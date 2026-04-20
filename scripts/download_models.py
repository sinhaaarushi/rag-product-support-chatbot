"""One-shot download of the Hugging Face models this project needs.

Run this once on a new machine to populate ``C:\\models`` (or whatever paths you
have set in ``EMBEDDING_MODEL_LOCAL_PATH`` / ``HF_CHAT_MODEL_LOCAL_PATH``). After
that, the app can run fully offline — ``OFFLINE_ONLY=true`` just loads the files
from disk.

Usage (PowerShell)::

    python scripts\\download_models.py

Usage (bash)::

    python scripts/download_models.py

The script is idempotent: if the target folder already contains the required
files, it skips the download. Pass ``--force`` to re-download anyway.

Only two model repos are fetched — the embedding model and the seq2seq chat
model — and they are pinned by name in ``config.py``. Nothing else is reached
out to.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# Files that must be present for SentenceTransformer(..., local_files_only=True)
# to load the embedding model without reaching the network.
_EMBEDDING_SENTINELS = ("config.json", "modules.json", "sentence_bert_config.json")

# Quantized chat model distributed as a single .gguf file. We only need the
# Q4_K_M quantization: ~1.8 GB at full 3B parameters, which delivers
# near-parity quality with the full-precision checkpoint while still fitting
# in laptop RAM.
_CHAT_GGUF_REPO = "bartowski/Qwen2.5-3B-Instruct-GGUF"
_CHAT_GGUF_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# Cross-encoder re-ranker. This is a ~175 MB MiniLM model trained on MS MARCO
# that re-scores (question, chunk) pairs. Worth its weight: it's the single
# biggest quality lever in most RAG stacks.
_RERANKER_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_SENTINELS = ("config.json", "tokenizer.json")


def _windows_fallback(name: str) -> Path:
    """``C:\\models\\<name>``-style fallback, safe on Windows.

    ``Path("C:") / "models"`` is drive-*relative* on Windows (resolves against
    the current directory of drive C:), so it silently writes into wherever
    the script happens to run. Append an explicit separator to get an absolute
    path rooted at the drive.
    """
    drive = os.getenv("SystemDrive", "C:")
    return Path(drive + os.sep) / "models" / name


def _default_target(env_value: str, fallback: Path) -> Path:
    """Resolve where a model should be written.

    Prefers the env-configured local path (so ``EMBEDDING_MODEL_LOCAL_PATH`` is
    the single source of truth), and falls back to ``C:\\models\\<name>`` so the
    script is useful on a fresh machine even before ``.env`` has been set up.
    """
    return Path(env_value).resolve() if env_value else fallback


def _already_downloaded(target: Path, sentinels: tuple[str, ...]) -> bool:
    return target.is_dir() and all((target / name).is_file() for name in sentinels)


def _download_snapshot(repo_id: str, target: Path, force: bool) -> None:
    """Fetch a full HF repo snapshot (used for the embedding model).

    Lazy-imported so a missing ``huggingface_hub`` only errors when we
    actually try to download, not on ``--help``.
    """
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"  -> {repo_id}")
    print(f"     into {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        # We serve PyTorch via sentence-transformers, so skip the flax / TF /
        # ONNX / OpenVINO weights. Keeps the download lean on slow networks.
        ignore_patterns=[
            "*.onnx",
            "onnx/*",
            "openvino/*",
            "*.msgpack",
            "flax_model.*",
            "tf_model.*",
            "*.h5",
        ],
        force_download=force,
    )


def _download_single_file(repo_id: str, filename: str, target: Path, force: bool) -> None:
    """Fetch one specific file from a HF repo (used for the GGUF chat model).

    ``snapshot_download`` would pull every quantization level in the repo
    (Q2, Q3, ..., Q8 -- several GBs each). We only want Q4_K_M, so we call
    ``hf_hub_download`` directly to stay lean.
    """
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"  -> {repo_id} / {filename}")
    print(f"     into {target}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target),
        force_download=force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if sentinel files are already present.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Do not download the embedding model.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Do not download the chat model.",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Do not download the cross-encoder re-ranker.",
    )
    args = parser.parse_args()

    embedding_target = _default_target(
        config.EMBEDDING_MODEL_LOCAL_PATH,
        _windows_fallback("all-MiniLM-L6-v2"),
    )
    # The chat target can be either a folder (the whole repo snapshot) or a
    # specific .gguf path. For the GGUF flow we want the folder, and the
    # single-file downloader drops the .gguf inside it.
    raw_chat_target = (
        Path(config.HF_CHAT_MODEL_LOCAL_PATH).resolve()
        if config.HF_CHAT_MODEL_LOCAL_PATH
        else _windows_fallback("qwen2.5-3b-instruct")
    )
    chat_folder = (
        raw_chat_target if raw_chat_target.suffix.lower() != ".gguf" else raw_chat_target.parent
    )

    reranker_target = _default_target(
        config.RERANKER_MODEL_LOCAL_PATH,
        _windows_fallback("ms-marco-MiniLM-L6-v2"),
    )

    embedding_done = not args.skip_embedding and _already_downloaded(
        embedding_target, _EMBEDDING_SENTINELS
    )
    chat_done = not args.skip_chat and (chat_folder / _CHAT_GGUF_FILE).is_file()
    reranker_done = not args.skip_reranker and _already_downloaded(
        reranker_target, _RERANKER_SENTINELS
    )

    print("Download plan:")
    if not args.skip_embedding:
        status = "ok (skip)" if embedding_done and not args.force else "pending"
        print(f"  [Embedding] {config.EMBEDDING_MODEL_NAME} -> {embedding_target}  [{status}]")
    if not args.skip_chat:
        status = "ok (skip)" if chat_done and not args.force else "pending"
        print(f"  [Chat]      {_CHAT_GGUF_REPO} / {_CHAT_GGUF_FILE} -> {chat_folder}  [{status}]")
    if not args.skip_reranker:
        status = "ok (skip)" if reranker_done and not args.force else "pending"
        print(f"  [Re-ranker] {_RERANKER_REPO} -> {reranker_target}  [{status}]")
    if args.skip_embedding and args.skip_chat and args.skip_reranker:
        print("  (nothing to do -- all skips set)")
        return 0
    print()

    try:
        if not args.skip_embedding:
            print("[Embedding]")
            if embedding_done and not args.force:
                print(f"  already present at {embedding_target}, skipping.")
            else:
                _download_snapshot(config.EMBEDDING_MODEL_NAME, embedding_target, args.force)
                missing = [
                    name for name in _EMBEDDING_SENTINELS if not (embedding_target / name).is_file()
                ]
                if missing:
                    print(f"  WARNING: expected files missing: {missing}", file=sys.stderr)
                else:
                    print("  ok")

        if not args.skip_chat:
            print("[Chat]")
            if chat_done and not args.force:
                print(f"  already present at {chat_folder / _CHAT_GGUF_FILE}, skipping.")
            else:
                _download_single_file(_CHAT_GGUF_REPO, _CHAT_GGUF_FILE, chat_folder, args.force)
                if not (chat_folder / _CHAT_GGUF_FILE).is_file():
                    print("  WARNING: GGUF file not found after download.", file=sys.stderr)
                else:
                    print("  ok")

        if not args.skip_reranker:
            print("[Re-ranker]")
            if reranker_done and not args.force:
                print(f"  already present at {reranker_target}, skipping.")
            else:
                _download_snapshot(_RERANKER_REPO, reranker_target, args.force)
                missing = [
                    name for name in _RERANKER_SENTINELS if not (reranker_target / name).is_file()
                ]
                if missing:
                    print(f"  WARNING: expected files missing: {missing}", file=sys.stderr)
                else:
                    print("  ok")
    except Exception as exc:  # noqa: BLE001 -- surface the error message front and centre
        print(f"  FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    print("Done. You can now run the indexing pipeline and chat offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
