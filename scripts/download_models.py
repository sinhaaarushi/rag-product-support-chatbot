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

# Files that must be present for AutoTokenizer / AutoModelForSeq2SeqLM to load
# the chat model offline. flan-t5-base ships SentencePiece + tokenizer.json.
_CHAT_SENTINELS = ("config.json", "tokenizer.json", "spiece.model")


def _default_target(env_value: str, fallback: Path) -> Path:
    """Resolve where a model should be written.

    Prefers the env-configured local path (so ``EMBEDDING_MODEL_LOCAL_PATH`` is
    the single source of truth), and falls back to a sensible default under
    ``C:\\models`` so the script is useful on a fresh machine even before ``.env``
    has been set up.
    """
    return Path(env_value).resolve() if env_value else fallback


def _already_downloaded(target: Path, sentinels: tuple[str, ...]) -> bool:
    return target.is_dir() and all((target / name).is_file() for name in sentinels)


def _download(repo_id: str, target: Path, force: bool) -> None:
    # Imported lazily so a missing `huggingface_hub` only fails when we actually
    # try to download, not on `--help`.
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"  -> {repo_id}")
    print(f"     into {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        # Real files on Windows. Symlinks inside user dirs trip up venvs and
        # antivirus software more often than they're worth.
        local_dir_use_symlinks=False,
        # We want everything the tokenizer / model loaders might look for.
        # Blocking msgpack/safetensors would save a few MB but introduce
        # surprise failures the first time someone flips a loader flag.
        ignore_patterns=["*.onnx", "*.msgpack", "onnx/*", "openvino/*"],
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
    args = parser.parse_args()

    embedding_target = _default_target(
        config.EMBEDDING_MODEL_LOCAL_PATH,
        Path(os.getenv("SystemDrive", "C:")) / "models" / "all-MiniLM-L6-v2",
    )
    chat_target = _default_target(
        config.HF_CHAT_MODEL_LOCAL_PATH,
        Path(os.getenv("SystemDrive", "C:")) / "models" / "flan-t5-base",
    )

    plan: list[tuple[str, str, Path, tuple[str, ...]]] = []
    if not args.skip_embedding:
        plan.append(
            ("Embedding", config.EMBEDDING_MODEL_NAME, embedding_target, _EMBEDDING_SENTINELS)
        )
    if not args.skip_chat:
        plan.append(("Chat", config.HF_CHAT_MODEL, chat_target, _CHAT_SENTINELS))

    if not plan:
        print("Nothing to do (both --skip-embedding and --skip-chat were set).")
        return 0

    print("Download plan:")
    for label, repo, target, _ in plan:
        print(f"  [{label}] {repo} -> {target}")
    print()

    for label, repo, target, sentinels in plan:
        print(f"[{label}]")
        if not args.force and _already_downloaded(target, sentinels):
            print(f"  already present at {target}, skipping. (use --force to re-fetch)")
            continue
        try:
            _download(repo, target, args.force)
        except Exception as exc:  # noqa: BLE001 — we want the error message front and centre
            print(f"  FAILED: {exc}", file=sys.stderr)
            return 1
        missing = [name for name in sentinels if not (target / name).is_file()]
        if missing:
            print(f"  WARNING: downloaded but expected files missing: {missing}", file=sys.stderr)
        else:
            print("  ok")

    print()
    print("Done. You can now run the indexing pipeline offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
