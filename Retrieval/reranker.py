"""Cross-encoder re-ranking of retrieved chunks.

FAISS kNN compares two embeddings produced independently; a cross-encoder
reads the (question, chunk) pair jointly and can spot relevance that pure
cosine similarity misses -- think of it as "shallow reading" vs "quick
glance". In practice, adding this stage lifts top-1 precision noticeably
with a small CPU cost because the model is tiny (~80 MB MiniLM).

We keep this optional behind ``config.USE_RERANKER`` so it's easy to A/B
against the pure kNN path, and so tests that don't care about re-ranking
can turn it off without monkey-patching imports.

Loading is lazy + cached: the first query pays the ~2 s model load once,
then every subsequent query reuses the cached instance.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import config

_cross_encoder: Any | None = None
_cross_encoder_lock = threading.Lock()


def _resolve_reranker_ref() -> tuple[str, bool]:
    """Return (model_ref, local_files_only) for the cross-encoder.

    Mirrors the pattern used for the embedding and chat models: when running
    in OFFLINE_ONLY mode we insist on a local path; otherwise fall back to
    the repo name so a first-run still works online.
    """
    if config.OFFLINE_ONLY:
        if not config.RERANKER_MODEL_LOCAL_PATH:
            raise RuntimeError(
                "OFFLINE_ONLY is enabled but RERANKER_MODEL_LOCAL_PATH is not set. "
                "Run scripts/download_models.py or set USE_RERANKER=false to skip."
            )
        path = Path(config.RERANKER_MODEL_LOCAL_PATH).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Re-ranker path not found: {path}")
        return str(path), True
    model_ref = config.RERANKER_MODEL_LOCAL_PATH or config.RERANKER_MODEL_NAME
    return model_ref, bool(config.RERANKER_MODEL_LOCAL_PATH)


def _load_cross_encoder() -> Any:
    """Load + cache the sentence-transformers ``CrossEncoder``.

    Lazy-imported (and the whole function is a no-op if called twice) so
    ``USE_RERANKER=false`` has zero import-time cost.
    """
    global _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        from sentence_transformers import CrossEncoder

        model_ref, local_files_only = _resolve_reranker_ref()
        _cross_encoder = CrossEncoder(model_ref, local_files_only=local_files_only)
        return _cross_encoder


def _min_max_normalize(values: list[float]) -> list[float]:
    """Scale ``values`` into the 0-1 range for blending.

    Cross-encoder scores are un-normalised logits (often negative) while FAISS
    cosine scores are already in [0, 1]. Blending them directly would always
    let one dominate. Min-max normalizing both sides first gives a fair mix.
    Uniform inputs (all equal) are returned as 0.5 across the board so the
    re-ranker simply passes through the weighted kNN order in that edge case.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / span for v in values]


def rerank(
    query: str,
    chunk_texts: list[str],
    weighted_scores: list[float],
) -> list[float]:
    """Return a blended relevance score for each chunk, same order as input.

    Callers pass the chunk texts (for the cross-encoder) and the existing
    weighted FAISS scores (for the blend). We don't reorder here -- we just
    score -- so the caller can combine these with whatever other metadata
    they track on each chunk (page number, document type, ...).

    The blend follows ``config.RERANKER_BLEND``: higher values trust the
    cross-encoder more. 0.85 is a good default because the cross-encoder is
    far more reliable at relevance, but keeping 15% of the weighted score
    means PSS / FAQ priority still tilts ties.
    """
    if not chunk_texts:
        return []
    model = _load_cross_encoder()
    pairs = [(query, text) for text in chunk_texts]
    ce_raw_scores = [float(s) for s in model.predict(pairs, show_progress_bar=False)]
    ce_norm = _min_max_normalize(ce_raw_scores)
    kn_norm = _min_max_normalize(weighted_scores)

    blend = config.RERANKER_BLEND
    return [blend * ce + (1.0 - blend) * kn for ce, kn in zip(ce_norm, kn_norm, strict=True)]
