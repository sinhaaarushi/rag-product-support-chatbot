"""
Generate dense embeddings using sentence-transformers (all-MiniLM-L6-v2 by default).
"""

from __future__ import annotations

import threading
from pathlib import Path

from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None:
            if config.OFFLINE_ONLY:
                if not config.EMBEDDING_MODEL_LOCAL_PATH:
                    raise RuntimeError(
                        "OFFLINE_ONLY is enabled but EMBEDDING_MODEL_LOCAL_PATH is not set."
                    )
                local_path = Path(config.EMBEDDING_MODEL_LOCAL_PATH).resolve()
                if not local_path.exists():
                    raise FileNotFoundError(f"Embedding model path not found: {local_path}")
                _model = SentenceTransformer(str(local_path), local_files_only=True)
            else:
                model_name_or_path = (
                    config.EMBEDDING_MODEL_LOCAL_PATH
                    if config.EMBEDDING_MODEL_LOCAL_PATH
                    else config.EMBEDDING_MODEL_NAME
                )
                _model = SentenceTransformer(model_name_or_path)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Encode a single text chunk into a float vector (dimension matches config).
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    model = _get_model()
    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype("float32").tolist()


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch encode many chunks (faster for indexing)."""
    if not texts:
        return []
    model = _get_model()
    arrays = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [row.astype("float32").tolist() for row in arrays]
