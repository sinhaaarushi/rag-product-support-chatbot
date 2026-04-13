"""
Local FAISS vector store with persisted metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config


def _ensure_store_dir() -> None:
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    # Vectors are normalized in embedding generator; inner product ~= cosine similarity.
    return faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)


def _read_metadata() -> list[dict[str, Any]]:
    if not config.FAISS_METADATA_FILE.exists():
        return []
    raw = config.FAISS_METADATA_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return json.loads(raw)


def _write_metadata(items: list[dict[str, Any]]) -> None:
    config.FAISS_METADATA_FILE.write_text(
        json.dumps(items, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_or_create_index() -> faiss.Index:
    _ensure_store_dir()
    if config.FAISS_INDEX_FILE.exists():
        return faiss.read_index(str(config.FAISS_INDEX_FILE))
    return _new_index()


def save_index(index: faiss.Index) -> None:
    _ensure_store_dir()
    faiss.write_index(index, str(config.FAISS_INDEX_FILE))


def clear_store() -> None:
    if config.FAISS_INDEX_FILE.exists():
        config.FAISS_INDEX_FILE.unlink()
    if config.FAISS_METADATA_FILE.exists():
        config.FAISS_METADATA_FILE.unlink()


def add_embeddings(records: list[dict[str, Any]]) -> int:
    """
    records items require: text, embedding, document_name, document_type, weight, chunk_index
    """
    if not records:
        return 0

    index = load_or_create_index()
    metadata = _read_metadata()

    vectors = np.asarray([r["embedding"] for r in records], dtype="float32")
    if vectors.ndim != 2 or vectors.shape[1] != config.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding shape mismatch. Expected (*, {config.EMBEDDING_DIMENSION})."
        )

    index.add(vectors)
    for r in records:
        metadata.append(
            {
                "text": r["text"],
                "document_name": r["document_name"],
                "document_type": r["document_type"],
                "weight": float(r["weight"]),
                "chunk_index": int(r["chunk_index"]),
            }
        )

    save_index(index)
    _write_metadata(metadata)
    return len(records)


def search(query_vector: list[float], k: int) -> list[dict[str, Any]]:
    index = load_or_create_index()
    metadata = _read_metadata()
    if index.ntotal == 0:
        return []

    q = np.asarray([query_vector], dtype="float32")
    scores, ids = index.search(q, k)
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if idx >= len(metadata):
            continue
        results.append(
            {
                "score": float(score),
                "metadata": metadata[idx],
            }
        )
    return results


def get_store_stats() -> dict[str, Any]:
    index = load_or_create_index()
    metadata = _read_metadata()
    return {
        "index_path": str(config.FAISS_INDEX_FILE),
        "metadata_path": str(config.FAISS_METADATA_FILE),
        "vector_count": int(index.ntotal),
        "metadata_count": len(metadata),
    }
