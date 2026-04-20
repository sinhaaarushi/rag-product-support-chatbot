"""
Local FAISS vector store with persisted metadata.
"""

from __future__ import annotations

import json
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
    raw_json = config.FAISS_METADATA_FILE.read_text(encoding="utf-8").strip()
    if not raw_json:
        return []
    return json.loads(raw_json)


def _write_metadata(records: list[dict[str, Any]]) -> None:
    config.FAISS_METADATA_FILE.write_text(
        json.dumps(records, ensure_ascii=True, indent=2),
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
    """Append chunk records to the FAISS index and the metadata sidecar.

    Each record must carry: ``text``, ``embedding``, ``document_name``,
    ``document_type``, ``weight``, ``chunk_index``.

    Index and metadata are written atomically enough for this single-writer
    use case: FAISS writes to a temp file and renames, json is written in one
    shot. If either write fails the caller retries the batch.
    """
    if not records:
        return 0

    index = load_or_create_index()
    metadata = _read_metadata()

    vectors = np.asarray([r["embedding"] for r in records], dtype="float32")
    if vectors.ndim != 2 or vectors.shape[1] != config.EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding shape mismatch. Expected (*, {config.EMBEDDING_DIMENSION}).")

    index.add(vectors)
    for record in records:
        metadata.append(
            {
                "text": record["text"],
                "document_name": record["document_name"],
                "document_type": record["document_type"],
                "weight": float(record["weight"]),
                "chunk_index": int(record["chunk_index"]),
            }
        )

    save_index(index)
    _write_metadata(metadata)
    return len(records)


def search(query_vector: list[float], k: int) -> list[dict[str, Any]]:
    """Return the top-k FAISS hits with attached metadata.

    FAISS returns ``-1`` in the id array for positions it couldn't fill (e.g.
    when the index has fewer than k vectors). We drop those and also guard
    against an id overflowing the metadata list, which would only happen if
    the index and metadata fell out of sync on disk — a rebuild fixes it.
    """
    index = load_or_create_index()
    metadata = _read_metadata()
    if index.ntotal == 0:
        return []

    query_matrix = np.asarray([query_vector], dtype="float32")
    scores, hit_ids = index.search(query_matrix, k)
    hits: list[dict[str, Any]] = []
    for score, hit_id in zip(scores[0], hit_ids[0], strict=False):
        if hit_id < 0 or hit_id >= len(metadata):
            continue
        hits.append(
            {
                "score": float(score),
                "metadata": metadata[hit_id],
            }
        )
    return hits


def get_store_stats() -> dict[str, Any]:
    index = load_or_create_index()
    metadata = _read_metadata()
    return {
        "index_path": str(config.FAISS_INDEX_FILE),
        "metadata_path": str(config.FAISS_METADATA_FILE),
        "vector_count": int(index.ntotal),
        "metadata_count": len(metadata),
    }


def unique_indexed_document_names() -> list[str]:
    """Distinct document_name values in metadata (order stable)."""
    metadata = _read_metadata()
    return sorted({str(m.get("document_name", "")) for m in metadata if m.get("document_name")})
