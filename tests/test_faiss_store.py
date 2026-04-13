"""FAISS store add/search with isolated paths."""

from __future__ import annotations

import numpy as np
import pytest

import config
from VectorStore import faiss_store as fs


def _fake_record(i: int, dim: int) -> dict:
    # Normalized random-ish vector for IP search
    v = np.random.randn(dim).astype("float32")
    v /= np.linalg.norm(v) + 1e-9
    return {
        "text": f"chunk {i}",
        "embedding": v.tolist(),
        "document_name": "PSS_test.pdf",
        "document_type": "PSS",
        "weight": 1.5,
        "chunk_index": i,
    }


def test_add_and_search_roundtrip(isolated_vector_store, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 32)
    fs.clear_store()
    records = [_fake_record(i, 32) for i in range(5)]
    n = fs.add_embeddings(records)
    assert n == 5
    stats = fs.get_store_stats()
    assert stats["vector_count"] == 5
    assert stats["metadata_count"] == 5

    q = np.asarray(records[2]["embedding"], dtype="float32")
    q /= np.linalg.norm(q) + 1e-9
    hits = fs.search(q.tolist(), k=3)
    assert len(hits) >= 1
    assert hits[0]["metadata"]["chunk_index"] == 2


def test_unique_indexed_document_names(isolated_vector_store, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 32)
    fs.clear_store()
    r0 = _fake_record(0, 32)
    r1 = {**_fake_record(1, 32), "document_name": "FAQ_other.pdf"}
    fs.add_embeddings([r0, r1])
    names = fs.unique_indexed_document_names()
    assert names == ["FAQ_other.pdf", "PSS_test.pdf"]


def test_add_rejects_wrong_dimension(isolated_vector_store, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 8)
    fs.clear_store()
    bad = _fake_record(0, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        fs.add_embeddings([bad])
