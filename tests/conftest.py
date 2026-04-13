"""Pytest fixtures: isolated FAISS paths so tests never touch real Data/vector_store."""

from __future__ import annotations

import pytest

import config


@pytest.fixture
def isolated_vector_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    vs = tmp_path / "vector_store"
    vs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", vs)
    monkeypatch.setattr(config, "FAISS_INDEX_FILE", vs / "index.faiss")
    monkeypatch.setattr(config, "FAISS_METADATA_FILE", vs / "metadata.json")
    return vs
