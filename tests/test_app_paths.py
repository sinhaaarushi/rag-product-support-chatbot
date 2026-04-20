"""Path-resolver guards in App.app — block indexing outside the documents dir."""

from __future__ import annotations

import pytest

import config
from App.app import _resolve_folder_path, _resolve_path


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", docs)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR", False)
    return docs


def test_resolve_path_accepts_file_inside_documents_dir(docs_dir):
    pdf = docs_dir / "PSS" / "PSS_a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    resolved = _resolve_path(pdf)
    assert resolved == pdf.resolve()


def test_resolve_path_rejects_file_outside_documents_dir(docs_dir, tmp_path):
    outside = tmp_path / "elsewhere" / "PSS_a.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="must be inside"):
        _resolve_path(outside)


def test_resolve_path_allows_outside_when_flag_enabled(docs_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_INDEX_OUTSIDE_DOCUMENTS_DIR", True)
    outside = tmp_path / "elsewhere" / "PSS_a.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF")
    resolved = _resolve_path(outside)
    assert resolved == outside.resolve()


def test_resolve_folder_path_only_accepts_documents_dir(docs_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="must be documents directory"):
        _resolve_folder_path(other)
