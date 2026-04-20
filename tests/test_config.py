"""Config helpers and document naming rules."""

from __future__ import annotations

import re

import config


def test_valid_doc_names_match_pattern():
    assert config.is_valid_doc_name("PSS_motor_v1.pdf") is True
    assert config.is_valid_doc_name("FAQ_returns.pdf") is True
    assert config.is_valid_doc_name("manual_installation.pdf") is True
    assert config.is_valid_doc_name("guide_quickstart.pdf") is True
    assert config.is_valid_doc_name("nested/PSS_motor_v1.pdf") is True


def test_invalid_doc_names():
    assert config.is_valid_doc_name("random.pdf") is False
    assert config.is_valid_doc_name("PSS.pdf") is False


def test_doc_name_pattern_is_valid_regex():
    re.compile(config.DOC_NAME_PATTERN)


def test_document_storage_key_uses_relative_path(monkeypatch, tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", docs)
    nested = docs / "PSS" / "PSS_motor_v1.pdf"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"%PDF-1.4")
    assert config.document_storage_key(nested) == "PSS/PSS_motor_v1.pdf"


def test_document_storage_key_outside_docs_is_basename(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", docs)
    other = tmp_path / "elsewhere" / "PSS_x.pdf"
    other.parent.mkdir(parents=True)
    other.touch()
    assert config.document_storage_key(other) == "PSS_x.pdf"
