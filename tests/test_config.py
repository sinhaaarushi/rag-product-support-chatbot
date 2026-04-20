"""Config helpers and document naming rules."""

from __future__ import annotations

import re

import config


def test_default_pattern_accepts_any_pdf():
    """Default policy is permissive — type is inferred from the folder layout."""
    assert config.is_valid_doc_name("PSS_motor_v1.pdf") is True
    assert config.is_valid_doc_name("any_product_name.pdf") is True
    assert config.is_valid_doc_name("Report.PDF") is True
    assert config.is_valid_doc_name("nested/PSS_motor_v1.pdf") is True


def test_default_pattern_rejects_non_pdf():
    assert config.is_valid_doc_name("notes.txt") is False
    assert config.is_valid_doc_name("readme") is False


def test_strict_pattern_via_override(monkeypatch):
    monkeypatch.setattr(
        config,
        "DOC_NAME_PATTERN",
        r"^(PSS|FAQ|manual|guide)_[A-Za-z0-9._-]+\.pdf$",
    )
    assert config.is_valid_doc_name("PSS_motor_v1.pdf") is True
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
