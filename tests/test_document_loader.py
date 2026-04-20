"""Tests for the page-aware PDF loader.

These tests use pdfplumber with in-memory fakes rather than real PDFs so
they stay fast and don't need fixture files checked in. What we care about
here is the *shape* of the output (one record per page, 1-indexed page
numbers, whitespace cleaned) — pdfplumber's own text extraction is covered
by its own test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from Ingestion.document_loader import PdfPage, load_pdf, load_pdf_pages


class _FakePage:
    """Minimal stand-in for a pdfplumber Page object."""

    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePdf:
    """Context-managed stand-in for pdfplumber.open()."""

    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> _FakePdf:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patched_open(pages: list[_FakePage]):
    """Shortcut: patch pdfplumber.open to return the given fake pages."""
    mock = MagicMock(return_value=_FakePdf(pages))
    return patch("Ingestion.document_loader.pdfplumber.open", mock)


def test_load_pdf_pages_yields_one_record_per_nonempty_page(tmp_path):
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"irrelevant")  # only needs to exist

    pages = [_FakePage("Hello world"), _FakePage("Second page body")]
    with _patched_open(pages):
        result = load_pdf_pages(pdf_file)

    assert [p.page_number for p in result] == [1, 2]
    assert all(isinstance(p, PdfPage) for p in result)
    assert result[0].text == "Hello world"
    assert result[1].text == "Second page body"


def test_load_pdf_pages_drops_empty_pages_but_preserves_indexing(tmp_path):
    """A scanned page returns None; page numbers must stay 1-indexed from
    the physical PDF and not be renumbered over the dropped gap.
    """
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"irrelevant")

    pages = [_FakePage("Page one"), _FakePage(None), _FakePage("Page three")]
    with _patched_open(pages):
        result = load_pdf_pages(pdf_file)

    assert [p.page_number for p in result] == [1, 3]


def test_load_pdf_pages_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        load_pdf_pages(missing)


def test_load_pdf_compat_shim_joins_pages(tmp_path):
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"irrelevant")

    pages = [_FakePage("First."), _FakePage("Second.")]
    with _patched_open(pages):
        joined = load_pdf(pdf_file)

    assert "First." in joined and "Second." in joined
