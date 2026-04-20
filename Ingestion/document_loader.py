"""Load PDF documents with per-page text extraction via pdfplumber.

The pipeline indexes chunks *per page* so that retrieval can cite the exact
page a fact came from. Everything downstream (chunking, metadata, retrieval)
assumes the ``page_number`` key is present on every record, so the loader
is the single source of truth for that field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class PdfPage:
    """One page of extracted PDF text.

    ``page_number`` is 1-indexed to match what a human sees in a PDF reader
    (page 1 is the first page), not pdfplumber's 0-indexed internal counter.
    ``text`` is already whitespace-normalized so callers can chunk it directly.
    """

    page_number: int
    text: str


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trim edges."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(file_path: str | Path) -> list[PdfPage]:
    """Extract each page of a PDF as its own ``PdfPage`` record.

    Pages that pdfplumber can't extract text from (scanned images without OCR,
    form XObjects it can't recurse into, etc.) return ``None`` and are dropped
    from the result — empty pages add noise without adding retrievable facts.
    This means the returned list may be shorter than the physical page count
    and may have gaps in ``page_number``; both are expected.

    Raises ``FileNotFoundError`` if the path does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[PdfPage] = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text:
                continue
            cleaned = _normalize_whitespace(raw_text)
            if cleaned:
                pages.append(PdfPage(page_number=page_index, text=cleaned))
    return pages


def load_pdf(file_path: str | Path) -> str:
    """Compatibility shim: return the whole PDF as one joined string.

    New code should prefer ``load_pdf_pages`` so page numbers can flow into
    FAISS metadata. This function stays so any caller that only needs a text
    blob (ad-hoc scripts, smoke tests) keeps working with no changes.
    """
    joined = "\n\n".join(page.text for page in load_pdf_pages(file_path))
    return _normalize_whitespace(joined)
