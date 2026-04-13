"""
Load PDF documents and return plain text using pdfplumber.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trim edges."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(file_path: str | Path) -> str:
    """
    Open a PDF, extract text from every page, and return a single clean string.

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)

    return _normalize_whitespace("\n\n".join(parts))
