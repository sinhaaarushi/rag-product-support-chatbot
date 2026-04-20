"""
Manual smoke test: load every PDF under Data/documents, chunk, print stats.

Not a pytest file. Run from the project root:

    python scripts/smoke_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from Ingestion.document_loader import load_pdf
from pipeline import list_pdf_files_under
from Processing.chunking import chunk_text


def main() -> None:
    docs_dir = config.DOCUMENTS_DIR
    pdfs = list_pdf_files_under(docs_dir)
    if not pdfs:
        print(f"No PDFs in {docs_dir}. Add a .pdf to test extraction and chunking.")
        return

    for pdf in pdfs:
        rel = config.document_storage_key(pdf)
        print("File:", rel)
        try:
            text = load_pdf(pdf)
        except Exception as exc:
            print(f"  ERROR loading: {exc}")
            continue
        chunks = chunk_text(
            text,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        print("  Chars:", len(text))
        print("  Chunks:", len(chunks))
        if chunks:
            print("  First chunk preview:", chunks[0][:200].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
