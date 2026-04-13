"""
Quick manual smoke test: load PDFs from Data/documents, chunk, print stats.
Run from project root: python test_loader.py
"""

from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Ingestion.document_loader import load_pdf
from Processing.chunking import chunk_text

import config


def main() -> None:
    docs_dir = config.DOCUMENTS_DIR
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {docs_dir}. Add a .pdf to test extraction and chunking.")
        return

    for pdf in pdfs:
        print("File:", pdf.name)
        text = load_pdf(pdf)
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
