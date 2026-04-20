"""Full index rebuild from the command line.

Clears the FAISS store and re-indexes every PDF under ``Data/documents/`` in
document order, printing one status line per file and a final summary. Useful
for cold-starts, CI, and anyone who'd rather not boot Streamlit just to kick
off a rebuild.

Usage (PowerShell)::

    python scripts\rebuild_index.py

Environment: the offline-only config applies — ``EMBEDDING_MODEL_LOCAL_PATH``
must point at a populated model directory (see ``scripts/download_models.py``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from App.app import (
    documents_sync_report,
    preflight_check,
    rebuild_index_from_documents,
)
from pipeline import list_pdf_files_under


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    print("=" * 72)
    print(" Preflight")
    print("=" * 72)
    checks = preflight_check()
    for k, v in checks.items():
        print(f"  {k}: {v}")
    if not checks.get("ready"):
        print("\nPreflight FAILED. Aborting rebuild.", file=sys.stderr)
        return 1

    print()
    print("=" * 72)
    print(" Documents on disk")
    print("=" * 72)
    pdfs = list_pdf_files_under(config.DOCUMENTS_DIR)
    total_bytes = sum(p.stat().st_size for p in pdfs)
    print(f"  Location      : {config.DOCUMENTS_DIR}")
    print(f"  Files         : {len(pdfs)}")
    print(f"  Combined size : {_format_bytes(total_bytes)}")

    print()
    print("=" * 72)
    print(" Rebuilding FAISS index (this will clear existing chunks)")
    print("=" * 72)
    start = time.monotonic()
    result = rebuild_index_from_documents()
    elapsed = time.monotonic() - start

    print()
    print("=" * 72)
    print(" Results")
    print("=" * 72)
    print(f"  Documents indexed     : {result.get('documents_indexed')}")
    print(f"  Total chunks indexed  : {result.get('total_chunks_indexed')}")
    print(f"  Elapsed               : {elapsed:.1f}s")

    per_doc = result.get("results", [])
    by_type: dict[str, list[int]] = {}
    for d in per_doc:
        by_type.setdefault(d.get("document_type", "unknown"), []).append(
            int(d.get("chunks_indexed", 0))
        )
    print()
    print("  Chunks by document type:")
    for dtype, counts in sorted(by_type.items()):
        print(f"    {dtype:<10} files={len(counts):>3}  chunks={sum(counts):>6}")

    zero = [d for d in per_doc if int(d.get("chunks_indexed", 0)) == 0]
    if zero:
        print()
        print(f"  WARNING: {len(zero)} document(s) produced 0 chunks:")
        for d in zero[:10]:
            print(f"    {d.get('document_name')}")
        if len(zero) > 10:
            print(f"    ... and {len(zero) - 10} more")

    print()
    print("=" * 72)
    print(" Sync report")
    print("=" * 72)
    sync = documents_sync_report()
    print(f"  on_disk   : {sync['pdf_count_on_disk']}")
    print(f"  in_index  : {sync['indexed_document_count']}")
    print(f"  in_sync   : {sync['in_sync']}")
    if not sync["in_sync"]:
        if sync["only_on_disk"]:
            print(f"  only_on_disk: {sync['only_on_disk']}")
        if sync["only_in_index"]:
            print(f"  only_in_index: {sync['only_in_index']}")

    return 0 if sync["in_sync"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
