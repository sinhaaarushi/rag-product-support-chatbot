"""One-shot sample query against the indexed documents.

Useful as an end-to-end sanity check after a rebuild. Prints the LLM answer,
the backend that served it, and the top retrieved chunks with their document
type, raw score, and weight-boosted score.

Usage::

    python scripts\\sample_query.py "What cloud services are offered?"
    python scripts\\sample_query.py "backup policy" --role sales --top 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _log(message: str) -> None:
    """Print with a timestamp and flush immediately.

    Why manual flushing: on Windows, Python buffers stdout when the output
    is piped or captured, which makes the script look hung for minutes
    during the first model load. Flushing on every progress line means the
    caller sees exactly where the work is sitting right now.
    """
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--role", choices=["customer", "sales"], default="customer")
    parser.add_argument("--top", type=int, default=5, help="Number of source rows to print.")
    args = parser.parse_args()

    _log("importing backend (first call also triggers transformers / torch import)...")
    # Import deliberately inside main so the log line above prints before
    # the ~5-10 s of heavy Python import kicks in.
    from App.app import query_documents
    from Retrieval.retriever import retrieve_for_query

    _log(f"running retrieval for: {args.query!r}")
    retrieval_start = time.perf_counter()
    retrieved = retrieve_for_query(args.query)
    _log(
        f"retrieval done in {time.perf_counter() - retrieval_start:.1f}s ({len(retrieved)} chunks)"
    )

    _log("calling LLM (first call loads the ~1 GB GGUF model; later calls are faster)...")
    llm_start = time.perf_counter()
    result = query_documents(args.query, role=args.role, include_sources=True)
    _log(f"LLM done in {time.perf_counter() - llm_start:.1f}s")

    print()
    print(f"QUERY   : {args.query}")
    print(f"ROLE    : {args.role}")
    print(f"BACKEND : {result['llm_backend']}")
    print()
    print("ANSWER  :")
    print(result["answer"])
    print()
    print(f"TOP {args.top} SOURCES (after weighted re-ranking):")
    for source in result.get("sources", [])[: args.top]:
        doc_type = source["document_type"]
        raw_score = source["score"]
        boosted_score = source["boosted_score"]
        doc_name = source["document_name"]
        print(f"  {doc_type:<5}  score={raw_score:.3f}  boosted={boosted_score:.3f}  {doc_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
