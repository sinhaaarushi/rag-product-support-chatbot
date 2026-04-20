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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from App.app import query_documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--role", choices=["customer", "sales"], default="customer")
    parser.add_argument("--top", type=int, default=5, help="Number of source rows to print.")
    args = parser.parse_args()

    result = query_documents(args.query, role=args.role, include_sources=True)

    print(f"QUERY   : {args.query}")
    print(f"ROLE    : {args.role}")
    print(f"BACKEND : {result['llm_backend']}")
    print()
    print("ANSWER  :")
    print(result["answer"])
    print()
    print(f"TOP {args.top} SOURCES (after weighted re-ranking):")
    for s in result.get("sources", [])[: args.top]:
        dtype = s["document_type"]
        score = s["score"]
        boosted = s["boosted_score"]
        name = s["document_name"]
        print(f"  {dtype:<5}  score={score:.3f}  boosted={boosted:.3f}  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
