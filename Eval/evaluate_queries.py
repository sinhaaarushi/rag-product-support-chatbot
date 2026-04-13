from __future__ import annotations

import json
from pathlib import Path

from App.app import query_documents


def run_eval(queries_file: str | Path = Path("Eval/sample_queries.json")) -> list[dict]:
    path = Path(queries_file).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for item in data:
        query = item["query"]
        role = item.get("role", "customer")
        resp = query_documents(query=query, role=role, include_sources=True)
        out.append(
            {
                "query": query,
                "role": role,
                "answer": resp.get("answer", ""),
                "source_count": len(resp.get("sources", [])),
            }
        )
    return out


if __name__ == "__main__":
    results = run_eval()
    print(json.dumps(results, indent=2))
