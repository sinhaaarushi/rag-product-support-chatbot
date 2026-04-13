"""Embed query and retrieve top chunks from local FAISS store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from Embeddings.embedding_generator import embed_text
from VectorStore.faiss_store import search


@dataclass
class RetrievedChunk:
    """One chunk returned to the LLM."""

    text: str
    document_name: str
    document_type: str
    weight: float
    score: float
    boosted_score: float


def infer_document_type_from_name(file_name: str) -> str:
    """
    Heuristic from basename: leading prefix PSS_ / FAQ_ / manual_ / guide_ (case-insensitive);
    otherwise 'manual'/'guide' substring; else 'default'.
    """
    base = Path(file_name).name
    lower = base.lower()
    first = lower.split("_", 1)[0] if "_" in lower else ""
    if first == "pss":
        return "PSS"
    if first == "faq":
        return "FAQ"
    if "manual" in lower:
        return "manual"
    if "guide" in lower:
        return "guide"
    return "default"


def retrieve_for_query(
    query: str,
    top_k: int | None = None,
    candidate_pool: int | None = None,
) -> list[RetrievedChunk]:
    """
    1) Embed query
    2) kNN search for a larger candidate set
    3) Re-rank by FAISS score * stored weight (PSS boost via higher weight)
    4) Return top_k chunks
    """
    top_k = top_k if top_k is not None else config.TOP_K
    candidate_pool = (
        candidate_pool if candidate_pool is not None else config.RETRIEVAL_CANDIDATES
    )
    candidate_pool = max(candidate_pool, top_k)

    query_vector = embed_text(query)
    hits = search(query_vector, k=candidate_pool)

    ranked: list[RetrievedChunk] = []
    for hit in hits:
        src = hit.get("metadata", {})
        score = float(hit.get("score", 0.0))
        weight = float(src.get("weight", 1.0))
        boosted = score * weight
        ranked.append(
            RetrievedChunk(
                text=src.get("text", ""),
                document_name=src.get("document_name", ""),
                document_type=src.get("document_type", "default"),
                weight=weight,
                score=score,
                boosted_score=boosted,
            )
        )

    ranked.sort(key=lambda c: c.boosted_score, reverse=True)
    return ranked[:top_k]


def chunks_to_context_dicts(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Serialize chunks for API responses or logging."""
    return [
        {
            "text": c.text,
            "document_name": c.document_name,
            "document_type": c.document_type,
            "weight": c.weight,
            "score": c.score,
            "boosted_score": c.boosted_score,
        }
        for c in chunks
    ]
