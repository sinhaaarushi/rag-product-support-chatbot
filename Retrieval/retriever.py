"""Embed query and retrieve top chunks from local FAISS store."""

from __future__ import annotations

import re
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


# Folder-name synonyms (lowercased). Order is irrelevant — type is keyed by folder.
_FOLDER_TYPE_MAP: dict[str, str] = {
    "faq": "FAQ",
    "faqs": "FAQ",
    "frequentlyaskedquestions": "FAQ",
    "pss": "PSS",
    "productspecsheet": "PSS",
    "productspecsheets": "PSS",
    "productspecificationsheet": "PSS",
    "productspecificationsheets": "PSS",
    "manual": "manual",
    "manuals": "manual",
    "guide": "guide",
    "guides": "guide",
}

# Token synonyms for filename-based inference. Same idea as above but matched
# against tokens in the basename (split on non-alphanumeric).
_TOKEN_TYPE_MAP: dict[str, str] = {
    "faq": "FAQ",
    "faqs": "FAQ",
    "frequentlyaskedquestions": "FAQ",
    "pss": "PSS",
    "productspecsheet": "PSS",
    "productspecsheets": "PSS",
    "productspecificationsheet": "PSS",
    "productspecificationsheets": "PSS",
    "manual": "manual",
    "guide": "guide",
}

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _normalize(token: str) -> str:
    return token.lower().replace("_", "").replace("-", "").replace(" ", "")


def infer_document_type_from_name(file_name: str | Path) -> str:
    """Infer the document type for a PDF.

    Strategy (most specific wins):
      1. Walk the path's folder components from deepest to shallowest. The first
         folder whose normalized name maps to a known type (see ``_FOLDER_TYPE_MAP``)
         decides the type. This means ``PSS/FAQs/file.pdf`` resolves to ``FAQ``,
         not ``PSS``.
      2. Fall back to tokens in the basename (split on non-alphanumeric chars).
         Synonyms like ``ProductSpecSheet`` and ``FrequentlyAskedQuestions``
         resolve to ``PSS`` and ``FAQ`` respectively.
      3. Otherwise return ``"default"``.

    The function accepts either a relative path (preferred — same key as stored
    in metadata) or a bare filename. Matching is case-insensitive and ignores
    underscores, hyphens, and spaces.
    """
    path = Path(str(file_name))
    parts = list(path.parts)
    folders = parts[:-1] if len(parts) > 1 else []

    for folder in reversed(folders):
        norm = _normalize(folder)
        if norm in _FOLDER_TYPE_MAP:
            return _FOLDER_TYPE_MAP[norm]

    stem = path.stem
    tokens = [_normalize(tok) for tok in _TOKEN_SPLIT.split(stem) if tok]
    for token in tokens:
        if token in _TOKEN_TYPE_MAP:
            return _TOKEN_TYPE_MAP[token]

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
    candidate_pool = candidate_pool if candidate_pool is not None else config.RETRIEVAL_CANDIDATES
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
