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
    """One chunk returned to the LLM.

    ``page_number`` is 1-indexed and matches the physical page a human would
    see in a PDF reader. Value is 0 for records indexed before page tracking
    was added; the UI treats 0 as "unknown page" and simply omits it.
    """

    text: str
    document_name: str
    document_type: str
    weight: float
    score: float
    boosted_score: float
    page_number: int = 0


# Folder-name synonyms (lowercased, with common punctuation stripped).
# The document type comes exclusively from the folder a PDF lives under; filenames
# are never consulted. If a file is not under any recognized folder, its type is
# ``"default"`` — which means: move it into the right folder, don't rename it.
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


def _normalize(token: str) -> str:
    return token.lower().replace("_", "").replace("-", "").replace(" ", "")


def infer_document_type_from_name(file_name: str | Path) -> str:
    """Infer the document type for a PDF from its folder path.

    Strategy: walk the path's folder components from deepest to shallowest and
    return the type of the first folder that matches ``_FOLDER_TYPE_MAP``. So
    ``PSS/FAQ/returns.pdf`` resolves to ``FAQ`` (not ``PSS``) because the deeper
    folder wins. Filenames are never consulted — organization is the source of
    truth, which keeps the rule predictable and forces clean folder layout.

    Accepts either a relative path (preferred — same key as stored in metadata)
    or a bare filename. Matching is case-insensitive and ignores underscores,
    hyphens, and spaces, so ``PSS``, ``pss``, ``Product Spec Sheets``, and
    ``product-spec-sheets`` all map to ``PSS``.

    Returns ``"default"`` when the file is not under any recognized folder.
    """
    path = Path(str(file_name))
    folders = list(path.parts)[:-1]

    for folder in reversed(folders):
        norm = _normalize(folder)
        if norm in _FOLDER_TYPE_MAP:
            return _FOLDER_TYPE_MAP[norm]

    return "default"


def retrieve_for_query(
    query: str,
    top_k: int | None = None,
    candidate_pool: int | None = None,
) -> list[RetrievedChunk]:
    """Run the full retrieval pipeline for a single user query.

    Stages:
      1. Embed the query with the same sentence-transformers model the index
         was built with.
      2. kNN against FAISS for a wider candidate pool (``candidate_pool``) so
         the re-ranker has real options, not just the top-3 "obvious" hits.
      3. Apply the stored per-document weight (PSS boost, FAQ boost, ...) so
         preferred document types get a tie-breaker bump.
      4. If ``USE_RERANKER`` is on, score every candidate with a cross-encoder
         that reads (question, chunk) jointly and blend that with the
         weighted score. This is where the biggest quality wins come from --
         cross-encoders are far more accurate than pure kNN for relevance.
      5. Return the top-``top_k`` chunks, ordered by the final blended score.
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
                page_number=int(src.get("page_number", 0)),
            )
        )

    if config.USE_RERANKER and ranked:
        # Imported lazily so USE_RERANKER=false means the sentence-transformers
        # CrossEncoder never even loads -- saves ~2 s on startup and ~80 MB RAM.
        from Retrieval.reranker import rerank

        final_scores = rerank(
            query=query,
            chunk_texts=[c.text for c in ranked],
            weighted_scores=[c.boosted_score for c in ranked],
        )
        # Overwriting boosted_score with the blended score keeps downstream
        # code (context_dicts serialisation, out-of-scope gate) untouched.
        for chunk, blended in zip(ranked, final_scores, strict=True):
            chunk.boosted_score = blended

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
            "page_number": c.page_number,
        }
        for c in chunks
    ]
