"""
Split long text into overlapping chunks for embedding and retrieval.

Design note
-----------
This is a deliberately simple character-window chunker with overlap. It is
deterministic, dependency-free, and easy to reason about, which is the right
default for a baseline RAG pipeline.

Known trade-offs:
- It does not respect sentence or paragraph boundaries, so a chunk can split
  a sentence in half. The overlap (default 50 chars) is what mitigates this.
- It counts characters, not tokens. Embedding models care about tokens, so a
  500-char chunk maps to roughly 80-130 tokens for English prose, well within
  the all-MiniLM-L6-v2 256-token limit.

Upgrade paths when retrieval quality becomes the bottleneck:
- Sentence-aware splitting (e.g. spaCy or nltk sent_tokenize) before windowing.
- Token-aware splitting using the embedding model's tokenizer for tight bounds.
- Recursive splitters (langchain's RecursiveCharacterTextSplitter) for code or
  structured documents.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Args:
        text: The raw text to split. May be empty or whitespace-only.
        chunk_size: Maximum characters per chunk. Must be > 0.
        overlap: Characters of context to repeat between adjacent chunks.
            Must satisfy ``0 <= overlap < chunk_size``. The overlap exists so
            that phrases straddling a window boundary are still indexable.

    Returns:
        A list of non-empty, non-whitespace-only chunks in document order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks
