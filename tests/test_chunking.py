"""Unit tests for text chunking."""

from __future__ import annotations

import pytest

from Processing.chunking import chunk_text


def test_chunk_basic():
    text = "a" * 100
    out = chunk_text(text, chunk_size=30, overlap=5)
    assert len(out) >= 2
    assert all(len(c) <= 30 for c in out)


def test_chunk_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_empty_or_whitespace_only():
    assert chunk_text("   \n  ", chunk_size=10, overlap=2) == []
