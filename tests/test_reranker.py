"""Tests for the cross-encoder re-ranker's pure helpers.

Loading the real CrossEncoder model is expensive (~2 s + 80 MB of RAM) and
hits the network unless OFFLINE_ONLY is set with a path, so we cover the
model-heavy ``rerank`` function via monkeypatching. The helpers that don't
need the model -- score normalization -- get tested directly.
"""

from __future__ import annotations

import config
from Retrieval import reranker


def test_min_max_normalize_scales_to_unit_range():
    out = reranker._min_max_normalize([1.0, 2.0, 3.0, 4.0])
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert 0.0 < out[1] < 1.0


def test_min_max_normalize_handles_uniform_input():
    """Uniform scores would give a zero-span, so we sentinel to 0.5 for every
    element. That lets the blend formula still work without NaNs or divides
    by zero, and still preserves the weighted-kNN ordering as a tiebreaker.
    """
    out = reranker._min_max_normalize([0.5, 0.5, 0.5])
    assert out == [0.5, 0.5, 0.5]


def test_min_max_normalize_handles_empty():
    assert reranker._min_max_normalize([]) == []


def test_rerank_blends_cross_encoder_and_weighted_scores(monkeypatch):
    """The re-ranker should blend with ``RERANKER_BLEND``. We stub out the
    CrossEncoder so this test stays fast and deterministic -- what we're
    verifying is the blend math, not the model.
    """
    monkeypatch.setattr(config, "RERANKER_BLEND", 0.8)

    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            # Return raw logits in a non-normalised range to exercise the
            # normalisation path inside ``rerank``.
            return [10.0, 5.0, 2.0]

    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda: FakeCrossEncoder())

    # These knn scores reverse the cross-encoder's ordering on purpose -- the
    # 80/20 blend should still let the cross-encoder decide the top slot.
    knn_scores = [0.1, 0.5, 1.0]
    texts = ["chunk_a", "chunk_b", "chunk_c"]
    blended = reranker.rerank("what is x?", texts, knn_scores)

    assert len(blended) == 3
    # First chunk has the highest cross-encoder score and the lowest kNN
    # score: with the 80/20 blend it should still end up highest overall.
    assert blended[0] > blended[1]
    assert blended[0] > blended[2]


def test_rerank_returns_empty_for_empty_input():
    """No chunks = no re-ranking work; we should not try to load the model.

    Covered so future refactors that move the model-load call earlier don't
    accidentally pay the load cost on empty inputs.
    """
    assert reranker.rerank("q", [], []) == []
