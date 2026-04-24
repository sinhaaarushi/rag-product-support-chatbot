"""Metadata helpers on the query path (defensible source-match labels)."""

from __future__ import annotations

from App.app import _heuristic_follow_ups, _source_match_label


def test_source_match_label_tiers() -> None:
    assert _source_match_label(0.55) == "High"
    assert _source_match_label(0.6) == "High"
    assert _source_match_label(0.45) == "Medium"
    assert _source_match_label(0.5) == "Medium"
    assert _source_match_label(0.35) == "Low"
    assert _source_match_label(0.44) == "Low"


def test_heuristic_follow_ups_single_and_multi() -> None:
    one = _heuristic_follow_ups([{"document_name": "X/a.pdf", "page_number": 0}])
    assert len(one) == 1
    assert "a.pdf" in one[0]
    two = _heuristic_follow_ups(
        [
            {"document_name": "A/x.pdf", "page_number": 1},
            {"document_name": "B/y.pdf", "page_number": 2},
        ]
    )
    assert len(two) == 2
    assert "x.pdf" in two[0] and "y.pdf" in two[0]
