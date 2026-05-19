"""Tests for the append-only answer feedback log."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from Utils.feedback_history import (
    answer_feedback_log_path_for_caption,
    append_answer_feedback,
    load_answer_feedback_newest_first,
)


def test_append_and_load_feedback_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "ANSWER_FEEDBACK_LOG", path)
    append_answer_feedback(
        question="What is SLA?",
        answer="The SLA is source-backed.",
        role="customer",
        rating=4,
    )
    append_answer_feedback(
        question="What plans are available?",
        answer="The available plans are listed in the docs.",
        role="sales",
        rating=5,
    )
    rows = load_answer_feedback_newest_first(max_entries=50)
    assert len(rows) == 2
    assert rows[0]["question"] == "What plans are available?"
    assert rows[0]["rating"] == 5
    assert rows[1]["role"] == "customer"


def test_feedback_rejects_invalid_rating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ANSWER_FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    with pytest.raises(ValueError, match="between 1 and 5"):
        append_answer_feedback(
            question="Question?",
            answer="Answer.",
            role="internal",
            rating=6,
        )


def test_feedback_loader_skips_malformed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "ANSWER_FEEDBACK_LOG", path)
    path.write_text(
        "\n".join(
            [
                "{not json",
                '{"question": "q", "answer": "a", "rating": 0}',
                '{"question": "q", "answer": "a", "rating": 3, "role": "internal"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_answer_feedback_newest_first()
    assert len(rows) == 1
    assert rows[0]["rating"] == 3


def test_answer_feedback_log_path_for_caption_under_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    log = root / "Data" / "logs" / "answer_feedback.jsonl"
    log.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "ANSWER_FEEDBACK_LOG", log)
    assert answer_feedback_log_path_for_caption() == "Data/logs/answer_feedback.jsonl"
