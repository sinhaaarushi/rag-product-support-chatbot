"""Tests for the append-only user question log."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from Utils.question_history import (
    append_user_question,
    format_ts_short,
    load_user_questions_newest_first,
    question_normalization_key,
    top_questions_by_frequency,
    user_questions_log_path_for_caption,
)


def test_append_and_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "q.jsonl"
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", path)
    append_user_question("What is SLA?", "internal")
    append_user_question("Pricing?", "customer")
    rows = load_user_questions_newest_first(max_entries=50)
    assert len(rows) == 2
    assert rows[0]["question"] == "Pricing?"
    assert rows[1]["question"] == "What is SLA?"
    assert rows[0]["role"] == "customer"


def test_format_ts_short_accepts_z_suffix() -> None:
    s = format_ts_short("2026-04-28T12:00:00+00:00")
    assert "2026-04-28" in s
    assert "UTC" in s


def test_malformed_lines_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "q.jsonl"
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", path)
    path.write_text("{not json\n", encoding="utf-8")
    assert load_user_questions_newest_first() == []


def test_user_questions_log_path_for_caption_under_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    log = root / "Data" / "logs" / "q.jsonl"
    log.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", log)
    assert user_questions_log_path_for_caption() == "Data/logs/q.jsonl"


def test_user_questions_log_path_for_caption_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere" / "q.jsonl"
    elsewhere.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", elsewhere)
    assert user_questions_log_path_for_caption() == str(elsewhere)


def test_question_normalization_key_collapses_space_and_case() -> None:
    assert question_normalization_key("  What\tis  SLA?  ") == question_normalization_key(
        "what is sla?"
    )


def test_top_questions_by_frequency_orders_by_count_and_recency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "freq.jsonl"
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", path)
    lines = [
        '{"ts": "2026-01-01T10:00:00+00:00", "role": "customer", "question": "beta?"}',
        '{"ts": "2026-01-02T10:00:00+00:00", "role": "customer", "question": "alpha?"}',
        '{"ts": "2026-01-03T10:00:00+00:00", "role": "customer", "question": "Alpha?"}',
        '{"ts": "2026-01-04T10:00:00+00:00", "role": "customer", "question": "beta?"}',
        '{"ts": "2026-01-05T10:00:00+00:00", "role": "customer", "question": "gamma?"}',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    top2 = top_questions_by_frequency(top_k=2)
    assert top2 == ["beta?", "Alpha?"]
    top3 = top_questions_by_frequency(top_k=3)
    assert top3 == ["beta?", "Alpha?", "gamma?"]


def test_top_questions_by_frequency_empty_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "empty.jsonl"
    monkeypatch.setattr(config, "USER_QUESTIONS_LOG", path)
    assert top_questions_by_frequency(top_k=4) == []
