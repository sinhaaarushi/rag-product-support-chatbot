"""Append-only JSONL log of answer feedback.

Ratings are collected from every audience after an answer, but the dashboard
only exposes the feedback history in Internal mode.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import config

_lock = threading.Lock()


class FeedbackRecord(TypedDict, total=False):
    ts: str
    role: str
    question: str
    answer: str
    rating: int


def answer_feedback_log_path_for_caption() -> str:
    """Path shown in the Internal panel; stable when the log is outside the repo."""
    log = config.ANSWER_FEEDBACK_LOG
    root = config.PROJECT_ROOT
    try:
        return log.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return str(log)


def append_answer_feedback(
    *,
    question: str,
    answer: str,
    role: str,
    rating: int,
) -> None:
    """Write one answer rating when the user picks 1-5 stars."""
    if rating < 1 or rating > 5:
        raise ValueError(f"Feedback rating must be between 1 and 5: {rating}")
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return
    rec: FeedbackRecord = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": str(role),
        "question": q,
        "answer": a,
        "rating": rating,
    }
    path: Path = config.ANSWER_FEEDBACK_LOG
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def load_answer_feedback_newest_first(
    *,
    max_entries: int | None = None,
) -> list[FeedbackRecord]:
    """Parse the feedback log and return newest valid records first."""
    cap = max_entries if max_entries is not None else config.ANSWER_FEEDBACK_PANEL_MAX
    path = config.ANSWER_FEEDBACK_LOG
    if not path.is_file():
        return []
    with _lock:
        raw = path.read_text(encoding="utf-8", errors="replace")
    indexed_rows: list[tuple[int, FeedbackRecord]] = []
    for line_index, raw_line in enumerate(raw.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        question = str(obj.get("question", "")).strip()
        answer = str(obj.get("answer", "")).strip()
        try:
            rating = int(obj.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if not question or not answer or rating < 1 or rating > 5:
            continue
        rec: FeedbackRecord = {
            "ts": str(obj.get("ts", "")),
            "role": str(obj.get("role", "")),
            "question": question,
            "answer": answer,
            "rating": rating,
        }
        indexed_rows.append((line_index, rec))
    indexed_rows.sort(key=lambda pair: -pair[0])
    return [rec for (_line_index, rec) in indexed_rows[:cap]]
