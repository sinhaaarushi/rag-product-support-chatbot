"""Append-only JSONL log of user questions (all roles).

Stored under ``Data/logs/`` (gitignored). Used by the internal dashboard
panel and fed from ``query_documents`` so CLI/eval paths record too.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import config

_lock = threading.Lock()


def _iso_timestamp_for_sort(iso_ts: str) -> float:
    """Monotonic-ish key so frequency ties honour recency."""
    raw = str(iso_ts or "").strip()
    if not raw:
        return 0.0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


class QuestionRecord(TypedDict, total=False):
    ts: str
    role: str
    question: str


def user_questions_log_path_for_caption() -> str:
    """Path to show in the Internal panel; stable when the log lies outside ``PROJECT_ROOT``."""
    log = config.USER_QUESTIONS_LOG
    root = config.PROJECT_ROOT
    try:
        return log.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return str(log)


def append_user_question(question: str, role: str) -> None:
    """Write one JSON line. Idempotent with empty input."""
    q = (question or "").strip()
    if not q:
        return
    rec: QuestionRecord = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": str(role),
        "question": q,
    }
    path: Path = config.USER_QUESTIONS_LOG
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def question_normalization_key(question: str) -> str:
    """Collapse runs of whitespace and case-fold so near-duplicates tally together."""
    return " ".join((question or "").strip().split()).casefold()


def top_questions_by_frequency(*, top_k: int) -> list[str]:
    """Return the ``top_k`` most-asked questions from the append-only JSONL log.

    Grouping uses ``question_normalization_key``. The label shown for each group
    is the **latest** wording seen (file order is oldest-to-newest, so newer lines win).
    Ties break toward the wording asked **most recently** (latest ``ts`` in the log wins).

    Reads the whole log once; keep ``USER_QUESTIONS_LOG`` bounded in production if
    the file grows unwieldy.
    """
    if top_k <= 0:
        return []
    path = config.USER_QUESTIONS_LOG
    if not path.is_file():
        return []
    with _lock:
        raw = path.read_text(encoding="utf-8", errors="replace")
    aggregated: dict[str, tuple[int, str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        q_raw = str(obj.get("question", "")).strip()
        if not q_raw:
            continue
        nk = question_normalization_key(q_raw)
        ts = str(obj.get("ts", ""))
        bucket = aggregated.get(nk)
        if bucket is None:
            aggregated[nk] = (1, q_raw, ts)
        else:
            count, _old_text, _old_ts = bucket
            aggregated[nk] = (count + 1, q_raw, ts)

    ranked = sorted(
        aggregated.values(),
        key=lambda record: (-record[0], -_iso_timestamp_for_sort(record[2])),
    )
    return [display for (_count, display, _ts) in ranked[:top_k]]


def load_user_questions_newest_first(*, max_entries: int | None = None) -> list[QuestionRecord]:
    """Parse the log; return newest first. No filtering by audience — every stored role is included."""
    cap = max_entries if max_entries is not None else config.USER_QUESTIONS_PANEL_MAX
    path = config.USER_QUESTIONS_LOG
    if not path.is_file():
        return []
    with _lock:
        raw = path.read_text(encoding="utf-8", errors="replace")
    indexed_rows: list[tuple[int, QuestionRecord]] = []
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
        q = str(obj.get("question", "")).strip()
        if not q:
            continue
        rec: QuestionRecord = {
            "ts": str(obj.get("ts", "")),
            "role": str(obj.get("role", "")),
            "question": q,
        }
        indexed_rows.append((line_index, rec))
    indexed_rows.sort(
        key=lambda pair: (
            -_iso_timestamp_for_sort(pair[1].get("ts", "")),
            -pair[0],
        ),
    )
    rows = [rec for (_, rec) in indexed_rows[:cap]]
    return rows


def format_ts_short(iso_ts: str) -> str:
    """Human-readable UTC stamp for the panel; falls back to raw string."""
    if not iso_ts:
        return "—"
    try:
        if iso_ts.endswith("Z"):
            iso_ts = iso_ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_ts[:19] if len(iso_ts) >= 19 else iso_ts
