"""Tests for trading_agent.events — fallback JSONL replay.

emit() itself is tested via every node-level test (it's the audit pipe
they all depend on). What needs its own coverage is the replay path,
because it only fires when there's accumulated buffered data and the
truncate-first pattern is easy to get subtly wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.events import replay_fallback_events


@pytest.fixture
def tmp_fallback(tmp_path: Path, monkeypatch):
    """Point events._fallback_path at a per-test tmpfile."""
    p = tmp_path / "fallback.jsonl"
    monkeypatch.setattr("trading_agent.events._fallback_path", lambda: p)
    return p


def _event_row(symbol: str = "US.AAPL") -> dict:
    return {
        "ts": "2026-05-13T12:00:00+00:00",
        "run_id": "test_run",
        "trigger": "intraday_monitor",
        "agent": "test_agent",
        "event_type": "test_event",
        "severity": 0,
        "payload": {"symbol": symbol},
        "cost_usd": None,
    }


def test_replay_noop_when_file_missing(tmp_fallback):
    """Returns clean zeros without touching DB when there's nothing to replay."""
    assert not tmp_fallback.exists()
    out = replay_fallback_events()
    assert out == {"replayed": 0, "failed": 0, "file_existed": False}


def test_replay_noop_when_file_empty(tmp_fallback):
    """Empty file is treated the same as missing — no DB hit."""
    tmp_fallback.write_text("")
    out = replay_fallback_events()
    assert out == {"replayed": 0, "failed": 0, "file_existed": False}


def test_replay_inserts_and_truncates(tmp_fallback):
    """Happy path: 3 buffered rows → 3 inserts → file empty."""
    rows = [_event_row(s) for s in ("US.AAPL", "US.SPY", "US.QQQ")]
    tmp_fallback.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    insert_calls: list[tuple] = []

    @pytest.MonkeyPatch().context()
    def _():
        pass

    # Fake cursor that records inserts and succeeds.
    fake_cur = MagicMock()

    def _execute(sql, params):
        insert_calls.append(params)
    fake_cur.execute.side_effect = _execute

    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False

    with patch("trading_agent.events.cursor", return_value=fake_ctx):
        out = replay_fallback_events()

    assert out["replayed"] == 3
    assert out["failed"] == 0
    assert out["file_existed"] is True
    assert len(insert_calls) == 3
    # File should be empty now (no leftover lines)
    assert tmp_fallback.read_text() in ("", "\n")


def test_replay_requeues_failed_rows(tmp_fallback):
    """Insert failure on row 2 → row 2 gets appended back to file."""
    rows = [_event_row(s) for s in ("US.AAPL", "US.SPY", "US.QQQ")]
    tmp_fallback.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    call_count = {"n": 0}

    def _execute(sql, params):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("connection dropped mid-replay")

    fake_cur = MagicMock()
    fake_cur.execute.side_effect = _execute
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False

    with patch("trading_agent.events.cursor", return_value=fake_ctx):
        out = replay_fallback_events()

    assert out["replayed"] == 2  # rows 1 + 3 succeeded
    assert out["failed"] == 1    # row 2 raised
    # The failed row should be back in the file, ready for next tick.
    requeued = tmp_fallback.read_text().strip().split("\n")
    assert len(requeued) == 1
    assert json.loads(requeued[0])["payload"]["symbol"] == "US.SPY"


def test_replay_drops_malformed_rows(tmp_fallback):
    """Malformed JSON → dropped (counted as failed, NOT requeued)."""
    good = json.dumps(_event_row("US.AAPL"))
    tmp_fallback.write_text(good + "\n" + "this is not json\n")

    fake_cur = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False

    with patch("trading_agent.events.cursor", return_value=fake_ctx):
        out = replay_fallback_events()

    assert out["replayed"] == 1
    assert out["failed"] == 1
    # Malformed line was NOT requeued (would loop forever otherwise).
    assert tmp_fallback.read_text() in ("", "\n")


def test_replay_max_lines_leaves_tail_for_next_tick(tmp_fallback):
    """If file has more rows than ``max_lines``, only the head is replayed
    and the tail stays for next tick (avoids blocking healthcheck)."""
    rows = [_event_row(f"US.X{i}") for i in range(10)]
    tmp_fallback.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    fake_cur = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False

    with patch("trading_agent.events.cursor", return_value=fake_ctx):
        out = replay_fallback_events(max_lines=3)

    assert out["replayed"] == 3
    # 7 remaining lines should be in the file
    remaining = [l for l in tmp_fallback.read_text().splitlines() if l.strip()]
    assert len(remaining) == 7
