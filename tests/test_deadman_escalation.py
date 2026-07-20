"""Deadman escalation on trade-engine silence (revival plan Week 1 Step 7).

The 6/20-7/8 OAuth expiry killed all EC2 LLM calls; healthcheck emitted
trade_engine_silence_detected 471 times with zero escalation. The ladder
under test: > 24 market-hours silent → priority-5 push de-bounced to one
per 6h; > 48 market-hours → auto-halt new opens via the SAME
data/halt.flag the /halt endpoint writes. Never auto-close, un-halt is
manual only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.events import SEV_CRITICAL


@pytest.fixture()
def halt_flag(tmp_path, monkeypatch):
    """Point the one true halt flag at a tmp path (same object the /halt
    endpoint and the premarket dispatch gate use)."""
    from trading_agent import halt_endpoint
    flag = tmp_path / "halt.flag"
    monkeypatch.setattr(halt_endpoint, "HALT_FLAG", flag)
    return flag


def _run_escalation(monkeypatch, silent_hours: float, emitted: list,
                    alerted: list, last_ts=None):
    """Drive _deadman_escalation with a fixed market-hours silence and a
    REAL de-bounce predicate backed by the emitted-events list (stands in
    for the agent_events query so the two-call de-bounce flow is honest)."""
    from trading_agent.graph.nodes import health_nodes as hn

    monkeypatch.setattr(
        "trading_agent.market_calendar.market_hours_between",
        lambda start, end: silent_hours,
    )
    monkeypatch.setattr(
        hn, "_was_deadman_push_recent",
        lambda: any(e["event_type"] == "deadman_silence_push" for e in emitted),
    )
    monkeypatch.setattr(hn, "emit", lambda **kw: emitted.append(kw))
    monkeypatch.setattr(hn, "_alert_ops",
                        lambda **kw: alerted.append(kw))
    ts = last_ts if last_ts is not None else (
        datetime.now(timezone.utc) - timedelta(days=10))
    hn._deadman_escalation("test_run", "healthcheck", ts)


def _events(emitted: list, event_type: str) -> list:
    return [e for e in emitted if e["event_type"] == event_type]


# ---------------------------------------------------------------------------
# 24h rung — push, de-bounced
# ---------------------------------------------------------------------------


def test_25h_pushes_once_with_debounce(monkeypatch, halt_flag):
    """25 market-hours silent → one priority-5 push; a second healthcheck
    tick inside the 6h de-bounce window emits nothing new."""
    emitted: list = []
    alerted: list = []
    _run_escalation(monkeypatch, 25.0, emitted, alerted)
    _run_escalation(monkeypatch, 25.0, emitted, alerted)  # next tick

    pushes = _events(emitted, "deadman_silence_push")
    assert len(pushes) == 1
    assert pushes[0]["payload"]["silent_hours"] == 25.0
    assert len(alerted) == 1
    assert "silent" in alerted[0]["title"].lower()
    # No halt at this rung, and never any auto-close of positions.
    assert not halt_flag.exists()
    assert _events(emitted, "deadman_auto_halt") == []


def test_below_24h_is_silent(monkeypatch, halt_flag):
    emitted: list = []
    alerted: list = []
    _run_escalation(monkeypatch, 5.0, emitted, alerted)
    assert emitted == []
    assert alerted == []
    assert not halt_flag.exists()


def test_never_dispatched_skips_escalation(monkeypatch, halt_flag):
    """Fresh deploy with no dispatch history: no heartbeat to measure from,
    auto-halting would fail the wrong direction."""
    from trading_agent.graph.nodes import health_nodes as hn
    emitted: list = []
    monkeypatch.setattr(hn, "emit", lambda **kw: emitted.append(kw))
    hn._deadman_escalation("test_run", "healthcheck", None)
    assert emitted == []
    assert not halt_flag.exists()


# ---------------------------------------------------------------------------
# 48h rung — auto-halt via the real halt mechanism
# ---------------------------------------------------------------------------


def test_49h_sets_halt_flag_via_real_mechanism(monkeypatch, halt_flag):
    emitted: list = []
    alerted: list = []
    _run_escalation(monkeypatch, 49.0, emitted, alerted)

    # The SAME flag file the /halt endpoint writes now exists.
    assert halt_flag.exists()
    halts = _events(emitted, "deadman_auto_halt")
    assert len(halts) == 1
    assert halts[0]["payload"] == {"silent_hours": 49.0}
    assert halts[0]["severity"] == SEV_CRITICAL
    # Both the halt alert and the 24h-rung push went out at priority 5
    # (_alert_ops is the fixed priority-5 ops sender).
    assert len(alerted) == 2
    assert any("AUTO-HALT" in a["title"] for a in alerted)


def test_49h_second_tick_does_not_rehalt(monkeypatch, halt_flag):
    """An existing flag (ours or the operator's) means never re-emit the
    halt — and un-halt stays manual: escalation never removes the flag."""
    emitted: list = []
    alerted: list = []
    _run_escalation(monkeypatch, 49.0, emitted, alerted)
    _run_escalation(monkeypatch, 50.0, emitted, alerted)
    assert len(_events(emitted, "deadman_auto_halt")) == 1
    assert halt_flag.exists()


# ---------------------------------------------------------------------------
# De-bounce query + node wiring
# ---------------------------------------------------------------------------


def test_debounce_query_true_when_recent_push_row():
    from trading_agent.graph.nodes.health_nodes import _was_deadman_push_recent
    fake_cur = MagicMock()
    fake_cur.fetchone.return_value = (1,)
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False
    with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
        assert _was_deadman_push_recent() is True


def test_debounce_query_fails_loud():
    """DB error → False → the priority-5 push still goes out."""
    from trading_agent.graph.nodes.health_nodes import _was_deadman_push_recent
    with patch("trading_agent.store.postgres.cursor",
               side_effect=Exception("conn refused")):
        assert _was_deadman_push_recent() is False


def test_silence_check_invokes_escalation_with_last_ts(monkeypatch):
    """trade_engine_silence_check hands the last-dispatch ts to the
    escalation ladder before its own wall-clock alert logic."""
    from trading_agent.graph.nodes import health_nodes as hn

    last_ts = datetime.now(timezone.utc) - timedelta(days=2)
    fake_cur = MagicMock()
    fake_cur.fetchone.return_value = (last_ts, 2.0)  # 2 days ago: not silent
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_cur
    fake_ctx.__exit__.return_value = False

    seen: list = []
    monkeypatch.setattr(
        hn, "_deadman_escalation",
        lambda run_id, trigger, ts: seen.append((run_id, trigger, ts)),
    )
    emitted: list = []
    monkeypatch.setattr(hn, "emit", lambda **kw: emitted.append(kw))
    with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
        hn.trade_engine_silence_check(
            {"run_id": "test_run", "trigger": "healthcheck"})
    assert seen == [("test_run", "healthcheck", last_ts)]
    # Wall-clock path unaffected: 2 days < 4 → engine reported active.
    assert [e["event_type"] for e in emitted] == ["trade_engine_active"]


# ---------------------------------------------------------------------------
# Market-hours silence arithmetic (fallback window)
# ---------------------------------------------------------------------------


def test_market_hours_between_skips_weekend():
    from trading_agent.market_calendar import _fallback_hours_between
    # Fri 2026-06-05 19:00 UTC → Mon 2026-06-08 14:30 UTC:
    # 1h of Friday's session + 0 all weekend + 1h of Monday's session.
    start = datetime(2026, 6, 5, 19, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc)
    assert _fallback_hours_between(start, end) == pytest.approx(2.0)


def test_market_hours_between_weekend_only_is_zero():
    from trading_agent.market_calendar import _fallback_hours_between
    start = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)   # Sat
    end = datetime(2026, 6, 7, 23, 59, tzinfo=timezone.utc)   # Sun
    assert _fallback_hours_between(start, end) == 0.0


def test_market_hours_between_public_fn_never_raises():
    from trading_agent.market_calendar import market_hours_between
    start = datetime(2026, 6, 5, 19, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc)
    assert isinstance(market_hours_between(start, end), float)
    assert market_hours_between(end, start) == 0.0
