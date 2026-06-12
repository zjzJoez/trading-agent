"""Phase 2.6 — outcome enrichment math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from trading_agent.learning.outcome import (
    CLOSED_OUTCOMES,
    _bucket_realized_r,
    check_outcome_completeness,
    compute_outcome,
    enrich_closed_trades,
    outcome_for_pnl,
)


def _pg_ctx(cur: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def test_realized_r_winner():
    row = {
        "entry_price": 10.0,
        "exit_price": 12.0,
        "stop": 9.0,
        "opened_at": datetime(2026, 4, 28, 13, tzinfo=timezone.utc),
        "closed_at": datetime(2026, 4, 30, 13, tzinfo=timezone.utc),
        "broker_fill_json": {},
    }
    m = compute_outcome(row)
    # entry-stop=1, exit-entry=2 → realized R = 2.0
    assert m.realized_r is not None
    assert abs(m.realized_r - 2.0) < 1e-6
    assert m.holding_days == 2.0


def test_realized_r_loser():
    row = {
        "entry_price": 10.0,
        "exit_price": 8.5,
        "stop": 9.0,
        "opened_at": datetime(2026, 4, 28, 13, tzinfo=timezone.utc),
        "closed_at": datetime(2026, 4, 28, 15, tzinfo=timezone.utc),
        "broker_fill_json": {},
    }
    m = compute_outcome(row)
    # entry-stop=1, exit-entry=-1.5 → realized R = -1.5
    assert abs(m.realized_r + 1.5) < 1e-6
    assert m.holding_days < 1.0


def test_holding_days_handles_missing_dates():
    row = {
        "entry_price": 10.0, "exit_price": 11.0, "stop": 9.0,
        "opened_at": None, "closed_at": None, "broker_fill_json": {},
    }
    m = compute_outcome(row)
    assert m.holding_days is None


def test_slippage_bps_computed_when_both_prices_present():
    row = {
        "entry_price": 10.0, "exit_price": 11.0, "stop": 9.0,
        "opened_at": datetime.now(timezone.utc),
        "closed_at": datetime.now(timezone.utc) + timedelta(days=1),
        "broker_fill_json": {"requested_price": 10.00, "avg_fill_price": 10.05},
    }
    m = compute_outcome(row)
    # 5 cents over 10.00 → 50 bps
    assert m.slippage_bps == 50.0


def test_no_stop_means_no_realized_r():
    row = {
        "entry_price": 10.0, "exit_price": 11.0, "stop": None,
        "opened_at": datetime.now(timezone.utc),
        "closed_at": datetime.now(timezone.utc),
        "broker_fill_json": {},
    }
    m = compute_outcome(row)
    assert m.realized_r is None


def test_bucket_realized_r_boundaries():
    assert _bucket_realized_r(2.5) == "win_2R+"
    assert _bucket_realized_r(1.0) == "win_1R"
    assert _bucket_realized_r(0.4) == "win_partial"
    assert _bucket_realized_r(-0.3) == "small_loss"
    assert _bucket_realized_r(-0.9) == "full_R_loss"
    assert _bucket_realized_r(-1.5) == "exceeded_R_loss"


# -- writer/reader outcome-literal alignment ----------------------------------

def test_outcome_for_pnl_emits_only_closed_literals():
    assert outcome_for_pnl(123.45) == "WIN"
    assert outcome_for_pnl(-0.01) == "LOSS"
    assert outcome_for_pnl(0.0) == "SCRATCH"
    assert set(CLOSED_OUTCOMES) == {"WIN", "LOSS", "SCRATCH"}


def test_journal_mcp_close_trade_literal_matches_closed_outcomes():
    """The journal MCP writer's accepted outcomes == the readers' filter set."""
    from typing import get_args, get_type_hints

    from trading_agent.mcp_servers.journal.server import close_trade
    hints = get_type_hints(close_trade)
    assert set(get_args(hints["outcome"])) == set(CLOSED_OUTCOMES)


def test_enrich_reader_filters_on_writer_literals():
    """Regression: the enrichment query filtered on CLOSED/STOPPED/TARGET_HIT,
    which no writer produces — trade_outcome_features stayed permanently
    empty.  The reader must parameterize on CLOSED_OUTCOMES."""
    cur = MagicMock()
    cur.fetchall.return_value = []
    with patch("trading_agent.learning.outcome.cursor", return_value=_pg_ctx(cur)):
        assert enrich_closed_trades() == []
    sql, params = cur.execute.call_args[0]
    assert "= ANY(%s)" in sql
    assert params[0] == list(CLOSED_OUTCOMES)
    for dead in ("'CLOSED'", "'STOPPED'", "'TARGET_HIT'"):
        assert dead not in sql


def test_enrich_closed_trades_enriches_win_row():
    """End-to-end through the mock DB: a WIN row (the literal the intraday
    exit path actually writes) must produce a trade_outcome_features insert."""
    opened = datetime(2026, 6, 1, 14, 30, tzinfo=timezone.utc)
    closed = datetime(2026, 6, 3, 14, 30, tzinfo=timezone.utc)
    row = (7, 10.0, 12.0, 9.0, opened, closed, {}, None, None, None, None,
           "WIN", None, None)
    cur = MagicMock()
    cur.fetchall.return_value = [row]
    cur.fetchone.return_value = (99,)  # RETURNING id of the insert
    with patch("trading_agent.learning.outcome.cursor", return_value=_pg_ctx(cur)), \
         patch("trading_agent.learning.outcome.emit"), \
         patch("trading_agent.learning.excursion.read_extremes",
               return_value=(None, None)):
        enriched = enrich_closed_trades()
    assert enriched == [7]
    inserts = [c.args[0] for c in cur.execute.call_args_list
               if "INSERT INTO trade_outcome_features" in c.args[0]]
    assert len(inserts) == 1


# -- nightly completeness check -----------------------------------------------

def test_completeness_check_alerts_on_gap():
    cur = MagicMock()
    cur.fetchone.return_value = (3,)
    with patch("trading_agent.learning.outcome.cursor", return_value=_pg_ctx(cur)):
        with patch("trading_agent.notify.send") as ntfy:
            assert check_outcome_completeness() == 3
    ntfy.assert_called_once()
    assert ntfy.call_args.kwargs["priority"] == 4


def test_completeness_check_quiet_when_complete():
    cur = MagicMock()
    cur.fetchone.return_value = (0,)
    with patch("trading_agent.learning.outcome.cursor", return_value=_pg_ctx(cur)):
        with patch("trading_agent.notify.send") as ntfy:
            assert check_outcome_completeness() == 0
    ntfy.assert_not_called()


def test_completeness_check_degrades_on_db_error():
    with patch("trading_agent.learning.outcome.cursor",
               side_effect=Exception("db down")):
        with patch("trading_agent.notify.send") as ntfy:
            assert check_outcome_completeness() == -1
    ntfy.assert_not_called()
