"""Phase 2.6 — outcome enrichment math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_agent.learning.outcome import _bucket_realized_r, compute_outcome


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
