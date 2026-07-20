"""R8 daily -2R circuit breaker — order-path tests.

Revival plan Week 1 Step 7: once TODAY's realized losing trades sum to
-2R (R = sizing.MAX_SINGLE_RISK_PCT x live equity, the same R1 risk
unit), NEW opens are refused for the rest of the trading day. Closes are
never touched, and total store blindness fails closed for opens only.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

EQUITY = 100_000.0
# R = 2.5% x 100k = $2,500 → breaker trips at -$5,000.
R_UNIT = 0.025 * EQUITY


@pytest.fixture()
def guard_db(tmp_path: Path, monkeypatch):
    """Tmp trader.db + redirected audit/sector paths for the guard module.

    Also pins the Postgres side of the breaker to "reachable, no losses"
    so every test states its PG behavior explicitly (the real cursor would
    otherwise fail as unreachable and add a store-degraded warn).
    """
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent import order_guard as og

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(og, "AUDIT_PATH", tmp_path / "hook_audit.log")
    monkeypatch.setattr(og, "SECTORS_CSV", tmp_path / "sectors.csv")
    monkeypatch.setattr(og, "_pg_todays_losses", lambda: 0.0)
    db_mod.migrate(db_file)
    yield db_file


def _insert_thesis(ticker: str) -> int:
    from trading_agent.db import connection
    created = datetime.now(timezone.utc).isoformat()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, 'LONG', 't', 'i', 'open')",
            (created, ticker),
        )
        return int(cur.lastrowid)


def _insert_open_trade(symbol: str, asset_type: str = "STK", qty: float = 10,
                       entry_price: float = 100.0) -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES (?, ?, 'BUY', ?, ?, ?, 'OPEN', 1)",
            (symbol, asset_type, qty, entry_price,
             datetime.now(timezone.utc).isoformat()),
        )
        return int(cur.lastrowid)


def _insert_closed_trade(pnl: float, *, provenance: str = "agent",
                         days_ago: float = 0.0, symbol: str = "US.XYZ") -> int:
    from trading_agent.db import connection
    closed = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    outcome = "LOSS" if pnl < 0 else ("WIN" if pnl > 0 else "SCRATCH")
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, closed_at, outcome, pnl, is_paper, provenance) "
            "VALUES (?, 'STK', 'BUY', 10, 100.0, ?, ?, ?, ?, 1, ?)",
            (symbol, closed, closed, outcome, pnl, provenance),
        )
        return int(cur.lastrowid)


def _eval_stock_open(ticker: str = "AAPL"):
    from trading_agent.order_guard import evaluate_order
    return evaluate_order("stock", {
        "symbol": f"US.{ticker}", "side": "BUY", "qty": 10,
        "price": 100.0, "stop": 95.0, "target": 110.0,
    }, equity=EQUITY, cash=EQUITY)


def _eval_stock_close(symbol: str):
    from trading_agent.order_guard import evaluate_order
    return evaluate_order("stock", {
        "symbol": symbol, "side": "SELL", "qty": 10, "price": 90.0,
    }, equity=EQUITY, cash=EQUITY)


# ---------------------------------------------------------------------------
# The trip wire itself
# ---------------------------------------------------------------------------


def test_open_blocked_at_minus_2_1R(guard_db):
    from trading_agent.order_guard import R8
    _insert_closed_trade(-1.0 * R_UNIT)
    _insert_closed_trade(-1.1 * R_UNIT)   # total -2.1R today
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert not d.allowed
    assert d.reason == "daily loss circuit breaker tripped (R8)"
    assert any(v.rule == R8 for v in d.violations)


def test_close_allowed_when_breaker_tripped(guard_db):
    """Closing/reducing must never be trapped by the breaker."""
    _insert_closed_trade(-2.1 * R_UNIT)
    _insert_open_trade("US.AAPL")
    d = _eval_stock_close("US.AAPL")
    assert d.allowed, d.violations_json()
    assert d.intent == "close"


def test_open_allowed_at_minus_1_5R(guard_db):
    from trading_agent.order_guard import R8
    _insert_closed_trade(-1.5 * R_UNIT)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert d.allowed, d.violations_json()
    assert not any(v.rule == R8 for v in d.violations)


def test_wins_do_not_rearm_the_breaker(guard_db):
    """Losses-only sum: a green morning must not offset a red afternoon."""
    _insert_closed_trade(+4.0 * R_UNIT)
    _insert_closed_trade(-2.1 * R_UNIT)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert not d.allowed
    assert d.reason == "daily loss circuit breaker tripped (R8)"


def test_yesterdays_losses_do_not_count(guard_db):
    _insert_closed_trade(-3.0 * R_UNIT, days_ago=1.5)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert d.allowed, d.violations_json()


def test_excluded_provenance_does_not_trip(guard_db):
    """real_mirror / virtual_backfill / dry_run rows are not the system's
    own live results and must not trip the breaker."""
    for prov in ("real_mirror", "virtual_backfill", "dry_run"):
        _insert_closed_trade(-10.0 * R_UNIT, provenance=prov)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert d.allowed, d.violations_json()


def test_postgres_losses_count_toward_the_sum(guard_db, monkeypatch):
    """Autonomous-graph trades live only in journal_trades — their losses
    must reach the breaker too."""
    from trading_agent import order_guard as og
    monkeypatch.setattr(og, "_pg_todays_losses", lambda: -2.1 * R_UNIT)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert not d.allowed
    assert d.reason == "daily loss circuit breaker tripped (R8)"


# ---------------------------------------------------------------------------
# Store-failure semantics
# ---------------------------------------------------------------------------


def _raise(*_a, **_k):
    raise RuntimeError("store unreachable")


def test_one_store_down_warns_and_uses_other(guard_db, monkeypatch):
    from trading_agent import order_guard as og
    from trading_agent.order_guard import R8, R8_STORE_DEGRADED
    monkeypatch.setattr(og, "_pg_todays_losses", _raise)
    _insert_closed_trade(-2.1 * R_UNIT)
    _insert_thesis("AAPL")
    d = _eval_stock_open("AAPL")
    assert not d.allowed
    assert any(v.rule == R8 for v in d.violations)
    assert any(v.rule == R8_STORE_DEGRADED and v.severity == "warn"
               for v in d.violations)


def test_store_blind_open_blocked(guard_db, monkeypatch):
    """BOTH journals unreachable + open → block (fail-closed on total
    blindness: the breaker cannot be proven un-tripped)."""
    from trading_agent import order_guard as og
    from trading_agent.order_guard import R8_STORE_BLIND
    _insert_thesis("AAPL")
    monkeypatch.setattr(og, "_sqlite_todays_losses", _raise)
    monkeypatch.setattr(og, "_pg_todays_losses", _raise)
    d = _eval_stock_open("AAPL")
    assert not d.allowed
    assert d.reason == "daily loss circuit breaker tripped (R8)"
    assert any(v.rule == R8_STORE_BLIND for v in d.violations)


def test_store_blind_close_allowed(guard_db, monkeypatch):
    """A dead DB must not brick a close — R8 never evaluates for closes."""
    from trading_agent import order_guard as og
    _insert_open_trade("US.AAPL")
    monkeypatch.setattr(og, "_sqlite_todays_losses", _raise)
    monkeypatch.setattr(og, "_pg_todays_losses", _raise)
    d = _eval_stock_close("US.AAPL")
    assert d.allowed, d.violations_json()
    assert d.intent == "close"


def test_check_daily_loss_breaker_close_intent_is_noop(guard_db, monkeypatch):
    """Unit-level: intent='close' returns [] without touching any store."""
    from trading_agent import order_guard as og
    monkeypatch.setattr(og, "_sqlite_todays_losses", _raise)
    monkeypatch.setattr(og, "_pg_todays_losses", _raise)
    assert og.check_daily_loss_breaker(EQUITY, "close") == []


def test_combo_open_also_gated_by_breaker(guard_db, monkeypatch):
    """evaluate_combo runs R8 too — a spread is still a NEW position."""
    from trading_agent import order_guard as og
    from trading_agent.order_guard import R8, evaluate_combo
    _insert_closed_trade(-2.1 * R_UNIT)
    _insert_thesis("MRVL")
    monkeypatch.setattr(og, "fetch_option_quote",
                        lambda code, timeout_s=5.0:
                        {"bid_price": 1.00, "ask_price": 1.02,
                         "option_open_interest": 1200})
    exp = (datetime.now(timezone.utc).date()
           + timedelta(days=30)).strftime("%y%m%d")
    d = evaluate_combo({
        "short_leg_symbol": f"US.MRVL{exp}P00280000",
        "long_leg_symbol": f"US.MRVL{exp}P00270000",
        "contracts": 1, "short_price": 3.0, "long_price": 1.5,
    }, equity=EQUITY, cash=EQUITY)
    assert not d.allowed
    assert any(v.rule == R8 for v in d.violations)
