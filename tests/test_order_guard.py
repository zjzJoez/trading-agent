"""Shared order-guard engine — thesis freshness, SELL-to-open block, audit.

This is the engine that BOTH the moomoo MCP server (authoritative,
in-process) and the Claude Code PreToolUse hook run. The regression
anchors here are the 2026-06 incidents: the MRVL 2x / AAPL 8x spreads
that entered with zero guard evaluation, and the AAPL 8x SELL-to-open
300P leg (~$235k assignment exposure, zero portfolio heat).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture()
def guard_db(tmp_path: Path, monkeypatch):
    """Tmp trader.db + redirected audit/sector paths for the guard module."""
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent import order_guard as og

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(og, "AUDIT_PATH", tmp_path / "hook_audit.log")
    monkeypatch.setattr(og, "SECTORS_CSV", tmp_path / "sectors.csv")
    db_mod.migrate(db_file)
    yield db_file


def _insert_thesis(ticker: str, minutes_ago: float = 0.0, status: str = "open") -> int:
    from trading_agent.db import connection
    created = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, 'LONG', 't', 'i', ?)",
            (created, ticker, status),
        )
        return int(cur.lastrowid)


def _insert_open_trade(symbol: str, asset_type: str = "OPT", qty: float = 1,
                       entry_price: float = 1.0) -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES (?, ?, 'BUY', ?, ?, ?, 'OPEN', 1)",
            (symbol, asset_type, qty, entry_price,
             datetime.now(timezone.utc).isoformat()),
        )
        return int(cur.lastrowid)


def _eval(kind, params, equity=100_000.0, cash=100_000.0):
    from trading_agent.order_guard import evaluate_order
    return evaluate_order(kind, params, equity=equity, cash=cash)


# ---------------------------------------------------------------------------
# SELL-to-open hard block (the AAPL 8x 300P leg)
# ---------------------------------------------------------------------------


def test_sell_to_open_option_blocked_even_with_fresh_thesis(guard_db):
    """The AAPL incident: 8x SELL-to-open 300P. Even with a fresh thesis,
    no SELL may open an option position until combos are first-class."""
    from trading_agent.order_guard import R_NO_SHORT_OPEN
    _insert_thesis("AAPL")
    d = _eval("option", {
        "option_symbol": "US.AAPL260720P00300000", "side": "SELL",
        "contracts": 8, "price": 2.50, "strategy_label": "earnings_directional_debit_spread",
    })
    assert not d.allowed
    assert d.intent == "open"
    assert any(v.rule == R_NO_SHORT_OPEN for v in d.violations)


def test_sell_to_close_option_allowed_without_thesis(guard_db):
    """SELL of an option we hold long = close. No thesis gate, no R1
    max_loss=inf block (the d089349 regression)."""
    sym = "US.AAPL260720C00200000"
    _insert_open_trade(sym, qty=6, entry_price=9.0)
    d = _eval("option", {
        "option_symbol": sym, "side": "SELL", "contracts": 6, "price": 9.55,
    })
    assert d.allowed, d.violations_json()
    assert d.intent == "close"


# ---------------------------------------------------------------------------
# Thesis freshness
# ---------------------------------------------------------------------------


def test_open_without_thesis_blocked(guard_db):
    d = _eval("option", {
        "option_symbol": "US.MRVL260702C00290000", "side": "BUY",
        "contracts": 2, "price": 1.00,
    })
    assert not d.allowed
    assert "no open thesis" in d.reason


def test_stale_thesis_blocked(guard_db):
    _insert_thesis("MRVL", minutes_ago=11)
    d = _eval("option", {
        "option_symbol": "US.MRVL260702C00290000", "side": "BUY",
        "contracts": 2, "price": 1.00,
    })
    assert not d.allowed
    assert "no open thesis" in d.reason


def test_fresh_thesis_buy_option_allowed(guard_db):
    tid = _insert_thesis("MRVL")
    d = _eval("option", {
        "option_symbol": "US.MRVL260702C00290000", "side": "BUY",
        "contracts": 1, "price": 1.00, "delta": 0.40,
    })
    assert d.allowed, d.violations_json()
    assert d.thesis_id == tid


def test_stock_sell_of_held_position_is_close(guard_db):
    """Stock exits must not demand a fresh thesis — with the gate now
    server-side, blocking exits would trap positions."""
    _insert_open_trade("US.NVDA", asset_type="STK", qty=10, entry_price=100.0)
    d = _eval("stock", {"symbol": "US.NVDA", "side": "SELL", "qty": 10, "price": 120.0})
    assert d.allowed, d.violations_json()
    assert d.intent == "close"


# ---------------------------------------------------------------------------
# Sizing pass-through (R1)
# ---------------------------------------------------------------------------


def test_r1_oversize_stock_blocked(guard_db):
    from trading_agent.sizing import R1
    _insert_thesis("NVDA")
    # risk = |200-180| * 200 = $4,000 > 2.5% of $100k ($2,500)
    d = _eval("stock", {
        "symbol": "US.NVDA", "side": "BUY", "qty": 200, "price": 200.0,
        "stop": 180.0, "target": 260.0,
    })
    assert not d.allowed
    assert any(v.rule == R1 and v.severity == "block" for v in d.violations)


# ---------------------------------------------------------------------------
# Audit trail — what the nightly reconciliation joins against
# ---------------------------------------------------------------------------


def test_audit_decision_writes_db_and_jsonl(guard_db, tmp_path):
    from trading_agent import order_guard as og
    from trading_agent.db import connection

    _insert_thesis("MRVL")
    d = _eval("option", {
        "option_symbol": "US.MRVL260702C00290000", "side": "BUY",
        "contracts": 1, "price": 1.00,
    })
    og.audit_decision(d, guard_name="server_order_guard",
                      tool_name="place_paper_option_order")

    with connection() as conn:
        rows = conn.execute(
            "SELECT hook_name, tool_name, decision, payload FROM hook_audit_log"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["hook_name"] == "server_order_guard"
    assert rows[0]["decision"] == "allow"
    payload = json.loads(rows[0]["payload"])
    assert payload["symbol"] == "US.MRVL260702C00290000"

    jsonl = (tmp_path / "hook_audit.log").read_text().strip().splitlines()
    assert len(jsonl) == 1
    assert json.loads(jsonl[0])["decision"] == "allow"


def test_audit_block_recorded(guard_db):
    from trading_agent import order_guard as og
    from trading_agent.db import connection

    d = _eval("option", {
        "option_symbol": "US.AAPL260720P00300000", "side": "SELL",
        "contracts": 8, "price": 2.50,
    })
    og.audit_decision(d, guard_name="server_order_guard",
                      tool_name="place_paper_option_order")
    with connection() as conn:
        row = conn.execute("SELECT decision, payload FROM hook_audit_log").fetchone()
    assert row["decision"] == "block"
    assert json.loads(row["payload"])["symbol"] == "US.AAPL260720P00300000"
