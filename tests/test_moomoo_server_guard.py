"""Server-side enforcement: the moomoo order tools ARE the choke point.

Before 2026-06 the risk gate lived only in the Claude Code PreToolUse
hook, so the autonomous graph / other MCP clients / scripts placed orders
with zero evaluation (MRVL 2x + AAPL 8x spreads). These tests pin the new
invariant: place_paper_order / place_paper_option_order refuse violating
orders BEFORE any broker call, and audit every evaluation to
hook_audit_log.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def guard_db(tmp_path: Path, monkeypatch):
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


class _StubTradeCtx:
    """Fakes the two OpenSecTradeContext calls the order tools make."""

    def __init__(self, equity: float = 100_000.0, fail_accinfo: bool = False):
        self.equity = equity
        self.fail_accinfo = fail_accinfo
        self.place_order_calls: list[dict] = []

    def accinfo_query(self, **kwargs):
        if self.fail_accinfo:
            return -1, "simulated OpenD outage"
        return 0, pd.DataFrame([{"total_assets": self.equity, "cash": self.equity}])

    def place_order(self, **kwargs):
        self.place_order_calls.append(kwargs)
        return 0, pd.DataFrame([{
            "order_id": "STUB-1", "code": kwargs["code"],
            "trd_side": str(kwargs["trd_side"]), "qty": kwargs["qty"],
            "price": kwargs["price"], "dealt_qty": 0, "dealt_avg_price": 0,
            "order_status": "SUBMITTING",
        }])


@pytest.fixture()
def stub_ctx(guard_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    ctx = _StubTradeCtx()
    monkeypatch.setattr(server, "_trade", lambda: ctx)
    yield ctx


def _insert_thesis(ticker: str) -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, 'LONG', 't', 'i', 'open')",
            (datetime.now(timezone.utc).isoformat(), ticker),
        )
        return int(cur.lastrowid)


def test_sell_to_open_option_refused_before_broker(stub_ctx):
    from trading_agent.mcp_servers.moomoo import server
    _insert_thesis("AAPL")
    resp = server.place_paper_option_order(
        option_symbol="US.AAPL260720P00300000", side="SELL",
        contracts=8, price=2.50, thesis_id=1,
    )
    assert resp["order_blocked"] is True
    assert "rows" not in resp  # posttool_fill_capture must see a no-op
    assert stub_ctx.place_order_calls == []
    rules = {v["rule"] for v in resp["violations"]}
    assert "R_short_option_open_blocked" in rules


def test_no_thesis_order_refused(stub_ctx):
    from trading_agent.mcp_servers.moomoo import server
    resp = server.place_paper_order(
        symbol="US.NVDA", side="BUY", qty=10, price=100.0, thesis_id=999,
        stop=95.0, target=115.0,
    )
    assert resp["order_blocked"] is True
    assert "no open thesis" in resp["reason"]
    assert stub_ctx.place_order_calls == []


def test_valid_order_places_and_audits(stub_ctx):
    from trading_agent.db import connection
    from trading_agent.mcp_servers.moomoo import server

    _insert_thesis("NVDA")
    resp = server.place_paper_order(
        symbol="US.NVDA", side="BUY", qty=10, price=100.0, thesis_id=1,
        stop=95.0, target=115.0, strategy_label="trend",
    )
    assert "order_blocked" not in resp
    assert resp["rows"][0]["order_id"] == "STUB-1"
    assert len(stub_ctx.place_order_calls) == 1
    assert stub_ctx.place_order_calls[0]["code"] == "US.NVDA"

    with connection() as conn:
        row = conn.execute(
            "SELECT hook_name, decision FROM hook_audit_log"
        ).fetchone()
    assert row["hook_name"] == "server_order_guard"
    assert row["decision"] == "allow"


def test_equity_lookup_failure_fails_closed(guard_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    ctx = _StubTradeCtx(fail_accinfo=True)
    monkeypatch.setattr(server, "_trade", lambda: ctx)
    _insert_thesis("NVDA")
    resp = server.place_paper_order(
        symbol="US.NVDA", side="BUY", qty=10, price=100.0, thesis_id=1,
        stop=95.0, target=115.0,
    )
    assert resp["order_blocked"] is True
    assert "cannot verify sizing" in resp["reason"]
    assert ctx.place_order_calls == []
