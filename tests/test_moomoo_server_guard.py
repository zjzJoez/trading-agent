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
from datetime import datetime, timedelta, timezone
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


def _future_opt(ticker: str = "MRVL", days: int = 30, right: str = "C",
                strike: float = 290.0) -> str:
    """Evergreen ~`days`-DTE option symbol (keeps R5's [14,60] DTE band
    satisfied as real time advances)."""
    exp = (datetime.now(timezone.utc).date() + timedelta(days=days)).strftime("%y%m%d")
    return f"US.{ticker}{exp}{right}{int(round(strike * 1000)):08d}"


_MRVL_OPT = _future_opt("MRVL", 30, "C", 290.0)


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


def test_illiquid_option_open_refused_before_broker(stub_ctx, monkeypatch):
    """R5d wired through the order tool: the autonomous synthesizer's
    buy-at-the-ask path gets a live spread/OI check at the choke point,
    and the violation surfaces in the tool's response."""
    from trading_agent import order_guard as og
    from trading_agent.mcp_servers.moomoo import server

    _insert_thesis("MRVL")
    monkeypatch.setattr(
        og, "fetch_option_quote",
        lambda code, timeout_s=5.0: {
            "bid_price": 1.00, "ask_price": 1.40, "option_open_interest": 50,
        },
    )
    resp = server.place_paper_option_order(
        option_symbol=_MRVL_OPT, side="BUY",
        contracts=1, price=1.40, thesis_id=1,
    )
    assert resp["order_blocked"] is True
    assert "rows" not in resp
    assert stub_ctx.place_order_calls == []
    rules = {v["rule"] for v in resp["violations"]}
    assert "R5d_illiquid" in rules


def test_liquid_option_open_places(stub_ctx, monkeypatch):
    from trading_agent import order_guard as og
    from trading_agent.mcp_servers.moomoo import server

    _insert_thesis("MRVL")
    monkeypatch.setattr(
        og, "fetch_option_quote",
        lambda code, timeout_s=5.0: {
            "bid_price": 1.00, "ask_price": 1.02, "option_open_interest": 1200,
        },
    )
    resp = server.place_paper_option_order(
        option_symbol=_MRVL_OPT, side="BUY",
        contracts=1, price=1.02, thesis_id=1, delta=0.40,
    )
    assert "order_blocked" not in resp
    assert len(stub_ctx.place_order_calls) == 1
    assert stub_ctx.place_order_calls[0]["code"] == _MRVL_OPT


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


# ---------------------------------------------------------------------------
# place_paper_option_combo (defined-risk vertical, area A)
# ---------------------------------------------------------------------------

_GOOD_QUOTE = {"bid_price": 1.00, "ask_price": 1.02, "option_open_interest": 1200}
_SHORT_PUT = _future_opt("SPY", 30, "P", 100.0)
_LONG_PUT = _future_opt("SPY", 30, "P", 95.0)


def _patch_combo_quote(monkeypatch, row=None):
    from trading_agent import order_guard as og
    monkeypatch.setattr(og, "fetch_option_quote",
                        lambda code, timeout_s=5.0: dict(row or _GOOD_QUOTE))


def test_valid_combo_places_both_legs_long_first(stub_ctx, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _insert_thesis("SPY")
    _patch_combo_quote(monkeypatch)
    resp = server.place_paper_option_combo(
        short_leg_symbol=_SHORT_PUT, long_leg_symbol=_LONG_PUT,
        contracts=1, short_price=2.0, long_price=0.8, thesis_id=1,
        strategy_label="credit_put_spread_30_45",
    )
    assert "order_blocked" not in resp
    assert resp["combo"] is True
    assert resp["net_credit"] == 2.0 - 0.8
    assert resp["max_loss"] == (5.0 - 1.2) * 100.0
    assert "rows" not in resp and len(resp["combo_rows"]) == 2
    # protective LONG leg placed FIRST, then the SHORT leg
    assert len(stub_ctx.place_order_calls) == 2
    assert stub_ctx.place_order_calls[0]["code"] == _LONG_PUT
    assert str(stub_ctx.place_order_calls[0]["trd_side"]).endswith("BUY")
    assert stub_ctx.place_order_calls[1]["code"] == _SHORT_PUT


def test_combo_not_defined_risk_blocked_before_broker(stub_ctx, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _insert_thesis("SPY")
    _patch_combo_quote(monkeypatch)
    # long strike ABOVE short on a put spread → not capped → R5e block
    resp = server.place_paper_option_combo(
        short_leg_symbol=_future_opt("SPY", 30, "P", 95.0),
        long_leg_symbol=_future_opt("SPY", 30, "P", 100.0),
        contracts=1, short_price=2.0, long_price=0.8, thesis_id=1,
    )
    assert resp["order_blocked"] is True
    assert "combo_rows" not in resp and "rows" not in resp
    assert stub_ctx.place_order_calls == []
    assert "R5e_combo_defined_risk" in {v["rule"] for v in resp["violations"]}


def test_combo_short_leg_rejection_rolls_back_long(guard_db, monkeypatch):
    """Broker fills the long leg then rejects the short — the tool must close
    the long so nothing is left as an unintended naked long."""
    from trading_agent.mcp_servers.moomoo import server

    class _FailShortLeg(_StubTradeCtx):
        def place_order(self, **kwargs):
            idx = len(self.place_order_calls)
            self.place_order_calls.append(kwargs)
            if idx == 1:  # the SHORT opening leg (2nd call) is rejected
                return -1, "simulated short-leg rejection"
            return 0, pd.DataFrame([{
                "order_id": f"STUB-{idx}", "code": kwargs["code"],
                "trd_side": str(kwargs["trd_side"]), "qty": kwargs["qty"],
                "price": kwargs["price"], "order_status": "SUBMITTING",
            }])

    ctx = _FailShortLeg()
    monkeypatch.setattr(server, "_trade", lambda: ctx)
    _insert_thesis("SPY")
    _patch_combo_quote(monkeypatch)
    resp = server.place_paper_option_combo(
        short_leg_symbol=_SHORT_PUT, long_leg_symbol=_LONG_PUT,
        contracts=1, short_price=2.0, long_price=0.8, thesis_id=1,
    )
    assert resp.get("combo") is False
    assert resp.get("combo_rollback") is True
    assert resp.get("rolled_back_long") is True
    # 3 calls: long BUY open, short SELL (fails), long SELL-to-close rollback
    assert len(ctx.place_order_calls) == 3
    assert ctx.place_order_calls[0]["code"] == _LONG_PUT
    assert str(ctx.place_order_calls[2]["trd_side"]).endswith("SELL")
    assert ctx.place_order_calls[2]["code"] == _LONG_PUT


def test_combo_requires_fresh_thesis(stub_ctx, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _patch_combo_quote(monkeypatch)
    resp = server.place_paper_option_combo(
        short_leg_symbol=_SHORT_PUT, long_leg_symbol=_LONG_PUT,
        contracts=1, short_price=2.0, long_price=0.8, thesis_id=1,
    )
    assert resp["order_blocked"] is True
    assert "no open thesis" in resp["reason"]
    assert stub_ctx.place_order_calls == []
