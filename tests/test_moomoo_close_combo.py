"""close_paper_option_combo — the atomic two-leg close tool (M1-0.1).

Placement order is load-bearing: BUY-to-close the SHORT leg FIRST (removes
assignment risk), then SELL-to-close the LONG leg. The rollback matrix is
deliberately asymmetric with the open path: a just-bought-back short is
NEVER re-sold (that would re-open the assignment risk the ordering exists
to remove) — the residual long wing is retried via the pending machinery.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


def _future_opt(ticker="SPY", days=40, right="P", strike=100.0) -> str:
    exp = (datetime.now(UTC).date()
           + timedelta(days=days)).strftime("%y%m%d")
    return f"US.{ticker}{exp}{right}{int(round(strike * 1000)):08d}"


SHORT_PUT = _future_opt("SPY", 40, "P", 100.0)
LONG_PUT = _future_opt("SPY", 40, "P", 95.0)


@pytest.fixture()
def close_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent import order_guard as og

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(og, "AUDIT_PATH", tmp_path / "hook_audit.log")
    db_mod.migrate(db_file)
    # Postgres never reachable in tests — the sqlite journal is the proof.
    def _boom():
        raise ConnectionError("no postgres in tests")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    yield db_file


def _journal_open_combo(contracts: float = 2.0,
                        short=SHORT_PUT, long=LONG_PUT) -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES (?, 'OPT', 'SELL', ?, 1.2, "
            "?, 'OPEN', 1)",
            (short, contracts, datetime.now(UTC).isoformat()),
        )
        tid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, datetime.now(UTC).isoformat(), json.dumps({
                "combo": True, "short_leg": short, "long_leg": long,
                "net_credit": 1.2, "width": 5.0, "max_loss": 380.0,
                "contracts": contracts, "broker_order_ids": ["L1", "S1"],
            })),
        )
        conn.commit()
    return tid


class _CloseStubCtx:
    """Stubs position_list_query + place_order + modify_order for the tool."""

    def __init__(self, positions=None, reject_leg_indexes=(),
                 leg1_dealt_qty=0.0, cancel_fails=False):
        self.positions = positions if positions is not None else [
            {"code": SHORT_PUT, "qty": 2, "position_side": "SHORT"},
            {"code": LONG_PUT, "qty": 2, "position_side": "LONG"},
        ]
        self.reject_leg_indexes = set(reject_leg_indexes)
        self.leg1_dealt_qty = leg1_dealt_qty
        self.cancel_fails = cancel_fails
        self.place_order_calls: list[dict] = []
        self.modify_order_calls: list[dict] = []

    def position_list_query(self, **kw):
        return 0, pd.DataFrame(self.positions or [
            {"code": "US.NONE", "qty": 0, "position_side": "LONG"}])

    def place_order(self, **kw):
        idx = len(self.place_order_calls)
        self.place_order_calls.append(kw)
        if idx in self.reject_leg_indexes:
            return -1, f"simulated rejection of leg {idx}"
        dealt = self.leg1_dealt_qty if idx == 0 else 0.0
        return 0, pd.DataFrame([{
            "order_id": f"CLOSE-{idx}", "code": kw["code"],
            "trd_side": str(kw["trd_side"]), "qty": kw["qty"],
            "price": kw["price"], "dealt_qty": dealt,
            "dealt_avg_price": kw["price"] if dealt else 0,
            "order_status": "FILLED_ALL" if dealt else "SUBMITTING",
        }])

    def modify_order(self, **kw):
        self.modify_order_calls.append(kw)
        if self.cancel_fails:
            return -1, "cancel failed — racing a fill"
        return 0, pd.DataFrame([{"order_id": kw.get("order_id", ""),
                                 "order_status": "CANCELLED_ALL"}])


@pytest.fixture()
def stub(close_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    ctx = _CloseStubCtx()
    monkeypatch.setattr(server, "_trade", lambda: ctx)
    yield ctx


def _swap_ctx(monkeypatch, ctx):
    from trading_agent.mcp_servers.moomoo import server
    monkeypatch.setattr(server, "_trade", lambda: ctx)
    return ctx


def _audit_rows():
    from trading_agent.db import connection
    with connection() as conn:
        return conn.execute(
            "SELECT tool_name, decision FROM hook_audit_log "
            "WHERE tool_name = 'close_paper_option_combo'").fetchall()


def test_refused_without_matching_open_combo_row(stub):
    """Never close what the journal doesn't know (same principle as route's
    manual-position refusal)."""
    from trading_agent.mcp_servers.moomoo import server
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close_blocked"] is True
    assert resp["reason"] == "no_matching_open_combo"
    assert stub.place_order_calls == []
    assert "combo" not in resp and "rows" not in resp


def test_refused_when_journal_units_below_requested(stub):
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo(contracts=1.0)
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close_blocked"] is True
    assert resp["reason"] == "no_matching_open_combo"


def test_refused_when_broker_not_short_the_leg(close_db, monkeypatch):
    """A BUY on a flat short leg would OPEN a long; a SELL on a flat long
    leg would OPEN a short — broker proof is mandatory."""
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    ctx = _swap_ctx(monkeypatch, _CloseStubCtx(positions=[
        # short leg NOT held short (e.g. already bought back)
        {"code": LONG_PUT, "qty": 2, "position_side": "LONG"},
    ]))
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close_blocked"] is True
    assert resp["reason"] == "broker_position_mismatch"
    assert ctx.place_order_calls == []


def test_refused_on_non_vertical_legs(stub):
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    # long strike ABOVE short on a put spread → does not cap
    resp = server.close_paper_option_combo(
        short_leg_symbol=_future_opt("SPY", 40, "P", 95.0),
        long_leg_symbol=_future_opt("SPY", 40, "P", 100.0),
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close_blocked"] is True
    assert resp["reason"] == "long_leg_does_not_cap_short"
    assert stub.place_order_calls == []


def test_short_close_rejected_places_nothing(close_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    ctx = _swap_ctx(monkeypatch, _CloseStubCtx(reject_leg_indexes={0}))
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close"] is False
    assert resp["nothing_placed"] is True
    # only the rejected BTC attempt hit the broker; no STC, no cancels
    assert len(ctx.place_order_calls) == 1
    assert ctx.modify_order_calls == []


def test_short_close_placed_first_buy_side(stub):
    """Leg ordering: BTC short (BUY) first, STC long (SELL) second."""
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close"] is True
    assert len(stub.place_order_calls) == 2
    assert stub.place_order_calls[0]["code"] == SHORT_PUT
    assert str(stub.place_order_calls[0]["trd_side"]).endswith("BUY")
    assert stub.place_order_calls[1]["code"] == LONG_PUT
    assert str(stub.place_order_calls[1]["trd_side"]).endswith("SELL")
    assert resp["short_close_order_id"] == "CLOSE-0"
    assert resp["long_close_order_id"] == "CLOSE-1"
    assert len(resp["close_rows"]) == 2
    # fill-capture-hook safety: response never looks like an opening order
    assert "combo" not in resp and "rows" not in resp


def test_long_close_rejected_cancels_working_short_close(close_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    ctx = _swap_ctx(monkeypatch, _CloseStubCtx(reject_leg_indexes={1}))
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close"] is False
    assert resp["combo_rollback"] is True
    assert resp["rollback_action"] == "cancelled_working_short_close"
    assert len(ctx.modify_order_calls) == 1
    assert ctx.modify_order_calls[0]["order_id"] == "CLOSE-0"


def test_leg1_filled_leg2_rejected_keeps_long_reports_partial(close_db, monkeypatch):
    """Deliberate asymmetry with the open path: the dealt short buy-back is
    NEVER undone by re-selling. The residual long wing is defined-risk and
    the pending machinery retries its close."""
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    ctx = _swap_ctx(monkeypatch, _CloseStubCtx(
        reject_leg_indexes={1}, leg1_dealt_qty=2.0))
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close"] == "partial"
    assert resp["long_leg_unclosed"] is True
    assert resp["short_close_order_id"] == "CLOSE-0"
    assert resp["close_rows"][0]["order_id"] == "CLOSE-0"
    # rollback never touched the broker again
    assert len(ctx.place_order_calls) == 2
    assert ctx.modify_order_calls == []


def test_rollback_never_places_sell_on_short_leg(close_db, monkeypatch):
    """Across BOTH leg-2 failure modes, no order ever SELLs the short leg
    (that would be a SELL-to-open re-creating assignment risk)."""
    from trading_agent.mcp_servers.moomoo import server
    for kwargs in ({"reject_leg_indexes": {1}},
                   {"reject_leg_indexes": {1}, "leg1_dealt_qty": 2.0},
                   {"reject_leg_indexes": {1}, "cancel_fails": True}):
        _journal_open_combo()
        ctx = _swap_ctx(monkeypatch, _CloseStubCtx(**kwargs))
        server.close_paper_option_combo(
            short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
            contracts=2, short_price=0.60, long_price=0.20)
        sells_on_short = [
            c for c in ctx.place_order_calls
            if c["code"] == SHORT_PUT and str(c["trd_side"]).endswith("SELL")]
        assert sells_on_short == []


def test_cancel_failed_racing_fill_reports_partial_with_order_id(close_db, monkeypatch):
    from trading_agent.mcp_servers.moomoo import server
    _journal_open_combo()
    _swap_ctx(monkeypatch, _CloseStubCtx(
        reject_leg_indexes={1}, cancel_fails=True))
    resp = server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    assert resp["combo_close"] is False
    assert resp["partial"] is True
    assert resp["short_close_order_id"] == "CLOSE-0"
    assert resp["rollback_action"] == "cancel_failed_racing_fill"


def test_audits_allow_and_block(stub):
    from trading_agent.mcp_servers.moomoo import server
    # block (no journal row)
    server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    # allow
    _journal_open_combo()
    server.close_paper_option_combo(
        short_leg_symbol=SHORT_PUT, long_leg_symbol=LONG_PUT,
        contracts=2, short_price=0.60, long_price=0.20)
    decisions = [r["decision"] for r in _audit_rows()]
    assert "block" in decisions and "allow" in decisions
