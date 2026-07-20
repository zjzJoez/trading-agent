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


def _eval(kind, params, equity=100_000.0, cash=100_000.0, **kw):
    from trading_agent.order_guard import evaluate_order
    return evaluate_order(kind, params, equity=equity, cash=cash, **kw)


# A snapshot row that clears R5d: spread 2/101 ≈ 2.0% of mid, OI 1200.
GOOD_QUOTE = {"bid_price": 1.00, "ask_price": 1.02, "option_open_interest": 1200}


def _patch_quote(monkeypatch, row):
    """Stub the R5d live-snapshot fetch — tests must never touch OpenD."""
    from trading_agent import order_guard as og
    monkeypatch.setattr(og, "fetch_option_quote", lambda code, timeout_s=5.0: row)


def _future_opt(ticker: str = "MRVL", days: int = 30, right: str = "C",
                strike: float = 290.0) -> str:
    """Option symbol with a live ~`days`-DTE expiry so R5's [14,60] DTE band
    keeps passing as real time advances. (Hardcoded 260702 symbols went stale
    once their DTE fell below 14 — this keeps the fixtures evergreen.)"""
    exp = (datetime.now(timezone.utc).date() + timedelta(days=days)).strftime("%y%m%d")
    return f"US.{ticker}{exp}{right}{int(round(strike * 1000)):08d}"


# Evergreen ~30-DTE option used across the buy/audit/R5d fixtures.
_MRVL_OPT = _future_opt("MRVL", 30, "C", 290.0)


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
        "option_symbol": _MRVL_OPT, "side": "BUY",
        "contracts": 2, "price": 1.00,
    })
    assert not d.allowed
    assert "no open thesis" in d.reason


def test_stale_thesis_blocked(guard_db):
    _insert_thesis("MRVL", minutes_ago=11)
    d = _eval("option", {
        "option_symbol": _MRVL_OPT, "side": "BUY",
        "contracts": 2, "price": 1.00,
    })
    assert not d.allowed
    assert "no open thesis" in d.reason


def test_fresh_thesis_buy_option_allowed(guard_db, monkeypatch):
    tid = _insert_thesis("MRVL")
    _patch_quote(monkeypatch, GOOD_QUOTE)
    d = _eval("option", {
        "option_symbol": _MRVL_OPT, "side": "BUY",
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
# Spec status gate (interim convexity retirement, 2026-07-20)
# ---------------------------------------------------------------------------


def test_convexity_open_blocked_at_order_tools_even_with_fresh_thesis(guard_db, monkeypatch):
    """Retirement enforcement of record: a BUY-to-open long premium order
    labeled into the shadow_only convexity spec is refused by the moomoo
    guard even when thesis freshness / bands / sizing would all pass."""
    from trading_agent.sizing import R_SPEC_STATUS
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, GOOD_QUOTE)
    d = _eval("option", {
        "option_symbol": _MRVL_OPT, "side": "BUY",
        "contracts": 1, "price": 1.00, "delta": 0.40,
        "strategy_label": "directional_long_call",
    })
    assert not d.allowed
    assert any(v.rule == R_SPEC_STATUS for v in d.violations)


def test_convexity_close_still_allowed_after_retirement(guard_db):
    """Retirement must never strand a position: SELL-to-close of an existing
    legacy convexity position passes the full guard, no thesis needed."""
    sym = "US.AAPL260720C00200000"
    _insert_open_trade(sym, qty=2, entry_price=2.0)
    d = _eval("option", {
        "option_symbol": sym, "side": "SELL", "contracts": 2, "price": 2.50,
        "strategy_label": "directional_long_call",
    })
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


def test_audit_decision_writes_db_and_jsonl(guard_db, tmp_path, monkeypatch):
    from trading_agent import order_guard as og
    from trading_agent.db import connection

    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, GOOD_QUOTE)
    d = _eval("option", {
        "option_symbol": _MRVL_OPT, "side": "BUY",
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
    assert payload["symbol"] == _MRVL_OPT

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


# ---------------------------------------------------------------------------
# R5d option liquidity gate — opening orders only, closes always exempt.
# The skill prose (spread < 10% of mid, OI > 100) was the ONLY screening
# until 2026-06; the autonomous synthesizer bought at the ask unchecked.
# ---------------------------------------------------------------------------


def _buy_open_option(**over):
    params = {
        "option_symbol": _MRVL_OPT, "side": "BUY",
        "contracts": 1, "price": 1.00, "delta": 0.40,
    }
    params.update(over)
    return params


def test_r5d_wide_spread_blocked(guard_db, monkeypatch):
    from trading_agent.order_guard import R5D
    _insert_thesis("MRVL")
    # spread 0.20 on mid 1.10 = 18.2% of mid — way past the 5% cap.
    _patch_quote(monkeypatch, {"bid_price": 1.00, "ask_price": 1.20,
                               "option_open_interest": 1200})
    d = _eval("option", _buy_open_option())
    assert not d.allowed
    assert d.reason == "option liquidity gate failed (R5d)"
    assert any(v.rule == R5D and v.severity == "block" and "spread" in v.message
               for v in d.violations)


def test_r5d_low_open_interest_blocked(guard_db, monkeypatch):
    """OI 120 cleared the old prose threshold (>100) — R5d demands >= 500."""
    from trading_agent.order_guard import R5D
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, {"bid_price": 1.00, "ask_price": 1.02,
                               "option_open_interest": 120})
    d = _eval("option", _buy_open_option())
    assert not d.allowed
    assert any(v.rule == R5D and "open interest" in v.message
               for v in d.violations)


def test_r5d_alternate_field_spellings_handled(guard_db, monkeypatch):
    """Some snapshot payloads spell the fields 'bid'/'ask'/'open_interest'."""
    from trading_agent.order_guard import R5D
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, {"bid": 1.00, "ask": 1.02, "open_interest": 3})
    d = _eval("option", _buy_open_option())
    assert not d.allowed
    assert any(v.rule == R5D for v in d.violations)


def test_r5d_liquid_contract_allowed(guard_db, monkeypatch):
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, GOOD_QUOTE)
    d = _eval("option", _buy_open_option())
    assert d.allowed, d.violations_json()
    assert not any(v.rule.startswith("R5d") for v in d.violations)


def test_r5d_quote_fetch_failure_blocks_unquotable(guard_db, monkeypatch):
    """Fetch raising (OpenD down, timeout) → fail-closed for OPENS: an entry
    you can't price honestly is an entry you skip."""
    from trading_agent.order_guard import R5D_UNQUOTABLE

    def _boom(code, timeout_s=5.0):
        raise RuntimeError("OpenD unreachable")

    _insert_thesis("MRVL")
    monkeypatch.setattr("trading_agent.order_guard.fetch_option_quote", _boom)
    d = _eval("option", _buy_open_option())
    assert not d.allowed
    assert any(v.rule == R5D_UNQUOTABLE for v in d.violations)


def test_r5d_zero_bid_blocks_unquotable(guard_db, monkeypatch):
    """A 0 bid means no one is on that side — unquotable, not 'spread=ask'."""
    from trading_agent.order_guard import R5D_UNQUOTABLE
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, {"bid_price": 0, "ask_price": 1.02,
                               "option_open_interest": 1200})
    d = _eval("option", _buy_open_option())
    assert not d.allowed
    assert any(v.rule == R5D_UNQUOTABLE for v in d.violations)


def test_r5d_oi_missing_warns_not_blocks(guard_db, monkeypatch):
    """Some snapshots omit OI entirely; spread is the primary protection,
    so a tight-spread contract passes with a warn."""
    from trading_agent.order_guard import R5D_OI_UNKNOWN
    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, {"bid_price": 1.00, "ask_price": 1.02})
    d = _eval("option", _buy_open_option())
    assert d.allowed, d.violations_json()
    assert any(w.startswith(R5D_OI_UNKNOWN) for w in d.warns)


def test_r5d_sell_to_close_exempt_even_with_terrible_quote(guard_db, monkeypatch):
    """Closes are NEVER liquidity-gated — blocking an exit is the d089349
    failure mode. The gate must not even fetch a quote for a close."""
    sym = "US.AAPL260720C00200000"
    _insert_open_trade(sym, qty=6, entry_price=9.0)

    def _must_not_fetch(code, timeout_s=5.0):
        raise AssertionError("R5d fetched a quote for a CLOSING order")

    monkeypatch.setattr(
        "trading_agent.order_guard.fetch_option_quote", _must_not_fetch)
    d = _eval("option", {
        "option_symbol": sym, "side": "SELL", "contracts": 6, "price": 9.55,
    })
    assert d.allowed, d.violations_json()
    assert d.intent == "close"


def test_r5d_stock_orders_unaffected(guard_db, monkeypatch):
    _insert_thesis("NVDA")

    def _must_not_fetch(code, timeout_s=5.0):
        raise AssertionError("R5d fetched a quote for a stock order")

    monkeypatch.setattr(
        "trading_agent.order_guard.fetch_option_quote", _must_not_fetch)
    d = _eval("stock", {
        "symbol": "US.NVDA", "side": "BUY", "qty": 10, "price": 100.0,
        "stop": 95.0, "target": 115.0,
    })
    assert d.allowed, d.violations_json()


def test_r5d_caller_supplied_quote_skips_fetch(guard_db, monkeypatch):
    """A caller that already holds a fresh snapshot passes it through via
    option_quote= — evaluate_order must not double-fetch."""
    _insert_thesis("MRVL")

    def _must_not_fetch(code, timeout_s=5.0):
        raise AssertionError("fetched despite option_quote in scope")

    monkeypatch.setattr(
        "trading_agent.order_guard.fetch_option_quote", _must_not_fetch)
    d = _eval("option", _buy_open_option(), option_quote=dict(GOOD_QUOTE))
    assert d.allowed, d.violations_json()


def test_r5d_block_lands_in_audit_log(guard_db, monkeypatch):
    """R5d must be auditable like every other rule — the nightly reconcile
    joins fills against these rows."""
    import json as _json
    from trading_agent import order_guard as og
    from trading_agent.db import connection
    from trading_agent.order_guard import R5D

    _insert_thesis("MRVL")
    _patch_quote(monkeypatch, {"bid_price": 1.00, "ask_price": 1.40,
                               "option_open_interest": 1200})
    d = _eval("option", _buy_open_option(price=1.40))
    og.audit_decision(d, guard_name="server_order_guard",
                      tool_name="place_paper_option_order")
    with connection() as conn:
        row = conn.execute("SELECT decision, payload FROM hook_audit_log").fetchone()
    assert row["decision"] == "block"
    rules = {v["rule"] for v in _json.loads(row["payload"])["violations"]}
    assert R5D in rules


# ---- fetch_option_quote: lazy moomoo-server import, swallow-all ----


def _stub_server_module(monkeypatch, get_quote):
    import sys
    import types
    monkeypatch.setitem(
        sys.modules, "trading_agent.mcp_servers.moomoo.server",
        types.SimpleNamespace(get_quote=get_quote),
    )


def test_fetch_option_quote_swallows_server_errors(monkeypatch):
    from trading_agent.order_guard import fetch_option_quote

    def _boom(symbols):
        raise RuntimeError("get_market_snapshot failed")

    _stub_server_module(monkeypatch, _boom)
    assert fetch_option_quote(_MRVL_OPT) is None


def test_fetch_option_quote_empty_rows_is_none(monkeypatch):
    from trading_agent.order_guard import fetch_option_quote
    _stub_server_module(monkeypatch, lambda symbols: {"rows": []})
    assert fetch_option_quote(_MRVL_OPT) is None


def test_fetch_option_quote_returns_first_row(monkeypatch):
    from trading_agent.order_guard import fetch_option_quote
    row = {"bid_price": 1.0, "ask_price": 1.02, "option_open_interest": 900}
    _stub_server_module(monkeypatch, lambda symbols: {"rows": [row]})
    assert fetch_option_quote(_MRVL_OPT) == row


# ---- dual-store close-intent (post-merge regression: Postgres journal) ----

class _PgCtx:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        import unittest.mock as m
        cur = m.MagicMock()
        cur.fetchone.return_value = self._rows[0] if self._rows else None
        return cur

    def __exit__(self, *a):
        return False


def test_sell_intent_close_when_open_trade_lives_only_in_postgres(guard_db, monkeypatch):
    """Autonomous-graph entries are journaled ONLY in Postgres journal_trades.
    With the SQLite-only lookup, every autonomous SELL-to-close was classified
    as SELL-to-open and hard-blocked — the guard would have prevented the
    system from exiting its own positions."""
    from trading_agent import order_guard as og
    monkeypatch.setattr(
        "trading_agent.store.postgres.cursor", lambda: _PgCtx([(1,)]))
    assert og.has_existing_open_for("US.QQQ260626C734000") is True


def test_sell_intent_open_when_both_stores_say_no(guard_db, monkeypatch):
    from trading_agent import order_guard as og
    monkeypatch.setattr(
        "trading_agent.store.postgres.cursor", lambda: _PgCtx([]))
    assert og.has_existing_open_for("US.QQQ260626C734000") is False


def test_sell_intent_open_when_sqlite_says_no_and_postgres_unreachable(guard_db, monkeypatch):
    """Phase-1 Mac has no Postgres server. SQLite's definitive 'no open
    trade' must classify as OPENING — failing open to 'close' here would
    silently disable the SELL-to-open block in interactive trading."""
    from trading_agent import order_guard as og

    def _boom():
        raise ConnectionError("no postgres on this box")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    assert og.has_existing_open_for("US.QQQ260626C734000") is False


# ---------------------------------------------------------------------------
# Multi-leg defined-risk combos (evaluate_combo, area A)
# ---------------------------------------------------------------------------


def _eval_combo(params, equity=100_000.0, cash=100_000.0, **kw):
    from trading_agent.order_guard import evaluate_combo
    return evaluate_combo(params, equity=equity, cash=cash, **kw)


def _combo_params(short_strike=100.0, long_strike=95.0, short_price=2.0,
                  long_price=0.8, contracts=1, right="P", ticker="SPY"):
    return {
        "short_leg_symbol": _future_opt(ticker, 30, right, short_strike),
        "long_leg_symbol": _future_opt(ticker, 30, right, long_strike),
        "contracts": contracts, "short_price": short_price,
        "long_price": long_price, "strategy_label": "credit_put_spread_30_45",
    }


def test_valid_credit_put_spread_combo_allowed(guard_db, monkeypatch):
    """Fresh thesis + liquid legs + long caps short → the combo opens, even
    though its short leg would be hard-blocked as a single-leg SELL-to-open."""
    _insert_thesis("SPY")
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    d = _eval_combo(_combo_params())
    assert d.allowed, d.violations_json()
    assert d.legs is not None and len(d.legs) == 2
    assert d.proposed["max_loss"] == (5.0 - 1.2) * 100.0  # 380


def test_combo_requires_fresh_thesis(guard_db, monkeypatch):
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    d = _eval_combo(_combo_params())  # no thesis inserted
    assert not d.allowed
    assert "no open thesis" in d.reason


def test_combo_not_defined_risk_blocked(guard_db, monkeypatch):
    """Put spread where the long strike is ABOVE the short = not capped → R5e."""
    from trading_agent.sizing import R5E
    _insert_thesis("SPY")
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    d = _eval_combo(_combo_params(short_strike=95.0, long_strike=100.0))
    assert not d.allowed
    assert any(v.rule == R5E for v in d.violations)


def test_combo_illiquid_leg_blocked(guard_db, monkeypatch):
    """A wide-spread quote on the legs trips R5d (liquidity) → blocked."""
    _insert_thesis("SPY")
    # bid/ask 1.00/1.40 → ~33% spread, far over the 5% R5d cap
    _patch_quote(monkeypatch, {"bid_price": 1.00, "ask_price": 1.40,
                               "option_open_interest": 1200})
    d = _eval_combo(_combo_params())
    assert not d.allowed
    assert "R5d" in d.reason


def test_combo_unparseable_leg_symbol_blocked(guard_db, monkeypatch):
    _insert_thesis("SPY")
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    params = _combo_params()
    params["short_leg_symbol"] = "GARBAGE"
    d = _eval_combo(params)
    assert not d.allowed
    assert "unparseable" in d.reason


def test_combo_legs_must_share_underlying(guard_db, monkeypatch):
    _insert_thesis("SPY")
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    params = _combo_params()
    params["long_leg_symbol"] = _future_opt("QQQ", 30, "P", 95.0)  # different underlying
    d = _eval_combo(params)
    assert not d.allowed
    assert "different underlying" in d.reason


def test_single_leg_sell_to_open_still_blocked(guard_db):
    """Regression guard: the combo path must NOT have loosened the single-leg
    SELL-to-open hard block."""
    from trading_agent.order_guard import R_NO_SHORT_OPEN
    _insert_thesis("SPY")
    d = _eval("option", {
        "option_symbol": _future_opt("SPY", 30, "P", 100.0), "side": "SELL",
        "contracts": 1, "price": 2.0, "strategy_label": "x",
    })
    assert not d.allowed
    assert any(v.rule == R_NO_SHORT_OPEN for v in d.violations)


def test_open_combo_counts_at_max_loss_for_r3(guard_db):
    """An open combo journaled at net_credit (1.2) must register its true
    max_loss (380) as existing exposure, not 1.2×100=120 — otherwise a
    follow-on order under-counts R3 ticker concentration."""
    import json as _json

    from trading_agent.db import connection
    from trading_agent.order_guard import load_open_positions, load_sector_map
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) "
            "VALUES (?, 'OPT', 'SELL', 1, 1.2, ?, 'OPEN', 1)",
            ("US.SPY260717P00100000",
             datetime.now(timezone.utc).isoformat()),
        )
        tid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, datetime.now(timezone.utc).isoformat(),
             _json.dumps({"combo": True, "max_loss": 380.0, "net_credit": 1.2})),
        )
        conn.commit()
    opens = load_open_positions(load_sector_map())
    spy = next(p for p in opens if p.ticker == "SPY")
    assert spy.risk_notional == 380.0
    assert spy.notional == 380.0  # NOT 120


def test_combo_response_returns_guard_verified_thesis_id(guard_db, monkeypatch):
    """evaluate_combo runs on the underlying's fresh thesis; the placement
    summary must carry that guard-verified id so the journal FK never breaks
    on a caller-supplied bad id."""
    tid = _insert_thesis("SPY")
    _patch_quote(monkeypatch, dict(GOOD_QUOTE))
    d = _eval_combo(_combo_params())
    assert d.allowed
    assert d.thesis_id == tid
