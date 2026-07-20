"""posttool_fill_capture — combo (defined-risk vertical) journaling (area A).

A place_paper_option_combo response (combo=True, combo_rows=[long, short])
must be journaled as ONE OPT trades row (entry_price=net_credit, short-leg
symbol) with both broker order ids + width/max_loss in the snapshot — never
two side-split rows (which would double-count R2 and mis-score the post-mortem).
"""
from __future__ import annotations

import dataclasses
import io
import json

import pytest


@pytest.fixture()
def post_db(tmp_path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent.hooks import posttool_fill_capture as pf

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(pf, "AUDIT_PATH", tmp_path / "hook_audit.log")
    db_mod.migrate(db_file)
    yield db_file


def _run_post(monkeypatch, payload) -> int:
    from trading_agent.hooks import posttool_fill_capture as pf
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return pf.main()


def _insert_thesis(ticker: str = "SPY") -> int:
    """trades.thesis_id is a FK → theses.id; production always has one."""
    from datetime import datetime, timezone

    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, 'LONG', 't', 'i', 'open')",
            (datetime.now(timezone.utc).isoformat(), ticker),
        )
        conn.commit()
        return int(cur.lastrowid)


def _combo_payload(thesis_id: int = 7, **over):
    resp = {
        "combo": True, "thesis_id": thesis_id,
        "strategy_label": "credit_put_spread_30_45",
        "net_credit": 1.2, "width": 5.0, "max_loss": 380.0,
        "combo_rows": [
            {"order_id": "L1", "code": "US.SPY260717P00095000"},
            {"order_id": "S1", "code": "US.SPY260717P00100000"},
        ],
    }
    resp.update(over)
    return {
        "tool_name": "mcp__moomoo-mcp__place_paper_option_combo",
        "tool_input": {
            "short_leg_symbol": "US.SPY260717P00100000",
            "long_leg_symbol": "US.SPY260717P00095000",
            "contracts": 1, "strategy_label": "credit_put_spread_30_45",
            "thesis_id": thesis_id,
        },
        "tool_response": resp,
    }


def test_combo_journals_exactly_one_row(post_db, monkeypatch):
    from trading_agent.db import connection
    tid = _insert_thesis("SPY")
    rc = _run_post(monkeypatch, _combo_payload(thesis_id=tid))
    assert rc == 0
    with connection() as conn:
        rows = conn.execute(
            "SELECT symbol, asset_type, side, qty, entry_price, strategy_label "
            "FROM trades"
        ).fetchall()
        snaps = conn.execute("SELECT payload FROM market_snapshots").fetchall()
    assert len(rows) == 1  # ONE row for the whole spread, not two
    r = rows[0]
    assert r["symbol"] == "US.SPY260717P00100000"  # short leg is the anchor
    assert r["asset_type"] == "OPT"
    assert r["entry_price"] == 1.2  # net credit
    assert r["strategy_label"] == "credit_put_spread_30_45"
    assert len(snaps) == 1
    payload_j = json.loads(snaps[0]["payload"])
    assert payload_j["combo"] is True
    assert payload_j["broker_order_ids"] == ["L1", "S1"]
    assert payload_j["max_loss"] == 380.0


def test_combo_rollback_response_journals_nothing(post_db, monkeypatch):
    """A rolled-back combo (combo=False) must NOT create a trade row."""
    from trading_agent.db import connection
    payload = _combo_payload()
    payload["tool_response"] = {
        "combo": False, "combo_rollback": True, "rolled_back_long": True,
        "thesis_id": 7, "reason": "short leg rejected",
    }
    rc = _run_post(monkeypatch, payload)
    assert rc == 0
    with connection() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()["c"]
    assert n == 0


class _PgCaptureCtx:
    calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        _PgCaptureCtx.calls.append((sql, params))


def test_combo_dual_writes_postgres_row_with_combo_plan(post_db, monkeypatch):
    """D2: the exit engine reads ONLY Postgres journal_trades — the combo
    branch must dual-write an engine-visible row there: symbol=short leg,
    side=SELL, entry_price=net_credit, exit_plan=combo_exit_plan (50%-PT +
    21-DTE), broker_fill_json.combo = canonical payload, 2-leg entry fees."""
    _PgCaptureCtx.calls = []
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        lambda: _PgCaptureCtx())
    tid = _insert_thesis("SPY")
    rc = _run_post(monkeypatch, _combo_payload(thesis_id=tid))
    assert rc == 0
    inserts = [(sql, p) for sql, p in _PgCaptureCtx.calls
               if "INSERT INTO journal_trades" in sql]
    assert len(inserts) == 1
    _sql, params = inserts[0]
    symbol, qty, entry_price, boid, plan_json, fill_json = params
    assert symbol == "US.SPY260717P00100000"       # short leg anchors
    assert qty == 1.0
    assert entry_price == 1.2                      # net credit
    assert boid == "L1"                            # long leg placed first
    plan = json.loads(plan_json)
    assert plan["direction"] == "SHORT"
    assert plan["hard_target"] == 0.6              # 50% of net credit
    assert plan["hard_stop"] == 5.0                # width (inert)
    assert plan["dte_rules"]["force_exit_at_dte"] == 21
    assert plan["scale_out_ladder"] == []          # whole units only (C7)
    assert plan["trail_stop"] is None
    fill = json.loads(fill_json)
    assert fill["combo"]["short_leg"] == "US.SPY260717P00100000"
    assert fill["combo"]["long_leg"] == "US.SPY260717P00095000"
    assert fill["combo"]["broker_order_ids"] == ["L1", "S1"]
    assert fill["provenance"] == "agent"
    assert fill["entry_fees"] == 2.0               # 2 legs × $1 × 1 contract


def test_combo_sqlite_failure_still_attempts_pg_dual_write(post_db, monkeypatch):
    """A Mac-side SQLite hiccup must NOT abort the Postgres dual-write —
    the PG row is the engine-visibility write (exit plan, 21-DTE, 50%-PT),
    the more safety-critical of the two. Without it the freshly-opened combo
    is invisible to the exit engine entirely."""
    from trading_agent.hooks import posttool_fill_capture as pf

    def _sqlite_boom(**kw):
        raise RuntimeError("sqlite locked")
    monkeypatch.setattr(pf, "_insert_trade_row", _sqlite_boom)
    _PgCaptureCtx.calls = []
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        lambda: _PgCaptureCtx())
    tid = _insert_thesis("SPY")
    rc = _run_post(monkeypatch, _combo_payload(thesis_id=tid))
    assert rc == 0
    inserts = [(sql, p) for sql, p in _PgCaptureCtx.calls
               if "INSERT INTO journal_trades" in sql]
    assert len(inserts) == 1, "PG dual-write must run despite SQLite failure"
    symbol = inserts[0][1][0]
    assert symbol == "US.SPY260717P00100000"


def test_combo_pg_dual_write_failure_keeps_sqlite_row_and_exits_zero(post_db, monkeypatch):
    """The hook is telemetry, never a gate: Postgres down → audit line,
    exit 0, SQLite combo row still lands."""
    from trading_agent.db import connection

    def _boom():
        raise ConnectionError("no postgres on this box")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    tid = _insert_thesis("SPY")
    rc = _run_post(monkeypatch, _combo_payload(thesis_id=tid))
    assert rc == 0
    with connection() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()["c"]
    assert n == 1
