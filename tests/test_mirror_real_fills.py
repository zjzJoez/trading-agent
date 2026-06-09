"""Tests for the real-account -> paper-journal shadow mirror job."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_journal_db(tmp_path: Path, monkeypatch):
    """Point CONFIG.db_path at a tmp file and apply schema."""
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


_FAKE_ORDERS = {
    "rows": [
        {  # stock buy -> STK / LONG
            "code": "US.NVDA", "trd_side": "BUY", "order_status": "FILLED_ALL",
            "order_id": "A1", "dealt_qty": 5, "dealt_avg_price": 200.0,
            "create_time": "2026-06-05 12:00:00",
        },
        {  # option sell-to-close a call -> OPT / SHORT_CALL
            "code": "US.MRVL260702C300000", "trd_side": "SELL", "order_status": "FILLED_ALL",
            "order_id": "A2", "dealt_qty": 2, "dealt_avg_price": 26.0,
            "create_time": "2026-06-02 10:00:00",
        },
        {  # cancelled -> ignored
            "code": "US.INTC", "trd_side": "SELL", "order_status": "CANCELLED_ALL",
            "order_id": "A3", "dealt_qty": 0, "dealt_avg_price": 0,
            "create_time": "2026-06-05 12:41:00",
        },
        {  # zero-fill failed -> ignored
            "code": "US.WDCX", "trd_side": "BUY", "order_status": "FAILED",
            "order_id": "A4", "dealt_qty": 0, "dealt_avg_price": 0,
            "create_time": "2026-06-03 09:38:00",
        },
    ]
}


def _run(monkeypatch, argv):
    from trading_agent.jobs import mirror_real_fills as m
    monkeypatch.setattr(m, "_fetch_orders", lambda since, acc_index: list(_FAKE_ORDERS["rows"]))
    return m.main(argv)


def test_mirror_inserts_classifies_and_skips_unfilled(tmp_journal_db, monkeypatch):
    from trading_agent.db import connection

    assert _run(monkeypatch, []) == 0

    with connection() as conn:
        rows = conn.execute(
            "SELECT broker_order_id, symbol, asset_type, side, strategy_label, is_paper "
            "FROM trades ORDER BY id"
        ).fetchall()
    by = {r["broker_order_id"]: r for r in rows}

    # Only the two FILLED_ALL orders were mirrored.
    assert set(by) == {"REALMIRROR-A1", "REALMIRROR-A2"}
    # All shadow rows tagged + paper.
    assert all(r["strategy_label"] == "shadow_real_account" and r["is_paper"] == 1 for r in rows)
    # Classification.
    assert by["REALMIRROR-A1"]["asset_type"] == "STK" and by["REALMIRROR-A1"]["side"] == "BUY"
    assert by["REALMIRROR-A2"]["asset_type"] == "OPT" and by["REALMIRROR-A2"]["side"] == "SELL"


def test_shadow_theses_are_not_open(tmp_journal_db, monkeypatch):
    """Shadow theses must be marked 'triggered' so they don't pollute the open
    set or the PreToolUse order gate."""
    from trading_agent.db import connection

    assert _run(monkeypatch, []) == 0
    with connection() as conn:
        open_count = conn.execute("SELECT COUNT(*) FROM theses WHERE status = 'open'").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM theses").fetchone()[0]
    assert total == 2
    assert open_count == 0


def test_mirror_is_idempotent(tmp_journal_db, monkeypatch):
    from trading_agent.db import connection

    assert _run(monkeypatch, []) == 0
    with connection() as conn:
        n1 = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    # Second run must insert nothing new (dedup by REALMIRROR-<order_id>).
    assert _run(monkeypatch, []) == 0
    with connection() as conn:
        n2 = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert n1 == 2
    assert n2 == 2


def test_dry_run_writes_nothing(tmp_journal_db, monkeypatch):
    from trading_agent.db import connection

    assert _run(monkeypatch, ["--dry-run"]) == 0
    with connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert n == 0
