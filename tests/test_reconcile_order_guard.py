"""Nightly guard reconciliation — fills with no guard evaluation must alert."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def recon_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _insert_fill(symbol: str, hours_ago: float = 1.0, broker_order_id: str = "B1",
                 strategy_label: str | None = None) -> int:
    from trading_agent.db import connection
    opened = (_now() - timedelta(hours=hours_ago)).isoformat()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, broker_order_id, strategy_label, is_paper) "
            "VALUES (?, 'OPT', 'BUY', 2, 1.5, ?, 'OPEN', ?, ?, 1)",
            (symbol, opened, broker_order_id, strategy_label),
        )
        return int(cur.lastrowid)


def _insert_guard_row(symbol: str, hours_ago: float = 1.0, decision: str = "allow",
                      hook_name: str = "server_order_guard") -> None:
    from trading_agent.db import connection
    created = (_now() - timedelta(hours=hours_ago)).isoformat()
    with connection() as conn:
        conn.execute(
            "INSERT INTO hook_audit_log (created_at, hook_name, tool_name, "
            "decision, reason, payload) VALUES (?, ?, 'place_paper_option_order', "
            "?, 'r', ?)",
            (created, hook_name, decision, json.dumps({"symbol": symbol})),
        )


def _run(argv=None):
    from trading_agent.jobs import reconcile_order_guard as job
    # --skip-positions keeps the guard-audit unit tests hermetic (no OpenD).
    return job.main(argv or ["--no-notify", "--skip-positions"])


def test_guarded_fill_is_clean(recon_db, capsys):
    _insert_fill("US.MRVL260702C00290000", hours_ago=2.0)
    _insert_guard_row("US.MRVL260702C00290000", hours_ago=2.01)
    assert _run() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["guarded"] == 1
    assert out["findings"] == []


def test_unguarded_fill_alerts(recon_db, capsys):
    """The MRVL/AAPL incident shape: a fill with zero guard evaluation."""
    _insert_fill("US.AAPL260720P00300000", hours_ago=2.0)
    assert _run() == 1
    out = json.loads(capsys.readouterr().out)
    assert len(out["findings"]) == 1
    assert out["findings"][0]["status"] == "unguarded"


def test_blocked_but_filled_alerts(recon_db, capsys):
    """A guard said BLOCK and the fill exists anyway — loudest case."""
    _insert_fill("US.AAPL260720P00300000", hours_ago=2.0)
    _insert_guard_row("US.AAPL260720P00300000", hours_ago=2.01, decision="block")
    assert _run() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["findings"][0]["status"] == "blocked_but_filled"


def test_real_mirror_fills_exempt(recon_db, capsys):
    _insert_fill("US.NVDA", hours_ago=2.0, broker_order_id="REALMIRROR-123",
                 strategy_label="shadow_real_account")
    assert _run() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fills_checked"] == 0


def test_old_fills_outside_window_ignored(recon_db, capsys):
    _insert_fill("US.NVDA", hours_ago=50.0)  # beyond the 26h default
    assert _run() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fills_checked"] == 0


def test_guard_row_within_match_window_counts(recon_db, capsys):
    """Virtual-fill path: guard ran in the order tool, journal row landed
    later via record_virtual_fill. Default window is 45 min."""
    _insert_fill("US.MRVL260702C00290000", hours_ago=2.0)
    _insert_guard_row("US.MRVL260702C00290000", hours_ago=2.5)  # 30 min earlier
    assert _run() == 0


def test_notify_fires_on_findings(recon_db):
    _insert_fill("US.AAPL260720P00300000", hours_ago=2.0)
    with patch("trading_agent.notify.send") as sent:
        from trading_agent.jobs import reconcile_order_guard as job
        assert job.main(["--skip-positions"]) == 1
    assert sent.call_count == 1
    args, kwargs = sent.call_args
    assert args[0] == "risk"
    assert kwargs["priority"] == 5


# ---------------------------------------------------------------------------
# Position reconciliation — broker positions vs journal OPEN rows
# ---------------------------------------------------------------------------

def _compare(broker, rows):
    from trading_agent.jobs.reconcile_order_guard import compare_positions
    return compare_positions(broker, rows)


def _row(symbol, side="BUY", qty=1.0, trade_id=1, store="sqlite"):
    return {"store": store, "trade_id": trade_id, "symbol": symbol,
            "side": side, "qty": qty}


def test_position_without_journal_row():
    """The 6/2 NVDA phantom exit: broker holds it, journal thinks it's flat."""
    findings = _compare({"US.NVDA": 112.0}, [])
    assert len(findings) == 1
    assert findings[0]["status"] == "position_without_journal_row"
    assert findings[0]["symbol"] == "US.NVDA"
    assert findings[0]["broker_qty"] == 112.0


def test_journal_row_without_position():
    """The ids 9-12 shape: journal OPEN rows for a vanished broker position."""
    findings = _compare({}, [
        _row("US.MRVL260702C290000", "BUY", 2.0, trade_id=9),
        _row("US.MRVL260702C300000", "SELL", 2.0, trade_id=10),
    ])
    statuses = {f["symbol"]: f["status"] for f in findings}
    assert statuses == {
        "US.MRVL260702C290000": "journal_row_without_position",
        "US.MRVL260702C300000": "journal_row_without_position",
    }
    by_symbol = {f["symbol"]: f for f in findings}
    assert by_symbol["US.MRVL260702C290000"]["journal_qty"] == 2.0
    assert by_symbol["US.MRVL260702C300000"]["journal_qty"] == -2.0
    assert by_symbol["US.MRVL260702C290000"]["trade_ids"] == [9]


def test_position_qty_mismatch():
    findings = _compare({"US.NVDA": 112.0}, [_row("US.NVDA", "BUY", 50.0)])
    assert len(findings) == 1
    assert findings[0]["status"] == "position_qty_mismatch"
    assert findings[0]["journal_qty"] == 50.0


def test_matched_positions_clean():
    findings = _compare(
        {"US.NVDA": 112.0, "US.AAPL260710P300000": -8.0},
        [_row("US.NVDA", "BUY", 112.0, trade_id=1),
         _row("US.AAPL260710P300000", "SELL", 8.0, trade_id=2, store="postgres")],
    )
    assert findings == []


def test_partial_stores_suppress_broker_side_findings():
    """Mac shape: Postgres unreachable → a broker position absent from the
    visible journals must NOT alert (it may be journaled in the brain's
    store), but a visible OPEN row with no broker position still must."""
    from trading_agent.jobs.reconcile_order_guard import compare_positions
    findings = compare_positions(
        {"US.NVDA": 112.0},
        [_row("US.MRVL260702C290000", "BUY", 2.0, trade_id=9)],
        all_stores_visible=False,
    )
    assert [f["status"] for f in findings] == ["journal_row_without_position"]
    assert findings[0]["symbol"] == "US.MRVL260702C290000"


def test_positions_check_end_to_end(recon_db, capsys):
    """Full main() run with patched broker/pg loaders: the NVDA case."""
    from trading_agent.jobs import reconcile_order_guard as job
    with patch.object(job, "_load_broker_positions", return_value={"US.NVDA": 112.0}), \
         patch.object(job, "_load_open_rows_pg", return_value=[]):
        assert job.main(["--no-notify"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["positions_checked"] is True
    assert out["position_stores"] == ["sqlite", "postgres"]
    assert len(out["position_findings"]) == 1
    assert out["position_findings"][0]["status"] == "position_without_journal_row"


def test_positions_check_skipped_when_broker_offline(recon_db, capsys):
    """OpenD offline → check skipped with a flag, no alert, exit 0."""
    from trading_agent.jobs import reconcile_order_guard as job
    with patch.object(job, "_load_broker_positions", return_value=None):
        assert job.main(["--no-notify"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["positions_checked"] is False
    assert out["position_findings"] == []


def test_open_mirror_rows_excluded_from_positions(recon_db, capsys):
    """real_mirror OPEN rows are journal-only by design — never a mismatch."""
    _insert_fill("US.NVDA", hours_ago=2.0, broker_order_id="REALMIRROR-9",
                 strategy_label="shadow_real_account")
    from trading_agent.jobs import reconcile_order_guard as job
    with patch.object(job, "_load_broker_positions", return_value={}), \
         patch.object(job, "_load_open_rows_pg", return_value=[]):
        assert job.main(["--no-notify"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["position_findings"] == []


def test_missing_audit_table_is_a_finding(recon_db, capsys):
    """Schema never migrated (the Mac 2026-07-08 state) must alert, not crash."""
    from trading_agent.db import connection
    with connection() as conn:
        conn.execute("DROP TABLE hook_audit_log")
    assert _run() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["findings"][0]["status"] == "audit_log_unreadable"


# ---------------------------------------------------------------------------
# Combo leg expansion — a vertical is ONE journal row but TWO broker legs
# ---------------------------------------------------------------------------

SHORT_LEG = "US.SPY260828P620000"
LONG_LEG = "US.SPY260828P610000"


def _insert_combo_fill(contracts: float = 2.0, with_snapshot: bool = True) -> int:
    """Journal a vertical the way posttool_fill_capture's combo path does:
    one SELL row keyed by the short leg, legs only in market_snapshots."""
    from trading_agent.db import connection
    opened = (_now() - timedelta(hours=2.0)).isoformat()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, broker_order_id, strategy_label, is_paper) "
            "VALUES (?, 'OPT', 'SELL', ?, 1.55, ?, 'OPEN', 'B77', "
            "'credit_put_spread_30_45', 1)",
            (SHORT_LEG, contracts, opened),
        )
        trade_id = int(cur.lastrowid)
        if with_snapshot:
            conn.execute(
                "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
                "VALUES (?, ?, ?)",
                (trade_id, opened, json.dumps({
                    "combo": True, "short_leg": SHORT_LEG, "long_leg": LONG_LEG,
                    "broker_order_ids": ["B77", "B78"], "net_credit": 1.55,
                    "width": 10.0, "max_loss": 845.0, "contracts": contracts,
                })),
            )
    return trade_id


def test_combo_both_legs_reconcile_clean(recon_db, capsys):
    """Broker short+long legs vs the single combo journal row: no findings."""
    _insert_combo_fill(contracts=2.0)
    _insert_guard_row(SHORT_LEG, hours_ago=2.0)
    from trading_agent.jobs import reconcile_order_guard as job
    broker = {SHORT_LEG: -2.0, LONG_LEG: 2.0}
    with patch.object(job, "_load_broker_positions", return_value=broker), \
         patch.object(job, "_load_open_rows_pg", return_value=[]):
        assert job.main(["--no-notify"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["position_findings"] == []


def test_combo_without_snapshot_still_flags_long_leg(recon_db, capsys):
    """No snapshot -> no expansion -> the long leg alerts (safe direction)."""
    _insert_combo_fill(contracts=2.0, with_snapshot=False)
    _insert_guard_row(SHORT_LEG, hours_ago=2.0)
    from trading_agent.jobs import reconcile_order_guard as job
    broker = {SHORT_LEG: -2.0, LONG_LEG: 2.0}
    with patch.object(job, "_load_broker_positions", return_value=broker), \
         patch.object(job, "_load_open_rows_pg", return_value=[]):
        assert job.main(["--no-notify"]) == 1
    out = json.loads(capsys.readouterr().out)
    statuses = {(f["status"], f["symbol"]) for f in out["position_findings"]}
    assert ("position_without_journal_row", LONG_LEG) in statuses


def test_expand_combo_rows_shape(recon_db):
    """Expansion appends exactly one synthetic BUY row per combo, same trade_id."""
    trade_id = _insert_combo_fill(contracts=3.0)
    from trading_agent.jobs.reconcile_order_guard import (
        _load_open_rows_sqlite, expand_combo_rows,
    )
    rows = expand_combo_rows(_load_open_rows_sqlite())
    synthetic = [r for r in rows if r["symbol"] == LONG_LEG]
    assert len(synthetic) == 1
    assert synthetic[0] == {
        "store": "sqlite", "trade_id": trade_id,
        "symbol": LONG_LEG, "side": "BUY", "qty": 3.0,
    }
