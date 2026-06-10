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
    return job.main(argv or ["--no-notify"])


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
        assert job.main([]) == 1
    assert sent.call_count == 1
    args, kwargs = sent.call_args
    assert args[0] == "risk"
    assert kwargs["priority"] == 5
