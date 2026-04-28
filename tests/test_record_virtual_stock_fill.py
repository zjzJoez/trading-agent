"""Smoke test for record_virtual_stock_fill — mirrors record_virtual_fill but
for stock/ETF positions. Verifies the row lands in `trades` with the expected
fields, joins on theses for `get_open_positions_with_thesis`, and aggregates
under `generate_post_mortem_prompt` by `strategy_label`.
"""
from __future__ import annotations

import dataclasses
from datetime import UTC
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_journal_db(tmp_path: Path, monkeypatch):
    """Point CONFIG.db_path at a tmp file and apply schema. Yields the path.

    `Config` is a frozen dataclass, so we swap the global CONFIG with a
    replace()d copy. Both `connection()` and `migrate()` read CONFIG at
    call-time, so this catches them.
    """
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    # db_mod imports CONFIG by name into its namespace at module import; patch there too.
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


def _insert_thesis(ticker: str, direction: str = "LONG") -> int:
    from datetime import datetime

    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, ?, ?, ?, 'open')",
            (datetime.now(UTC).isoformat(), ticker, direction,
             "shadow tracking", "stop on close below 175"),
        )
        return cur.lastrowid


def test_record_virtual_stock_fill_inserts_row(tmp_journal_db):
    """Tool inserts a STK row with VIRTUAL- broker_order_id and is_paper=1."""
    from trading_agent.db import connection
    from trading_agent.mcp_servers.journal.server import record_virtual_stock_fill

    thesis_id = _insert_thesis("AMZN")
    res = record_virtual_stock_fill(
        ticker="amzn",  # exercise upper() normalization
        side="BUY",
        qty=5,
        price=185.42,
        thesis_id=thesis_id,
        strategy_label="shadow_real_account",
        reasoning="mirror live AMZN holding from real account",
    )

    assert res["thesis_id"] == thesis_id
    assert res["is_paper"] is True
    assert res["broker_order_id"].startswith("VIRTUAL-")
    assert res["trade_id"] > 0

    with connection() as conn:
        row = conn.execute(
            "SELECT symbol, asset_type, side, qty, entry_price, strategy_label, "
            "broker_order_id, is_paper, outcome, stop, target, exit_price, pnl "
            "FROM trades WHERE id = ?",
            (res["trade_id"],),
        ).fetchone()

    assert row is not None
    assert row["symbol"] == "AMZN"
    assert row["asset_type"] == "STK"
    assert row["side"] == "BUY"
    assert row["qty"] == 5
    assert row["entry_price"] == 185.42
    assert row["strategy_label"] == "shadow_real_account"
    assert row["broker_order_id"] == res["broker_order_id"]
    assert row["is_paper"] == 1
    assert row["outcome"] == "OPEN"
    # No option-side fields populated for a stock fill.
    assert row["stop"] is None
    assert row["target"] is None
    assert row["exit_price"] is None
    assert row["pnl"] is None


def test_get_open_positions_includes_stock_fill(tmp_journal_db):
    """get_open_positions_with_thesis joins the stock trade row with its thesis."""
    from trading_agent.mcp_servers.journal.server import (
        get_open_positions_with_thesis,
        record_virtual_stock_fill,
    )

    thesis_id = _insert_thesis("MSFT")
    record_virtual_stock_fill(
        ticker="MSFT", side="BUY", qty=10, price=420.10,
        thesis_id=thesis_id, strategy_label="shadow_real_account",
    )

    res = get_open_positions_with_thesis()
    assert res["count"] >= 1
    msft = [r for r in res["rows"] if r["symbol"] == "MSFT"]
    assert len(msft) == 1
    r = msft[0]
    assert r["asset_type"] == "STK"
    assert r["side"] == "BUY"
    assert r["qty"] == 10
    assert r["thesis_id"] == thesis_id
    assert r["ticker"] == "MSFT"
    assert r["thesis_status"] == "open"


def test_post_mortem_aggregates_stock_by_strategy(tmp_journal_db):
    """generate_post_mortem_prompt aggregates closed stock trades by
    strategy_label exactly like options trades."""
    from datetime import datetime, timedelta

    from trading_agent.db import connection
    from trading_agent.mcp_servers.journal.server import (
        close_trade,
        generate_post_mortem_prompt,
        record_virtual_stock_fill,
    )

    thesis_id = _insert_thesis("NVDA")
    fill = record_virtual_stock_fill(
        ticker="NVDA", side="BUY", qty=3, price=900.0,
        thesis_id=thesis_id, strategy_label="shadow_real_account",
    )

    # Backdate opened_at so it falls inside our window even on slow CI.
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with connection() as conn:
        conn.execute(
            "UPDATE trades SET opened_at = ? WHERE id = ?",
            (yesterday, fill["trade_id"]),
        )

    close_trade(
        trade_id=fill["trade_id"],
        exit_price=920.0,
        outcome="WIN",
        pnl=60.0,  # 3 * (920 - 900)
    )

    since = (datetime.now(UTC) - timedelta(days=2)).date().isoformat()
    pm = generate_post_mortem_prompt(since_date=since)

    assert pm["closed_trade_count"] >= 1
    bucket = pm["by_strategy"].get("shadow_real_account")
    assert bucket is not None, f"missing shadow_real_account in {pm['by_strategy']}"
    assert bucket["n"] >= 1
    assert bucket["wins"] >= 1
    assert bucket["pnl"] >= 60.0
