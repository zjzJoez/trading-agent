"""Journal decontamination (Phase 0.4): provenance classification, seeded
lesson re-sourcing, P&L recompute, and the aggregate/retrieval hygiene that
depends on them."""
from __future__ import annotations

import dataclasses
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent
           / "scripts" / "migrate_journal_provenance.py")
_spec = importlib.util.spec_from_file_location("migrate_journal_provenance", _SCRIPT)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


@pytest.fixture()
def tmp_journal_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(
        config_mod.CONFIG, db_path=db_file, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


# --- classification rules (mirror the live DB's contamination cases) -------

@pytest.mark.parametrize("boid,reasoning,expected", [
    ("REALMIRROR-12345", "real-account mirror; real_order_id=1", "real_mirror"),
    ("3040008", "delta=0.48, dte=14 [dry-run: cancelled before fill]", "dry_run"),
    ("VIRTUAL-4d2154be", "Short leg... Mirroring real fill from Apr 24-27 "
                         "(scaled 1→5).", "virtual_backfill"),
    ("VIRTUAL-aaaa", "Broker rejected paper order; mid fallback", "virtual"),
    ("3103243", None, "agent"),
    (None, None, "agent"),
])
def test_classify(boid, reasoning, expected):
    assert mig._classify(boid, reasoning) == expected


def test_recompute_pnl_sell_to_open_inverts_sign():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (entry_price REAL, exit_price REAL, qty REAL,"
                 " asset_type TEXT, side TEXT)")
    conn.execute("INSERT INTO t VALUES (1.50, 3.00, 5, 'OPT', 'SELL')")
    row = conn.execute("SELECT * FROM t").fetchone()
    # short leg: (1.50-3.00)×5×100 = -750
    assert mig._recompute_pnl(row) == pytest.approx(-750.0)


# --- end-to-end on a synthetic contaminated DB -----------------------------

def _seed_contaminated_db():
    from trading_agent.db import connection
    now = datetime.now(UTC).isoformat()
    with connection() as conn:
        # Seeded "lesson" written BEFORE any trade exists
        conn.execute(
            "INSERT INTO notes (created_at, topic, text, tags, source) "
            "VALUES ('2026-04-21T17:30:00+00:00', 'lesson-x', 'fabricated', "
            "'lesson', 'post_mortem')")
        # Trades: agent loss (with id-8-style misreported pnl), virtual
        # backfill, real mirror, dry run
        rows = [
            # symbol, asset, side, qty, entry, exit, pnl, outcome, boid, reasoning
            ("US.QQQ260626C734000", "OPT", "BUY", 4, 16.76, 16.09, -136.0,
             "LOSS", "3104672", None),
            ("US.XLE260522C59000", "OPT", "SELL", 5, 1.5, 3.65, -1075.0,
             "LOSS", "VIRTUAL-4d2154be1be3",
             "Short leg. Mirroring real fill from Apr 24-27 (scaled 1→5)."),
            ("US.GLD260522C445000", "OPT", "BUY", 1, 8.0, None, None,
             "OPEN", "REALMIRROR-77", "real-account mirror; real_order_id=77"),
            ("US.SPY260522C500000", "OPT", "BUY", 1, 2.0, 2.0, 0.0,
             "SCRATCH", "3040008", "[dry-run: cancelled before fill]"),
        ]
        for r in rows:
            conn.execute(
                "INSERT INTO trades (symbol, asset_type, side, qty, "
                "entry_price, exit_price, pnl, outcome, broker_order_id, "
                "reasoning, opened_at, closed_at, is_paper) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (*r, "2026-04-21T19:31:00+00:00",
                 now if r[7] != "OPEN" else None),
            )


def test_migration_end_to_end(tmp_journal_db, capsys, monkeypatch):
    _seed_contaminated_db()
    monkeypatch.setattr(
        "sys.argv",
        ["migrate_journal_provenance.py", "--db", str(tmp_journal_db)])
    assert mig.main() == 0

    from trading_agent.db import connection
    with connection() as conn:
        provs = dict(conn.execute(
            "SELECT broker_order_id, provenance FROM trades").fetchall())
        assert provs["3104672"] == "agent"
        assert provs["VIRTUAL-4d2154be1be3"] == "virtual_backfill"
        assert provs["REALMIRROR-77"] == "real_mirror"
        assert provs["3040008"] == "dry_run"

        # id-8-style mismatch: -136 reported vs (16.09-16.76)*4*100 = -268
        row = conn.execute(
            "SELECT pnl, pnl_recomputed, pnl_mismatch FROM trades "
            "WHERE broker_order_id='3104672'").fetchone()
        assert row["pnl"] == pytest.approx(-136.0)          # untouched
        assert row["pnl_recomputed"] == pytest.approx(-268.0)
        assert row["pnl_mismatch"] == 1

        # SELL-to-open backfill: (1.5-3.65)*5*100 = -1075 → consistent
        row = conn.execute(
            "SELECT pnl_recomputed, pnl_mismatch FROM trades "
            "WHERE broker_order_id LIKE 'VIRTUAL-%'").fetchone()
        assert row["pnl_recomputed"] == pytest.approx(-1075.0)
        assert row["pnl_mismatch"] == 0

        # Seeded lesson excluded from post_mortem retrieval, preserved
        src = conn.execute(
            "SELECT source FROM notes WHERE topic='lesson-x'").fetchone()[0]
        assert src == "seeded_prior"

    # Backup exists next to the DB
    backups = list(tmp_journal_db.parent.glob("*.bak-*"))
    assert backups, "migration must write a backup before touching the DB"


def test_migration_is_idempotent(tmp_journal_db, monkeypatch):
    _seed_contaminated_db()
    monkeypatch.setattr(
        "sys.argv",
        ["migrate_journal_provenance.py", "--db", str(tmp_journal_db)])
    assert mig.main() == 0
    assert mig.main() == 0  # second run: no error, same result
    from trading_agent.db import connection
    with connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE provenance IS NULL").fetchone()[0]
    assert n == 0


# --- downstream hygiene -----------------------------------------------------

def test_post_mortem_aggregates_split_by_provenance(tmp_journal_db, monkeypatch):
    _seed_contaminated_db()
    monkeypatch.setattr(
        "sys.argv",
        ["migrate_journal_provenance.py", "--db", str(tmp_journal_db)])
    assert mig.main() == 0

    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-04-01")
    # Agent bucket: only the QQQ trade; pnl taken from pnl_recomputed
    # because the self-reported figure failed the leg-arithmetic audit.
    agent_pnls = {k: v["pnl"] for k, v in out["by_strategy"].items()}
    assert sum(v["n"] for v in out["by_strategy"].values()) == 1
    assert sum(agent_pnls.values()) == pytest.approx(-268.0)
    # The 5x-scaled XLE backfill and the dry run live in the shadow bucket
    assert sum(v["n"] for v in out["by_strategy_shadow"].values()) == 2


def test_search_past_trades_excludes_backfills(tmp_journal_db, monkeypatch):
    _seed_contaminated_db()
    monkeypatch.setattr(
        "sys.argv",
        ["migrate_journal_provenance.py", "--db", str(tmp_journal_db)])
    assert mig.main() == 0

    import struct

    from trading_agent.mcp_servers.journal import server as js
    monkeypatch.setattr(
        js, "_embed", lambda text: struct.pack("384f", *([0.0] * 384)))
    out = js.search_past_trades("XLE")
    boids_returned = set()
    from trading_agent.db import connection
    with connection() as conn:
        for r in out["trades_lexical"]:
            b = conn.execute("SELECT broker_order_id FROM trades WHERE id=?",
                             (r["trade_id"],)).fetchone()[0]
            boids_returned.add(b)
    assert "VIRTUAL-4d2154be1be3" not in boids_returned  # scaled backfill out
