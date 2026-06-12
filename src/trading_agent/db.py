"""SQLite connection helper + migration runner.

Loads the sqlite-vec extension so `vec0` virtual tables work. Every caller
(MCP servers, hooks, jobs) should go through `connect()` — it sets pragmas
and loads extensions uniformly.
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from .config import CONFIG, ensure_dirs

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with sqlite-vec loaded and sane pragmas."""
    ensure_dirs()
    path = db_path or CONFIG.db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@contextmanager
def cursor(db_path: Path | None = None) -> Iterator[sqlite3.Cursor]:
    conn = connect(db_path)
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Full Connection context manager — commit on success, rollback on
    exception, and **always close**.

    Prefer this over `with connect() as conn`: sqlite3.Connection's native
    context manager auto-commits but does not close, which leaks
    connections when a tool is invoked repeatedly.
    """
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the original CREATE TABLE shipped. CREATE TABLE IF NOT
# EXISTS cannot retrofit them onto existing databases, so migrate() ALTERs
# any that are missing. Keep in sync with schema.sql's trades definition.
_TRADE_COLUMN_ADDS: dict[str, str] = {
    "provenance": "TEXT NOT NULL DEFAULT 'agent'",
    "fees": "REAL",
    "pnl_recomputed": "REAL",
    "pnl_mismatch": "INTEGER DEFAULT 0",
    "exit_order_id": "TEXT",
    "exit_state_json": "TEXT",
}


def _ensure_trade_columns(conn: sqlite3.Connection) -> list[str]:
    """ALTER TABLE trades to add any missing post-v1 columns. Idempotent."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    added: list[str] = []
    for name, decl in _TRADE_COLUMN_ADDS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {decl}")
            added.append(name)
    return added


def migrate(db_path: Path | None = None) -> None:
    """Apply schema.sql + column retrofits. Idempotent."""
    sql = SCHEMA_PATH.read_text()
    conn = connect(db_path)
    try:
        conn.executescript(sql)
        added = _ensure_trade_columns(conn)
        if added:
            print(f"[db.migrate] trades columns added: {', '.join(added)}",
                  file=sys.stderr)
        conn.commit()
    finally:
        conn.close()


def migrate_cli() -> None:
    """`ta-migrate` entry point."""
    migrate()
    print(f"Migrated {CONFIG.db_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_cli()
    else:
        print("usage: python -m trading_agent.db migrate", file=sys.stderr)
        sys.exit(2)
