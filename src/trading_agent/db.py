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


def migrate(db_path: Path | None = None) -> None:
    """Apply schema.sql. Idempotent — uses IF NOT EXISTS throughout."""
    sql = SCHEMA_PATH.read_text()
    conn = connect(db_path)
    try:
        conn.executescript(sql)
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
