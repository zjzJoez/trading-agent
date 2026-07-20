"""Phase 2 Postgres connection pool + migrations runner.

The DSN comes from `~/.env.postgres` on EC2 (chmod 600). Locally, point
`POSTGRES_DSN` at any Postgres you have for unit tests, or skip — the
sqlite trader.db still works for Phase 1 manual override sessions.

All Phase 2 tables (regime_*, risk_*, learning_*, agent_events, journal_*
mirror) live in Postgres; Phase 1 sqlite is preserved as a read-only
cache for backwards compatibility.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

# Migrations live next to the package, ../../migrations/*.sql relative to here.
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

_pool: ConnectionPool | None = None


def _resolve_dsn() -> str:
    """Read POSTGRES_DSN from env, falling back to ~/.env.postgres on EC2."""
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        return dsn
    env_file = Path.home() / ".env.postgres"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("POSTGRES_DSN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "POSTGRES_DSN not set and ~/.env.postgres not readable. "
        "On EC2 run install_postgres.sh; locally export POSTGRES_DSN=..."
    )


def get_pool(min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    """Lazy singleton — first caller creates the pool, others reuse it."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_resolve_dsn(),
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": False},
            open=True,
        )
    return _pool


def shutdown(timeout: float = 5.0) -> None:
    """Close the singleton ConnectionPool. Idempotent.

    Paired with the same cleanup hook in ``graph/checkpointer.py`` — both
    pools spawn non-daemon worker + scheduler threads that block interpreter
    exit. Call from ``orchestrator.main()``'s finally clause before
    ``os._exit(0)``.
    """
    global _pool
    if _pool is not None:
        try:
            _pool.close(timeout=timeout)
        except Exception as e:
            log.warning("store pool close failed: %s", e)
        _pool = None


@contextmanager
def conn() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool; commits on success, rolls back on error."""
    pool = get_pool()
    with pool.connection() as c:
        yield c


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Borrow a cursor (auto-commits on context exit)."""
    with conn() as c:
        with c.cursor() as cur:
            yield cur


# ---------------------------------------------------------------------------
# Migrations runner
# ---------------------------------------------------------------------------


def applied_versions() -> set[str]:
    """Return the set of migration versions already applied. Empty set if the
    schema_migrations table doesn't exist yet."""
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'schema_migrations')"
            )
            if not cur.fetchone()[0]:
                return set()
            cur.execute("SELECT version FROM schema_migrations")
            return {row[0] for row in cur.fetchall()}
    except psycopg.OperationalError as e:
        log.error("Cannot connect to Postgres: %s", e)
        raise


def migrate(target_dir: Path = MIGRATIONS_DIR, dry_run: bool = False) -> list[str]:
    """Apply every *.sql file in `target_dir` whose version is not yet recorded
    in schema_migrations. Each file is its own transaction (BEGIN/COMMIT in the
    SQL itself). Returns the list of versions applied."""
    if not target_dir.is_dir():
        raise FileNotFoundError(f"migrations dir not found: {target_dir}")

    files = sorted(target_dir.glob("*.sql"))
    if not files:
        log.warning("No migration files found in %s", target_dir)
        return []

    already = applied_versions()
    applied: list[str] = []

    for f in files:
        version = f.stem  # e.g. '002_phase2'
        if version in already:
            log.info("[skip] %s already applied", version)
            continue
        if dry_run:
            log.info("[dry-run] would apply %s (%d bytes)", version, f.stat().st_size)
            applied.append(version)
            continue

        sql = f.read_text()
        log.info("[apply] %s", version)
        with conn() as c:
            with c.cursor() as cur:
                cur.execute(sql)
            # Each migration file does its own COMMIT internally.
            # Defensive close: ensure no pending tx.
            c.commit()
        applied.append(version)

    return applied


def migrate_cli() -> None:
    """Console entry point: `ta-pg-migrate`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dry_run = "--dry-run" in sys.argv
    try:
        applied = migrate(dry_run=dry_run)
    except Exception as e:
        log.error("Migration failed: %s", e)
        sys.exit(1)
    if applied:
        verb = "Would apply" if dry_run else "Applied"
        print(f"{verb}: {applied}")
    else:
        print("All migrations already applied.")


# ---------------------------------------------------------------------------
# Convenience: one-shot status check (used by scripts/status.py extension)
# ---------------------------------------------------------------------------


def health() -> dict:
    """Lightweight probe — used by health check graph + status.py."""
    try:
        with cursor() as cur:
            cur.execute("SELECT 1, current_database(), current_user")
            row = cur.fetchone()
            cur.execute(
                "SELECT count(*)::int FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            n_tables = cur.fetchone()[0]
            return {
                "ok": True,
                "database": row[1],
                "user": row[2],
                "public_tables": n_tables,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Journal queries
# ---------------------------------------------------------------------------


def get_open_journal_trades() -> list[dict] | None:
    """Return open journal_trades rows joined with journal_theses.

    Canonical replacement for journal-mcp ``get_open_positions_with_thesis``
    when called from the autonomous orchestrator: the MCP function reads the
    Phase-1 SQLite ``trades`` table, but autonomous fills only write to
    Postgres ``journal_trades`` (persist_trade_event), so SQLite is
    perpetually empty for fills made by the agent itself.

    Each row dict carries the broker symbol, trade/thesis ids, sizing and
    risk levels, the deterministic exit machinery's fields (``exit_plan``,
    ``mfe_so_far``, ``scale_rungs_taken``) plus the joined thesis fields
    needed by mark-to-market and the exit-trigger enrichment.
    ``strategy_label`` is extracted from ``broker_fill_json`` (it isn't a
    top-level column on the Postgres mirror).

    Rows are ordered newest-first; symbol-keyed consumers MUST keep
    first-row-wins semantics (setdefault) so a phantom (placed-never-filled)
    row sharing a symbol with a later real fill never shadows it.

    Returns ``None`` when the store cannot be read — callers must treat that
    as UNKNOWN, never as "no open trades": an EOD reconcile comparing against
    an empty set during a DB outage produces a confidently wrong all-flat
    report, and an exit pass would misclassify every position as manual.
    """
    sql = """
        SELECT
            t.id AS trade_id, t.symbol, t.asset_type, t.side,
            t.qty, t.entry_price, t.stop, t.target,
            t.exit_plan, t.mfe_so_far,
            COALESCE(t.scale_rungs_taken, 0) AS scale_rungs_taken,
            t.regime_downsized_at_label,
            t.opened_at, t.thesis_id,
            t.broker_fill_json->>'strategy_label' AS strategy_label,
            t.broker_fill_json->'combo' AS combo,
            th.direction, th.thesis_text, th.invalidation,
            th.status AS thesis_status
        FROM journal_trades t
        LEFT JOIN journal_theses th ON th.id = t.thesis_id
        WHERE t.outcome = 'OPEN'
        ORDER BY t.opened_at DESC
    """
    try:
        with cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning("get_open_journal_trades query failed: %s", e)
        return None


if __name__ == "__main__":
    migrate_cli()
