"""Rsync trader.ec2.db back to Mac and merge via INSERT OR IGNORE.

Runs on the Mac (not EC2) on a cron or manual invocation. Pulls the remote
DB over SSH to /tmp, then opens both DBs and merges:
  - notes: dedup on (created_at, topic, tags)
  - theses: dedup on (created_at, ticker, thesis_text)
  - post_mortems: dedup on (period_start, period_end)

We do NOT sync `trades`, `market_snapshots`, or `notes_vec`:
  - trades: only the Mac ever places orders (EC2 has no OpenD)
  - snapshots: same — tied to trades
  - notes_vec: embeddings are regenerated Mac-side after sync

Usage:
  python -m trading_agent.jobs.db_sync \\
    --remote ubuntu@dublin.example.com \\
    --remote-db ~/trading-agent/data/trader.ec2.db

Safe to re-run: INSERT OR IGNORE plus dedup indexes make it idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection, migrate


MERGE_TABLES = {
    # table -> tuple of columns used for idempotency
    "notes":        ("created_at", "topic", "text", "tags", "source"),
    "theses":       ("created_at", "ticker", "direction", "thesis_text",
                     "invalidation", "timeframe", "expected_return_pct",
                     "max_loss_pct", "status"),
    "post_mortems": ("period_start", "period_end", "summary",
                     "lessons_json", "created_at"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rsync(remote: str, remote_path: str, local_path: Path) -> None:
    src = f"{remote}:{remote_path}"
    cmd = ["rsync", "-az", "--partial", "--timeout=60", src, str(local_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


def _merge_table(local: sqlite3.Connection, remote: sqlite3.Connection,
                 table: str, dedup_cols: tuple) -> tuple[int, int]:
    """Returns (inserted, skipped)."""
    cols_meta = remote.execute(f"PRAGMA table_info({table})").fetchall()
    cols = [c[1] for c in cols_meta if c[1] != "id"]
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    rows = remote.execute(f"SELECT {col_list} FROM {table}").fetchall()
    inserted = skipped = 0
    for row in rows:
        # Composite dedup check.
        where = " AND ".join(f"{c} IS ?" for c in dedup_cols)
        dedup_values = [row[cols.index(c)] for c in dedup_cols]
        hit = local.execute(
            f"SELECT 1 FROM {table} WHERE {where} LIMIT 1",
            dedup_values,
        ).fetchone()
        if hit:
            skipped += 1
            continue
        local.execute(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            row,
        )
        inserted += 1
    local.commit()
    return inserted, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=False,
                    help="ssh target, e.g. ubuntu@dublin.example.com")
    ap.add_argument("--remote-db", default="~/trading-agent/data/trader.ec2.db",
                    help="path to EC2 DB")
    ap.add_argument("--local-ec2-db", default=None,
                    help="if set, skip rsync and use this local file "
                         "(useful for testing or when rsync already happened)")
    args = ap.parse_args()

    ensure_dirs()
    migrate(CONFIG.db_path)  # ensure local schema is current

    if args.local_ec2_db:
        ec2_path = Path(args.local_ec2_db)
        if not ec2_path.exists():
            print(f"ERROR: local ec2 db not found: {ec2_path}", file=sys.stderr)
            return 2
    else:
        if not args.remote:
            print("ERROR: --remote is required when --local-ec2-db is not set",
                  file=sys.stderr)
            return 2
        tmp = Path(tempfile.mkdtemp(prefix="db_sync_")) / "trader.ec2.db"
        print(f"[db_sync] rsync {args.remote}:{args.remote_db} -> {tmp}")
        _rsync(args.remote, args.remote_db, tmp)
        ec2_path = tmp

    report = {"ts": _now(), "source": str(ec2_path), "target": str(CONFIG.db_path),
              "tables": {}}

    with connection() as local:
        # Attach remote read-only.
        remote_conn = sqlite3.connect(f"file:{ec2_path}?mode=ro", uri=True)
        remote_conn.row_factory = sqlite3.Row
        try:
            for table, dedup_cols in MERGE_TABLES.items():
                # Skip if remote table is empty or missing.
                try:
                    n_remote = remote_conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    report["tables"][table] = {"skipped_reason": "missing on remote"}
                    continue
                if n_remote == 0:
                    report["tables"][table] = {"remote_count": 0, "inserted": 0}
                    continue
                ins, skip = _merge_table(local, remote_conn, table, dedup_cols)
                report["tables"][table] = {
                    "remote_count": n_remote,
                    "inserted": ins, "skipped": skip,
                }
        finally:
            remote_conn.close()

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
