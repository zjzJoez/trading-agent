"""Phase 0.4 — journal ledger decontamination (one-shot, idempotent).

The SQLite journal is the only ground truth the learning loop has, and it was
contaminated four ways:

  1. Virtual backfills of the operator's REAL trades sat indistinguishable
     from agent paper fills — including an XLE bear-call spread *scaled 1→5*,
     booking a 5x-amplified copy of a real loss into the paper record.
  2. 54 real-account mirror rows (REALMIRROR-*) shared aggregates with agent
     trades and double-counted one GLD loss already closed as a paper trade.
  3. Five "post-mortem lessons" were seeded two hours BEFORE the first trade
     existed, citing statistics found nowhere in the DB — yet retrievable by
     search_notes(source='post_mortem') as if learned from experience.
  4. close_trade trusted caller-supplied pnl; trade id 8 recorded -$136
     against leg arithmetic of -$268 and nothing flagged it.

This script: backs up the DB, applies the schema retrofit (provenance / fees /
pnl_recomputed / pnl_mismatch / exit_* columns via db.migrate), backfills
provenance from row markers, re-sources the pre-trade seeded lessons to
'seeded_prior' (excluded from post_mortem retrieval, preserved for audit),
and recomputes gross P&L from stored legs, flagging mismatches. It never
deletes or overwrites the original pnl — corrections live in new columns.

Usage:
    .venv/bin/python scripts/migrate_journal_provenance.py \
        --db /Users/joez/Desktop/trading-agent/data/trader.db [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Self-reported pnl vs leg arithmetic beyond max($1, 1%) → pnl_mismatch=1.
TOLERANCE_USD = 1.0


def _classify(broker_order_id: str | None, reasoning: str | None) -> str:
    boid = broker_order_id or ""
    txt = (reasoning or "").lower()
    if boid.startswith("REALMIRROR-"):
        return "real_mirror"
    if "[dry-run" in txt:
        return "dry_run"
    if boid.startswith("VIRTUAL-"):
        markers = ("mirroring real", "real fill", "real-account", "scaled 1")
        if any(m in txt for m in markers):
            return "virtual_backfill"
        return "virtual"
    return "agent"


def _recompute_pnl(row: sqlite3.Row) -> float | None:
    entry = row["entry_price"]
    exit_ = row["exit_price"]
    qty = row["qty"]
    if entry is None or exit_ is None or qty is None:
        return None
    mult = 100.0 if (row["asset_type"] or "").upper() == "OPT" else 1.0
    side = (row["side"] or "BUY").upper()
    if side == "SELL":  # SELL-to-open: profit when exit < entry
        gross = (float(entry) - float(exit_)) * float(qty) * mult
    else:
        gross = (float(exit_) - float(entry)) * float(qty) * mult
    return round(gross, 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found", file=sys.stderr)
        return 1

    # 1. Backup (skipped on dry-run; nothing will be written anyway)
    if not args.dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = args.db.with_suffix(f".db.bak-{stamp}")
        shutil.copy2(args.db, backup)
        # WAL sidecar carries un-checkpointed writes; copy it too if present.
        wal = Path(str(args.db) + "-wal")
        if wal.exists():
            shutil.copy2(wal, Path(str(backup) + "-wal"))
        print(f"backup: {backup}")

    # 2. Schema retrofit via the canonical migrate path
    if not args.dry_run:
        from trading_agent.db import migrate
        migrate(args.db)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row

    # On dry-run against an un-migrated DB the new columns may be absent.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    can_write = {"provenance", "pnl_recomputed", "pnl_mismatch"} <= cols
    if args.dry_run and not can_write:
        print("dry-run: provenance columns absent (un-migrated DB) — "
              "classification preview only")

    # 3. Provenance backfill
    counts: dict[str, int] = {}
    updates: list[tuple[str, int]] = []
    for row in conn.execute(
            "SELECT id, broker_order_id, reasoning FROM trades"):
        prov = _classify(row["broker_order_id"], row["reasoning"])
        counts[prov] = counts.get(prov, 0) + 1
        updates.append((prov, row["id"]))
    if not args.dry_run:
        conn.executemany(
            "UPDATE trades SET provenance = ? WHERE id = ?", updates)
    print(f"provenance: {dict(sorted(counts.items()))}")

    # 4. Seeded "lessons" — created before the first trade ever existed.
    first_trade = conn.execute(
        "SELECT MIN(opened_at) FROM trades").fetchone()[0]
    seeded = conn.execute(
        "SELECT id, created_at, topic FROM notes "
        "WHERE source = 'post_mortem' AND created_at < ?",
        (first_trade,),
    ).fetchall()
    for n in seeded:
        print(f"seeded lesson → seeded_prior: note {n['id']} "
              f"({n['created_at']}) {n['topic']}")
    if not args.dry_run and seeded:
        conn.executemany(
            "UPDATE notes SET source = 'seeded_prior' WHERE id = ?",
            [(n["id"],) for n in seeded],
        )

    # 5. P&L recompute + mismatch flags (closed trades with a recorded pnl)
    mismatches = []
    recomputed_n = 0
    for row in conn.execute(
            "SELECT id, entry_price, exit_price, qty, asset_type, side, pnl "
            "FROM trades WHERE outcome IN ('WIN','LOSS','SCRATCH')"):
        gross = _recompute_pnl(row)
        if gross is None:
            continue
        recomputed_n += 1
        pnl = row["pnl"]
        mismatch = 0
        if pnl is not None:
            tol = max(TOLERANCE_USD, abs(gross) * 0.01)
            mismatch = 1 if abs(float(pnl) - gross) > tol else 0
        if mismatch:
            mismatches.append((row["id"], pnl, gross))
        if not args.dry_run and can_write:
            conn.execute(
                "UPDATE trades SET pnl_recomputed = ?, pnl_mismatch = ? "
                "WHERE id = ?",
                (gross, mismatch, row["id"]),
            )
    print(f"pnl recomputed for {recomputed_n} closed trades")
    for tid, pnl, gross in mismatches:
        print(f"  MISMATCH trade {tid}: self-reported pnl={pnl} "
              f"vs leg arithmetic={gross}")

    if args.dry_run:
        conn.rollback()
        print("dry-run: no changes written")
    else:
        conn.commit()
        print("committed")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
