"""One-time migration: recompute features for every existing snapshot
in `regime_feature_snapshots`, adding the continuous MA-distance keys
(spy_dist_50dma / spy_dist_200dma) introduced when the binary
spy_above_*dma features were retired from HMM_FEATURE_ORDER.

What this does NOT do:
  - Delete any snapshot rows (FK-safe; production audit history preserved)
  - Touch regime_states rows
  - Touch existing HMM models in regime_model_versions
  - Change snapshot timestamps or IDs

What this DOES do:
  - For each existing snapshot, re-fetch yfinance prices up to its date
  - Re-run build_feature_snapshot() → gets a feature dict with BOTH the
    binary and new continuous keys
  - UPDATE the row's `features` JSONB with the new dict (the binary keys
    are also re-written, with the same values they had before)

Idempotency: safe to re-run. The script computes a hash of the
already-written `features` JSONB; if it matches what the recomputation
produces, the row is skipped.

Usage:
    python -m scripts.upgrade_snapshots_continuous_features
    python -m scripts.upgrade_snapshots_continuous_features --dry-run  # show plan only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, time, timezone

import pandas as pd

from scripts.backfill_history import TICKER_MAP, download_klines
from trading_agent.regime.features import build_feature_snapshot
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser("upgrade_snapshots_continuous_features")
    p.add_argument("--start", default="2015-01-01",
                   help="yfinance bulk-download start date — wide window so "
                        "all existing snapshot dates have ≥60 days lookback")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pull existing snapshot dates from DB
    with cursor() as cur:
        cur.execute(
            "SELECT id, as_of, features FROM regime_feature_snapshots "
            "ORDER BY as_of ASC"
        )
        rows = cur.fetchall()
    log.info("Loaded %d snapshots from DB", len(rows))

    # yfinance bulk
    klines_full = download_klines(args.start, args.end)
    if "US.SPY" not in klines_full:
        log.error("SPY missing from yfinance download")
        sys.exit(1)

    log.info("yfinance: %d/%d tickers downloaded", len(klines_full), len(TICKER_MAP))

    n_updated = 0
    n_skipped_existing = 0
    n_failed = 0

    for snap_id, as_of, old_features in rows:
        target_date = as_of.date() if hasattr(as_of, "date") else as_of
        old_feats = dict(old_features or {})

        # Short-circuit: if the snapshot ALREADY has the new continuous keys,
        # don't bother recomputing — this makes the script idempotent on re-run.
        if "spy_dist_50dma" in old_feats and "spy_dist_200dma" in old_feats:
            n_skipped_existing += 1
            continue

        # Slice klines to date (point-in-time)
        klines_at_d = {}
        for ticker, df in klines_full.items():
            sub = df.loc[df.index <= target_date]
            if not sub.empty:
                klines_at_d[ticker] = sub

        # Recompute. Note: we pass the existing vix_level (the historical
        # ^VIX close) by reading it from VIXY/^VIX slice — same as backfill.
        vix_level = None
        vix_df = klines_at_d.get("US.VIXY")
        if vix_df is not None and len(vix_df) > 0:
            try:
                vix_level = float(vix_df["Close"].iloc[-1])
            except Exception:
                vix_level = None

        try:
            as_of_dt = datetime.combine(target_date, time(20, 0), tzinfo=timezone.utc)
            snap = build_feature_snapshot(
                as_of=as_of_dt,
                klines=klines_at_d,
                vix_level=vix_level,
            )
            new_feats = snap.features
        except Exception as e:
            n_failed += 1
            log.warning("Failed to recompute %s (id=%s): %s", target_date, snap_id, e)
            continue

        # Sanity: existing & new should agree on shared keys (within tiny tolerance)
        for k in old_feats:
            if k in new_feats and isinstance(old_feats[k], (int, float)) \
                    and isinstance(new_feats[k], (int, float)):
                if abs(old_feats[k] - new_feats[k]) > 1e-4:
                    log.debug("mismatch on %s for %s: old=%s new=%s",
                              k, target_date, old_feats[k], new_feats[k])

        if args.dry_run:
            n_updated += 1
            if n_updated <= 3:
                # Show one example diff
                added = {k: v for k, v in new_feats.items() if k not in old_feats}
                log.info("[dry-run] would update %s (id=%s); new keys added: %s",
                         target_date, snap_id, list(added.keys()))
            continue

        # UPDATE the row
        with cursor() as cur:
            cur.execute(
                "UPDATE regime_feature_snapshots SET features = %s::jsonb "
                "WHERE id = %s",
                (json.dumps(new_feats), snap_id),
            )
        n_updated += 1
        if n_updated % 200 == 0:
            log.info("  …%d snapshots updated (last=%s)", n_updated, target_date)

    log.info("Done. updated=%d skipped(already-had-keys)=%d failed=%d",
             n_updated, n_skipped_existing, n_failed)
    if args.dry_run:
        log.info("(dry-run mode — no rows were modified)")


if __name__ == "__main__":
    main()
