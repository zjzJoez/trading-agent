"""Backfill historical regime features + states from yfinance daily klines.

End-state goal: 2000+ rows in regime_feature_snapshots covering 2018-now,
which is enough training data for the HMM (~4 states × 17 features needs
500+ obs minimum; 2000+ gives stable EM convergence and covers 1-2 full
market regime cycles including 2020 COVID + 2022 bear + 2023 rebound).

Why yfinance and not moomoo:
    moomoo's `get_historical_kline` hangs on multi-year queries (verified
    twice on EC2 5/15). yfinance bulk-downloads 19 tickers × 8 years in
    2.8s, supports the real ^VIX index (moomoo only has VIXY ETF proxy),
    and requires no API key or rate-limit management.

Why we don't store raw klines:
    The +25MB Postgres budget is for the processed snapshots only. Raw
    OHLCV stays in script memory, gets fed ticker-by-ticker into
    build_feature_snapshot() per trading day, and is dropped after the
    snapshot row is INSERTed. Keeps Postgres lean.

Idempotency:
    Skips trading days already present in regime_feature_snapshots
    (matched by DATE(as_of)). Safe to re-run after partial completion.

Usage:
    python -m scripts.backfill_history --start 2018-01-01 --end 2026-05-15

Notes:
    - First 60 trading days are skipped (insufficient lookback for WIN_LONG).
    - vix_level is overridden with the real ^VIX close per day, not the
      spy_rv_20-derived proxy that build_feature_snapshot defaults to.
    - regime_states get rule_based_fallback labels (model=None), with
      `confidence=0.60` per the fallback contract. These act as weak
      labels; once HMM is trained, train_hmm.py + a one-shot relabel
      pass can rewrite them with the real Viterbi-decoded states.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, time, timezone

import pandas as pd
import yfinance as yf

from trading_agent.regime.classifier import classify
from trading_agent.regime.features import build_feature_snapshot
from trading_agent.regime.persist import (
    insert_feature_snapshot,
    insert_regime_state,
)
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

# Map our moomoo-style tickers (used in features.py) to yfinance symbols.
# Note we use real ^VIX (the index) instead of VIXY (the ETF proxy);
# features.py's vix_chg_5 calculation is scale-agnostic, so the absolute
# levels are fine for HMM training in standardized space.
TICKER_MAP = {
    "US.SPY": "SPY", "US.QQQ": "QQQ", "US.IWM": "IWM",
    "US.TLT": "TLT", "US.HYG": "HYG", "US.GLD": "GLD", "US.UUP": "UUP",
    "US.XLK": "XLK", "US.XLF": "XLF", "US.XLE": "XLE", "US.XLV": "XLV",
    "US.XLY": "XLY", "US.XLP": "XLP", "US.XLI": "XLI", "US.XLB": "XLB",
    "US.XLU": "XLU", "US.XLRE": "XLRE", "US.XLC": "XLC",
    "US.VIXY": "^VIX",  # Use real VIX, not VIXY ETF
}


def download_klines(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Bulk-download all tickers via yfinance; return moomoo-keyed dict."""
    log.info("Downloading %d tickers from yfinance: %s → %s",
             len(TICKER_MAP), start, end)
    yf_tickers = list(TICKER_MAP.values())
    df = yf.download(
        yf_tickers, start=start, end=end,
        auto_adjust=True, group_by="ticker", progress=False,
        # Discrete download avoids partial-day NaN rows on the latest bar.
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for moomoo_ticker, yf_ticker in TICKER_MAP.items():
        try:
            sub = df[yf_ticker].dropna(how="all")
        except KeyError:
            log.warning("Ticker %s missing from yfinance result", yf_ticker)
            continue
        if sub.empty:
            log.warning("Ticker %s returned empty frame", yf_ticker)
            continue
        # Index normalized to plain `date` (drop time component).
        sub.index = pd.to_datetime(sub.index).date
        out[moomoo_ticker] = sub

    log.info("Downloaded %d/%d tickers OK", len(out), len(TICKER_MAP))
    return out


def existing_snapshot_dates() -> set:
    """Return set of dates already in regime_feature_snapshots for skip-logic."""
    with cursor() as cur:
        cur.execute(
            "SELECT DATE(as_of AT TIME ZONE 'UTC') FROM regime_feature_snapshots"
        )
        return {d for (d,) in cur.fetchall()}


def backfill_one_day(
    target_date,
    klines_full: dict[str, pd.DataFrame],
    *,
    prior_label: str | None,
) -> tuple[int, int, str]:
    """Build snapshot + classify + persist for one trading day.

    Returns (feature_snapshot_id, regime_state_id, label_predicted).

    `klines_full` is the complete history for all tickers; we slice each
    DataFrame up to and including `target_date` and pass that to the
    feature builder. This is O(N rows) per day, but cheap (~20ms/day);
    backfilling 2000 days takes ~40s, dominated by Postgres INSERTs.
    """
    klines_at_d: dict[str, pd.DataFrame] = {}
    for moomoo_ticker, df in klines_full.items():
        # df.loc[:date] is inclusive; we want "data available by EOD on target_date"
        sub = df.loc[df.index <= target_date]
        if not sub.empty:
            klines_at_d[moomoo_ticker] = sub

    # Override vix_level with real ^VIX close for this day (otherwise
    # build_feature_snapshot derives it from spy_rv_20 × 1.1 × 100, which
    # is a noisy proxy — we have the real index here)
    vix_level = None
    vix_df = klines_at_d.get("US.VIXY")
    if vix_df is not None and len(vix_df) > 0:
        # _close() will read 'Close' column for vix_chg_5/vix_pctl_252, but
        # vix_level is set independently before vix_chg_5 is computed.
        try:
            vix_level = float(vix_df["Close"].iloc[-1])
        except Exception:
            vix_level = None

    as_of_dt = datetime.combine(target_date, time(20, 0), tzinfo=timezone.utc)  # ~ NYSE close
    snap = build_feature_snapshot(
        as_of=as_of_dt,
        klines=klines_at_d,
        vix_level=vix_level,
    )

    # Classify with model=None → rule_based_fallback (confidence=0.60).
    # Once HMM is trained, a separate relabel script can rewrite these
    # using the real Viterbi-decoded states.
    pred = classify(snap, model=None, prior_label=prior_label)

    feature_id = insert_feature_snapshot(snap)
    state_id = insert_regime_state(
        pred,
        as_of=as_of_dt,
        feature_snapshot_id=feature_id,
        model_version_id=None,
        prior_label=prior_label,
        gate={},  # gate is computed at runtime, not historically meaningful
    )
    return feature_id, state_id, pred.label


def main():
    p = argparse.ArgumentParser("backfill_history")
    p.add_argument("--start", default="2018-01-01",
                   help="Earliest trading date to backfill (YYYY-MM-DD)")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Latest trading date (exclusive). Default: today.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N trading days (smoke testing)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    klines_full = download_klines(args.start, args.end)
    if "US.SPY" not in klines_full:
        log.error("SPY missing from yfinance download — cannot proceed")
        sys.exit(1)

    spy_dates = sorted(klines_full["US.SPY"].index)
    # Need at least 60 days of lookback before WIN_LONG features become valid.
    LOOKBACK_DAYS = 60
    if len(spy_dates) <= LOOKBACK_DAYS:
        log.error("Insufficient SPY history (%d days, need >%d)",
                  len(spy_dates), LOOKBACK_DAYS)
        sys.exit(1)
    target_dates = spy_dates[LOOKBACK_DAYS:]
    log.info("SPY data %s → %s; %d target days after %d-day lookback",
             spy_dates[0], spy_dates[-1], len(target_dates), LOOKBACK_DAYS)

    skip = existing_snapshot_dates()
    log.info("Skipping %d days already present in DB", len(skip))

    if args.limit:
        target_dates = target_dates[: args.limit]
        log.info("Limit applied: processing first %d target days", args.limit)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    prior_label: str | None = None

    for d in target_dates:
        if d in skip:
            n_skipped += 1
            # Don't track prior_label from skipped — we'll just use None for
            # the next eligible day. The rule_based_fallback only uses prior
            # for the "label_changed_low_confidence" transition rule, which
            # is bypassed when confidence == 0.60 (fixed).
            continue
        try:
            _fid, _sid, label = backfill_one_day(d, klines_full, prior_label=prior_label)
            n_done += 1
            prior_label = label
            if n_done % 100 == 0:
                log.info("  …%d days backfilled (last=%s, label=%s)", n_done, d, label)
        except Exception as e:
            n_failed += 1
            log.warning("Failed to backfill %s: %s", d, e)
            if n_failed > 50:
                log.error("Too many failures (>50); aborting")
                sys.exit(2)

    log.info(
        "Done. Backfilled: %d, Skipped (already present): %d, Failed: %d",
        n_done, n_skipped, n_failed,
    )

    # Quick sanity dump
    with cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(as_of), MAX(as_of) FROM regime_feature_snapshots")
        n, t0, t1 = cur.fetchone()
        log.info("regime_feature_snapshots: %d rows total, range=%s → %s", n, t0, t1)


if __name__ == "__main__":
    main()
