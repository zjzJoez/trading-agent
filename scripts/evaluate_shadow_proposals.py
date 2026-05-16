"""Daily cron: compute forward outcomes for shadow_proposals rows.

For each shadow_proposals row that is ≥20 trading days old and has no
`fwd_evaluated_at` set:
  1. Fetch the underlying ticker's price from yfinance for proposal_ts and
     proposal_ts + 1d / 5d / 20d (or as many days forward as exist).
  2. Compute simple returns at each horizon.
  3. Apply a direction sign: LONG / LONG_CALL → +1; LONG_PUT → -1; etc.
  4. Mark `fwd_directional_correct_*` based on whether direction matches
     the forward return sign.
  5. UPDATE the row with all the computed values + `fwd_evaluated_at = NOW()`.

This script is intentionally simple: it measures **underlying** direction
quality, not option PnL. Options pricing requires the historical option
chain at exit — we don't have that. The underlying direction proxy is a
reasonable signal for "did the LLM call the right direction" without
modeling vega/theta.

Usage:
    python -m scripts.evaluate_shadow_proposals
    python -m scripts.evaluate_shadow_proposals --min-age-days 5  # eval 5d-old too
    python -m scripts.evaluate_shadow_proposals --dry-run

Scheduled by cron AFTER 22:00 UTC trading-day evening so yfinance has
the closing print.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


# Direction → sign for "did the call direction match"
DIRECTION_SIGN = {
    "LONG": +1, "LONG_CALL": +1, "BUY": +1,
    "LONG_PUT": -1, "SHORT": -1, "SHORT_CALL": -1, "SELL": -1,
}


def fetch_ticker_close_series(ticker: str, start: date, end: date):
    """Daily close series from yfinance, indexed by date."""
    import yfinance as yf
    import pandas as pd
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=2)).isoformat(),
        progress=False, auto_adjust=True,
    )
    if df.empty:
        return None
    close = df["Close"]
    if hasattr(close, "iloc") and close.ndim == 2:
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).date
    return close


def compute_forward_return(close_series, proposal_date: date, horizon_days: int) -> float | None:
    """Compound return from proposal_date close to proposal_date + horizon_days
    (in trading days)."""
    if close_series is None:
        return None
    # Find the first index >= proposal_date (entry close)
    after = [d for d in close_series.index if d >= proposal_date]
    if len(after) < 2:
        return None
    entry_date = after[0]
    entry_price = float(close_series.loc[entry_date])
    if entry_price <= 0:
        return None
    # Take the close `horizon_days` trading days after entry_date
    later = [d for d in close_series.index if d > entry_date]
    if len(later) < horizon_days:
        return None
    exit_date = later[horizon_days - 1]
    exit_price = float(close_series.loc[exit_date])
    return exit_price / entry_price - 1.0


def main():
    p = argparse.ArgumentParser("evaluate_shadow_proposals")
    p.add_argument("--min-age-days", type=int, default=20,
                   help="Only evaluate proposals at least this many calendar "
                        "days old. Default 20 ≈ ~14 trading days, enough for "
                        "the 20d forward return to be computable.")
    p.add_argument("--limit", type=int, default=200,
                   help="Max proposals to evaluate this run.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_age_days)
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, proposal_ts, ticker, direction, final_action
            FROM shadow_proposals
            WHERE fwd_evaluated_at IS NULL
              AND proposal_ts <= %s
            ORDER BY proposal_ts ASC
            LIMIT %s
            """,
            (cutoff, args.limit),
        )
        rows = cur.fetchall()

    log.info("Pending evaluation: %d proposals older than %s",
             len(rows), cutoff.isoformat())

    if not rows:
        return

    # Group by ticker so we fetch each ticker's series once
    by_ticker: dict[str, list] = {}
    min_dt = max_dt = None
    for r in rows:
        t = (r[2] or "").upper()
        if not t:
            continue
        by_ticker.setdefault(t, []).append(r)
        dt = r[1].date() if hasattr(r[1], "date") else r[1]
        min_dt = dt if min_dt is None else min(min_dt, dt)
        max_dt = dt if max_dt is None else max(max_dt, dt)

    today = date.today()
    series_cache = {}
    for t in by_ticker:
        log.info("Fetching %s closes [%s, %s] …", t, min_dt, today)
        s = fetch_ticker_close_series(t, min_dt - timedelta(days=5), today)
        if s is None:
            log.warning("No yfinance data for %s — will mark proposals with null fwd", t)
        series_cache[t] = s

    n_evaluated = 0
    n_skipped = 0
    for t, proposals in by_ticker.items():
        s = series_cache.get(t)
        for (pid, p_ts, ticker, direction, final_action) in proposals:
            p_date = p_ts.date() if hasattr(p_ts, "date") else p_ts
            fwd_1d = compute_forward_return(s, p_date, 1) if s is not None else None
            fwd_5d = compute_forward_return(s, p_date, 5) if s is not None else None
            fwd_20d = compute_forward_return(s, p_date, 20) if s is not None else None

            sign = DIRECTION_SIGN.get(direction or "", 0)
            if sign == 0 or fwd_5d is None:
                correct_5d = None
            else:
                correct_5d = (sign * fwd_5d) > 0
            if sign == 0 or fwd_20d is None:
                correct_20d = None
            else:
                correct_20d = (sign * fwd_20d) > 0

            note = {
                "evaluated_via": "yfinance_underlying",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "fwd_5d_available": fwd_5d is not None,
                "fwd_20d_available": fwd_20d is not None,
            }

            if args.dry_run:
                log.info(
                    "[dry-run] id=%d ticker=%s dir=%s fwd_5d=%s fwd_20d=%s correct_5d=%s correct_20d=%s",
                    pid, ticker, direction, fwd_5d, fwd_20d, correct_5d, correct_20d,
                )
                n_evaluated += 1
                continue

            try:
                with cursor() as cur:
                    cur.execute(
                        """
                        UPDATE shadow_proposals
                        SET fwd_evaluated_at = NOW(),
                            fwd_1d_underlying_ret = %s,
                            fwd_5d_underlying_ret = %s,
                            fwd_20d_underlying_ret = %s,
                            fwd_directional_correct_5d = %s,
                            fwd_directional_correct_20d = %s,
                            fwd_eval_notes = %s::jsonb
                        WHERE id = %s
                        """,
                        (fwd_1d, fwd_5d, fwd_20d, correct_5d, correct_20d,
                         json.dumps(note), pid),
                    )
                n_evaluated += 1
            except Exception as e:
                log.warning("Failed to update id=%d: %s", pid, e)
                n_skipped += 1

    log.info("Done. Evaluated=%d, Skipped=%d, Total pending=%d",
             n_evaluated, n_skipped, len(rows))


if __name__ == "__main__":
    main()
