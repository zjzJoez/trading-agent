"""Record one daily mark for the active benchmark window.

Run on a daily cron (after EOD reconcile lands a portfolio_marks row).
The script is intentionally idempotent — a UNIQUE(window_id, mark_date)
constraint means re-running on the same trading day is a no-op (raises
on duplicate, exit 0 if --idempotent).

Pipeline
--------
1. Look up the single active row in `benchmark_windows`. If none → exit 0
   silently (benchmark window not started yet, nothing to record).
2. Read today's portfolio equity from the most recent `portfolio_marks`
   row. If the latest mark predates today by more than 2 days, error out
   — we don't want stale data corrupting the series.
3. Fetch SPY + TLT close from yfinance for today (or last trading day if
   the cron runs before yfinance has today's bar).
4. Compute the 60/40 baseline NAV per the construction recorded at T0:
     fixed_units:   spy_units * spy_today + tlt_units * tlt_today
     daily_rebal:   yesterday's baseline NAV × (0.6 * spy_ret + 0.4 * tlt_ret)
5. Compute cumulative returns from T0 for strategy / SPY / baseline.
6. INSERT into benchmark_marks (UNIQUE on (window_id, mark_date)).

Usage
-----
    # Normal cron use:
    python -m scripts.record_benchmark_mark

    # If today's mark already exists, exit cleanly without raising:
    python -m scripts.record_benchmark_mark --idempotent

The script does NOT close the window — that's a manual decision (operator
runs UPDATE benchmark_windows SET status='closed' when the eval period
ends and the post-mortem can begin).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def fetch_latest_close(ticker: str) -> tuple[date, float]:
    """Most recent yfinance close for `ticker`, returns (trade_date, close)."""
    import yfinance as yf

    today = date.today()
    start = (today - timedelta(days=10)).isoformat()
    end = (today + timedelta(days=2)).isoformat()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if hasattr(close, "iloc") and close.ndim == 2:
        close = close.iloc[:, 0]
    last_date = close.index[-1].date() if hasattr(close.index[-1], "date") else close.index[-1]
    return last_date, float(close.iloc[-1])


def get_active_window() -> dict | None:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, t0_ts, t0_spy_close, t0_tlt_close, t0_portfolio_nav,
                   t0_baseline_60_40_nav, baseline_construction
            FROM benchmark_windows
            WHERE status='active'
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "t0_ts": row[1],
            "t0_spy": float(row[2]),
            "t0_tlt": float(row[3]),
            "t0_nav": float(row[4]),
            "t0_baseline": float(row[5]),
            "construction": row[6],
        }


def latest_portfolio_mark() -> tuple[datetime, float] | None:
    with cursor() as cur:
        cur.execute(
            "SELECT as_of, equity FROM portfolio_marks ORDER BY as_of DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return row[0], float(row[1])


def last_baseline_nav(window_id: int) -> float | None:
    """Most recent benchmark_marks.baseline_60_40_nav for this window."""
    with cursor() as cur:
        cur.execute(
            "SELECT baseline_60_40_nav FROM benchmark_marks "
            "WHERE window_id=%s ORDER BY mark_date DESC LIMIT 1",
            (window_id,),
        )
        row = cur.fetchone()
        return float(row[0]) if row else None


def previous_spy_tlt(window_id: int) -> tuple[float, float] | None:
    """Most recent (spy_close, tlt_close) recorded for this window."""
    with cursor() as cur:
        cur.execute(
            "SELECT spy_close, tlt_close FROM benchmark_marks "
            "WHERE window_id=%s ORDER BY mark_date DESC LIMIT 1",
            (window_id,),
        )
        row = cur.fetchone()
        return (float(row[0]), float(row[1])) if row else None


def compute_baseline_nav(
    construction: str,
    t0_nav: float,
    t0_spy: float,
    t0_tlt: float,
    spy_now: float,
    tlt_now: float,
    window_id: int,
) -> float:
    """60/40 NAV at the current mark, using the construction set at T0."""
    if construction == "fixed_units":
        spy_units = 0.6 * t0_nav / t0_spy
        tlt_units = 0.4 * t0_nav / t0_tlt
        return spy_units * spy_now + tlt_units * tlt_now

    if construction == "daily_rebal":
        prev = last_baseline_nav(window_id)
        if prev is None:
            # First mark — same as fixed_units day-1
            return t0_nav  # SPY hasn't moved yet vs T0
        prev_prices = previous_spy_tlt(window_id)
        if prev_prices is None:
            return prev
        prev_spy, prev_tlt = prev_prices
        spy_ret = (spy_now / prev_spy) - 1.0 if prev_spy else 0.0
        tlt_ret = (tlt_now / prev_tlt) - 1.0 if prev_tlt else 0.0
        return prev * (1.0 + 0.6 * spy_ret + 0.4 * tlt_ret)

    raise ValueError(f"Unknown baseline_construction: {construction}")


def main():
    p = argparse.ArgumentParser("record_benchmark_mark")
    p.add_argument("--portfolio-nav", type=float, default=None,
                   help="Override portfolio NAV. If omitted, read from "
                        "latest portfolio_marks.equity (≤2 days old).")
    p.add_argument("--max-mark-age-days", type=int, default=2,
                   help="Refuse if portfolio_marks is older than this. "
                        "Prevents stale-data poisoning of the series.")
    p.add_argument("--idempotent", action="store_true",
                   help="Exit 0 silently if today's mark already exists.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    win = get_active_window()
    if not win:
        log.info("No active benchmark window. Exiting (nothing to record).")
        return

    # Portfolio NAV
    nav = args.portfolio_nav
    if nav is None:
        latest = latest_portfolio_mark()
        if latest is None:
            log.error("No portfolio_marks row exists and --portfolio-nav not given.")
            sys.exit(3)
        mark_ts, nav = latest
        age = datetime.now(timezone.utc) - mark_ts
        if age > timedelta(days=args.max_mark_age_days):
            log.error("Latest portfolio_marks is %s old (>%d days). Refusing to "
                      "record a stale mark; rerun EOD reconcile first.",
                      age, args.max_mark_age_days)
            sys.exit(4)

    # Prices
    spy_date, spy_close = fetch_latest_close("SPY")
    tlt_date, tlt_close = fetch_latest_close("TLT")
    if spy_date != tlt_date:
        log.warning("SPY/TLT close dates disagree (%s vs %s); using SPY's",
                    spy_date, tlt_date)
    mark_date = spy_date

    baseline_nav = compute_baseline_nav(
        construction=win["construction"],
        t0_nav=win["t0_nav"],
        t0_spy=win["t0_spy"],
        t0_tlt=win["t0_tlt"],
        spy_now=spy_close,
        tlt_now=tlt_close,
        window_id=win["id"],
    )

    cum_strategy = (nav / win["t0_nav"]) - 1.0
    cum_spy = (spy_close / win["t0_spy"]) - 1.0
    cum_baseline = (baseline_nav / win["t0_baseline"]) - 1.0

    # INSERT with idempotency handling
    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_marks
                    (window_id, ts, mark_date, spy_close, tlt_close,
                     portfolio_nav, baseline_60_40_nav,
                     cum_strategy_ret, cum_spy_ret, cum_baseline_ret)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (win["id"], mark_date, spy_close, tlt_close, nav, baseline_nav,
                 cum_strategy, cum_spy, cum_baseline),
            )
    except Exception as e:
        if "duplicate key" in str(e).lower() and args.idempotent:
            log.info("Mark for %s already exists; idempotent exit", mark_date)
            return
        raise

    print(f"✓ Recorded benchmark_marks row for window {win['id']} on {mark_date}:")
    print(f"  strategy NAV  = ${nav:,.2f}  (cum return {cum_strategy:+.2%})")
    print(f"  passive SPY   = ${spy_close:.4f}  (cum return {cum_spy:+.2%})")
    print(f"  60/40 baseline= ${baseline_nav:,.2f} (cum return {cum_baseline:+.2%})")


if __name__ == "__main__":
    main()
