"""Pre-register a paper-trading benchmark window — Option B from the
"how do we honestly evaluate this system" debate.

The goal is to anchor a measurement period at a specific T0 so the
6-8-week paper trading run can be compared apples-to-apples against
passive baselines. Without a pre-registered start, every "how did we
do" question dissolves into ambiguity ("vs. SPY since when?", "before
or after that disaster Friday?", etc.).

What this script does
---------------------
1. Reads current portfolio equity from `portfolio_marks` (latest row).
   Falls back to `--portfolio-nav` if no mark yet (cold start).
2. Fetches SPY + TLT closes at T0 from yfinance.
3. Computes the implicit 60/40 baseline NAV (same dollars as the
   portfolio at T0, split 60% SPY / 40% TLT, fixed-units construction
   per migration 007 default).
4. INSERTs a row into `benchmark_windows` with status='active'.
5. Refuses to start a second active window unless `--force` is given —
   measurement integrity depends on one window at a time.

The script does NOT mark today's portfolio state as a benchmark_mark.
Use record_benchmark_mark.py for that (daily cron).

Usage
-----
    python -m scripts.start_benchmark_window \\
        --note "first 6-week eval, post Risk-Council burn-in" \\
        --baseline-construction fixed_units

    # Cold start (no portfolio_marks row yet):
    python -m scripts.start_benchmark_window \\
        --portfolio-nav 100000 \\
        --note "T0 with synthetic NAV (pre-trading)"
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def fetch_close(ticker: str, near_date) -> float:
    """Fetch the most recent close for `ticker` on or before `near_date`."""
    import yfinance as yf
    from datetime import timedelta

    # Pull a 10-day window ending near_date+1 so we catch the latest trading
    # day even if near_date is a weekend or holiday.
    start = (near_date - timedelta(days=10)).isoformat()
    end = (near_date + timedelta(days=1)).isoformat()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} near {near_date}")
    close = df["Close"]
    if hasattr(close, "iloc") and close.ndim == 2:
        close = close.iloc[:, 0]
    return float(close.iloc[-1])


def latest_portfolio_nav() -> float | None:
    """Return equity from the most recent portfolio_marks row, or None."""
    with cursor() as cur:
        cur.execute(
            "SELECT equity FROM portfolio_marks ORDER BY as_of DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return float(row[0])


def has_active_window() -> bool:
    with cursor() as cur:
        cur.execute(
            "SELECT id FROM benchmark_windows WHERE status='active' LIMIT 1"
        )
        return cur.fetchone() is not None


def main():
    p = argparse.ArgumentParser("start_benchmark_window")
    p.add_argument("--portfolio-nav", type=float, default=None,
                   help="Override portfolio NAV at T0. If omitted, reads "
                        "latest portfolio_marks.equity. Required if no "
                        "portfolio_marks row exists yet.")
    p.add_argument("--note", default="",
                   help="Free-text rationale. Stored on the row; appears in "
                        "every per-window summary report.")
    p.add_argument("--baseline-construction", default="fixed_units",
                   choices=["fixed_units", "daily_rebal"],
                   help="How the 60/40 baseline is held. 'fixed_units' = "
                        "buy units at T0, hold forever (drifts with markets). "
                        "'daily_rebal' = rebalance to 60/40 each day "
                        "(ignores friction). Pick one; you cannot change "
                        "mid-window without invalidating comparisons.")
    p.add_argument("--force", action="store_true",
                   help="Start a new window even if an active one exists. "
                        "Closes the prior active window with status='aborted'.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if has_active_window() and not args.force:
        log.error(
            "An active benchmark window already exists. Pass --force to "
            "abort it and start a new one, or close it manually first."
        )
        sys.exit(2)

    # Portfolio NAV at T0
    nav = args.portfolio_nav
    if nav is None:
        nav = latest_portfolio_nav()
        if nav is None:
            log.error(
                "No portfolio_marks row exists and --portfolio-nav not given. "
                "Either run an EOD mark first or pass --portfolio-nav."
            )
            sys.exit(3)
        log.info("Using portfolio NAV from latest portfolio_marks: %.2f", nav)
    else:
        log.info("Using --portfolio-nav: %.2f", nav)

    t0 = datetime.now(timezone.utc)
    log.info("Fetching SPY + TLT closes near %s …", t0.date())
    spy_close = fetch_close("SPY", t0.date())
    tlt_close = fetch_close("TLT", t0.date())
    log.info("SPY=%.4f  TLT=%.4f", spy_close, tlt_close)

    # 60/40 baseline NAV at T0 by construction is just the same NAV. We store
    # it explicitly so a future query doesn't have to assume.
    baseline_nav = nav

    print()
    print("=" * 60)
    print(f"BENCHMARK WINDOW T0 — {t0.isoformat()}")
    print(f"  portfolio_nav     = ${nav:,.2f}")
    print(f"  spy_close         = ${spy_close:.4f}")
    print(f"  tlt_close         = ${tlt_close:.4f}")
    print(f"  baseline 60/40    = ${baseline_nav:,.2f} ({args.baseline_construction})")
    print(f"  note              = {args.note or '(none)'}")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run: not inserting")
        return

    with cursor() as cur:
        if args.force:
            cur.execute(
                "UPDATE benchmark_windows SET status='aborted', "
                "closed_at=NOW() WHERE status='active'"
            )
            log.info("Aborted %d prior active window(s)", cur.rowcount)
        cur.execute(
            """
            INSERT INTO benchmark_windows
                (t0_ts, t0_spy_close, t0_tlt_close, t0_portfolio_nav,
                 t0_baseline_60_40_nav, baseline_construction, status, note)
            VALUES (%s, %s, %s, %s, %s, %s, 'active', %s)
            RETURNING id
            """,
            (t0, spy_close, tlt_close, nav, baseline_nav,
             args.baseline_construction, args.note),
        )
        window_id = int(cur.fetchone()[0])

    print(f"\n✓ Inserted benchmark_windows id={window_id} (status=active)")
    print("Next step: schedule a daily cron to run record_benchmark_mark.py")
    print("           e.g.  16:30 ET on trading days (after EOD mark settles)")


if __name__ == "__main__":
    main()
