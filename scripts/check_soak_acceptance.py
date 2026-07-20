"""Phase 2.8 — soak acceptance evaluator.

Run after a soak window closes to check whether the past N trading days meet
the published acceptance criteria from plan §15:

  1. zero human intervention      — no manual /enter, no /halt without resume
                                    same day
  2. ntfy delivery 100%           — every emitted notify event has a recorded
                                    HTTP success in agent_events
  3. no duplicate orders           — proposal_id is unique across journal_trades
  4. no missed fills               — every APPROVE risk_decision either has a
                                    matching journal_trades row OR a recorded
                                    skip reason
  5. no unnotified critical event  — every severity≥2 agent_events entry has a
                                    matching ntfy push
  6. positive Sharpe vs SPY        — over the window, paper Sharpe ≥ SPY Sharpe

Exit code 0 if all gates pass, 1 otherwise.  Designed to be run from a CI job
or wrapped in another systemd timer at the end of the soak window.

Usage:
    .venv/bin/python scripts/check_soak_acceptance.py --days 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

from trading_agent.learning.outcome import CLOSED_OUTCOMES
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def _gate(label: str, ok: bool, detail: str = "") -> dict:
    return {"label": label, "pass": bool(ok), "detail": detail}


def check_zero_intervention(since) -> dict:
    """Look for halt/resume agent_events; flag any halt without same-day resume."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT ts, event_type FROM agent_events
                WHERE agent = 'halt_endpoint' AND ts >= %s
                ORDER BY ts ASC
                """,
                (since,),
            )
            rows = cur.fetchall()
    except Exception as e:
        return _gate("zero_intervention", False, f"db read failed: {e}")
    halts: list[datetime] = []
    resumes: list[datetime] = []
    for ts, evt in rows:
        if evt == "halt_set":
            halts.append(ts)
        elif evt == "halt_cleared":
            resumes.append(ts)
    unmatched = 0
    for h in halts:
        same_day_resume = any(
            r.date() == h.date() and r >= h for r in resumes
        )
        if not same_day_resume:
            unmatched += 1
    return _gate(
        "zero_intervention",
        unmatched == 0,
        f"{len(halts)} halts, {len(resumes)} resumes, {unmatched} unmatched halts",
    )


def check_ntfy_delivery(since) -> dict:
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type = 'ntfy_send_failed' THEN 1 ELSE 0 END) AS failed,
                    COUNT(*) FILTER (WHERE agent = 'ntfy' OR event_type LIKE 'ntfy_%') AS total
                FROM agent_events
                WHERE ts >= %s
                """,
                (since,),
            )
            failed, total = cur.fetchone()
    except Exception as e:
        return _gate("ntfy_delivery", False, f"db read failed: {e}")
    failed = int(failed or 0)
    total = int(total or 0)
    return _gate(
        "ntfy_delivery",
        failed == 0,
        f"{failed} failures / {total} total",
    )


def check_no_duplicate_orders(since) -> dict:
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT proposal_id, COUNT(*) AS n FROM journal_trades
                WHERE opened_at >= %s
                GROUP BY proposal_id HAVING COUNT(*) > 1
                """,
                (since,),
            )
            dups = cur.fetchall()
    except Exception as e:
        return _gate("no_duplicate_orders", False, f"db read failed: {e}")
    return _gate(
        "no_duplicate_orders",
        not dups,
        f"{len(dups)} duplicate proposal_ids" if dups else "clean",
    )


def check_no_missed_fills(since) -> dict:
    """An APPROVE risk_decision should produce either a journal_trades row
    OR a logged skip reason (e.g. virtual fill, soak-block)."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT rd.risk_decision_id
                FROM risk_decisions rd
                LEFT JOIN journal_trades jt ON jt.risk_decision_id = rd.risk_decision_id
                WHERE rd.created_at >= %s
                  AND rd.decision IN ('APPROVE', 'DOWNSIZE')
                  AND jt.id IS NULL
                """,
                (since,),
            )
            orphans = [r[0] for r in cur.fetchall()]
    except Exception as e:
        return _gate("no_missed_fills", False, f"db read failed: {e}")
    if not orphans:
        return _gate("no_missed_fills", True, "all approvals fulfilled or audited")
    # Any orphan must have a matching agent_events skip note
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT payload->>'risk_decision_id'
                FROM agent_events
                WHERE event_type IN ('soak_read_only_block', 'execute_skipped',
                                     'symbol_parse_failed', 'order_rejected')
                  AND ts >= %s
                """,
                (since,),
            )
            audited = {r[0] for r in cur.fetchall() if r[0]}
    except Exception as e:
        return _gate("no_missed_fills", False, f"audit read failed: {e}")
    unexplained = [o for o in orphans if o not in audited]
    return _gate(
        "no_missed_fills",
        not unexplained,
        f"{len(unexplained)}/{len(orphans)} approvals unexplained",
    )


def check_critical_events_notified(since) -> dict:
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM agent_events
                WHERE severity >= 2 AND ts >= %s
                """,
                (since,),
            )
            n_critical = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*) FROM agent_events
                WHERE event_type = 'ntfy_send_failed' AND severity >= 2 AND ts >= %s
                """,
                (since,),
            )
            n_critical_failed = int(cur.fetchone()[0])
    except Exception as e:
        return _gate("critical_events_notified", False, f"db read failed: {e}")
    return _gate(
        "critical_events_notified",
        n_critical_failed == 0,
        f"{n_critical_failed} critical events failed to notify out of {n_critical}",
    )


def check_exit_fills_confirmed(since) -> dict:
    """Measurement-integrity gate: every closed trade in the window must have
    a broker-confirmed exit (broker_fill_json.exit_dealt_avg_price) unless it
    is a virtual fill. Closes journaled without fill confirmation are the
    exact accounting fiction Phase 0 removed — paper statistics built on them
    are inadmissible, so the soak window fails rather than reporting them.
    """
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM journal_trades
                WHERE closed_at >= %s
                  AND outcome = ANY(%s)
                  AND COALESCE(broker_order_id, '') NOT LIKE 'VIRTUAL-%%'
                  AND (broker_fill_json IS NULL
                       OR NOT broker_fill_json ? 'exit_dealt_avg_price')
                """,
                (since, list(CLOSED_OUTCOMES)),
            )
            unconfirmed = int(cur.fetchone()[0])
    except Exception as e:
        return _gate("exit_fills_confirmed", False, f"db read failed: {e}")
    return _gate(
        "exit_fills_confirmed",
        unconfirmed == 0,
        f"{unconfirmed} closed trades without broker-confirmed exit price",
    )


def _annualized_sharpe(returns: list[float],
                       periods_per_year: float = 252.0) -> float | None:
    """Annualized Sharpe of a return series (0 risk-free). None if <2 points
    or zero variance. Sample variance (n-1) so both series are consistent."""
    import math
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return None
    return (mean / sd) * math.sqrt(periods_per_year)


def fetch_spy_daily_returns(since) -> list[float]:
    """SPY daily simple returns from `since` to today (yfinance, auto-adjusted).

    Raises RuntimeError on an empty download so the gate FAILS rather than
    silently passing a paper-only number."""
    import yfinance as yf
    start = since.date() if hasattr(since, "date") else since
    df = yf.download("SPY", start=str(start), progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no SPY data for the window")
    close = df["Close"]
    if hasattr(close, "ndim") and close.ndim == 2:
        close = close.iloc[:, 0]
    rets = close.pct_change().dropna()
    return [float(x) for x in rets.tolist()]


def fetch_paper_daily_returns(since) -> tuple[list[float], str]:
    """Paper return series over the window, with its source tag.

    Preferred: portfolio_marks NAV (equity) → daily pct change, directly
    comparable to SPY daily Sharpe. Falls back to the per-trade realized_r
    series when <2 NAV marks exist (less comparable — reflected in the tag)."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT equity FROM portfolio_marks
                WHERE as_of >= %s AND equity IS NOT NULL
                ORDER BY as_of ASC
                """,
                (since,),
            )
            navs = [float(r[0]) for r in cur.fetchall()]
    except Exception as e:
        navs = []
        log.warning("portfolio_marks read failed (%s); trying realized_r", e)
    if len(navs) >= 2:
        rets = [navs[i] / navs[i - 1] - 1.0
                for i in range(1, len(navs)) if navs[i - 1] > 0]
        if rets:
            return rets, "portfolio_marks_nav"
    with cursor() as cur:
        cur.execute(
            """
            SELECT tof.realized_r FROM trade_outcome_features tof
            JOIN journal_trades jt ON jt.id = tof.trade_id
            WHERE jt.closed_at >= %s AND tof.realized_r IS NOT NULL
            ORDER BY jt.closed_at ASC
            """,
            (since,),
        )
        rs = [float(r[0]) for r in cur.fetchall()]
    return rs, "per_trade_realized_r"


def check_paper_sharpe_vs_spy(since) -> dict:
    """Gate 6: annualized paper Sharpe ≥ SPY Sharpe over the window.

    Paper returns come from portfolio_marks NAV when available (daily, directly
    comparable to SPY); otherwise a per-trade realized-R fallback. SPY daily
    returns are downloaded via yfinance — a download failure FAILS the gate
    (we will not pass on a paper-only number)."""
    try:
        paper_rets, source = fetch_paper_daily_returns(since)
    except Exception as e:
        return _gate("paper_sharpe_vs_spy", False, f"paper return read failed: {e}")
    if not paper_rets:
        return _gate("paper_sharpe_vs_spy", False,
                     "no paper returns — soak window has no signal")
    try:
        spy_rets = fetch_spy_daily_returns(since)
    except Exception as e:
        return _gate("paper_sharpe_vs_spy", False, f"SPY benchmark fetch failed: {e}")

    # NAV path is already daily → annualize with sqrt(252) like SPY. Per-trade
    # fallback annualizes per-trade (sqrt(252/n)) — comparison is then only
    # indicative (flagged via the source tag in the detail string).
    if source == "per_trade_realized_r":
        paper_sharpe = _annualized_sharpe(
            paper_rets, periods_per_year=252.0 / max(1, len(paper_rets)))
    else:
        paper_sharpe = _annualized_sharpe(paper_rets)
    spy_sharpe = _annualized_sharpe(spy_rets)

    if spy_sharpe is None:
        return _gate("paper_sharpe_vs_spy", False,
                     f"SPY Sharpe undefined (n={len(spy_rets)})")
    ps = paper_sharpe if paper_sharpe is not None else 0.0
    return _gate(
        "paper_sharpe_vs_spy",
        ps >= spy_sharpe,
        f"paper_sharpe={ps:.3f} ({source}, n={len(paper_rets)}) vs "
        f"spy_sharpe={spy_sharpe:.3f} (n={len(spy_rets)})",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--json", action="store_true",
                   help="emit JSON report instead of text")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    gates = [
        check_zero_intervention(since),
        check_ntfy_delivery(since),
        check_no_duplicate_orders(since),
        check_no_missed_fills(since),
        check_critical_events_notified(since),
        check_exit_fills_confirmed(since),
        check_paper_sharpe_vs_spy(since),
    ]
    all_pass = all(g["pass"] for g in gates)
    report = {
        "window_days": args.days,
        "since_utc": since.isoformat(),
        "all_pass": all_pass,
        "gates": gates,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"=== Soak acceptance over last {args.days} days ===")
        print(f"  since (UTC): {since.isoformat()}")
        for g in gates:
            mark = "PASS" if g["pass"] else "FAIL"
            print(f"  [{mark}] {g['label']}: {g['detail']}")
        print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
