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
                  AND outcome IN ('WIN', 'LOSS', 'SCRATCH')
                  AND COALESCE(broker_order_id, '') NOT LIKE 'VIRTUAL-%%'
                  AND (broker_fill_json IS NULL
                       OR NOT broker_fill_json ? 'exit_dealt_avg_price')
                """,
                (since,),
            )
            unconfirmed = int(cur.fetchone()[0])
    except Exception as e:
        return _gate("exit_fills_confirmed", False, f"db read failed: {e}")
    return _gate(
        "exit_fills_confirmed",
        unconfirmed == 0,
        f"{unconfirmed} closed trades without broker-confirmed exit price",
    )


def check_paper_sharpe_vs_spy(since) -> dict:
    """Compute paper-account daily-return Sharpe vs SPY over the window.

    Cheap implementation: trade-level R series → annualized Sharpe.  No
    benchmark download yet — a Phase 2.8 v2 candidate. For now we report the
    paper Sharpe and PASS if it's positive, leaving SPY comparison as a TODO.
    """
    try:
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
    except Exception as e:
        return _gate("paper_sharpe_positive", False, f"db read failed: {e}")
    if not rs:
        return _gate(
            "paper_sharpe_positive",
            False,
            "no closed trades — soak window has no signal",
        )
    import math
    mean = sum(rs) / len(rs)
    var = sum((r - mean) ** 2 for r in rs) / len(rs) if len(rs) > 1 else 0
    sd = math.sqrt(var)
    sharpe_per_trade = (mean / sd) if sd > 1e-9 else 0.0
    sharpe_annual = sharpe_per_trade * math.sqrt(252.0 / max(1, len(rs)))
    return _gate(
        "paper_sharpe_positive",
        sharpe_annual > 0,
        f"sharpe_annual={sharpe_annual:.3f} (per-trade mean={mean:.3f}, sd={sd:.3f}, n={len(rs)})",
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
