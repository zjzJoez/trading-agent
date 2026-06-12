"""Nightly reconciliation: every opening fill must have a guard evaluation.

The risk gate (trading_agent.order_guard) runs inside the moomoo MCP order
tools and in the Claude Code PreToolUse hook, and every evaluation writes a
row to the ``hook_audit_log`` table. This job is the tripwire for the gap
that bit us on 2026-06-02/08 (MRVL 2x + AAPL 8x spreads entered with zero
guard evaluation): it joins recent ``trades`` rows against guard-evaluation
audit rows and alerts on any fill that no layer evaluated.

Classification per fill:
  guarded            — an 'allow' row for the same symbol within the match
                       window before the fill. Healthy.
  blocked_but_filled — only 'block' rows found. WORSE than unguarded: a
                       guard said no and a fill happened anyway (override or
                       direct journal write).
  unguarded          — no guard row at all. The fill bypassed every layer
                       (direct record_fill / record_virtual_fill call, or a
                       code path that skipped the gate).

Exemptions: shadow mirrors of the operator's real-account trades
(``broker_order_id LIKE 'REALMIRROR-%'`` / ``strategy_label =
'shadow_real_account'``) are deliberately journal-only and never pass the
gate.

Findings notify ntfy topic ``risk`` at priority 5 and the process exits 1
(0 when clean) so manual runs and CI fail loudly.

Usage:
    python -m trading_agent.jobs.reconcile_order_guard [--since-hours 26]
        [--match-window-min 45] [--no-notify]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from trading_agent import notify
from trading_agent.db import connection

JOB = "reconcile_order_guard"
GUARD_NAMES = ("server_order_guard", "pretool_order_guard")
MIRROR_PREFIX = "REALMIRROR-"
MIRROR_LABEL = "shadow_real_account"

# A guard evaluation may precede its journal row by minutes (e.g. the
# virtual-fill path: guard runs in place_paper_option_order, the broker
# rejects, the skill calls record_virtual_fill afterwards). It can also
# land slightly AFTER the trades row on clock skew, hence the small
# forward tolerance.
FORWARD_TOLERANCE = timedelta(minutes=5)


def _parse_ts(value: str | None) -> datetime | None:
    """ISO-8601 → aware datetime (naive treated as UTC). None on garbage."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class GuardRow:
    symbol: str
    decision: str          # allow/block
    hook_name: str
    created_at: datetime


def _load_guard_rows(cutoff: datetime) -> list[GuardRow]:
    out: list[GuardRow] = []
    with connection() as conn:
        rows = conn.execute(
            "SELECT created_at, hook_name, decision, payload FROM hook_audit_log "
            "WHERE hook_name IN (?, ?)",
            GUARD_NAMES,
        ).fetchall()
    for r in rows:
        ts = _parse_ts(r["created_at"])
        if ts is None or ts < cutoff:
            continue
        try:
            payload = json.loads(r["payload"] or "{}")
        except ValueError:
            payload = {}
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            continue
        out.append(GuardRow(
            symbol=symbol,
            decision=str(r["decision"]),
            hook_name=str(r["hook_name"]),
            created_at=ts,
        ))
    return out


def _load_fills(cutoff: datetime) -> list[dict]:
    """trades rows opened since `cutoff`, minus the real-account shadow
    mirrors. Timestamps are parsed in Python — trades.opened_at is ISO-8601
    with offsets, which SQLite string compares mishandle."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, symbol, side, qty, entry_price, opened_at,
                   broker_order_id, strategy_label
            FROM trades
            WHERE opened_at IS NOT NULL
            """
        ).fetchall()
    fills: list[dict] = []
    for r in rows:
        opened = _parse_ts(r["opened_at"])
        if opened is None or opened < cutoff:
            continue
        boid = r["broker_order_id"] or ""
        if boid.startswith(MIRROR_PREFIX) or r["strategy_label"] == MIRROR_LABEL:
            continue
        fills.append({
            "trade_id": r["id"],
            "symbol": r["symbol"],
            "side": r["side"],
            "qty": r["qty"],
            "entry_price": r["entry_price"],
            "opened_at": opened,
            "broker_order_id": boid,
            "strategy_label": r["strategy_label"],
        })
    return fills


def _classify(fill: dict, guard_rows: list[GuardRow], window: timedelta) -> str:
    lo = fill["opened_at"] - window
    hi = fill["opened_at"] + FORWARD_TOLERANCE
    matched = [
        g for g in guard_rows
        if g.symbol == fill["symbol"] and lo <= g.created_at <= hi
    ]
    if any(g.decision == "allow" for g in matched):
        return "guarded"
    if matched:
        return "blocked_but_filled"
    return "unguarded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Alert on opening fills with no guard-evaluation audit row"
    )
    ap.add_argument("--since-hours", type=float, default=26.0,
                    help="how far back to scan trades (default 26h — nightly run + slack)")
    ap.add_argument("--match-window-min", type=float, default=45.0,
                    help="guard row may precede the fill by up to this many minutes")
    ap.add_argument("--no-notify", action="store_true",
                    help="skip the ntfy alert (report + exit code only)")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.since_hours)
    window = timedelta(minutes=args.match_window_min)

    fills = _load_fills(cutoff)
    # Guard rows can predate the oldest fill by the match window.
    guard_rows = _load_guard_rows(cutoff - window)

    findings: list[dict] = []
    guarded = 0
    for fill in fills:
        status = _classify(fill, guard_rows, window)
        if status == "guarded":
            guarded += 1
            continue
        findings.append({
            "status": status,
            "trade_id": fill["trade_id"],
            "symbol": fill["symbol"],
            "side": fill["side"],
            "qty": fill["qty"],
            "entry_price": fill["entry_price"],
            "opened_at": fill["opened_at"].isoformat(),
            "broker_order_id": fill["broker_order_id"],
            "strategy_label": fill["strategy_label"],
        })

    summary = {
        "ts": now.isoformat(),
        "job": JOB,
        "since_hours": args.since_hours,
        "fills_checked": len(fills),
        "guarded": guarded,
        "findings": findings,
    }
    print(json.dumps(summary, indent=2, default=str))

    if findings and not args.no_notify:
        lines = [
            f"{f['status'].upper()}: {f['side']} {f['qty']:g} {f['symbol']} "
            f"@ {f['entry_price']:g} (trade #{f['trade_id']})"
            for f in findings[:10]
        ]
        if len(findings) > 10:
            lines.append(f"… and {len(findings) - 10} more")
        notify.send(
            "risk",
            title=f"🚨 {len(findings)} fill(s) bypassed the order guard",
            body="\n".join(lines)
            + "\n\nNo guard-evaluation row in hook_audit_log. "
              "Investigate the entry path before the next session.",
            priority=5,
            tags=["rotating_light"],
        )

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
