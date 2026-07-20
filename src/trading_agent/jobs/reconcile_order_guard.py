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

Second check — position reconciliation (added after the 2026-07-08 audit):
broker paper positions vs journal OPEN rows across both stores (local
sqlite + Postgres journal_trades when reachable). Catches both directions
of drift we've been bitten by:

  position_without_journal_row — broker holds it, journal thinks it's flat
      (the 6/2 NVDA phantom exit: close journaled at order placement, the
      exit order never filled, 112 shares sat unmanaged for 5 weeks)
  journal_row_without_position — journal OPEN, broker holds nothing
      (the MRVL/AAPL spreads ids 9-12: expired or vanished with no close)
  position_qty_mismatch        — both exist, net quantities disagree

The positions check is skipped (with a summary flag, not an alert) when
OpenD is offline, mirroring mirror_real_fills' no-op-when-offline rule.

Findings notify ntfy topic ``risk`` at priority 5 and the process exits 1
(0 when clean) so manual runs and CI fail loudly.

Usage:
    python -m trading_agent.jobs.reconcile_order_guard [--since-hours 26]
        [--match-window-min 45] [--no-notify] [--skip-positions]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
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


def _load_guard_rows(cutoff: datetime) -> list[GuardRow] | None:
    """None means the audit table itself is unreadable (e.g. schema never
    migrated on this host) — that's a finding, not a silent zero."""
    out: list[GuardRow] = []
    try:
        with connection() as conn:
            rows = conn.execute(
                "SELECT created_at, hook_name, decision, payload FROM hook_audit_log "
                "WHERE hook_name IN (?, ?)",
                GUARD_NAMES,
            ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[{JOB}] hook_audit_log unreadable: {e}", file=sys.stderr)
        return None
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


# ---------------------------------------------------------------------------
# Position reconciliation — broker positions vs journal OPEN rows
# ---------------------------------------------------------------------------

QTY_TOLERANCE = 1e-6


def _load_broker_positions() -> dict[str, float] | None:
    """Paper-account positions as {symbol: signed_qty} (SHORT → negative).
    None when the broker is unreachable (OpenD offline) — the check is then
    skipped with a summary flag, mirroring mirror_real_fills' no-op rule."""
    try:
        from trading_agent.mcp_servers.moomoo.server import get_positions
        rows = get_positions().get("rows") or []
    except Exception as e:
        print(f"[{JOB}] broker positions unavailable: {e}", file=sys.stderr)
        return None
    out: dict[str, float] = {}
    for r in rows:
        code = str(r.get("code") or "")
        try:
            qty = float(r.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if not code or abs(qty) < QTY_TOLERANCE:
            continue
        if str(r.get("position_side") or "LONG").upper().startswith("SHORT"):
            qty = -abs(qty)
        out[code] = out.get(code, 0.0) + qty
    return out


def _load_open_rows_sqlite() -> list[dict]:
    """Journal rows still OPEN in the local sqlite store (mirrors excluded)."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, symbol, side, qty, broker_order_id, strategy_label "
            "FROM trades WHERE outcome = 'OPEN'"
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        boid = r["broker_order_id"] or ""
        if boid.startswith(MIRROR_PREFIX) or r["strategy_label"] == MIRROR_LABEL:
            continue
        out.append({
            "store": "sqlite", "trade_id": int(r["id"]), "symbol": str(r["symbol"]),
            "side": str(r["side"] or "BUY").upper(), "qty": float(r["qty"] or 0.0),
        })
    return out


def _combo_snapshots_sqlite(trade_ids: list[int]) -> dict[int, dict]:
    """Latest combo market_snapshot payload per sqlite trade_id.

    Thin delegation to the canonical detector (trading_agent.combo) — this
    job used to keep a private copy of the query; the module is now the ONE
    place the detection rule lives. Best-effort semantics unchanged:
    unreadable payloads are skipped — the row then reconciles as a plain
    single-leg and the long leg surfaces as a finding, which is the safe
    direction (alert, never mask)."""
    from trading_agent.combo import sqlite_combo_payloads
    return sqlite_combo_payloads(trade_ids)


def expand_combo_rows(rows: list[dict]) -> list[dict]:
    """Expand combo journal rows into their broker-visible legs.

    The broker holds TWO positions per vertical (short leg + protective long
    leg) while the journal holds one SELL row on the short leg's symbol.
    Without expansion the long leg false-alarms as position_without_journal_row
    every night. Appends a synthetic BUY row for each combo's long leg, tagged
    with the same trade_id/store so findings still point at the real row.

    Handles BOTH stores: sqlite rows resolve their combo payload from
    market_snapshots; Postgres rows carry it inline (the ``combo`` key,
    selected from broker_fill_json->'combo' by _load_open_rows_pg)."""
    sqlite_ids = [r["trade_id"] for r in rows if r["store"] == "sqlite"]
    snapshots = _combo_snapshots_sqlite(sqlite_ids)
    extra: list[dict] = []
    for r in rows:
        if r["store"] == "sqlite":
            snap = snapshots.get(r["trade_id"])
        else:
            snap = r.get("combo")
        if not isinstance(snap, dict) or not snap.get("combo"):
            continue
        long_sym = str(snap.get("long_leg") or "")
        if not long_sym:
            continue
        try:
            contracts = float(snap.get("contracts") or r["qty"])
        except (TypeError, ValueError):
            contracts = r["qty"]
        extra.append({
            "store": r["store"], "trade_id": r["trade_id"],
            "symbol": long_sym, "side": "BUY", "qty": contracts,
        })
    return rows + extra


def _load_open_rows_pg() -> list[dict] | None:
    """Open rows from the Postgres journal (the EC2 brain's store). None when
    Postgres isn't configured/reachable from this host — the sqlite-only
    result is still useful, so this degrades rather than fails."""
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT id, symbol, side, qty, broker_order_id, "
                "broker_fill_json->'combo' "
                "FROM journal_trades WHERE outcome = 'OPEN'"
            )
            rows = cur.fetchall()
    except Exception as e:
        print(f"[{JOB}] postgres journal unavailable: {e}", file=sys.stderr)
        return None
    out: list[dict] = []
    for tid, symbol, side, qty, boid, combo_payload in rows:
        if str(boid or "").startswith(MIRROR_PREFIX):
            continue
        out.append({
            "store": "postgres", "trade_id": int(tid), "symbol": str(symbol),
            "side": str(side or "BUY").upper(), "qty": float(qty or 0.0),
            "combo": combo_payload if isinstance(combo_payload, dict) else None,
        })
    return out


def compare_positions(
    broker: dict[str, float],
    journal_rows: list[dict],
    *,
    all_stores_visible: bool = True,
) -> list[dict]:
    """Pure classification of broker {symbol: signed_qty} vs open journal rows.

    Journal net per symbol = Σ(+qty for BUY-to-open, −qty for SELL-to-open)
    across all stores (the sqlite and Postgres journals hold disjoint trades).

    ``all_stores_visible=False`` (e.g. the Mac, which can't reach the EC2
    Postgres) suppresses the broker-side findings — a broker position absent
    from the *visible* journals may be journaled in the store we can't see,
    so flagging it would false-alarm nightly for every brain-held position.
    journal_row_without_position stays valid either way: the broker view is
    always complete, so a visible OPEN row with no position is a true finding."""
    net: dict[str, float] = {}
    rows_by_symbol: dict[str, list[dict]] = {}
    for row in journal_rows:
        sign = 1.0 if row["side"] == "BUY" else -1.0
        net[row["symbol"]] = net.get(row["symbol"], 0.0) + sign * row["qty"]
        rows_by_symbol.setdefault(row["symbol"], []).append(row)

    findings: list[dict] = []
    if all_stores_visible:
        for symbol, b_qty in sorted(broker.items()):
            j_qty = net.get(symbol, 0.0)
            if abs(j_qty) < QTY_TOLERANCE:
                findings.append({
                    "status": "position_without_journal_row",
                    "symbol": symbol, "broker_qty": b_qty, "journal_qty": 0.0,
                    "trade_ids": [], "stores": [],
                })
            elif abs(j_qty - b_qty) > QTY_TOLERANCE:
                findings.append({
                    "status": "position_qty_mismatch",
                    "symbol": symbol, "broker_qty": b_qty, "journal_qty": j_qty,
                    "trade_ids": [r["trade_id"] for r in rows_by_symbol[symbol]],
                    "stores": sorted({r["store"] for r in rows_by_symbol[symbol]}),
                })
    for symbol, j_qty in sorted(net.items()):
        if symbol in broker or abs(j_qty) < QTY_TOLERANCE:
            continue
        findings.append({
            "status": "journal_row_without_position",
            "symbol": symbol, "broker_qty": 0.0, "journal_qty": j_qty,
            "trade_ids": [r["trade_id"] for r in rows_by_symbol[symbol]],
            "stores": sorted({r["store"] for r in rows_by_symbol[symbol]}),
        })
    return findings


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
    ap.add_argument("--skip-positions", action="store_true",
                    help="skip the broker-vs-journal position reconciliation")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.since_hours)
    window = timedelta(minutes=args.match_window_min)

    fills = _load_fills(cutoff)
    # Guard rows can predate the oldest fill by the match window.
    guard_rows = _load_guard_rows(cutoff - window)

    findings: list[dict] = []
    guarded = 0
    if guard_rows is None:
        # Missing/unreadable audit table: every fill below will classify as
        # unguarded (truthful — nothing audited them), but name the root
        # cause first so the alert isn't a head-scratcher.
        findings.append({"status": "audit_log_unreadable",
                         "detail": "hook_audit_log missing or unreadable — "
                                   "run trading_agent.db.migrate on this host"})
        guard_rows = []
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

    # ---- position reconciliation ----
    positions_checked = False
    position_stores: list[str] = []
    position_findings: list[dict] = []
    if not args.skip_positions:
        try:
            broker = _load_broker_positions()
            if broker is not None:
                positions_checked = True
                journal_rows = _load_open_rows_sqlite()
                position_stores.append("sqlite")
                pg_rows = _load_open_rows_pg()
                if pg_rows is not None:
                    journal_rows.extend(pg_rows)
                    position_stores.append("postgres")
                # Expand AFTER both stores load: Postgres-journaled verticals
                # need their long wing appended too, or it false-alarms
                # position_without_journal_row nightly.
                journal_rows = expand_combo_rows(journal_rows)
                position_findings = compare_positions(
                    broker, journal_rows,
                    all_stores_visible=pg_rows is not None,
                )
        finally:
            # The moomoo SDK context spawns non-daemon threads that hang the
            # interpreter at exit (the same failure orchestrator's
            # _shutdown_resources handles) — close it as soon as we're done.
            try:
                from trading_agent.mcp_servers.moomoo.server import (
                    shutdown as _mm_shutdown,
                )
                _mm_shutdown()
            except Exception:
                pass

    all_findings = findings + position_findings
    summary = {
        "ts": now.isoformat(),
        "job": JOB,
        "since_hours": args.since_hours,
        "fills_checked": len(fills),
        "guarded": guarded,
        "positions_checked": positions_checked,
        "position_stores": position_stores,
        "findings": findings,
        "position_findings": position_findings,
    }
    print(json.dumps(summary, indent=2, default=str))

    if all_findings and not args.no_notify:
        lines = []
        for f in findings[:10]:
            if f["status"] == "audit_log_unreadable":
                lines.append(f"AUDIT_LOG_UNREADABLE: {f['detail']}")
            else:
                lines.append(
                    f"{f['status'].upper()}: {f['side']} {f['qty']:g} {f['symbol']} "
                    f"@ {f['entry_price']:g} (trade #{f['trade_id']})"
                )
        for f in position_findings[:10]:
            ids = ",".join(str(i) for i in f["trade_ids"]) or "-"
            lines.append(
                f"{f['status'].upper()}: {f['symbol']} broker={f['broker_qty']:g} "
                f"journal={f['journal_qty']:g} (trades: {ids})"
            )
        hidden = len(all_findings) - len(lines)
        if hidden > 0:
            lines.append(f"… and {hidden} more")
        notify.send(
            "risk",
            title=f"🚨 {len(all_findings)} reconciliation finding(s): guard/positions",
            body="\n".join(lines)
            + "\n\nGuard findings = fills with no guard-evaluation row in "
              "hook_audit_log. Position findings = broker positions vs "
              "journal OPEN rows disagree. Investigate before the next session.",
            priority=5,
            tags=["rotating_light"],
        )

    return 1 if all_findings else 0


if __name__ == "__main__":
    rc = main()
    # Belt-and-braces vs any library non-daemon thread we don't know about:
    # flush, then os._exit so launchd/systemd oneshots never hang in
    # threading._shutdown() (observed 2026-07-08: the SDK context kept the
    # process alive 5+ min after the report printed; mirror_real_fills has
    # been stuck the same way for a day).
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(rc)
