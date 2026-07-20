"""Week 1 Step 7 acceptance simulation — deadman halt + daily -2R breaker.

Scripted end-to-end demonstration against a TEMP sqlite store (never the
real data/trader.db, never real ntfy, never the real halt flag):

  1. -2R circuit breaker trip: seed today's journal with closed losing
     trades summing to -2.1R, then drive the REAL order-guard entry point
     (evaluate_order). An opening BUY must be refused with rule
     R8_daily_loss_breaker; a SELL-to-close of an open position must pass.

  2. 48h deadman auto-halt: feed the REAL escalation ladder a
     last-dispatch timestamp 30 days old (>> 48 market-hours through the
     real market calendar) and verify it creates the SAME halt flag the
     /halt endpoint writes, emits deadman_auto_halt {silent_hours}, and
     never touches open positions. Un-halt stays manual — the script
     verifies the flag survives a repeat tick.

Exit 0 = both behaviors demonstrated. Exit != 0 = something regressed.

Run (from the repo root):
    .venv/bin/python scripts/smoke_deadman_breaker.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Temp store MUST be wired up before any trading_agent import reads config.
_TMP = Path(tempfile.mkdtemp(prefix="smoke_deadman_"))
os.environ["DB_PATH"] = str(_TMP / "trader_smoke.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EQUITY = 100_000.0
R_UNIT = None  # filled in main() from the real sizing constant

_failures: list[str] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def _seed_breaker_db() -> None:
    from trading_agent.db import connection, migrate
    migrate(Path(os.environ["DB_PATH"]))
    now = datetime.now(timezone.utc).isoformat()
    with connection() as conn:
        # Fresh thesis so the open reaches the sizing/breaker rules.
        conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, 'AAPL', 'LONG', 't', 'i', 'open')",
            (now,),
        )
        # Today's realized losses: -1.0R and -1.1R (agent provenance).
        for pnl in (-1.0 * R_UNIT, -1.1 * R_UNIT):
            conn.execute(
                "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
                "opened_at, closed_at, outcome, pnl, is_paper, provenance) "
                "VALUES ('US.XYZ', 'STK', 'BUY', 10, 100.0, ?, ?, 'LOSS', ?, 1, 'agent')",
                (now, now, pnl),
            )
        # An open position we will try to close while the breaker is tripped.
        conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) "
            "VALUES ('US.AAPL', 'STK', 'BUY', 10, 100.0, ?, 'OPEN', 1)",
            (now,),
        )


def simulate_breaker() -> None:
    print(f"\n== 1. Daily -2R circuit breaker (R = ${R_UNIT:,.0f}, "
          f"seeded losses = -2.1R) ==")
    from trading_agent import order_guard as og

    # Postgres is out of scope for this smoke: pin it reachable-and-flat so
    # the demonstration runs purely against the temp sqlite store.
    og._pg_todays_losses = lambda: 0.0

    _seed_breaker_db()

    d_open = og.evaluate_order("stock", {
        "symbol": "US.AAPL", "side": "BUY", "qty": 10,
        "price": 100.0, "stop": 95.0, "target": 110.0,
    }, equity=EQUITY, cash=EQUITY)
    _check("opening BUY refused", not d_open.allowed, d_open.reason)
    _check("refusal rule is R8_daily_loss_breaker",
           any(v.rule == og.R8 for v in d_open.violations))

    d_close = og.evaluate_order("stock", {
        "symbol": "US.AAPL", "side": "SELL", "qty": 10, "price": 90.0,
    }, equity=EQUITY, cash=EQUITY)
    _check("SELL-to-close still allowed", d_close.allowed,
           f"intent={d_close.intent}")


def simulate_deadman_halt() -> None:
    print("\n== 2. 48h deadman auto-halt (last dispatch 30 days ago) ==")
    from unittest.mock import patch

    from trading_agent import halt_endpoint
    from trading_agent.graph.nodes import health_nodes as hn

    flag = _TMP / "halt.flag"
    emitted: list[dict] = []
    notified: list[tuple] = []

    last_ts = datetime.now(timezone.utc) - timedelta(days=30)
    with patch.object(halt_endpoint, "HALT_FLAG", flag), \
         patch.object(hn, "emit", lambda **kw: emitted.append(kw)), \
         patch.object(hn, "_was_deadman_push_recent", lambda: False), \
         patch("trading_agent.notify.send",
               lambda *a, **kw: notified.append((a, kw)) or {"ok": True}):
        # Real market calendar: 30 wall-days >> 48 market-hours of silence.
        hn._deadman_escalation("smoke_run", "healthcheck", last_ts)
        halts = [e for e in emitted if e["event_type"] == "deadman_auto_halt"]
        _check("halt flag created (the /halt endpoint's own file)",
               flag.exists(), str(flag))
        _check("deadman_auto_halt emitted once", len(halts) == 1)
        _check("payload carries silent_hours",
               bool(halts) and halts[0]["payload"].get("silent_hours", 0) > 48,
               f"payload={halts[0]['payload'] if halts else None}")
        _check("priority-5 ntfy pushes captured (never sent for real)",
               len(notified) >= 1,
               f"{len(notified)} push(es), priority="
               f"{[kw.get('priority') for _, kw in notified]}")

        # Repeat tick: no re-halt, and the flag is NOT removed (un-halt is
        # manual via POST /resume only).
        hn._deadman_escalation("smoke_run", "healthcheck", last_ts)
        halts2 = [e for e in emitted if e["event_type"] == "deadman_auto_halt"]
        _check("second tick does not re-halt", len(halts2) == 1)
        _check("flag survives (un-halt is manual)", flag.exists())

    # Positions untouched: the OPEN row seeded in part 1 is still OPEN.
    from trading_agent.db import connection
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol='US.AAPL' "
            "AND outcome='OPEN'").fetchone()
    _check("no position was auto-closed", row[0] == 1)


def main() -> int:
    global R_UNIT
    from trading_agent.sizing import MAX_SINGLE_RISK_PCT
    R_UNIT = MAX_SINGLE_RISK_PCT * EQUITY

    print(f"temp store: {os.environ['DB_PATH']}")
    simulate_breaker()
    simulate_deadman_halt()

    if _failures:
        print(f"\nSMOKE FAILED — {len(_failures)} check(s): {_failures}")
        return 1
    print("\nSMOKE PASSED — -2R breaker trips, 48h deadman halts, closes flow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
