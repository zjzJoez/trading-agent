"""Mirror the operator's live-account fills into the paper journal (shadow).

The operator trades the brokerage account by hand. This job reads those filled
orders (READ-ONLY, via the existing moomoo read tool) and records each one as a
shadow thesis + paper fill in the journal — ``is_paper=1``,
``strategy_label='shadow_real_account'`` — so the post-mortem / learning loop can
reason about the operator's manual trades. It never places a live order.

Idempotent: every mirrored fill is stored with ``broker_order_id``
``REALMIRROR-<order_id>``; re-runs skip orders already mirrored. The shadow
thesis is immediately marked ``triggered`` so it never shows up in
``list_open_theses`` (and therefore can't accidentally satisfy the PreToolUse
order gate for a same-ticker live paper order).

Scope (v1): one journal row per filled order (a 1:1 fill log). It does NOT yet
net buys/sells into round-trips or auto-close on the offsetting fill — that
position-lifecycle matching is a follow-up. For now it gives the journal a
faithful, deduped record of the operator's real-account activity, cleanly
separable by ``strategy_label='shadow_real_account'``.

Usage:
    python -m trading_agent.jobs.mirror_real_fills [--since YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone

from trading_agent.db import connection
from trading_agent.mcp_servers.journal.server import (
    close_thesis,
    record_fill,
    record_thesis,
)

log = logging.getLogger(__name__)

MIRROR_LABEL = "shadow_real_account"
MIRROR_PREFIX = "REALMIRROR-"

# Option symbol after the market prefix: <UNDERLYING><YYMMDD><C|P><strike*1000>.
# Moomoo does NOT zero-pad the strike to 8 digits (e.g. MRVL260702C300000 is
# strike 300), so the strike group is variable-length.
_OPT_RE = re.compile(r"^([A-Z]+)\d{6}([CP])\d+$")
_LEADING_ALPHA = re.compile(r"^([A-Z]+)")


def _strip_market(code: str) -> str:
    return code.split(".", 1)[-1] if "." in code else code  # US.NVDA -> NVDA


def _classify(code: str, side: str) -> tuple[str, str, str]:
    """Return (asset_type, underlying, direction) for a broker symbol + side."""
    sym = _strip_market(code)
    m = _OPT_RE.match(sym)
    if m:
        underlying, cp = m.group(1), m.group(2)
        if side == "BUY":
            direction = "LONG_CALL" if cp == "C" else "LONG_PUT"
        else:
            direction = "SHORT_CALL" if cp == "C" else "SHORT_PUT"
        return "OPT", underlying, direction
    lead = _LEADING_ALPHA.match(sym)
    underlying = lead.group(1) if lead else sym
    direction = "LONG" if side == "BUY" else "SHORT"
    return "STK", underlying, direction


def _already_mirrored() -> set[str]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT broker_order_id FROM trades WHERE broker_order_id LIKE ?",
            (MIRROR_PREFIX + "%",),
        ).fetchall()
    return {r[0][len(MIRROR_PREFIX):] for r in rows if r[0]}


def _fetch_orders(since: str | None, acc_index: int) -> list[dict]:
    """Read filled real-account orders. Returns [] if the broker is unreachable
    (OpenD down) so a scheduled run is a graceful no-op, not a failure."""
    try:
        # Lazy import: keeps this module import-light and broker-SDK-free until a
        # real run actually needs the live read.
        from trading_agent.mcp_servers.moomoo.server import get_real_history_orders
        resp = get_real_history_orders(start=since, acc_index=acc_index)
    except Exception as e:  # noqa: BLE001 — broker/OpenD may be offline
        log.warning("[mirror_real_fills] broker read failed (OpenD down?): %s", e)
        return []
    return resp.get("rows", []) if isinstance(resp, dict) else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mirror real-account fills into the paper journal")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD; default = broker default window")
    ap.add_argument("--acc-index", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report what would be mirrored, write nothing")
    args = ap.parse_args(argv)

    rows = _fetch_orders(args.since, args.acc_index)
    seen = _already_mirrored()

    mirrored: list[dict] = []
    skipped = 0
    ignored = 0
    for o in rows:
        if o.get("order_status") != "FILLED_ALL":
            ignored += 1
            continue
        oid = str(o.get("order_id") or "")
        qty = float(o.get("dealt_qty") or 0)
        price = float(o.get("dealt_avg_price") or 0)
        if not oid or qty <= 0 or price <= 0:
            ignored += 1
            continue
        if oid in seen:
            skipped += 1
            continue

        code = o.get("code", "")
        side = "SELL" if str(o.get("trd_side", "BUY")).upper() == "SELL" else "BUY"
        asset_type, underlying, direction = _classify(code, side)
        when = o.get("create_time", "")

        rec = {
            "order_id": oid, "code": code, "side": side, "qty": qty,
            "price": price, "asset_type": asset_type, "underlying": underlying,
        }
        if args.dry_run:
            mirrored.append(rec)
            continue

        th = record_thesis(
            ticker=underlying,
            direction=direction,
            thesis_text=(
                f"Shadow mirror of real-account {side} {qty:g} {code} @ {price:g} "
                f"on {when} (real_order_id={oid}). Operator manual trade — "
                "recorded for post-mortem visibility only, not a live thesis."
            ),
            invalidation="n/a — shadow mirror of an operator manual trade",
            timeframe="shadow",
        )
        thesis_id = th["thesis_id"]
        fill = record_fill(
            broker_order_id=f"{MIRROR_PREFIX}{oid}",
            symbol=code,
            asset_type=asset_type,
            side=side,
            qty=qty,
            fill_price=price,
            thesis_id=thesis_id,
            strategy_label=MIRROR_LABEL,
            reasoning=f"real-account mirror; real_order_id={oid}; filled {when}",
        )
        # Keep the shadow thesis out of the 'open' set (and the order gate).
        close_thesis(thesis_id=thesis_id, status="triggered")
        rec["thesis_id"] = thesis_id
        rec["trade_id"] = fill["trade_id"]
        mirrored.append(rec)
        seen.add(oid)

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job": "mirror_real_fills",
        "dry_run": args.dry_run,
        "fetched": len(rows),
        "mirrored": len(mirrored),
        "skipped_already_mirrored": skipped,
        "ignored_unfilled_or_bad": ignored,
        "details": mirrored,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
