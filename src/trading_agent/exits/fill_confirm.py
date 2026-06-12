"""Exit-fill confirmation — the pending-close state machine.

Phase 0 measurement fix: ``route_exit_or_hold`` used to place a close order
and journal the close IMMEDIATELY at the position's last mark — never
confirming the broker fill, never paying the spread you actually cross to
hit the bid. That systematically inflated every recorded exit by ~half the
quoted spread and could mark positions closed whose close orders never
filled. From this change on:

  place close order → record ``pending_close`` state on the journal row
  → next intraday tick, ``sync_fill_status`` calls ``finalize_pending_exits``
  → FILLED_ALL: journal the close at the broker's dealt price, net of fees
  → dead with a dealt portion (CANCELLED_PART): book the dealt leg, re-place
    the remainder next tick
  → dead with nothing dealt: re-place next tick (state retained, so the
    duplicate-order guard stays armed)
  → resting past the reprice window with ZERO fill: cancel + re-place at a
    fresh bid-based marketable limit, up to ``MAX_REPRICE_ATTEMPTS``
  → vanished from the (today-only) broker order map: after
    ``MAX_MISSING_TICKS`` consecutive misses, reconcile against live
    positions — position gone → settle at the requested price flagged
    UNCONFIRMED (sev-2); position still held → re-place.

Core safety invariant: a partially-filled order is NEVER cancelled by this
module (only zero-fill orders are repriced). Dealt legs therefore only enter
the books from terminal order states, which makes double-counting and
over-close structurally impossible: every re-place is sized
``target − already_dealt``.

State lives next to the trade row in whichever store resolved it:
Postgres ``journal_trades.broker_fill_json['pending_close']`` or SQLite
``trades.exit_order_id`` + ``trades.exit_state_json``. Both survive process
death; the 15-minute tick cadence is the polling loop.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit

log = logging.getLogger(__name__)

PENDING_KEY = "pending_close"

# Re-place an unfilled exit at a fresh marketable limit after this long.
REPRICE_AFTER_SECONDS = 600
# After this many cancel+replace rounds, stop chasing and page the operator.
MAX_REPRICE_ATTEMPTS = 3
# Consecutive ticks an order may be absent from the broker order map before
# we reconcile against live positions (get_orders is today-only — an exit
# that settles after the day's last tick vanishes from tomorrow's map).
MAX_MISSING_TICKS = 2
# Marketable-limit buffers, mirroring the entry side's ask*1.02 convention.
SELL_CROSS_BUF = 0.98
BUY_CROSS_BUF = 1.02

_FILLED = {"FILLED_ALL"}
_DEAD = {"CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DELETED", "EXPIRED"}
# Live order statuses where a resting order may legitimately sit.
_PARTIAL = {"FILLED_PART"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pending_state(
    *,
    order_id: str,
    symbol: str,
    action: str,
    reason: str,
    qty: float,
    side: str,
    asset_type: str,
    requested_exit_price: float,
    quoted_bid: float | None,
    quoted_ask: float | None,
    thesis_id: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "order_id": str(order_id),
        "symbol": symbol,
        "action": action,
        "reason": reason[:280],
        "qty": float(qty),                 # total close target
        "dealt_legs": [],                  # [{qty, price}] from terminal states
        "side": side,
        "asset_type": asset_type,
        "requested_exit_price": float(requested_exit_price),
        "quoted_bid": quoted_bid,
        "quoted_ask": quoted_ask,
        "thesis_id": thesis_id,
        "source": source,
        "placed_at": _now_iso(),
        "attempts": 0,
        "missing_ticks": 0,
        "exhausted_alerted": False,
        "blocked_alerted": False,
    }


def _legs_qty(state: dict) -> float:
    return sum(float(leg.get("qty") or 0) for leg in (state.get("dealt_legs") or []))


def _remaining_qty(state: dict) -> float:
    return max(0.0, float(state.get("qty") or 0) - _legs_qty(state))


def _blended_exit(state: dict, final_qty: float, final_price: float) -> tuple[float, float]:
    """(total_dealt_qty, weighted_avg_price) across legs + the final fill."""
    legs = list(state.get("dealt_legs") or [])
    if final_qty > 0:
        legs.append({"qty": final_qty, "price": final_price})
    total = sum(float(x["qty"]) for x in legs)
    if total <= 0:
        return 0.0, float(final_price)
    avg = sum(float(x["qty"]) * float(x["price"]) for x in legs) / total
    return total, avg


# ---------------------------------------------------------------------------
# Pending-state storage
# ---------------------------------------------------------------------------

def write_pending_close_pg(trade_id: int, state: dict[str, Any]) -> bool:
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                UPDATE journal_trades
                SET broker_fill_json = COALESCE(broker_fill_json, '{}'::jsonb)
                      || jsonb_build_object(%s::text, %s::jsonb)
                WHERE id = %s AND outcome = 'OPEN'
                """,
                (PENDING_KEY, json.dumps(state, default=str), int(trade_id)),
            )
        return True
    except Exception as e:
        log.warning("write_pending_close_pg(%s) failed: %s", trade_id, e)
        return False


def write_pending_close_sqlite(trade_id: int, state: dict[str, Any]) -> bool:
    try:
        from trading_agent.db import connection
        with connection() as conn:
            conn.execute(
                "UPDATE trades SET exit_order_id = ?, exit_state_json = ? "
                "WHERE id = ? AND outcome = 'OPEN'",
                # order_id may be None between a dead order and its
                # replacement — keep the guard armed via a sentinel.
                (state.get("order_id") or "PENDING-REPLACE",
                 json.dumps(state, default=str), int(trade_id)),
            )
        return True
    except Exception as e:
        log.warning("write_pending_close_sqlite(%s) failed: %s", trade_id, e)
        return False


def write_pending_close(trade_id: int, state: dict[str, Any]) -> bool:
    if state.get("source") == "sqlite":
        return write_pending_close_sqlite(trade_id, state)
    return write_pending_close_pg(trade_id, state)


def pending_close_trade_ids_sqlite() -> set[int] | None:
    """SQLite trade ids with an in-flight close. None means the store could
    not be read — callers MUST fail closed (skip exits for that store), not
    treat it as 'no pending exits': forgetting an in-flight close and placing
    a second one over-closes the position."""
    try:
        from trading_agent.db import connection
        with connection() as conn:
            rows = conn.execute(
                "SELECT id FROM trades WHERE outcome = 'OPEN' "
                "AND exit_order_id IS NOT NULL",
            ).fetchall()
        return {int(r[0]) for r in rows}
    except Exception as e:
        log.warning("pending_close_trade_ids_sqlite failed: %s", e)
        return None


def pending_close_trade_ids_pg() -> set[int] | None:
    """Postgres twin of pending_close_trade_ids_sqlite. None on read error."""
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT id FROM journal_trades "
                "WHERE outcome = 'OPEN' AND broker_fill_json ? %s",
                (PENDING_KEY,),
            )
            return {int(r[0]) for r in cur.fetchall()}
    except Exception as e:
        log.warning("pending_close_trade_ids_pg failed: %s", e)
        return None


def _clear_pending_pg(trade_id: int) -> None:
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "UPDATE journal_trades SET broker_fill_json = "
                "broker_fill_json - %s WHERE id = %s",
                (PENDING_KEY, int(trade_id)),
            )
    except Exception as e:
        log.warning("_clear_pending_pg(%s) failed: %s", trade_id, e)


def _clear_pending_sqlite(trade_id: int) -> None:
    try:
        from trading_agent.db import connection
        with connection() as conn:
            conn.execute(
                "UPDATE trades SET exit_order_id = NULL, exit_state_json = NULL "
                "WHERE id = ?",
                (int(trade_id),),
            )
    except Exception as e:
        log.warning("_clear_pending_sqlite(%s) failed: %s", trade_id, e)


def _clear_pending(trade_id: int, state: dict) -> None:
    if state.get("source") == "sqlite":
        _clear_pending_sqlite(trade_id)
    else:
        _clear_pending_pg(trade_id)


def find_adoptable_close_order(
    order_map: dict[str, dict], symbol: str, side: str,
) -> tuple[str, float] | None:
    """A live order on this symbol+side that an earlier (crashed) run placed.

    route_exit_or_hold calls this BEFORE placing a close: a crash between
    placement and the pending-state write leaves an orphan resting order;
    adopting it instead of placing a second one prevents an over-close.
    Returns (order_id, qty) of the first live match, or None.
    """
    dead_or_done = _DEAD | _FILLED
    for oid, row in order_map.items():
        try:
            r_symbol = str(row.get("code") or row.get("symbol") or "")
            r_side = str(row.get("trd_side") or row.get("side") or "").upper()
            status = str(row.get("order_status")
                         or row.get("order_status_str") or "").upper()
            if (r_symbol == symbol and side.upper() in r_side
                    and status and status not in dead_or_done):
                return str(oid), float(row.get("qty") or 0)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Close-out helpers
# ---------------------------------------------------------------------------

def _outcome_for_pnl(pnl: float) -> str:
    # Mirrors learning.outcome semantics (WIN/LOSS/SCRATCH are the canonical
    # closed literals); kept local so this module has no learning dependency.
    # NOTE for merge: the optimistic-taussig branch adds the identical helper
    # to learning/outcome.py — consolidate onto that one once both land.
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "SCRATCH"


def _net_pnl_for_close(
    *,
    entry_price: float,
    dealt_exit_price: float,
    qty: float,
    asset_type: str,
    side_opened: str,
    entry_fees: float | None,
) -> tuple[float, float]:
    """(net_pnl, exit_fees). Both prices are dealt — only fees apply."""
    from trading_agent.execution_costs import fees_per_side
    mult = 100.0 if str(asset_type).upper() == "OPT" else 1.0
    if str(side_opened).upper() == "SELL":
        gross = (entry_price - dealt_exit_price) * qty * mult
    else:
        gross = (dealt_exit_price - entry_price) * qty * mult
    exit_fees = fees_per_side(qty, asset_type)
    if entry_fees is None:
        entry_fees = fees_per_side(qty, asset_type)
    return round(gross - float(entry_fees) - exit_fees, 2), exit_fees


def _settle(trade_id: int, state: dict, final_qty: float, final_price: float,
            *, unconfirmed: bool = False) -> tuple[bool, float]:
    """Close the journal row over the TOTAL dealt quantity (legs + final fill)
    at the blended exit price. Returns (ok, net_pnl)."""
    total_qty, avg_price = _blended_exit(state, final_qty, final_price)
    if total_qty <= 0:
        return False, 0.0
    if state.get("source") == "sqlite":
        return _close_sqlite(trade_id, state, avg_price, total_qty,
                             unconfirmed=unconfirmed)
    return _close_pg(trade_id, state, avg_price, total_qty,
                     unconfirmed=unconfirmed)


def _close_pg(trade_id: int, state: dict, dealt_price: float,
              dealt_qty: float, *, unconfirmed: bool = False) -> tuple[bool, float]:
    """Close a Postgres journal row at the dealt price. Returns (ok, net_pnl)."""
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT entry_price, qty, asset_type, side, thesis_id, "
                "broker_fill_json FROM journal_trades "
                "WHERE id = %s AND outcome = 'OPEN'",
                (int(trade_id),),
            )
            row = cur.fetchone()
        if row is None:
            # Already closed (double finalize) — clear any leftover state.
            _clear_pending_pg(trade_id)
            return False, 0.0
        entry_price, qty, asset_type, side, thesis_id, blob = row
        blob = blob if isinstance(blob, dict) else {}
        entry_fees = blob.get("entry_fees")
        net, exit_fees = _net_pnl_for_close(
            entry_price=float(entry_price or 0.0),
            dealt_exit_price=dealt_price,
            qty=float(dealt_qty or qty or 0.0),
            asset_type=str(asset_type or "OPT"),
            side_opened=str(side or "BUY"),
            entry_fees=float(entry_fees) if entry_fees is not None else None,
        )
        outcome = _outcome_for_pnl(net)
        close_reason = state.get("action") or "EXIT"
        if unconfirmed:
            close_reason += " [exit fill UNCONFIRMED — order vanished, position flat]"
        with cursor() as cur:
            cur.execute(
                """
                UPDATE journal_trades
                SET exit_price = %s, outcome = %s,
                    closed_at = COALESCE(closed_at, NOW()),
                    close_reason = COALESCE(close_reason, %s),
                    broker_fill_json = (broker_fill_json - %s)
                      || jsonb_build_object(
                           'exit_dealt_avg_price', CASE WHEN %s THEN NULL
                                                        ELSE %s::numeric END,
                           'exit_dealt_qty', %s::numeric,
                           'exit_fees', %s::numeric,
                           'net_pnl', %s::numeric,
                           'requested_exit_price', %s::numeric,
                           'exit_quoted_bid', %s::numeric,
                           'exit_quoted_ask', %s::numeric,
                           'exit_unconfirmed', %s,
                           'exit_fill_confirmed_at', NOW()::text)
                WHERE id = %s AND outcome = 'OPEN'
                """,
                (dealt_price, outcome, close_reason, PENDING_KEY,
                 unconfirmed, dealt_price, dealt_qty, exit_fees, net,
                 state.get("requested_exit_price"),
                 state.get("quoted_bid"), state.get("quoted_ask"),
                 unconfirmed, int(trade_id)),
            )
        # Thesis lifecycle — same store as the trade row.
        t_id = state.get("thesis_id") or thesis_id
        if t_id is not None:
            with cursor() as cur:
                cur.execute(
                    "UPDATE journal_theses SET status='triggered' "
                    "WHERE id = %s AND status='open'",
                    (int(t_id),),
                )
        return True, net
    except Exception as e:
        log.warning("_close_pg(%s) failed: %s", trade_id, e)
        return False, 0.0


def _close_sqlite(trade_id: int, state: dict, dealt_price: float,
                  dealt_qty: float, *, unconfirmed: bool = False) -> tuple[bool, float]:
    """Close a SQLite journal row at the dealt price via journal close_trade.

    The net P&L and exit fees are computed HERE over the actually-dealt
    quantity and passed explicitly — close_trade's own leg arithmetic uses
    the journal row's full qty, which is wrong whenever the broker dealt
    fewer units (prior scale rung, partial terminal)."""
    try:
        from trading_agent.db import connection
        from trading_agent.mcp_servers.journal.server import close_trade
        with connection() as conn:
            row = conn.execute(
                "SELECT entry_price, qty, asset_type, side, fees, outcome "
                "FROM trades WHERE id = ?",
                (int(trade_id),),
            ).fetchone()
        if row is None or row["outcome"] != "OPEN":
            _clear_pending_sqlite(trade_id)
            return False, 0.0
        net, exit_fees = _net_pnl_for_close(
            entry_price=float(row["entry_price"] or 0.0),
            dealt_exit_price=dealt_price,
            qty=float(dealt_qty or row["qty"] or 0.0),
            asset_type=str(row["asset_type"] or "OPT"),
            side_opened=str(row["side"] or "BUY"),
            entry_fees=float(row["fees"]) if row["fees"] is not None else None,
        )
        close_trade(
            trade_id=int(trade_id),
            exit_price=dealt_price,
            outcome=_outcome_for_pnl(net),
            pnl=net,
            exit_fees=exit_fees,
        )
        _clear_pending_sqlite(trade_id)
        return True, net
    except Exception as e:
        log.warning("_close_sqlite(%s) failed: %s", trade_id, e)
        return False, 0.0


# ---------------------------------------------------------------------------
# Placement / reprice
# ---------------------------------------------------------------------------

def _fresh_marketable_exit(symbol: str, side: str, fallback: float) -> tuple[float, float | None, float | None]:
    """(limit_price, quoted_bid, quoted_ask) off a fresh quote; fallback on error."""
    bid = ask = None
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
        for r in (get_quote([symbol]).get("rows") or []):
            b = r.get("bid_price") or r.get("bid")
            a = r.get("ask_price") or r.get("ask")
            bid = float(b) if b and float(b) > 0 else None
            ask = float(a) if a and float(a) > 0 else None
            break
    except Exception as e:
        log.warning("fresh exit quote %s failed: %s", symbol, e)
    if str(side).upper() == "SELL":
        base = bid if bid else fallback
        return round(base * SELL_CROSS_BUF, 2), bid, ask
    base = ask if ask else fallback
    return round(base * BUY_CROSS_BUF, 2), bid, ask


def _place_remainder(trade_id: int, state: dict, run_id: str, trigger: str) -> None:
    """Place a fresh close order for the not-yet-dealt remainder.

    Used when the previous order died (state.order_id is None) and after a
    zero-fill cancel in _reprice. Sized target − dealt legs, so a partial
    terminal can never lead to an over-close."""
    remaining = _remaining_qty(state)
    if remaining <= 0:
        # Everything already dealt via legs — settle with no final fill.
        ok, net = _settle(trade_id, state, 0.0, float(state["requested_exit_price"]))
        if ok:
            emit(run_id=run_id, trigger=trigger, agent="finalize_pending_exits",
                 event_type="position_closed",
                 payload={"trade_id": trade_id, "symbol": state["symbol"],
                          "net_pnl": net, "fill_confirmed": True,
                          "via": "legs_only"})
        return

    try:
        from trading_agent.mcp_servers.moomoo.server import (
            place_paper_option_order,
            place_paper_order,
        )
    except Exception as e:
        log.warning("_place_remainder: moomoo import failed: %s", e)
        return

    new_limit, bid, ask = _fresh_marketable_exit(
        state["symbol"], state["side"], float(state["requested_exit_price"]))
    try:
        if str(state.get("asset_type")) == "OPT":
            result = place_paper_option_order(
                option_symbol=state["symbol"], side=state["side"],
                contracts=int(remaining), price=new_limit,
                thesis_id=int(trade_id), strategy_label=None,
            )
        else:
            result = place_paper_order(
                symbol=state["symbol"], side=state["side"], qty=int(remaining),
                price=new_limit, thesis_id=int(trade_id), order_type="NORMAL",
            )
    except Exception as e:
        log.warning("_place_remainder %s failed: %s", state["symbol"], e)
        return

    # Forward-compat with the order-tool risk guard (confident-lovelace
    # branch): a refusal returns order_blocked=true with no rows. A guard
    # blocking a CLOSE is always a bug worth paging on — alert once, keep
    # the state so the close retries once the guard is fixed.
    if result.get("order_blocked"):
        if not state.get("blocked_alerted"):
            state["blocked_alerted"] = True
            emit(run_id=run_id, trigger=trigger, agent="finalize_pending_exits",
                 event_type="exit_order_blocked_by_guard", severity=2,
                 payload={"trade_id": trade_id, "symbol": state["symbol"],
                          "violations": result.get("violations")})
        write_pending_close(trade_id, state)
        return

    rows = result.get("rows") or []
    new_id = str(rows[0].get("order_id") or rows[0].get("orderID") or "").strip() if rows else ""
    if new_id:
        state.update({
            "order_id": new_id,
            "requested_exit_price": new_limit,
            "quoted_bid": bid,
            "quoted_ask": ask,
            "placed_at": _now_iso(),
            "missing_ticks": 0,
        })
        write_pending_close(trade_id, state)
        emit(run_id=run_id, trigger=trigger, agent="finalize_pending_exits",
             event_type="exit_order_replaced",
             payload={"trade_id": trade_id, "symbol": state["symbol"],
                      "new_order_id": new_id, "new_limit": new_limit,
                      "remaining_qty": remaining})
    else:
        # Keep the state (legs must not be lost); retry next tick.
        state["order_id"] = None
        write_pending_close(trade_id, state)
        emit(run_id=run_id, trigger=trigger, agent="finalize_pending_exits",
             event_type="exit_replace_failed", severity=2,
             payload={"trade_id": trade_id, "symbol": state["symbol"]})


def _reprice(trade_id: int, state: dict, order_row: dict, run_id: str,
             trigger: str) -> None:
    """Cancel a stale ZERO-FILL resting exit and re-place at a fresh limit.

    Safety invariant: an order with any dealt quantity is never cancelled —
    its legs enter the books only from its terminal state, so this function
    refuses to touch it (the caller routes partials to wait instead)."""
    symbol = state["symbol"]
    attempts = int(state.get("attempts") or 0)
    if attempts >= MAX_REPRICE_ATTEMPTS:
        if not state.get("exhausted_alerted"):
            state["exhausted_alerted"] = True
            write_pending_close(trade_id, state)
            emit(
                run_id=run_id, trigger=trigger, agent="finalize_pending_exits",
                event_type="exit_reprice_exhausted", severity=2,
                payload={"trade_id": trade_id, "symbol": symbol,
                         "order_id": state["order_id"], "attempts": attempts},
            )
        return

    dealt_now = 0.0
    try:
        dealt_now = float(order_row.get("dealt_qty") or 0)
    except (TypeError, ValueError):
        pass
    if dealt_now > 0:
        # Partially filled — never cancel. Let it finish or terminal-ize.
        return

    try:
        from trading_agent.mcp_servers.moomoo.server import cancel_paper_order
    except Exception as e:
        log.warning("_reprice: moomoo import failed: %s", e)
        return
    try:
        cancel_paper_order(order_id=state["order_id"])
    except Exception as e:
        # Could be racing a fill — leave the state intact; next tick's
        # order_map look-up settles it (or missing_ticks reconciliation).
        log.warning("_reprice cancel %s failed: %s", state["order_id"], e)
        return

    state["attempts"] = attempts + 1
    state["order_id"] = None  # replacement happens via _place_remainder
    write_pending_close(trade_id, state)
    _place_remainder(trade_id, state, run_id, trigger)


def _position_still_held(symbol: str) -> bool | None:
    """True/False from live broker positions; None when unknowable."""
    try:
        from trading_agent.mcp_servers.moomoo.server import get_positions
        for r in (get_positions().get("rows") or []):
            code = str(r.get("code") or r.get("symbol") or "")
            if code == symbol and float(r.get("qty") or 0) > 0:
                return True
        return False
    except Exception as e:
        log.warning("_position_still_held(%s) failed: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# The finalize pass — called from sync_fill_status every intraday tick
# ---------------------------------------------------------------------------

def _iter_pending() -> list[tuple[int, dict]]:
    """All (trade_id, state) pairs with an in-flight close, both stores."""
    out: list[tuple[int, dict]] = []
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT id, broker_fill_json->%s FROM journal_trades "
                "WHERE outcome = 'OPEN' AND broker_fill_json ? %s",
                (PENDING_KEY, PENDING_KEY),
            )
            for tid, st in cur.fetchall():
                st = st if isinstance(st, dict) else json.loads(st or "{}")
                st["source"] = "postgres"
                out.append((int(tid), st))
    except Exception as e:
        log.warning("_iter_pending: postgres read failed: %s", e)
    try:
        from trading_agent.db import connection
        with connection() as conn:
            rows = conn.execute(
                "SELECT id, exit_state_json FROM trades "
                "WHERE outcome = 'OPEN' AND exit_order_id IS NOT NULL",
            ).fetchall()
        for r in rows:
            try:
                st = json.loads(r["exit_state_json"] or "{}")
            except (TypeError, ValueError):
                st = {}
            if st:
                st["source"] = "sqlite"
                out.append((int(r["id"]), st))
    except Exception as e:
        log.warning("_iter_pending: sqlite read failed: %s", e)
    return out


def finalize_pending_exits(order_map: dict[str, dict], run_id: str,
                           trigger: str) -> dict:
    """Settle every in-flight close against the live broker order map.

    Returns counters for the caller's audit event.
    """
    confirmed = repriced = dead = waiting = replaced = unresolved = 0
    for trade_id, state in _iter_pending():
        order_id = state.get("order_id")

        # No live order (previous one died / replacement failed) → re-place.
        if not order_id:
            _place_remainder(trade_id, state, run_id, trigger)
            replaced += 1
            continue

        order = order_map.get(str(order_id))
        status = ""
        if order:
            status = str(order.get("order_status")
                         or order.get("order_status_str") or "").upper()

        if order is None:
            # get_orders is today-only: an exit that settled after the day's
            # last tick is invisible. Reconcile via live positions after
            # MAX_MISSING_TICKS consecutive misses.
            state["missing_ticks"] = int(state.get("missing_ticks") or 0) + 1
            if state["missing_ticks"] < MAX_MISSING_TICKS:
                write_pending_close(trade_id, state)
                waiting += 1
                continue
            held = _position_still_held(state["symbol"])
            if held is False:
                # Position is flat — the vanished order almost certainly
                # filled, but we never saw the dealt price. Settle at the
                # requested limit, FLAGGED UNCONFIRMED (the soak gate
                # rejects unconfirmed closes; the operator resolves them).
                ok, net = _settle(
                    trade_id, state, _remaining_qty(state),
                    float(state["requested_exit_price"]), unconfirmed=True)
                unresolved += 1
                emit(run_id=run_id, trigger=trigger,
                     agent="finalize_pending_exits",
                     event_type="exit_settled_unconfirmed", severity=2,
                     payload={"trade_id": trade_id, "symbol": state["symbol"],
                              "order_id": order_id, "ok": ok, "net_pnl": net})
            elif held is True:
                # Still holding — the order died invisibly. Re-place.
                state["order_id"] = None
                state["missing_ticks"] = 0
                write_pending_close(trade_id, state)
                _place_remainder(trade_id, state, run_id, trigger)
                replaced += 1
            else:
                # Broker unreachable for positions too — try again next tick.
                write_pending_close(trade_id, state)
                waiting += 1
            continue

        if status in _FILLED:
            dealt = order.get("dealt_avg_price") or order.get("dealt_avg_px")
            dealt_qty = order.get("dealt_qty") or order.get("qty") or _remaining_qty(state)
            try:
                dealt_f = float(dealt)
            except (TypeError, ValueError):
                dealt_f = float(state["requested_exit_price"])
            ok, net = _settle(trade_id, state, float(dealt_qty), dealt_f)
            if ok:
                confirmed += 1
                emit(
                    run_id=run_id, trigger=trigger,
                    agent="finalize_pending_exits",
                    event_type="position_closed",
                    payload={
                        "trade_id": trade_id, "symbol": state["symbol"],
                        "action": state.get("action"),
                        "qty_closed": float(dealt_qty) + _legs_qty(state),
                        "exit_price": dealt_f, "net_pnl": net,
                        "reason": state.get("reason"),
                        "order_id": order_id,
                        "fill_confirmed": True,
                    },
                )
                _notify_exit_filled(state, dealt_f, net)
        elif status in _DEAD:
            dead += 1
            dealt_qty = 0.0
            try:
                dealt_qty = float(order.get("dealt_qty") or 0)
            except (TypeError, ValueError):
                pass
            if dealt_qty > 0:
                # CANCELLED_PART etc — book the dealt leg, then re-place the
                # remainder (or settle if the legs now cover the target).
                dealt = order.get("dealt_avg_price") or order.get("dealt_avg_px")
                try:
                    leg_price = float(dealt)
                except (TypeError, ValueError):
                    leg_price = float(state["requested_exit_price"])
                state.setdefault("dealt_legs", []).append(
                    {"qty": dealt_qty, "price": leg_price})
                emit(run_id=run_id, trigger=trigger,
                     agent="finalize_pending_exits",
                     event_type="exit_partial_terminal", severity=1,
                     payload={"trade_id": trade_id, "symbol": state["symbol"],
                              "order_id": order_id, "dealt_qty": dealt_qty,
                              "dealt_price": leg_price,
                              "remaining": _remaining_qty(state)})
            state["order_id"] = None
            write_pending_close(trade_id, state)
            _place_remainder(trade_id, state, run_id, trigger)
        elif status in _PARTIAL:
            # Partially filled and still working — NEVER cancel. Wait for
            # completion or a terminal state; legs enter the books there.
            waiting += 1
        else:
            # Still resting. Reprice when stale (zero-fill orders only —
            # _reprice itself refuses orders with any dealt quantity).
            if state.get("missing_ticks"):
                state["missing_ticks"] = 0
                write_pending_close(trade_id, state)
            age = _age_seconds(state.get("placed_at"))
            if age is not None and age >= REPRICE_AFTER_SECONDS:
                _reprice(trade_id, state, order, run_id, trigger)
                repriced += 1
            else:
                waiting += 1
    return {"confirmed": confirmed, "repriced": repriced, "dead": dead,
            "waiting": waiting, "replaced": replaced, "unresolved": unresolved}


def _age_seconds(placed_at_iso: Any) -> float | None:
    try:
        placed = datetime.fromisoformat(str(placed_at_iso))
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - placed).total_seconds()
    except (TypeError, ValueError):
        return None


def _notify_exit_filled(state: dict, dealt: float, net: float) -> None:
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="trades",
            title=f"EXIT FILLED — {state['symbol']}",
            body=(
                f"Symbol: {state['symbol']}\n"
                f"Qty: {state['qty']:g} @ {dealt:.2f} (broker dealt)\n"
                f"Net P&L (after fees): {net:+.2f}\n"
                f"Reason: {state.get('reason')}"
            ),
            priority=4,
            tags=["white_check_mark", "chart_with_downwards_trend"],
        )
    except Exception as e:
        log.warning("_notify_exit_filled failed: %s", e)


__all__ = [
    "MAX_MISSING_TICKS",
    "MAX_REPRICE_ATTEMPTS",
    "PENDING_KEY",
    "REPRICE_AFTER_SECONDS",
    "build_pending_state",
    "finalize_pending_exits",
    "find_adoptable_close_order",
    "pending_close_trade_ids_pg",
    "pending_close_trade_ids_sqlite",
    "write_pending_close",
]
