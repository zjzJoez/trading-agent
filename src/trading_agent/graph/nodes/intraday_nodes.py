"""Phase 2.5 — Intraday monitor nodes.

Replaces stubs:
    refresh_quotes_and_greeks   fetch current marks + Greeks for open positions
    detect_exit_triggers        call exit-monitor LLM per position
    route_exit_or_hold          execute closes if any EXIT_* decision returned

Pipeline (intraday_monitor_graph):
    load_active_params → opend_health → load_open_positions
    → refresh_quotes_and_greeks  (this file)
    → update_excursions          (eod_learning.py — already real)
    → load_latest_regime → active_risk_snapshot
    → detect_exit_triggers       (this file)
    → route_exit_or_hold         (this file, places close orders + ends)
"""
from __future__ import annotations

import logging
import math

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_ticker(code: str) -> str:
    import re
    if not code:
        return ""
    s = code.split(".", 1)[-1]
    m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", s)
    return m.group(1) if m else s


# ---------------------------------------------------------------------------
# Node 0: sync_fill_status — reconcile journal entry-prices with broker fills
# ---------------------------------------------------------------------------

# Broker order statuses that mean "this order will never fill" → UNFILLED.
_DEAD_ORDER_STATUSES = {"CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DELETED", "EXPIRED"}
# Statuses that mean "filled" → sync the real avg fill price into the journal.
_FILLED_ORDER_STATUSES = {"FILLED_ALL", "FILLED_PART"}


def _order_status_of(order: dict | None) -> str:
    if not order:
        return ""
    return str(order.get("order_status")
               or order.get("order_status_str") or "").upper()


def _honest_entry_leg(dealt_avg: float, touch: float | None, *,
                      is_short: bool, underlying: str) -> float:
    """Clamp one combo ENTRY leg's dealt price to "fill no better than the
    touch" (M1-0.2 cost honesty).

    Moomoo SIMULATE books limit orders at the caller's limit price — a
    mid-priced entry "fills" at mid and never pays the spread. Honest
    accounting caps the fill at the quoted touch captured at placement:
      * short leg (we SOLD)   → can't receive more than the bid;
      * long leg  (we BOUGHT) → can't pay less than the ask.
    Missing touch → charge the calibrated per-underlying half-spread off
    the dealt price instead (global percent-of-mark fallback). The clamp is
    correct under either SIMULATE fill regime (books-at-limit or
    books-at-touch) — see the M1-0.2 cost audit Q4."""
    from trading_agent.execution_costs import half_spread_cost
    if touch is not None and touch > 0:
        return min(dealt_avg, touch) if is_short else max(dealt_avg, touch)
    hs = half_spread_cost(dealt_avg, 1, "OPT", underlying=underlying) / 100.0
    if is_short:
        return max(0.0, dealt_avg - hs)
    return dealt_avg + hs


def _sync_combo_entry(cursor, trade_id, symbol, meta, order_map,
                      *, run_id: str, trigger: str) -> None:
    """Combo twin of the single-order entry sync (3a).

    Resolves BOTH leg orders from the canonical combo payload and:
      * both FILLED_ALL → entry_price = the HONEST net credit:
        clamp(short_dealt_avg) − clamp(long_dealt_avg), each leg capped at
        the entry touch persisted in the combo payload (leg_quotes) via
        _honest_entry_leg; the raw dealt credit is kept alongside in
        broker_fill_json (entry_credit_raw / entry_credit_honest);
      * both dead      → outcome='UNFILLED' (same as single-leg);
      * mixed (one filled, one dead/absent) → sev-2 combo_entry_leg_mismatch,
        touch NOTHING (an operator problem, not a sync problem).
    Anything else (both still working, both absent) → leave alone.
    """
    orders_by_code: dict[str, dict] = {}
    for oid in meta.broker_order_ids:
        o = order_map.get(str(oid))
        if not o:
            continue
        code = str(o.get("code") or o.get("symbol") or "")
        orders_by_code.setdefault(code, o)
    short_o = orders_by_code.get(meta.short_leg)
    long_o = orders_by_code.get(meta.long_leg)
    # Positional fallback (long leg is placed FIRST in place_paper_option_combo)
    # for order rows that omit the code field.
    if short_o is None and long_o is None and len(meta.broker_order_ids) == 2:
        long_o = order_map.get(str(meta.broker_order_ids[0]))
        short_o = order_map.get(str(meta.broker_order_ids[1]))

    s_status = _order_status_of(short_o)
    l_status = _order_status_of(long_o)
    s_filled = s_status == "FILLED_ALL"
    l_filled = l_status == "FILLED_ALL"
    s_dead = s_status in _DEAD_ORDER_STATUSES
    l_dead = l_status in _DEAD_ORDER_STATUSES

    if s_filled and l_filled:
        try:
            s_avg = float(short_o.get("dealt_avg_price")
                          or short_o.get("dealt_avg_px") or 0)
            l_avg = float(long_o.get("dealt_avg_price")
                          or long_o.get("dealt_avg_px") or 0)
        except (TypeError, ValueError):
            return
        if s_avg <= 0 or l_avg <= 0:
            return
        raw_credit = round(s_avg - l_avg, 4)
        # M1-0.2 "fill no better than the touch": clamp each leg at the
        # entry quote captured with the R5d guard snapshot; a leg with no
        # quote is charged the calibrated per-underlying half-spread.
        underlying = _bare_ticker(meta.short_leg)
        honest_short = _honest_entry_leg(
            s_avg, meta.short_bid, is_short=True, underlying=underlying)
        honest_long = _honest_entry_leg(
            l_avg, meta.long_ask, is_short=False, underlying=underlying)
        honest_credit = round(honest_short - honest_long, 4)
        try:
            with cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_trades
                    SET entry_price = %s::numeric,
                        broker_fill_json = COALESCE(broker_fill_json, '{}'::jsonb)
                          || jsonb_build_object(
                               'status', 'FILLED_ALL'::text,
                               'combo_fill_synced', TRUE,
                               'avg_fill_price', %s::numeric,
                               'entry_credit_raw', %s::numeric,
                               'entry_credit_honest', %s::numeric,
                               'fill_synced_at', NOW()::text)
                    WHERE id = %s AND outcome = 'OPEN'
                    """,
                    (honest_credit, honest_credit, raw_credit,
                     honest_credit, int(trade_id)),
                )
        except Exception as e:
            log.warning("[sync_fill_status] combo fill update id=%s failed: %s",
                        trade_id, e)
    elif s_dead and l_dead:
        try:
            with cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_trades
                    SET outcome = 'UNFILLED', closed_at = NOW(),
                        close_reason = COALESCE(close_reason,
                            'sync_fill_status: combo legs ' || %s)
                    WHERE id = %s AND outcome = 'OPEN'
                    """,
                    (f"{s_status}/{l_status}", int(trade_id)),
                )
        except Exception as e:
            log.warning("[sync_fill_status] combo unfilled update id=%s failed: %s",
                        trade_id, e)
    elif (s_filled and (l_dead or long_o is None)) or \
            (l_filled and (s_dead or short_o is None)):
        # One leg filled, the sibling dead/absent — the "vertical" is legged
        # apart at the broker. Touch nothing; page the operator ONCE per
        # (symbol, day) — the condition persists every tick while the
        # operator reconciles, and repeated sev-2s train alert fatigue.
        if _alerted_today("combo_entry_leg_mismatch", symbol):
            log.info(
                "[sync_fill_status] combo %s legs still mismatched "
                "(%s/%s) — alerted today already",
                symbol, s_status or "ABSENT", l_status or "ABSENT",
            )
            return
        emit(
            run_id=run_id, trigger=trigger, agent="sync_fill_status",
            event_type="combo_entry_leg_mismatch", severity=2,
            payload={"trade_id": int(trade_id), "symbol": symbol,
                     "short_leg": meta.short_leg, "long_leg": meta.long_leg,
                     "short_status": s_status or "ABSENT",
                     "long_status": l_status or "ABSENT",
                     "action_required": "reconcile legs manually — one leg "
                                        "filled, the other is dead/absent"},
        )


def sync_fill_status(state: TradingGraphState) -> dict:
    """Reconcile each OPEN journal_trade's fill state against the live broker.

    The journal's broker_fill_json is stamped at PLACEMENT time (status
    SUBMITTED, fill_qty 0). It is never updated when the order later fills.
    6/2: IBM/SNOW showed SUBMITTED in the journal but were FILLED_ALL at the
    broker. Without this sync, (a) the journal misreports entry prices, and
    (b) reconcile_phantom_trades would WRONGLY void these real positions
    after 24h (it keys off the stale fill_qty=0).

    For each OPEN trade with a broker_order_id, look up the live order:
      * FILLED_*  → set entry_price = real dealt_avg_price, stamp
                    broker_fill_json {status, fill_qty, avg_fill_price}.
      * dead      → outcome='UNFILLED', closed_at=NOW.
      * pending   → leave (phantom-reconcile handles the truly-stuck after 24h).

    Best-effort: any broker/DB error skips the sync for this tick.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[intraday/sync_fill_status] run_id=%s", run_id)

    # 1. Live broker order map first — shared by the entry-price sync AND the
    # exit-fill finalize pass below. Broker down → nothing can settle this tick.
    order_map: dict[str, dict] = {}
    try:
        from trading_agent.mcp_servers.moomoo.server import get_orders
        for r in (get_orders().get("rows") or []):
            oid = str(r.get("order_id") or r.get("orderID") or "").strip()
            if oid:
                order_map[oid] = r
    except Exception as e:
        log.warning("[sync_fill_status] broker get_orders failed: %s — skip tick", e)
        return {}

    # 2. OPEN journal trades that have a broker order id (entry-side sync).
    # Postgres being down must not skip the exit finalize pass — the SQLite
    # side of finalize_pending_exits works without it.
    open_trades: list = []
    cursor = None
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, symbol, broker_order_id, entry_price,
                       broker_fill_json->'combo' AS combo
                FROM journal_trades
                WHERE outcome = 'OPEN' AND broker_order_id IS NOT NULL
                  AND broker_order_id <> ''
                """,
            )
            open_trades = cur.fetchall()
    except Exception as e:
        log.warning("[sync_fill_status] open-trades read failed: %s", e)

    from trading_agent.combo import combo_meta

    synced_filled = 0
    marked_unfilled = 0
    for row_t in open_trades:
        trade_id, symbol, broker_order_id, entry_price = row_t[:4]
        combo_raw = row_t[4] if len(row_t) > 4 else None
        # Combo rows: entry_price is the NET CREDIT across BOTH legs, and the
        # stored broker_order_id is the LONG leg's id (long is placed first in
        # place_paper_option_combo). The single-order sync below would clobber
        # the net credit with ONE leg's dealt price — combos resolve both leg
        # orders from the canonical payload instead.
        combo_m = combo_meta(combo_raw)
        if combo_m is not None:
            _sync_combo_entry(
                cursor, trade_id, symbol, combo_m, order_map,
                run_id=run_id, trigger=trigger,
            )
            continue
        if combo_raw:
            # Combo payload PRESENT but unparseable — fail closed. The
            # single-order branch below would clobber entry_price (the net
            # credit) with ONE leg's dealt price (broker_order_id is the
            # LONG leg's id). Skip the row and alert once per (symbol, day).
            log.warning(
                "[sync_fill_status] combo payload unparseable for trade "
                "%s (%s) — skipping single-order sync (fail closed)",
                trade_id, symbol,
            )
            if not _alerted_today("combo_payload_unparseable", symbol):
                emit(
                    run_id=run_id, trigger=trigger, agent="sync_fill_status",
                    event_type="combo_payload_unparseable", severity=2,
                    payload={"trade_id": int(trade_id), "symbol": symbol,
                             "action_required": "repair broker_fill_json"
                                                "->'combo' on this row — the"
                                                " engine HOLDs it until then"},
                )
            continue
        order = order_map.get(str(broker_order_id))
        if not order:
            continue
        status = str(order.get("order_status") or order.get("order_status_str") or "").upper()
        if status in _FILLED_ORDER_STATUSES:
            avg = order.get("dealt_avg_price") or order.get("dealt_avg_px")
            filled_qty = order.get("dealt_qty") or order.get("qty")
            try:
                avg_f = float(avg) if avg else None
            except (TypeError, ValueError):
                avg_f = None
            try:
                with cursor() as cur:
                    # Explicit casts — jsonb_build_object + COALESCE can't
                    # infer param types from a bare %s placeholder (psycopg
                    # sends them untyped), which raised "could not determine
                    # data type of parameter $2".
                    cur.execute(
                        """
                        UPDATE journal_trades
                        SET entry_price = COALESCE(%s::numeric, entry_price),
                            broker_fill_json = COALESCE(broker_fill_json, '{}'::jsonb)
                              || jsonb_build_object(
                                   'status', %s::text,
                                   'fill_qty', %s::text,
                                   'avg_fill_price', %s::numeric,
                                   'fill_synced_at', NOW()::text)
                        WHERE id = %s AND outcome = 'OPEN'
                        """,
                        (avg_f, status, str(filled_qty) if filled_qty is not None else None,
                         avg_f, int(trade_id)),
                    )
                synced_filled += 1
            except Exception as e:
                log.warning("[sync_fill_status] fill update id=%s failed: %s", trade_id, e)
        elif status in _DEAD_ORDER_STATUSES:
            try:
                with cursor() as cur:
                    cur.execute(
                        """
                        UPDATE journal_trades
                        SET outcome = 'UNFILLED', closed_at = NOW(),
                            close_reason = COALESCE(close_reason,
                                'sync_fill_status: broker order ' || %s)
                        WHERE id = %s AND outcome = 'OPEN'
                        """,
                        (status, int(trade_id)),
                    )
                marked_unfilled += 1
            except Exception as e:
                log.warning("[sync_fill_status] unfilled update id=%s failed: %s", trade_id, e)

    if synced_filled or marked_unfilled:
        emit(
            run_id=run_id, trigger=trigger, agent="sync_fill_status",
            event_type="fill_status_synced",
            payload={"filled": synced_filled, "unfilled": marked_unfilled,
                     "n_open_checked": len(open_trades)},
        )
        log.info("[sync_fill_status] filled=%d unfilled=%d", synced_filled, marked_unfilled)

    # 3. Exit-fill confirmation: settle every in-flight close placed by
    # route_exit_or_hold against the live order map — journal closes happen
    # HERE, at the broker's dealt price, never at placement time.
    try:
        from trading_agent.exits.fill_confirm import finalize_pending_exits
        stats = finalize_pending_exits(order_map, run_id, trigger)
        if any(stats.values()):
            emit(
                run_id=run_id, trigger=trigger, agent="sync_fill_status",
                event_type="pending_exits_finalized", payload=stats,
            )
            log.info("[sync_fill_status] pending exits: %s", stats)
    except Exception as e:
        log.warning("[sync_fill_status] finalize_pending_exits failed: %s", e)
    return {}


def _alerted_today(event_type: str, symbol: str) -> bool:
    """True iff we already fired `event_type` for this symbol today.

    Per-(event_type, symbol, day) dedup — without this, every 15-min
    intraday tick spams the same alert for a persistent condition (6/1
    incident: 22 position_no_exit_plan events on one QQQ contract before
    the operator noticed; same storm shape for the combo sev-2s while an
    operator is manually reconciling a legged-apart entry).
    """
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM agent_events
                WHERE event_type = %s
                  AND ts > DATE_TRUNC('day', NOW())
                  AND payload->>'symbol' = %s
                LIMIT 1
                """,
                (event_type, symbol),
            )
            return cur.fetchone() is not None
    except Exception:
        # On DB error, default to "yes we alerted" so we don't spam
        # if the DB is degraded.
        return True


def _no_plan_alerted_today(symbol: str) -> bool:
    """True iff we already fired position_no_exit_plan for this symbol today."""
    return _alerted_today("position_no_exit_plan", symbol)


def _is_option_code(code: str) -> bool:
    import re
    return bool(re.search(r"\d{6,8}[CP]\d+$", code or ""))


# ---------------------------------------------------------------------------
# Node 1: refresh_quotes_and_greeks
# ---------------------------------------------------------------------------

def refresh_quotes_and_greeks(state: TradingGraphState) -> dict:
    """Refresh last-price marks (and option Greeks if available) for every
    open position in state["positions"].

    Uses moomoo get_quote (batched, list[str] signature) for both stock and
    option symbols. Per the API docstring, get_quote returns greeks/iv/expiry
    in the same row when the symbol is an option. Failures degrade to stale
    marks so downstream nodes still function.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[intraday/refresh_quotes_and_greeks] run_id=%s", run_id)

    positions: list[dict] = list(state.get("positions") or [])
    if not positions:
        emit(run_id=run_id, trigger=trigger, agent="refresh_quotes_and_greeks",
             event_type="no_positions", payload={})
        return {}

    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
    except Exception as e:
        log.error("[refresh_quotes] moomoo import failed: %s", e)
        return {}

    refreshed: list[dict] = []
    stale_count = 0

    # Batch fetch all symbols in one get_quote call (signature: list[str])
    symbols_to_fetch = [p["symbol"] for p in positions if p.get("symbol")]
    quote_by_symbol: dict[str, dict] = {}
    try:
        result = get_quote(symbols_to_fetch)
        for row in (result.get("rows") or []):
            code = row.get("code") or row.get("symbol") or ""
            if code:
                quote_by_symbol[code] = row
    except Exception as e:
        log.warning("[refresh_quotes] get_quote batch failed: %s", e)

    for pos in positions:
        symbol = pos.get("symbol") or ""
        if not symbol:
            refreshed.append(pos)
            continue

        updated = dict(pos)
        row = quote_by_symbol.get(symbol)
        if row is None:
            stale_count += 1
            refreshed.append(updated)
            continue

        try:
            mark = float(row.get("last_price") or pos.get("mark") or 0)
            if mark > 0:
                updated["mark"] = mark

            # Live bid/ask: route_exit_or_hold prices SELL closes off the BID
            # (the side you can actually hit), and the quoted spread feeds the
            # execution-cost audit trail.
            bid_raw = row.get("bid_price") or row.get("bid")
            ask_raw = row.get("ask_price") or row.get("ask")
            if bid_raw is not None and float(bid_raw) > 0:
                updated["bid"] = float(bid_raw)
            if ask_raw is not None and float(ask_raw) > 0:
                updated["ask"] = float(ask_raw)

            if _is_option_code(symbol):
                # Options: greeks/iv/expiry come back in the same row per docstring
                if row.get("delta") is not None:
                    updated["delta"] = float(row["delta"])
                if row.get("gamma") is not None:
                    updated["gamma"] = float(row["gamma"])
                if row.get("vega") is not None:
                    updated["vega"] = float(row["vega"])
                if row.get("theta") is not None:
                    updated["theta"] = float(row["theta"])
                iv_raw = row.get("imp_volatility") or row.get("implied_volatility") or row.get("iv")
                if iv_raw is not None:
                    updated["iv"] = float(iv_raw)
                exp_raw = row.get("expiry_date") or row.get("strike_time")
                if exp_raw is not None:
                    from datetime import date
                    try:
                        exp = date.fromisoformat(str(exp_raw)[:10])
                        updated["dte"] = (exp - date.today()).days
                    except Exception:
                        pass
        except Exception as e:
            log.warning("[refresh_quotes] symbol=%s parse failed: %s", symbol, e)
            stale_count += 1

        refreshed.append(updated)

    emit(
        run_id=run_id, trigger=trigger, agent="refresh_quotes_and_greeks",
        event_type="quotes_refreshed",
        payload={"n_positions": len(refreshed), "stale_count": stale_count},
    )
    return {"positions": refreshed}


# ---------------------------------------------------------------------------
# Node 2: detect_exit_triggers
# ---------------------------------------------------------------------------

def _load_journal_enrichment_by_symbol() -> dict[str, dict]:
    """Return ``{broker_symbol: enrichment_dict}`` for every OPEN journal row.

    The exit LLM needs ``thesis_id``, ``thesis_text``, ``direction``,
    ``invalidation``, ``stop`` and ``target`` for each open position. These
    live in Postgres ``journal_trades`` + ``journal_theses``, NOT in the
    broker's ``get_positions()`` payload — so ``state["positions"]`` (which
    comes straight from moomoo) is missing all of them.

    Without this enrichment, ``_thesis_summary_for(pos)`` returns
    "(no thesis linked)" for every position even when a real thesis row
    exists (verified on the 5/12 NVDA SCRATCH self-exit). The exit_monitor
    LLM then sees ``thesis_summary: (no thesis linked)`` and rationally
    recommends EXIT_CAUTIOUS for "unvetted exposure" — closing the same
    position the agent just opened.

    Notes:
        * Uses Postgres, NOT the journal-mcp SQLite ``trades`` table — the
          MCP table is stale; autonomous fills only write to Postgres.
        * Single batched query, not per-position lookup, so the lookup cost
          is O(1) per intraday tick regardless of position count.
        * Returns ``{}`` on DB error — the caller falls back to the legacy
          per-position lookup in ``_thesis_summary_for``.
    """
    out: dict[str, dict] = {}
    try:
        from trading_agent.store.postgres import get_open_journal_trades
        rows = get_open_journal_trades()
    except Exception as e:
        log.warning("[detect_exit_triggers] journal enrichment query failed: %s", e)
        return out
    if rows is None:  # store unreadable — caller falls back to legacy lookup
        return out
    from trading_agent.combo import combo_meta
    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        # Rows are newest-first; first-row-wins so a phantom (placed-never-
        # filled) OPEN row can't shadow the later real fill on the same symbol.
        if str(symbol) in out:
            continue
        trade_id = row.get("trade_id")
        thesis_id = row.get("thesis_id")
        stop = row.get("stop")
        target = row.get("target")
        exit_plan = row.get("exit_plan")
        mfe_so_far = row.get("mfe_so_far")
        # Combo detection is fail-closed: a malformed payload parses to None
        # and the row degrades to "unknown" (HOLD), never to single-leg
        # management (which would orphan the protective long wing). A payload
        # that is PRESENT but unparseable is flagged so _merge_combo_positions
        # can enforce the HOLD (a bare combo=None would be dropped by the
        # enrichment merge and the row would silently manage as single-leg).
        combo_raw = row.get("combo")
        combo_parsed = combo_meta(combo_raw)
        out[str(symbol)] = {
            "trade_id": int(trade_id) if trade_id is not None else None,
            "thesis_id": int(thesis_id) if thesis_id is not None else None,
            "stop": float(stop) if stop is not None else None,
            "target": float(target) if target is not None else None,
            # exit_plan is JSONB; psycopg returns a dict directly.
            "exit_plan": exit_plan if isinstance(exit_plan, dict) else None,
            "mfe_so_far": float(mfe_so_far) if mfe_so_far is not None else None,
            "opened_at": row.get("opened_at"),
            "scale_rungs_taken": int(row.get("scale_rungs_taken") or 0),
            "regime_downsized_at_label": row.get("regime_downsized_at_label"),
            "thesis_status": row.get("thesis_status"),
            "direction": row.get("direction"),
            "thesis_text": row.get("thesis_text"),
            "invalidation": row.get("invalidation"),
            "combo": combo_parsed,
            "combo_unparseable": bool(combo_raw) and combo_parsed is None,
        }
    return out


def _leg_quote_of(pos: dict) -> dict:
    """{bid, ask, mark} snapshot of one broker leg position."""
    def _f(v):
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None
    return {"bid": _f(pos.get("bid")), "ask": _f(pos.get("ask")),
            "mark": _f(pos.get("mark"))}


def _merge_combo_positions(
    positions: list[dict], *, run_id: str, trigger: str,
    pending_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Collapse each journaled combo's two broker legs into ONE synthetic
    position marked net-of-both-legs, and remove the long wing from the
    evaluation list entirely (it must produce no decision and no
    ``position_no_exit_plan`` "manual trade" alert).

    ``pending_ids``: journal trade ids with a combo close in flight. Leg
    divergence is NORMAL while a trim/close's two legs fill across a tick
    boundary — those rows HOLD quietly (info log) instead of paging sev-2.

    Returns ``(merged_positions, forced_hold_decisions)``. Failure modes are
    fail-closed HOLDs (the combo's legs are pulled from evaluation so nothing
    downstream can single-leg-manage them):

      * combo payload present but unparseable (``combo_unparseable``), or a
        SHORT-direction exit plan with no parsed combo meta → sev-2
        ``combo_payload_unparseable``, HOLD — the degraded row must NEVER
        fall through to single-leg management (it would orphan the wing);
      * long leg missing at the broker → sev-2 ``combo_long_leg_orphaned``,
        HOLD reason ``combo_long_leg_missing``;
      * leg qty mismatch → sev-2 ``combo_leg_qty_mismatch``, HOLD;
      * either leg mark ≤ 0/stale, or spread mark ≤ 0 → HOLD reason
        ``combo_mark_invalid`` (a bogus low mark would falsely fire the
        SHORT-direction 50%-PT target — this guard is mandatory).

    Sev-2 emits are deduped per (event_type, symbol, day) via
    ``_alerted_today`` — repeats degrade to log lines.

    No combos (and no degraded rows) in the list → the positions list is
    returned unchanged (identity for single-leg flows).
    """
    pending_ids = pending_ids or set()
    drop_symbols: set[str] = set()
    synthetic_by_symbol: dict[str, dict] = {}
    forced_holds: list[dict] = []

    def _hold(symbol: str, reason: str) -> None:
        forced_holds.append({
            "symbol": symbol, "action": "HOLD",
            "exit_qty_factor": 0.0, "reason": reason,
        })

    def _emit_deduped(event_type: str, symbol: str, payload: dict) -> None:
        if _alerted_today(event_type, symbol):
            log.info("[detect_exit_triggers] %s %s persists — alerted "
                     "today already", event_type, symbol)
            return
        emit(run_id=run_id, trigger=trigger, agent="detect_exit_triggers",
             event_type=event_type, severity=2, payload=payload)

    # Fail-closed backstop (M1-0.1 degrade contract): a journal row whose
    # combo payload is present-but-unparseable — or that carries a
    # SHORT-direction exit plan with no combo meta (single-leg SELL-to-open
    # is hard-blocked, so this signature can only be a degraded combo) —
    # must HOLD, never fall through to single-leg management: the single-leg
    # path would BUY back the short leg alone and orphan the wing.
    for p in positions:
        sym = p.get("symbol")
        if not sym or p.get("combo") is not None:
            continue
        ep = p.get("exit_plan")
        short_plan = isinstance(ep, dict) and ep.get("direction") == "SHORT"
        if p.get("combo_unparseable") or short_plan:
            drop_symbols.add(sym)
            _hold(sym, "combo_payload_unparseable")
            _emit_deduped(
                "combo_payload_unparseable", sym,
                {"symbol": sym, "trade_id": p.get("trade_id"),
                 "note": "combo payload missing/unparseable on a SHORT row "
                         "— held fail-closed, never single-leg managed"})

    combo_shorts = [p for p in positions
                    if p.get("combo") is not None and p.get("symbol")]
    if not combo_shorts and not drop_symbols:
        return positions, []

    by_symbol: dict[str, dict] = {
        p["symbol"]: p for p in positions if p.get("symbol")}

    for short_pos in combo_shorts:
        meta = short_pos["combo"]
        symbol = short_pos["symbol"]
        trade_id = short_pos.get("trade_id")
        close_in_flight = trade_id is not None and trade_id in pending_ids
        drop_symbols.add(symbol)  # short leg never evaluates as single-leg
        long_pos = by_symbol.get(meta.long_leg)
        if long_pos is None:
            if close_in_flight:
                log.info(
                    "[detect_exit_triggers] combo %s wing absent while a "
                    "close is in flight (trade %s) — normal pending state",
                    symbol, trade_id,
                )
            else:
                _emit_deduped(
                    "combo_long_leg_orphaned", symbol,
                    {"symbol": symbol, "short_leg": meta.short_leg,
                     "long_leg": meta.long_leg,
                     "note": "protective wing not among broker "
                             "positions — combo held fail-closed"})
            _hold(symbol, "combo_close_in_flight" if close_in_flight
                  else "combo_long_leg_missing")
            continue
        drop_symbols.add(meta.long_leg)
        short_qty = abs(float(short_pos.get("qty") or 0))
        long_qty = abs(float(long_pos.get("qty") or 0))
        if abs(short_qty - long_qty) > 0:
            if close_in_flight:
                log.info(
                    "[detect_exit_triggers] combo %s leg qtys diverge "
                    "(%g/%g) while a close is in flight (trade %s) — "
                    "normal pending state",
                    symbol, short_qty, long_qty, trade_id,
                )
            else:
                _emit_deduped(
                    "combo_leg_qty_mismatch", symbol,
                    {"symbol": symbol, "short_qty": short_qty,
                     "long_qty": long_qty, "long_leg": meta.long_leg})
            _hold(symbol, "combo_close_in_flight" if close_in_flight
                  else "combo_leg_qty_mismatch")
            continue
        short_q = _leg_quote_of(short_pos)
        long_q = _leg_quote_of(long_pos)
        short_mark = short_q["mark"]
        long_mark = long_q["mark"]
        if short_mark is None or long_mark is None:
            _hold(symbol, "combo_mark_invalid")
            continue
        spread_mark = round(short_mark - long_mark, 4)
        if spread_mark <= 0:
            _hold(symbol, "combo_mark_invalid")
            continue
        synth = dict(short_pos)
        synth["asset_type"] = "OPT"
        synth["qty"] = short_qty                 # whole spread units
        synth["entry_price"] = meta.net_credit   # per-unit net credit
        synth["mark"] = spread_mark              # net-of-both-legs value
        synth["combo_legs"] = {"short": short_q, "long": long_q}
        synthetic_by_symbol[symbol] = synth

    merged: list[dict] = []
    for p in positions:
        sym = p.get("symbol")
        if sym in synthetic_by_symbol:
            merged.append(synthetic_by_symbol.pop(sym))
        elif sym in drop_symbols:
            continue
        else:
            merged.append(p)
    return merged, forced_holds


def detect_exit_triggers(state: TradingGraphState) -> dict:
    """Deterministic exit-rule executor — no LLM in the hot path.

    For each open position, loads the ExitPlan stored at entry time (or
    falls back to journal_trades.stop / .target for legacy rows) and
    consults trading_agent.exits.hard_executor.hard_exit_decision. The
    decision is pure code:

      P0 regime kill switch (CRISIS) → exit
      P1 hard_stop                   → exit
      P2 DTE rules (options)         → exit / upgrade-to-intrinsic-floor
      P3 hard_target + scale-out     → full / partial exit
      P4 trailing stop (post-engage) → exit
      P5 max age                     → exit

    See docs/position_management.md for the full rule table + thresholds.

    Writes decisions into state["journal"]["exit_decisions"] as a list of
    dicts: {symbol, action, exit_qty_factor, reason}.

    History: this node previously called the exit_monitor LLM (Haiku) per
    position. We replaced it on 2026-05-22 after audit of a SPY 742C trade
    that had R:R 0.6:1 and an LLM-set stop only 55% below entry — the
    LLM was making subjective calls without a deterministic safety net.
    Hard rules + R7 R:R floor at entry close that gap.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[intraday/detect_exit_triggers] run_id=%s", run_id)

    positions: list[dict] = list(state.get("positions") or [])
    if not positions:
        return {}

    regime = state.get("regime") or {}
    regime_label = str(regime.get("label") or "")

    # Enrich each broker position with its journal_trades + journal_theses row.
    # The hard executor needs: stop, target, exit_plan (JSONB), mfe_so_far,
    # opened_at. Without enrichment we'd HOLD everything (no plan = safe path).
    enrichment_by_symbol = _load_journal_enrichment_by_symbol()
    enriched_count = 0
    if enrichment_by_symbol:
        for pos in positions:
            symbol = pos.get("symbol") or ""
            enrichment = enrichment_by_symbol.get(symbol)
            if not enrichment:
                continue
            for k, v in enrichment.items():
                if v is not None and k not in pos:
                    pos[k] = v
            enriched_count += 1
    log.info(
        "[detect_exit_triggers] enriched %d/%d positions from Postgres journal",
        enriched_count, len(positions),
    )

    # In-flight combo closes: leg divergence (qtys, missing wing, residual
    # wing) is NORMAL while a close's two legs fill across a tick boundary —
    # those rows hold quietly instead of paging sev-2 every tick. None
    # (store unreadable) degrades to "no suppression" (alerting is the safe
    # direction when we can't prove a close explains the divergence).
    pending_ids: set[int] = set()
    try:
        from trading_agent.exits.fill_confirm import pending_close_trade_ids_pg
        pending_ids = pending_close_trade_ids_pg() or set()
    except Exception as e:
        log.warning("[detect_exit_triggers] pending-close lookup failed: %s", e)

    # Leg symbols of combos whose close is in flight: the residual wing (or
    # short leg) surfacing alone at the broker mid-close must NOT fire the
    # "manual trade" position_no_exit_plan alert.
    in_flight_combo_legs: set[str] = set()
    for enr in enrichment_by_symbol.values():
        c_meta = enr.get("combo")
        t_id = enr.get("trade_id")
        if c_meta is not None and t_id is not None and t_id in pending_ids:
            in_flight_combo_legs.update({c_meta.short_leg, c_meta.long_leg})

    # Combo collapse: each journaled vertical's two broker legs become ONE
    # synthetic position marked net-of-both-legs (entry=net_credit, mark=
    # spread value); the long wing is removed from evaluation entirely.
    # Failure modes come back as forced fail-closed HOLD decisions.
    positions, combo_hold_decisions = _merge_combo_positions(
        positions, run_id=run_id, trigger=trigger, pending_ids=pending_ids)

    # Underlying spot prices for option intrinsic-floor checks. Best effort:
    # if moomoo is down we still run with intrinsic=None and the executor
    # will skip the intrinsic-floor branch (it'll just not upgrade the stop).
    underlying_marks: dict[str, float] = {}
    underlying_syms = {
        _bare_ticker(p.get("symbol") or "")
        for p in positions
        if (p.get("asset_type") == "OPT" or _is_option_code(p.get("symbol") or ""))
    }
    underlying_syms.discard("")
    if underlying_syms:
        try:
            from trading_agent.mcp_servers.moomoo.server import get_quote
            q = get_quote([f"US.{t}" for t in underlying_syms])
            for r in (q.get("rows") or []):
                code = r.get("code") or ""
                bare = _bare_ticker(code)
                last = r.get("last_price") or r.get("cur_price")
                if bare and last:
                    underlying_marks[bare] = float(last)
        except Exception as e:
            log.warning("[detect_exit_triggers] underlying quote fetch failed: %s", e)

    try:
        from datetime import datetime, timezone
        from trading_agent.exits import hard_exit_decision
        from trading_agent.exits.hard_executor import load_exit_plan
    except Exception as e:
        log.error("[detect_exit_triggers] hard_executor import failed: %s", e)
        return {}

    now_utc = datetime.now(timezone.utc)
    decisions: list[dict] = list(combo_hold_decisions)
    hold_count = len(combo_hold_decisions)
    exit_count = 0
    no_plan_count = 0

    for pos in positions:
        symbol = pos.get("symbol") or ""
        if not symbol:
            continue
        plan = load_exit_plan(
            raw=pos.get("exit_plan"),
            fallback_stop=pos.get("stop"),
            fallback_target=pos.get("target"),
        )
        if plan is None and symbol in in_flight_combo_legs:
            # Residual leg of a combo whose close is in flight (e.g. the
            # wing still working after the short leg's BTC filled). A normal
            # pending state, not an unmanaged manual position — HOLD quietly.
            decisions.append({
                "symbol": symbol,
                "action": "HOLD",
                "exit_qty_factor": 0.0,
                "reason": "combo_close_in_flight_residual_leg",
            })
            hold_count += 1
            continue
        if plan is None:
            # No plan, no stop, no target. This is almost always a MANUAL
            # trade: the user placed the order directly on moomoo, bypassing
            # the system, so no journal_trades row exists.
            #
            # Behavior:
            #   1. Default HOLD on this tick (we cannot evaluate — no rules).
            #   2. Emit sev-1 event once per (symbol, day) so the dashboard
            #      sees it but every 15-min tick doesn't spam (6/1 incident:
            #      22 events on one position in one day).
            #
            # The user is responsible for managing manual positions.
            no_plan_count += 1
            decisions.append({
                "symbol": symbol,
                "action": "HOLD",
                "exit_qty_factor": 0.0,
                "reason": "no_exit_plan_and_no_legacy_stop_target_manual_trade",
            })
            if not _no_plan_alerted_today(symbol):
                emit(
                    run_id=run_id, trigger=trigger, agent="detect_exit_triggers",
                    event_type="position_no_exit_plan",
                    severity=1,
                    payload={
                        "symbol": symbol,
                        "interpretation": "likely_manual_trade_not_in_journal",
                        "action_required": "either close on broker or add "
                                           "journal_trades row with exit_plan",
                    },
                )
            hold_count += 1
            continue

        underlying = _bare_ticker(symbol)
        decision = hard_exit_decision(
            pos=pos,
            plan=plan,
            regime_label=regime_label,
            now_utc=now_utc,
            mfe_so_far=pos.get("mfe_so_far"),
            underlying_mark=underlying_marks.get(underlying),
            scale_rungs_taken=int(pos.get("scale_rungs_taken") or 0),
            regime_downsized_at_label=pos.get("regime_downsized_at_label"),
            thesis_status=pos.get("thesis_status"),
        )

        if decision is None:
            action = "HOLD"
            qty_factor = 0.0
            reason = "no_rule_triggered"
        else:
            action = decision.action
            qty_factor = decision.exit_qty_factor
            reason = decision.reason

        full_qty = float(pos.get("qty") or 0)
        decision_extra: dict = {}
        if pos.get("combo") is not None:
            # Combo whole-unit semantics (P0b "trim 50%" on 1–2 lot
            # verticals): a unit is ONE spread (both legs together) — never
            # leg apart, never fractionate. floor(), not round(): round(1.5)
            # would trim 2 of 3 lots where the plan says floor(3/2)=1.
            if action != "HOLD":
                units_to_close = int(math.floor(full_qty * qty_factor)) \
                    if qty_factor < 1.0 else int(full_qty)
                if units_to_close >= 1:
                    decision_extra = {"combo": True,
                                      "close_units": units_to_close}
                else:
                    # qty==1 downsize: trimming half a spread is impossible.
                    # Skip, but carry the regime label so route stamps
                    # regime_downsized_at_label WITHOUT an order — the stamp
                    # is both the plan's "skip (emit event)" dedup and the
                    # once-per-episode guard.
                    decision_extra = {"combo": True}
                    if action == "EXIT_REGIME_DOWNSIZE":
                        decision_extra["stamp_regime_label"] = regime_label
                    reason = (
                        f"combo_trim_skipped_whole_unit: {action} factor="
                        f"{qty_factor:.2f} on qty={int(full_qty)}: "
                        f"{reason[:120]}"
                    )
                    action = "HOLD"
                    qty_factor = 0.0
        elif action != "HOLD" and 0.0 < qty_factor < 1.0 and full_qty <= 1:
            # Sanity guard (single-leg only): a partial-exit signal on a
            # 1-contract option position is physically impossible (options
            # don't fractionate). The downstream route_exit_or_hold would
            # `max(1, round(1 * 0.5)) = 1` and force a FULL close,
            # contradicting the partial signal. Demote to HOLD so the next
            # tick re-evaluates (likely hits target if mark stays up).
            log.info(
                "[detect_exit_triggers] %s demoting %s (factor=%.2f, qty=%.0f) → HOLD",
                symbol, action, qty_factor, full_qty,
            )
            reason = (
                f"demoted_from_{action}_partial_impossible_on_qty_{int(full_qty)}: "
                f"{reason[:120]}"
            )
            action = "HOLD"
            qty_factor = 0.0

        decisions.append({
            "symbol": symbol,
            "action": action,
            "exit_qty_factor": qty_factor,
            "reason": reason,
            # carried so route_exit_or_hold can stamp the downsize guard column
            "regime_label": regime_label,
            **decision_extra,
        })
        if action == "HOLD":
            hold_count += 1
        else:
            exit_count += 1
        log.info("[detect_exit_triggers] %s → %s (%s)", symbol, action, reason[:80])

    emit(
        run_id=run_id, trigger=trigger, agent="detect_exit_triggers",
        event_type="exit_decisions",
        payload={
            "hold": hold_count, "exit": exit_count,
            "no_plan": no_plan_count,
            "decisions": decisions,
        },
    )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["exit_decisions"] = decisions
    # positions is the combo-merged list: route_exit_or_hold needs the
    # synthetic combo positions (meta + per-leg quotes) to place the atomic
    # two-leg close. Identity when no combos are present.
    return {"journal": journal_payload, "positions": positions}


# ---------------------------------------------------------------------------
# Node 3: route_exit_or_hold
# ---------------------------------------------------------------------------

def _route_combo_close(
    *,
    dec: dict,
    pos: dict,
    trade_id: int,
    thesis_id: int | None,
    run_id: str,
    trigger: str,
    broker_order_rows: dict[str, dict],
    moomoo_available: bool,
) -> str:
    """Place an atomic two-leg close for one combo decision.

    Returns ``"pending"`` when a pending combo state now tracks live broker
    orders, else ``"failed"`` (deferred — detect re-fires next tick).

    Invariants (fail-closed, per M1-0.1):
      * the journal row is NEVER closed here — settlement happens in
        fill_confirm once BOTH legs confirm;
      * a pending-state write failure never falls back to a single-leg
        journal close (it can't represent two legs) — sev-2 alert, row
        stays OPEN, the live orders surface via next tick's orphan check;
      * any live BUY on the short leg / SELL on the long leg defers the
        close one tick (orphan adoption is not extended to combos in v1).
    """
    from trading_agent.exits.fill_confirm import (
        BUY_CROSS_BUF,
        SELL_CROSS_BUF,
        build_pending_combo_state,
        find_adoptable_close_order,
        write_pending_close,
    )

    meta = pos["combo"]
    symbol = pos.get("symbol") or meta.short_leg
    action = dec.get("action", "")
    reason = dec.get("reason", "")
    legs = pos.get("combo_legs") or {}
    short_q = legs.get("short") or {}
    long_q = legs.get("long") or {}
    full_qty = int(float(pos.get("qty") or 0))
    close_units = int(dec.get("close_units") or full_qty)
    is_trim = 0 < close_units < full_qty

    # Pre-placement safety: a live close order on either leg means a previous
    # run placed this close and died before the pending-state write. Defer
    # one tick fail-closed rather than double-placing.
    live_short = find_adoptable_close_order(
        broker_order_rows, meta.short_leg, "BUY")
    live_long = find_adoptable_close_order(
        broker_order_rows, meta.long_leg, "SELL")
    if live_short or live_long:
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="combo_close_possible_orphan", severity=1,
            payload={"symbol": symbol, "trade_id": trade_id,
                     "live_short_close": live_short and live_short[0],
                     "live_long_close": live_long and live_long[0],
                     "note": "live close order on a combo leg — deferring "
                             "one tick (no combo orphan adoption in v1)"},
        )
        return "failed"

    # Per-leg marketable limits toward the executable side.
    short_base = short_q.get("ask") or short_q.get("mark")
    long_base = long_q.get("bid") or long_q.get("mark")
    if not short_base or not long_base or not moomoo_available:
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="exit_order_failed", severity=1,
            payload={"symbol": symbol, "action": action, "reason": reason,
                     "combo": True, "detail": "no leg quotes or moomoo down"},
        )
        return "failed"
    short_price = round(float(short_base) * BUY_CROSS_BUF, 2)
    long_price = round(float(long_base) * SELL_CROSS_BUF, 2)

    try:
        from trading_agent.mcp_servers.moomoo.server import (
            close_paper_option_combo,
        )
        result = close_paper_option_combo(
            short_leg_symbol=meta.short_leg,
            long_leg_symbol=meta.long_leg,
            contracts=close_units,
            short_price=short_price,
            long_price=long_price,
            trade_id=int(trade_id),
        )
    except Exception as e:
        log.warning("[route_exit_or_hold] combo close %s failed: %s", symbol, e)
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="exit_order_failed", severity=1,
            payload={"symbol": symbol, "action": action, "reason": reason,
                     "combo": True, "detail": repr(e)},
        )
        return "failed"

    if result.get("combo_close_blocked") or result.get("order_blocked"):
        # The close tool's own gate refused a journal-backed exit — always a
        # bug worth paging on (closes must never be trapped).
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="exit_order_blocked_by_guard", severity=2,
            payload={"symbol": symbol, "trade_id": trade_id, "combo": True,
                     "reason": result.get("reason")},
        )
        return "failed"

    cc = result.get("combo_close")
    partial = (cc == "partial") or bool(result.get("partial"))
    if cc is not True and not partial:
        # Nothing live at the broker (leg1 rejected, or clean rollback of a
        # working BTC order) — retry next tick.
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="exit_order_failed", severity=1,
            payload={"symbol": symbol, "action": action, "reason": reason,
                     "combo": True,
                     "rollback": bool(result.get("combo_rollback")),
                     "detail": result.get("reason")},
        )
        return "failed"

    short_oid = result.get("short_close_order_id") or None
    long_oid = (None if partial
                else (result.get("long_close_order_id") or None))
    state = build_pending_combo_state(
        symbol=symbol,
        action=action,
        reason=reason,
        units=close_units,
        settle_mode="trim" if is_trim else "full",
        net_credit=meta.net_credit,
        short_leg=meta.short_leg,
        long_leg=meta.long_leg,
        short_order_id=short_oid,
        long_order_id=long_oid,
        short_price=short_price,
        long_price=long_price,
        short_bid=short_q.get("bid"), short_ask=short_q.get("ask"),
        long_bid=long_q.get("bid"), long_ask=long_q.get("ask"),
        thesis_id=thesis_id,
        source="postgres",
        # Per-leg broker qty at placement — the vanished-order branch
        # reconciles TRIMS by quantity (presence alone can't detect a late
        # invisible fill on a partial close and would over-trim).
        broker_qty_before=full_qty,
    )
    if not write_pending_close(trade_id, state):
        # The legacy "immediate cost-honest close" fallback is NOT valid for
        # combos (it can't represent two legs). The broker orders are live;
        # next tick's orphan check defers and the operator is paged. NEVER
        # single-leg-close a combo row.
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="combo_pending_write_failed", severity=2,
            payload={"symbol": symbol, "trade_id": trade_id,
                     "short_close_order_id": short_oid,
                     "long_close_order_id": long_oid,
                     "risk": "two live close orders untracked — journal row "
                             "left OPEN; reconcile manually"},
        )
        return "failed"

    if partial:
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="combo_close_partial_leg", severity=2,
            payload={"symbol": symbol, "trade_id": trade_id,
                     "short_close_order_id": short_oid,
                     "note": "short leg buy-back live/dealt, long-leg close "
                             "not placed — fill_confirm retries the wing"},
        )

    # Trims stamp regime_downsized_at_label immediately (the stamp IS the
    # idempotency guard; a lost stamp would re-trim next tick).
    if is_trim and action == "EXIT_REGIME_DOWNSIZE":
        try:
            from trading_agent.store.postgres import cursor
            with cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_trades
                    SET regime_downsized_at_label = %s
                    WHERE id = %s AND outcome = 'OPEN'
                    """,
                    (dec.get("regime_label"), trade_id),
                )
        except Exception as e:
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="downsize_guard_unpersisted", severity=1,
                payload={"symbol": symbol, "trade_id": trade_id,
                         "regime_label": dec.get("regime_label"),
                         "error": repr(e),
                         "risk": "next tick may re-trim 50% — stamp "
                                 "regime_downsized_at_label manually"},
            )

    emit(
        run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
        event_type="exit_order_placed",
        payload={
            "symbol": symbol, "action": action, "qty": close_units,
            "combo": True, "settle_mode": "trim" if is_trim else "full",
            "short_close_order_id": short_oid,
            "long_close_order_id": long_oid,
            "short_limit": short_price, "long_limit": long_price,
            "reason": reason, "trade_id": trade_id,
        },
    )
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="trades",
            title=f"EXIT ORDER {action.replace('EXIT_', '')} — "
                  f"{_bare_ticker(symbol)} vertical",
            body=(
                f"Spread: short {meta.short_leg} / long {meta.long_leg}\n"
                f"Units: {close_units} ({'trim' if is_trim else 'full'})\n"
                f"BTC short @ {short_price:.2f}, STC long @ {long_price:.2f}\n"
                f"Awaiting broker fills — P&L records when BOTH legs confirm\n"
                f"Reason: {reason}"
            ),
            priority=4,
            tags=["rotating_light", "chart_with_downwards_trend"],
        )
    except Exception as e:
        log.warning("[route_exit_or_hold] combo ntfy failed: %s", e)
    return "pending"


def route_exit_or_hold(state: TradingGraphState) -> dict:
    """Execute paper close orders for any EXIT_* decisions.

    For each decision where action != 'HOLD':
      - Places a paper close order via moomoo (market sell, qty * factor)
      - Closes the journal_trade row via journal-mcp close_trade
      - Sends ntfy notification on the 'trades' topic

    Returns empty dict; all side-effects are the work.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[intraday/route_exit_or_hold] run_id=%s", run_id)

    journal = state.get("journal") or {}
    decisions: list[dict] = journal.get("exit_decisions") or []
    # stamp_regime_label rides on a combo qty==1 downsize SKIP (action HOLD):
    # no order, but route must still stamp regime_downsized_at_label so the
    # skip doesn't re-fire every tick — include those in the work list.
    exits = [d for d in decisions
             if d.get("action", "HOLD") != "HOLD" or d.get("stamp_regime_label")]

    if not exits:
        emit(run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
             event_type="all_held", payload={"n_positions": len(decisions)})
        return {}

    positions_by_symbol: dict[str, dict] = {
        p["symbol"]: p for p in (state.get("positions") or []) if p.get("symbol")
    }

    try:
        from trading_agent.mcp_servers.moomoo.server import (
            place_paper_option_order,
            place_paper_order,
        )
        moomoo_available = True
    except Exception as e:
        log.error("[route_exit_or_hold] moomoo import failed: %s", e)
        moomoo_available = False

    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_available = True
    except Exception:
        ntfy_available = False

    # Build a one-shot map of journal trade_ids by symbol so we never close
    # a position the journal doesn't know about (i.e. user-placed manual fills).
    # Postgres journal_trades is the SOLE source: autonomous fills only write
    # there (persist_trade_event); the Phase-1 SQLite mirror is perpetually
    # empty for them, so consulting it first could only resolve the wrong row
    # (e.g. an operator manual fill shadowing the agent's own trade).
    # thesis_id is included so close_thesis can fire after a successful exit —
    # otherwise the thesis stays status='open' forever and creates a zombie row
    # that blocks /eod-review hygiene checks.
    # None means the store could not be read — UNKNOWN, never "no trades":
    # resolving exits against an empty map would misclassify every position
    # as manual and skip the very closes that reduce risk.
    journal_trade_id_by_symbol: dict[str, tuple[int, int | None]] | None = {}
    try:
        from trading_agent.store.postgres import get_open_journal_trades
        rows = get_open_journal_trades()
        if rows is None:
            journal_trade_id_by_symbol = None
        else:
            for row in rows:
                sym = row.get("symbol")
                tid = row.get("trade_id")
                th_id = row.get("thesis_id")
                if sym and tid is not None:
                    # Newest-first ordering + setdefault = newest OPEN row
                    # wins, matching the pre-port resolver semantics (a
                    # phantom row must not receive the real trade's close).
                    journal_trade_id_by_symbol.setdefault(
                        str(sym),
                        (int(tid), int(th_id) if th_id is not None else None),
                    )
    except Exception as e:
        log.warning("[route_exit_or_hold] postgres journal lookup failed: %s", e)
        journal_trade_id_by_symbol = None

    if journal_trade_id_by_symbol is None:
        log.warning(
            "[route_exit_or_hold] journal unreadable — failing closed, "
            "all %d exits deferred one tick", len(exits),
        )
        emit(
            run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
            event_type="exit_pass_deferred_journal_unreadable", severity=2,
            payload={"n_exits": len(exits),
                     "symbols": [d["symbol"] for d in exits]},
        )
        return {}

    # In-flight closes (pending fill confirmation) — placing a second close
    # order for the same trade would over-close the position at the broker.
    # The lookup returns None on store error; that means "unknown", and
    # unknown must FAIL CLOSED (skip all exits this tick) — forgetting an
    # in-flight close and placing a second one is unrecoverable, a one-tick
    # delay is not.
    try:
        from trading_agent.exits.fill_confirm import (
            find_adoptable_close_order,
            pending_close_trade_ids_pg,
        )
        pending_ids_pg: set[int] | None = pending_close_trade_ids_pg()
    except Exception as e:
        log.warning("[route_exit_or_hold] pending-close lookup failed: %s", e)
        find_adoptable_close_order = None  # type: ignore[assignment]
        pending_ids_pg = None

    # Live broker orders, for orphan adoption: a crash between a previous
    # run's order placement and its pending-state write leaves a resting
    # close order nothing tracks — adopting it beats double-placing.
    broker_order_rows: dict[str, dict] = {}
    if moomoo_available:
        try:
            from trading_agent.mcp_servers.moomoo.server import get_orders
            for _r in (get_orders().get("rows") or []):
                _oid = str(_r.get("order_id") or _r.get("orderID") or "").strip()
                if _oid:
                    broker_order_rows[_oid] = _r
        except Exception as e:
            log.warning("[route_exit_or_hold] broker order scan failed: %s", e)

    closed_symbols: list[str] = []
    pending_symbols: list[str] = []
    failed_symbols: list[str] = []
    skipped_no_journal: list[str] = []

    for dec in exits:
        symbol = dec["symbol"]
        action = dec["action"]
        qty_factor = float(dec.get("exit_qty_factor") or 1.0)
        reason = dec.get("reason", "")
        dec_regime_label = dec.get("regime_label")

        pos = positions_by_symbol.get(symbol)
        if not pos:
            log.warning("[route_exit_or_hold] position for %s not in state — skip", symbol)
            failed_symbols.append(symbol)
            continue

        # Safety guard — only close positions the journal knows about. Manual
        # broker positions without a thesis must be left alone for the operator
        # to handle. The exit-monitor LLM may flag them, but this node refuses.
        resolved = journal_trade_id_by_symbol.get(symbol)
        if resolved is None:
            log.warning(
                "[route_exit_or_hold] %s has no journal trade — refusing to close (manual position)",
                symbol,
            )
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="exit_skipped_no_journal",
                severity=1,
                payload={"symbol": symbol, "action": action, "reason": reason},
            )
            skipped_no_journal.append(symbol)
            continue
        trade_id, thesis_id = resolved

        # Combo stamp-only path: a qty==1 EXIT_REGIME_DOWNSIZE was skipped
        # (whole units only — half a spread cannot trim). Stamp the guard
        # column WITHOUT an order; the stamp is the once-per-episode dedup.
        if dec.get("stamp_regime_label") and action == "HOLD":
            try:
                from trading_agent.store.postgres import cursor
                with cursor() as cur:
                    cur.execute(
                        """
                        UPDATE journal_trades
                        SET regime_downsized_at_label = %s
                        WHERE id = %s AND outcome = 'OPEN'
                        """,
                        (dec["stamp_regime_label"], trade_id),
                    )
                emit(
                    run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                    event_type="combo_trim_skipped_single_unit", severity=1,
                    payload={"symbol": symbol, "trade_id": trade_id,
                             "regime_label": dec["stamp_regime_label"],
                             "reason": reason,
                             "note": "1-lot vertical cannot trim 50% in whole "
                                     "units — skipped, guard stamped"},
                )
            except Exception as e:
                # No order was placed — a failed stamp just retries next
                # tick. Nothing at the broker can double-trim.
                log.warning(
                    "[route_exit_or_hold] combo skip-stamp %s failed: %s",
                    symbol, e,
                )
            continue

        if pending_ids_pg is None:
            log.warning(
                "[route_exit_or_hold] %s: pending-close state unreadable — "
                "failing closed, exit deferred one tick",
                symbol,
            )
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="exit_deferred_pending_unknown", severity=1,
                payload={"symbol": symbol, "store": "postgres"},
            )
            failed_symbols.append(symbol)
            continue
        if trade_id in pending_ids_pg:
            log.info(
                "[route_exit_or_hold] %s trade %s already has an exit in "
                "flight — awaiting fill confirmation, skipping",
                symbol, trade_id,
            )
            pending_symbols.append(symbol)
            continue

        # Combo close path: every trigger fired on a combo (profit-take,
        # 21-DTE, P0b trim, stop/thesis-broken) goes through the atomic
        # two-leg close tool — never the single-leg placement below. Gated
        # on the POSITION's meta (not just the decision flag) so a combo can
        # never leak into single-leg placement.
        if pos.get("combo") is not None:
            status = _route_combo_close(
                dec=dec, pos=pos, trade_id=trade_id, thesis_id=thesis_id,
                run_id=run_id, trigger=trigger,
                broker_order_rows=broker_order_rows,
                moomoo_available=moomoo_available,
            )
            if status == "pending":
                pending_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
            continue

        full_qty = float(pos.get("qty") or 0)
        close_qty = max(1, round(full_qty * qty_factor)) if qty_factor < 1.0 else full_qty
        if close_qty <= 0:
            failed_symbols.append(symbol)
            continue

        asset_type = pos.get("asset_type", "STK")
        # Close direction depends on what we're long/short. A SHORT option
        # position is closed by BUYING it back; a LONG position is closed
        # by SELLING. Plan direction lives in the enrichment.
        plan_dir = "LONG"
        ep = pos.get("exit_plan")
        if isinstance(ep, dict):
            plan_dir = ep.get("direction") or "LONG"
        side_close = "BUY" if plan_dir == "SHORT" else "SELL"

        # Backstop (M1-0.1 degrade contract): a SHORT-direction plan with no
        # combo meta is a degraded combo row (single-leg SELL-to-open is
        # hard-blocked, so no legitimate single-leg SHORT exists). Placing
        # the single-leg BUY-back would leg the vertical apart and orphan
        # the wing — refuse loudly instead. detect already forces HOLD for
        # this signature; this guard catches any path that slips through.
        if plan_dir == "SHORT" and pos.get("combo") is None:
            log.error(
                "[route_exit_or_hold] %s: SHORT-direction plan with no combo "
                "meta — refusing single-leg close (would orphan the wing)",
                symbol,
            )
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="combo_single_leg_close_refused", severity=2,
                payload={"symbol": symbol, "trade_id": trade_id,
                         "action": action, "reason": reason,
                         "note": "degraded combo row — repair the combo "
                                 "payload; the engine will not leg it apart"},
            )
            failed_symbols.append(symbol)
            continue
        placed_order_id: str | None = None

        # Marketable limit toward the EXECUTABLE side. The old code priced at
        # the last mark — for a SELL that's ~half a spread above what the
        # market pays, so journaled exits were systematically inflated and
        # wide-spread orders could rest unfilled while the journal said
        # "closed". Mirrors the entry side's fresh-ask×1.02 convention.
        from trading_agent.exits.fill_confirm import (
            BUY_CROSS_BUF,
            SELL_CROSS_BUF,
            build_pending_state,
            write_pending_close,
        )
        mark_px = float(pos.get("mark") or pos.get("entry_price") or 0)
        quoted_bid = pos.get("bid")
        quoted_ask = pos.get("ask")
        if side_close == "SELL":
            base_px = float(quoted_bid) if quoted_bid else mark_px
            exit_price = round(base_px * SELL_CROSS_BUF, 2)
        else:
            base_px = float(quoted_ask) if quoted_ask else mark_px
            exit_price = round(base_px * BUY_CROSS_BUF, 2)

        # Any sub-1.0 factor is a partial close — EXIT_TARGET scale-out rungs
        # AND EXIT_REGIME_DOWNSIZE (0.5). Gating on EXIT_TARGET alone would
        # mis-route a regime downsize as a FULL close (close_trade + close_thesis).
        is_partial = (0.0 < qty_factor < 1.0)

        # Orphan adoption (full closes only): a live same-symbol same-side
        # order means a previous run placed this close and died before the
        # pending-state write — track THAT order instead of doubling it.
        adopted = None
        if (not is_partial and broker_order_rows
                and find_adoptable_close_order is not None):
            adopted = find_adoptable_close_order(
                broker_order_rows, symbol, side_close)
            if adopted:
                placed_order_id = adopted[0]
                log.warning(
                    "[route_exit_or_hold] %s adopting orphan close order %s "
                    "(crash-window leftover) — no new order placed",
                    symbol, placed_order_id,
                )
                emit(
                    run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                    event_type="exit_order_adopted", severity=1,
                    payload={"symbol": symbol, "order_id": placed_order_id},
                )

        if moomoo_available and not adopted:
            try:
                if asset_type == "OPT":
                    result = place_paper_option_order(
                        option_symbol=symbol,
                        side=side_close,
                        contracts=int(close_qty),
                        price=exit_price,
                        thesis_id=int(trade_id),  # use journal trade_id as link
                        strategy_label=pos.get("strategy_label"),
                        delta=pos.get("delta"),
                        dte=pos.get("dte"),
                    )
                else:
                    result = place_paper_order(
                        symbol=symbol,
                        side=side_close,
                        qty=int(close_qty),
                        price=exit_price,
                        thesis_id=int(trade_id),
                        order_type="NORMAL",
                    )
                # Both place_paper_order and place_paper_option_order return
                # {thesis_id, ..., rows: [{order_id, order_status, ...}, ...]}
                # — NOT a top-level "order_id". Pattern matches trade_nodes.execute_paper_order.
                rows = result.get("rows") or []
                placed_order_id = None
                if rows:
                    placed_order_id = (
                        str(rows[0].get("order_id") or rows[0].get("orderID") or "").strip()
                        or None
                    )
                if not placed_order_id and result.get("virtual_fill_suggested"):
                    # Moomoo rejected paper order; fall through to virtual close in journal.
                    log.warning(
                        "[route_exit_or_hold] %s virtual fallback (broker rejected): %s",
                        symbol, result.get("reason"),
                    )
                if not placed_order_id and result.get("order_blocked"):
                    # Server-side risk gate refused the close — should not
                    # happen for a journal-backed exit (closes are exempt);
                    # surface loudly so the gap gets investigated.
                    log.error(
                        "[route_exit_or_hold] %s close REFUSED by order guard: %s %s",
                        symbol, result.get("reason"), result.get("violations"),
                    )
                if placed_order_id:
                    log.info(
                        "[route_exit_or_hold] %s order placed %s qty=%s order_id=%s",
                        asset_type, symbol, close_qty, placed_order_id,
                    )
            except Exception as e:
                log.warning("[route_exit_or_hold] order placement %s failed: %s", symbol, e)
                placed_order_id = None

        if placed_order_id is None:
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="exit_order_failed",
                severity=1,
                payload={"symbol": symbol, "action": action, "reason": reason},
            )
            failed_symbols.append(symbol)
            continue

        # ------------------------------------------------------------------
        # Partial vs full close routing.
        #
        # A scale-out rung returns action=EXIT_TARGET with 0 < factor < 1.
        # That's a PARTIAL close — the broker order trimmed the position
        # but the journal_trades row must stay outcome='OPEN' so the
        # residual quantity keeps being monitored. We only need to bump
        # scale_rungs_taken so the same rung doesn't fire next tick.
        #
        # Full closes (factor=1.0 OR any non-EXIT_TARGET action) flow
        # through close_trade + close_thesis as before.
        # ------------------------------------------------------------------
        closed_ok = False
        exit_pending = False
        if is_partial:
            try:
                from trading_agent.store.postgres import cursor
                with cursor() as cur:
                    if action == "EXIT_REGIME_DOWNSIZE":
                        # Stamp the guard so the downsize won't re-fire every
                        # tick while this degrade label persists. (We do NOT
                        # bump scale_rungs_taken — this isn't a ladder rung.)
                        cur.execute(
                            """
                            UPDATE journal_trades
                            SET regime_downsized_at_label = %s
                            WHERE id = %s AND outcome = 'OPEN'
                            """,
                            (dec_regime_label, trade_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE journal_trades
                            SET scale_rungs_taken = COALESCE(scale_rungs_taken, 0) + 1
                            WHERE id = %s AND outcome = 'OPEN'
                            """,
                            (trade_id,),
                        )
                closed_ok = True
                log.info(
                    "[route_exit_or_hold] %s partial close (factor=%.2f, %s) — "
                    "trade %s stays OPEN",
                    symbol, qty_factor, action, trade_id,
                )
            except Exception as e:
                log.warning(
                    "[route_exit_or_hold] partial-close rung increment "
                    "%s failed: %s", symbol, e,
                )
                if action == "EXIT_REGIME_DOWNSIZE":
                    # The stamp IS the idempotency guard. The broker already
                    # trimmed, but with regime_downsized_at_label unpersisted
                    # the NEXT tick would trim ANOTHER 50%. Do NOT mark this a
                    # clean success — surface a sev-1 alert so the operator
                    # fixes the unpersisted guard before the position bleeds out.
                    closed_ok = False
                    emit(
                        run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                        event_type="downsize_guard_unpersisted", severity=1,
                        payload={"symbol": symbol, "trade_id": trade_id,
                                 "regime_label": dec_regime_label,
                                 "error": repr(e),
                                 "risk": "next tick may re-trim 50% — stamp "
                                         "regime_downsized_at_label manually"},
                    )
                else:
                    # Scale rung: best-effort — broker already trimmed and the
                    # rung counter is advisory; next tick re-evaluates safely.
                    closed_ok = True
        else:
            # Full close — the journal is NOT closed here anymore. The close
            # order just placed must be CONFIRMED at the broker's dealt price
            # before any P&L is recorded; the old place-and-journal-at-mark
            # flow inflated exits by ~half the spread and could mark unfilled
            # positions closed. sync_fill_status finalizes next tick.
            pending = build_pending_state(
                order_id=placed_order_id,
                symbol=symbol,
                action=action,
                reason=reason,
                qty=close_qty,
                side=side_close,
                asset_type=asset_type,
                requested_exit_price=exit_price,
                quoted_bid=float(quoted_bid) if quoted_bid else None,
                quoted_ask=float(quoted_ask) if quoted_ask else None,
                thesis_id=thesis_id,
                source="postgres",
                expected_qty_before_close=full_qty,
            )
            if write_pending_close(trade_id, pending):
                exit_pending = True
                log.info(
                    "[route_exit_or_hold] %s exit order %s pending fill "
                    "confirmation (limit=%.2f, trade_id=%s)",
                    symbol, placed_order_id, exit_price, trade_id,
                )
            else:
                # Pending-state write failed — we can't track the order, so
                # fall back to the legacy immediate close rather than strand
                # the position. Cost-honest: round-trip fees charged, exit at
                # the executable-side limit, and flagged unconfirmed.
                entry_price_j = float(pos.get("entry_price") or 0)
                mult_j = 100 if asset_type == "OPT" else 1
                # SHORT positions invert PnL sign: you collect premium on
                # entry (cash IN), pay to close (cash OUT) → profit = entry - exit.
                if plan_dir == "SHORT":
                    gross_j = (entry_price_j - exit_price) * close_qty * mult_j
                else:
                    gross_j = (exit_price - entry_price_j) * close_qty * mult_j
                try:
                    from trading_agent.execution_costs import fees_per_side
                    pnl_j = round(gross_j - 2 * fees_per_side(close_qty, asset_type), 2)
                except Exception as e:
                    log.warning("[route_exit_or_hold] fee model failed: %s", e)
                    pnl_j = round(gross_j, 2)
                from trading_agent.learning.outcome import outcome_for_pnl
                outcome_j = outcome_for_pnl(pnl_j)
                close_reason_j = f"{action} [exit fill UNCONFIRMED — pending-state write failed]"

                # Postgres is the sole journal for autonomous trades — the
                # resolver above only yields Postgres rows.
                try:
                    from trading_agent.store.postgres import cursor
                    with cursor() as cur:
                        cur.execute(
                            """
                            UPDATE journal_trades
                            SET exit_price = %s,
                                outcome = %s,
                                closed_at = COALESCE(closed_at, NOW()),
                                close_reason = COALESCE(close_reason, %s)
                            WHERE id = %s AND outcome = 'OPEN'
                            """,
                            (exit_price, outcome_j, close_reason_j, trade_id),
                        )
                    closed_ok = True
                    log.info("[route_exit_or_hold] %s journal closed via postgres trade_id=%s", symbol, trade_id)
                except Exception as e:
                    log.warning("[route_exit_or_hold] postgres close %s failed: %s", symbol, e)

        # Thesis flip (only on full closes — partial closes leave it open
        # so subsequent rungs can still close out cleanly). Without this,
        # theses pile up status='open' forever (the May 2026 AAPL/XLE/
        # GLD/AMZN cleanup is the canonical example). Postgres only — the
        # thesis row was written alongside the Postgres trade row.
        if closed_ok and not is_partial and thesis_id is not None:
            try:
                from trading_agent.store.postgres import cursor
                with cursor() as cur:
                    # Include 'thesis_broken' so an EXIT_EVENT full close still
                    # records the terminal 'triggered' status (the trade
                    # executed and resolved), not just plain 'open' theses.
                    cur.execute(
                        "UPDATE journal_theses SET status='triggered' "
                        "WHERE id = %s AND status IN ('open','thesis_broken')",
                        (thesis_id,),
                    )
                log.info(
                    "[route_exit_or_hold] thesis %s closed via postgres",
                    thesis_id,
                )
            except Exception as e:
                log.warning(
                    "[route_exit_or_hold] postgres close_thesis(%s) failed: %s",
                    thesis_id, e,
                )

        if exit_pending:
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="exit_order_placed",
                payload={
                    "symbol": symbol,
                    "action": action,
                    "qty": close_qty,
                    "limit_price": exit_price,
                    "reason": reason,
                    "order_id": placed_order_id,
                    "trade_id": trade_id,
                },
            )
            pending_symbols.append(symbol)
        else:
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="position_closed",
                payload={
                    "symbol": symbol,
                    "action": action,
                    "qty_closed": close_qty,
                    "exit_price": exit_price,
                    "reason": reason,
                    "order_id": placed_order_id,
                    "fill_confirmed": False,
                },
            )
            closed_symbols.append(symbol)

        if ntfy_available:
            try:
                entry = float(pos.get("entry_price") or 0)
                # SHORT inverts sign — see PnL note above.
                if plan_dir == "SHORT":
                    pnl_pct = ((entry - exit_price) / entry * 100) if entry else 0.0
                else:
                    pnl_pct = ((exit_price - entry) / entry * 100) if entry else 0.0
                verb = "EXIT ORDER" if exit_pending else "EXIT"
                ntfy_send(
                    topic="trades",
                    title=f"{verb} {action.replace('EXIT_', '')} — {_bare_ticker(symbol)}",
                    body=(
                        f"Symbol: {symbol}\n"
                        f"Qty: {int(close_qty)}\n"
                        f"Limit @ {exit_price:.2f}  Entry @ {entry:.2f}  "
                        f"(~{pnl_pct:+.1f}% to limit)\n"
                        + ("Awaiting broker fill — P&L records on confirmation\n"
                           if exit_pending else "")
                        + f"Reason: {reason}"
                    ),
                    priority=4,
                    tags=["rotating_light", "chart_with_downwards_trend"],
                )
            except Exception as e:
                log.warning("[route_exit_or_hold] ntfy failed: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
        event_type="exit_pass_complete",
        payload={
            "closed": closed_symbols,
            "pending_fill": pending_symbols,
            "failed": failed_symbols,
            "held": [d["symbol"] for d in decisions if d.get("action") == "HOLD"],
        },
    )

    # Audibility watchdogs — run every intraday tick (5-min cadence) so
    # dispatch silent-die / LLM schema-violation alerts fire within
    # ~5 min instead of ~60 min (healthcheck cadence). Watchdogs are
    # idempotent — they have their own per-event cooldowns, so running
    # them twice (intraday + healthcheck) doesn't double-alert.
    try:
        from trading_agent.graph.nodes.health_nodes import (
            _check_dispatch_silent_die,
            _check_llm_schema_violations,
        )
        _check_dispatch_silent_die(run_id, trigger)
        _check_llm_schema_violations(run_id, trigger)
    except Exception as e:
        log.warning("[route_exit_or_hold] watchdog failed: %s", e)
    return {}


__all__ = [
    "refresh_quotes_and_greeks",
    "detect_exit_triggers",
    "route_exit_or_hold",
]
