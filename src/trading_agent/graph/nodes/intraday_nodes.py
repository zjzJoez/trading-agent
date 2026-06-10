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

    try:
        from trading_agent.store.postgres import cursor
    except Exception as e:
        log.warning("[sync_fill_status] postgres import failed: %s", e)
        return {}

    # 1. OPEN journal trades that have a broker order id
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, symbol, broker_order_id, entry_price
                FROM journal_trades
                WHERE outcome = 'OPEN' AND broker_order_id IS NOT NULL
                  AND broker_order_id <> ''
                """,
            )
            open_trades = cur.fetchall()
    except Exception as e:
        log.warning("[sync_fill_status] open-trades read failed: %s", e)
        return {}
    if not open_trades:
        return {}

    # 2. Live broker order map: order_id → (status, dealt_qty, dealt_avg_price)
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

    synced_filled = 0
    marked_unfilled = 0
    for trade_id, symbol, broker_order_id, entry_price in open_trades:
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
    return {}


def _no_plan_alerted_today(symbol: str) -> bool:
    """True iff we already fired position_no_exit_plan for this symbol today.

    Per-symbol per-day dedup — without this, every 15-min intraday tick
    spams the same alert for a persistent manual position (6/1 incident:
    22 events on one QQQ contract before operator noticed).
    """
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM agent_events
                WHERE event_type = 'position_no_exit_plan'
                  AND ts > DATE_TRUNC('day', NOW())
                  AND payload->>'symbol' = %s
                LIMIT 1
                """,
                (symbol,),
            )
            return cur.fetchone() is not None
    except Exception:
        # On DB error, default to "yes we alerted" so we don't spam
        # if the DB is degraded.
        return True


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
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT t.symbol, t.id AS trade_id, t.thesis_id,
                       t.stop, t.target, t.exit_plan,
                       t.mfe_so_far, t.opened_at,
                       COALESCE(t.scale_rungs_taken, 0) AS scale_rungs_taken,
                       th.direction, th.thesis_text, th.invalidation
                FROM journal_trades t
                LEFT JOIN journal_theses th ON th.id = t.thesis_id
                WHERE t.outcome = 'OPEN'
                """,
            )
            for row in cur.fetchall():
                (symbol, trade_id, thesis_id, stop, target, exit_plan,
                 mfe_so_far, opened_at, scale_rungs_taken,
                 direction, thesis_text, invalidation) = row
                if not symbol:
                    continue
                out[str(symbol)] = {
                    "trade_id": int(trade_id) if trade_id is not None else None,
                    "thesis_id": int(thesis_id) if thesis_id is not None else None,
                    "stop": float(stop) if stop is not None else None,
                    "target": float(target) if target is not None else None,
                    # exit_plan is JSONB; psycopg returns a dict directly.
                    "exit_plan": exit_plan if isinstance(exit_plan, dict) else None,
                    "mfe_so_far": float(mfe_so_far) if mfe_so_far is not None else None,
                    "opened_at": opened_at,
                    "scale_rungs_taken": int(scale_rungs_taken or 0),
                    "direction": direction,
                    "thesis_text": thesis_text,
                    "invalidation": invalidation,
                }
    except Exception as e:
        log.warning("[detect_exit_triggers] journal enrichment query failed: %s", e)
    return out


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
    decisions: list[dict] = []
    hold_count = 0
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
        )

        if decision is None:
            action = "HOLD"
            qty_factor = 0.0
            reason = "no_rule_triggered"
        else:
            action = decision.action
            qty_factor = decision.exit_qty_factor
            reason = decision.reason

        # Sanity guard: a partial-exit signal on a 1-contract option position
        # is physically impossible (options don't fractionate). The downstream
        # route_exit_or_hold would `max(1, round(1 * 0.5)) = 1` and force a
        # FULL close, contradicting the partial signal. Demote to HOLD so
        # the next tick re-evaluates (likely hits target if mark stays up).
        full_qty = float(pos.get("qty") or 0)
        if action != "HOLD" and 0.0 < qty_factor < 1.0 and full_qty <= 1:
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
    return {"journal": journal_payload}


# ---------------------------------------------------------------------------
# Node 3: route_exit_or_hold
# ---------------------------------------------------------------------------

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
    exits = [d for d in decisions if d.get("action", "HOLD") != "HOLD"]

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
    # Multi-source resolver: SQLite journal MCP first (canonical), then
    # Postgres journal_trades as a fallback when SQLite is empty / out of sync.
    # Value is (trade_id, thesis_id, source) where source ∈ {"sqlite", "postgres"}.
    # thesis_id is included so close_thesis can fire after a successful exit —
    # otherwise the thesis stays status='open' forever and creates a zombie row
    # that blocks /eod-review hygiene checks.
    journal_trade_id_by_symbol: dict[str, tuple[int, int | None, str]] = {}

    # Source 1: SQLite journal MCP
    journal_close_thesis = None  # type: ignore[assignment]
    try:
        from trading_agent.mcp_servers.journal.server import (
            close_thesis as journal_close_thesis,  # noqa: F811 - assignment from import
            close_trade as journal_close_trade,
            get_open_positions_with_thesis,
        )
        for row in (get_open_positions_with_thesis().get("rows") or []):
            sym = row.get("symbol")
            tid = row.get("trade_id")
            th_id = row.get("thesis_id")
            if sym and tid:
                journal_trade_id_by_symbol[str(sym)] = (
                    int(tid),
                    int(th_id) if th_id is not None else None,
                    "sqlite",
                )
    except Exception as e:
        log.warning("[route_exit_or_hold] sqlite journal lookup failed: %s", e)
        journal_close_trade = None  # type: ignore[assignment]

    # Source 2: Postgres journal_trades (fallback for symbols not in SQLite)
    needed_symbols = {d["symbol"] for d in exits} - set(journal_trade_id_by_symbol)
    if needed_symbols:
        try:
            from trading_agent.store.postgres import cursor
            with cursor() as cur:
                cur.execute(
                    """
                    SELECT id, symbol, thesis_id FROM journal_trades
                    WHERE outcome = 'OPEN' AND symbol = ANY(%s)
                    ORDER BY opened_at DESC
                    """,
                    (list(needed_symbols),),
                )
                for tid, sym, th_id in cur.fetchall():
                    journal_trade_id_by_symbol.setdefault(
                        str(sym),
                        (
                            int(tid),
                            int(th_id) if th_id is not None else None,
                            "postgres",
                        ),
                    )
        except Exception as e:
            log.warning("[route_exit_or_hold] postgres journal fallback failed: %s", e)

    closed_symbols: list[str] = []
    failed_symbols: list[str] = []
    skipped_no_journal: list[str] = []

    for dec in exits:
        symbol = dec["symbol"]
        action = dec["action"]
        qty_factor = float(dec.get("exit_qty_factor") or 1.0)
        reason = dec.get("reason", "")

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
        trade_id, thesis_id, trade_id_source = resolved

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
        placed_order_id: str | None = None
        exit_price = float(pos.get("mark") or pos.get("entry_price") or 0)

        if moomoo_available:
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
        is_partial = (action == "EXIT_TARGET" and 0.0 < qty_factor < 1.0)

        closed_ok = False
        if is_partial:
            try:
                from trading_agent.store.postgres import cursor
                with cursor() as cur:
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
                    "[route_exit_or_hold] %s partial close (factor=%.2f) — "
                    "rung incremented; trade %s stays OPEN",
                    symbol, qty_factor, trade_id,
                )
            except Exception as e:
                log.warning(
                    "[route_exit_or_hold] partial-close rung increment "
                    "%s failed: %s", symbol, e,
                )
                # Treat as best-effort — broker already trimmed the position.
                # Don't fail the whole loop; the next tick re-evaluates.
                closed_ok = True
        else:
            # Full close — journal close (trade row + thesis row).
            entry_price_j = float(pos.get("entry_price") or 0)
            mult_j = 100 if asset_type == "OPT" else 1
            # SHORT positions invert PnL sign: you collect premium on entry
            # (cash IN), pay to close (cash OUT). So profit = entry - exit.
            if plan_dir == "SHORT":
                pnl_j = round((entry_price_j - exit_price) * close_qty * mult_j, 2)
            else:
                pnl_j = round((exit_price - entry_price_j) * close_qty * mult_j, 2)
            outcome_j = "WIN" if pnl_j > 0 else ("LOSS" if pnl_j < 0 else "SCRATCH")

            if trade_id_source == "sqlite" and journal_close_trade is not None:
                try:
                    journal_close_trade(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        outcome=outcome_j,
                        pnl=pnl_j,
                    )
                    closed_ok = True
                except Exception as e:
                    log.warning("[route_exit_or_hold] sqlite close_trade %s failed: %s", symbol, e)
            if not closed_ok:
                # Either source was postgres, or sqlite path raised — write directly to Postgres.
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
                            (exit_price, outcome_j, action, trade_id),
                        )
                    closed_ok = True
                    log.info("[route_exit_or_hold] %s journal closed via postgres trade_id=%s", symbol, trade_id)
                except Exception as e:
                    log.warning("[route_exit_or_hold] postgres close %s failed: %s", symbol, e)

        # Thesis flip (only on full closes — partial closes leave it open
        # so subsequent rungs can still close out cleanly). Without this,
        # theses pile up status='open' forever (the May 2026 AAPL/XLE/
        # GLD/AMZN cleanup is the canonical example).
        if closed_ok and not is_partial and thesis_id is not None:
            thesis_note = f"trade {trade_id} closed via {action} ({reason})"[:280]
            thesis_closed = False
            # Route the thesis close to the SAME store as the trade close —
            # they're written together at thesis_record time, so a postgres
            # trade implies a postgres thesis. Without this gate, a postgres
            # trade would try sqlite close_thesis first; sqlite MCP returns
            # "ok" without raising even when the row is absent, leaving
            # `thesis_closed=True` and skipping the postgres UPDATE. Result:
            # journal_theses stays status='open' forever and accumulates
            # zombies (the May 2026 AAPL/XLE/GLD/AMZN cleanup pattern).
            if trade_id_source == "sqlite" and journal_close_thesis is not None:
                try:
                    journal_close_thesis(
                        thesis_id=thesis_id,
                        status="triggered",
                        note=thesis_note,
                    )
                    thesis_closed = True
                except Exception as e:
                    log.warning(
                        "[route_exit_or_hold] sqlite close_thesis(%s) failed: %s",
                        thesis_id, e,
                    )
            if not thesis_closed:
                try:
                    from trading_agent.store.postgres import cursor
                    with cursor() as cur:
                        cur.execute(
                            "UPDATE journal_theses SET status='triggered' "
                            "WHERE id = %s AND status='open'",
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
            },
        )

        if ntfy_available:
            try:
                entry = float(pos.get("entry_price") or 0)
                # SHORT inverts sign — see PnL note above.
                if plan_dir == "SHORT":
                    pnl_pct = ((entry - exit_price) / entry * 100) if entry else 0.0
                else:
                    pnl_pct = ((exit_price - entry) / entry * 100) if entry else 0.0
                ntfy_send(
                    topic="trades",
                    title=f"EXIT {action.replace('EXIT_', '')} — {_bare_ticker(symbol)}",
                    body=(
                        f"Symbol: {symbol}\n"
                        f"Qty closed: {int(close_qty)}\n"
                        f"Exit @ {exit_price:.2f}  Entry @ {entry:.2f}  "
                        f"({pnl_pct:+.1f}%)\n"
                        f"Reason: {reason}"
                    ),
                    priority=4,
                    tags=["rotating_light", "chart_with_downwards_trend"],
                )
            except Exception as e:
                log.warning("[route_exit_or_hold] ntfy failed: %s", e)

        closed_symbols.append(symbol)

    emit(
        run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
        event_type="exit_pass_complete",
        payload={
            "closed": closed_symbols,
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
