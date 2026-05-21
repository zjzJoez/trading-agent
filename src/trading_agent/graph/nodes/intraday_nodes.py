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

import json
import logging
from typing import Any

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
                       t.stop, t.target,
                       th.direction, th.thesis_text, th.invalidation
                FROM journal_trades t
                LEFT JOIN journal_theses th ON th.id = t.thesis_id
                WHERE t.outcome = 'OPEN'
                """,
            )
            for row in cur.fetchall():
                symbol, trade_id, thesis_id, stop, target, direction, thesis_text, invalidation = row
                if not symbol:
                    continue
                out[str(symbol)] = {
                    "trade_id": int(trade_id) if trade_id is not None else None,
                    "thesis_id": int(thesis_id) if thesis_id is not None else None,
                    "stop": float(stop) if stop is not None else None,
                    "target": float(target) if target is not None else None,
                    "direction": direction,
                    "thesis_text": thesis_text,
                    "invalidation": invalidation,
                }
    except Exception as e:
        log.warning("[detect_exit_triggers] journal enrichment query failed: %s", e)
    return out


def _thesis_summary_for(pos: dict) -> str:
    """Format a thesis summary string for the exit-monitor LLM prompt.

    Prefers fields already on ``pos`` (populated by
    ``_load_journal_enrichment_by_symbol`` upstream). Falls back to a
    direct ``journal_theses`` query if ``thesis_id`` is present but the
    rich fields aren't — kept for backwards-compat with any future caller
    that doesn't run the enrichment step.
    """
    # Preferred path: enrichment already attached the JOIN result.
    if pos.get("thesis_text") or pos.get("direction"):
        direction = pos.get("direction") or "?"
        text = str(pos.get("thesis_text") or "")[:200]
        inval = pos.get("invalidation") or "n/a"
        return f"direction={direction} | thesis={text} | invalidation={inval}"

    thesis_id = pos.get("thesis_id")
    if not thesis_id:
        return "(no thesis linked)"
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT direction, thesis_text, invalidation FROM journal_theses WHERE id = %s",
                (int(thesis_id),),
            )
            row = cur.fetchone()
        if row:
            return f"direction={row[0]} | thesis={str(row[1])[:200]} | invalidation={row[2] or 'n/a'}"
    except Exception as e:
        log.warning("thesis_summary_for(%s): %s", thesis_id, e)
    return "(thesis not found)"


def _format_exit_prompt(pos: dict, regime: dict, prior_regime_label: str) -> str:
    mark = float(pos.get("mark") or pos.get("entry_price") or 0)
    entry = float(pos.get("entry_price") or 0)
    stop = pos.get("stop")
    target = pos.get("target")
    dte = pos.get("dte")

    prompt_parts = [
        f"position:",
        f"  symbol: {pos.get('symbol')}",
        f"  underlying: {pos.get('underlying')}",
        f"  asset_type: {pos.get('asset_type')}",
        f"  side: {pos.get('side')}",
        f"  qty: {pos.get('qty')}",
        f"  entry_price: {entry}",
        f"  mark: {mark}",
        f"  stop: {stop}",
        f"  target: {target}",
        f"  age_minutes: {pos.get('age_minutes', 0)}",
        f"  delta: {pos.get('delta')}",
        f"  iv: {pos.get('iv')}",
        f"  dte: {dte}",
        f"  unrealized_pnl: {pos.get('unrealized_pnl', 0)}",
        f"  thesis_summary: {_thesis_summary_for(pos)}",
        f"",
        f"current_quote:",
        f"  last: {mark}",
        f"",
        f"regime_state:",
        f"  label: {regime.get('label', 'VOLATILE_TRANSITION')}",
        f"  confidence: {regime.get('confidence', 0):.2f}",
        f"  gate: {json.dumps(regime.get('gate') or {})}",
        f"",
        f"prior_regime_label: {prior_regime_label}",
        f"",
        f"flagged_news_today: []",
        f"",
        f"Respond per your output schema.",
    ]
    return "\n".join(prompt_parts)


def detect_exit_triggers(state: TradingGraphState) -> dict:
    """Call the exit-monitor LLM for each open position.

    Writes decisions into state["journal"]["exit_decisions"] as a list of
    dicts: {symbol, action, exit_qty_factor, reason}.

    Best-effort per position: a failed LLM call defaults to HOLD so the
    position isn't silently abandoned.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[intraday/detect_exit_triggers] run_id=%s", run_id)

    positions: list[dict] = list(state.get("positions") or [])
    if not positions:
        return {}

    regime = state.get("regime") or {}
    # Simple approximation for prior label: not in state yet, use same label
    prior_regime_label = regime.get("label", "VOLATILE_TRANSITION")

    # Enrich each broker position with its journal_trades + journal_theses row
    # (thesis_id, thesis_text, direction, invalidation, stop, target). Without
    # this, _thesis_summary_for sees pos.thesis_id=None and the LLM concludes
    # "no thesis = unvetted exposure → EXIT_CAUTIOUS" — closing the same
    # position we just opened. Empty map ⇒ legacy per-position fallback.
    enrichment_by_symbol = _load_journal_enrichment_by_symbol()
    enriched_count = 0
    if enrichment_by_symbol:
        for pos in positions:
            symbol = pos.get("symbol") or ""
            enrichment = enrichment_by_symbol.get(symbol)
            if not enrichment:
                continue
            # Don't clobber broker-provided fields (qty, mark, etc.) — only
            # fill in keys that aren't already populated. The enrichment
            # keys (trade_id, thesis_id, stop, target, direction, thesis_text,
            # invalidation) don't overlap with broker fields today; the
            # ``k not in pos`` guard makes that invariant explicit so a
            # future field-name collision fails-safe (broker wins).
            for k, v in enrichment.items():
                if v is not None and k not in pos:
                    pos[k] = v
            enriched_count += 1
    log.info(
        "[detect_exit_triggers] enriched %d/%d positions from Postgres journal",
        enriched_count, len(positions),
    )

    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import ExitMonitorOutput
        router = get_router()
    except Exception as e:
        log.error("[detect_exit_triggers] LLM router unavailable: %s", e)
        return {}

    decisions: list[dict] = []
    hold_count = 0
    exit_count = 0

    for pos in positions:
        symbol = pos.get("symbol") or ""
        if not symbol:
            continue
        prompt = _format_exit_prompt(pos, regime, prior_regime_label)
        try:
            res = router.call("exit_monitor", prompt, schema=ExitMonitorOutput, timeout_s=60)
            parsed: ExitMonitorOutput | None = (
                res.parsed if isinstance(res.parsed, ExitMonitorOutput) else None
            )
            if parsed is None:
                action = "HOLD"
                reason = "no_parsed_output"
                qty_factor = 0.0
            else:
                action = parsed.action
                reason = parsed.reason
                qty_factor = parsed.exit_qty_factor
        except Exception as e:
            log.warning("[detect_exit_triggers] symbol=%s LLM failed: %s — defaulting HOLD", symbol, e)
            action = "HOLD"
            reason = f"llm_error: {e}"
            qty_factor = 0.0
            # Tracking event for escalation. One failure is fine (default HOLD
            # is safe), but a *streak* of failures means every position is
            # frozen at HOLD with no LLM oversight — stops won't fire, regime
            # changes will be ignored. Audit row enables a window query below.
            emit(
                run_id=run_id, trigger=trigger, agent="detect_exit_triggers",
                event_type="exit_monitor_llm_failed",
                severity=1,
                payload={"symbol": symbol, "error": str(e)[:300]},
            )

        # Sanity guard: a partial-exit signal on a 1-contract option position
        # is physically impossible (options don't fractionate). The downstream
        # route_exit_or_hold would `max(1, round(1 * 0.5)) = 1` and force a
        # FULL close, contradicting the LLM's "let some run" signal. Demote
        # the action to HOLD so the LLM's intent is preserved. (5/12 NVDA
        # EXIT_CAUTIOUS @ factor 0.5 actually closed 100% — that's the bug.)
        full_qty = float(pos.get("qty") or 0)
        if action != "HOLD" and 0.0 < qty_factor < 1.0 and full_qty <= 1:
            original_action = action
            log.info(
                "[detect_exit_triggers] %s demoting %s (factor=%.2f, qty=%.0f) → HOLD: "
                "partial exit physically impossible on 1-contract position",
                symbol, original_action, qty_factor, full_qty,
            )
            reason = (
                f"demoted_from_{original_action}_partial_impossible_on_qty_{int(full_qty)}: "
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
        payload={"hold": hold_count, "exit": exit_count, "decisions": decisions},
    )

    # Escalation: if exit_monitor has failed N+ times in the last hour, the
    # LLM channel is effectively dead — positions are silently HOLDing, no
    # one is monitoring stops or regime changes. Fire a high-priority alert.
    _maybe_escalate_exit_monitor_failures(run_id, trigger)

    journal_payload = dict(state.get("journal") or {})
    journal_payload["exit_decisions"] = decisions
    return {"journal": journal_payload}


# ---------------------------------------------------------------------------
# Escalation helper — bridge between event-level failures and ops alerts.
# ---------------------------------------------------------------------------

# A streak of LLM failures means every open position is frozen at HOLD without
# real oversight. Threshold is intentionally tight — 3 failures in 60 min
# matches the systemd 5-min intraday cadence: 12 ticks per hour, so 3 failures
# is a ~25% failure rate that almost certainly means the channel is down, not
# a transient rate-limit blip.
_EXIT_LLM_FAIL_THRESHOLD = 3
_EXIT_LLM_FAIL_WINDOW_MIN = 60
# Don't spam — one alert per cooldown window even if failures keep accumulating.
_EXIT_LLM_FAIL_ALERT_COOLDOWN_MIN = 60


def _maybe_escalate_exit_monitor_failures(run_id: str, trigger: str) -> None:
    """Count recent exit_monitor LLM failures; alert ops if past threshold.

    Best-effort: any DB or ntfy error is swallowed so a flaky audit DB
    can't bring down the main intraday loop.
    """
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM agent_events
                WHERE event_type = 'exit_monitor_llm_failed'
                  AND ts > NOW() - (%s::text || ' minutes')::interval
                """,
                (str(_EXIT_LLM_FAIL_WINDOW_MIN),),
            )
            fail_count = int(cur.fetchone()[0] or 0)
            if fail_count < _EXIT_LLM_FAIL_THRESHOLD:
                return
            # Cooldown — did we already alert in the past cooldown window?
            cur.execute(
                """
                SELECT 1 FROM agent_events
                WHERE event_type = 'exit_monitor_persistent_failure'
                  AND ts > NOW() - (%s::text || ' minutes')::interval
                LIMIT 1
                """,
                (str(_EXIT_LLM_FAIL_ALERT_COOLDOWN_MIN),),
            )
            if cur.fetchone() is not None:
                return  # already alerted recently, suppress
    except Exception as e:
        log.warning("[detect_exit_triggers] escalation count query failed: %s", e)
        return

    emit(
        run_id=run_id, trigger=trigger, agent="detect_exit_triggers",
        event_type="exit_monitor_persistent_failure",
        severity=2,
        payload={
            "fail_count": fail_count,
            "window_minutes": _EXIT_LLM_FAIL_WINDOW_MIN,
            "threshold": _EXIT_LLM_FAIL_THRESHOLD,
        },
    )
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="ops",
            title="exit_monitor LLM DOWN",
            body=(
                f"{fail_count} exit_monitor LLM failures in the last "
                f"{_EXIT_LLM_FAIL_WINDOW_MIN} min (threshold={_EXIT_LLM_FAIL_THRESHOLD}).\n"
                f"All open positions are HOLDing with no LLM oversight. "
                f"Stops still fire via deterministic check, but regime/thesis "
                f"changes are not being evaluated. Check claude_code / codex / "
                f"deepseek channels."
            ),
            priority=5,
            tags=["rotating_light", "warning"],
        )
    except Exception as e:
        log.warning("[detect_exit_triggers] escalation ntfy failed: %s", e)


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
        side_close = "SELL"  # we only hold long positions
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

        # Order placed — close journal entry on whichever source the trade lives in.
        entry_price_j = float(pos.get("entry_price") or 0)
        mult_j = 100 if asset_type == "OPT" else 1
        pnl_j = round((exit_price - entry_price_j) * close_qty * mult_j, 2)
        outcome_j = "WIN" if pnl_j > 0 else ("LOSS" if pnl_j < 0 else "SCRATCH")

        closed_ok = False
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

        # Once the trade row is closed, also flip the parent thesis to
        # status='triggered' so the pretool freshness gate stops treating it
        # as a live thesis and /eod-review stops flagging it as a zombie.
        # Without this, theses pile up status='open' forever (the May 2026
        # AAPL/XLE/GLD/AMZN cleanup is the canonical example).
        if closed_ok and thesis_id is not None:
            thesis_note = f"trade {trade_id} closed via {action} ({reason})"[:280]
            thesis_closed = False
            if journal_close_thesis is not None:
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
