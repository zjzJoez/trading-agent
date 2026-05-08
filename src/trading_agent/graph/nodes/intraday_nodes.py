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

    Uses moomoo get_quote for underlying prices and get_option_chain_snapshot
    for option contracts.  Failures per-position are logged and skipped; the
    position keeps its stale mark so downstream nodes still function.
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
        from trading_agent.mcp_servers.moomoo.server import (
            get_quote,
            get_option_chain_snapshot,
        )
    except Exception as e:
        log.error("[refresh_quotes] moomoo import failed: %s", e)
        return {}

    refreshed: list[dict] = []
    stale_count = 0

    for pos in positions:
        symbol = pos.get("symbol") or ""
        if not symbol:
            refreshed.append(pos)
            continue

        updated = dict(pos)
        try:
            if _is_option_code(symbol):
                snap = get_option_chain_snapshot(symbol=symbol)
                rows = snap.get("rows") or []
                if rows:
                    row = rows[0]
                    mark = float(
                        row.get("last_price") or row.get("option_premium") or pos["mark"]
                    )
                    updated["mark"] = mark
                    # Greeks — moomoo returns them in the snapshot
                    if row.get("delta") is not None:
                        updated["delta"] = float(row["delta"])
                    if row.get("gamma") is not None:
                        updated["gamma"] = float(row["gamma"])
                    if row.get("vega") is not None:
                        updated["vega"] = float(row["vega"])
                    if row.get("theta") is not None:
                        updated["theta"] = float(row["theta"])
                    if row.get("imp_volatility") is not None:
                        updated["iv"] = float(row["imp_volatility"])
                    # DTE from row
                    if row.get("expiry_date") is not None:
                        from datetime import date
                        try:
                            exp = date.fromisoformat(str(row["expiry_date"]))
                            updated["dte"] = (exp - date.today()).days
                        except Exception:
                            pass
                else:
                    stale_count += 1
            else:
                # Stock underlying
                underlying = pos.get("underlying") or symbol
                qr = get_quote(symbol=underlying)
                rows = qr.get("rows") or []
                if rows:
                    row = rows[0]
                    mark = float(row.get("last_price") or pos["mark"])
                    updated["mark"] = mark
                else:
                    stale_count += 1
        except Exception as e:
            log.warning("[refresh_quotes] symbol=%s failed: %s", symbol, e)
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

def _thesis_summary_for(pos: dict) -> str:
    """Pull a short thesis text from Postgres journal_theses."""
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
            cancel_paper_order,
            get_orders,
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

    closed_symbols: list[str] = []
    failed_symbols: list[str] = []

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

        full_qty = float(pos.get("qty") or 0)
        close_qty = max(1, round(full_qty * qty_factor)) if qty_factor < 1.0 else full_qty
        if close_qty <= 0:
            failed_symbols.append(symbol)
            continue

        asset_type = pos.get("asset_type", "STK")
        is_option = asset_type == "OPT"
        side_close = "SELL"  # we only hold long positions

        placed_order_id: str | None = None
        exit_price = float(pos.get("mark") or pos.get("entry_price") or 0)
        thesis_id = pos.get("thesis_id") or 0

        if moomoo_available:
            try:
                if asset_type == "OPT":
                    # Options: use the option-specific placer (limit-price only, no MARKET)
                    from trading_agent.mcp_servers.moomoo.server import place_paper_option_order
                    result = place_paper_option_order(
                        option_symbol=symbol,
                        side=side_close,
                        contracts=int(close_qty),
                        price=exit_price,
                        thesis_id=int(thesis_id),
                        strategy_label=pos.get("strategy_label"),
                        delta=pos.get("delta"),
                        dte=pos.get("dte"),
                    )
                else:
                    # Stocks: limit at mark; NORMAL order type avoids wide fills
                    result = place_paper_order(
                        symbol=symbol,
                        side=side_close,
                        qty=int(close_qty),
                        price=exit_price,
                        thesis_id=int(thesis_id),
                        order_type="NORMAL",
                    )
                placed_order_id = str(result.get("order_id") or "")
                log.info(
                    "[route_exit_or_hold] %s order placed %s qty=%s order_id=%s",
                    asset_type, symbol, close_qty, placed_order_id,
                )
            except Exception as e:
                log.warning("[route_exit_or_hold] order placement %s failed: %s", symbol, e)
                placed_order_id = None

        if placed_order_id is None:
            # Order placement failed — do NOT journal close or notify as closed.
            # Operator will see the failed event in agent_events and can act manually.
            emit(
                run_id=run_id, trigger=trigger, agent="route_exit_or_hold",
                event_type="exit_order_failed",
                severity=1,
                payload={"symbol": symbol, "action": action, "reason": reason},
            )
            failed_symbols.append(symbol)
            continue

        # Order placed — journal the close and notify
        try:
            from trading_agent.mcp_servers.journal.server import close_trade as journal_close_trade
            journal_close_trade(
                symbol=symbol,
                exit_price=exit_price,
                close_reason=action,
            )
        except Exception as e:
            log.warning("[route_exit_or_hold] journal close_trade %s failed: %s", symbol, e)

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
    return {}


__all__ = [
    "refresh_quotes_and_greeks",
    "detect_exit_triggers",
    "route_exit_or_hold",
]
