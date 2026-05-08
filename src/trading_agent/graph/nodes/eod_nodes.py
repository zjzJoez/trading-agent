"""Phase 2.5 — EOD review nodes.

Replaces stubs:
    reconcile_journal              compare moomoo positions vs journal open trades
    mark_to_market                 fetch closing marks for every open position
    persist_daily_marks            write portfolio_marks row to Postgres
    update_regime_accuracy_labels  annotate yesterday's regime_state with realized direction
    generate_eod_digest            compose daily summary text (LLM)
    ntfy_daily_summary             push digest to ntfy digest topic

Pipeline (eod_review_graph):
    reconcile_journal → mark_to_market → persist_daily_marks
    → update_regime_accuracy_labels → enrich_outcomes (eod_learning)
    → promote_or_rollback (eod_learning) → generate_eod_digest → ntfy_daily_summary
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_agent.events import SEV_WARN, emit
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


def _get_moomoo_positions() -> tuple[dict[str, dict], float, float]:
    """Return (symbol → row, equity, cash) from moomoo paper account.
    Returns ({}, 100_000, 100_000) on any error."""
    try:
        from trading_agent.mcp_servers.moomoo.server import get_account_info, get_positions
        ai = get_account_info()
        ai_row = (ai.get("rows") or [{}])[0]
        equity = float(ai_row.get("total_assets") or ai_row.get("net_cash_power") or 100_000)
        cash = float(ai_row.get("cash") or ai_row.get("avl_withdrawal_cash") or equity)
        pl = get_positions()
        by_symbol: dict[str, dict] = {}
        for row in (pl.get("rows") or []):
            sym = row.get("code") or row.get("symbol") or ""
            if sym:
                by_symbol[sym] = row
        return by_symbol, equity, cash
    except Exception as e:
        log.warning("[eod] moomoo positions fetch failed: %s", e)
        return {}, 100_000.0, 100_000.0


def _get_journal_open_trades() -> list[dict]:
    """Return open journal_trades rows from SQLite via journal MCP server."""
    try:
        from trading_agent.mcp_servers.journal.server import get_open_positions_with_thesis
        result = get_open_positions_with_thesis()
        return list(result.get("rows") or [])
    except Exception as e:
        log.warning("[eod] journal open positions fetch failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Node 1: reconcile_journal
# ---------------------------------------------------------------------------

def reconcile_journal(state: TradingGraphState) -> dict:
    """Compare moomoo paper-account positions against open journal trades.

    Mismatches are emitted as SEV_WARN events and included in the EOD
    digest so the operator knows to manually reconcile.  We never auto-close
    without an explicit risk decision.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/reconcile_journal] run_id=%s", run_id)

    moomoo_positions, equity, cash = _get_moomoo_positions()
    journal_trades = _get_journal_open_trades()

    moomoo_symbols = set(moomoo_positions.keys())
    journal_symbols = {t.get("symbol", "") for t in journal_trades if t.get("symbol")}

    only_in_moomoo = moomoo_symbols - journal_symbols
    only_in_journal = journal_symbols - moomoo_symbols
    matched = moomoo_symbols & journal_symbols

    discrepancies: list[dict] = []
    for sym in only_in_moomoo:
        discrepancies.append({"symbol": sym, "issue": "in_broker_not_journal"})
    for sym in only_in_journal:
        discrepancies.append({"symbol": sym, "issue": "in_journal_not_broker"})

    if discrepancies:
        emit(
            run_id=run_id, trigger=trigger, agent="reconcile_journal",
            event_type="reconcile_discrepancy",
            severity=SEV_WARN,
            payload={
                "only_in_moomoo": list(only_in_moomoo),
                "only_in_journal": list(only_in_journal),
                "matched": len(matched),
            },
        )
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="reconcile_journal",
            event_type="reconcile_ok",
            payload={"matched": len(matched)},
        )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["reconcile"] = {
        "matched": list(matched),
        "only_in_moomoo": list(only_in_moomoo),
        "only_in_journal": list(only_in_journal),
        "open_trades": journal_trades,
        "discrepancy_count": len(discrepancies),
    }
    return {
        "journal": journal_payload,
        "account": {"equity": equity, "cash": cash},
    }


# ---------------------------------------------------------------------------
# Node 2: mark_to_market
# ---------------------------------------------------------------------------

def mark_to_market(state: TradingGraphState) -> dict:
    """Fetch closing marks for every open position and compute unrealized PnL.

    Enriches each position dict with `mark`, `unrealized_pnl`, and (for
    options) fresh `iv` and `dte`.  The enriched list is stored in
    state["positions"] for persist_daily_marks to persist.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/mark_to_market] run_id=%s", run_id)

    journal = state.get("journal") or {}
    open_trades: list[dict] = journal.get("open_trades") or _get_journal_open_trades()

    if not open_trades:
        emit(run_id=run_id, trigger=trigger, agent="mark_to_market",
             event_type="no_open_positions", payload={})
        return {}

    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
        moomoo_ok = True
    except Exception as e:
        log.error("[mark_to_market] moomoo import failed: %s", e)
        moomoo_ok = False

    # Batch fetch all symbols in one call (signature: list[str])
    quote_by_symbol: dict[str, dict] = {}
    if moomoo_ok and open_trades:
        symbols_to_fetch = [t.get("symbol") for t in open_trades if t.get("symbol")]
        try:
            result = get_quote(symbols_to_fetch)
            for r in (result.get("rows") or []):
                code = r.get("code") or r.get("symbol") or ""
                if code:
                    quote_by_symbol[code] = r
        except Exception as e:
            log.warning("[mark_to_market] get_quote batch failed: %s", e)

    marked_positions: list[dict] = []
    total_unrealized_pnl = 0.0

    for trade in open_trades:
        symbol = trade.get("symbol") or ""
        entry = float(trade.get("entry_price") or 0)
        qty = float(trade.get("qty") or 0)
        is_opt = _is_option_code(symbol)
        mark = entry  # default: no change
        iv: float | None = None
        dte: int | None = None

        r = quote_by_symbol.get(symbol)
        if r:
            try:
                mark = float(r.get("last_price") or entry)
                if is_opt:
                    iv_raw = r.get("imp_volatility") or r.get("implied_volatility") or r.get("iv")
                    if iv_raw is not None:
                        iv = float(iv_raw)
                    exp_raw = r.get("expiry_date") or r.get("strike_time")
                    if exp_raw is not None:
                        from datetime import date
                        try:
                            exp = date.fromisoformat(str(exp_raw)[:10])
                            dte = (exp - date.today()).days
                        except Exception:
                            pass
            except Exception as e:
                log.warning("[mark_to_market] %s parse failed: %s", symbol, e)

        mult = 100 if is_opt else 1
        unrealized_pnl = (mark - entry) * qty * mult
        total_unrealized_pnl += unrealized_pnl

        pos = {
            "symbol": symbol,
            "asset_type": "OPT" if is_opt else "STK",
            "side": trade.get("side", "BUY"),
            "qty": qty,
            "entry_price": entry,
            "mark": mark,
            "stop": trade.get("stop"),
            "target": trade.get("target"),
            "unrealized_pnl": unrealized_pnl,
            "iv": iv,
            "dte": dte,
            "thesis_id": trade.get("thesis_id"),
            "strategy_label": trade.get("strategy_label"),
        }
        marked_positions.append(pos)

    emit(
        run_id=run_id, trigger=trigger, agent="mark_to_market",
        event_type="marks_computed",
        payload={
            "n_positions": len(marked_positions),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        },
    )

    journal_payload = dict(journal)
    journal_payload["marked_positions"] = marked_positions
    journal_payload["total_unrealized_pnl"] = round(total_unrealized_pnl, 2)
    return {"journal": journal_payload, "positions": marked_positions}


# ---------------------------------------------------------------------------
# Node 3: persist_daily_marks
# ---------------------------------------------------------------------------

def persist_daily_marks(state: TradingGraphState) -> dict:
    """Write a portfolio_marks row to Postgres with today's EOD snapshot."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/persist_daily_marks] run_id=%s", run_id)

    account = state.get("account") or {}
    equity = float(account.get("equity") or 100_000.0)
    cash = float(account.get("cash") or equity)
    journal = state.get("journal") or {}
    positions = journal.get("marked_positions") or state.get("positions") or []

    # Aggregate Greeks across option positions
    agg_greeks: dict[str, float] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for p in positions:
        for g in ("delta", "gamma", "vega", "theta"):
            if p.get(g) is not None:
                agg_greeks[g] += float(p[g]) * float(p.get("qty") or 0)

    # Factor exposures: simple long_equity / long_options split
    long_eq_notional = sum(
        float(p.get("mark") or 0) * float(p.get("qty") or 0)
        for p in positions if p.get("asset_type") == "STK"
    )
    long_opt_notional = sum(
        float(p.get("mark") or 0) * float(p.get("qty") or 0) * 100
        for p in positions if p.get("asset_type") == "OPT"
    )
    exposures = {
        "long_equity_notional": round(long_eq_notional, 2),
        "long_option_notional": round(long_opt_notional, 2),
    }

    pnl_day = round(journal.get("total_unrealized_pnl") or 0.0, 2)
    pnl_payload = {"day": pnl_day, "week": None, "mtd": None, "ytd": None}

    now_utc = datetime.now(timezone.utc)

    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO portfolio_marks
                    (as_of, equity, cash, positions, greeks, exposures, pnl)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (
                    now_utc,
                    equity,
                    cash,
                    json.dumps([
                        {k: v for k, v in p.items() if v is not None}
                        for p in positions
                    ], default=str),
                    json.dumps(agg_greeks),
                    json.dumps(exposures),
                    json.dumps(pnl_payload),
                ),
            )
            marks_id = int(cur.fetchone()[0])
        status = "ok"
    except Exception as e:
        log.warning("[persist_daily_marks] DB insert failed: %s", e)
        marks_id = -1
        status = f"failed: {e}"

    emit(
        run_id=run_id, trigger=trigger, agent="persist_daily_marks",
        event_type="marks_persisted",
        payload={
            "marks_id": marks_id,
            "equity": equity,
            "pnl_day": pnl_day,
            "n_positions": len(positions),
            "status": status,
        },
    )
    return {}


# ---------------------------------------------------------------------------
# Node 4: update_regime_accuracy_labels
# ---------------------------------------------------------------------------

def update_regime_accuracy_labels(state: TradingGraphState) -> dict:
    """Annotate yesterday's regime_states row with whether its directional
    call was accurate based on SPY's next-day realized return.

    Accuracy heuristic:
        BULL_TREND / RANGE_LOW_VOL → expect SPY ≥ 0% today → "correct" if SPY closed up
        BEAR_TREND / CRISIS        → expect SPY < 0% today → "correct" if SPY closed down
        VOLATILE_TRANSITION        → no directional expectation → "neutral"

    Updates regime_states.accuracy_label (TEXT column).  If the column
    doesn't exist yet (pre-migration), silently skips — non-blocking.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/update_regime_accuracy_labels] run_id=%s", run_id)

    # Fetch SPY close for today
    spy_return: float | None = None
    try:
        from datetime import date as _date, timedelta as _td
        from trading_agent.mcp_servers.moomoo.server import get_historical_kline
        today = _date.today()
        start = (today - _td(days=7)).isoformat()   # 7-day buffer covers weekends/holidays
        end = today.isoformat()
        klines = get_historical_kline(symbol="US.SPY", start=start, end=end, ktype="K_DAY", max_count=5)
        rows = klines.get("rows") or []
        if len(rows) >= 2:
            # column name from moomoo API is "close", not "close_price"
            prev_close = float(rows[-2].get("close") or rows[-2].get("close_price") or 0)
            today_close = float(rows[-1].get("close") or rows[-1].get("close_price") or 0)
            if prev_close > 0:
                spy_return = (today_close - prev_close) / prev_close
    except Exception as e:
        log.warning("[update_regime_accuracy_labels] SPY kline failed: %s", e)

    if spy_return is None:
        emit(run_id=run_id, trigger=trigger, agent="update_regime_accuracy_labels",
             event_type="spy_return_unavailable", payload={})
        return {}

    BULL_LABELS = {"BULL_TREND", "RANGE_LOW_VOL"}
    BEAR_LABELS = {"BEAR_TREND", "CRISIS"}

    # Find yesterday's regime_states row
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    accuracy_label: str

    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, label
                FROM regime_states
                WHERE created_at >= %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (yesterday.replace(hour=0, minute=0, second=0, microsecond=0),),
            )
            row = cur.fetchone()

        if row is None:
            emit(run_id=run_id, trigger=trigger, agent="update_regime_accuracy_labels",
                 event_type="no_regime_row_yesterday", payload={"spy_return": round(spy_return, 4)})
            return {}

        regime_id, regime_label = int(row[0]), str(row[1])

        if regime_label in BULL_LABELS:
            accuracy_label = "correct" if spy_return >= 0 else "incorrect"
        elif regime_label in BEAR_LABELS:
            accuracy_label = "correct" if spy_return < 0 else "incorrect"
        else:
            accuracy_label = "neutral"

        # Write accuracy_label to regime_states (migration 003 adds the columns).
        with cursor() as cur:
            cur.execute(
                """
                UPDATE regime_states
                SET accuracy_label = %s, spy_next_day_return = %s
                WHERE id = %s
                """,
                (accuracy_label, round(spy_return, 6), regime_id),
            )

    except Exception as e:
        log.warning("[update_regime_accuracy_labels] DB failed: %s", e)
        accuracy_label = "error"
        regime_label = "unknown"
        regime_id = -1

    emit(
        run_id=run_id, trigger=trigger, agent="update_regime_accuracy_labels",
        event_type="accuracy_labeled",
        payload={
            "regime_id": regime_id,
            "regime_label": regime_label,
            "spy_return": round(spy_return, 4),
            "accuracy_label": accuracy_label,
        },
    )
    return {}


# ---------------------------------------------------------------------------
# Node 5: generate_eod_digest
# ---------------------------------------------------------------------------

def _build_eod_digest_prompt(state: TradingGraphState) -> str:
    account = state.get("account") or {}
    journal = state.get("journal") or {}
    positions = journal.get("marked_positions") or state.get("positions") or []
    regime = state.get("regime") or {}
    reconcile = journal.get("reconcile") or {}

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    equity = float(account.get("equity") or 100_000.0)
    pnl_day = float(journal.get("total_unrealized_pnl") or 0.0)
    n_open = len(positions)
    discrepancies = reconcile.get("discrepancy_count", 0)

    pos_lines = []
    for p in positions[:10]:
        sym = p.get("symbol", "?")
        mark = p.get("mark", 0)
        entry = p.get("entry_price", 0)
        pnl = p.get("unrealized_pnl", 0)
        dte = p.get("dte")
        dte_str = f" DTE={dte}" if dte is not None else ""
        pos_lines.append(f"  {sym}: mark={mark:.2f} entry={entry:.2f} pnl={pnl:+.0f}{dte_str}")

    lines = [
        f"date: {now_utc}",
        f"equity: ${equity:,.0f}",
        f"day_unrealized_pnl: ${pnl_day:+,.0f}",
        f"open_positions: {n_open}",
        f"regime: {regime.get('label', 'UNKNOWN')} (conf={regime.get('confidence', 0):.2f})",
        f"reconcile_discrepancies: {discrepancies}",
        f"",
        f"positions:",
    ] + pos_lines + [
        f"",
        f"Generate a concise EOD digest (max 200 words). "
        f"Cover: overall account health, key positions to watch, any thesis invalidations, "
        f"tomorrow's key risks. "
        f"Be direct — this is the operator's end-of-day push notification.",
    ]
    return "\n".join(lines)


def generate_eod_digest(state: TradingGraphState) -> dict:
    """Call ntfy-digest-composer (Haiku) to produce the EOD summary text."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/generate_eod_digest] run_id=%s", run_id)

    prompt = _build_eod_digest_prompt(state)
    digest_title = "EOD Summary"
    digest_body = "(digest unavailable)"
    priority = 3
    tags: list[str] = ["chart_with_upwards_trend"]

    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import NtfyDigestOutput
        router = get_router()
        res = router.call("ntfy_digest_composer", prompt, schema=NtfyDigestOutput, timeout_s=90)
        parsed: NtfyDigestOutput | None = (
            res.parsed if isinstance(res.parsed, NtfyDigestOutput) else None
        )
        if parsed:
            digest_title = parsed.title[:80]
            digest_body = parsed.body_md[:1000]
            priority = parsed.priority
            tags = list(parsed.tags or tags)
    except Exception as e:
        log.warning("[generate_eod_digest] LLM failed: %s", e)
        # Fallback: construct a simple digest from state
        account = state.get("account") or {}
        journal = state.get("journal") or {}
        equity = float(account.get("equity") or 100_000.0)
        pnl = float(journal.get("total_unrealized_pnl") or 0.0)
        n = len(journal.get("marked_positions") or [])
        digest_title = f"EOD {datetime.now(timezone.utc).strftime('%m-%d')}"
        digest_body = f"Equity: ${equity:,.0f}  |  Day PnL: ${pnl:+,.0f}  |  {n} open positions"

    emit(
        run_id=run_id, trigger=trigger, agent="generate_eod_digest",
        event_type="digest_generated",
        payload={"title": digest_title, "body_len": len(digest_body)},
    )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["eod_digest"] = {
        "title": digest_title,
        "body": digest_body,
        "priority": priority,
        "tags": tags,
    }
    return {"journal": journal_payload}


# ---------------------------------------------------------------------------
# Node 6: ntfy_daily_summary
# ---------------------------------------------------------------------------

def ntfy_daily_summary(state: TradingGraphState) -> dict:
    """Push the EOD digest to ntfy and write a final agent_event."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/ntfy_daily_summary] run_id=%s", run_id)

    journal = state.get("journal") or {}
    digest = journal.get("eod_digest") or {}

    title = str(digest.get("title") or "EOD Summary")
    body = str(digest.get("body") or "No digest generated.")
    priority = int(digest.get("priority") or 3)
    tags = list(digest.get("tags") or ["chart_with_upwards_trend"])

    ntfy_status = "skipped"
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="digest",
            title=title,
            body=body,
            priority=priority,
            tags=tags,
        )
        ntfy_status = "sent"
        log.info("[ntfy_daily_summary] digest sent: %s", title)
    except Exception as e:
        log.warning("[ntfy_daily_summary] ntfy failed: %s", e)
        ntfy_status = f"failed: {e}"

    emit(
        run_id=run_id, trigger=trigger, agent="ntfy_daily_summary",
        event_type="eod_digest_sent",
        payload={"title": title, "status": ntfy_status},
    )
    return {}


__all__ = [
    "reconcile_journal",
    "mark_to_market",
    "persist_daily_marks",
    "update_regime_accuracy_labels",
    "generate_eod_digest",
    "ntfy_daily_summary",
]
