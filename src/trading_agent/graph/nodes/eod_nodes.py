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


def _get_journal_open_trades() -> list[dict] | None:
    """Return open journal_trades rows from Postgres (joined with theses).

    Postgres ``journal_trades`` is the canonical source — autonomous fills
    only write there, not to the Phase-1 SQLite mirror. ``None`` means the
    store could not be read; callers must treat that as UNKNOWN — comparing
    against an empty set during a DB outage would produce a confidently
    wrong all-flat reconcile report.
    """
    try:
        from trading_agent.store.postgres import get_open_journal_trades
    except Exception as e:
        log.warning("[eod] postgres import failed: %s", e)
        return None
    rows = get_open_journal_trades()
    return None if rows is None else list(rows)


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
    if journal_trades is None:
        # Journal unreadable — a discrepancy report against an empty set
        # would flag every broker position as in_broker_not_journal and
        # read as a confidently wrong "all flat" digest. Say so instead.
        emit(
            run_id=run_id, trigger=trigger, agent="reconcile_journal",
            event_type="journal_unreadable", severity=2,
            payload={"n_broker_positions": len(moomoo_positions)},
        )
        journal_payload = dict(state.get("journal") or {})
        journal_payload["reconcile"] = {
            "matched": [], "only_in_moomoo": [], "only_in_journal": [],
            "open_trades": [], "discrepancy_count": 0,
            "journal_unreadable": True,
        }
        return {
            "journal": journal_payload,
            "account": {"equity": equity, "cash": cash},
        }

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
# Node 1a: reconcile_phantom_trades
# ---------------------------------------------------------------------------

# A "phantom" trade row is one where:
#   * journal_trades.outcome = 'OPEN'
#   * the broker_fill_json shows fill_qty / dealt_qty = 0
#     (order was SUBMITTED but never executed — broker rejected, cancelled,
#     or the order timed out)
#   * the row has been open longer than PHANTOM_AGE_HOURS
#
# Without this sweep, phantom rows show up as "live positions" in the
# enrichment query inside intraday_nodes — the hard executor evaluates
# rules against an imaginary mark and the dashboard misreports portfolio
# state. The May 2026 SPY 742C trade (#4) is the canonical example: order
# 3074924 was SUBMITTING then orphaned; the journal still says OPEN
# 8 days later.
PHANTOM_AGE_HOURS = 24


def reconcile_phantom_trades(state: TradingGraphState) -> dict:
    """Close out journal_trades rows for orders the broker never filled.

    Looks at broker_fill_json for fill_qty / dealt_qty / dealt_avg_price
    fields; if any one of them is zero AND the trade is older than
    PHANTOM_AGE_HOURS, mark outcome='UNFILLED' and close the parent thesis.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/reconcile_phantom_trades] run_id=%s", run_id)

    voided: list[dict] = []
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=PHANTOM_AGE_HOURS)

    try:
        from trading_agent.store.postgres import cursor
    except Exception as e:
        log.warning("[reconcile_phantom_trades] postgres import failed: %s", e)
        return {}

    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, symbol, thesis_id, opened_at,
                       broker_fill_json, broker_order_id
                FROM journal_trades
                WHERE outcome = 'OPEN'
                  AND opened_at < %s
                """,
                (cutoff_dt,),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("[reconcile_phantom_trades] scan failed: %s", e)
        return {}

    for (trade_id, symbol, thesis_id, opened_at, broker_fill_json,
         broker_order_id) in rows:
        fill = broker_fill_json if isinstance(broker_fill_json, dict) else {}
        # Three independent signals — any one missing fill is enough.
        fill_qty = _safe_float(fill.get("fill_qty"))
        dealt_qty: float | None = None
        avg_price: float | None = None
        raw = fill.get("raw_response")
        if isinstance(raw, list) and raw:
            r0 = raw[0] if isinstance(raw[0], dict) else {}
            dealt_qty = _safe_float(r0.get("dealt_qty"))
            avg_price = _safe_float(r0.get("dealt_avg_price"))
        avg_fill = _safe_float(fill.get("avg_fill_price"))

        # "phantom" = every available fill-signal reads zero
        signals = [s for s in (fill_qty, dealt_qty, avg_price, avg_fill)
                   if s is not None]
        if not signals or any(s > 0 for s in signals):
            continue

        try:
            with cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_trades
                    SET outcome = 'UNFILLED',
                        closed_at = NOW(),
                        close_reason = COALESCE(close_reason,
                            'reconcile_phantom_trades: broker never filled')
                    WHERE id = %s AND outcome = 'OPEN'
                    """,
                    (int(trade_id),),
                )
        except Exception as e:
            log.warning(
                "[reconcile_phantom_trades] UPDATE journal_trades(%s) failed: %s",
                trade_id, e,
            )
            continue

        # Close parent thesis too — without a fill, the thesis is moot.
        if thesis_id is not None:
            try:
                with cursor() as cur:
                    cur.execute(
                        "UPDATE journal_theses SET status='void' "
                        "WHERE id=%s AND status='open'",
                        (int(thesis_id),),
                    )
            except Exception as e:
                log.warning(
                    "[reconcile_phantom_trades] thesis(%s) update failed: %s",
                    thesis_id, e,
                )

        voided.append({
            "trade_id": int(trade_id),
            "symbol": symbol,
            "thesis_id": int(thesis_id) if thesis_id is not None else None,
            "opened_at": opened_at.isoformat() if opened_at else None,
            "broker_order_id": broker_order_id,
        })

    if voided:
        emit(
            run_id=run_id, trigger=trigger, agent="reconcile_phantom_trades",
            event_type="phantom_trades_voided",
            severity=1,  # this is a real ops signal — broker rejected something
            payload={"count": len(voided), "items": voided[:20]},
        )
        log.warning(
            "[reconcile_phantom_trades] voided %d phantom trade(s)",
            len(voided),
        )
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="reconcile_phantom_trades",
            event_type="phantom_trades_clean",
            payload={},
        )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["phantom_trades_voided"] = voided
    return {"journal": journal_payload}


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Node 1b: auto_void_stale_theses
# ---------------------------------------------------------------------------

# Any thesis older than this with no OPEN trade linked is considered abandoned
# and gets auto-voided. Picked to be safely longer than every typical catalyst
# window (earnings: ~5d, FOMC: ~2d, geopolitical hedge: ~30d). 14d is long
# enough that a real thesis-still-active scenario would have already been
# converted into a position or refreshed.
STALE_THESIS_AGE_DAYS = 14


def auto_void_stale_theses(state: TradingGraphState) -> dict:
    """Sweep open theses with no linked OPEN trade older than N days and void them.

    Two zombie classes exist:
      1. Thesis recorded but never executed (no trade row) — happens when
         the operator records a thesis via /research then doesn't pull the
         trigger. These linger as status='open' forever.
      2. Thesis with closed trade but never closed itself — handled inline
         by route_exit_or_hold (sibling fix in same change).

    This node handles class 1. We void rather than 'triggered' since the
    invalidation never actually fired — the thesis simply went stale.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/auto_void_stale_theses] run_id=%s", run_id)

    voided: list[dict] = []
    # journal_close_thesis is optional — we may be running on EC2 with an
    # empty sqlite journal, in which case the Postgres sweep is the only
    # path that does real work. Import failure is informational, not fatal.
    journal_close_thesis = None  # type: ignore[assignment]
    connection = None
    try:
        from trading_agent.mcp_servers.journal.server import (
            close_thesis as journal_close_thesis,  # noqa: F811 - reassigned from import
            connection,  # noqa: F811 - reassigned from import
        )
    except Exception as e:
        log.warning("[auto_void_stale_theses] sqlite journal import failed: %s", e)

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=STALE_THESIS_AGE_DAYS)
    cutoff = cutoff_dt.isoformat()

    # ---- Sweep 1: sqlite journal-mcp (local dev / shadow journal) -----------
    sqlite_rows: list[tuple] = []
    if connection is not None:
        try:
            with connection() as conn:
                sqlite_rows = conn.execute(
                    """
                    SELECT t.id, t.ticker, t.created_at
                    FROM theses t
                    WHERE t.status = 'open'
                      AND t.created_at < ?
                      AND NOT EXISTS (
                          SELECT 1 FROM trades tr
                          WHERE tr.thesis_id = t.id
                            AND tr.outcome = 'OPEN'
                      )
                    """,
                    (cutoff,),
                ).fetchall()
        except Exception as e:
            log.warning("[auto_void_stale_theses] sqlite scan failed: %s", e)
            sqlite_rows = []

    for row in sqlite_rows:
        thesis_id = row[0]
        ticker = row[1]
        created_at = row[2]
        if journal_close_thesis is None:
            continue
        try:
            journal_close_thesis(
                thesis_id=int(thesis_id),
                status="void",
                note=(
                    f"auto_void_stale_theses: {STALE_THESIS_AGE_DAYS}d age cutoff, "
                    f"no OPEN trade linked. Created {created_at}."
                ),
            )
            voided.append({"thesis_id": int(thesis_id), "ticker": ticker,
                           "created_at": str(created_at), "source": "sqlite"})
        except Exception as e:
            log.warning("[auto_void_stale_theses] sqlite close_thesis(%s) failed: %s",
                        thesis_id, e)

    # ---- Sweep 2: Postgres journal_theses (production canonical on EC2) -----
    # The 5/22 AAPL/XLE/GLD/AMZN zombie cleanup proved that all real theses
    # on EC2 live in Postgres — sqlite is empty there. Without this sweep,
    # auto_void_stale_theses does nothing on prod.
    #
    # Status routing distinguishes two outcomes:
    #   * If the thesis has ANY linked trade rows (even already-closed),
    #     archive as 'triggered' — the idea was actually tested in market.
    #     Marking it 'void' would lose the win/loss history.
    #   * Pure orphans (no trade row ever) → 'void' — the operator never
    #     pulled the trigger, idea went stale.
    # (5/22 audit: QQQ #1 and SPY #2 were both 24d stale opens BUT both
    # had WIN trades — they should be 'triggered', not 'void'.)
    try:
        from trading_agent.store.postgres import cursor as pg_cursor
        with pg_cursor() as cur:
            cur.execute(
                """
                SELECT th.id, th.ticker, th.created_at,
                       EXISTS (
                           SELECT 1 FROM journal_trades tr
                           WHERE tr.thesis_id = th.id
                       ) AS has_any_trade,
                       (
                           SELECT outcome FROM journal_trades tr
                           WHERE tr.thesis_id = th.id
                             AND tr.outcome != 'OPEN'
                           ORDER BY closed_at DESC NULLS LAST
                           LIMIT 1
                       ) AS latest_outcome
                FROM journal_theses th
                WHERE th.status = 'open'
                  AND th.created_at < %s
                  AND NOT EXISTS (
                      SELECT 1 FROM journal_trades tr
                      WHERE tr.thesis_id = th.id
                        AND tr.outcome = 'OPEN'
                  )
                """,
                (cutoff_dt,),
            )
            pg_rows = cur.fetchall()
        for thesis_id, ticker, created_at, has_any_trade, latest_outcome in pg_rows:
            new_status = "triggered" if has_any_trade else "void"
            try:
                with pg_cursor() as cur:
                    cur.execute(
                        "UPDATE journal_theses SET status=%s "
                        "WHERE id=%s AND status='open'",
                        (new_status, int(thesis_id)),
                    )
                voided.append({
                    "thesis_id": int(thesis_id),
                    "ticker": ticker,
                    "created_at": created_at.isoformat() if created_at else None,
                    "source": "postgres",
                    "new_status": new_status,
                    "latest_outcome": latest_outcome,
                })
            except Exception as e:
                log.warning("[auto_void_stale_theses] postgres update(%s) failed: %s",
                            thesis_id, e)
    except Exception as e:
        log.warning("[auto_void_stale_theses] postgres scan failed: %s", e)

    if voided:
        emit(
            run_id=run_id, trigger=trigger, agent="auto_void_stale_theses",
            event_type="stale_theses_voided",
            payload={"count": len(voided), "items": voided[:20]},
        )
        log.info("[auto_void_stale_theses] voided %d stale theses", len(voided))
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="auto_void_stale_theses",
            event_type="stale_theses_clean",
            payload={},
        )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["stale_theses_voided"] = voided
    return {"journal": journal_payload}


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
    open_trades: list[dict] | None = (
        journal.get("open_trades") or _get_journal_open_trades())

    if open_trades is None:
        # Journal unreadable — "no marks tonight" must be loud, not a quiet
        # no_open_positions that reads as an intentionally flat book.
        emit(run_id=run_id, trigger=trigger, agent="mark_to_market",
             event_type="journal_unreadable", severity=2, payload={})
        return {}
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

    from trading_agent.combo import combo_meta

    # Batch fetch all symbols in one call (signature: list[str]). Combo rows
    # (symbol = short leg, entry = net credit, side = SELL) need BOTH legs
    # quoted — marking them at the short leg's price alone is sign-inverted
    # and wing-blind (M1-0.1 net-of-legs marking).
    quote_by_symbol: dict[str, dict] = {}
    if moomoo_ok and open_trades:
        symbols_to_fetch = []
        for t in open_trades:
            meta = combo_meta(t.get("combo"))
            if meta is not None:
                symbols_to_fetch.extend([meta.short_leg, meta.long_leg])
            elif t.get("symbol"):
                symbols_to_fetch.append(t.get("symbol"))
        symbols_to_fetch = list(dict.fromkeys(symbols_to_fetch))
        try:
            result = get_quote(symbols_to_fetch)
            for r in (result.get("rows") or []):
                code = r.get("code") or r.get("symbol") or ""
                if code:
                    quote_by_symbol[code] = r
        except Exception as e:
            log.warning("[mark_to_market] get_quote batch failed: %s", e)

    def _pos_last(row: dict | None) -> float | None:
        try:
            v = float((row or {}).get("last_price") or 0)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    marked_positions: list[dict] = []
    total_unrealized_pnl = 0.0
    combo_skipped = 0

    for trade in open_trades:
        symbol = trade.get("symbol") or ""
        entry = float(trade.get("entry_price") or 0)
        qty = float(trade.get("qty") or 0)

        # Combo rows: mark net-of-both-legs with SHORT polarity. entry_price
        # is the net CREDIT, so unrealized = (entry − spread_value) × qty ×
        # 100 — marking the short leg alone with LONG polarity reports a
        # healthy credit spread as a loss. Either leg unquotable → skip the
        # row loudly (mirrors learning/excursion.py) rather than mark junk.
        meta = combo_meta(trade.get("combo"))
        if meta is not None:
            short_last = _pos_last(quote_by_symbol.get(meta.short_leg))
            long_last = _pos_last(quote_by_symbol.get(meta.long_leg))
            if short_last is None or long_last is None:
                combo_skipped += 1
                log.warning(
                    "[mark_to_market] combo %s skipped: leg unquotable "
                    "(short=%s long=%s)", symbol, short_last, long_last,
                )
                continue
            mark = round(short_last - long_last, 4)
            unrealized_pnl = (entry - mark) * qty * 100.0
            total_unrealized_pnl += unrealized_pnl
            marked_positions.append({
                "symbol": symbol,
                "asset_type": "OPT",
                "side": trade.get("side", "SELL"),
                "qty": qty,
                "entry_price": entry,
                "mark": mark,
                "stop": trade.get("stop"),
                "target": trade.get("target"),
                "unrealized_pnl": unrealized_pnl,
                "combo": True,
                "thesis_id": trade.get("thesis_id"),
                "strategy_label": trade.get("strategy_label"),
            })
            continue

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
            "combo_skipped_unquotable": combo_skipped,
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
# Node 4b: compute_ops_summary
# ---------------------------------------------------------------------------

# Sister of generate_eod_digest. The LLM digest is operator-facing prose;
# this is the cold-numbers operational rollup ("today: 6 trigger fires,
# 1 proposal, 1 DEFER, 0 fills, 2 alerts, $0.43 LLM cost"). Sent to the
# `ops` topic so it sits with healthchecks/deploys, not the `digest` topic
# (which is the LLM narrative — different consumer use case).
#
# The 23-day-silent-trade-engine incident is what motivated this: if this
# digest had existed earlier we'd have seen "0 candidate_entry dispatches"
# every day for 23 days and noticed by day 2.


def compute_ops_summary(state: TradingGraphState) -> dict:
    """Aggregate today's agent_events into a programmatic ops digest.

    Bucketing (UTC day):
      - trigger fires by type (count)
      - candidates dispatched + DEFER/VETO/APPROVE breakdown
      - severity > 0 events (count + tail of details)
      - LLM cost sum from cost_usd column
      - burn-in counter snapshot
      - any failed systemd units (joined to failed_units_check event)
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[eod/compute_ops_summary] run_id=%s", run_id)

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary: dict[str, Any] = {"date": today_utc}

    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            # Trigger fires today (run_start events) — grouped by trigger
            cur.execute(
                """
                SELECT trigger, COUNT(*)
                FROM agent_events
                WHERE event_type = 'run_start'
                  AND ts::date = CURRENT_DATE
                GROUP BY trigger
                ORDER BY trigger
                """,
            )
            summary["trigger_fires"] = {row[0]: int(row[1]) for row in cur.fetchall()}

            # Risk-council decisions today
            cur.execute(
                """
                SELECT payload->>'decision' AS decision, COUNT(*)
                FROM agent_events
                WHERE event_type = 'decision_finalized'
                  AND ts::date = CURRENT_DATE
                GROUP BY decision
                """,
            )
            summary["decisions"] = {row[0] or "UNKNOWN": int(row[1]) for row in cur.fetchall()}

            # Dispatches (candidate_entry_dispatched)
            cur.execute(
                """
                SELECT COUNT(*) AS dispatched,
                       COUNT(*) FILTER (WHERE event_type = 'candidate_entry_dispatch_failed') AS failed
                FROM agent_events
                WHERE event_type IN ('candidate_entry_dispatched',
                                     'candidate_entry_dispatch_failed')
                  AND ts::date = CURRENT_DATE
                """,
            )
            row = cur.fetchone() or (0, 0)
            summary["dispatches"] = {"ok": int(row[0] or 0), "failed": int(row[1] or 0)}

            # Severity > 0 events — count + tail
            cur.execute(
                """
                SELECT COUNT(*) FROM agent_events
                WHERE severity > 0 AND ts::date = CURRENT_DATE
                """,
            )
            summary["alerts_total"] = int((cur.fetchone() or (0,))[0] or 0)

            cur.execute(
                """
                SELECT ts::time, agent, event_type
                FROM agent_events
                WHERE severity > 0 AND ts::date = CURRENT_DATE
                ORDER BY id DESC LIMIT 5
                """,
            )
            summary["alerts_recent"] = [
                {"t": str(row[0])[:8], "agent": row[1], "event": row[2]}
                for row in cur.fetchall()
            ]

            # LLM cost
            cur.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM agent_events
                WHERE cost_usd IS NOT NULL AND ts::date = CURRENT_DATE
                """,
            )
            summary["llm_cost_usd"] = float((cur.fetchone() or (0,))[0] or 0)

            # Burn-in count
            cur.execute("SELECT COUNT(*) FROM journal_trades")
            summary["filled_trades_lifetime"] = int((cur.fetchone() or (0,))[0] or 0)

    except Exception as e:
        log.warning("[compute_ops_summary] postgres query failed: %s", e)
        summary["db_error"] = str(e)[:200]

    # Burn-in threshold (also defined in risk/llm.py; kept here for digest
    # readability — if you bump it there, also bump here).
    BURN_IN_TARGET = 12
    filled = summary.get("filled_trades_lifetime", 0)
    burn_in_remaining = max(0, BURN_IN_TARGET - filled)

    # Compose the body
    triggers = summary.get("trigger_fires", {})
    decisions = summary.get("decisions", {})
    dispatches = summary.get("dispatches", {})
    alerts = summary.get("alerts_total", 0)
    cost = summary.get("llm_cost_usd", 0)

    body_lines = [
        f"{today_utc} UTC",
        "",
        f"Trigger fires: " + (
            ", ".join(f"{k}={v}" for k, v in sorted(triggers.items())) or "none"
        ),
        f"Dispatches: ok={dispatches.get('ok', 0)} failed={dispatches.get('failed', 0)}",
        f"Decisions: " + (
            ", ".join(f"{k}={v}" for k, v in sorted(decisions.items())) or "none"
        ),
        f"Alerts (sev>0): {alerts}",
        f"LLM cost today: ${cost:.2f}",
        f"Burn-in: {filled}/{BURN_IN_TARGET} fills ({burn_in_remaining} remaining)",
    ]

    if summary.get("alerts_recent"):
        body_lines.append("")
        body_lines.append("Recent alerts:")
        for a in summary["alerts_recent"]:
            body_lines.append(f"  {a['t']} {a['agent']} → {a['event']}")

    body = "\n".join(body_lines)

    # Choose priority: anything > 0 alerts gets default-priority (3); zero
    # alerts gets quiet (1) so phone doesn't buzz nightly when nothing's wrong.
    priority = 3 if alerts > 0 or dispatches.get("failed", 0) > 0 else 1

    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="ops",
            title=f"Daily ops summary — {today_utc}",
            body=body,
            priority=priority,
            tags=["bar_chart"],
        )
        ntfy_status = "sent"
    except Exception as e:
        log.warning("[compute_ops_summary] ntfy failed: %s", e)
        ntfy_status = f"failed: {e}"

    emit(
        run_id=run_id, trigger=trigger, agent="compute_ops_summary",
        event_type="ops_summary_pushed",
        payload={"summary": summary, "ntfy_status": ntfy_status, "priority": priority},
    )

    journal_payload = dict(state.get("journal") or {})
    journal_payload["ops_summary"] = summary
    return {"journal": journal_payload}


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
