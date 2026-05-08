"""Phase 2.5 — real implementation of the proposal/sizing/positions nodes.

Replaces stubs:
    load_open_positions          (moomoo.get_account_info + get_positions)
    research_ticker              (4 analysts via OAuth router, 2-at-a-time)
    build_trade_proposal         (trader-synthesizer LLM)
    deterministic_sizing         (R1-R6 from sizing.py; downsize qty to fit caps)

These leave the rest of the graph (debate, journal RAG, executor, fill capture)
on stubs for now — those are the next Phase-2.5 sub-tasks.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState
from trading_agent.sizing import (
    OpenPosition as SizingOpenPosition,
    ProposedTrade,
    SizingContext,
    blockers,
    check as sizing_check,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — moomoo row → graph state shape
# ---------------------------------------------------------------------------


def _bare_ticker(code: str) -> str:
    """Strip prefix and option strike — `US.SPY260530C00720000` → `SPY`."""
    if not code:
        return ""
    s = code.split(".", 1)[-1]
    m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", s)
    if m:
        return m.group(1)
    return s


def _is_option_code(code: str) -> bool:
    return bool(re.search(r"\d{6,8}[CP]\d+$", code or ""))


def _moomoo_row_to_open_position(row: dict) -> dict:
    """Translate one moomoo position-list row into our state shape."""
    code = row.get("code") or row.get("symbol") or ""
    is_opt = _is_option_code(code)
    qty = float(row.get("qty") or 0.0)
    side = "BUY" if qty >= 0 else "SELL"
    entry = float(row.get("cost_price") or row.get("avg_price") or 0.0)
    mark = float(row.get("nominal_price") or row.get("current_price") or entry)
    bare = _bare_ticker(code)
    from trading_agent import sectors as sectors_lookup
    return {
        "symbol": code,
        "underlying": "US." + bare if "." in code else bare,
        "asset_type": "OPT" if is_opt else "STK",
        "side": side,
        "qty": abs(qty),
        "entry_price": entry,
        "mark": mark,
        "stop": None,
        "target": None,
        "delta": None,
        "gamma": None,
        "vega": None,
        "theta": None,
        "iv": None,
        "notional": abs(qty) * (100.0 if is_opt else 1.0) * mark,
        "unrealized_pnl": float(row.get("pl_val") or 0.0),
        "thesis_id": None,
        "sector": sectors_lookup.lookup(bare),
        "strategy_label": None,
        "age_minutes": 0.0,
    }


# ---------------------------------------------------------------------------
# load_latest_regime — read the most recent regime_states row from Postgres.
# Falls back to safe-mode VOLATILE_TRANSITION when no row exists yet.
# ---------------------------------------------------------------------------


def load_latest_regime(state: TradingGraphState) -> dict:
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/load_latest_regime] run_id=%s", run_id)

    regime_payload: dict[str, Any] = {
        "label": "VOLATILE_TRANSITION",
        "confidence": 0.0,
        "gate": {"size_multiplier": 0.0, "allow_new_entries": False, "require_llm_risk_review": True},
    }
    try:
        from trading_agent.regime.persist import get_latest_regime_state

        latest = get_latest_regime_state()
        if latest:
            regime_payload = {
                "state_id": latest.get("state_id"),
                "label": latest.get("label", "VOLATILE_TRANSITION"),
                "confidence": float(latest.get("confidence") or 0.0),
                "probabilities": latest.get("probabilities") or {},
                "crisis_flags": latest.get("crisis_flags") or [],
                "degradation_level": int(latest.get("degradation_level") or 0),
                "gate": latest.get("gate") or regime_payload["gate"],
            }
    except Exception as e:
        log.error("[trade/load_latest_regime] DB read failed: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="load_latest_regime",
        event_type="regime_loaded",
        payload={"label": regime_payload["label"], "confidence": regime_payload["confidence"]},
    )
    return {"regime": regime_payload}


# ---------------------------------------------------------------------------
# load_open_positions
# ---------------------------------------------------------------------------


def load_open_positions(state: TradingGraphState) -> dict:
    """Fetch paper-account positions + equity from moomoo OpenD."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/load_open_positions] run_id=%s", run_id)

    positions: list[dict] = []
    equity: float = 100_000.0
    cash: float = 100_000.0
    fetch_status = "ok"

    try:
        from trading_agent.mcp_servers.moomoo.server import (
            get_account_info,
            get_positions,
        )

        ai = get_account_info()
        if ai.get("rows"):
            row = ai["rows"][0]
            equity = float(row.get("total_assets") or row.get("net_cash_power") or equity)
            cash = float(row.get("cash") or row.get("avl_withdrawal_cash") or cash)

        pl = get_positions()
        for r in pl.get("rows") or []:
            try:
                positions.append(_moomoo_row_to_open_position(r))
            except Exception as e:
                log.warning("skip bad position row: %s", e)
    except Exception as e:
        log.error("[trade/load_open_positions] moomoo fetch failed: %s", e)
        fetch_status = "failed"

    emit(
        run_id=run_id, trigger=trigger, agent="load_open_positions",
        event_type="positions_loaded",
        payload={"n_positions": len(positions), "equity": equity, "fetch_status": fetch_status},
    )

    return {
        "positions": positions,
        "account": {"equity": equity, "cash": cash, "fetch_status": fetch_status},
    }


# ---------------------------------------------------------------------------
# research_ticker — 4 analysts via OAuth router
# ---------------------------------------------------------------------------


def research_ticker(state: TradingGraphState) -> dict:
    """Run 4 analysts (technical / fundamental / news / sentiment) for ONE ticker.

    Concurrency: max 2 in parallel (matches `OAuthLLMRouter` 2-vCPU semaphore +
    Codex CLI startup overhead). We run two pairs sequentially:
      pair-1: technical_analyst (Sonnet) + news_analyst (Haiku)
      pair-2: sentiment_analyst (Haiku) + fundamental_analyst (Codex)
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    target = (state.get("research") or {}).get("target_ticker") or _first_candidate_ticker(state)
    if not target:
        log.warning("[trade/research_ticker] no target ticker in state — skipping")
        return {}

    log.info("[trade/research_ticker] run_id=%s ticker=%s", run_id, target)

    regime = state.get("regime") or {}
    regime_label = regime.get("label", "VOLATILE_TRANSITION")
    regime_conf = regime.get("confidence", 0.5)
    allow_new = regime.get("gate", {}).get("allow_new_entries", True)

    # Pre-fetch the data each analyst needs so they don't have to call MCP tools.
    klines_block = _prefetch_klines_block(target)
    fundamentals_block = _prefetch_fundamentals_block(target)
    news_block = _prefetch_news_block(target)
    insider_block = _prefetch_insider_block(target)

    common_header = (
        f"ticker: {target}\n"
        f"current_regime:\n  label: {regime_label}\n  confidence: {regime_conf:.2f}\n"
        f"  allow_new_entries: {allow_new}\n"
        f"lookback_days: 60\n"
    )

    prompt_for = {
        "technical_analyst": common_header + "\n" + klines_block + "\n\nRespond per your schema.",
        "news_analyst": common_header + "\n" + news_block + "\n\nRespond per your schema.",
        "sentiment_analyst": common_header + "\n" + insider_block + "\n\nRespond per your schema.",
        "fundamental_analyst": common_header + "\n" + fundamentals_block + "\n\nRespond per your schema.",
    }

    from trading_agent.llm import get_router

    router = get_router()
    reports: dict[str, Any] = {"target_ticker": target, "skipped": []}

    pairs = [
        ("technical_analyst", "news_analyst"),
        ("sentiment_analyst", "fundamental_analyst"),
    ]

    for pair in pairs:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {
                ex.submit(_call_role_safe, router, role, prompt_for[role]): role for role in pair
            }
            for fut in futs:
                role = futs[fut]
                try:
                    parsed = fut.result()
                    reports[role] = parsed
                except Exception as e:
                    log.warning("[research_ticker] %s failed: %s", role, e)
                    reports["skipped"].append({"role": role, "reason": str(e)[:200]})

    # Stash the raw option chain text block alongside the analyst reports so
    # the trader-synthesizer sees it verbatim (analysts' structured outputs
    # drop the per-contract moomoo codes).
    reports["_option_chain_block"] = klines_block

    emit(
        run_id=run_id, trigger=trigger, agent="research_ticker",
        event_type="research_completed",
        payload={
            "ticker": target,
            "roles_succeeded": [k for k in (
                "technical_analyst", "news_analyst", "sentiment_analyst", "fundamental_analyst"
            ) if k in reports],
            "n_skipped": len(reports["skipped"]),
        },
    )
    return {"research": {**(state.get("research") or {}), **reports}}


def _call_role_safe(router, role: str, prompt: str) -> dict | None:
    """Wrap router.call → return parsed dict or None on failure."""
    from trading_agent.llm.schemas import SCHEMA_FOR_ROLE

    schema = SCHEMA_FOR_ROLE.get(role)
    res = router.call(role, prompt, schema=schema, timeout_s=240)
    if res.parsed is None:
        return None
    if hasattr(res.parsed, "model_dump"):
        return res.parsed.model_dump()
    return res.parsed if isinstance(res.parsed, dict) else None


# ---------------------------------------------------------------------------
# Pre-fetch helpers — gather upstream data into compact text blocks the
# analyst LLMs read directly, so they never need to call MCP tools.
# ---------------------------------------------------------------------------


def _prefetch_klines_block(ticker: str) -> str:
    """Recent 1D bars + 30d ATM IV summary, formatted as a compact text block."""
    from datetime import date, timedelta

    code = f"US.{ticker}" if "." not in ticker else ticker
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=90)).isoformat()
    lines = ["klines_1d (most-recent-last):"]
    try:
        from trading_agent.mcp_servers.moomoo.server import get_historical_kline

        bars = get_historical_kline(code, start=start, end=end, ktype="K_DAY", max_count=60).get("rows") or []
        for r in bars[-40:]:
            lines.append(
                f"  {(r.get('time_key') or '')[:10]} O={float(r.get('open', 0)):.2f} "
                f"H={float(r.get('high', 0)):.2f} L={float(r.get('low', 0)):.2f} "
                f"C={float(r.get('close', 0)):.2f} V={int(r.get('volume', 0))}"
            )
    except Exception as e:
        log.warning("klines prefetch failed for %s: %s", ticker, e)
        lines.append("  (no data — moomoo fetch failed)")

    # Bypass Phase-1 snapshot helper (picks 0DTE today's expiry); compose directly
    # with list_option_expiries → pick first ≥7 DTE → get_option_chain → get_quote.
    try:
        chain_block = _prefetch_option_chain(code, target_dte_min=7, target_dte_max=45,
                                              strikes_each_side=4)
        if chain_block:
            lines.append("")
            lines.append(chain_block)
    except Exception as e:
        log.warning("option chain prefetch failed for %s: %s", ticker, e)
    return "\n".join(lines)


def _prefetch_option_chain(
    underlying: str, *, target_dte_min: int = 7, target_dte_max: int = 45,
    strikes_each_side: int = 4,
) -> str:
    """Return a text block of near-money strikes across SEVERAL expiries.

    Each row includes the EXACT moomoo code so the trader can pick one
    verbatim (rather than hallucinating an arbitrary expiry).
    """
    from trading_agent.mcp_servers.moomoo.server import (
        get_option_chain,
        get_quote,
        list_option_expiries,
    )

    expiries = list_option_expiries(underlying).get("rows") or []
    chosen: list[tuple[str, int]] = []
    for r in expiries:
        try:
            dte = int(r.get("option_expiry_date_distance", 0) or 0)
        except Exception:
            dte = 0
        if target_dte_min <= dte <= target_dte_max:
            chosen.append((r.get("strike_time"), dte))
    # Pick up to 3 expiries: shortest, medium, longest within window
    if len(chosen) > 3:
        chosen = [chosen[0], chosen[len(chosen) // 2], chosen[-1]]
    if not chosen:
        return "option_chain_summary: (no expiry in target DTE window)"

    spot = 0.0
    try:
        und_q = get_quote([underlying]).get("rows") or []
        if und_q:
            spot = float(und_q[0].get("last_price") or 0.0)
    except Exception as e:
        log.warning("underlying quote fetch failed: %s", e)

    blocks: list[str] = [f"option_chain_summary: spot={spot:.2f}"]
    blocks.append(
        "  IMPORTANT: when emitting your TradeProposal, the `symbol` field "
        "MUST be one of the moomoo codes below verbatim. Do NOT invent expiry "
        "dates or strike codes — the broker will reject anything off-chain."
    )
    for expiry, dte in chosen:
        chain = get_option_chain(underlying, expiry=expiry).get("rows") or []
        if not chain:
            continue
        by_side: dict[str, list[dict]] = {"CALL": [], "PUT": []}
        for r in chain:
            side = r.get("option_type") or r.get("type")
            strike = float(r.get("strike_price") or 0)
            if side in by_side and strike > 0:
                by_side[side].append({"code": r.get("code"), "strike": strike})

        selected_codes: list[str] = []
        for side in ("CALL", "PUT"):
            rows = by_side[side]
            rows.sort(key=lambda x: abs(x["strike"] - spot))
            for r in rows[:strikes_each_side]:
                selected_codes.append(r["code"])
        if not selected_codes:
            continue
        quotes = get_quote(selected_codes).get("rows") or []
        blocks.append(f"\nexpiry={expiry} dte={dte}:")
        for q in quotes:
            side = q.get("option_type", "?")
            strike = float(q.get("option_strike_price") or 0)
            iv_pct = float(q.get("option_implied_volatility") or 0) / 100.0
            delta = float(q.get("option_delta") or 0)
            oi = int(q.get("option_open_interest") or 0)
            bid = float(q.get("bid_price") or 0)
            ask = float(q.get("ask_price") or 0)
            code = q.get("code", "")
            blocks.append(
                f"  {side} K={strike:.0f} IV={iv_pct:.1%} Δ={delta:+.3f} "
                f"OI={oi} bid={bid:.2f} ask={ask:.2f} code={code}"
            )
    return "\n".join(blocks)


def _prefetch_fundamentals_block(ticker: str) -> str:
    """Latest 10-Q-ish summary. For now best-effort; deeper EDGAR fetch is Phase 2.5b."""
    return (
        "latest_filings:\n"
        f"  ticker: {ticker}\n"
        "  (10-Q/10-K body not yet wired into the graph; respond using only the\n"
        "   ticker name and current_regime context. Mark missing fields with `unknown`\n"
        "   or sensible defaults; set `revenue_yoy_pct` to null and `margin_trend` to\n"
        "   \"UNKNOWN\".)"
    )


def _prefetch_news_block(ticker: str) -> str:
    """Recent SEC filings (8-K / 10-Q / 10-K) with brief metadata."""
    import asyncio
    from trading_agent.mcp_servers.edgar.server import get_recent_filings_for_ticker

    lines = []
    try:
        bare = _bare_ticker(ticker if "." in ticker else f"US.{ticker}")
        result = asyncio.run(
            get_recent_filings_for_ticker(
                bare, limit=10, form_types=["8-K", "10-Q", "10-K"]
            )
        )
        rows = result.get("rows") or []
        if not rows:
            lines.append("recent_filings: [] (none in window)")
        else:
            lines.append(f"recent_filings (count={len(rows)}, newest first):")
            for r in rows[:8]:
                items = ", ".join((r.get("items") or "").split(",")[:4])[:120]
                lines.append(
                    f"  - {r.get('filing_date', '?')} {r.get('form', '?')} "
                    f"acc={r.get('accession_number', '?')} items=[{items}]"
                )
        lines.append("news_cache: [] (Phase 2.5b will wire a news cache)")
        lines.append("next_earnings_date: null  (not yet computed)")
    except Exception as e:
        log.warning("EDGAR news prefetch failed for %s: %s", ticker, e)
        lines = [
            "recent_filings: [] (EDGAR fetch failed)",
            f"  error: {str(e)[:200]}",
            "news_cache: []",
            "next_earnings_date: null",
        ]
    return "\n".join(lines)


def _prefetch_insider_block(ticker: str) -> str:
    """Form 4 (insider transactions) recent filings."""
    import asyncio
    from trading_agent.mcp_servers.edgar.server import get_insider_transactions

    lines = []
    try:
        bare = _bare_ticker(ticker if "." in ticker else f"US.{ticker}")
        result = asyncio.run(get_insider_transactions(bare, limit=15))
        rows = result.get("rows") or []
        if not rows:
            lines.append("insider_transactions: [] (no Form 4 in window)")
        else:
            lines.append(f"insider_transactions (Form 4, count={len(rows)}, newest first):")
            for r in rows[:10]:
                lines.append(
                    f"  - {r.get('filing_date', '?')} acc={r.get('accession_number', '?')} "
                    f"primary={r.get('primary_document', '?')}"
                )
    except Exception as e:
        log.warning("EDGAR insider prefetch failed for %s: %s", ticker, e)
        lines = [f"insider_transactions: [] (EDGAR fetch failed: {str(e)[:200]})"]
    lines.append("rag_lessons: (see past_trades_on_this_ticker section in trader prompt)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# create_or_refresh_thesis
# ---------------------------------------------------------------------------


def create_or_refresh_thesis(state: TradingGraphState) -> dict:
    """Insert a journal_theses row for the current proposal so audit hooks
    have a thesis_id to bind to. Also updates state["proposal"]["thesis_id"]
    so downstream executor + persist_trade_event can reference it.

    No-op if no proposal yet (trader declined or upstream stub didn't run).
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/create_or_refresh_thesis] run_id=%s", run_id)

    proposal = state.get("proposal") or {}
    if not proposal.get("ticker"):
        return {}

    ticker = (proposal.get("ticker") or "").upper()
    direction = proposal.get("direction", "LONG")
    thesis_text = (
        proposal.get("strategy_label", "auto") + " | "
        + (proposal.get("proposal_notes") or "autonomous proposal")
    )[:1500]
    invalidation = (
        f"price closes below stop {proposal.get('stop')} OR regime flips to CRISIS "
        f"OR thesis_invalid_news_within_24h"
    )[:500]
    timeframe = "swing"  # default; trader-synthesizer doesn't yet emit a timeframe field
    expected_return_pct = float(proposal.get("expected_return_pct") or 0.0)
    max_loss_pct = float(proposal.get("max_loss_pct") or 0.0)
    regime_state_id = state.get("regime", {}).get("state_id")

    thesis_id: int | None = None
    try:
        from trading_agent.store.postgres import cursor

        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO journal_theses
                    (created_at, ticker, direction, thesis_text, invalidation,
                     timeframe, return_pct, loss_pct, status, regime_state_id)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, 'open', %s)
                RETURNING id
                """,
                (
                    ticker, direction, thesis_text, invalidation,
                    timeframe, expected_return_pct, max_loss_pct, regime_state_id,
                ),
            )
            thesis_id = int(cur.fetchone()[0])
    except Exception as e:
        log.error("[create_or_refresh_thesis] DB insert failed: %s", e)
        emit(
            run_id=run_id, trigger=trigger, agent="create_or_refresh_thesis",
            event_type="persist_failed", payload={"error": str(e)[:300]}, severity=2,
        )
        return {}

    emit(
        run_id=run_id, trigger=trigger, agent="create_or_refresh_thesis",
        event_type="thesis_recorded",
        payload={"thesis_id": thesis_id, "ticker": ticker, "direction": direction},
    )
    new_proposal = {**proposal, "thesis_id": thesis_id}
    return {"proposal": new_proposal}


# ---------------------------------------------------------------------------
# execute_paper_order + capture_fill
# ---------------------------------------------------------------------------


def _to_moomoo_option_symbol(human: str, *, default_underlying: str = "SPY") -> str | None:
    """Translate human-readable option symbol into moomoo code.

    Accepts:
      - "US.SPY260529C00720000"   (already moomoo format → return as-is)
      - "SPY  260528C00715000"     (already moomoo without prefix → add US.)
      - "SPY 2026-05-29 720C"      (human → translate)
      - "SPY 2026-05-29 $720 Call" (human → translate)
    """
    s = (human or "").strip()
    if not s:
        return None
    if s.startswith("US."):
        return s.replace("  ", " ").replace(" ", "")
    # Already moomoo-style without US. prefix: "SPY  260528C00715000"
    m = re.match(r"^([A-Z\.]+)\s*(\d{6})([CP])(\d{8})$", s.replace(" ", ""))
    if m:
        return f"US.{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"
    # Human form: "SPY 2026-05-29 720C" or "SPY 2026-05-29 $720 Call"
    m = re.match(
        r"^([A-Z\.]+)\s+(\d{4})-(\d{2})-(\d{2})\s+\$?(\d+(?:\.\d+)?)\s*([CcPp])(?:all|ut)?\s*$",
        s,
    )
    if m:
        ticker = m.group(1)
        yy = m.group(2)[-2:]
        mm = m.group(3)
        dd = m.group(4)
        strike = int(round(float(m.group(5)) * 1000))
        cp = m.group(6).upper()
        return f"US.{ticker}{yy}{mm}{dd}{cp}{strike:08d}"
    log.warning("could not parse option symbol: %r", human)
    return None


def execute_paper_order(state: TradingGraphState) -> dict:
    """Place the approved order on the moomoo paper account."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/execute_paper_order] run_id=%s", run_id)

    risk = state.get("risk") or {}
    proposal = state.get("proposal") or {}
    if risk.get("decision") not in ("APPROVE", "DOWNSIZE"):
        log.info("[execute_paper_order] decision=%s — no order to place", risk.get("decision"))
        return {}

    # Phase 2.8 soak gate — READ_ONLY blocks all new entries
    from trading_agent.learning.soak import current_phase, is_new_entry_allowed, tiny_paper_qty_cap
    phase = current_phase()
    if not is_new_entry_allowed(phase):
        log.info("[execute_paper_order] soak_phase=%s blocks new entries", phase.value)
        emit(
            run_id=run_id, trigger=trigger, agent="execute_paper_order",
            event_type="soak_read_only_block",
            payload={"soak_phase": phase.value, "proposal_id": proposal.get("proposal_id")},
        )
        return {}

    approved_qty = int(risk.get("approved_qty") or 0)
    # TINY_PAPER soak phase clamps to 1 contract regardless of sizing decision
    cap = tiny_paper_qty_cap(phase)
    if cap is not None and approved_qty > cap:
        log.info("[execute_paper_order] soak_phase=%s caps qty %d→%d",
                 phase.value, approved_qty, cap)
        approved_qty = cap

    if approved_qty <= 0:
        log.info("[execute_paper_order] approved_qty=0 — nothing to place")
        return {}

    raw_symbol = proposal.get("symbol", "")
    moomoo_symbol = _to_moomoo_option_symbol(raw_symbol)
    if not moomoo_symbol:
        log.error("[execute_paper_order] could not parse symbol %r — aborting", raw_symbol)
        emit(
            run_id=run_id, trigger=trigger, agent="execute_paper_order",
            event_type="symbol_parse_failed",
            payload={"raw_symbol": raw_symbol}, severity=2,
        )
        return {}

    # Cap entry price at the risk decision's max_entry_price (1.005× proposal entry)
    entry_price = float(proposal.get("entry_price") or 0.0)
    max_price = float(risk.get("max_entry_price") or entry_price * 1.005)
    place_price = round(min(entry_price, max_price), 2)

    try:
        from trading_agent.mcp_servers.moomoo.server import place_paper_option_order

        result = place_paper_option_order(
            option_symbol=moomoo_symbol,
            side="BUY",
            contracts=approved_qty,
            price=place_price,
            thesis_id=int(proposal.get("thesis_id") or 0),
            strategy_label=proposal.get("strategy_label"),
            delta=proposal.get("option_delta"),
            dte=proposal.get("option_dte"),
        )
    except Exception as e:
        log.error("[execute_paper_order] place_paper_option_order raised: %s", e)
        emit(
            run_id=run_id, trigger=trigger, agent="execute_paper_order",
            event_type="placement_exception", payload={"error": str(e)[:300]}, severity=2,
        )
        return {}

    rows = result.get("rows") or []
    order_id = None
    if rows:
        order_id = rows[0].get("order_id") or rows[0].get("orderID")

    if not order_id and result.get("virtual_fill_suggested"):
        emit(
            run_id=run_id, trigger=trigger, agent="execute_paper_order",
            event_type="rejected_virtual_fallback",
            payload={"hint": result.get("hint"), "reason": result.get("reason")},
        )
        return {"order": {"placed": False, "virtual": True, "reason": result.get("reason")}}

    emit(
        run_id=run_id, trigger=trigger, agent="execute_paper_order",
        event_type="order_placed",
        payload={
            "order_id": order_id, "symbol": moomoo_symbol, "qty": approved_qty,
            "price": place_price, "side": "BUY",
        },
    )

    try:
        from trading_agent.notify import send as ntfy_send

        ntfy_send(
            "trades",
            title=f"Order placed: {proposal.get('ticker', '?')} {approved_qty}x",
            body=(
                f"symbol: {moomoo_symbol}\n"
                f"order_id: {order_id}\n"
                f"limit: {place_price:.2f}\n"
                f"strategy: {proposal.get('strategy_label')}\n"
                f"thesis_id: {proposal.get('thesis_id')}"
            ),
            priority=4,
            tags=["money_with_wings"],
        )
    except Exception as e:
        log.warning("ntfy trade send failed: %s", e)

    return {
        "order": {
            "placed": True,
            "order_id": order_id,
            "symbol": moomoo_symbol,
            "qty": approved_qty,
            "limit_price": place_price,
            "raw_response": rows,
        }
    }


def persist_trade_event(state: TradingGraphState) -> dict:
    """Insert a journal_trades row tying together proposal + risk_decision +
    regime_state + thesis + broker order. This is the canonical fill record.

    If `state["fill"]` shows the order is FILLED, we record entry_price = avg
    fill price; otherwise we record the limit price as a placeholder and the
    EOD reconciler can update later.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/persist_trade_event] run_id=%s", run_id)

    order = state.get("order") or {}
    proposal = state.get("proposal") or {}
    fill = state.get("fill") or {}
    risk = state.get("risk") or {}
    regime = state.get("regime") or {}
    learning = state.get("learning") or {}

    if not order.get("placed") and not order.get("virtual"):
        log.info("[persist_trade_event] no order placed — skipping")
        return {}

    broker_order_id = str(order.get("order_id") or "")
    symbol = order.get("symbol") or proposal.get("symbol", "")
    asset_type = proposal.get("asset_type", "OPT")
    side = "BUY"
    qty = float(order.get("qty") or risk.get("approved_qty") or 0)
    entry_price = float(fill.get("avg_fill_price") or order.get("limit_price") or 0)

    import json as _json

    try:
        from trading_agent.store.postgres import cursor

        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO journal_trades
                    (thesis_id, symbol, asset_type, side, qty, entry_price,
                     stop, target, outcome, broker_order_id, is_paper, opened_at,
                     proposal_id, risk_decision_id, params_version_id,
                     entry_regime_state_id, broker_fill_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s, TRUE, NOW(),
                        %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (
                    proposal.get("thesis_id"), symbol, asset_type, side, qty,
                    entry_price, proposal.get("stop"), proposal.get("target"),
                    broker_order_id, proposal.get("proposal_id"),
                    risk.get("risk_decision_id"),
                    learning.get("params_version_id"),
                    regime.get("state_id"),
                    _json.dumps(
                        {
                            **fill,
                            "raw_response": order.get("raw_response"),
                            # Phase 2.6.5 — stash a few proposal fields for the
                            # diversity-archive cell calculation post-close
                            "option_dte": proposal.get("option_dte"),
                            "option_iv": proposal.get("option_iv"),
                            "option_delta": proposal.get("option_delta"),
                            "strategy_label": proposal.get("strategy_label"),
                        },
                        default=str,
                    ),
                ),
            )
            trade_id = int(cur.fetchone()[0])
    except Exception as e:
        log.error("[persist_trade_event] DB insert failed: %s", e)
        emit(
            run_id=run_id, trigger=trigger, agent="persist_trade_event",
            event_type="persist_failed", payload={"error": str(e)[:300]}, severity=2,
        )
        return {}

    emit(
        run_id=run_id, trigger=trigger, agent="persist_trade_event",
        event_type="trade_recorded",
        payload={
            "trade_id": trade_id, "broker_order_id": broker_order_id,
            "symbol": symbol, "qty": qty, "entry_price": entry_price,
            "thesis_id": proposal.get("thesis_id"),
        },
    )
    return {"order": {**order, "trade_id": trade_id}}


def ntfy_trade_event(state: TradingGraphState) -> dict:
    """Send the final trade-fill notification (in addition to the
    placement notification fired inside execute_paper_order)."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/ntfy_trade_event] run_id=%s", run_id)

    order = state.get("order") or {}
    fill = state.get("fill") or {}
    proposal = state.get("proposal") or {}

    if not order.get("placed"):
        return {}

    status = fill.get("status", "UNKNOWN")
    fill_qty = float(fill.get("fill_qty") or 0)
    avg_px = float(fill.get("avg_fill_price") or 0)

    if status == "FILLED_ALL":
        title = f"Filled: {proposal.get('ticker', '?')} {fill_qty:.0f}@{avg_px:.2f}"
        priority = 4
        tags = ["money_with_wings", "white_check_mark"]
    elif status in ("CANCELLED_ALL", "FAILED"):
        title = f"Order {status}: {proposal.get('ticker', '?')}"
        priority = 4
        tags = ["x", "warning"]
    else:
        title = f"Order {status}: {proposal.get('ticker', '?')}"
        priority = 3
        tags = ["clock1"]

    body = (
        f"symbol: {order.get('symbol')}\n"
        f"order_id: {order.get('order_id')}\n"
        f"qty: {order.get('qty')} @ limit {order.get('limit_price')}\n"
        f"fill_qty: {fill_qty} @ avg {avg_px:.2f}\n"
        f"strategy: {proposal.get('strategy_label')}\n"
        f"thesis_id: {proposal.get('thesis_id')}"
    )

    try:
        from trading_agent.notify import send as ntfy_send

        ntfy_send("trades", title=title, body=body, priority=priority, tags=tags)
    except Exception as e:
        log.warning("ntfy_trade_event send failed: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="ntfy_trade_event",
        event_type="trade_notification_sent",
        payload={"status": status, "fill_qty": fill_qty, "avg_px": avg_px},
    )
    return {}


def capture_fill(state: TradingGraphState) -> dict:
    """Poll moomoo briefly for a fill on the just-placed order. Records final state."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/capture_fill] run_id=%s", run_id)

    order = state.get("order") or {}
    order_id = order.get("order_id")
    if not order_id:
        log.info("[capture_fill] no order_id in state — skipping")
        return {}

    import time

    final_status = "UNKNOWN"
    fill_qty = 0.0
    avg_fill_price = 0.0
    try:
        from trading_agent.mcp_servers.moomoo.server import get_orders

        # Brief polling: 5 attempts × 2 s = 10 s. After-hours fills are rare;
        # we don't block the orchestrator for long. EOD reconciler picks up later.
        for _ in range(5):
            rows = get_orders().get("rows") or []
            row = next((r for r in rows if str(r.get("order_id")) == str(order_id)), None)
            if row:
                final_status = str(row.get("order_status") or "UNKNOWN")
                fill_qty = float(row.get("dealt_qty") or 0)
                avg_fill_price = float(row.get("dealt_avg_price") or 0)
                if final_status in ("FILLED_ALL", "CANCELLED_ALL", "FAILED"):
                    break
            time.sleep(2.0)
    except Exception as e:
        log.warning("capture_fill poll failed: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="capture_fill",
        event_type="fill_captured",
        payload={
            "order_id": order_id, "status": final_status,
            "fill_qty": fill_qty, "avg_fill_price": avg_fill_price,
        },
    )
    return {
        "fill": {
            "order_id": order_id,
            "status": final_status,
            "fill_qty": fill_qty,
            "avg_fill_price": avg_fill_price,
        }
    }


def _first_candidate_ticker(state: dict) -> str | None:
    cands = state.get("candidates") or []
    if cands:
        c = cands[0]
        if isinstance(c, dict):
            return c.get("ticker")
    return None


# ---------------------------------------------------------------------------
# researcher_debate — bull + bear pods (≤2 rounds, regime-gated)
# ---------------------------------------------------------------------------


def researcher_debate(state: TradingGraphState) -> dict:
    """Run a structured bull-vs-bear debate over the analyst reports.

    Constraints (from plan §5.2 and §11.6):
    - ≤2 rounds; round 2 only if budget allows AND round 1 didn't converge
    - Skipped entirely in CRISIS regime
    - Bull = Claude Sonnet, Bear = Codex GPT-5.5 (model-family diversity)

    Stores transcript + a convergence summary in `state["debate"]`.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/researcher_debate] run_id=%s", run_id)

    regime = state.get("regime") or {}
    if regime.get("label") == "CRISIS":
        emit(
            run_id=run_id, trigger=trigger, agent="researcher_debate",
            event_type="skipped", payload={"reason": "regime_crisis"},
        )
        return {}

    research = state.get("research") or {}
    target = research.get("target_ticker") or _first_candidate_ticker(state)
    if not target:
        return {}

    from trading_agent.llm import get_router
    from trading_agent.llm.schemas import BearResearcherOutput, BullResearcherOutput

    router = get_router()
    base_ctx = _format_debate_context(target, research, regime)

    bull_history: list[dict] = []
    bear_history: list[dict] = []
    rounds_done = 0

    for rnd in (1, 2):
        bull_prompt = base_ctx + (
            f"\nround_number: {rnd}\n"
            + (f"bear_last_turn:\n{bear_history[-1] if bear_history else 'none'}\n" if rnd > 1 else "")
            + "\nRespond per your output schema."
        )
        try:
            res = router.call("bull_researcher", bull_prompt, schema=BullResearcherOutput, timeout_s=180)
            bull_parsed = res.parsed.model_dump() if res.parsed and hasattr(res.parsed, "model_dump") else None
        except Exception as e:
            log.warning("[debate] bull round %d failed: %s", rnd, e)
            bull_parsed = None
        if bull_parsed:
            bull_history.append(bull_parsed)

        bear_prompt = base_ctx + (
            f"\nround_number: {rnd}\n"
            + (f"bull_last_turn:\n{bull_history[-1] if bull_history else 'none'}\n")
            + "\nRespond per your output schema."
        )
        try:
            res = router.call("bear_researcher", bear_prompt, schema=BearResearcherOutput, timeout_s=180)
            bear_parsed = res.parsed.model_dump() if res.parsed and hasattr(res.parsed, "model_dump") else None
        except Exception as e:
            log.warning("[debate] bear round %d failed: %s", rnd, e)
            bear_parsed = None
        if bear_parsed:
            bear_history.append(bear_parsed)

        rounds_done = rnd
        # Skip round 2 if regime is BEAR/TRANSITION (cost discipline) or if both
        # sides already concede the same conclusion.
        if regime.get("label") in {"BEAR_TREND", "VOLATILE_TRANSITION"}:
            break
        # Convergence: round 1 produced clear positions on both sides → call it done
        if rnd == 1 and bull_parsed and bear_parsed:
            break

    emit(
        run_id=run_id, trigger=trigger, agent="researcher_debate",
        event_type="debate_done",
        payload={
            "ticker": target, "rounds": rounds_done,
            "bull_turns": len(bull_history), "bear_turns": len(bear_history),
        },
    )

    return {
        "debate": {
            "ticker": target,
            "rounds_done": rounds_done,
            "bull_history": bull_history,
            "bear_history": bear_history,
        }
    }


def _format_debate_context(target: str, research: dict, regime: dict) -> str:
    import json as _json
    parts = [
        f"ticker: {target}",
        f"current_regime:",
        f"  label: {regime.get('label', 'UNKNOWN')}",
        f"  confidence: {regime.get('confidence', 0):.2f}",
    ]
    for role in ("technical_analyst", "news_analyst", "sentiment_analyst", "fundamental_analyst"):
        rep = research.get(role)
        if rep:
            parts.append(f"\n--- {role} report ---")
            parts.append(_json.dumps(rep, indent=2)[:1800])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# retrieve_past_lessons — Postgres-backed RAG over journal_trades
# ---------------------------------------------------------------------------


def retrieve_past_lessons(state: TradingGraphState) -> dict:
    """Pull similar past trades + closed-trade outcomes from Postgres so the
    trader has historical context. Phase-2.5 simple version: filter by
    ticker + strategy_label; richer vector search lands in Phase 2.6.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/retrieve_past_lessons] run_id=%s", run_id)

    research = state.get("research") or {}
    target = research.get("target_ticker") or _first_candidate_ticker(state)
    if not target:
        return {}

    bare = _bare_ticker(target if "." in target else f"US.{target}")
    lessons: list[dict] = []
    try:
        from trading_agent.store.postgres import cursor

        with cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.symbol, t.qty, t.entry_price, t.exit_price,
                       t.outcome, t.close_reason, t.opened_at, t.closed_at,
                       th.direction, th.thesis_text
                FROM journal_trades t
                LEFT JOIN journal_theses th ON th.id = t.thesis_id
                WHERE t.symbol LIKE %s
                ORDER BY t.opened_at DESC
                LIMIT 5
                """,
                (f"%{bare}%",),
            )
            for row in cur.fetchall():
                lessons.append({
                    "trade_id": row[0], "symbol": row[1], "qty": float(row[2] or 0),
                    "entry": float(row[3] or 0), "exit": float(row[4] or 0) if row[4] else None,
                    "outcome": row[5], "close_reason": row[6],
                    "opened_at": str(row[7]), "closed_at": str(row[8]) if row[8] else None,
                    "direction": row[9], "thesis": (row[10] or "")[:300],
                })
    except Exception as e:
        log.warning("[retrieve_past_lessons] DB read failed: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="retrieve_past_lessons",
        event_type="lessons_retrieved",
        payload={"ticker": bare, "n_lessons": len(lessons)},
    )
    return {"research": {**research, "rag_lessons": lessons}}


# ---------------------------------------------------------------------------
# build_trade_proposal — trader-synthesizer (Opus)
# ---------------------------------------------------------------------------


def build_trade_proposal(state: TradingGraphState) -> dict:
    """Call trader-synthesizer to produce a typed TradeProposal.

    On decline_to_trade=true → returns no proposal (later nodes see no
    state["proposal"] and the chain naturally routes to persist_defer).
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/build_trade_proposal] run_id=%s", run_id)

    research = state.get("research") or {}
    regime = state.get("regime") or {}
    account = state.get("account") or {"equity": 100_000.0}
    positions = state.get("positions") or []

    target = research.get("target_ticker") or _first_candidate_ticker(state)
    if not target:
        log.warning("[build_trade_proposal] no target ticker — declining")
        return {}

    # Stuff the debate transcript into `research` so _format_trader_prompt sees it.
    research = {**research, "_debate": state.get("debate")}

    prompt = _format_trader_prompt(
        ticker=target, research=research, regime=regime,
        account_equity=float(account.get("equity", 100_000.0)),
        n_positions=len(positions),
        heat_pct=_estimate_heat_pct(positions, account),
    )

    from trading_agent.llm import get_router
    from trading_agent.llm.schemas import TraderSynthesizerOutput

    router = get_router()
    res = router.call("trader_synthesizer", prompt, schema=TraderSynthesizerOutput, timeout_s=180)
    parsed: TraderSynthesizerOutput | None = (
        res.parsed if isinstance(res.parsed, TraderSynthesizerOutput) else None
    )

    if parsed is None or parsed.decline_to_trade or parsed.proposal is None:
        emit(
            run_id=run_id, trigger=trigger, agent="build_trade_proposal",
            event_type="declined",
            payload={
                "ticker": target,
                "reason": (parsed.decline_reason if parsed else "no_parsed_output"),
            },
        )
        return {"research": {**research, "trader_decline": True}}

    p = parsed.proposal
    proposal_dict = {
        "proposal_id": f"prop_{run_id[:30]}",
        "ticker": p.ticker,
        "symbol": p.symbol,
        "asset_type": p.asset_type,
        "direction": p.direction,
        "side": "BUY",
        "qty": float(p.qty_request),
        "entry_price": float(p.entry_price),
        "strategy_label": p.strategy_label,
        "thesis_id": 0,  # populated by create_or_refresh_thesis when wired
        "params_version_id": 0,
        "stop": float(p.stop),
        "target": float(p.target),
        "expected_return_pct": float(p.expected_return_pct),
        "max_loss_pct": float(p.max_loss_pct),
    }
    if p.option_delta is not None:
        proposal_dict["option_delta"] = float(p.option_delta)
    if p.option_dte is not None:
        proposal_dict["option_dte"] = int(p.option_dte)
    if p.option_iv is not None:
        proposal_dict["option_iv"] = float(p.option_iv)

    emit(
        run_id=run_id, trigger=trigger, agent="build_trade_proposal",
        event_type="proposal_built",
        payload={
            "ticker": p.ticker, "symbol": p.symbol, "qty": p.qty_request,
            "direction": p.direction, "strategy": p.strategy_label,
        },
    )
    return {"proposal": proposal_dict}


def _format_trader_prompt(
    *, ticker: str, research: dict, regime: dict,
    account_equity: float, n_positions: int, heat_pct: float,
) -> str:
    parts = [
        f"ticker: {ticker}",
        f"current_regime:",
        f"  label: {regime.get('label', 'UNKNOWN')}",
        f"  confidence: {regime.get('confidence', 0):.2f}",
        f"  allow_new_entries: {regime.get('gate', {}).get('allow_new_entries', True)}",
        f"  size_multiplier: {regime.get('gate', {}).get('size_multiplier', 1.0):.2f}",
        f"account_state:",
        f"  equity: {account_equity:.0f}",
        f"  current_heat_pct: {heat_pct:.2%}",
        f"  open_positions_count: {n_positions}",
    ]
    import json as _json
    for role in ("technical_analyst", "news_analyst", "sentiment_analyst", "fundamental_analyst"):
        rep = research.get(role)
        if rep:
            parts.append(f"\n--- {role} report ---")
            parts.append(_json.dumps(rep, indent=2)[:3000])

    # Bull/Bear debate transcript
    parts.append(_format_debate_for_trader(research))

    # RAG lessons from past trades on this ticker
    lessons = research.get("rag_lessons") or []
    if lessons:
        parts.append("\n--- past_trades_on_this_ticker (most-recent-first) ---")
        for L in lessons[:5]:
            parts.append(_json.dumps(L)[:600])

    # Pass the raw market-data block (klines + option chain with moomoo codes)
    # so the trader can pick a verbatim contract code.
    chain_block = research.get("_option_chain_block")
    if chain_block:
        parts.append("\n--- raw market data (option chain has the moomoo codes you must use verbatim) ---")
        parts.append(chain_block[:6000])

    parts.append("\nRespond per your output schema.")
    return "\n".join(parts)


def _format_debate_for_trader(research: dict) -> str:
    """The debate state is stored on `state['debate']`, but build_trade_proposal
    only sees `research`. Caller stuffs the latest debate dict into research
    before invoking the trader (see build_trade_proposal flow). If absent,
    return empty so the trader knows debate didn't run."""
    deb = research.get("_debate")
    if not deb:
        return "\n--- debate ---\n(debate skipped or not yet wired)"
    import json as _json
    return (
        f"\n--- debate ({deb.get('rounds_done', 0)} rounds) ---\n"
        f"bull_history: {_json.dumps(deb.get('bull_history') or [])[:2000]}\n"
        f"bear_history: {_json.dumps(deb.get('bear_history') or [])[:2000]}"
    )


def _estimate_heat_pct(positions: list[dict], account: dict) -> float:
    eq = float(account.get("equity") or 1.0)
    total = 0.0
    for p in positions:
        if p.get("asset_type") == "OPT":
            total += float(p.get("entry_price") or 0.0) * float(p.get("qty") or 0.0) * 100.0
        else:
            stop = p.get("stop")
            if stop and stop > 0 and stop < (p.get("entry_price") or 0):
                total += (float(p["entry_price"]) - float(stop)) * float(p.get("qty") or 0.0)
            else:
                total += 0.05 * float(p.get("entry_price") or 0.0) * float(p.get("qty") or 0.0)
    return total / eq if eq > 0 else 0.0


# ---------------------------------------------------------------------------
# deterministic_sizing — R1-R6
# ---------------------------------------------------------------------------


def deterministic_sizing(state: TradingGraphState) -> dict:
    """Apply R1-R6. Downsize qty to the maximum that passes; if zero passes, mark proposal infeasible."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/deterministic_sizing] run_id=%s", run_id)

    proposal = state.get("proposal")
    if not proposal:
        log.info("[deterministic_sizing] no proposal in state — skipping")
        return {}

    account = state.get("account") or {"equity": 100_000.0}
    positions = state.get("positions") or []
    equity = float(account.get("equity", 100_000.0))

    from trading_agent import sectors as sectors_lookup
    sectors_loaded = sectors_lookup.known_count() > 0

    open_positions = tuple(
        SizingOpenPosition(
            symbol=p.get("symbol", ""),
            ticker=_bare_ticker(p.get("symbol", "")),
            asset_type=p.get("asset_type", "STK"),
            qty=float(p.get("qty", 0)),
            entry_price=float(p.get("entry_price", 0)),
            stop=p.get("stop"),
            sector=p.get("sector") or sectors_lookup.lookup(_bare_ticker(p.get("symbol", ""))),
            strategy_label=p.get("strategy_label"),
        )
        for p in positions
    )
    ctx = SizingContext(equity=equity, opens=open_positions, sector_lookup_available=sectors_loaded)

    requested_qty = float(proposal.get("qty", 0))
    final_qty = 0.0
    final_violations: list = []

    if requested_qty <= 0:
        emit(
            run_id=run_id, trigger=trigger, agent="deterministic_sizing",
            event_type="zero_qty_proposal", payload={"ticker": proposal.get("ticker", "?")},
        )
        return {"sizing": {"r1_r6_violations": [], "approved_qty": 0.0, "infeasible": True}}

    # Soak TINY_PAPER cap: hard ceiling before R1-R6 candidates are tried
    try:
        from trading_agent.learning.soak import tiny_paper_qty_cap
        soak_cap = tiny_paper_qty_cap()
        if soak_cap is not None and requested_qty > soak_cap:
            emit(
                run_id=run_id, trigger=trigger, agent="deterministic_sizing",
                event_type="tiny_paper_cap_applied",
                payload={"requested": requested_qty, "cap": soak_cap},
            )
            requested_qty = float(soak_cap)
    except Exception as e:
        log.warning("[deterministic_sizing] soak cap check failed: %s", e)

    # Try requested qty, then 75%, 50%, 25%, 10% — coarse but sufficient for R1
    candidate_qtys = [
        requested_qty,
        max(1, int(requested_qty * 0.75)),
        max(1, int(requested_qty * 0.5)),
        max(1, int(requested_qty * 0.25)),
        max(1, int(requested_qty * 0.1)),
    ]
    seen: set[int] = set()
    for q in candidate_qtys:
        qi = int(q)
        if qi in seen or qi <= 0:
            continue
        seen.add(qi)
        proposed = ProposedTrade(
            ticker=proposal.get("ticker", ""),
            asset_type=proposal.get("asset_type", "OPT"),
            side="BUY",
            qty=float(qi),
            entry_price=float(proposal.get("entry_price", 0)),
            stop=proposal.get("stop"),
            strategy_label=proposal.get("strategy_label"),
            sector=proposal.get("sector") or sectors_lookup.lookup(proposal.get("ticker", "")),
            delta=proposal.get("option_delta"),
            dte=proposal.get("option_dte"),
        )
        violations = sizing_check(ctx, proposed)
        if not blockers(violations):
            final_qty = float(qi)
            final_violations = violations
            break
        final_violations = violations  # keep last for reporting

    new_proposal = {**proposal, "qty": final_qty}
    if final_qty == 0:
        emit(
            run_id=run_id, trigger=trigger, agent="deterministic_sizing",
            event_type="infeasible",
            payload={
                "ticker": proposal.get("ticker", "?"),
                "blockers": [b.rule for b in blockers(final_violations)],
            },
        )
        return {
            "proposal": new_proposal,
            "sizing": {
                "r1_r6_violations": [
                    {"rule": v.rule, "severity": v.severity, "message": v.message[:200]}
                    for v in final_violations
                ],
                "approved_qty": 0.0,
                "infeasible": True,
            },
        }

    emit(
        run_id=run_id, trigger=trigger, agent="deterministic_sizing",
        event_type="sized",
        payload={
            "ticker": proposal.get("ticker", "?"),
            "requested_qty": requested_qty,
            "approved_qty": final_qty,
            "warnings": [v.rule for v in final_violations if v.severity == "warn"],
        },
    )
    return {
        "proposal": new_proposal,
        "sizing": {
            "r1_r6_violations": [
                {"rule": v.rule, "severity": v.severity, "message": v.message[:200]}
                for v in final_violations
            ],
            "approved_qty": final_qty,
            "infeasible": False,
        },
    }


# ---------------------------------------------------------------------------
# regime_execution_gate
# ---------------------------------------------------------------------------


def regime_execution_gate(state: TradingGraphState) -> dict:
    """Apply the regime gate (size_multiplier + allow_new_entries) + soak-phase gate."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/regime_execution_gate] run_id=%s", run_id)

    proposal = state.get("proposal")
    if not proposal:
        return {}

    # Soak-phase gate: READ_ONLY blocks all new entries regardless of regime
    try:
        from trading_agent.learning.soak import is_new_entry_allowed, current_phase
        soak_phase = current_phase()
        if not is_new_entry_allowed(soak_phase):
            emit(
                run_id=run_id, trigger=trigger, agent="regime_execution_gate",
                event_type="blocked",
                payload={"reason": "soak_read_only", "soak_phase": soak_phase.value},
            )
            return {"proposal": {**proposal, "qty": 0.0}}
    except Exception as e:
        log.warning("[regime_execution_gate] soak check failed: %s", e)

    regime = state.get("regime") or {}
    gate = regime.get("gate") or {}
    if not gate.get("allow_new_entries", True):
        emit(
            run_id=run_id, trigger=trigger, agent="regime_execution_gate",
            event_type="blocked", payload={"reason": "regime_blocks_new_entries", "label": regime.get("label")},
        )
        return {"proposal": {**proposal, "qty": 0.0}}

    mult = float(gate.get("size_multiplier", 1.0))
    qty = float(proposal.get("qty", 0))
    new_qty = qty * mult
    if proposal.get("asset_type") == "OPT":
        new_qty = float(int(new_qty))

    if new_qty != qty:
        emit(
            run_id=run_id, trigger=trigger, agent="regime_execution_gate",
            event_type="downsized",
            payload={"requested": qty, "size_multiplier": mult, "approved": new_qty,
                     "label": regime.get("label")},
        )
    return {"proposal": {**proposal, "qty": new_qty}}


# ---------------------------------------------------------------------------
# Entry-termination nodes — VETO / DEFER paths
# ---------------------------------------------------------------------------

def persist_veto(state: TradingGraphState) -> dict:
    """Persist a VETO outcome to agent_events + risk_decisions audit trail."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/persist_veto] run_id=%s", run_id)

    risk = state.get("risk") or {}
    proposal = state.get("proposal") or {}
    hard_violations = risk.get("hard_violations") or []
    reasons = risk.get("reasons") or []

    emit(
        run_id=run_id, trigger=trigger, agent="persist_veto",
        event_type="veto_persisted",
        severity=1,  # warn — a veto is notable
        payload={
            "ticker": proposal.get("ticker", "?"),
            "proposal_id": proposal.get("proposal_id"),
            "hard_violations": hard_violations,
            "reasons": reasons,
        },
    )
    return {}


def ntfy_risk_block(state: TradingGraphState) -> dict:
    """Push a ntfy notification for a VETO-blocked trade."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/ntfy_risk_block] run_id=%s", run_id)

    risk = state.get("risk") or {}
    proposal = state.get("proposal") or {}
    ticker = proposal.get("ticker", "?")
    hard_violations = risk.get("hard_violations") or []
    reasons = risk.get("reasons") or []

    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="risk",
            title=f"VETO — {ticker}",
            body=(
                f"Trade blocked for {ticker}.\n"
                f"Hard violations: {', '.join(str(v) for v in hard_violations) or 'none'}\n"
                f"Reasons: {'; '.join(str(r) for r in reasons[:3])}"
            ),
            priority=4,
            tags=["no_entry", "rotating_light"],
        )
    except Exception as e:
        log.warning("[ntfy_risk_block] ntfy failed: %s", e)

    emit(run_id=run_id, trigger=trigger, agent="ntfy_risk_block",
         event_type="veto_notification_sent", payload={"ticker": ticker})
    return {}


def persist_defer(state: TradingGraphState) -> dict:
    """Persist a DEFER outcome — market conditions not suitable right now."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/persist_defer] run_id=%s", run_id)

    risk = state.get("risk") or {}
    proposal = state.get("proposal") or {}
    reasons = risk.get("reasons") or []

    emit(
        run_id=run_id, trigger=trigger, agent="persist_defer",
        event_type="defer_persisted",
        payload={
            "ticker": proposal.get("ticker", "?"),
            "proposal_id": proposal.get("proposal_id"),
            "reasons": reasons,
        },
    )
    return {}


def ntfy_defer(state: TradingGraphState) -> dict:
    """Push a ntfy notification for a DEFER decision."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[trade/ntfy_defer] run_id=%s", run_id)

    risk = state.get("risk") or {}
    proposal = state.get("proposal") or {}
    ticker = proposal.get("ticker", "?")
    reasons = risk.get("reasons") or []

    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="risk",
            title=f"DEFER — {ticker}",
            body=(
                f"Entry deferred for {ticker}.\n"
                f"Reasons: {'; '.join(str(r) for r in reasons[:3]) or 'regime or risk conditions'}"
            ),
            priority=2,
            tags=["hourglass", "chart_with_upwards_trend"],
        )
    except Exception as e:
        log.warning("[ntfy_defer] ntfy failed: %s", e)

    emit(run_id=run_id, trigger=trigger, agent="ntfy_defer",
         event_type="defer_notification_sent", payload={"ticker": ticker})
    return {}
