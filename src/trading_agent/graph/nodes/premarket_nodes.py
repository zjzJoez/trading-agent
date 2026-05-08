"""Phase 2.5 — Premarket scan downstream nodes.

Replaces stubs:
    collect_watchlist_data    fetch quotes + basic metrics for watchlist tickers
    rank_candidates           call scout LLM to score + filter to top-N
    ntfy_scan_digest          push ranked candidates to ntfy trades topic

Pipeline (premarket_scan_graph):
    collect_macro_market_data → compute_regime_features → classify_regime
    → maybe_llm_regime_review → persist_regime
    → collect_watchlist_data    (this file)
    → rank_candidates           (this file)
    → ntfy_scan_digest          (this file)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_quote_for_ticker(ticker: str) -> dict[str, Any]:
    """Return a minimal quote dict for one ticker.  Never raises."""
    symbol = f"US.{ticker}" if not ticker.startswith("US.") else ticker
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
        qr = get_quote(symbol=symbol)
        rows = qr.get("rows") or []
        if rows:
            r = rows[0]
            return {
                "ticker": ticker,
                "symbol": symbol,
                "last": float(r.get("last_price") or 0),
                "volume": float(r.get("volume") or 0),
                "turnover": float(r.get("turnover") or 0),
                "change_pct": float(r.get("change_rate") or r.get("price_spread") or 0),
                "high": float(r.get("high_price") or 0),
                "low": float(r.get("low_price") or 0),
                "open": float(r.get("open_price") or 0),
                "prev_close": float(r.get("prev_close_price") or 0),
            }
    except Exception as e:
        log.warning("[premarket] quote %s failed: %s", ticker, e)
    return {"ticker": ticker, "symbol": symbol, "last": 0.0, "error": "fetch_failed"}


def _fetch_recent_filing_headline(ticker: str) -> str | None:
    """Best-effort: get the most recent EDGAR filing type for context."""
    try:
        from trading_agent.mcp_servers.edgar.server import get_recent_filings_for_ticker
        result = get_recent_filings_for_ticker(ticker=ticker, limit=1)
        filings = result.get("filings") or []
        if filings:
            f = filings[0]
            return f"{f.get('form','?')} filed {f.get('filed','?')}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Node 1: collect_watchlist_data
# ---------------------------------------------------------------------------

def collect_watchlist_data(state: TradingGraphState) -> dict:
    """Fetch live quotes for every ticker in state["watchlist"].

    If watchlist is empty, falls back to a default set of liquid large-caps
    so the premarket scan always has something to rank.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[premarket/collect_watchlist_data] run_id=%s", run_id)

    watchlist: list[str] = list(state.get("watchlist") or [])
    if not watchlist:
        # Default watchlist — broad coverage across sectors
        watchlist = [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META",
            "GOOGL", "TSLA", "JPM", "GLD", "TLT", "XLE", "XLF",
        ]
        log.info("[premarket/collect_watchlist_data] using default watchlist (%d tickers)", len(watchlist))

    quote_data: dict[str, Any] = {}
    for ticker in watchlist:
        quote_data[ticker] = _fetch_quote_for_ticker(ticker)

    # Also grab one recent EDGAR filing per ticker (best-effort, sequential)
    for ticker in watchlist[:6]:  # limit to top 6 to avoid rate-limit
        headline = _fetch_recent_filing_headline(ticker)
        if headline:
            quote_data[ticker]["recent_filing"] = headline

    emit(
        run_id=run_id, trigger=trigger, agent="collect_watchlist_data",
        event_type="watchlist_data_collected",
        payload={
            "n_tickers": len(watchlist),
            "fetch_errors": sum(1 for v in quote_data.values() if "error" in v),
        },
    )
    return {"market_data": quote_data, "watchlist": watchlist}


# ---------------------------------------------------------------------------
# Node 2: rank_candidates
# ---------------------------------------------------------------------------

def _build_scout_prompt(
    watchlist: list[str],
    market_data: dict[str, Any],
    regime: dict[str, Any],
) -> str:
    regime_label = regime.get("label", "VOLATILE_TRANSITION")
    regime_conf = float(regime.get("confidence") or 0.0)
    gate = regime.get("gate") or {}
    allow_entries = gate.get("allow_new_entries", True)
    size_mult = float(gate.get("size_multiplier") or 1.0)

    lines = [
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"regime: {regime_label} (confidence={regime_conf:.2f})",
        f"allow_new_entries: {allow_entries}",
        f"size_multiplier: {size_mult:.2f}",
        f"",
        f"watchlist_quotes:",
    ]
    for ticker in watchlist:
        q = market_data.get(ticker) or {}
        if q.get("error"):
            lines.append(f"  {ticker}: (quote unavailable)")
            continue
        last = q.get("last", 0)
        chg = q.get("change_pct", 0)
        vol = q.get("volume", 0)
        filing = q.get("recent_filing", "")
        filing_str = f" | {filing}" if filing else ""
        lines.append(
            f"  {ticker}: last={last:.2f} chg={chg:+.2%} vol={vol:.0f}{filing_str}"
        )

    lines += [
        f"",
        f"Rank the watchlist tickers from most to least actionable as of today's open. "
        f"Score 0.0–1.0. Include only tickers where there is a concrete near-term setup "
        f"(momentum, mean-reversion, event-driven, or regime-aligned). "
        f"Skip tickers with no clear edge or that contradict the current regime. "
        f"Return at most 5 candidates. "
        f"Respond per your ScoutOutput schema.",
    ]
    return "\n".join(lines)


def rank_candidates(state: TradingGraphState) -> dict:
    """Call the scout LLM (Haiku) to score and filter the watchlist.

    Output is a ranked list of candidates stored in state["candidates"].
    If the LLM fails, falls back to top-3 by absolute change_pct.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[premarket/rank_candidates] run_id=%s", run_id)

    watchlist = list(state.get("watchlist") or [])
    market_data = dict(state.get("market_data") or {})
    regime = state.get("regime") or {}

    if not watchlist:
        return {}

    prompt = _build_scout_prompt(watchlist, market_data, regime)
    candidates: list[dict] = []

    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import ScoutOutput
        router = get_router()
        res = router.call("scout", prompt, schema=ScoutOutput, timeout_s=120)
        parsed: ScoutOutput | None = (
            res.parsed if isinstance(res.parsed, ScoutOutput) else None
        )
        if parsed and parsed.candidates:
            candidates = [
                {"ticker": c.ticker, "score": c.score, "reason": c.reason}
                for c in sorted(parsed.candidates, key=lambda x: -x.score)
            ]
            skipped = [s.ticker for s in (parsed.skipped or [])]
            log.info("[rank_candidates] top candidates: %s", [c["ticker"] for c in candidates[:3]])
        else:
            log.warning("[rank_candidates] no candidates returned — falling back")
    except Exception as e:
        log.warning("[rank_candidates] scout LLM failed: %s — using change_pct fallback", e)

    # Fallback: rank by |change_pct| if LLM gave nothing
    if not candidates:
        ranked_tickers = sorted(
            watchlist,
            key=lambda t: abs(float((market_data.get(t) or {}).get("change_pct") or 0)),
            reverse=True,
        )
        candidates = [
            {"ticker": t, "score": 0.5, "reason": "fallback_by_abs_change_pct"}
            for t in ranked_tickers[:3]
        ]

    emit(
        run_id=run_id, trigger=trigger, agent="rank_candidates",
        event_type="candidates_ranked",
        payload={
            "n_candidates": len(candidates),
            "top_ticker": candidates[0]["ticker"] if candidates else None,
        },
    )
    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 3: ntfy_scan_digest
# ---------------------------------------------------------------------------

def ntfy_scan_digest(state: TradingGraphState) -> dict:
    """Push a pre-market scan summary to ntfy trades topic."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[premarket/ntfy_scan_digest] run_id=%s", run_id)

    candidates: list[dict] = list(state.get("candidates") or [])
    regime = state.get("regime") or {}
    regime_label = regime.get("label", "UNKNOWN")
    gate = regime.get("gate") or {}
    allow_entries = gate.get("allow_new_entries", True)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not candidates:
        body = f"Regime: {regime_label} | No actionable candidates today."
        title = f"Premarket Scan — {now_utc}"
    else:
        top = candidates[:5]
        lines = [f"Regime: {regime_label} | entries {'OPEN' if allow_entries else 'BLOCKED'}"]
        for i, c in enumerate(top, 1):
            lines.append(f"{i}. {c['ticker']} (score={c['score']:.2f}) — {c['reason'][:80]}")
        body = "\n".join(lines)
        top_ticker = top[0]["ticker"] if top else "?"
        title = f"Scan {now_utc[:10]} — {top_ticker} leads"

    ntfy_status = "skipped"
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="trades",
            title=title,
            body=body,
            priority=3,
            tags=["mag", "chart_with_upwards_trend"],
        )
        ntfy_status = "sent"
    except Exception as e:
        log.warning("[ntfy_scan_digest] ntfy failed: %s", e)
        ntfy_status = f"failed: {e}"

    emit(
        run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
        event_type="scan_digest_sent",
        payload={
            "n_candidates": len(candidates),
            "regime_label": regime_label,
            "allow_new_entries": allow_entries,
            "ntfy_status": ntfy_status,
        },
    )
    return {}


__all__ = [
    "collect_watchlist_data",
    "rank_candidates",
    "ntfy_scan_digest",
]
