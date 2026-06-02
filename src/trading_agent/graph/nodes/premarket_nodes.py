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

def _fetch_quotes_batch(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Batch fetch quotes for multiple tickers; returns ticker → quote dict.
    Never raises."""
    out: dict[str, dict[str, Any]] = {}
    symbols = [f"US.{t}" if not t.startswith("US.") else t for t in tickers]
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
        result = get_quote(symbols)
        rows_by_code: dict[str, dict] = {}
        for r in (result.get("rows") or []):
            code = r.get("code") or r.get("symbol") or ""
            if code:
                rows_by_code[code] = r
        for ticker, sym in zip(tickers, symbols):
            r = rows_by_code.get(sym)
            if r:
                out[ticker] = {
                    "ticker": ticker,
                    "symbol": sym,
                    "last": float(r.get("last_price") or 0),
                    "volume": float(r.get("volume") or 0),
                    "turnover": float(r.get("turnover") or 0),
                    "change_pct": float(r.get("change_rate") or 0),
                    "high": float(r.get("high_price") or 0),
                    "low": float(r.get("low_price") or 0),
                    "open": float(r.get("open_price") or 0),
                    "prev_close": float(r.get("prev_close_price") or 0),
                }
            else:
                out[ticker] = {"ticker": ticker, "symbol": sym, "last": 0.0, "error": "fetch_failed"}
    except Exception as e:
        log.warning("[premarket] get_quote batch failed: %s", e)
        for ticker, sym in zip(tickers, symbols):
            out[ticker] = {"ticker": ticker, "symbol": sym, "last": 0.0, "error": "fetch_failed"}
    return out


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
        # Source of truth for the dynamic watchlist lives in Postgres
        # watchlist_members (migration 010), refreshed daily by
        # scripts/refresh_dynamic_watchlist.py. Reads through the same
        # helper so this graph node and standalone jobs stay aligned.
        try:
            import sys
            from pathlib import Path
            scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from refresh_dynamic_watchlist import get_active_watchlist  # type: ignore
            watchlist = get_active_watchlist()
            log.info("[premarket/collect_watchlist_data] DB watchlist loaded (%d tickers)", len(watchlist))
        except Exception as e:
            log.warning("[premarket/collect_watchlist_data] DB watchlist read failed: %s — using hardcoded fallback", e)
            watchlist = [
                "SPY", "QQQ", "IWM",
                "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
            ]

    quote_data = _fetch_quotes_batch(watchlist)

    # Also grab one recent EDGAR filing per ticker (best-effort, sequential)
    for ticker in watchlist[:6]:  # limit to top 6 to avoid rate-limit
        headline = _fetch_recent_filing_headline(ticker)
        if headline and ticker in quote_data:
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
        f"Rank the watchlist tickers from most to least actionable as of today's open. ",
        f"Score 0.0–1.0. Include only tickers where there is a concrete near-term setup ",
        f"(momentum, mean-reversion, event-driven, or regime-aligned).",
        f"",
        f"PREFERENCE ORDER (most→least preferred when ranking):",
        f"  1. Single-name catalyst plays: recent filings (8-K, 4), earnings within ",
        f"     2 weeks, news-driven gaps, sector rotation candidates.",
        f"  2. High-volume single names with above-average premarket move ",
        f"     (|chg_pct| ≥ 1.5%).",
        f"  3. Mag-7 / large-caps with a SPECIFIC thesis tied to today (not just ",
        f"     'good company, will go up').",
        f"  4. Broad ETFs (SPY, QQQ, IWM) — ONLY when there is an explicit ",
        f"     macro thesis (FOMC, CPI print, NFP, sector breadth turn). Default ",
        f"     ETF score should be ≤ 0.5; anything > 0.7 needs explicit ",
        f"     justification in the `reason` field naming the macro trigger.",
        f"",
        f"Reject silently (don't put in candidates list): tickers with no clear edge, ",
        f"tickers that contradict the current regime, or ETFs without a macro thesis.",
        f"",
        f"Return at most 5 candidates. Quality over quantity — if only 1 ticker has ",
        f"a real catalyst, return just that one. Respond per your ScoutOutput schema.",
    ]
    return "\n".join(lines)


def rank_candidates(state: TradingGraphState) -> dict:
    """Call the scout LLM (Haiku) to score and filter the watchlist.

    Output is a ranked list of candidates stored in state["candidates"].
    If the LLM fails, falls back to top-3 by absolute change_pct.

    Fallback is intentionally LOUD: 5/22-6/1 incident showed Haiku
    intermittently returning empty / list-wrapped output, triggering
    silent fallback to score=0.5 (below DISPATCH_MIN_SCORE 0.55) — so
    8 of 9 trading days dispatched 0 candidates with 0 alerts. Now
    every fallback emits sev=2 + ntfy "scout LLM degraded".
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
    failure_mode: str | None = None
    llm_exception: str | None = None
    raw_sample: str | None = None  # captured raw scout text for post-mortem

    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import ScoutOutput
        router = get_router()
        res = router.call("scout", prompt, schema=ScoutOutput, timeout_s=120)
        # Capture the raw model text so a degraded run records what the
        # model ACTUALLY said — not a reconstruction. (advisor 6/2: stop
        # inferring from a 4ms log gap; read the bytes.) Tickers + scores
        # only, nothing sensitive.
        raw_sample = (getattr(res, "raw_text", "") or "")[:500]
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
        elif parsed is None:
            failure_mode = "schema_parse_failed"
            log.warning("[rank_candidates] scout returned malformed output, parsed=None")
        else:
            failure_mode = "empty_candidates"
            log.warning("[rank_candidates] scout returned empty candidates list")
    except Exception as e:
        failure_mode = "llm_exception"
        llm_exception = str(e)[:200]
        log.warning("[rank_candidates] scout LLM raised: %s", e)

    # Fallback: rank by |change_pct| if LLM gave nothing.
    #
    # Score is 0.0, NOT 0.5. The fallback means "scout produced no real
    # signal" — that is a DO-NOT-DISPATCH state, not moderate conviction.
    # The old 0.5 score made a failed scout look almost dispatchable and
    # forced DISPATCH_MIN_SCORE artificially high (0.55) just to exclude
    # fallbacks. With scout on Sonnet (reliable) and fallback scored 0.0,
    # the threshold can sit at the natural "real signal" level (0.50) and
    # fallbacks are cleanly excluded. The candidates are still surfaced in
    # the digest (so the operator sees what moved) but won't dispatch.
    if not candidates:
        ranked_tickers = sorted(
            watchlist,
            key=lambda t: abs(float((market_data.get(t) or {}).get("change_pct") or 0)),
            reverse=True,
        )
        candidates = [
            {"ticker": t, "score": 0.0, "reason": "fallback_by_abs_change_pct_NO_LLM_SIGNAL"}
            for t in ranked_tickers[:3]
        ]
        # Loud alert — silent degradation is exactly what caused the
        # 5/22 → 6/1 0-trades-for-11-days incident. We want a phone ping
        # on every degraded run so the operator knows the system is
        # effectively offline for new entries.
        emit(
            run_id=run_id, trigger=trigger, agent="rank_candidates",
            event_type="scout_llm_degraded",
            severity=2,
            payload={
                "failure_mode": failure_mode or "unknown",
                "llm_exception": llm_exception,
                "n_watchlist": len(watchlist),
                "fallback_top": [c["ticker"] for c in candidates[:3]],
                # The actual model output (truncated) — so the next failure
                # is read from bytes, not reconstructed from a log gap.
                "raw_sample": raw_sample,
            },
        )
        try:
            from trading_agent.notify import send as ntfy_send
            ntfy_send(
                topic="ops",
                title="Scout LLM degraded — premarket dispatch will likely skip",
                body=(
                    f"failure_mode: {failure_mode or 'unknown'}\n"
                    f"falling back to abs(change_pct) ranking with score=0.5\n"
                    f"DISPATCH_MIN_SCORE is 0.55 → fallback candidates "
                    f"will be SKIPPED.\n"
                    f"Top fallback: {[c['ticker'] for c in candidates[:3]]}\n"
                    f"Investigate: tail /var/log/trading-agent-brain.err | "
                    f"grep scout"
                ),
                priority=4,
                tags=["warning", "robot"],
            )
        except Exception as e:
            log.warning("[rank_candidates] degraded-alert ntfy failed: %s", e)

    # Persist per-ticker scout scores into watchlist_members so the
    # daily refresh's eviction logic can see scout-flat tickers and
    # rotate them out after the threshold window. Best-effort; the
    # watchlist table may not exist on stale deployments.
    try:
        from trading_agent.store.postgres import cursor as _pgcur
        with _pgcur() as cur:
            for c in candidates:
                cur.execute(
                    """
                    UPDATE watchlist_members
                       SET last_scout_score = %s,
                           last_scout_ts    = NOW()
                     WHERE ticker = %s
                    """,
                    (float(c.get("score") or 0), c.get("ticker")),
                )
    except Exception as e:
        log.debug("[rank_candidates] scout-score writeback skipped: %s", e)

    emit(
        run_id=run_id, trigger=trigger, agent="rank_candidates",
        event_type="candidates_ranked",
        payload={
            "n_candidates": len(candidates),
            "top_ticker": candidates[0]["ticker"] if candidates else None,
            # Include top-3 with scores for downstream debugging + dispatch decision
            "top": [
                {"ticker": c["ticker"], "score": round(c["score"], 3), "reason": c["reason"][:120]}
                for c in candidates[:3]
            ],
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

    # ---- Autonomous handoff: dispatch candidate_entry for top pick ----
    _dispatch_candidate_entry_if_eligible(state, candidates, regime)
    return {}


# ---------------------------------------------------------------------------
# Autonomous handoff to candidate_entry
# ---------------------------------------------------------------------------

# Score threshold below which we do NOT dispatch. Above this, scout has at
# least moderate conviction and the full pipeline (Bull/Bear debate + Risk
# Council) is worth the LLM spend.
#
# Tuning history:
#   0.7  → too tight; daily-best frequently sat at 0.60-0.65, system never
#          dispatched
#   0.6  → still too tight in aggressive mode. 5/22 audit: NVDA scored 0.56
#          ("203M premarket vol vs 40M baseline; AI sector momentum") —
#          legitimate setup but missed cutoff by 0.04
#   0.55 → still excluded today's legit NVDA (0.52). The 0.55 floor only
#          existed to clear the 0.50 fallback score — but that fallback
#          score was itself wrong (a failed scout isn't "moderate
#          conviction").
#   0.50 → fallback score is now 0.0 (see rank_candidates), so the floor
#          no longer needs headroom above it. 0.50 = "any genuine scout
#          signal dispatches; the downstream trader-synthesizer is the
#          real quality gate (it DECLINED on 5/28 when no edge existed)."
DISPATCH_MIN_SCORE = 0.50

# Don't dispatch the same ticker more than once in this many days. Prevents
# the scout's daily "top pick" from collapsing onto a handful of names if the
# same underlying keeps rotating to the top of the watchlist.
SAME_TICKER_COOLDOWN_DAYS = 7


def _dispatch_candidate_entry_if_eligible(
    state: TradingGraphState,
    candidates: list[dict],
    regime: dict,
) -> None:
    """Spin off the candidate_entry subgraph for the highest-scoring candidate.

    Conditions for dispatch (all must hold):
      * at least one candidate, top score ≥ DISPATCH_MIN_SCORE
      * halt flag absent
      * soak phase allows new entries
      * regime gate allows new entries

    Mechanism: ``subprocess.Popen`` with ``start_new_session=True`` so the
    child outlives the premarket graph's runtime. The child runs the full
    candidate_entry pipeline (research → debate → propose → size → risk
    council → execute or veto) independently.

    Never raises — failures only log + emit. Worst case the user gets an
    audit row with rationale and the candidate is skipped that day.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]

    if not candidates:
        return
    top = candidates[0]
    top_score = float(top.get("score") or 0.0)
    ticker = (top.get("ticker") or "").strip().upper()
    if not ticker:
        return

    # Guard 1: score threshold
    if top_score < DISPATCH_MIN_SCORE:
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_skipped",
            payload={"ticker": ticker, "score": top_score, "reason": "below_score_threshold",
                     "threshold": DISPATCH_MIN_SCORE},
        )
        return

    # Guard 2: halt flag
    from pathlib import Path
    halt_flag = Path.home() / "trading-agent" / "data" / "halt.flag"
    if halt_flag.exists():
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_skipped",
            payload={"ticker": ticker, "reason": "halt_flag_set"},
        )
        return

    # Guard 3: soak phase
    try:
        from trading_agent.learning.soak import current_phase, is_new_entry_allowed
        phase = current_phase()
        if not is_new_entry_allowed(phase):
            emit(
                run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
                event_type="candidate_entry_skipped",
                payload={"ticker": ticker, "reason": "soak_read_only", "soak_phase": phase.value},
            )
            return
    except Exception as e:
        log.warning("[dispatch] soak check failed: %s", e)

    # Guard 4: regime gate
    gate = regime.get("gate") or {}
    if not gate.get("allow_new_entries", True):
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_skipped",
            payload={"ticker": ticker, "reason": "regime_blocks_new_entries",
                     "regime_label": regime.get("label")},
        )
        return

    # Guard 5: position-aware skip — already exposed to this ticker or sector
    exposure = _existing_exposure(ticker)
    if exposure:
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_skipped",
            payload={"ticker": ticker, "reason": exposure["reason"],
                     "detail": exposure["detail"]},
        )
        return

    # Guard 6: cooldown — recent FILLED entry on same ticker (VETO/DEFER do
    # not lock; only an executed fill within the window blocks re-dispatch).
    cd = _in_dispatch_cooldown(ticker, days=SAME_TICKER_COOLDOWN_DAYS)
    if cd:
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_skipped",
            payload={"ticker": ticker, "reason": "ticker_cooldown",
                     "last_fill_age_days": cd["age_days"],
                     "cooldown_days": SAME_TICKER_COOLDOWN_DAYS},
        )
        return

    # All guards passed — dispatch via systemd unit (NOT subprocess.Popen).
    #
    # Why systemd unit instead of subprocess.Popen(start_new_session=True):
    #
    # subprocess.Popen with start_new_session=True changes the SESSION ID but
    # does NOT leave the parent's systemd cgroup. When the parent unit
    # (trading-agent-brain@premarket_scan.service) ExecStart returns, systemd
    # defaults to KillMode=control-group and SIGTERMs every PID in the cgroup
    # — including the "detached" child. We saw this happen on 5/13 12:31:
    # SPY was dispatched, child PID 96421 silently died before LangGraph
    # could emit `run_start`, and nothing wrote to the journal or log files
    # because Python was killed during import.
    #
    # The fix is to start a SEPARATE systemd unit per ticker. systemd already
    # provides `trading-agent-candidate-entry@.service` as a template — each
    # instance lives in its own cgroup, has its own 15-min TimeoutStartSec,
    # its own journalctl trace (`journalctl -u trading-agent-candidate-entry@SPY`),
    # and its own EnvironmentFile loading. The parent unit exiting no longer
    # touches it.
    #
    # Requires sudoers entry on EC2:
    #   ubuntu ALL=(root) NOPASSWD: /bin/systemctl start trading-agent-candidate-entry@*.service, \
    #                               /bin/systemctl reset-failed trading-agent-candidate-entry@*.service
    # See deploy/ec2/INSTALL.md for the install snippet.
    import subprocess
    unit_name = f"trading-agent-candidate-entry@{ticker}.service"

    try:
        # Clear any prior "failed" state of this instance — systemctl start
        # silently no-ops if the unit is in failed state (e.g. previous run
        # hit TimeoutStartSec). reset-failed is a noop when state is clean.
        subprocess.run(
            ["sudo", "-n", "/bin/systemctl", "reset-failed", unit_name],
            check=False, capture_output=True, timeout=10,
        )
        # Fire-and-forget start. --no-block means systemctl returns
        # immediately; the unit runs asynchronously in its own cgroup.
        result = subprocess.run(
            ["sudo", "-n", "/bin/systemctl", "start", "--no-block", unit_name],
            check=False, capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"systemctl start exit={result.returncode}: "
                f"{result.stderr.decode('utf-8', 'replace')[:300]}"
            )
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_dispatched",
            payload={
                "ticker": ticker,
                "score": top_score,
                "reason": top.get("reason", "")[:160],
                "unit": unit_name,
            },
        )
        log.info(
            "[dispatch] candidate_entry started for %s via systemd (score=%.2f)",
            ticker, top_score,
        )
    except Exception as e:
        log.error("[dispatch] systemctl start failed for %s: %s", ticker, e)
        emit(
            run_id=run_id, trigger=trigger, agent="ntfy_scan_digest",
            event_type="candidate_entry_dispatch_failed",
            severity=2,
            payload={"ticker": ticker, "error": str(e)[:200]},
        )


# ---------------------------------------------------------------------------
# Dispatch-guard helpers
# ---------------------------------------------------------------------------

def _existing_exposure(ticker: str) -> dict | None:
    """Return a reason dict if we already have exposure to this ticker.

    Two checks:
      1. Same underlying — moomoo broker has a non-zero qty position OR
         Postgres journal_trades has an OPEN entry for the same underlying.
      2. Same GICS sector — any open broker position whose bare ticker
         maps to the same sector via sectors.lookup().

    Both checks ignore zombie positions (qty=0). Returns None when no
    relevant exposure exists, i.e. dispatch is allowed to proceed.
    """
    from trading_agent import sectors as sectors_lookup

    norm_ticker = (ticker or "").strip().upper()
    if not norm_ticker:
        return None
    target_sector = sectors_lookup.lookup(norm_ticker)

    # Pull open broker positions
    broker_positions: list[dict] = []
    try:
        from trading_agent.mcp_servers.moomoo.server import get_positions
        rows = (get_positions().get("rows") or [])
        broker_positions = [r for r in rows if float(r.get("qty") or 0) != 0]
    except Exception as e:
        log.warning("[exposure] broker positions fetch failed: %s", e)

    for p in broker_positions:
        code = p.get("code") or p.get("symbol") or ""
        # Strip option strike to get the underlying ticker.
        import re
        bare = code.split(".", 1)[-1]
        m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", bare)
        underlying = (m.group(1) if m else bare).upper()
        if underlying == norm_ticker:
            return {"reason": "already_holding_same_underlying",
                    "detail": f"{code} qty={p.get('qty')}"}
        if target_sector and target_sector == sectors_lookup.lookup(underlying):
            return {"reason": "already_exposed_to_sector",
                    "detail": f"{underlying} in {target_sector}"}

    # Postgres journal — catch OPEN trades broker forgot to clear
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT symbol FROM journal_trades
                WHERE outcome = 'OPEN' AND qty > 0
                """,
            )
            for (sym,) in cur.fetchall():
                import re
                bare = (sym or "").split(".", 1)[-1]
                m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", bare)
                underlying = (m.group(1) if m else bare).upper()
                if underlying == norm_ticker:
                    return {"reason": "journal_open_same_underlying",
                            "detail": str(sym)}
    except Exception as e:
        log.warning("[exposure] journal check failed: %s", e)

    return None


def _in_dispatch_cooldown(ticker: str, days: int = 7) -> dict | None:
    """Return cooldown info if `ticker` has had a FILLED entry within `days`,
    else None.

    Cooldown semantics (revised):
        * Locks the ticker only after a successful fill — event ``trade_recorded``
          fired by ``persist_trade_event`` after the ``journal_trades`` insert.
        * VETO and DEFER outcomes do NOT lock the ticker. The LLM can re-evaluate
          the same name on the next trading day — a Risk Council veto is "this
          setup isn't safe right now", not "blacklist this name for a week".
        * Legacy fills (before ``ticker`` was added to the payload) are caught
          by a regex against ``payload->>'symbol'`` (moomoo format
          ``US.<TICKER><YYMMDD><C|P><strike>``). The digit-boundary anchor
          ``([0-9]|$)`` ensures ``NVDA`` doesn't false-match ``NVDAB``.

    Returns None on any DB error — cooldown is best-effort, not authoritative.
    A position-aware skip (``_existing_exposure``) is the real defence against
    double-opens; cooldown is just to keep the scout from churning on one name.
    """
    try:
        from trading_agent.store.postgres import cursor
        norm = (ticker or "").strip().upper()
        if not norm:
            return None
        # Regex parameter — anchor on US.<TICKER> + digit boundary
        symbol_regex = rf"^US\.{norm}([0-9]|$)"
        with cursor() as cur:
            cur.execute(
                """
                SELECT MAX(ts) FROM agent_events
                WHERE event_type = 'trade_recorded'
                  AND (payload->>'ticker' = %s
                       OR (payload->>'symbol') ~ %s)
                  AND ts > NOW() - (%s::text || ' days')::interval
                """,
                (norm, symbol_regex, str(days)),
            )
            row = cur.fetchone()
            if row and row[0]:
                from datetime import datetime, timezone
                last_ts = row[0]
                age_days = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
                return {"last_ts": last_ts.isoformat(), "age_days": round(age_days, 2)}
    except Exception as e:
        log.warning("[cooldown] db check failed: %s", e)
    return None


__all__ = [
    "collect_watchlist_data",
    "rank_candidates",
    "ntfy_scan_digest",
]
