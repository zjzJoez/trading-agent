"""Persist + update rows in shadow_proposals.

Lifecycle of a row:
    1. build_trade_proposal succeeds → insert_shadow_proposal(...)
       creates a row with final_action='pending', returns shadow_id
    2. deterministic_sizing → update_shadow_deterministic(...)
    3. regime_execution_gate → update_shadow_gate(...)
    4. (optional) LLM council → update_shadow_council(...)
    5. final outcome node (persist_veto / persist_defer / execute_paper_order)
       → finalize_shadow_proposal(...) sets final_action

Each helper is idempotent on (shadow_id, field) — re-running the same node
won't corrupt the row. Failures are best-effort logged and swallowed: the
shadow log must NEVER block the production trade flow.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def _capture_universe_snapshot() -> str:
    """Snapshot the active watchlist as JSON for embedding in shadow_proposals.

    Critical for downstream analysis: the dynamic watchlist means the LLM's
    selection universe varies daily. Without this column, "LLM directional
    skill" analysis would conflate skill with universe quality. See
    migrations/011_shadow_proposals_universe.sql.

    Best-effort: if the watchlist module isn't importable (older deploys
    without migration 010 applied), returns an empty snapshot — still
    valid JSON so the INSERT doesn't fail.
    """
    try:
        # Lazy import — refresh script lives in scripts/, not the trading_agent
        # package, so we import via the absolute path on sys.path.
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from refresh_dynamic_watchlist import get_universe_snapshot  # type: ignore
        return json.dumps(get_universe_snapshot(), default=str)
    except Exception as e:
        log.debug("[shadow] universe snapshot unavailable: %s", e)
        from datetime import datetime, timezone
        return json.dumps({
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "universe_size": 0,
            "tickers": [],
            "error": str(e),
        })


def insert_shadow_proposal(
    *,
    run_id: str,
    trigger: str,
    proposal: dict[str, Any],
    regime: dict[str, Any] | None,
) -> int | None:
    """Insert a fresh shadow_proposals row. Returns its id, or None on failure.

    Best-effort: any DB error is logged and swallowed; returning None
    signals callers to skip later UPDATE calls (the row doesn't exist).
    """
    universe_json = _capture_universe_snapshot()
    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO shadow_proposals (
                    proposal_ts, run_id, trigger,
                    ticker, symbol, asset_type, direction,
                    qty_request, entry_price, stop, target,
                    strategy_label, expected_return_pct, max_loss_pct,
                    proposal_json,
                    regime_state_id, regime_label, regime_confidence,
                    final_action,
                    universe_snapshot
                ) VALUES (
                    NOW(), %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s::jsonb,
                    %s, %s, %s,
                    'pending',
                    %s::jsonb
                )
                RETURNING id
                """,
                (
                    run_id, trigger,
                    proposal.get("ticker"), proposal.get("symbol"),
                    proposal.get("asset_type"), proposal.get("direction"),
                    proposal.get("qty"), proposal.get("entry_price"),
                    proposal.get("stop"), proposal.get("target"),
                    proposal.get("strategy_label"),
                    proposal.get("expected_return_pct"), proposal.get("max_loss_pct"),
                    json.dumps(proposal, default=str),
                    (regime or {}).get("state_id"),
                    (regime or {}).get("label"),
                    (regime or {}).get("confidence"),
                    universe_json,
                ),
            )
            shadow_id = int(cur.fetchone()[0])
        # Touch the watchlist row so eviction logic sees recent activity.
        # Best-effort: any failure is silent (watchlist may not exist yet).
        try:
            with cursor() as cur2:
                cur2.execute(
                    """
                    UPDATE watchlist_members
                       SET last_proposed_at = NOW()
                     WHERE ticker = %s
                    """,
                    (proposal.get("ticker"),),
                )
        except Exception:
            pass
        return shadow_id
    except Exception as e:
        log.warning("[shadow] insert failed (non-fatal): %s", e)
        return None


def update_shadow_deterministic(
    shadow_id: int,
    *,
    decision: str,
    factor: float,
    reasons: list[str] | None = None,
) -> None:
    if shadow_id is None:
        return
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE shadow_proposals
                SET deterministic_decision = %s,
                    deterministic_factor = %s,
                    deterministic_reasons = %s::jsonb
                WHERE id = %s
                """,
                (decision, factor, json.dumps(reasons or []), int(shadow_id)),
            )
    except Exception as e:
        log.warning("[shadow] deterministic update failed (id=%s): %s", shadow_id, e)


def update_shadow_gate(
    shadow_id: int,
    *,
    decision: str,
    size_factor: float,
    block_reason: str | None = None,
) -> None:
    if shadow_id is None:
        return
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE shadow_proposals
                SET gate_decision = %s,
                    gate_size_factor = %s,
                    gate_block_reason = %s
                WHERE id = %s
                """,
                (decision, size_factor, block_reason, int(shadow_id)),
            )
    except Exception as e:
        log.warning("[shadow] gate update failed (id=%s): %s", shadow_id, e)


def update_shadow_council(
    shadow_id: int,
    *,
    decision: str,
    size_factor: float,
    reasons: list[str] | None = None,
) -> None:
    if shadow_id is None:
        return
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE shadow_proposals
                SET council_invoked = TRUE,
                    council_decision = %s,
                    council_size_factor = %s,
                    council_reasons = %s::jsonb
                WHERE id = %s
                """,
                (decision, size_factor, json.dumps(reasons or []), int(shadow_id)),
            )
    except Exception as e:
        log.warning("[shadow] council update failed (id=%s): %s", shadow_id, e)


def finalize_shadow_proposal(
    shadow_id: int,
    *,
    final_action: str,
    journal_thesis_id: int | None = None,
) -> None:
    """Mark the final outcome. `final_action` should be one of:
    EXECUTED | VETOED_BY_GATE | VETOED_BY_RISK | VETOED_BY_COUNCIL |
    DEFERRED | DECLINED_BY_TRADER | REJECTED_SPEC_BAND.
    """
    if shadow_id is None:
        return
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE shadow_proposals
                SET final_action = %s,
                    journal_thesis_id = %s
                WHERE id = %s
                """,
                (final_action, journal_thesis_id, int(shadow_id)),
            )
    except Exception as e:
        log.warning("[shadow] finalize failed (id=%s): %s", shadow_id, e)
