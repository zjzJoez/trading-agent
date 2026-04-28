"""Phase 2.7.5 — weekly_learning_graph nodes.

Once a week (Saturday post-close — see ``trading-agent-learning.timer``)
the LLM Critic reviews:

  * recent closed-trade outcomes (``trade_outcome_features``)
  * the per-cell champion archive (``param_archive_cells``)
  * the current ACTIVE param values
  * the param bounds + recent param changes (anti-thrashing)

…and proposes 0–3 bounded parameter mutations.  Each accepted proposal is:

  1. validated against ``PARAM_BOUNDS`` + ``FROZEN_HARD_CAPS``
  2. replayed via ``replay.replay_param_version`` against control history
  3. inserted as a CANARY param_version (single family at a time, 10% traffic)
     via ``canary.insert_canary``

The Critic is intentionally weak — it can only propose; it never promotes
to ACTIVE directly.  Promotion still requires Phase 2.7's Wilson LB +
profit factor + drawdown gates after N_canary_min trades.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState
from trading_agent.learning.canary import (
    DEFAULT_INITIAL_TRAFFIC_PCT,
    MultiFamilyCanary,
    family_of_params,
    insert_canary,
    list_active_canaries,
)
from trading_agent.learning.params import (
    FROZEN_HARD_CAPS,
    PARAM_BOUNDS,
    load_active_params,
)
from trading_agent.learning.replay import replay_param_version
from trading_agent.learning.archive import archive_summary
from trading_agent.notify.ntfy import send as ntfy_send
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 7
MAX_PROPOSALS_PER_RUN = 3


# ---------------------------------------------------------------------------
# Outcome aggregator
# ---------------------------------------------------------------------------

def _last_n_days_outcomes(days: int = LOOKBACK_DAYS) -> dict[str, Any]:
    """Pull a compact summary of recent closed trades for the Critic prompt."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    out: dict[str, Any] = {"window_days": days, "rows": []}
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT jt.id, jt.symbol, jt.outcome, jt.opened_at, jt.closed_at,
                       tof.realized_r, tof.holding_days, tof.mae, tof.mfe,
                       tof.param_version_id, rs.label AS regime,
                       jt.broker_fill_json->>'strategy_label' AS strategy_label
                FROM journal_trades jt
                LEFT JOIN trade_outcome_features tof ON tof.trade_id = jt.id
                LEFT JOIN regime_states rs ON rs.id = jt.entry_regime_state_id
                WHERE jt.closed_at >= %s
                ORDER BY jt.closed_at DESC
                LIMIT 100
                """,
                (since,),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("_last_n_days_outcomes failed: %s", e)
        return out
    for r in rows:
        (tid, sym, outcome, opened_at, closed_at, realized_r, hold, mae, mfe,
         pvid, regime, strategy) = r
        out["rows"].append({
            "trade_id": int(tid),
            "symbol": str(sym),
            "outcome": str(outcome),
            "regime": regime,
            "strategy_label": strategy,
            "realized_R": float(realized_r) if realized_r is not None else None,
            "holding_days": float(hold) if hold is not None else None,
            "mae": float(mae) if mae is not None else None,
            "mfe": float(mfe) if mfe is not None else None,
            "params_version_id": int(pvid) if pvid is not None else None,
            "opened_at": opened_at.isoformat() if opened_at else None,
            "closed_at": closed_at.isoformat() if closed_at else None,
        })
    return out


def _recent_param_changes(weeks: int = 4) -> list[dict[str, Any]]:
    """Anti-thrashing: surface params that were already touched recently."""
    since = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, status, created_at, params, rationale
                FROM param_versions
                WHERE created_at >= %s AND status IN ('CANARY', 'ACTIVE', 'REJECTED')
                ORDER BY created_at DESC
                """,
                (since,),
            )
            rows = cur.fetchall()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for rid, status, created_at, params, rationale in rows:
        keys = sorted(list((params or {}).keys())) if isinstance(params, dict) else []
        out.append({
            "id": int(rid), "status": str(status),
            "created_at": created_at.isoformat() if created_at else None,
            "param_keys_touched": keys,
            "rationale": (rationale or "")[:200],
        })
    return out


# ---------------------------------------------------------------------------
# Critic call + proposal validation
# ---------------------------------------------------------------------------

def _format_critic_prompt(
    weekly: dict[str, Any],
    archive: list[dict[str, Any]],
    current_params: dict[str, Any],
    recent_changes: list[dict[str, Any]],
) -> str:
    bounds = {
        k: {"min": b.min, "max": b.max, "default": b.default,
            "family": b.family.value, "integer": b.integer}
        for k, b in PARAM_BOUNDS.items()
    }
    import json as _json
    return _json.dumps({
        "weekly_stats": {
            "n_closed_trades": len(weekly["rows"]),
            "window_days": weekly["window_days"],
        },
        "closed_trades": weekly["rows"][:50],
        "current_params": current_params,
        "param_bounds": bounds,
        "frozen_hard_caps": sorted(FROZEN_HARD_CAPS),
        "diversity_archive_summary": archive[:30],
        "recent_param_changes": recent_changes,
        "constraints": {
            "max_proposals": MAX_PROPOSALS_PER_RUN,
            "single_family_per_proposal": True,
            "must_pass_replay_validation": True,
        },
    }, default=str)


def _proposal_key_in_bounds(name: str, value: float) -> bool:
    bound = PARAM_BOUNDS.get(name)
    if bound is None or name in FROZEN_HARD_CAPS:
        return False
    return bound.min <= value <= bound.max


def _accept_or_reject_proposal(
    proposal: dict[str, Any],
) -> tuple[bool, str]:
    name = proposal.get("param_name")
    val = proposal.get("proposed_value")
    if not isinstance(name, str) or val is None:
        return False, "malformed proposal"
    if name in FROZEN_HARD_CAPS:
        return False, f"key {name} is frozen"
    if name not in PARAM_BOUNDS:
        return False, f"key {name} not in PARAM_BOUNDS"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False, "value not numeric"
    if not _proposal_key_in_bounds(name, f):
        b = PARAM_BOUNDS[name]
        return False, f"value {f} outside [{b.min}, {b.max}]"
    return True, "ok"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def collect_outcomes_node(state: TradingGraphState) -> dict:
    """Pull the last 7 days of closed trades + archive top into state."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/collect_outcomes] run_id=%s", run_id)

    weekly = _last_n_days_outcomes()
    archive = archive_summary(limit=30)
    recent = _recent_param_changes()
    current = load_active_params()
    current_params = {**{k: b.default for k, b in PARAM_BOUNDS.items()},
                      **current.values}

    learning = state.get("learning") or {}
    learning = {
        **learning,
        "weekly": weekly,
        "archive": archive,
        "recent_param_changes": recent,
        "current_params": current_params,
        "active_param_version_id": current.version_id,
    }
    emit(
        run_id=run_id, trigger=trigger, agent="collect_outcomes_weekly",
        event_type="weekly_corpus_built",
        payload={
            "n_closed_trades": len(weekly["rows"]),
            "n_archive_cells": len(archive),
            "n_recent_changes": len(recent),
        },
    )
    return {"learning": learning}


def critic_propose_node(state: TradingGraphState) -> dict:
    """Call the Codex Learning Critic for parameter proposals."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/critic_propose] run_id=%s", run_id)

    learning = state.get("learning") or {}
    weekly = learning.get("weekly") or {"window_days": LOOKBACK_DAYS, "rows": []}
    archive = learning.get("archive") or []
    current_params = learning.get("current_params") or {}
    recent_changes = learning.get("recent_param_changes") or []

    if len(weekly.get("rows", [])) == 0:
        emit(
            run_id=run_id, trigger=trigger, agent="critic_propose",
            event_type="skipped",
            payload={"reason": "no closed trades in window"},
        )
        return {"learning": {**learning, "critic_proposals": []}}

    try:
        from trading_agent.llm import get_router
    except Exception as e:
        log.warning("[critic_propose] router unavailable: %s", e)
        return {"learning": {**learning, "critic_proposals": []}}

    from trading_agent.llm.schemas import LearningCriticOutput
    router = get_router()
    prompt = _format_critic_prompt(weekly, archive, current_params, recent_changes)

    try:
        llm_result = router.call(
            role="learning_critic",
            user_prompt=prompt,
            schema=LearningCriticOutput,
            timeout_s=240,
        )
    except Exception as e:
        log.warning("[critic_propose] router.call failed: %s", e)
        return {"learning": {**learning, "critic_proposals": []}}

    parsed = llm_result.parsed
    if hasattr(parsed, "model_dump"):
        result = parsed.model_dump()
    elif isinstance(parsed, dict):
        result = parsed
    else:
        log.warning("[critic_propose] unexpected parsed type: %s", type(parsed))
        return {"learning": {**learning, "critic_proposals": []}}

    proposals = result.get("proposals") or []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for p in proposals[:MAX_PROPOSALS_PER_RUN]:
        ok, reason = _accept_or_reject_proposal(p)
        (accepted if ok else rejected).append({
            **(p if isinstance(p, dict) else {"raw": str(p)}),
            "validation": reason,
        })
    emit(
        run_id=run_id, trigger=trigger, agent="critic_propose",
        event_type="proposals_received",
        payload={
            "n_total": len(proposals),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
        },
    )
    return {"learning": {
        **learning,
        "critic_proposals": accepted,
        "critic_rejected": rejected,
        "critic_summary_md": result.get("weekly_summary_md"),
    }}


def replay_validate_node(state: TradingGraphState) -> dict:
    """Replay each accepted proposal against history.  Drop ones that don't
    score better than the current control via ``replay.replay_param_version``."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/replay_validate] run_id=%s", run_id)

    learning = state.get("learning") or {}
    proposals = learning.get("critic_proposals") or []
    if not proposals:
        return {}

    from trading_agent.learning.params import ParamResolver
    control = load_active_params()
    validated: list[dict[str, Any]] = []
    for p in proposals:
        name = p["param_name"]
        candidate_values = {**control.values, name: float(p["proposed_value"])}
        candidate = ParamResolver(
            version_id=None, status="DRAFT",
            scope="global", scope_key=None,
            values=candidate_values,
        )
        try:
            report = replay_param_version(candidate, baseline=control)
            delta = report.shadow.composite_score - report.actual.composite_score
            p_with_score = {
                **p,
                "replay_n_trades": report.n_trades_evaluated,
                "control_composite": round(report.actual.composite_score, 4),
                "candidate_composite": round(report.shadow.composite_score, 4),
                "composite_delta": round(delta, 4),
                "passed_replay": delta >= 0,
            }
        except Exception as e:
            p_with_score = {**p, "passed_replay": False, "replay_error": str(e)[:200]}
        validated.append(p_with_score)
    emit(
        run_id=run_id, trigger=trigger, agent="replay_validate",
        event_type="replay_done",
        payload={"n_validated": sum(1 for v in validated if v.get("passed_replay"))},
    )
    return {"learning": {**learning, "critic_proposals": validated}}


def promote_to_canary_node(state: TradingGraphState) -> dict:
    """For proposals that passed replay AND target a family with no live
    canary, insert as CANARY status with the configured initial traffic pct."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/promote_to_canary] run_id=%s", run_id)

    learning = state.get("learning") or {}
    proposals = learning.get("critic_proposals") or []
    parent_id = learning.get("active_param_version_id")
    inserted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    actives = list_active_canaries()
    occupied: set[str] = {a.family.value for a in actives}

    for p in proposals:
        if not p.get("passed_replay"):
            skipped.append({**p, "skip_reason": "did not pass replay"})
            continue
        params = {p["param_name"]: float(p["proposed_value"])}
        try:
            fam = family_of_params(params)
        except MultiFamilyCanary as e:
            skipped.append({**p, "skip_reason": str(e)})
            continue
        if fam.value in occupied:
            skipped.append({**p, "skip_reason": f"family {fam.value} already has live canary"})
            continue
        try:
            cv = insert_canary(
                parent_version_id=int(parent_id) if parent_id is not None else 1,
                params=params,
                rationale=p.get("rationale", "weekly Critic proposal"),
                created_by="phase_2_7_5_critic",
                initial_traffic_pct=DEFAULT_INITIAL_TRAFFIC_PCT,
            )
        except Exception as e:
            skipped.append({**p, "skip_reason": f"insert failed: {e}"})
            continue
        occupied.add(fam.value)
        inserted.append({**p, "canary_version_id": cv, "family": fam.value})

    emit(
        run_id=run_id, trigger=trigger, agent="promote_to_canary",
        event_type="canary_insertions",
        payload={
            "n_inserted": len(inserted),
            "n_skipped": len(skipped),
            "inserted": inserted,
        },
    )
    return {"learning": {
        **learning,
        "promoted_canaries": inserted,
        "skipped_proposals": skipped,
    }}


def ntfy_learning_digest_node(state: TradingGraphState) -> dict:
    """Push the weekly summary + accepted/skipped proposals to the
    `learning` ntfy topic."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/ntfy_digest] run_id=%s", run_id)

    learning = state.get("learning") or {}
    summary = (learning.get("critic_summary_md") or "")[:1000]
    inserted = learning.get("promoted_canaries") or []
    skipped = learning.get("skipped_proposals") or []
    body_lines = [
        f"weekly_learning_run {run_id}",
        f"closed_trades_in_window: {len(learning.get('weekly', {}).get('rows', []))}",
        f"canary_inserts: {len(inserted)}",
        f"skipped: {len(skipped)}",
        "",
        summary,
    ]
    body = "\n".join(body_lines)[:3500]
    try:
        ntfy_send(
            topic="learning",
            title="weekly_learning_digest",
            body=body,
            priority=3,
        )
    except Exception as e:
        log.warning("[ntfy_learning_digest] send failed: %s", e)
    emit(
        run_id=run_id, trigger=trigger, agent="ntfy_learning_digest",
        event_type="digest_sent",
        payload={"n_inserted": len(inserted), "n_skipped": len(skipped)},
    )
    return {}


__all__ = [
    "collect_outcomes_node",
    "critic_propose_node",
    "ntfy_learning_digest_node",
    "promote_to_canary_node",
    "replay_validate_node",
]
