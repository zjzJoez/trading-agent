"""Phase 2.6 — LangGraph nodes for the online-learning loop.

Two nodes are exposed:

  * ``load_active_params``  — resolves the current ACTIVE param_versions row
    once per graph run and snapshots it onto ``state["learning"]``.  Every
    downstream consumer (sizing, regime gate, risk, trader) sees the same
    values, so the decision has a single ``params_version_id`` for audit.

  * ``shadow_track``  — best-effort: replays every ``SHADOW``-status param
    version against the just-sized proposal and writes the counterfactual
    metrics to ``learning_events`` + ``learning_assignments``.  Never alters
    the proposal; never raises.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState
from trading_agent.learning.params import load_active_params
from trading_agent.learning.shadow import ShadowInputs, run_shadows

log = logging.getLogger(__name__)


def load_active_params_node(state: TradingGraphState) -> dict:
    """Resolve the currently ACTIVE global parameter set for this run."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/load_active_params] run_id=%s", run_id)

    resolver = load_active_params(scope="global", scope_key=None)
    learning_payload = state.get("learning") or {}
    learning_payload = {
        **learning_payload,
        "params": resolver.to_audit_dict(),
        "params_version_id": resolver.version_id,
    }

    emit(
        run_id=run_id, trigger=trigger, agent="load_active_params",
        event_type="params_resolved",
        payload={
            "version_id": resolver.version_id,
            "status": resolver.status,
            "n_overrides": len(resolver.values),
        },
    )
    return {"learning": learning_payload}


def shadow_track_node(state: TradingGraphState) -> dict:
    """Replay all SHADOW param versions against the sized proposal."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/shadow_track] run_id=%s", run_id)

    proposal = state.get("proposal")
    sizing = state.get("sizing") or {}
    if not proposal or sizing.get("infeasible"):
        return {}

    learning_payload = state.get("learning") or {}
    control_dict = learning_payload.get("params") or {}

    from trading_agent.learning.params import ParamResolver
    control = ParamResolver(
        version_id=control_dict.get("version_id"),
        status=control_dict.get("status", "DEFAULTS"),
        scope=control_dict.get("scope", "global"),
        scope_key=control_dict.get("scope_key"),
        values=dict(control_dict.get("values") or {}),
    )

    qty_actual = int(float(sizing.get("approved_qty") or proposal.get("qty") or 0))
    if qty_actual <= 0:
        return {}

    entry_price = float(proposal.get("entry_price") or 0.0)
    asset_type = proposal.get("asset_type", "OPT")
    contract_mult = 100 if asset_type == "OPT" else 1
    stop = proposal.get("stop")
    if stop is not None:
        max_loss_per_contract = abs(entry_price - float(stop)) * contract_mult
    else:
        max_loss_per_contract = entry_price * contract_mult * 0.05

    regime = state.get("regime") or {}
    account = state.get("account") or {}

    inputs = ShadowInputs(
        run_id=run_id,
        proposal_id=str(proposal.get("proposal_id") or ""),
        ticker=str(proposal.get("ticker") or ""),
        strategy_label=str(proposal.get("strategy_label") or ""),
        equity=float(account.get("equity") or 100_000.0),
        entry_price=entry_price,
        contract_multiplier=contract_mult,
        qty_actual=qty_actual,
        max_loss_per_contract=max_loss_per_contract,
        regime_label=regime.get("label"),
        feature_snapshot=None,  # Phase 2.6.5: refetch via feature_snapshot_id
        control=control,
    )

    try:
        records = run_shadows(inputs, scope="global")
    except Exception as e:
        log.warning("[learning/shadow_track] run_shadows failed: %s", e)
        records = []

    emit(
        run_id=run_id, trigger=trigger, agent="shadow_track",
        event_type="shadow_done",
        payload={
            "n_shadows": len(records),
            "shadow_version_ids": [r.shadow_param_version_id for r in records],
        },
    )

    learning_payload = {
        **learning_payload,
        "n_shadows": len(records),
    }
    return {"learning": learning_payload}


__all__ = ["load_active_params_node", "shadow_track_node"]
