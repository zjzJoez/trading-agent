"""Phase 2.6 — Shadow tracker: at-decision-time counterfactual recorder.

A "shadow" is a candidate parameter version that runs *in parallel* with the
control on the same proposal but never alters the actual order.  We log the
counterfactual outcome to ``learning_events`` and the assignment to
``learning_assignments`` so the weekly_learning_graph can score the candidate
without re-running expensive LLMs.

Deterministic-replay scope (cheap counterfactuals — Phase 2.6):
    * sizing_aggression — re-evaluate R1 / R5 soft caps and downsize qty
    * regime_size_multipliers — re-multiply sized qty by candidate multiplier
    * regime_thresholds — re-run crisis_overlay against the persisted
      feature snapshot to see if a different VIX/SPY threshold flips the label

Replay-only scope (Phase 2.6 cannot shadow at decision time — Phase 2.7+):
    * entry_filters — would change which candidates the LLM trader sees
    * candidate_count — would change scout output upstream
    * stop_distances — only knowable from the realized price path post-close

The tracker is best-effort: any failure is logged + emitted to agent_events but
never re-raised.  Shadow tracking must never block a live decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from trading_agent.events import emit
from trading_agent.learning.params import (
    PARAM_BOUNDS,
    ParamFamily,
    ParamResolver,
    list_versions,
)
from trading_agent.regime.classifier import crisis_overlay
from trading_agent.regime.features import FeatureSnapshot
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counterfactual computation
# ---------------------------------------------------------------------------

@dataclass
class SizingCounterfactual:
    """Result of replaying R1/R5/regime caps under a candidate resolver."""

    qty_actual: int
    qty_shadow: int
    notional_actual: float
    notional_shadow: float
    max_loss_actual: float
    max_loss_shadow: float
    blocking_rule: str | None
    multiplier_actual: float
    multiplier_shadow: float

    @property
    def qty_delta(self) -> int:
        return self.qty_shadow - self.qty_actual

    def to_payload(self) -> dict[str, Any]:
        return {
            "qty_actual": self.qty_actual,
            "qty_shadow": self.qty_shadow,
            "qty_delta": self.qty_delta,
            "notional_actual": round(self.notional_actual, 2),
            "notional_shadow": round(self.notional_shadow, 2),
            "max_loss_actual": round(self.max_loss_actual, 2),
            "max_loss_shadow": round(self.max_loss_shadow, 2),
            "blocking_rule": self.blocking_rule,
            "multiplier_actual": self.multiplier_actual,
            "multiplier_shadow": self.multiplier_shadow,
        }


@dataclass
class CrisisCounterfactual:
    """Result of replaying crisis_overlay under candidate thresholds."""

    is_crisis_actual: bool
    is_crisis_shadow: bool
    flags_actual: list[str]
    flags_shadow: list[str]

    @property
    def flips(self) -> bool:
        return self.is_crisis_actual != self.is_crisis_shadow

    def to_payload(self) -> dict[str, Any]:
        return {
            "is_crisis_actual": self.is_crisis_actual,
            "is_crisis_shadow": self.is_crisis_shadow,
            "flags_actual": list(self.flags_actual),
            "flags_shadow": list(self.flags_shadow),
            "flips": self.flips,
        }


@dataclass
class ShadowRunRecord:
    """What gets persisted for one (proposal × shadow_version) pair."""

    run_id: str
    proposal_id: str
    ticker: str
    strategy_label: str
    baseline_param_version_id: int | None
    shadow_param_version_id: int
    sizing: SizingCounterfactual | None = None
    crisis: CrisisCounterfactual | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _regime_size_mult_key(label: str | None) -> str | None:
    """Map a regime label to the matching size_mult_* param key."""
    if not label:
        return None
    mapping = {
        "BULL_TREND": "size_mult_bull_trend",
        "RANGE_LOW_VOL": "size_mult_range_low_vol",
        "VOLATILE_TRANSITION": "size_mult_volatile_transition",
        "BEAR_TREND": "size_mult_bear_trend",
        "CRISIS": None,  # hard zero, never shadowed
    }
    return mapping.get(label)


def replay_sizing_caps(
    *,
    equity: float,
    entry_price: float,
    contract_multiplier: int,
    qty_actual: int,
    max_loss_per_contract: float,
    regime_label: str | None,
    control: ParamResolver,
    shadow: ParamResolver,
) -> SizingCounterfactual:
    """Re-derive qty_shadow under the shadow resolver.

    Mirrors the qty-shrink loop in ``trade_nodes.deterministic_sizing`` but
    parameterised by the resolver.  Hard caps from ``sizing.py`` (R1=0.02
    notional risk, R5=0.01 notional ceiling) are still applied — soft knobs
    can only tighten further.
    """
    contracts_to_qty = max(1, int(contract_multiplier))
    notional_per_contract = entry_price * contracts_to_qty

    # Hard caps from sizing.py — never crossed
    HARD_R1_PCT = 0.020
    HARD_R5_PCT = 0.010

    def _resolve_qty(r: ParamResolver, mult: float) -> tuple[int, float, float, str | None]:
        soft_r1 = min(r.get("r1_soft_cap_pct"), HARD_R1_PCT)
        soft_r5 = min(r.get("r5_soft_notional_pct"), HARD_R5_PCT)
        r1_budget = soft_r1 * equity
        r5_cap = soft_r5 * equity
        # apply regime multiplier first
        target_qty = max(0, int(qty_actual * (mult / max(0.0001, _control_mult)) ))
        qty = target_qty
        block: str | None = None
        while qty > 0:
            ml = max_loss_per_contract * qty
            nm = notional_per_contract * qty
            if ml <= r1_budget and nm <= r5_cap:
                break
            if ml > r1_budget:
                block = "R1"
            elif nm > r5_cap:
                block = "R5"
            qty -= 1
        return qty, qty * notional_per_contract, qty * max_loss_per_contract, block

    mult_key = _regime_size_mult_key(regime_label)
    _control_mult = control.get(mult_key) if mult_key else 1.0
    shadow_mult = shadow.get(mult_key) if mult_key else 1.0

    # control branch: identity (we already know qty_actual was sized)
    notional_actual = qty_actual * notional_per_contract
    max_loss_actual = qty_actual * max_loss_per_contract

    qty_shadow, notional_shadow, ml_shadow, block_shadow = _resolve_qty(
        shadow, shadow_mult
    )

    return SizingCounterfactual(
        qty_actual=qty_actual,
        qty_shadow=qty_shadow,
        notional_actual=notional_actual,
        notional_shadow=notional_shadow,
        max_loss_actual=max_loss_actual,
        max_loss_shadow=ml_shadow,
        blocking_rule=block_shadow,
        multiplier_actual=float(_control_mult),
        multiplier_shadow=float(shadow_mult),
    )


def replay_crisis_overlay(
    snapshot: FeatureSnapshot,
    control: ParamResolver,
    shadow: ParamResolver,
) -> CrisisCounterfactual:
    """Re-run the deterministic crisis overlay under candidate thresholds."""
    def _params(r: ParamResolver) -> dict[str, float]:
        # Align with regime/classifier.py DEFAULT_CRISIS_PARAMS key names.
        # Note: spy_5d_drop is a fraction in classifier (e.g. -0.07) but our
        # param family stores it as percent (-7.0) — convert here.
        return {
            "vix_abs_high": float(r.get("crisis_vix_level")),
            "vix_5d_change_high": float(r.get("crisis_vix_delta_5d")),
            "spy_5d_drop": float(r.get("crisis_spy_5d_drop_pct")) / 100.0,
        }

    a = crisis_overlay(snapshot, params=_params(control))
    b = crisis_overlay(snapshot, params=_params(shadow))
    return CrisisCounterfactual(
        is_crisis_actual=bool(a.is_crisis),
        is_crisis_shadow=bool(b.is_crisis),
        flags_actual=list(a.flags),
        flags_shadow=list(b.flags),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def list_active_shadows(scope: str = "global") -> list[dict[str, Any]]:
    """Return all SHADOW-status param_versions rows for this scope.

    Defensive: returns ``[]`` if the table is unreachable so a live decision
    never blocks on shadow tracking infrastructure.
    """
    try:
        return list_versions(status="SHADOW", scope=scope, limit=25)
    except Exception as e:
        log.warning("list_active_shadows: DB read failed (%s)", e)
        return []


def _materialize_shadow_resolver(version_row: dict[str, Any]) -> ParamResolver:
    """Build a ParamResolver from a list_versions() row."""
    return ParamResolver(
        version_id=int(version_row["id"]),
        status=str(version_row["status"]),
        scope=str(version_row["scope"]),
        scope_key=version_row.get("scope_key"),
        values={k: v for k, v in (version_row.get("params") or {}).items()
                if k in PARAM_BOUNDS},
    )


def record_shadow_run(rec: ShadowRunRecord) -> None:
    """Persist one shadow record: 1 learning_assignments + 1 learning_events.

    Best-effort; logs and continues on any DB failure.
    """
    payload: dict[str, Any] = {
        "ticker": rec.ticker,
        "strategy_label": rec.strategy_label,
        "baseline_param_version_id": rec.baseline_param_version_id,
        "shadow_param_version_id": rec.shadow_param_version_id,
    }
    if rec.sizing is not None:
        payload["sizing"] = rec.sizing.to_payload()
    if rec.crisis is not None:
        payload["crisis"] = rec.crisis.to_payload()
    if rec.extra:
        payload["extra"] = rec.extra

    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO learning_assignments
                    (proposal_id, ticker, strategy_label,
                     baseline_param_version_id, assigned_param_version_id,
                     assignment_bucket, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    rec.proposal_id,
                    rec.ticker,
                    rec.strategy_label,
                    rec.baseline_param_version_id,
                    rec.shadow_param_version_id,
                    0,  # bucket=0 means "shadow only, never actuated"
                    "phase_2_6_shadow",
                ),
            )
    except Exception as e:
        log.warning("record_shadow_run: assignments insert failed (%s)", e)

    emit(
        run_id=rec.run_id,
        trigger="candidate_entry",
        agent="shadow_tracker",
        event_type="shadow_run",
        payload=payload,
    )

    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO learning_events
                    (event_type, param_version_id, payload)
                VALUES ('shadow_run', %s, %s::jsonb)
                """,
                (rec.shadow_param_version_id, _json_dumps(payload)),
            )
    except Exception as e:
        log.warning("record_shadow_run: learning_events insert failed (%s)", e)


def _json_dumps(d: dict[str, Any]) -> str:
    import json
    return json.dumps(d, default=str)


# ---------------------------------------------------------------------------
# Top-level entry point used from the graph
# ---------------------------------------------------------------------------

@dataclass
class ShadowInputs:
    """Decision-time inputs the trade graph hands to the shadow tracker."""

    run_id: str
    proposal_id: str
    ticker: str
    strategy_label: str
    equity: float
    entry_price: float
    contract_multiplier: int
    qty_actual: int
    max_loss_per_contract: float
    regime_label: str | None
    feature_snapshot: FeatureSnapshot | None
    control: ParamResolver


def run_shadows(inputs: ShadowInputs, scope: str = "global") -> list[ShadowRunRecord]:
    """Iterate over all SHADOW-status param_versions and record counterfactuals.

    Phase 2.6: always returns the list (possibly empty) so the caller can
    propagate counts to ``agent_events``.  Never raises.
    """
    records: list[ShadowRunRecord] = []
    try:
        shadows = list_active_shadows(scope=scope)
    except Exception as e:
        log.warning("run_shadows: list_active_shadows failed (%s)", e)
        return records

    for sv in shadows:
        shadow_resolver = _materialize_shadow_resolver(sv)
        rec = ShadowRunRecord(
            run_id=inputs.run_id,
            proposal_id=inputs.proposal_id,
            ticker=inputs.ticker,
            strategy_label=inputs.strategy_label,
            baseline_param_version_id=inputs.control.version_id,
            shadow_param_version_id=int(sv["id"]),
        )
        try:
            rec.sizing = replay_sizing_caps(
                equity=inputs.equity,
                entry_price=inputs.entry_price,
                contract_multiplier=inputs.contract_multiplier,
                qty_actual=inputs.qty_actual,
                max_loss_per_contract=inputs.max_loss_per_contract,
                regime_label=inputs.regime_label,
                control=inputs.control,
                shadow=shadow_resolver,
            )
        except Exception as e:
            log.warning("run_shadows: replay_sizing_caps failed (%s)", e)
            rec.extra["sizing_error"] = str(e)

        if inputs.feature_snapshot is not None:
            try:
                rec.crisis = replay_crisis_overlay(
                    inputs.feature_snapshot, inputs.control, shadow_resolver
                )
            except Exception as e:
                log.warning("run_shadows: replay_crisis_overlay failed (%s)", e)
                rec.extra["crisis_error"] = str(e)

        record_shadow_run(rec)
        records.append(rec)

    return records


__all__ = [
    "ShadowInputs",
    "ShadowRunRecord",
    "SizingCounterfactual",
    "CrisisCounterfactual",
    "list_active_shadows",
    "replay_sizing_caps",
    "replay_crisis_overlay",
    "record_shadow_run",
    "run_shadows",
]
