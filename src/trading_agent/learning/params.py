"""Phase 2.6 — Parameter families, bounds, and ACTIVE-version resolver.

The online-learning loop mutates a small, bounded set of *soft* parameters.
This module is the single source of truth for:

  * which keys exist (allow-list)
  * which family each key belongs to
  * what range each key may legally take (bounds)
  * the code-level default value (used when no ACTIVE param_versions row exists)
  * how to resolve a `ParamResolver` for a single graph run

Hard risk caps and safety toggles are explicitly absent from the allow-list.
Any attempt to resolve a key not registered here raises ``UnknownParamKey``.

Schema reference (migrations/002_phase2.sql):

    param_versions(id, status, scope, scope_key, params jsonb, bounds jsonb,
                   parent_version_id, traffic_pct, ...)

Statuses: ``DRAFT`` | ``SHADOW`` | ``CANARY`` | ``ACTIVE`` | ``RETIRED`` |
``REJECTED``.  Phase 2.6 only reads/writes ``DRAFT``, ``SHADOW``, and
``ACTIVE`` — canary lifecycle states are owned by Phase 2.7.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family taxonomy
# ---------------------------------------------------------------------------

class ParamFamily(str, Enum):
    """The five mutable families enumerated in plan §8 + skill SKILL.md."""

    SIZING_AGGRESSION = "sizing_aggression"
    STOP_DISTANCES = "stop_distances"
    ENTRY_FILTERS = "entry_filters"
    REGIME_THRESHOLDS = "regime_thresholds"
    REGIME_SIZE_MULTIPLIERS = "regime_size_multipliers"


@dataclass(frozen=True)
class ParamBound:
    """Allow-list entry: one mutable parameter.

    ``canary_eligible`` — True only when LIVE decision code outside learning/
    consumes the key.  Background (ITEM 11): every Phase-2.7 canary before
    this flag existed was an A/A test — params_version_id changed on the
    journal row but no decision path read the value, so the promotion gates
    were comparing two identical treatments and any PROMOTE was pure noise.
    ``canary.route_canary`` skips ineligible keys so traffic is never spent
    on a canary whose only effect is a label; the CI guard in
    tests/learning/test_param_consumers.py greps the src tree to keep the
    flag honest (eligible key with no consumer outside learning/ = red).
    """

    family: ParamFamily
    default: float | int
    min: float | int
    max: float | int
    description: str
    integer: bool = False
    canary_eligible: bool = True

    def clamp(self, value: float | int) -> float | int:
        v = max(self.min, min(self.max, value))
        return int(round(v)) if self.integer else float(v)


# ---------------------------------------------------------------------------
# Allow-list — anything not here is rejected by the resolver
# ---------------------------------------------------------------------------
#
# Conservative bounds.  Values outside these ranges are treated as
# operator/configuration error and clamped (with a warning).  Ranges were
# chosen to never weaken Phase-1 hard safety:
#
#  * sizing_aggression keys never exceed Phase-1 R1/R5 caps
#  * regime_size_multipliers cap at 1.0 (no upgrade beyond BULL_TREND baseline)
#  * stop_distances narrow band so a learner can't disable stops
#  * regime_thresholds stay inside the v3 §6.2 envelope (0.75..0.95 vol pct)
#

PARAM_BOUNDS: dict[str, ParamBound] = {
    # -- sizing_aggression -------------------------------------------------
    # Note: R1/R5 hard caps (MAX_SINGLE_RISK_PCT / MAX_OPTION_NOTIONAL_PCT)
    # live in sizing.py and are NOT mutable.  The keys below are
    # *additional, more conservative* knobs the learner may tighten
    # further, never loosen past the Phase-1 floor.
    # Live consumer: trade_nodes.deterministic_sizing (_apply_soft_sizing_caps).
    "r1_soft_cap_pct": ParamBound(
        family=ParamFamily.SIZING_AGGRESSION,
        default=0.020, min=0.005, max=0.020,
        description="Soft per-trade risk cap; clamped at hard R1=0.02.",
    ),
    "r5_soft_notional_pct": ParamBound(
        family=ParamFamily.SIZING_AGGRESSION,
        default=0.010, min=0.003, max=0.010,
        description="Soft option-notional cap; clamped at hard R5=0.01.",
    ),
    "r3_ticker_exposure_pct": ParamBound(
        family=ParamFamily.SIZING_AGGRESSION,
        default=0.10, min=0.05, max=0.10,
        description="Per-ticker total exposure ceiling.",
    ),

    # -- stop_distances ----------------------------------------------------
    # Live consumer: trade_nodes.persist_trade_event's default-exit-plan
    # fallback (_stop_distance_fallback) — used only when the trader omits
    # an explicit stop, so the resolver can never override a chosen stop.
    "implicit_stop_frac": ParamBound(
        family=ParamFamily.STOP_DISTANCES,
        default=0.05, min=0.025, max=0.10,
        description="Fallback stop fraction when proposal lacks an explicit stop.",
    ),
    "option_premium_stop_pct": ParamBound(
        family=ParamFamily.STOP_DISTANCES,
        default=0.50, min=0.30, max=0.70,
        description="Premium-stop trigger; close at this fraction of entry premium.",
    ),
    # NOTE: default_stop_atr_mult was PRUNED (area E). No ATR series is
    # threaded into the trade path, so wiring a consumer would be fictional and
    # the system trades long-premium options almost exclusively (STK-direction
    # stops fall back to implicit_stop_frac). Re-add it only alongside a real
    # ATR feed on the proposal.

    # -- entry_filters -----------------------------------------------------
    # min_dte is the only surviving entry filter. min_setup_quality and
    # max_iv_percentile_long_premium were PRUNED (area E): the trader proposal
    # carries no setup-quality score and no IV-PERCENTILE series, so neither
    # had a real field to consume — a canary on them was fictional.
    # Live consumer for min_dte: trade_nodes.deterministic_sizing entry filter,
    # which runs AFTER assign_canary so per-ticker attribution is valid.
    "min_dte": ParamBound(
        family=ParamFamily.ENTRY_FILTERS,
        default=7, min=3, max=21, integer=True,
        description="Minimum days-to-expiry on new option entries.",
    ),

    # -- regime_thresholds -------------------------------------------------
    # v3 §6.2: VIX≥35, VIX 5d Δ≥+10, SPY 5d ≤−7%, corr≥0.80 + RV≥30, HYG ≤−5%
    # Two flags trip CRISIS.  These knobs let the learner tune sensitivity
    # within the published envelope.
    # LIVE consumer (area E): graph/nodes/regime_nodes.classify_regime now
    # threads these into classify() (crisis_params + confidence thresholds),
    # so an ACTIVE swap actually changes the live classifier and shadow replay
    # is non-fictional.  They stay canary_eligible=False NOT for lack of a
    # consumer but because the classifier runs at premarket/global scope while
    # canary routing is per-(ticker,date) at candidate_entry time — there is no
    # global-scope canary routing path yet, so an A/B canary can't attribute.
    "crisis_vix_level": ParamBound(
        family=ParamFamily.REGIME_THRESHOLDS,
        default=35.0, min=30.0, max=45.0,
        description="VIX absolute-level crisis flag.",
        canary_eligible=False,
    ),
    "crisis_vix_delta_5d": ParamBound(
        family=ParamFamily.REGIME_THRESHOLDS,
        default=10.0, min=7.0, max=15.0,
        description="VIX 5-day delta crisis flag.",
        canary_eligible=False,
    ),
    "crisis_spy_5d_drop_pct": ParamBound(
        family=ParamFamily.REGIME_THRESHOLDS,
        default=-7.0, min=-10.0, max=-5.0,
        description="SPY 5-day return crisis flag (negative pct).",
        canary_eligible=False,
    ),
    "regime_confidence_transition": ParamBound(
        family=ParamFamily.REGIME_THRESHOLDS,
        default=0.55, min=0.45, max=0.65,
        description="Below this HMM confidence → force VOLATILE_TRANSITION.",
        canary_eligible=False,
    ),
    "regime_confidence_llm_review": ParamBound(
        family=ParamFamily.REGIME_THRESHOLDS,
        default=0.65, min=0.55, max=0.75,
        description="Below this HMM confidence → request LLM second opinion.",
        canary_eligible=False,
    ),

    # NOTE: max_candidates_per_scout (family candidate_count) was PRUNED
    # (area E). The scout's top-N cap runs in premarket, upstream of
    # assign_canary and in a separate process, governed by DISPATCH_MIN_SCORE /
    # MAX_DISPATCH_PER_SCAN (not learnable) — it gates which tickers dispatch,
    # not any single trade's outcome, so there is no per-trade counterfactual.

    # -- regime_size_multipliers ------------------------------------------
    # CRISIS multiplier is INTENTIONALLY ABSENT — it is a hard zero.
    # Live consumer: trade_nodes.regime_execution_gate — effective multiplier
    # is min(persisted gate multiplier, resolver value), tighten-only.
    "size_mult_bull_trend": ParamBound(
        family=ParamFamily.REGIME_SIZE_MULTIPLIERS,
        default=1.00, min=0.50, max=1.00,
        description="Size multiplier under BULL_TREND.",
    ),
    "size_mult_range_low_vol": ParamBound(
        family=ParamFamily.REGIME_SIZE_MULTIPLIERS,
        default=0.75, min=0.30, max=1.00,
        description="Size multiplier under RANGE_LOW_VOL.",
    ),
    "size_mult_volatile_transition": ParamBound(
        family=ParamFamily.REGIME_SIZE_MULTIPLIERS,
        default=0.50, min=0.10, max=0.75,
        description="Size multiplier under VOLATILE_TRANSITION.",
    ),
    "size_mult_bear_trend": ParamBound(
        family=ParamFamily.REGIME_SIZE_MULTIPLIERS,
        default=0.50, min=0.10, max=0.75,
        description="Size multiplier under BEAR_TREND.",
    ),
}


# Frozen list of keys that MUST NEVER appear in a param_versions row.  Risk
# hard caps and the simulate/real toggle live here for explicit documentation.
FROZEN_HARD_CAPS: frozenset[str] = frozenset({
    "MAX_SINGLE_RISK_PCT",
    "MAX_OPTION_NOTIONAL_PCT",
    "MAX_CONCURRENT_OPENS",
    "MAX_TICKER_EXPOSURE_PCT",
    "MAX_SAME_SECTOR_OPENS",
    "OPTION_DELTA_MIN",
    "OPTION_DELTA_MAX",
    "EARNINGS_LOCK_DAYS",
    "size_mult_crisis",
    "enable_real",
    "trd_env",
})


class UnknownParamKey(KeyError):
    """Raised when a param_versions row contains a key outside the allow-list."""


class FrozenParamMutationAttempt(RuntimeError):
    """Raised if a param_versions row tries to mutate a hard-cap key."""


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

@dataclass
class ParamResolver:
    """A single parameter snapshot for one graph run.

    Built once per subgraph invocation by ``load_active_params()`` and threaded
    through ``state["params"]`` so every consumer (sizing, regime gates, risk,
    trader synthesizer) sees the same values for one decision.
    """

    version_id: int | None
    status: str  # DRAFT | SHADOW | CANARY | ACTIVE | DEFAULTS
    scope: str
    scope_key: str | None
    values: dict[str, float | int] = field(default_factory=dict)

    def get(self, key: str, default: float | int | None = None) -> float | int:
        if key not in PARAM_BOUNDS:
            raise UnknownParamKey(f"{key!r} is not a registered mutable param")
        if key in self.values:
            return self.values[key]
        if default is not None:
            return default
        return PARAM_BOUNDS[key].default

    def family(self, family: ParamFamily) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for k, b in PARAM_BOUNDS.items():
            if b.family is family:
                out[k] = self.get(k)
        return out

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "status": self.status,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "values": dict(self.values),
        }


def _validate_payload(params: dict[str, Any]) -> dict[str, float | int]:
    """Reject unknown / frozen keys; clamp to bounds; coerce to numeric."""
    cleaned: dict[str, float | int] = {}
    for k, v in params.items():
        if k in FROZEN_HARD_CAPS:
            raise FrozenParamMutationAttempt(
                f"param_versions row attempts to mutate frozen key {k!r}"
            )
        if k not in PARAM_BOUNDS:
            raise UnknownParamKey(f"{k!r} is not a registered mutable param")
        bound = PARAM_BOUNDS[k]
        try:
            numeric = int(v) if bound.integer else float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"param {k!r} value {v!r} not numeric") from e
        clamped = bound.clamp(numeric)
        if clamped != numeric:
            log.warning(
                "param %s value %s outside [%s, %s]; clamped to %s",
                k, numeric, bound.min, bound.max, clamped,
            )
        cleaned[k] = clamped
    return cleaned


def load_active_params(
    scope: str = "global",
    scope_key: str | None = None,
) -> ParamResolver:
    """Resolve the most recent ACTIVE row for (scope, scope_key).

    Falls back to a defaults-only resolver (``status=DEFAULTS``,
    ``version_id=None``) if no ACTIVE row exists or the database is degraded.
    Never raises — Phase 2.6 must never block a trade because the learning
    table is empty or unavailable.
    """
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, status, scope, scope_key, params
                FROM param_versions
                WHERE status = 'ACTIVE' AND scope = %s
                  AND (scope_key IS NOT DISTINCT FROM %s)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (scope, scope_key),
            )
            row = cur.fetchone()
    except Exception as e:
        log.warning("load_active_params: DB read failed (%s); using defaults", e)
        return ParamResolver(
            version_id=None, status="DEFAULTS", scope=scope, scope_key=scope_key
        )

    if not row:
        return ParamResolver(
            version_id=None, status="DEFAULTS", scope=scope, scope_key=scope_key
        )

    rid, status, sc, sk, params = row
    raw = params if isinstance(params, dict) else json.loads(params or "{}")
    try:
        cleaned = _validate_payload(raw)
    except (UnknownParamKey, FrozenParamMutationAttempt, ValueError) as e:
        log.error(
            "param_versions row id=%s rejected (%s); using defaults", rid, e
        )
        return ParamResolver(
            version_id=None, status="DEFAULTS", scope=scope, scope_key=scope_key
        )

    return ParamResolver(
        version_id=int(rid),
        status=str(status),
        scope=str(sc),
        scope_key=sk,
        values=cleaned,
    )


def insert_param_version(
    *,
    status: str,
    scope: str = "global",
    scope_key: str | None = None,
    params: dict[str, float | int],
    bounds: dict[str, dict[str, float | int]] | None = None,
    parent_version_id: int | None = None,
    rationale: str | None = None,
    created_by: str = "system",
    traffic_pct: float = 0.0,
) -> int:
    """Insert a new param_versions row and return its id.

    Validates that every key is in the allow-list and clamps values to bounds
    before insert.  Hard-cap mutation attempts raise.
    """
    cleaned = _validate_payload(params)
    bounds_payload = bounds or {
        k: {"min": PARAM_BOUNDS[k].min, "max": PARAM_BOUNDS[k].max}
        for k in cleaned
    }
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO param_versions
                (created_by, parent_version_id, status, scope, scope_key,
                 traffic_pct, params, bounds, rationale)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            RETURNING id
            """,
            (
                created_by,
                parent_version_id,
                status,
                scope,
                scope_key,
                traffic_pct,
                json.dumps(cleaned),
                json.dumps(bounds_payload),
                rationale,
            ),
        )
        return int(cur.fetchone()[0])


def list_versions(
    status: str | None = None,
    scope: str = "global",
    scope_key: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """List recent param_versions rows for diagnostics."""
    where = ["scope = %s"]
    args: list[Any] = [scope]
    if scope_key is not None:
        where.append("scope_key = %s")
        args.append(scope_key)
    if status:
        where.append("status = %s")
        args.append(status)
    args.append(limit)
    with cursor() as cur:
        cur.execute(
            f"""
            SELECT id, created_at, status, scope, scope_key, traffic_pct,
                   params, rationale, parent_version_id
            FROM param_versions
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            args,
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        rid, created_at, st, sc, sk, traffic, params, rat, parent = r
        out.append({
            "id": int(rid),
            "created_at": created_at,
            "status": st,
            "scope": sc,
            "scope_key": sk,
            "traffic_pct": float(traffic),
            "params": params if isinstance(params, dict) else json.loads(params or "{}"),
            "rationale": rat,
            "parent_version_id": parent,
        })
    return out


def seed_baseline_if_absent(scope: str = "global") -> int | None:
    """Insert a baseline ACTIVE row from defaults if none exists.

    Idempotent — if an ACTIVE row already exists for ``(scope, NULL)``, returns
    its id without writing.  Designed to be safe to call from a migration step
    or from an orchestrator startup hook.
    """
    existing = load_active_params(scope=scope, scope_key=None)
    if existing.version_id is not None:
        log.info(
            "seed_baseline_if_absent: scope=%s already has ACTIVE id=%s",
            scope, existing.version_id,
        )
        return existing.version_id
    defaults = {k: b.default for k, b in PARAM_BOUNDS.items()}
    new_id = insert_param_version(
        status="ACTIVE",
        scope=scope,
        scope_key=None,
        params=defaults,
        rationale="Phase 2.6 baseline seed — code-level defaults",
        created_by="phase_2_6_seed",
        traffic_pct=100.0,
    )
    log.info("seed_baseline_if_absent: inserted ACTIVE param_version id=%s", new_id)
    return new_id


def defaults_resolver() -> ParamResolver:
    """Return a resolver wired to code-level defaults (no DB access)."""
    return ParamResolver(
        version_id=None, status="DEFAULTS", scope="global", scope_key=None
    )
