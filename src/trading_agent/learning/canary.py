"""Phase 2.7 — Canary routing: stable-hash A/B over CANARY param_versions.

Bridges Phase 2.6.5's diversity archive with Phase 2.7.5's LLM Critic by
giving any ``param_versions status=CANARY`` row a way to drive a small slice
of *real* paper trades — not just shadow-replays.

Routing contract:
    bucket(ticker, date, family) = sha256("ticker|date|family")[:4] mod 100
    if bucket < canary.traffic_pct:  use canary
    else                            use control

The hash inputs are stable per (ticker, date, family) so every fire of
``candidate_entry_graph`` for the same ticker on the same day picks the same
side.  Different families hash independently, so a single ticker can route
its sizing-aggression knob to the canary while keeping its regime-thresholds
on the control.

Single-active-canary-per-family is enforced at insert time (see
``learning.params.insert_param_version``) and re-checked here so a stale row
can't double-route.  Hard caps and the simulate/real toggle remain frozen.

Traffic ramp:
    new CANARY rows start at 10% traffic.
    after 10 trades clean (no rollback fired), the eod promote-or-rollback
    node bumps the row to 20% and emits a ``learning_events.canary_ramped``.
    after N_canary_min=20 trades, promote.evaluate_canary checks gates.

Live decision flow (per candidate_entry):
    load_active_params_node      -> control resolver in state["learning"]
    research_ticker (target known)
    assign_canary_node           -> for each active canary family, hash;
                                    if bucket<traffic, overlay canary keys;
                                    record learning_assignments row
    -> downstream sizing/regime/risk/proposal see the routed values
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trading_agent.learning.params import (
    PARAM_BOUNDS,
    ParamFamily,
    ParamResolver,
    list_versions,
    load_active_params,
)
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

DEFAULT_INITIAL_TRAFFIC_PCT = 10.0
RAMPED_TRAFFIC_PCT = 20.0
RAMP_AFTER_N_TRADES = 10
N_CANARY_MIN = 20  # min trades before promotion is even considered


# ---------------------------------------------------------------------------
# Family inference
# ---------------------------------------------------------------------------

class MultiFamilyCanary(ValueError):
    """Raised when a CANARY row's params jsonb spans multiple families."""


def family_of_params(params: dict[str, Any]) -> ParamFamily:
    """Infer the single family this param_version mutates.

    A canary row should target ONE knob family at a time.  Multi-family
    rows confound attribution and are rejected here.
    """
    families: set[ParamFamily] = set()
    for k in params:
        bound = PARAM_BOUNDS.get(k)
        if bound is None:
            # Unknown keys would have been rejected at insert time; defensive
            continue
        families.add(bound.family)
    if len(families) == 0:
        raise MultiFamilyCanary("canary row has no recognised mutable params")
    if len(families) > 1:
        raise MultiFamilyCanary(
            f"canary row spans {len(families)} families: "
            f"{sorted(f.value for f in families)}"
        )
    return next(iter(families))


def ineligible_keys(params: dict[str, Any]) -> list[str]:
    """Recognised keys in ``params`` that have NO live consumer.

    ``ParamBound.canary_eligible`` is the per-key source of truth (ITEM 11):
    a canary touching any ineligible key would be an A/A test — the journal
    rows get a different params_version_id but the decision path never reads
    the value, so promotion gates compare two identical treatments.
    """
    return sorted(
        k for k in params
        if k in PARAM_BOUNDS and not PARAM_BOUNDS[k].canary_eligible
    )


# ---------------------------------------------------------------------------
# Bucket math
# ---------------------------------------------------------------------------

def bucket_for(ticker: str, date_str: str, family: ParamFamily | str) -> int:
    """Stable hash → bucket integer in [0, 100)."""
    fam = family.value if isinstance(family, ParamFamily) else str(family)
    h = hashlib.sha256(f"{ticker}|{date_str}|{fam}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def date_key(ts: str | datetime | None = None) -> str:
    """Date stamp used as the second hash input.  Day-resolution is enough —
    intraday re-entries on the same ticker on the same day must route the
    same way for the gates to be statistically interpretable."""
    if isinstance(ts, datetime):
        d = ts.astimezone(timezone.utc) if ts.tzinfo else ts
    elif isinstance(ts, str) and ts:
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            d = datetime.now(timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Listing + family resolution
# ---------------------------------------------------------------------------

@dataclass
class ActiveCanary:
    """A row from `param_versions status=CANARY` plus inferred family."""

    version_id: int
    parent_version_id: int | None
    family: ParamFamily
    traffic_pct: float
    params: dict[str, float | int]
    created_at: datetime | None
    rationale: str | None = None


def list_active_canaries() -> list[ActiveCanary]:
    """Return well-formed CANARY rows.  Multi-family or empty rows are
    skipped + logged so a single bad row can't poison routing."""
    out: list[ActiveCanary] = []
    try:
        rows = list_versions(status="CANARY", limit=50)
    except Exception as e:
        log.warning("list_active_canaries: db read failed (%s)", e)
        return out
    seen_families: set[ParamFamily] = set()
    for r in rows:
        # Clamp values at read time, exactly like the ACTIVE read path
        # (params._validate_payload): route_canary overlays these verbatim
        # onto the live resolver, and the stop-distance consumer applies
        # them without its own re-validation — an out-of-band DB edit must
        # not be able to push a live stop outside the declared bounds.
        params = {k: PARAM_BOUNDS[k].clamp(v)
                  for k, v in (r.get("params") or {}).items()
                  if k in PARAM_BOUNDS}
        try:
            family = family_of_params(params)
        except MultiFamilyCanary as e:
            log.warning(
                "list_active_canaries: skipping bad CANARY id=%s (%s)",
                r["id"], e,
            )
            continue
        if family in seen_families:
            log.warning(
                "list_active_canaries: duplicate canary in family=%s "
                "(extra row id=%s ignored)",
                family.value, r["id"],
            )
            continue
        seen_families.add(family)
        out.append(ActiveCanary(
            version_id=int(r["id"]),
            parent_version_id=r.get("parent_version_id"),
            family=family,
            traffic_pct=float(r.get("traffic_pct") or 0.0),
            params=params,
            created_at=r.get("created_at"),
            rationale=r.get("rationale"),
        ))
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    family: ParamFamily
    canary_version_id: int
    bucket: int
    traffic_pct: float
    routed_to_canary: bool
    overlaid_keys: list[str] = field(default_factory=list)


@dataclass
class RoutedResolver:
    """Output of `route_canary` — final resolver + audit trail.

    ``skipped_ineligible`` rows received NO traffic and NO
    learning_assignments record (assign_canary_node only iterates the two
    routed lists) — they're surfaced in the audit dict so an operator can
    see a mis-created canary is being ignored rather than silently 50/50'd.
    """

    resolver: ParamResolver
    routed_to_canary: list[RouteDecision] = field(default_factory=list)
    routed_to_control: list[RouteDecision] = field(default_factory=list)
    skipped_ineligible: list[dict[str, Any]] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "resolver_version_id": self.resolver.version_id,
            "resolver_status": self.resolver.status,
            "n_canary_routes_taken": len(self.routed_to_canary),
            "skipped_ineligible": list(self.skipped_ineligible),
            "decisions": [
                {
                    "family": d.family.value,
                    "canary_version_id": d.canary_version_id,
                    "bucket": d.bucket,
                    "traffic_pct": d.traffic_pct,
                    "routed_to_canary": d.routed_to_canary,
                    "overlaid_keys": list(d.overlaid_keys),
                }
                for d in (self.routed_to_canary + self.routed_to_control)
            ],
        }


def route_canary(
    control: ParamResolver,
    *,
    ticker: str,
    ts: str | datetime | None = None,
    canaries: list[ActiveCanary] | None = None,
) -> RoutedResolver:
    """Decide control vs canary per family and overlay canary keys onto a
    fresh ParamResolver.

    The control resolver is never mutated.  The returned resolver carries
    the control's defaults + canary keys for each family that hashed below
    its traffic_pct.
    """
    canaries = canaries if canaries is not None else list_active_canaries()
    routed_to_canary: list[RouteDecision] = []
    routed_to_control: list[RouteDecision] = []

    overlaid_values = dict(control.values)
    final_status = control.status
    final_version_id = control.version_id
    d_str = date_key(ts)
    skipped_ineligible: list[dict[str, Any]] = []

    for c in canaries:
        # ITEM 11 guard: never route traffic to a canary whose only effect
        # is a label.  A row touching any key without a live consumer is an
        # A/A test by construction — burn zero traffic on it.
        bad_keys = ineligible_keys(c.params)
        if bad_keys:
            log.warning(
                "route_canary: skipping CANARY id=%s — keys %s have no live "
                "consumer (canary_eligible=False)",
                c.version_id, bad_keys,
            )
            skipped_ineligible.append({
                "canary_version_id": c.version_id,
                "family": c.family.value,
                "ineligible_keys": bad_keys,
            })
            continue
        b = bucket_for(ticker, d_str, c.family)
        decision = RouteDecision(
            family=c.family,
            canary_version_id=c.version_id,
            bucket=b,
            traffic_pct=c.traffic_pct,
            routed_to_canary=(b < c.traffic_pct),
            overlaid_keys=[],
        )
        if decision.routed_to_canary:
            for k, v in c.params.items():
                overlaid_values[k] = v
                decision.overlaid_keys.append(k)
            # When we route to ANY canary, mark the resolver as CANARY-touched
            final_status = "CANARY"
            final_version_id = c.version_id
            routed_to_canary.append(decision)
        else:
            routed_to_control.append(decision)

    final = ParamResolver(
        version_id=final_version_id,
        status=final_status,
        scope=control.scope,
        scope_key=control.scope_key,
        values=overlaid_values,
    )
    return RoutedResolver(
        resolver=final,
        routed_to_canary=routed_to_canary,
        routed_to_control=routed_to_control,
        skipped_ineligible=skipped_ineligible,
    )


# ---------------------------------------------------------------------------
# learning_assignments writer
# ---------------------------------------------------------------------------

def record_assignment(
    *,
    proposal_id: str,
    ticker: str,
    strategy_label: str,
    baseline_param_version_id: int | None,
    assigned_param_version_id: int,
    bucket: int,
    reason: str,
) -> int | None:
    """Insert a learning_assignments row.  Returns the new id or None on err."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO learning_assignments
                    (proposal_id, ticker, strategy_label,
                     baseline_param_version_id, assigned_param_version_id,
                     assignment_bucket, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    proposal_id,
                    ticker,
                    strategy_label,
                    baseline_param_version_id,
                    assigned_param_version_id,
                    bucket,
                    reason,
                ),
            )
            return int(cur.fetchone()[0])
    except Exception as e:
        log.warning("record_assignment failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Traffic ramp helpers
# ---------------------------------------------------------------------------

def n_trades_for_canary(canary_version_id: int) -> int:
    """How many journal_trades have so far been routed to this CANARY."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM journal_trades
                WHERE params_version_id = %s
                """,
                (canary_version_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        log.warning("n_trades_for_canary failed: %s", e)
        return 0


def maybe_ramp_traffic(canary_version_id: int) -> bool:
    """Ramp 10% → 20% after first ``RAMP_AFTER_N_TRADES`` clean trades.

    "Clean" here means simply: no rollback fired yet.  If a rollback
    occurred the row has status=REJECTED and this function returns False.
    Returns True if a ramp UPDATE was applied.

    Atomic: the conditional UPDATE eliminates the SELECT/UPDATE TOCTOU —
    concurrent eod runs cannot double-ramp because only the first one
    matches ``traffic_pct < RAMPED_TRAFFIC_PCT`` and reserves the row.
    """
    n = n_trades_for_canary(canary_version_id)
    if n < RAMP_AFTER_N_TRADES:
        return False
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE param_versions
                SET traffic_pct = %s
                WHERE id = %s
                  AND status = 'CANARY'
                  AND traffic_pct < %s
                RETURNING traffic_pct
                """,
                (RAMPED_TRAFFIC_PCT, canary_version_id, RAMPED_TRAFFIC_PCT),
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute(
                """
                INSERT INTO learning_events (event_type, param_version_id, payload)
                VALUES ('canary_ramped', %s, %s::jsonb)
                """,
                (
                    canary_version_id,
                    json.dumps({
                        "to_pct": RAMPED_TRAFFIC_PCT,
                        "n_trades": n,
                    }),
                ),
            )
            return True
    except Exception as e:
        log.warning("maybe_ramp_traffic failed for id=%s: %s", canary_version_id, e)
        return False


# ---------------------------------------------------------------------------
# Insertion helper for Phase 2.7.5 (LLM Critic creates CANARYs)
# ---------------------------------------------------------------------------

# Derived from the per-key ``ParamBound.canary_eligible`` flags so there is
# ONE source of truth (params.py) for "can this knob be A/B-routed live".
# A family lands here when NO key in it is eligible. After area E pruned the
# consumer-less keys and wired min_dte, the only fully-ineligible family is:
#   * REGIME_THRESHOLDS — the classifier now consumes these (classify_regime
#     threads them into classify()), so an ACTIVE swap is real, BUT it runs at
#     premarket/global scope while canary routing is per-(ticker,date) at
#     candidate_entry time. There is no global-scope canary routing path yet,
#     so an A/B canary on them can't attribute — keep them out of live routing.
# The per-key checks in ``insert_canary`` / ``route_canary`` remain as defence
# for any FUTURE mixed family (an eligible family carrying an ineligible key);
# none exists in PARAM_BOUNDS today.
CANARY_INELIGIBLE_FAMILIES: frozenset[ParamFamily] = frozenset(
    fam for fam in ParamFamily
    if not any(
        b.canary_eligible for b in PARAM_BOUNDS.values() if b.family is fam
    )
)


class CanaryFamilyIneligible(RuntimeError):
    """Raised when a CANARY targets a family that can't yet be routed live."""


def insert_canary(
    *,
    parent_version_id: int,
    params: dict[str, float | int],
    rationale: str,
    created_by: str,
    initial_traffic_pct: float = DEFAULT_INITIAL_TRAFFIC_PCT,
) -> int:
    """Create a CANARY row.  Validates single-family + no active canary
    in that family; raises on conflict so the caller learns immediately."""
    fam = family_of_params(params)  # raises MultiFamilyCanary if invalid
    if fam in CANARY_INELIGIBLE_FAMILIES:
        raise CanaryFamilyIneligible(
            f"family {fam.value} cannot be routed live yet — "
            "use SHADOW status until node-ordering rearchitect lands"
        )
    # Per-key check for any FUTURE mixed family (an eligible family carrying
    # an ineligible key): refuse at creation time so the caller learns
    # immediately instead of route_canary silently skipping the row forever.
    # No such family exists in PARAM_BOUNDS today (area E pruned the last one).
    bad = ineligible_keys(params)
    if bad:
        raise CanaryFamilyIneligible(
            f"keys {bad} have no live consumer (canary_eligible=False) — "
            "a canary on them is an A/A test; use SHADOW status instead"
        )
    actives = list_active_canaries()
    for a in actives:
        if a.family == fam and a.version_id != parent_version_id:
            raise RuntimeError(
                f"family {fam.value} already has active canary id={a.version_id}; "
                "retire or reject it before opening another"
            )
    from trading_agent.learning.params import insert_param_version
    new_id = insert_param_version(
        status="CANARY",
        scope="global",
        scope_key=None,
        params=params,
        parent_version_id=parent_version_id,
        rationale=rationale,
        created_by=created_by,
        traffic_pct=initial_traffic_pct,
    )
    log.info(
        "insert_canary: id=%s family=%s parent=%s traffic=%s",
        new_id, fam.value, parent_version_id, initial_traffic_pct,
    )
    return new_id


__all__ = [
    "CANARY_INELIGIBLE_FAMILIES",
    "DEFAULT_INITIAL_TRAFFIC_PCT",
    "RAMPED_TRAFFIC_PCT",
    "RAMP_AFTER_N_TRADES",
    "N_CANARY_MIN",
    "ActiveCanary",
    "CanaryFamilyIneligible",
    "MultiFamilyCanary",
    "RouteDecision",
    "RoutedResolver",
    "bucket_for",
    "date_key",
    "family_of_params",
    "ineligible_keys",
    "insert_canary",
    "list_active_canaries",
    "maybe_ramp_traffic",
    "n_trades_for_canary",
    "record_assignment",
    "route_canary",
]
