"""ITEM 11 — paired counterfactual evaluation for qty-only canaries.

Why this exists: traffic-split canary stats are the WRONG instrument for
parameter families whose only effect is position size.  A qty-only change
(sizing_aggression soft caps, regime_size_multipliers) rescales each trade's
realized R deterministically — ``replay.replay_one`` computes the exact
counterfactual (qty_shadow / realized_r_shadow) for any closed trade, no
matter which arm it was routed to.  Compared to the traffic split, the
paired replay:

  * is deterministic — no assignment-hash noise, rerunning gives the same
    answer;
  * uses 100% of the sample — every closed trade since the canary's
    created_at is a paired observation, instead of throwing away the
    control arm's trades for the treatment estimate;
  * has no attribution problem — d_i = r_shadow_i − r_actual_i differences
    cancel the per-trade variance (ticker, regime, thesis quality) that
    dominates small-n arm-vs-arm comparisons.

Decision rule on the paired differences d_i:

    PROMOTE  iff  LB95(mean d) > 0  and  n >= N_PAIRED_MIN
    REJECT   iff  UB95(mean d) < 0
    RETAIN   otherwise

Non-qty families keep promote.py's traffic-split evaluation — their effects
(different stops, different entries) are NOT post-hoc computable from the
realized fills, so the split is the only unbiased estimator there.

NOTE for the merger: ``_mean_ci_95`` deliberately duplicates the
lcb-of-mean math a sibling cluster is adding to ``learning/_stats.py``
(``lcb_mean``) — consolidate the two after both branches land.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from trading_agent.learning.params import (
    PARAM_BOUNDS,
    ParamFamily,
    ParamResolver,
    defaults_resolver,
)
from trading_agent.learning.replay import (
    ClosedTradeRow,
    fetch_closed_trades,
    replay_one,
)
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

# Families whose entire causal effect is a qty rescale of the same fills.
QTY_ONLY_FAMILIES: frozenset[ParamFamily] = frozenset({
    ParamFamily.SIZING_AGGRESSION,
    ParamFamily.REGIME_SIZE_MULTIPLIERS,
})

# Minimum paired sample before either tail of the CI may decide.  Matches
# canary.N_CANARY_MIN in spirit, but counts CLOSED trades with a realized_r
# — not routed journal rows.
N_PAIRED_MIN = 20

Z_95 = 1.959963984540054  # two-sided 95%, same constant as promote.WILSON_Z_95


def is_qty_only_params(params: dict[str, Any]) -> bool:
    """True iff every recognised key sits in a qty-only family.

    Unrecognised keys are ignored (they were rejected at insert time);
    an empty / fully-unrecognised dict returns False so the caller falls
    back to the traffic-split path rather than promoting a no-op row.
    """
    fams = {
        PARAM_BOUNDS[k].family for k in params if k in PARAM_BOUNDS
    }
    return bool(fams) and fams <= QTY_ONLY_FAMILIES


def _mean_ci_95(ds: list[float]) -> tuple[float, float, float]:
    """(mean, lb95, ub95) of the paired differences, normal approximation.

    Sample sd (ddof=1).  With sd == 0 the CI collapses onto the mean —
    correct behavior: identical candidate ⇒ d ≡ 0 ⇒ lb = ub = 0 ⇒ RETAIN
    (no evidence either way), never PROMOTE.
    """
    n = len(ds)
    if n == 0:
        return 0.0, float("-inf"), float("inf")
    mean = sum(ds) / n
    if n < 2:
        return mean, float("-inf"), float("inf")
    var = sum((d - mean) ** 2 for d in ds) / (n - 1)
    half = Z_95 * math.sqrt(var / n)
    return mean, mean - half, mean + half


@dataclass
class CounterfactualDecision:
    """Outcome of one paired-counterfactual evaluation (promote-compatible)."""

    canary_version_id: int
    decision: Literal["PROMOTE", "RETAIN", "REJECT"]
    rationale: list[str] = field(default_factory=list)
    n_pairs: int = 0
    n_skipped_missing_outcome: int = 0
    mean_diff: float = 0.0
    lb95: float = 0.0
    ub95: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "canary_version_id": self.canary_version_id,
            "decision": self.decision,
            "method": "paired_counterfactual",
            "rationale": list(self.rationale),
            "n_pairs": self.n_pairs,
            "n_skipped_missing_outcome": self.n_skipped_missing_outcome,
            "mean_diff": round(self.mean_diff, 6),
            "lb95": round(self.lb95, 6),
            "ub95": round(self.ub95, 6),
        }


def _resolver_for_version(version_id: int) -> ParamResolver | None:
    """Materialize a resolver from one param_versions row, or None on failure."""
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT status, scope, scope_key, params "
                "FROM param_versions WHERE id = %s",
                (version_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        log.warning("_resolver_for_version(%s) failed: %s", version_id, e)
        return None
    if row is None:
        return None
    status, scope, scope_key, params = row
    if not isinstance(params, dict):
        import json
        params = json.loads(params or "{}")
    return ParamResolver(
        version_id=int(version_id),
        status=str(status),
        scope=str(scope),
        scope_key=scope_key,
        values={
            k: PARAM_BOUNDS[k].clamp(v)
            for k, v in params.items() if k in PARAM_BOUNDS
        },
    )


def paired_diffs(
    trades: list[ClosedTradeRow],
    candidate: ParamResolver,
    baseline: ParamResolver,
) -> tuple[list[float], int]:
    """d_i = r_shadow_i − r_actual_i for every trade with a realized outcome.

    Trades with realized_r IS NULL are EXCLUDED (counted, not zero-filled):
    a missing outcome enrichment is not a 0R trade, and zero-filled pairs
    would shrink the mean toward 0 and bias the gate to RETAIN forever.
    """
    ds: list[float] = []
    n_skipped = 0
    for t in trades:
        if t.realized_r is None:
            n_skipped += 1
            continue
        rep = replay_one(t, candidate, baseline=baseline)
        ds.append(rep.realized_r_shadow - rep.realized_r_actual)
    return ds, n_skipped


def evaluate_qty_only(
    *,
    canary_version_id: int,
    candidate_params: dict[str, Any],
    baseline_version_id: int | None,
    since: datetime | None,
    trades: list[ClosedTradeRow] | None = None,
    baseline: ParamResolver | None = None,
) -> CounterfactualDecision:
    """Exact paired counterfactual over all closed trades since ``since``.

    ``trades`` / ``baseline`` are injectable for tests and offline analysis;
    production callers (promote.evaluate_canary dispatch) pass neither.
    """
    candidate = ParamResolver(
        version_id=canary_version_id,
        status="CANARY",
        scope="global",
        scope_key=None,
        values={
            k: PARAM_BOUNDS[k].clamp(v)
            for k, v in candidate_params.items() if k in PARAM_BOUNDS
        },
    )
    if baseline is None:
        baseline = (
            _resolver_for_version(int(baseline_version_id))
            if baseline_version_id is not None else None
        ) or defaults_resolver()
    if trades is None:
        # ALL closed trades since the canary's creation — not just the
        # routed slice; that's the whole point of the paired design.
        trades = fetch_closed_trades(since=since, limit=1000)

    ds, n_skipped = paired_diffs(trades, candidate, baseline)
    n = len(ds)
    mean, lb, ub = _mean_ci_95(ds)

    stats_line = (
        f"paired counterfactual: n={n} (skipped {n_skipped} without "
        f"realized_r), mean_d={mean:.4f}, "
        f"CI95=[{lb:.4f}, {ub:.4f}] vs baseline "
        f"version_id={baseline.version_id}"
    )

    if n < N_PAIRED_MIN:
        return CounterfactualDecision(
            canary_version_id=canary_version_id, decision="RETAIN",
            rationale=[
                f"only {n}/{N_PAIRED_MIN} paired closed trades since "
                "canary creation — wait for more samples",
                stats_line,
            ],
            n_pairs=n, n_skipped_missing_outcome=n_skipped,
            mean_diff=mean, lb95=lb, ub95=ub,
        )

    if lb > 0:
        decision: Literal["PROMOTE", "RETAIN", "REJECT"] = "PROMOTE"
        verdict = "LB95(mean d) > 0 — candidate dominates baseline"
    elif ub < 0:
        decision = "REJECT"
        verdict = "UB95(mean d) < 0 — candidate is confidently worse"
    else:
        decision = "RETAIN"
        verdict = "CI95 straddles 0 — no significant difference yet"

    return CounterfactualDecision(
        canary_version_id=canary_version_id, decision=decision,
        rationale=[stats_line, verdict],
        n_pairs=n, n_skipped_missing_outcome=n_skipped,
        mean_diff=mean, lb95=lb, ub95=ub,
    )


def maybe_evaluate_qty_only(
    canary_version_id: int,
) -> CounterfactualDecision | None:
    """Dispatch helper for promote.evaluate_canary.

    Returns a decision ONLY when the row is a live CANARY whose changed keys
    are all qty-only; returns None otherwise — including on ANY read failure
    — so promote.py falls through to the traffic-split path, which carries
    its own fail-safe gating (RETAIN on thin/odd evidence).
    """
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT status, parent_version_id, params, created_at "
                "FROM param_versions WHERE id = %s",
                (canary_version_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        log.warning(
            "maybe_evaluate_qty_only(%s): row fetch failed (%s) — "
            "falling back to traffic-split evaluation",
            canary_version_id, e,
        )
        return None
    if row is None:
        return None
    status, parent_id, params, created_at = row
    if str(status) != "CANARY":
        return None
    if not isinstance(params, dict):
        import json
        try:
            params = json.loads(params or "{}")
        except Exception:
            return None
    if not is_qty_only_params(params):
        return None
    try:
        return evaluate_qty_only(
            canary_version_id=canary_version_id,
            candidate_params=params,
            baseline_version_id=int(parent_id) if parent_id is not None else None,
            since=created_at,
        )
    except Exception as e:
        log.warning(
            "maybe_evaluate_qty_only(%s): evaluation failed (%s) — "
            "falling back to traffic-split evaluation",
            canary_version_id, e,
        )
        return None


__all__ = [
    "N_PAIRED_MIN",
    "QTY_ONLY_FAMILIES",
    "Z_95",
    "CounterfactualDecision",
    "evaluate_qty_only",
    "is_qty_only_params",
    "maybe_evaluate_qty_only",
    "paired_diffs",
]
