"""Phase 2.7 — Canary promotion + auto-rollback gates.

Decisions for a CANARY param_version row are made daily by
``promote_or_rollback_node`` inside ``eod_review_graph``:

  1. **Hard rollback** — fires if any of:
       a. ≥ 1 hard-risk breach (R1–R6 deterministic block) in first 5 trades
       b. 5 consecutive losing days
     → status = REJECTED with a structured ``learning_events.canary_rolled_back``.

  2. **Promotion** — only considered after ``N_canary_min = 20`` trades, AND
     only when BOTH canary and control have ≥ ``N_canary_min`` closed outcomes
     in ``trade_outcome_features`` (thin evidence fails safe to RETAIN — see
     the comment at the evidence gate).  Gates:
       - PRIMARY (Phase 2 item 10): LB95(mean realized R) of the canary ≥
         max(0, control's mean-R point estimate) — the canary must be
         credibly non-negative AND not credibly worse than control
         (one-sided Student-t bound, ``_stats.lcb_mean``)
       - profit factor ≥ control × 0.95
       - max drawdown ≤ control × 1.1
     Win rate and its Wilson LB are DIAGNOSTICS only: still computed and
     reported in the rationale, never gating.  Re-keyed from the old
     Wilson-LB-on-win-rate gate because win rate is not expectancy — the
     convexity track's declared shape (strategy_specs.py) is 30–45% WR with
     fat winners, which a win-rate gate structurally rejects while letting
     high-WR books that bleed R per trade promote.
     If all three pass → status = ACTIVE; old ACTIVE → RETIRED;
     ``params_history`` snapshot row written; ``learning_events.canary_promoted``.

  3. **Retain** — fewer than ``N_canary_min`` trades, no breach yet.  No-op,
     traffic ramp may still happen via ``canary.maybe_ramp_traffic``.

This module does the math + persistence; the LangGraph node in
``graph/nodes/eod_learning.py`` calls into ``evaluate_and_apply()`` once per
EOD run.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from trading_agent.learning._stats import lcb_mean
from trading_agent.learning.canary import (
    N_CANARY_MIN,
    n_trades_for_canary,
)
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

PROFIT_FACTOR_FRAC = 0.95
MDD_FRAC = 1.10
ROLLBACK_LOSING_DAYS = 5
ROLLBACK_FIRST_N_TRADES = 5
WILSON_Z_95 = 1.959963984540054  # two-sided 95%

DecisionStr = Literal["PROMOTE", "RETAIN", "REJECT"]


# ---------------------------------------------------------------------------
# Wilson interval — DIAGNOSTIC only since the item-10 re-key.  Win rate is
# not expectancy; this is reported in rationale but never gates promotion.
# ---------------------------------------------------------------------------

def wilson_lb(wins: int, n: int, z: float = WILSON_Z_95) -> float:
    """Wilson 95% CI lower bound on the proportion ``wins/n``.

    Returns 0.0 when n=0 (no evidence of any positive rate).
    """
    if n <= 0:
        return 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return max(0.0, (center - half) / denom)


# ---------------------------------------------------------------------------
# Outcome aggregates per param_version
# ---------------------------------------------------------------------------

@dataclass
class OutcomeStats:
    n: int
    wins: int
    losses: int
    win_rate: float          # diagnostic — never gates (item-10 re-key)
    wilson_lb: float         # diagnostic — never gates (item-10 re-key)
    gross_win: float
    gross_loss: float
    profit_factor: float
    cum_R: float
    max_drawdown_R: float
    mean_R: float            # point estimate of expectancy per trade
    lcb_mean_R: float | None  # LB95(mean R); None when n < 2 (no evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "wilson_lb": round(self.wilson_lb, 4),
            "gross_win": round(self.gross_win, 4),
            "gross_loss": round(self.gross_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "cum_R": round(self.cum_R, 4),
            "max_drawdown_R": round(self.max_drawdown_R, 4),
            "mean_R": round(self.mean_R, 4),
            "lcb_mean_R": (round(self.lcb_mean_R, 4)
                           if self.lcb_mean_R is not None else None),
        }


def _stats_for_version(param_version_id: int) -> OutcomeStats:
    """Aggregate trade_outcome_features for one param_version chronologically.

    Joins through journal_trades to use opened_at as the chronological key
    (so the cumulative drawdown is in trade order, not insert order).
    """
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT tof.realized_r
                FROM trade_outcome_features tof
                JOIN journal_trades jt ON jt.id = tof.trade_id
                WHERE tof.param_version_id = %s
                  AND tof.realized_r IS NOT NULL
                ORDER BY jt.opened_at ASC
                """,
                (param_version_id,),
            )
            rs = [float(r[0]) for r in cur.fetchall()]
    except Exception as e:
        log.warning("_stats_for_version(%s) failed: %s", param_version_id, e)
        rs = []
    return _aggregate(rs)


def _aggregate(rs: list[float]) -> OutcomeStats:
    from trading_agent.learning._stats import series_max_drawdown
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    losses = sum(1 for r in rs if r < 0)
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))
    win_rate = (wins / n) if n > 0 else 0.0
    wlb = wilson_lb(wins, n)
    if gross_loss > 1e-9:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    cum, mdd = series_max_drawdown(rs)
    return OutcomeStats(
        n=n, wins=wins, losses=losses,
        win_rate=win_rate, wilson_lb=wlb,
        gross_win=gross_win, gross_loss=gross_loss,
        profit_factor=pf,
        cum_R=cum, max_drawdown_R=mdd,
        mean_R=(cum / n) if n > 0 else 0.0,
        lcb_mean_R=lcb_mean(rs),
    )


# ---------------------------------------------------------------------------
# Rollback triggers
# ---------------------------------------------------------------------------

def first_n_trades_breach(canary_version_id: int, n: int = ROLLBACK_FIRST_N_TRADES) -> bool:
    """True if any of the canary's first ``n`` trades hit a hard-risk
    block — i.e., the deterministic_risk_guardrails decision was VETO
    on the proposal that ultimately got routed through this canary, OR
    the trade closed at ≤ −1.5R (deeper than its own stop).
    """
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT tof.realized_r
                FROM trade_outcome_features tof
                JOIN journal_trades jt ON jt.id = tof.trade_id
                WHERE tof.param_version_id = %s
                  AND tof.realized_r IS NOT NULL
                ORDER BY jt.opened_at ASC
                LIMIT %s
                """,
                (canary_version_id, n),
            )
            rs = [float(r[0]) for r in cur.fetchall()]
    except Exception as e:
        log.warning("first_n_trades_breach: db read failed (%s)", e)
        return False
    if not rs:
        return False
    # A deeper-than-stop close (≤ −1.5R) is the deterministic indicator
    # we use here.  Strict R-trigger violations are surfaced separately
    # via agent_events / risk_decisions.hard_violations — see Phase 2.7.5.
    return any(r <= -1.5 for r in rs)


def consecutive_losing_days(canary_version_id: int) -> int:
    """Count consecutive days (most recent backwards) where every closed
    trade routed to this canary lost on a calendar-UTC basis."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT
                    DATE(jt.closed_at AT TIME ZONE 'UTC') AS d,
                    SUM(CASE WHEN tof.realized_r > 0 THEN 1 ELSE 0 END) AS wins,
                    COUNT(*) AS n
                FROM trade_outcome_features tof
                JOIN journal_trades jt ON jt.id = tof.trade_id
                WHERE tof.param_version_id = %s
                  AND jt.closed_at IS NOT NULL
                  AND tof.realized_r IS NOT NULL
                GROUP BY DATE(jt.closed_at AT TIME ZONE 'UTC')
                ORDER BY d DESC
                """,
                (canary_version_id,),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("consecutive_losing_days failed: %s", e)
        return 0
    streak = 0
    for d, wins, n in rows:
        if int(wins) == 0 and int(n) > 0:
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromoteDecision:
    canary_version_id: int
    decision: DecisionStr
    rationale: list[str] = field(default_factory=list)
    canary_stats: OutcomeStats | None = None
    control_stats: OutcomeStats | None = None
    rollback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "canary_version_id": self.canary_version_id,
            "decision": self.decision,
            "rationale": list(self.rationale),
            "rollback_reason": self.rollback_reason,
        }
        if self.canary_stats:
            d["canary_stats"] = self.canary_stats.to_dict()
        if self.control_stats:
            d["control_stats"] = self.control_stats.to_dict()
        return d


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def evaluate_canary(canary_version_id: int) -> PromoteDecision:
    """Decide whether to promote, retain, or reject a canary row."""
    # 0. Sanity — only consider rows still flagged CANARY
    with cursor() as cur:
        cur.execute(
            "SELECT status, parent_version_id FROM param_versions WHERE id = %s",
            (canary_version_id,),
        )
        row = cur.fetchone()
    if row is None:
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="RETAIN",
            rationale=["row not found"],
        )
    status, parent_id = row
    if str(status) != "CANARY":
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="RETAIN",
            rationale=[f"status={status} — not a live canary"],
        )

    # 1. Hard rollback triggers
    if first_n_trades_breach(canary_version_id):
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="REJECT",
            rationale=["hard-risk breach in first 5 trades (≤ -1.5R)"],
            rollback_reason="first5_breach",
        )
    streak = consecutive_losing_days(canary_version_id)
    if streak >= ROLLBACK_LOSING_DAYS:
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="REJECT",
            rationale=[f"{streak} consecutive losing days — rollback gate"],
            rollback_reason=f"losing_streak_{streak}",
        )

    # 2. Sample-size gate
    n = n_trades_for_canary(canary_version_id)
    if n < N_CANARY_MIN:
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="RETAIN",
            rationale=[f"only {n}/{N_CANARY_MIN} trades — wait for more samples"],
        )

    # 3. Stats compare — gates are only meaningful with closed-outcome
    #    evidence on BOTH sides, so missing or thin trade_outcome_features
    #    data must fail safe to RETAIN, never PROMOTE.  The expectancy gate
    #    already fails on an empty canary (lcb_mean_R is None), but the
    #    PF/MDD comparisons still pass vacuously against an empty CONTROL
    #    (x >= 0×0.95, y <= 0×1.1 only when y=0) — keep the explicit n>=20
    #    both-sides gate so thin evidence is named in the rationale instead
    #    of silently surviving whichever comparisons happen to be vacuous.
    #    (n_trades_for_canary above counts routed journal rows, which can hit
    #    20 while enrichment lags or breaks — it is not outcome evidence.)
    canary_stats = _stats_for_version(canary_version_id)
    control_id = parent_id  # most natural baseline is the parent
    control_stats = (
        _stats_for_version(int(control_id)) if control_id is not None
        else _aggregate([])
    )
    if canary_stats.n < N_CANARY_MIN or control_stats.n < N_CANARY_MIN:
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="RETAIN",
            rationale=[
                f"insufficient closed-outcome evidence: canary n={canary_stats.n}, "
                f"control n={control_stats.n} — need ≥ {N_CANARY_MIN} each "
                "before gates can run"
            ],
            canary_stats=canary_stats, control_stats=control_stats,
        )

    rationale: list[str] = []

    # PRIMARY gate (item 10): expectancy, not win rate.  LB95(mean R) must
    # clear max(0, control mean R) — credibly non-negative AND not credibly
    # worse than control.  A high-WR canary that bleeds R per trade can no
    # longer promote, and a 35%-WR convexity canary with fat winners no
    # longer needs a win-rate gate its spec never declared it would pass.
    control_floor = max(0.0, control_stats.mean_R)
    pass_expectancy = (
        canary_stats.lcb_mean_R is not None
        and canary_stats.lcb_mean_R >= control_floor
    )
    lb_txt = ("None" if canary_stats.lcb_mean_R is None
              else f"{canary_stats.lcb_mean_R:.4f}")
    rationale.append(
        f"LB95(mean R)={lb_txt} vs floor=max(0, control mean R="
        f"{control_stats.mean_R:.4f})={control_floor:.4f} → "
        f"{'pass' if pass_expectancy else 'fail'}"
    )

    # DIAGNOSTIC only — reported for the post-mortem narrative, never
    # gating: win rate is not expectancy (item-10 re-key).
    rationale.append(
        f"diagnostic (non-gating): win_rate={canary_stats.win_rate:.4f} "
        f"(wilson_lb={canary_stats.wilson_lb:.4f}) vs control win_rate="
        f"{control_stats.win_rate:.4f}"
    )

    pass_pf = canary_stats.profit_factor >= control_stats.profit_factor * PROFIT_FACTOR_FRAC
    rationale.append(
        f"profit_factor={canary_stats.profit_factor:.4f} vs "
        f"{control_stats.profit_factor * PROFIT_FACTOR_FRAC:.4f} "
        f"({PROFIT_FACTOR_FRAC:.2f}× control={control_stats.profit_factor:.4f}) → "
        f"{'pass' if pass_pf else 'fail'}"
    )

    pass_mdd = canary_stats.max_drawdown_R <= control_stats.max_drawdown_R * MDD_FRAC
    rationale.append(
        f"max_drawdown={canary_stats.max_drawdown_R:.4f} vs "
        f"{control_stats.max_drawdown_R * MDD_FRAC:.4f} "
        f"({MDD_FRAC:.2f}× control={control_stats.max_drawdown_R:.4f}) → "
        f"{'pass' if pass_mdd else 'fail'}"
    )

    if pass_expectancy and pass_pf and pass_mdd:
        return PromoteDecision(
            canary_version_id=canary_version_id, decision="PROMOTE",
            rationale=rationale,
            canary_stats=canary_stats, control_stats=control_stats,
        )
    return PromoteDecision(
        canary_version_id=canary_version_id, decision="RETAIN",
        rationale=rationale,
        canary_stats=canary_stats, control_stats=control_stats,
    )


# ---------------------------------------------------------------------------
# Apply outcomes
# ---------------------------------------------------------------------------

def apply_promotion(decision: PromoteDecision) -> bool:
    """For PROMOTE decisions: flip the canary to ACTIVE, retire the old
    ACTIVE in the same scope, and snapshot to params_history.

    Returns True on success.  Best-effort logged failure on errors.
    """
    if decision.decision != "PROMOTE":
        return False
    cv = decision.canary_version_id
    try:
        with cursor() as cur:
            # Lock the canary row first
            cur.execute(
                "SELECT scope, scope_key, params FROM param_versions WHERE id = %s FOR UPDATE",
                (cv,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            scope, scope_key, params = row

            # Retire any existing ACTIVE row in the same scope
            cur.execute(
                """
                UPDATE param_versions
                SET status = 'RETIRED', retired_at = NOW()
                WHERE status = 'ACTIVE' AND scope = %s
                  AND (scope_key IS NOT DISTINCT FROM %s)
                """,
                (scope, scope_key),
            )

            # Promote
            cur.execute(
                """
                UPDATE param_versions
                SET status = 'ACTIVE', promoted_at = NOW(),
                    traffic_pct = 100.0,
                    replay_metrics = %s::jsonb
                WHERE id = %s
                """,
                (
                    json.dumps(decision.to_dict(), default=str),
                    cv,
                ),
            )

            # params_history snapshot
            snapshot = params if isinstance(params, dict) else (
                json.loads(params or "{}") if isinstance(params, str) else {}
            )
            cur.execute(
                """
                INSERT INTO params_history (promoted_param_version_id, snapshot)
                VALUES (%s, %s::jsonb)
                """,
                (cv, json.dumps({
                    "params": snapshot,
                    "decision": decision.to_dict(),
                }, default=str)),
            )

            cur.execute(
                """
                INSERT INTO learning_events (event_type, param_version_id, payload)
                VALUES ('canary_promoted', %s, %s::jsonb)
                """,
                (cv, json.dumps(decision.to_dict(), default=str)),
            )
        return True
    except Exception as e:
        log.error("apply_promotion(%s) failed: %s", cv, e)
        return False


def apply_rollback(decision: PromoteDecision) -> bool:
    """For REJECT decisions: flip the canary to REJECTED, write the
    rationale into learning_events.canary_rolled_back."""
    if decision.decision != "REJECT":
        return False
    cv = decision.canary_version_id
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE param_versions
                SET status = 'REJECTED', retired_at = NOW(),
                    replay_metrics = %s::jsonb
                WHERE id = %s
                """,
                (
                    json.dumps(decision.to_dict(), default=str),
                    cv,
                ),
            )
            cur.execute(
                """
                INSERT INTO learning_events (event_type, param_version_id, payload)
                VALUES ('canary_rolled_back', %s, %s::jsonb)
                """,
                (cv, json.dumps(decision.to_dict(), default=str)),
            )
        return True
    except Exception as e:
        log.error("apply_rollback(%s) failed: %s", cv, e)
        return False


def evaluate_and_apply_all() -> list[dict[str, Any]]:
    """Run evaluate_canary on every CANARY row + apply the outcome.

    Designed to be called from the eod_review graph.  Returns a list of
    decision dicts so the caller can log to agent_events.
    """
    out: list[dict[str, Any]] = []
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT id FROM param_versions WHERE status = 'CANARY'",
            )
            ids = [int(r[0]) for r in cur.fetchall()]
    except Exception as e:
        log.warning("evaluate_and_apply_all: list failed: %s", e)
        return out

    for cv_id in ids:
        try:
            d = evaluate_canary(cv_id)
            if d.decision == "PROMOTE":
                apply_promotion(d)
            elif d.decision == "REJECT":
                apply_rollback(d)
            # Traffic ramp on RETAIN
            if d.decision == "RETAIN":
                from trading_agent.learning.canary import maybe_ramp_traffic
                maybe_ramp_traffic(cv_id)
            out.append(d.to_dict())
        except Exception as e:
            log.warning("evaluate_and_apply_all: id=%s failed: %s", cv_id, e)
    return out


__all__ = [
    "DecisionStr",
    "MDD_FRAC",
    "OutcomeStats",
    "PROFIT_FACTOR_FRAC",
    "PromoteDecision",
    "ROLLBACK_FIRST_N_TRADES",
    "ROLLBACK_LOSING_DAYS",
    "WILSON_Z_95",
    "_aggregate",
    "_stats_for_version",
    "apply_promotion",
    "apply_rollback",
    "consecutive_losing_days",
    "evaluate_and_apply_all",
    "evaluate_canary",
    "first_n_trades_breach",
    "wilson_lb",
]
