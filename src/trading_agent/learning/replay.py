"""Phase 2.6 — Historical replay of decisions under candidate parameters.

Drives the weekly_learning_graph: given a candidate ``param_versions`` row,
re-derive what each closed paper trade would have looked like and produce a
composite score so the loop can promote / reject the candidate.

Composite score (QuantBook L17, QuantEvolve §3):

    score = sharpe + IR − λ · max_drawdown          (λ = 0.5)

where ``sharpe`` is the per-trade Sharpe of realized R, ``IR`` is the
information ratio against the production baseline R series, and
``max_drawdown`` is the largest peak-to-trough drawdown of the cumulative R
curve.

Replay scope (what's deterministic):

    * sizing_aggression   — re-derive qty under shadow caps; scale realized R
      linearly by qty_shadow / qty_actual.  Linear-scaling assumption is
      documented; it ignores second-order effects (slippage on larger fills).
    * regime_size_multipliers — same scaling, but mediated through the
      regime label active at entry.
    * regime_thresholds   — re-run crisis_overlay on the persisted feature
      snapshot.  If a candidate flips the entry day to CRISIS, the trade is
      replayed as ``skipped`` (R = 0 contributes nothing to numerator/denom).

Stop_distances and entry_filters are out of scope for Phase 2.6 replay; they
need an intraday price path or a re-run of the LLM trader respectively.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_agent.learning.params import (
    PARAM_BOUNDS,
    ParamResolver,
    defaults_resolver,
    load_active_params,
)
from trading_agent.learning.shadow import (
    _regime_size_mult_key,
    replay_crisis_overlay,
    replay_sizing_caps,
)
from trading_agent.regime.features import FeatureSnapshot
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

LAMBDA_MDD = 0.5


# ---------------------------------------------------------------------------
# Trade fetch
# ---------------------------------------------------------------------------

@dataclass
class ClosedTradeRow:
    """One closed paper trade plus its full audit chain (read-only)."""

    trade_id: int
    ticker: str
    strategy_label: str | None
    entry_price: float
    exit_price: float
    stop: float | None
    qty: int
    contract_multiplier: int
    opened_at: datetime
    closed_at: datetime
    params_version_id: int | None
    entry_regime_label: str | None
    entry_feature_snapshot: FeatureSnapshot | None
    realized_r: float | None
    equity_at_entry: float


def _row_to_feature_snapshot(features: dict[str, Any] | None,
                              as_of: datetime) -> FeatureSnapshot | None:
    if not features:
        return None
    fs = features if isinstance(features, dict) else json.loads(features)
    return FeatureSnapshot(
        as_of=as_of,
        features={k: float(v) for k, v in (fs.get("features") or fs).items()
                  if isinstance(v, (int, float))},
    )


def fetch_closed_trades(
    since: datetime | None = None,
    limit: int = 200,
) -> list[ClosedTradeRow]:
    """Pull closed trades joined with regime + outcome rows."""
    since = since or (datetime.now(timezone.utc) - timedelta(days=180))
    rows: list[ClosedTradeRow] = []
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT
                  jt.id, jt.symbol, jt.entry_price, jt.exit_price, jt.stop,
                  jt.qty, jt.opened_at, jt.closed_at, jt.params_version_id,
                  rs.label, rfs.features, jt.broker_fill_json,
                  tof.realized_r, jth.strategy_label
                FROM journal_trades jt
                LEFT JOIN regime_states rs ON rs.id = jt.entry_regime_state_id
                LEFT JOIN regime_feature_snapshots rfs
                       ON rfs.id = rs.feature_snapshot_id
                LEFT JOIN trade_outcome_features tof ON tof.trade_id = jt.id
                LEFT JOIN journal_theses jth ON jth.id = jt.thesis_id
                WHERE jt.outcome IN ('CLOSED', 'STOPPED', 'TARGET_HIT')
                  AND jt.closed_at >= %s
                ORDER BY jt.closed_at ASC
                LIMIT %s
                """,
                (since, limit),
            )
            raw = cur.fetchall()
    except Exception as e:
        log.warning("fetch_closed_trades failed: %s", e)
        return rows

    for r in raw:
        (tid, sym, entry, exit_, stop, qty, opened_at, closed_at, pvid,
         regime_label, features_json, fill_blob,
         realized_r, strategy) = r
        # Asset type: assume OPT for option symbols (US.XYZnnnnCnnnnnnnn pattern)
        contract_mult = 100 if "C" in sym or "P" in sym else 1
        equity = 100_000.0  # default; if fill_blob has equity_at_entry use it
        fb = fill_blob if isinstance(fill_blob, dict) else (
            json.loads(fill_blob) if isinstance(fill_blob, str) and fill_blob else {})
        if isinstance(fb, dict) and fb.get("equity_at_entry"):
            try:
                equity = float(fb["equity_at_entry"])
            except (TypeError, ValueError):
                pass

        rows.append(ClosedTradeRow(
            trade_id=int(tid),
            ticker=str(sym),
            strategy_label=strategy,
            entry_price=float(entry),
            exit_price=float(exit_) if exit_ is not None else float(entry),
            stop=float(stop) if stop is not None else None,
            qty=int(qty),
            contract_multiplier=contract_mult,
            opened_at=opened_at,
            closed_at=closed_at,
            params_version_id=int(pvid) if pvid is not None else None,
            entry_regime_label=regime_label,
            entry_feature_snapshot=_row_to_feature_snapshot(
                features_json, opened_at
            ),
            realized_r=float(realized_r) if realized_r is not None else None,
            equity_at_entry=equity,
        ))
    return rows


# ---------------------------------------------------------------------------
# Counterfactual P&L per trade
# ---------------------------------------------------------------------------

@dataclass
class TradeReplay:
    trade_id: int
    qty_actual: int
    qty_shadow: int
    realized_r_actual: float
    realized_r_shadow: float
    skipped_by_regime: bool


def replay_one(
    trade: ClosedTradeRow,
    candidate: ParamResolver,
    *,
    baseline: ParamResolver | None = None,
) -> TradeReplay:
    """Replay one closed trade under the candidate resolver.

    Returns the (actual, shadow) pair so the caller can aggregate stats.
    """
    baseline = baseline or defaults_resolver()

    # 1. crisis-threshold flip → trade would have been skipped
    skipped = False
    if trade.entry_feature_snapshot is not None:
        cf = replay_crisis_overlay(trade.entry_feature_snapshot, baseline, candidate)
        # Only the candidate's view matters for "would we have entered?".
        # If the candidate sees CRISIS at entry, the regime gate would have
        # blocked the trade (size_mult_crisis is hard zero).
        if cf.is_crisis_shadow and not cf.is_crisis_actual:
            skipped = True

    # 2. sizing-cap counterfactual
    max_loss_per_contract = (
        abs(trade.entry_price - trade.stop) * trade.contract_multiplier
        if trade.stop is not None else
        trade.entry_price * trade.contract_multiplier * 0.05
    )
    sizing = replay_sizing_caps(
        equity=trade.equity_at_entry,
        entry_price=trade.entry_price,
        contract_multiplier=trade.contract_multiplier,
        qty_actual=trade.qty,
        max_loss_per_contract=max_loss_per_contract,
        regime_label=trade.entry_regime_label,
        control=baseline,
        shadow=candidate,
    )

    realized_r_actual = trade.realized_r if trade.realized_r is not None else 0.0
    if skipped or sizing.qty_shadow == 0:
        realized_r_shadow = 0.0
    else:
        # Linear P&L scaling; ignores slippage non-linearity
        ratio = sizing.qty_shadow / max(1, sizing.qty_actual)
        realized_r_shadow = realized_r_actual * ratio

    return TradeReplay(
        trade_id=trade.trade_id,
        qty_actual=sizing.qty_actual,
        qty_shadow=sizing.qty_shadow,
        realized_r_actual=realized_r_actual,
        realized_r_shadow=realized_r_shadow,
        skipped_by_regime=skipped,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class ReplayMetrics:
    n_trades: int
    win_rate: float
    mean_R: float
    profit_factor: float
    sharpe: float
    max_drawdown_R: float
    cum_R: float
    composite_score: float
    sample: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "mean_R": round(self.mean_R, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown_R": round(self.max_drawdown_R, 4),
            "cum_R": round(self.cum_R, 4),
            "composite_score": round(self.composite_score, 4),
        }


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 1e-12 else default


def _aggregate(rs: list[float], baseline_rs: list[float] | None = None) -> ReplayMetrics:
    n = len(rs)
    if n == 0:
        return ReplayMetrics(0, 0, 0, 0, 0, 0, 0, 0, [])

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    win_rate = len(wins) / n
    mean = sum(rs) / n
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = _safe_div(gross_win, gross_loss, default=float(gross_win) if gross_win > 0 else 0.0)

    var = sum((r - mean) ** 2 for r in rs) / n if n > 1 else 0.0
    sd = math.sqrt(var)
    sharpe = _safe_div(mean, sd) * math.sqrt(252.0 / max(1, n))

    from trading_agent.learning._stats import series_max_drawdown
    cum, mdd = series_max_drawdown(rs)

    # Information ratio vs baseline (if provided)
    if baseline_rs and len(baseline_rs) == n:
        diffs = [a - b for a, b in zip(rs, baseline_rs)]
        diff_mean = sum(diffs) / n
        diff_var = sum((d - diff_mean) ** 2 for d in diffs) / n if n > 1 else 0.0
        diff_sd = math.sqrt(diff_var)
        ir = _safe_div(diff_mean, diff_sd) * math.sqrt(252.0 / max(1, n))
    else:
        ir = 0.0

    composite = sharpe + ir - LAMBDA_MDD * mdd
    return ReplayMetrics(
        n_trades=n,
        win_rate=win_rate,
        mean_R=mean,
        profit_factor=pf,
        sharpe=sharpe,
        max_drawdown_R=mdd,
        cum_R=cum,
        composite_score=composite,
        sample=list(rs),
    )


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

@dataclass
class ReplayReport:
    candidate_version_id: int | None
    baseline_version_id: int | None
    n_trades_evaluated: int
    n_trades_skipped_by_candidate: int
    actual: ReplayMetrics
    shadow: ReplayMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_version_id": self.candidate_version_id,
            "baseline_version_id": self.baseline_version_id,
            "n_trades_evaluated": self.n_trades_evaluated,
            "n_trades_skipped_by_candidate": self.n_trades_skipped_by_candidate,
            "actual": self.actual.to_dict(),
            "shadow": self.shadow.to_dict(),
            "composite_delta": round(
                self.shadow.composite_score - self.actual.composite_score, 4
            ),
        }


def replay_param_version(
    candidate: ParamResolver,
    *,
    since: datetime | None = None,
    limit: int = 200,
    baseline: ParamResolver | None = None,
) -> ReplayReport:
    """Replay a candidate over closed trades and return paired metrics.

    Phase 2.6 ergonomics: caller passes a ``ParamResolver``; the function does
    not insert a new ``param_versions`` row.  Use ``persist_replay_metrics``
    to attach the report to an existing version.
    """
    baseline = baseline or load_active_params()
    trades = fetch_closed_trades(since=since, limit=limit)

    actual_R: list[float] = []
    shadow_R: list[float] = []
    n_skipped = 0
    for t in trades:
        rep = replay_one(t, candidate, baseline=baseline)
        actual_R.append(rep.realized_r_actual)
        shadow_R.append(rep.realized_r_shadow)
        if rep.skipped_by_regime:
            n_skipped += 1

    return ReplayReport(
        candidate_version_id=candidate.version_id,
        baseline_version_id=baseline.version_id,
        n_trades_evaluated=len(trades),
        n_trades_skipped_by_candidate=n_skipped,
        actual=_aggregate(actual_R),
        shadow=_aggregate(shadow_R, baseline_rs=actual_R),
    )


def persist_replay_metrics(version_id: int, metrics: dict[str, Any]) -> None:
    """Write the replay report into ``param_versions.replay_metrics``."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                UPDATE param_versions
                SET replay_metrics = %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(metrics, default=str), version_id),
            )
    except Exception as e:
        log.warning("persist_replay_metrics: write failed (%s)", e)


__all__ = [
    "ClosedTradeRow",
    "TradeReplay",
    "ReplayMetrics",
    "ReplayReport",
    "fetch_closed_trades",
    "replay_one",
    "replay_param_version",
    "persist_replay_metrics",
]


# Silence unused import lint — kept for downstream callers.
_ = (PARAM_BOUNDS, _regime_size_mult_key)
