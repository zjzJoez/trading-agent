"""Phase 2.6 — Outcome enrichment writer.

Closes the audit loop: when a paper trade exits (CLOSED in journal_trades),
compute the realized statistics needed for shadow scoring and weekly
post-mortem, and write them to ``trade_outcome_features``.

Phase 2.6 scope: deterministic post-close metrics from journal_trades + the
broker fill blob.  MAE/MFE intentionally degraded to ``None`` here — they
require the intraday price path which we don't yet persist; Phase 2.6.5 adds
mid-day quote snapshots, at which point this module fills them in.

The module is also responsible for stitching together the audit chain:

    journal_trades
        → entry_regime_state_id, exit_regime_state_id
        → risk_decisions.risk_snapshot_id (via journal_trades.risk_decision_id)
        → params_version_id (live ACTIVE at decision time)

so weekly_learning can join on ``trade_outcome_features.param_version_id`` and
get a clean strategy × regime × params triple per closed trade.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


@dataclass
class OutcomeMetrics:
    """Deterministic metrics computed from journal_trades + fill JSON."""

    realized_r: float | None
    holding_days: float | None
    slippage_bps: float | None
    option_iv_change: float | None
    mae: float | None
    mfe: float | None


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_outcome(trade_row: dict[str, Any]) -> OutcomeMetrics:
    """Derive realized stats from one journal_trades row.

    Inputs needed: ``entry_price``, ``exit_price``, ``stop``, ``opened_at``,
    ``closed_at``, ``broker_fill_json``.  Anything missing → metric is ``None``.
    """
    entry = _safe_float(trade_row.get("entry_price"))
    exit_ = _safe_float(trade_row.get("exit_price"))
    stop = _safe_float(trade_row.get("stop"))
    opened_at = trade_row.get("opened_at")
    closed_at = trade_row.get("closed_at")

    realized_r: float | None = None
    if entry is not None and exit_ is not None and stop is not None:
        risk_per_unit = abs(entry - stop)
        if risk_per_unit > 1e-9:
            realized_r = (exit_ - entry) / risk_per_unit

    holding_days: float | None = None
    if isinstance(opened_at, datetime) and isinstance(closed_at, datetime):
        delta = closed_at - opened_at
        holding_days = round(delta.total_seconds() / 86400.0, 4)

    slippage_bps: float | None = None
    fill_blob = trade_row.get("broker_fill_json") or {}
    if isinstance(fill_blob, str):
        try:
            fill_blob = json.loads(fill_blob)
        except (TypeError, ValueError):
            fill_blob = {}
    requested = _safe_float(fill_blob.get("requested_price"))
    filled = _safe_float(fill_blob.get("avg_fill_price"))
    if requested is not None and filled is not None and requested > 1e-9:
        slippage_bps = round((filled - requested) / requested * 10_000.0, 2)

    option_iv_change = _safe_float(fill_blob.get("iv_change"))

    return OutcomeMetrics(
        realized_r=realized_r,
        holding_days=holding_days,
        slippage_bps=slippage_bps,
        option_iv_change=option_iv_change,
        mae=None,  # Phase 2.6.5 — needs intraday quote path
        mfe=None,
    )


def write_outcome_features(
    trade_id: int,
    metrics: OutcomeMetrics,
    *,
    entry_regime_state_id: int | None = None,
    exit_regime_state_id: int | None = None,
    risk_snapshot_id: int | None = None,
    param_version_id: int | None = None,
    feature: dict[str, Any] | None = None,
    label: dict[str, Any] | None = None,
) -> int | None:
    """Insert one trade_outcome_features row.  Returns row id or None on error."""
    feature = feature or {}
    if metrics.realized_r is not None:
        feature.setdefault("realized_r", metrics.realized_r)
    if metrics.holding_days is not None:
        feature.setdefault("holding_days", metrics.holding_days)
    if metrics.slippage_bps is not None:
        feature.setdefault("slippage_bps", metrics.slippage_bps)
    if metrics.option_iv_change is not None:
        feature.setdefault("option_iv_change", metrics.option_iv_change)

    label = label or {}
    if metrics.realized_r is not None:
        label.setdefault("win", metrics.realized_r > 0)
        label.setdefault("R_bucket", _bucket_realized_r(metrics.realized_r))

    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO trade_outcome_features
                    (trade_id, entry_regime_state_id, exit_regime_state_id,
                     risk_snapshot_id, param_version_id,
                     mae, mfe, realized_r, holding_days, slippage_bps,
                     option_iv_change, feature, label)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (
                    trade_id,
                    entry_regime_state_id,
                    exit_regime_state_id,
                    risk_snapshot_id,
                    param_version_id,
                    metrics.mae,
                    metrics.mfe,
                    metrics.realized_r,
                    metrics.holding_days,
                    metrics.slippage_bps,
                    metrics.option_iv_change,
                    json.dumps(feature, default=str),
                    json.dumps(label, default=str),
                ),
            )
            return int(cur.fetchone()[0])
    except Exception as e:
        log.warning("write_outcome_features failed for trade_id=%s: %s", trade_id, e)
        return None


def _bucket_realized_r(r: float) -> str:
    if r >= 2.0:
        return "win_2R+"
    if r >= 1.0:
        return "win_1R"
    if r >= 0:
        return "win_partial"
    if r >= -0.5:
        return "small_loss"
    if r >= -1.0:
        return "full_R_loss"
    return "exceeded_R_loss"


def enrich_closed_trades(limit: int = 50) -> list[int]:
    """Find closed journal_trades rows that lack a trade_outcome_features row,
    compute their metrics, and persist.

    Returns a list of journal_trades.id values that were enriched.

    Designed to be called by ``eod_review_graph`` (already wired) and by
    ``weekly_learning_graph`` (Phase 2.7).
    """
    enriched: list[int] = []
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT jt.id, jt.entry_price, jt.exit_price, jt.stop,
                       jt.opened_at, jt.closed_at, jt.broker_fill_json,
                       jt.entry_regime_state_id, jt.exit_regime_state_id,
                       jt.params_version_id, jt.risk_decision_id, jt.outcome
                FROM journal_trades jt
                LEFT JOIN trade_outcome_features tof ON tof.trade_id = jt.id
                WHERE jt.outcome IN ('CLOSED', 'STOPPED', 'TARGET_HIT')
                  AND tof.id IS NULL
                ORDER BY jt.closed_at DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("enrich_closed_trades: read failed (%s)", e)
        return enriched

    for r in rows:
        (tid, entry, exit_, stop, opened_at, closed_at, fill,
         entry_rs, exit_rs, pvid, rdid, outcome) = r
        trade_row = {
            "id": tid,
            "entry_price": entry,
            "exit_price": exit_,
            "stop": stop,
            "opened_at": opened_at,
            "closed_at": closed_at,
            "broker_fill_json": fill,
            "outcome": outcome,
        }
        metrics = compute_outcome(trade_row)
        risk_snapshot_id = _lookup_risk_snapshot_id(rdid)
        new_id = write_outcome_features(
            int(tid),
            metrics,
            entry_regime_state_id=int(entry_rs) if entry_rs is not None else None,
            exit_regime_state_id=int(exit_rs) if exit_rs is not None else None,
            risk_snapshot_id=risk_snapshot_id,
            param_version_id=int(pvid) if pvid is not None else None,
        )
        if new_id is not None:
            enriched.append(int(tid))

    if enriched:
        emit(
            run_id=f"outcome_enrich_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
            trigger="eod_review",
            agent="learning_outcome",
            event_type="outcomes_enriched",
            payload={"trade_ids": enriched, "n": len(enriched)},
        )
    return enriched


def _lookup_risk_snapshot_id(risk_decision_id: str | None) -> int | None:
    """Resolve risk_decisions.risk_decision_id (text) → risk_snapshot_id."""
    if not risk_decision_id:
        return None
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT risk_snapshot_id FROM risk_decisions "
                "WHERE risk_decision_id = %s LIMIT 1",
                (risk_decision_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
    except Exception as e:
        log.warning("_lookup_risk_snapshot_id failed (%s)", e)
    return None


__all__ = [
    "OutcomeMetrics",
    "compute_outcome",
    "write_outcome_features",
    "enrich_closed_trades",
]
