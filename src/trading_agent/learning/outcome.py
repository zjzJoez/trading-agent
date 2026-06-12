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

    Virtual fills (``broker_order_id`` starts with ``VIRTUAL-``) prefer
    ``broker_fill_json.{entry_price,exit_price,stop}`` over the columns,
    since the columns may carry placeholder zeros for shadow-only stock
    positions logged via ``record_virtual_stock_fill``.
    """
    fill_blob = trade_row.get("broker_fill_json") or {}
    if isinstance(fill_blob, str):
        try:
            fill_blob = json.loads(fill_blob)
        except (TypeError, ValueError):
            fill_blob = {}
    if not isinstance(fill_blob, dict):
        fill_blob = {}

    is_virtual = str(trade_row.get("broker_order_id") or "").startswith("VIRTUAL-")

    if is_virtual:
        entry = (_safe_float(fill_blob.get("entry_price"))
                 or _safe_float(trade_row.get("entry_price")))
        exit_ = (_safe_float(fill_blob.get("exit_price"))
                 or _safe_float(trade_row.get("exit_price")))
        stop = (_safe_float(fill_blob.get("stop"))
                or _safe_float(trade_row.get("stop")))
    else:
        entry = _safe_float(trade_row.get("entry_price"))
        exit_ = _safe_float(trade_row.get("exit_price"))
        stop = _safe_float(trade_row.get("stop"))
    opened_at = trade_row.get("opened_at")
    closed_at = trade_row.get("closed_at")

    realized_r: float | None = None
    if entry is not None and exit_ is not None and stop is not None:
        risk_per_unit = abs(entry - stop)
        if risk_per_unit > 1e-9:
            # Net of round-trip fees (in price units) when the caller supplies
            # asset_type — production enrichment always does. Friction-free
            # realized_r overstated every downstream gate (replay scoring,
            # canary promotion, the soak Sharpe gate) by the full cost stack.
            fee_unit = 0.0
            asset_type = trade_row.get("asset_type")
            if asset_type:
                try:
                    from trading_agent.execution_costs import (
                        round_trip_fee_per_unit,
                    )
                    fee_unit = round_trip_fee_per_unit(str(asset_type))
                except Exception as e:
                    log.warning("fee-per-unit lookup failed: %s", e)
            realized_r = (exit_ - entry - fee_unit) / risk_per_unit

    holding_days: float | None = None
    if isinstance(opened_at, datetime) and isinstance(closed_at, datetime):
        delta = closed_at - opened_at
        holding_days = round(delta.total_seconds() / 86400.0, 4)

    slippage_bps: float | None = None
    requested = _safe_float(fill_blob.get("requested_price"))
    filled = _safe_float(fill_blob.get("avg_fill_price"))
    if requested is not None and filled is not None and requested > 1e-9:
        slippage_bps = round((filled - requested) / requested * 10_000.0, 2)

    option_iv_change = _safe_float(fill_blob.get("iv_change"))

    # Phase 2.7.5 — MAE / MFE read from journal_trades persisted by
    # intraday update_excursions_node.  None when the trade closed before
    # any intraday refresh fired (e.g. immediate same-tick close).
    mae: float | None = None
    mfe: float | None = None
    tid = trade_row.get("id")
    if tid is not None:
        try:
            from trading_agent.learning.excursion import read_extremes
            mae, mfe = read_extremes(int(tid))
        except Exception as e:
            log.warning("read_extremes(%s) failed: %s", tid, e)

    return OutcomeMetrics(
        realized_r=realized_r,
        holding_days=holding_days,
        slippage_bps=slippage_bps,
        option_iv_change=option_iv_change,
        mae=mae,
        mfe=mfe,
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

    Phase 2.6.5: also updates the diversity archive
    (``param_archive_cells``) with the realized R, electing a per-cell
    champion when the candidate beats the current one.
    """
    enriched: list[int] = []
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT jt.id, jt.entry_price, jt.exit_price, jt.stop,
                       jt.opened_at, jt.closed_at, jt.broker_fill_json,
                       jt.entry_regime_state_id, jt.exit_regime_state_id,
                       jt.params_version_id, jt.risk_decision_id, jt.outcome,
                       jt.broker_fill_json->>'strategy_label' AS strategy_label,
                       rs.label AS entry_regime_label, jt.asset_type
                FROM journal_trades jt
                LEFT JOIN trade_outcome_features tof ON tof.trade_id = jt.id
                LEFT JOIN regime_states rs ON rs.id = jt.entry_regime_state_id
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
         entry_rs, exit_rs, pvid, rdid, outcome,
         strategy_label, entry_regime_label, asset_type) = r
        trade_row = {
            "id": tid,
            "entry_price": entry,
            "exit_price": exit_,
            "stop": stop,
            "opened_at": opened_at,
            "closed_at": closed_at,
            "broker_fill_json": fill,
            "outcome": outcome,
            "asset_type": asset_type,
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
            # Phase 2.6.5 — diversity archive update (best-effort; isolated
            # from the outcome write so a failure here can't lose the audit row)
            if metrics.realized_r is not None and pvid is not None:
                _try_archive_update(
                    closed_at=closed_at,
                    realized_R=metrics.realized_r,
                    param_version_id=int(pvid),
                    fill_blob=fill,
                    strategy_label=strategy_label,
                    entry_regime_label=entry_regime_label,
                )

    if enriched:
        emit(
            run_id=f"outcome_enrich_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
            trigger="eod_review",
            agent="learning_outcome",
            event_type="outcomes_enriched",
            payload={"trade_ids": enriched, "n": len(enriched)},
        )
    return enriched


def _try_archive_update(
    *,
    closed_at,
    realized_R: float,
    param_version_id: int,
    fill_blob,
    strategy_label: str | None,
    entry_regime_label: str | None,
) -> None:
    """Best-effort hook from outcome enrichment into the diversity archive.

    Pulls option_dte / option_iv from the broker_fill_json (or proposal
    field stashed therein) and computes the (regime, strategy, dte_bucket,
    iv_bucket) cell, then calls archive.update_champion. Never raises.
    """
    try:
        from trading_agent.learning.archive import (
            cell_for_trade, update_champion,
        )
    except Exception as e:
        log.warning("archive import failed: %s", e)
        return

    fb = fill_blob if isinstance(fill_blob, dict) else (
        json.loads(fill_blob) if isinstance(fill_blob, str) and fill_blob else {}
    )
    option_dte = _safe_float(fb.get("option_dte")) if isinstance(fb, dict) else None
    option_iv = _safe_float(fb.get("option_iv")) if isinstance(fb, dict) else None
    cell = cell_for_trade(
        regime_label=entry_regime_label,
        strategy_label=strategy_label,
        option_dte=option_dte,
        option_iv=option_iv,
    )
    res = update_champion(
        cell, param_version_id, realized_R, trade_at=closed_at,
    )
    log.info(
        "[archive] cell=%s pv=%s R=%.4f n=%s score=%s champion_changed=%s",
        cell.to_text(), param_version_id, realized_R,
        res.get("n_trades"), res.get("score_after"),
        res.get("champion_changed"),
    )


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
