"""Persist regime feature snapshots + classifications to Postgres.

Tables (from migrations/002_phase2.sql):
    regime_model_versions
    regime_feature_snapshots
    regime_states
    regime_llm_reviews

Each call returns the new row id so the caller can store FK references.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from trading_agent.regime.classifier import HMMModel, RegimePrediction
from trading_agent.regime.features import FeatureSnapshot
from trading_agent.regime.llm_review import LLMReview
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def insert_feature_snapshot(snap: FeatureSnapshot) -> int:
    """Persist a feature vector. Returns the new row id."""
    payload = snap.to_db_payload()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO regime_feature_snapshots
                (as_of, source_freshness, features, data_quality)
            VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (
                snap.as_of,
                json.dumps(payload["source_freshness"]),
                json.dumps(payload["features"]),
                json.dumps(payload["data_quality"]),
            ),
        )
        return int(cur.fetchone()[0])


def insert_regime_state(
    pred: RegimePrediction,
    *,
    as_of: datetime,
    feature_snapshot_id: int | None,
    model_version_id: int | None,
    prior_label: str | None = None,
    gate: dict[str, Any] | None = None,
) -> int:
    payload = pred.to_db_payload()
    payload["gate"] = gate or {}
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO regime_states
                (as_of, model_version_id, feature_snapshot_id,
                 label, prior_label, pending_label, confidence,
                 probabilities, degradation_level, crisis_flags, gate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (
                as_of,
                model_version_id,
                feature_snapshot_id,
                pred.label,
                prior_label,
                pred.pending_label,
                float(pred.confidence),
                json.dumps(payload["probabilities"]),
                int(pred.degradation_level),
                json.dumps(pred.crisis_flags),
                json.dumps(payload["gate"]),
            ),
        )
        return int(cur.fetchone()[0])


def insert_model_version(
    model: HMMModel,
    *,
    train_start: str | None = None,
    train_end: str | None = None,
    metrics: dict[str, Any] | None = None,
    status: str = "active",
) -> int:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO regime_model_versions
                (model_type, feature_set_version, train_start, train_end,
                 params_json, metrics_json, status)
            VALUES ('hmm_gaussian_v1', %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            RETURNING id
            """,
            (
                "v1.0",
                train_start,
                train_end,
                json.dumps(model.to_json()),
                json.dumps(metrics or {}),
                status,
            ),
        )
        return int(cur.fetchone()[0])


def insert_llm_review(
    *,
    regime_state_id: int,
    trigger_reason: str,
    prompt_hash: str,
    review: LLMReview,
    applied_action: str,
) -> int:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO regime_llm_reviews
                (regime_state_id, trigger_reason, prompt_hash,
                 response, applied_action, cost_usd)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (
                regime_state_id,
                trigger_reason,
                prompt_hash,
                json.dumps({
                    "review_label": review.review_label,
                    "confidence_adjustment": review.confidence_adjustment,
                    "risk_notes": review.risk_notes,
                    "must_defer_new_entries": review.must_defer_new_entries,
                }),
                applied_action,
                review.cost_usd,
            ),
        )
        return int(cur.fetchone()[0])


def get_active_model_id() -> int | None:
    """Return the id of the most recent 'active' regime model, or None."""
    with cursor() as cur:
        cur.execute(
            "SELECT id FROM regime_model_versions WHERE status='active' "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def get_active_model() -> HMMModel | None:
    """Load the active HMM model from Postgres. Returns None if none trained yet."""
    with cursor() as cur:
        cur.execute(
            "SELECT params_json FROM regime_model_versions "
            "WHERE status='active' ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return HMMModel.from_json(row[0])


def get_model_by_id(model_id: int) -> tuple[HMMModel, dict] | None:
    """Load a specific (possibly shadow/retired) HMM by id.

    Used by `walkforward_backtest.py` to load the OOS evaluation model
    without disturbing the active-model lookup path that production uses.
    Returns (HMMModel, metrics_json_dict). None if id doesn't exist.
    """
    with cursor() as cur:
        cur.execute(
            "SELECT params_json, metrics_json, status, train_start, train_end "
            "FROM regime_model_versions WHERE id=%s",
            (int(model_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        params_json, metrics_json, status, train_start, train_end = row
        info = {
            "id": int(model_id),
            "status": status,
            "train_start": str(train_start) if train_start else None,
            "train_end": str(train_end) if train_end else None,
            "metrics": metrics_json or {},
        }
        return HMMModel.from_json(params_json), info


def get_latest_regime_state() -> dict | None:
    """Return the most recent regime_states row as a dict (for graph nodes)."""
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, as_of, label, confidence, pending_label,
                   degradation_level, crisis_flags, probabilities, gate
            FROM regime_states
            ORDER BY as_of DESC, id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "state_id": int(row[0]),
            "as_of": row[1].isoformat() if row[1] else None,
            "label": row[2],
            "confidence": float(row[3]),
            "pending_label": row[4],
            "degradation_level": int(row[5]),
            "crisis_flags": row[6] or [],
            "probabilities": row[7] or {},
            "gate": row[8] or {},
        }
