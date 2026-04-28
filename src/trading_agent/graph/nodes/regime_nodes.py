"""Real regime nodes for the LangGraph orchestrator (replaces stubs §6).

Wires together:
    features.build_feature_snapshot
    classifier.classify (Layers 0+1)
    llm_review.stub_llm_review (Phase 2.4 swaps for real LLM call)
    persist.* (Postgres)
    notify.send (ntfy)

Each node respects the graph's state shape — it reads what it needs from
state and writes back a partial state update.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.notify import send as ntfy_send
from trading_agent.regime.classifier import classify
from trading_agent.regime.features import (
    ALL_TICKERS,
    build_feature_snapshot,
)
from trading_agent.regime.gates import regime_size_multiplier
from trading_agent.regime.llm_review import (
    apply_review_to_prediction,
    real_llm_review,
    stub_llm_review,
)
from trading_agent.regime.persist import (
    get_active_model,
    get_latest_regime_state,
    insert_feature_snapshot,
    insert_llm_review,
    insert_regime_state,
)

log = logging.getLogger(__name__)


def collect_macro_market_data(state: dict) -> dict:
    """Pull klines for all regime-relevant tickers via Phase-1 moomoo MCP module.

    Phase 2.2 leaves this minimal — Phase 2.5 wires the real moomoo fetch.
    For now we set a placeholder dict so downstream nodes can run with no
    klines (they'll go to fallback / safe-mode paths in tests).
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[regime/collect_macro_market_data] run_id=%s", run_id)

    # Real fetch via moomoo-api → OpenD :11111. Falls back to cache if OpenD
    # is down. Returns klines as plain dicts for state serialization (the
    # actual DataFrames are reconstructed in compute_regime_features).
    from trading_agent.regime.data_fetch import fetch_for_features

    klines: dict[str, Any] = {}
    fetched = 0
    try:
        df_dict = fetch_for_features()
        for tkr, df in df_dict.items():
            # Stash as records — DataFrame doesn't survive LangGraph's state copy
            klines[tkr] = {
                "columns": list(df.columns),
                "index": [str(i) for i in df.index],
                "data": df.values.tolist(),
            }
            fetched += 1
    except Exception as e:
        log.error("[regime/collect_macro_market_data] fetch failed: %s", e)

    market_data = {
        "tickers_requested": list(ALL_TICKERS),
        "klines_fetched": fetched,
        "klines": klines,
        "fetch_status": "ok" if fetched > 0 else "failed",
    }

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="collect_macro_market_data",
        event_type="data_fetched",
        payload={
            "klines_fetched": fetched,
            "tickers": len(ALL_TICKERS),
            "fetch_status": market_data["fetch_status"],
        },
    )
    return {"market_data": market_data}


def compute_regime_features(state: dict) -> dict:
    """Build the standardized feature vector. Stores in state for the next node."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[regime/compute_regime_features] run_id=%s", run_id)

    # Reconstruct DataFrames from the records-format klines stashed by the
    # previous node. (LangGraph state can't carry DataFrames natively.)
    import pandas as pd

    raw = state.get("market_data", {}).get("klines", {}) or {}
    klines: dict[str, pd.DataFrame] = {}
    for tkr, payload in raw.items():
        if isinstance(payload, pd.DataFrame):
            klines[tkr] = payload
            continue
        try:
            df = pd.DataFrame(
                payload["data"],
                columns=payload["columns"],
                index=pd.to_datetime(payload["index"]).date,
            )
            klines[tkr] = df
        except Exception as e:
            log.warning("Failed to reconstruct klines for %s: %s", tkr, e)

    snap = build_feature_snapshot(klines=klines)

    feature_payload = snap.to_db_payload()
    emit(
        run_id=run_id,
        trigger=trigger,
        agent="compute_regime_features",
        event_type="features_computed",
        payload={
            "degradation_level": snap.degradation_level,
            "missing": snap.data_quality.get("missing", []),
            "n_features": len(snap.features),
        },
    )
    return {
        # Keep feature snapshot inside the `regime` slot so LangGraph's state
        # schema (TradingGraphState) doesn't filter it out.
        "regime": {
            **state.get("regime", {}),
            "_features_snapshot": {
                "as_of": snap.as_of.isoformat(),
                **feature_payload,
                "_obj_features": snap.features,
                "_obj_data_quality": snap.data_quality,
            },
        }
    }


def classify_regime(state: dict) -> dict:
    """Run Layer 0 (crisis overlay) + Layer 1 (HMM) → RegimePrediction."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[regime/classify_regime] run_id=%s", run_id)

    fs = state.get("regime", {}).get("_features_snapshot")
    if not fs:
        log.warning("classify_regime called with no feature snapshot in state")
        return {}

    # Reconstruct a minimal FeatureSnapshot for the classifier
    from trading_agent.regime.features import FeatureSnapshot

    snap = FeatureSnapshot(
        as_of=datetime.fromisoformat(fs["as_of"]),
    )
    snap.features = fs["_obj_features"]
    snap.data_quality = fs["_obj_data_quality"]

    # Pull active HMM model (None if not yet trained — falls back to rules)
    model = get_active_model()
    prior_state = get_latest_regime_state()
    prior_label = prior_state["label"] if prior_state else None

    pred = classify(snap, model=model, prior_label=prior_label)

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="classify_regime",
        event_type="regime_classified",
        payload={
            "label": pred.label,
            "confidence": pred.confidence,
            "needs_llm_review": pred.needs_llm_review,
            "review_reason": pred.review_reason,
            "crisis_flags": pred.crisis_flags,
            "degradation_level": pred.degradation_level,
            "had_active_model": model is not None,
        },
    )

    return {
        "regime": {
            **state.get("regime", {}),  # preserve _features_snapshot
            "label": pred.label,
            "confidence": pred.confidence,
            "pending_label": pred.pending_label,
            "probabilities": {k: float(v) for k, v in pred.probabilities.items()},
            "crisis_flags": pred.crisis_flags,
            "degradation_level": pred.degradation_level,
            "needs_llm_review": pred.needs_llm_review,
            "review_reason": pred.review_reason,
            "_pred": pred.__dict__,
        }
    }


def maybe_llm_regime_review(state: dict) -> dict:
    """Run LLM second-opinion ONLY when classify_regime says it's needed.

    Phase 2.2 returns the stub LLM result (CONFIRM, no change).
    Phase 2.4 wires `OAuthLLMRouter.call("regime_reviewer", ...)`.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    regime = state.get("regime", {})
    if not regime.get("needs_llm_review"):
        log.info("[regime/maybe_llm_regime_review] skipped — confidence ok")
        emit(
            run_id=run_id,
            trigger=trigger,
            agent="maybe_llm_regime_review",
            event_type="skipped",
            payload={"reason": "confidence_above_threshold"},
        )
        return {}

    # Reconstruct a RegimePrediction for apply_review_to_prediction
    from trading_agent.regime.classifier import RegimePrediction

    pred = RegimePrediction(**regime["_pred"])
    fs = regime.get("_features_snapshot") or {}
    raw_feats = {
        k: v for k, v in (fs.get("features") or {}).items()
        if isinstance(v, (int, float))
    }
    # Pick top 20 most "interesting" features by absolute value, but EXCLUDE
    # bounded 0/1 indicator flags from the ranking so they aren't drowned out.
    notable = [(k, v) for k, v in raw_feats.items() if not k.endswith(("_above_50dma", "_above_200dma"))]
    notable.sort(key=lambda kv: -abs(kv[1]))
    top = dict(notable[:20])
    # Always include the regime indicator flags (they're cheap signal)
    for k, v in raw_feats.items():
        if k.endswith(("_above_50dma", "_above_200dma")) and len(top) < 25:
            top[k] = v

    review = real_llm_review(pred, top_features=top or None)
    new_pred = apply_review_to_prediction(pred, review)

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="maybe_llm_regime_review",
        event_type="llm_review_applied",
        payload={
            "review_label": review.review_label,
            "confidence_adjustment": review.confidence_adjustment,
            "result_label": new_pred.label,
            "result_confidence": new_pred.confidence,
            "is_stub": True,
        },
    )

    return {
        "regime": {
            **regime,
            "label": new_pred.label,
            "confidence": new_pred.confidence,
            "pending_label": new_pred.pending_label,
            "_pred": new_pred.__dict__,
            "_llm_review": {
                "label": review.review_label,
                "confidence_adjustment": review.confidence_adjustment,
                "risk_notes": review.risk_notes,
            },
        }
    }


def persist_regime(state: dict) -> dict:
    """Write the feature snapshot + regime state (+ optional LLM review) to Postgres."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[regime/persist_regime] run_id=%s", run_id)

    regime = state.get("regime", {})
    fs = regime.get("_features_snapshot")
    if not fs or not regime.get("_pred"):
        log.warning("persist_regime called with incomplete state (fs=%s, pred=%s)",
                    bool(fs), bool(regime.get("_pred")))
        return {}

    from trading_agent.regime.features import FeatureSnapshot

    snap = FeatureSnapshot(as_of=datetime.fromisoformat(fs["as_of"]))
    snap.features = fs["_obj_features"]
    snap.data_quality = fs["_obj_data_quality"]
    snap.source_freshness = fs.get("source_freshness", {})

    feature_id = insert_feature_snapshot(snap)

    from trading_agent.regime.classifier import RegimePrediction

    pred = RegimePrediction(**regime["_pred"])
    prior_state = get_latest_regime_state()

    # Compute and inject the gate dict so downstream graphs can read it directly
    size_mult = regime_size_multiplier(pred.label, pred.confidence)
    gate = {
        "size_multiplier": size_mult,
        "allow_new_entries": pred.label != "CRISIS",
        "require_llm_risk_review": pred.label in {"VOLATILE_TRANSITION", "BEAR_TREND"},
    }

    state_id = insert_regime_state(
        pred,
        as_of=snap.as_of,
        feature_snapshot_id=feature_id,
        model_version_id=None,  # rule-based fallback for now
        prior_label=prior_state["label"] if prior_state else None,
        gate=gate,
    )

    # If we ran an LLM review, persist that too
    review_blob = regime.get("_llm_review")
    if review_blob:
        from trading_agent.regime.llm_review import LLMReview

        review = LLMReview(
            review_label=review_blob["label"],
            confidence_adjustment=review_blob["confidence_adjustment"],
            risk_notes=review_blob.get("risk_notes", []),
        )
        insert_llm_review(
            regime_state_id=state_id,
            trigger_reason=pred.review_reason or "auto",
            prompt_hash="stub_phase_2_2",
            review=review,
            applied_action=review.review_label,
        )

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="persist_regime",
        event_type="regime_persisted",
        payload={
            "regime_state_id": state_id,
            "feature_snapshot_id": feature_id,
            "label": pred.label,
            "size_multiplier": size_mult,
        },
    )

    # Also notify on regime *changes* (not every persist)
    prior_label = prior_state["label"] if prior_state else None
    if prior_label != pred.label:
        ntfy_send(
            "risk",
            title=f"📊 Regime → {pred.label}",
            body=(
                f"Confidence: {pred.confidence:.2f}\n"
                f"Prior: {prior_label or '(none)'}\n"
                f"Size multiplier: {size_mult:.2f}\n"
                f"Crisis flags: {pred.crisis_flags or 'none'}"
            ),
            priority=4 if pred.label != "CRISIS" else 5,
            tags=["chart_with_upwards_trend"],
        )

    return {
        "regime": {
            **regime,
            "state_id": state_id,
            "feature_snapshot_id": feature_id,
            "gate": gate,
        }
    }
