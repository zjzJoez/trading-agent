"""LLM second-opinion on regime classification.

Wired in Phase 2.4 (oauth_router). For now exposes a stub interface so
the graph nodes that need it can be written end-to-end.

Hard rules (never change):
    - LLM CAN: CONFIRM, DOWNGRADE_TO_TRANSITION, DOWNGRADE_TO_CRISIS,
               DATA_QUALITY_DEFER
    - LLM CANNOT: upgrade to a more permissive regime
    - LLM CANNOT: change numeric thresholds
    - Every review is persisted to regime_llm_reviews

Triggered when:
    - confidence < 0.65
    - label changed today
    - data quality warning
    - HMM disagrees with rule overlay (Phase 2.4 wiring)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from trading_agent.regime.classifier import RegimePrediction

log = logging.getLogger(__name__)

ReviewLabel = Literal[
    "CONFIRM",
    "DOWNGRADE_TO_TRANSITION",
    "DOWNGRADE_TO_CRISIS",
    "DATA_QUALITY_DEFER",
]


@dataclass
class LLMReview:
    review_label: ReviewLabel
    confidence_adjustment: float = 0.0    # -0.10 to 0.0; cannot raise confidence
    risk_notes: list[str] = None  # type: ignore[assignment]
    must_defer_new_entries: bool = False
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.risk_notes is None:
            self.risk_notes = []
        # Hard guardrail: cannot raise confidence
        if self.confidence_adjustment > 0:
            log.warning(
                "LLM tried to raise confidence (+%.2f) — clamping to 0",
                self.confidence_adjustment,
            )
            self.confidence_adjustment = 0.0


def apply_review_to_prediction(
    pred: RegimePrediction,
    review: LLMReview,
) -> RegimePrediction:
    """Apply an LLM review to a prediction. Hard guardrails enforced here so
    even a misbehaving LLM cannot escalate to a more permissive label.
    """
    # Confidence can only go down
    new_conf = max(0.0, min(1.0, pred.confidence + review.confidence_adjustment))

    if review.review_label == "CONFIRM":
        return RegimePrediction(
            **{**pred.__dict__, "confidence": new_conf}
        )

    if review.review_label == "DOWNGRADE_TO_CRISIS":
        # Always allowed (more conservative)
        new_pred = RegimePrediction(
            label="CRISIS",
            confidence=max(new_conf, 0.95),
            probabilities={"CRISIS": 1.0},
            crisis_flags=pred.crisis_flags + ["llm_downgrade"],
            degradation_level=pred.degradation_level,
            needs_llm_review=False,
            review_reason="llm_downgrade_to_crisis",
        )
        return new_pred

    if review.review_label == "DOWNGRADE_TO_TRANSITION":
        # Only if the original is more permissive than transition
        if pred.label in {"BULL_TREND", "RANGE_LOW_VOL", "BEAR_TREND"}:
            return RegimePrediction(
                label="VOLATILE_TRANSITION",
                confidence=new_conf,
                pending_label=pred.label,
                probabilities={**pred.probabilities, "VOLATILE_TRANSITION": new_conf},
                crisis_flags=pred.crisis_flags,
                degradation_level=pred.degradation_level,
                needs_llm_review=False,
                review_reason="llm_downgrade_to_transition",
            )
        return RegimePrediction(**{**pred.__dict__, "confidence": new_conf})

    if review.review_label == "DATA_QUALITY_DEFER":
        # Force degradation_level >= 1; downstream gates will block new entries
        return RegimePrediction(
            label="VOLATILE_TRANSITION",
            confidence=new_conf,
            pending_label=pred.label,
            probabilities=pred.probabilities,
            crisis_flags=pred.crisis_flags,
            degradation_level=max(pred.degradation_level, 1),
            needs_llm_review=False,
            review_reason="llm_data_quality_defer",
        )

    # Unknown review_label — fail safe (defer new entries)
    return RegimePrediction(
        label="VOLATILE_TRANSITION",
        confidence=0.0,
        pending_label=pred.label,
        probabilities=pred.probabilities,
        crisis_flags=pred.crisis_flags,
        degradation_level=max(pred.degradation_level, 1),
        needs_llm_review=False,
        review_reason="llm_unknown_label_fail_safe",
    )


def stub_llm_review(pred: RegimePrediction) -> LLMReview:
    """Phase 2.2 placeholder. Returns CONFIRM with no adjustment.

    Kept for unit tests. Production graph calls `real_llm_review` which
    routes through the OAuth router to Claude Sonnet 4.6.
    """
    log.info("[stub_llm_review] returning CONFIRM (used in tests / no-LLM mode)")
    return LLMReview(
        review_label="CONFIRM",
        confidence_adjustment=0.0,
        risk_notes=["stub: no LLM"],
    )


def real_llm_review(pred: RegimePrediction, *, top_features: dict | None = None,
                    prior_label: str | None = None) -> LLMReview:
    """Phase 2.4: route the regime review through the OAuth router.

    Falls back to `stub_llm_review` on any failure (router not configured,
    schema violation, timeout). Hard guardrails in `apply_review_to_prediction`
    cap the LLM's authority — it cannot upgrade confidence or label.

    `top_features` is a {name: raw_value} mapping (NOT z-scores). The features
    have natural units (VIX in points, returns as fractions, momentum as
    fractions, MA-above flags as 0/1). The LLM's prompt makes this explicit
    so it doesn't misinterpret raw values as standard deviations.
    """
    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import RegimeReviewerOutput

        router = get_router()
        prompt_lines = [
            f"proposed_label: {pred.label}",
            f"proposed_confidence: {pred.confidence:.2f}",
            f"prior_label: {prior_label or 'none'}",
            f"crisis_flags: {pred.crisis_flags or 'none'}",
            f"degradation_level: {pred.degradation_level}",
        ]
        if top_features:
            prompt_lines.append("\ntop_features (raw values, not z-scores; units inline):")
            for k, v in top_features.items():
                if isinstance(v, (int, float)):
                    prompt_lines.append(f"  {k}: {v:.4f}")
                else:
                    prompt_lines.append(f"  {k}: {v}")
            prompt_lines.append(
                "\nFeature units cheat-sheet: vix_level=VIX index points (~12-30 normal); "
                "*_ret_*=fractional returns over period; *_rv_*=annualized realized vol "
                "(0.10-0.25 normal); *_pctl_*=percentile [0-1]; *_above_*dma=1 if above, "
                "0 if below; breadth_*_pct=fraction of stocks meeting condition; "
                "avg_pairwise_corr=Pearson correlation [0-1]."
            )
        prompt_lines.append("\nReview and respond per your output schema.")
        result = router.call(
            "regime_reviewer",
            "\n".join(prompt_lines),
            schema=RegimeReviewerOutput,
        )
        if result.parsed is None or not isinstance(result.parsed, RegimeReviewerOutput):
            log.warning("[real_llm_review] router returned no parsed output → CONFIRM")
            return stub_llm_review(pred)
        out = result.parsed
        return LLMReview(
            review_label=out.review_label,  # type: ignore[arg-type]
            confidence_adjustment=out.confidence_adjustment,
            risk_notes=list(out.risk_notes),
            must_defer_new_entries=out.must_defer_new_entries,
        )
    except Exception as e:
        log.error("[real_llm_review] failed → fail-safe CONFIRM: %s", e)
        return stub_llm_review(pred)
