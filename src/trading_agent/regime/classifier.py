"""Regime classifier — hard crisis overlay + Gaussian HMM + LLM gate.

Layer 0 — Crisis overlay (deterministic):
    Hard rules on VIX, SPY, correlation, credit. Two flags → CRISIS,
    independent of HMM output. Catches gap-risk events that HMMs trained
    on quiet history will smooth over.

Layer 1 — Gaussian HMM:
    4-state HMM trained on standardized features. Emission means are
    interpretable; calibration maps {state_idx -> regime_label}. The
    HMM owns "BULL_TREND, RANGE_LOW_VOL, BEAR_TREND, VOLATILE_TRANSITION"
    — never CRISIS (Layer 0 does that).

Layer 2 — LLM second-opinion (in llm_review.py):
    Triggered only when confidence < 0.65, label changed today, or DQ
    warning. Can downgrade/defer. Cannot upgrade.

For Phase 2.2 we ship Layers 0 and 1. Layer 2 is wired in Phase 2.4
(LLM oauth_router).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trading_agent.regime.features import (
    HMM_FEATURE_ORDER,
    FeatureSnapshot,
)

log = logging.getLogger(__name__)

RegimeLabel = Literal[
    "BULL_TREND",
    "RANGE_LOW_VOL",
    "VOLATILE_TRANSITION",
    "BEAR_TREND",
    "CRISIS",
]

# All regime labels as a fixed-order tuple (for serialization, UI sorting, etc.)
ALL_REGIMES: tuple[RegimeLabel, ...] = (
    "BULL_TREND",
    "RANGE_LOW_VOL",
    "VOLATILE_TRANSITION",
    "BEAR_TREND",
    "CRISIS",
)


# ---------------------------------------------------------------------------
# Layer 0 — Hard crisis overlay
# ---------------------------------------------------------------------------


@dataclass
class CrisisOverlayResult:
    is_crisis: bool
    flags: list[str]
    detail: dict[str, float]


# Default thresholds per §6.2 of plan v3.
# These can be overridden via data/regime_params.json (Online Learning §8 will
# tune them within bounded ranges).
DEFAULT_CRISIS_PARAMS = {
    "vix_abs_high": 35.0,
    "vix_5d_change_high": 10.0,
    "spy_5d_drop": -0.07,
    "corr_high": 0.80,
    "rv_high": 0.30,
    "hyg_5d_drop": -0.05,
    "min_flags_to_crisis": 2,
}


def crisis_overlay(
    snapshot: FeatureSnapshot,
    *,
    params: dict[str, float] | None = None,
) -> CrisisOverlayResult:
    """Layer 0. Returns CRISIS if at least `min_flags_to_crisis` rules trip."""
    p = {**DEFAULT_CRISIS_PARAMS, **(params or {})}
    f = snapshot.features

    flags: list[str] = []
    detail: dict[str, float] = {}

    if f.get("vix_level", 0) >= p["vix_abs_high"]:
        flags.append("vix_abs_high")
    if f.get("vix_chg_5", 0) >= p["vix_5d_change_high"]:
        flags.append("vix_5d_change_high")
    if f.get("spy_ret_5", 0) <= p["spy_5d_drop"]:
        flags.append("spy_5d_drop")
    if (
        f.get("avg_pairwise_corr_20", 0) >= p["corr_high"]
        and f.get("spy_rv_20", 0) >= p["rv_high"]
    ):
        flags.append("corr_rv_combo")
    if f.get("hyg_mom_20", 0) <= p["hyg_5d_drop"]:
        flags.append("hyg_drop")

    detail["vix_level"] = f.get("vix_level", 0)
    detail["vix_chg_5"] = f.get("vix_chg_5", 0)
    detail["spy_ret_5"] = f.get("spy_ret_5", 0)
    detail["avg_pairwise_corr_20"] = f.get("avg_pairwise_corr_20", 0)
    detail["spy_rv_20"] = f.get("spy_rv_20", 0)
    detail["hyg_mom_20"] = f.get("hyg_mom_20", 0)

    return CrisisOverlayResult(
        is_crisis=len(flags) >= int(p["min_flags_to_crisis"]),
        flags=flags,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Layer 1 — Gaussian HMM (4 states) + calibration → regime label
# ---------------------------------------------------------------------------


@dataclass
class HMMModel:
    """Serializable HMM artifact. Persisted to regime_model_versions."""

    n_states: int
    feature_order: list[str]
    means: list[list[float]]                # n_states × n_features
    covars: list[list[list[float]]]         # n_states × n_features × n_features
    transmat: list[list[float]]             # n_states × n_states
    startprob: list[float]                  # n_states
    feature_means: list[float]              # n_features (training-time standardization)
    feature_stds: list[float]               # n_features
    state_to_label: dict[int, RegimeLabel]  # calibration mapping
    train_meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON cannot key by int; convert state_to_label keys to str
        d["state_to_label"] = {str(k): v for k, v in self.state_to_label.items()}
        return d

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "HMMModel":
        if isinstance(blob.get("state_to_label", {}), dict):
            blob = dict(blob)
            blob["state_to_label"] = {int(k): v for k, v in blob["state_to_label"].items()}
        return cls(**blob)


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    s = std.copy()
    s[s == 0] = 1.0
    return (x - mean) / s


def _gaussian_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Multivariate normal log-density — handcoded so we don't pull a heavy dep
    just for inference. Training uses hmmlearn which is heavier."""
    d = len(x)
    diff = x - mu
    sign, logdet = np.linalg.slogdet(cov + 1e-9 * np.eye(d))
    if sign <= 0:
        return -1e9
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    return float(-0.5 * (d * math.log(2 * math.pi) + logdet + diff @ inv @ diff))


def hmm_predict(
    model: HMMModel,
    snapshot: FeatureSnapshot,
) -> tuple[int, np.ndarray]:
    """Single-step posterior over hidden states given the latest feature vector.

    Uses a stationary-distribution prior and emission probabilities to compute
    P(state | obs). Returns (argmax_state, probabilities).

    For a richer posterior over a sequence, use forward-backward via hmmlearn
    in batch mode (Phase 2.2.1 — backfill). For real-time decisions, this
    one-shot Bayes posterior is sufficient.

    Feature ordering: builds the input vector from `snapshot.features` using
    `model.feature_order` — NOT the module-level HMM_FEATURE_ORDER. This
    matters when the codebase's feature schema has evolved since the model
    was trained: an old model continues to work as long as the snapshot
    still contains the keys it was trained on.
    """
    n = model.n_states
    # Build the input vector in THIS model's feature order, falling back to
    # the module-level HMM_FEATURE_ORDER only if the model has no embedded
    # order (shouldn't happen for serialized HMMs).
    feature_order = model.feature_order or [
        # late import to avoid circular dep
        k for k in __import__(
            "trading_agent.regime.features", fromlist=["HMM_FEATURE_ORDER"]
        ).HMM_FEATURE_ORDER
    ]
    x = np.array(
        [snapshot.features.get(k, 0.0) for k in feature_order],
        dtype=float,
    )
    mean = np.array(model.feature_means)
    std = np.array(model.feature_stds)
    z = _standardize(x, mean, std)

    log_prior = np.log(np.array(model.startprob) + 1e-12)
    log_emit = np.zeros(n)
    for s in range(n):
        mu = np.array(model.means[s])
        cov = np.array(model.covars[s])
        log_emit[s] = _gaussian_logpdf(z, mu, cov)

    log_post = log_prior + log_emit
    # softmax for numerical stability
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()
    return int(np.argmax(post)), post


# ---------------------------------------------------------------------------
# Public API — combine Layers 0 + 1 into a labeled prediction
# ---------------------------------------------------------------------------


@dataclass
class RegimePrediction:
    label: RegimeLabel
    confidence: float
    pending_label: RegimeLabel | None = None     # set when low-confidence transition
    probabilities: dict[RegimeLabel, float] = field(default_factory=dict)
    crisis_flags: list[str] = field(default_factory=list)
    degradation_level: int = 0
    needs_llm_review: bool = False
    review_reason: str | None = None
    model_version_id: int | None = None
    feature_snapshot_id: int | None = None

    def to_db_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "pending_label": self.pending_label,
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "crisis_flags": self.crisis_flags,
            "degradation_level": int(self.degradation_level),
            "gate": {},  # filled in by gates.py before persistence
        }


CONFIDENCE_TRANSITION_THRESHOLD = 0.55
CONFIDENCE_LLM_REVIEW_THRESHOLD = 0.65
CONFIDENCE_PENDING_THRESHOLD = 0.75


def classify(
    snapshot: FeatureSnapshot,
    model: HMMModel | None,
    *,
    crisis_params: dict[str, float] | None = None,
    prior_label: RegimeLabel | None = None,
) -> RegimePrediction:
    """Top-level classify: Layer 0 → Layer 1 → uncertainty rules.

    `prior_label` is the previous tick's label, used to detect transitions
    that should soft-land in VOLATILE_TRANSITION.
    """
    # ---- DQ gate first: missing critical data → safe-mode TRANSITION ----
    deg = snapshot.degradation_level
    if deg >= 2:
        return RegimePrediction(
            label="VOLATILE_TRANSITION",
            confidence=0.0,
            probabilities={"VOLATILE_TRANSITION": 1.0},
            degradation_level=deg,
            needs_llm_review=False,
            review_reason="data_quality_critical_missing_safe_mode",
        )

    # ---- Layer 0: Crisis overlay ----
    crisis = crisis_overlay(snapshot, params=crisis_params)
    if crisis.is_crisis:
        return RegimePrediction(
            label="CRISIS",
            confidence=1.0,
            probabilities={"CRISIS": 1.0},
            crisis_flags=crisis.flags,
            degradation_level=deg,
            needs_llm_review=False,
            review_reason="crisis_overlay_hit",
        )

    # ---- Layer 1: HMM (if model available; else rule-based fallback) ----
    if model is None:
        return _rule_based_fallback(snapshot, crisis.flags, deg, prior_label)

    state_idx, post = hmm_predict(model, snapshot)
    label_for_state = model.state_to_label.get(state_idx, "VOLATILE_TRANSITION")
    confidence = float(post[state_idx])

    # Build probabilities dict in regime-label space (multiple states may map
    # to the same regime label; sum their masses)
    probs: dict[RegimeLabel, float] = {r: 0.0 for r in ALL_REGIMES if r != "CRISIS"}
    for s in range(model.n_states):
        lbl = model.state_to_label.get(s, "VOLATILE_TRANSITION")
        probs[lbl] = probs.get(lbl, 0.0) + float(post[s])

    label: RegimeLabel = label_for_state
    pending_label = None
    review_reason = None

    # ---- Uncertainty rules ----
    if confidence < CONFIDENCE_TRANSITION_THRESHOLD:
        # Force soft-land state
        pending_label = label
        label = "VOLATILE_TRANSITION"
        review_reason = "confidence_below_transition_threshold"
    elif (
        prior_label
        and prior_label != label
        and confidence < CONFIDENCE_PENDING_THRESHOLD
    ):
        # Label changed but not confident enough — gate as TRANSITION until
        # confidence stabilizes; remember the pending direction
        pending_label = label
        label = "VOLATILE_TRANSITION"
        review_reason = "label_changed_low_confidence"

    needs_llm = (
        confidence < CONFIDENCE_LLM_REVIEW_THRESHOLD
        or label == "VOLATILE_TRANSITION"
        or (prior_label is not None and prior_label != label)
        or deg > 0
    )

    return RegimePrediction(
        label=label,
        confidence=confidence,
        pending_label=pending_label,
        probabilities=probs,
        crisis_flags=crisis.flags,
        degradation_level=deg,
        needs_llm_review=needs_llm,
        review_reason=review_reason,
    )


def _rule_based_fallback(
    snapshot: FeatureSnapshot,
    crisis_flags: list[str],
    degradation_level: int,
    prior_label: RegimeLabel | None,
) -> RegimePrediction:
    """Safety net used until an HMM is trained.

    Simple, conservative: trend score + RV bucket → label. Confidence is
    fixed at 0.6 to ensure LLM review fires (Phase 2.4 will catch rule
    blind spots).
    """
    f = snapshot.features
    trend = (
        0.5 * f.get("spy_above_50dma", 0)
        + 0.3 * (1.0 if f.get("spy_ret_20", 0) > 0 else 0.0)
        + 0.2 * f.get("spy_above_200dma", 0)
    )
    rv = f.get("spy_rv_20", 0.18)

    # Order matters: ultra-low vol + flat returns short-circuits to RANGE
    # before trend logic, because a flat tape with positive trend score
    # belongs in RANGE_LOW_VOL, not BULL_TREND (which implies movement).
    if rv < 0.12 and abs(f.get("spy_ret_20", 0)) < 0.015:
        label: RegimeLabel = "RANGE_LOW_VOL"
    elif trend > 0.6 and rv < 0.20:
        label = "BULL_TREND"
    elif trend < 0.3 and f.get("spy_ret_20", 0) < -0.02:
        label = "BEAR_TREND"
    else:
        label = "VOLATILE_TRANSITION"

    confidence = 0.60   # always invite LLM review for rule-based fallback
    return RegimePrediction(
        label=label,
        confidence=confidence,
        probabilities={label: confidence},
        crisis_flags=crisis_flags,
        degradation_level=degradation_level,
        needs_llm_review=True,
        review_reason="rule_based_fallback_no_hmm",
    )


# ---------------------------------------------------------------------------
# Model load/save — used by training jobs (Phase 2.2.1)
# ---------------------------------------------------------------------------


def save_model(model: HMMModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model.to_json(), indent=2))


def load_model(path: str | Path) -> HMMModel | None:
    p = Path(path)
    if not p.is_file():
        return None
    return HMMModel.from_json(json.loads(p.read_text()))
