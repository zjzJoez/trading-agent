"""Phase 2.2 — Regime Detection Agent.

Three-layer classifier (§6 of plan v3):
    Layer 0  Hard crisis overlay  (deterministic, <1ms)
    Layer 1  Gaussian HMM         (4 hidden states, deterministic given features)
    Layer 2  LLM second-opinion   (downgrade-only, only on ambiguity)

Five regime states:  BULL_TREND | RANGE_LOW_VOL | VOLATILE_TRANSITION | BEAR_TREND | CRISIS
"""

from trading_agent.regime.classifier import (  # noqa: F401
    classify,
    crisis_overlay,
)
from trading_agent.regime.features import build_feature_snapshot  # noqa: F401
from trading_agent.regime.gates import (  # noqa: F401
    GateResult,
    gate_trade_for_regime,
    regime_size_multiplier,
)
