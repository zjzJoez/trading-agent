---
name: regime-detection
description: How to read market regime classifier output (HMM + crisis overlay) and what each regime implies for trading.
---

# Regime Detection

The autonomous system classifies the current market into one of 5 regimes. This shapes every downstream decision.

## States

| Label | Trigger conditions (rough) | Trading posture |
| --- | --- | --- |
| `BULL_TREND` | SPY 50d > 200d, 5d return > 0, vol < 75th pctile, breadth positive | Full long bias; full size multiplier (1.0) |
| `RANGE_LOW_VOL` | SPY trendless, 20d realized vol < 60th pctile, breadth flat | Long with smaller size (0.7) |
| `VOLATILE_TRANSITION` | Vol elevated, breadth ambiguous, possible flip | Long allowed but heavily downsized (0.4); LLM risk council mandatory |
| `BEAR_TREND` | SPY 50d < 200d, 5d return < 0, vol elevated, breadth negative | Long puts only; size 0.5 |
| `CRISIS` | VIX >85th pctile + breadth collapse + correlated drawdown | NO new entries; only reduce-risk |

## Inputs available

A `RegimeState` object on the LangGraph state has:
- `label`: one of the 5 above
- `confidence`: 0.0–1.0 (combined from HMM posterior + crisis overlay)
- `pending_label`: a state the system is transitioning to (after 3 consecutive higher-confidence reads)
- `crisis_flags`: list (vix_spike_85th, breadth_collapse, correlated_drawdown_5pct, vix_term_inversion)
- `gate.size_multiplier`: 0.0–1.0 multiplier applied to deterministic sizing
- `gate.allow_new_entries`: bool (false in CRISIS)
- `gate.require_llm_risk_review`: bool (true in TRANSITION/BEAR)

## Slow-on/fast-off rule

Confidence rises slowly: a regime change to a more risk-on tier requires 3 consecutive days of higher posterior. Confidence drops fast: any single CRISIS-overlay trigger immediately moves to CRISIS regardless of HMM.

## How LLMs use this

- LLMs CANNOT re-classify the regime upward. They can only DOWNGRADE confidence or DEFER.
- Every analyst report prefaces its read with the current regime so the chain stays consistent.

## What the evidence supports (and what it doesn't)

Treat the regime layer as **drawdown control, not alpha**. The walk-forward
record (2017 → 2026-06-12, 5 seeds): overlay Sharpe 0.908 ± 0.029 vs SPY
0.837. The +0.071 difference is ~0.2 standard errors — not distinguishable
from zero — and the overlay trails SPY buy-and-hold by ~142pp cumulatively
(avg 0.69× exposure). What IS demonstrated: 2022-bear protection (+9–10pp)
and max drawdown ~−14–15% vs SPY's ~−25%.

Two caveats when reasoning about regime output:

- That OOS window was reused across the v3–v5 design iterations, so it's
  burned for model selection. Data from 2026-06-13 forward is a frozen
  holdout for v6+ (`src/trading_agent/regime/holdout.py`) — never run a
  model-selection evaluation into it without `--break-glass`.
- The walk-forward validated an auto-calibrated architecture; production
  runs a human-calibrated artifact (`scripts/promote_hmm.py`). To measure
  the deployed model itself, use `scripts/validate_production_hmm.py`.

Practical implication: lean on the regime label to SIZE DOWN and to block
entries (CRISIS/BEAR gates), not as a directional return forecast.
