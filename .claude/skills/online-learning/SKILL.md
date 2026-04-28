---
name: online-learning
description: Mutable parameter families with bounds, shadow→canary→promote pipeline, and statistical gates for parameter promotion.
---

# Online Learning

The system mutates a small set of bounded parameters weekly based on closed-trade stats and post-mortems. Hard risk caps are NEVER mutated by this loop.

## Mutable parameter families

| Family | Examples | Bounds |
| --- | --- | --- |
| sizing_aggression | `r1_max_pos_pct`, `r3_dollar_floor` | conservative ≤ value ≤ aggressive |
| stop_distances | `default_stop_atr_mult`, `option_premium_stop_pct` | 0.5× ≤ value ≤ 2.0× current |
| entry_filters | `min_setup_quality`, `min_iv_percentile_to_skip` | exposed ranges per family |
| regime_thresholds | crisis-overlay vol percentile | 0.75 ≤ value ≤ 0.95 |
| candidate_count | `max_candidates_per_scout` | 1 ≤ value ≤ 12 |

## Pipeline

1. **Spec** — Learning Critic proposes a parameter change with rationale, expected impact, and min_canary_trades.
2. **Synthesis** — System validates bounds, anti-thrashing (no change if same param touched in last 4 weeks).
3. **Implementation** — New value committed as a `param_versions` row with `mode=CANARY`.
4. **Validation** — N trades run with the canary version. Statistical gates checked:
   - Wilson 95% CI lower bound on win rate ≥ control's mean
   - Profit factor ≥ control's profit factor × 0.95
   - Max drawdown ≤ control's max drawdown × 1.1
5. **Analysis** — If gates pass, promote to `mode=ACTIVE`. If fail, revert.

## Shadow vs canary

- **Shadow** — proposed parameter runs in parallel with control on the SAME trades; only logged, not actuated. Used to gather signal cheaply.
- **Canary** — parameter actually drives a slice (e.g. 25%) of trades. Statistically gated.

## Hard rules

- N_canary_min ≥ 20 trades for any parameter family before considering promotion.
- No more than 1 active canary per family at a time.
- Risk hard-caps NEVER mutated; only "soft" parameters above.
- If a canary fails its drawdown gate within the first 5 trades, immediate auto-revert.
