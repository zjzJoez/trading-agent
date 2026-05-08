---
name: portfolio-risk
description: Per-regime guardrail tables, correlation/factor/Greeks concepts, heat metric, and the risk decision taxonomy.
---

# Portfolio Risk

The Active Risk Agent applies deterministic per-regime caps, then optionally invokes a 3-step LLM council on borderline cases. LLMs can only DOWNGRADE — never widen — the deterministic envelope.

## Per-regime hard caps (fractions of equity)

| Cap | BULL | RANGE | TRANSITION | BEAR | CRISIS |
| --- | :-: | :-: | :-: | :-: | :-: |
| `heat_pct_max` | 6% | 6% | 3% | 3% | 1.5% |
| `single_underlying_delta_dollar_max` | 15% | 15% | 7.5% | 7.5% | 0% |
| `portfolio_net_delta_max` | 35% | 35% | 15% | 15% | 0% |
| `vega_10vol_max` (10vol shock) | 1% | 1% | 0.5% | 0.5% | 0.25% |
| `gamma_2sigma_max` (2σ shock) | 1.5% | 1.5% | 0.75% | 0.75% | 0.25% |
| `avg_pairwise_corr_max` | 0.65 | 0.65 | 0.55 | 0.55 | n/a |
| `same_expiry_premium_max` | 2% | 2% | 1% | 1% | 0% |
| `factor_exposure_max` | 30% | 30% | 20% | 20% | 5% |

## Decision taxonomy

- `APPROVE` — all guardrails clear; trade proceeds at requested qty.
- `DOWNSIZE` — one or more guardrails exceeded by ≤1.5×; qty multiplied by `suggested_qty_factor`.
- `VETO` — guardrail breach exceeds 1.5× cap, OR real_trading_marker, OR regime forbids new entries.
- `DEFER` — data_quality.degradation_level ≥ 2 (missing critical greeks/quotes).

## Heat metric

Sum of (option premium at risk + stock distance-to-stop × shares) divided by equity. Long premium uses full premium as max loss. Stocks without explicit stops use 5% implicit.

## Correlation crowding

Pairwise Pearson on daily returns over the last 60 sessions. The `avg_pairwise_corr_max` is checked against the WORST off-diagonal correlation in the open-position set.

## Council interaction

- LLM council runs when deterministic decision != APPROVE, OR regime ∈ {TRANSITION, BEAR, CRISIS}, OR is_earnings_play, OR canary_param_version.
- 3-step chain: Conservative → Opportunity → Arbiter.
- Arbiter cannot upgrade beyond deterministic ceiling.
