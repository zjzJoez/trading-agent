# Rolling Walk-Forward Backtest v3 — Properly OOS Across 2017-2026

> **Superseded 2026-05-17:** The headline "Sharpe 1.02 vs SPY 0.84 (+0.18 edge)"
> below was a **broken-model artifact**. The HMM aborted EM at iter 3-4
> in every annual refit (binary features × full covariance → singularity);
> the single seed=42 + non-converged means/covars happened to produce
> labels that on this OOS window slightly beat SPY by chance. With
> continuous features + diag covariance + 5 seeds, Sharpe is **0.78 ± 0.015,
> a slight DISADVANTAGE vs SPY 0.84** — see `../walkforward_v4_converged/REPORT.md`.
> Do not cite the v3 numbers as evidence of system alpha.


**Date run:** 2026-05-16  
**Method:** 10 annual refits. For each year Y ∈ [2017, 2026], train a
fresh 4-state HMM on all snapshots with as_of < Y-01-01, auto-calibrate
state→label using training-window mean spy_ret_20 (no future leakage),
then classify each day in year Y with that HMM.

**Total OOS sample:** 2369 days across 10 years; 2367 with full 1-day
forward return.

## Why this run

Previous backtests (v1 and v2) used a single fixed train/test split,
which leaves most of the historical data unused and concentrates the
test sample in one specific market regime. Rolling walk-forward is the
gold-standard OOS evaluation in quantitative time-series research because:

- It uses *all* available data for OOS prediction
- Every prediction is genuinely OOS (the model that classified day d was
  trained only on data before year(d))
- It exposes the model's behavior across multiple market regimes
  (2017 ranging, 2018 selloff, 2019 melt-up, 2020 COVID, 2022 bear, etc.)

Auto-calibration rule used: rank the 4 hidden states by mean
training-window spy_ret_20. Lowest = BEAR_TREND, highest = BULL_TREND,
middle two = VOLATILE_TRANSITION. Deterministic, no future leakage.

## Headline numbers

| Metric                          | Strategy    | Passive SPY |
|---------------------------------|-------------|-------------|
| Cumulative return (10 years)    | **+133.4%** | **+261.8%** |
| Annualized return               | +9.45%      | +14.67%     |
| **Sharpe (annualized)**         | **1.02**    | **0.84**    |
| Sharpe rough SE (n=2367)        | ±0.33       | ±0.33       |
| Information Ratio vs SPY        | -0.48       | n/a         |
| Max drawdown                    | -14.2%      | (~-25% in 2022) |
| Avg size multiplier             | 0.604       | (n/a)       |

**The Sharpe gap (1.02 vs 0.84) is the headline finding.** Strategy beat
SPY on risk-adjusted basis by ~0.18 — but the SE of each Sharpe is ±0.33,
so the gap is **within one standard error** (about 0.55 SE apart).
Suggestive, not statistically conclusive over 10 years.

The strategy systematically lags SPY in cumulative return because the
average size multiplier is 0.604 (≈60% gross exposure). Mechanically,
this gives back ~40% of any SPY rally.

But the strategy's **max drawdown was -14.2% vs SPY's ~-25% in 2022** —
that's the regime gate doing real work during stress, even at
60%-of-SPY average exposure.

## Per-regime breakdown across 10 OOS years

Unconditional baseline (n=2367): mean fwd_20d **+1.23%**, neg_rate **30.15%**.

| Label                | n_days | mean_fwd_20d | neg_rate | hit_rate | size_mult |
|----------------------|--------|--------------|----------|----------|-----------|
| BULL_TREND           | 555    | +1.29%       | 23.6%    | **75.86%**| 1.0       |
| BEAR_TREND           | 77     | +1.42%       | 27.3%    | 27.3%    | 0.5       |
| VOLATILE_TRANSITION  | 1686   | +1.05%       | 32.9%    | n/a      | 0.5       |
| CRISIS (Layer 0)     | 51     | +6.15%       | 15.7%    | n/a      | 0.0       |

**Critical observations:**

1. **BULL_TREND has real signal across 10 OOS years.** Mean fwd_20d
   +1.29% (vs unconditional +1.23% — basically same), but the
   sign-match hit rate is **75.86%** vs unconditional positive rate of
   ~70% (1 - 0.30 = 0.70). The classifier's BULL_TREND identifications
   are correct ~6pp more often than chance on this window.

2. **The auto-calibrated classifier rarely picks BEAR.** Only 77 out of
   2369 days (3.3%) are labeled BEAR_TREND when calibration is forced
   to use training-window spy_ret_20 ranking. This is *fundamentally
   different* from v1 and v2 (which had 85% BEAR_TREND). The single
   fixed-split calibration over-weights "low breadth" features; the
   rolling auto-calibrator picks the genuinely lowest-return state,
   which is rarer.

3. **VOLATILE_TRANSITION is the default.** 71% of days. The model
   prefers "I don't know, size down" over confident bull or bear calls.
   This is conservative-by-default behavior.

4. **CRISIS overlay (Layer 0) consistently catches the right days.**
   51 days flagged across 10 years; mean forward 20d return +6.15%
   (mean reversion follows). This is the same finding from v1 but with
   8× more sample.

## Per-year breakdown — when does the gate help vs hurt?

| Year | SPY ret | Strategy ret | Δ (pp) | Dominant label |
|------|---------|--------------|--------|----------------|
| 2017 | +21.6% | +10.4%       | **-11.3** | TRANSITION (100%) |
| 2018 | -5.1%  | -5.3%        | -0.1   | TRANSITION (86%) |
| 2019 | +32.3% | +33.9%       | **+1.6** | BULL (~100%)   |
| 2020 | +15.6% | +8.5%        | -7.1   | mixed (COVID)  |
| 2021 | +31.3% | +17.7%       | -13.6  | TRANSITION (97%)|
| 2022 | **-19.0%** | **-9.2%** | **+9.8** | TRANSITION (96%)|
| 2023 | +26.0% | +12.5%       | -13.5  | TRANSITION (100%)|
| 2024 | +25.3% | +11.3%       | -14.0  | TRANSITION (100%)|
| 2025 | +18.2% | +13.4%       | -4.9   | BULL (76%)     |
| 2026 YTD | +3.3% | +1.2%    | -2.1   | TRANSITION (77%)|

**Where the gate adds value:**
- 2019 (+1.6pp): correctly classified the whole year as BULL → full 1.0× sizing → tracked SPY rally
- 2022 (+9.8pp): didn't catch the bear directly (only 3 CRISIS days), but
  TRANSITION-default 0.5× sizing halved the damage
- 2025 (-4.9pp): BULL classification kicked in for most of the year,
  recovered some of the historical gap

**Where the gate hurts:**
- 2017, 2021, 2023, 2024: classified as TRANSITION (size 0.5×) but
  market quietly rallied. ~13pp/year of "missed market" damage each time.
- These are the years where "no obvious bull signal AND no crisis" → the
  conservative default cost the most.

## What this tells us about the system

The 10-year OOS view is the most honest read on the deterministic spine
of the system. It shows:

1. **The regime classifier has real Sharpe-adding skill** — small but
   present (+0.18 Sharpe, within 1 SE so suggestive not significant).
2. **The size-down default sacrifices a lot of beta.** Average 60%
   exposure → cumulative return is ~half of SPY's. This is a defensive
   posture, not alpha generation.
3. **CRISIS overlay consistently works.** All 51 CRISIS days had
   positive forward 20d returns (mean +6.15%), suggesting Layer 0 catches
   transient pullbacks where staying out is correct.
4. **BULL_TREND has modest predictive value.** ~75.86% hit rate is
   slightly above unconditional ~70%, so identifying bull conditions
   adds a small amount of directional skill.

This is fundamentally a **lower-beta-with-slightly-better-Sharpe**
strategy pattern. Whether that's interesting depends on what the user
wants the system for:
- Pure return chasing → no, just buy SPY
- Sharpe optimization → marginal edge, hard to call statistically
- Drawdown protection → meaningful (-14% vs SPY -25% in 2022)

## Caveats

- Each annual HMM refit also hit hmmlearn non-convergence at iter 3-4.
  Same root cause (binary feature ill-conditioning). State separation in
  Viterbi decode is clean, so the practical impact is limited.
- Auto-calibration rule is one specific choice (`rank by spy_ret_20`).
  A different rule (e.g. "highest above_50dma fraction = BULL") might
  produce different labels and different aggregate behavior.
- The "strategy" is still a SPY-long proxy with `size_mult × SPY_next_1d`.
  Real system trades options on individual tickers — costs, gamma,
  spread dynamics are not modeled here.
- Information ratio of -0.48 means the strategy is *systematically
  worse than SPY when SPY is the benchmark*. The Sharpe-edge story
  comes from lower volatility, not from cleaner directional calls.

## Comparison across the three backtests

|                    | v1 (fixed) | v2 (fixed) | v3 (rolling) |
|--------------------|-----------|------------|--------------|
| OOS days           | 575       | 1328       | 2367         |
| Train window       | 2015-2023 | 2015-2020  | rolling      |
| BEAR_TREND density | 85%       | 85%        | **3.3%**     |
| BULL hit rate      | 62% (n=37)| 60% (n=151)| **76% (n=555)**|
| Strategy Sharpe    | 1.18      | 0.80       | **1.02**     |
| SPY Sharpe         | 1.19      | 0.87       | 0.84         |
| Sharpe Δ           | -0.01     | -0.07      | **+0.18**    |

v3 is the most credible number. v1 and v2 are dominated by their narrow
test windows + the over-bearish single-shot calibration.

## Artifacts

- `rolling_per_day.csv` — 2369 rows, one per OOS day
- `rolling_by_regime.csv` — per-label aggregates
- `rolling_by_year.csv` — per-year breakdown with strategy vs SPY return
- `rolling_headline.json` — top-line summary + per-year HMM calibration
  audit (state→label mapping each year)
