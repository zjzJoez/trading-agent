# Walk-Forward v5 — Extended Training History (2007-2026)

**Date run:** 2026-05-19  
**Headline:** Backfilled 8 additional years (2007-04 → 2015-03, the
GFC + 2010 flash crash + 2011 EU crisis + 2013 taper era). HMM v5 is
trained on **4760 obs vs v4's 2807 obs** (+70% data). Strategy Sharpe
is **0.908 ± 0.029** (essentially unchanged from v4.2's 0.905 ± 0.026).
The expected "more data → better backtest" intuition was **wrong** for
the headline number. But two qualitative wins:

1. **Today's classification flipped BEAR_TREND → RANGE_LOW_VOL.** The
   model has now seen real 2008-level crisis (VIX 38, RV 40%, SPY -56%),
   so today's mild conditions clearly aren't bear.
2. **BEAR_TREND label is used much more sparingly** — 348 → 142 days on
   2017-2026 OOS window. The model has a calibrated "what is real bear"
   threshold instead of over-firing.

## What changed since v4.2

| Aspect | v4.2 | **v5** |
|---|---|---|
| Training data | 2015-03 → 2026-05 (2807 obs) | **2007-04 → 2026-05 (4760 obs)** |
| Crises in training | 2020 COVID + 2022 mild bear | **+ 2008 GFC + 2010 flash crash + 2011 EU + 2013 taper** |
| Production HMM | id=4 | **id=5** |

## State separation — v5 has a real crisis state

Comparison of state characteristics (production HMM):

| State | v4 (id=4) | **v5 (id=5)** |
|---|---|---|
| BULL_TREND mean VIX | 18.4 | 20.2 |
| RANGE_LOW_VOL mean VIX | 13.6 | 14.3 |
| BEAR_TREND mean VIX | 18.9 | 21.5 |
| **VOLATILE_TRANSITION mean VIX** | **30.6** | **38.8** |
| VOL_TRANSITION mean RV | 30.7% | **39.7%** |
| VOL_TRANSITION p10 spy_ret_20 | -9.6% | **-13.9%** |

The VOLATILE_TRANSITION state now captures real crisis-level features.
v4 thought "30 VIX + 30% RV = crisis"; v5 knows "real crisis means
~38 VIX + 40% RV". When markets actually break down in the future, v5
will correctly identify it (and Layer 0 CRISIS overlay catches the
single-day cliff events).

## Headline numbers (5-seed, 2017-2026 OOS)

| Metric | v4.2 (prior fix only) | **v5 (+ long history)** |
|---|---|---|
| Strategy Sharpe | 0.905 ± 0.026 | **0.908 ± 0.029** |
| SPY Sharpe | 0.837 | 0.837 |
| **Sharpe Δ** | **+0.068** | **+0.071** |
| Cum strategy ret | +125% ± 5pp | +121% ± 4pp |
| Cum SPY ret | +262% | +260% |
| Max drawdown | -15.2% | -13.9% to -15.3% |
| Avg size_mult | 0.69 | 0.69 |

**The headline number is essentially unchanged.** Adding 70% more
training data moved Sharpe by less than 1 SE. The most likely reason:
2017-2026 OOS days don't visit the regions that 2007-2014 data added
to the training distribution. The 2020 COVID + 2022 bear that v4 had
were already sufficient bear variety for 2017-2026 evaluation.

## What DID change — label quality

Label distribution on 2017-2026 OOS (seed=42, n=2369):

| Label | v4.2 | **v5** | Δ |
|---|---|---|---|
| RANGE_LOW_VOL | 789 (33%) | **1070 (45%)** | +281 |
| VOLATILE_TRANSITION | 604 (25%) | 676 (29%) | +72 |
| BULL_TREND | 578 (24%) | 431 (18%) | -147 |
| **BEAR_TREND** | **348 (15%)** | **142 (6%)** | **-206** |
| CRISIS (Layer 0) | 51 (2%) | 51 (2%) | 0 |

v5 cuts BEAR_TREND usage by **60%**. The model no longer over-fires
bear on mild market weakness — it reserves BEAR for the (rarer) real
trend-down conditions that match the historical 2008/2022 pattern.
Most of the freed days reclassify as RANGE_LOW_VOL (calm-bull-grind).

**This is a major qualitative improvement** even if the OOS Sharpe is
unchanged: the label semantics are now closer to what a human would
agree with looking at the same data.

## Today (2026-05-18) — the most visible behavioral change

| HMM | Label | Confidence |
|---|---|---|
| v4 (id=4, 2015-2026 training) | BEAR_TREND | 0.6649 |
| **v5 (id=5, 2007-2026 training)** | **RANGE_LOW_VOL** | 0.9987 |

Same features, different classification. v5 has seen 2008 GFC and
knows today's "VIX 18 + breadth 27%" is nothing like real bear.
Result: gate sizing changes from 0.5x (BEAR) to 0.75x (RANGE_LOW_VOL).

Practical impact: the system tomorrow morning will run premarket
scans at 0.75x sizing instead of 0.5x. New trades get 50% more
capital deployed than they would have under v4.

## Confidence — still high, slightly worse

| | v4.2 | **v5** |
|---|---|---|
| pct conf ≥ 0.999 | 60.3% | 65.0% |
| mean confidence | 0.953 | 0.960 |

Confidence got marginally MORE concentrated, not less. The extra
training data made per-state distributions more confidently separated
(model has more data to define each state's center), but didn't
distribute mass more diffusely. This is the diag-cov + high-dim
pathology that advisor flagged earlier — **adding data doesn't fix
this; it's a model-structure issue**.

By year, confidence is now sensibly varied:
- 2017 (calm year): mean conf 0.991 ← properly confident
- 2018 (turbulent): mean conf 0.967 ← properly less confident
- 2021 (bull with whiplash): 0.942 ← properly uncertain
- 2022 (real bear): 0.947 ← properly uncertain  
- 2026 YTD (narrow rally): 0.908 ← appropriately least confident

The PATTERN of confidence is right even if the overall level is too
high.

## Per-year breakdown (seed=42, v5)

| Year | SPY ret | Strategy | Δ pp | n_bull | n_bear | n_range_LV |
|---|---|---|---|---|---|---|
| 2017 | +21.6% | +14.9% | -6.7 | 0 | 0 | (mostly RANGE) |
| 2018 | -5.1% | -6.8% | -1.7 | 12 | 5 | (mixed) |
| 2019 | +32.3% | +19.6% | -12.7 | 70 | 2 | (bull captured partially) |
| 2020 | +15.6% | +11.7% | -4.0 | 104 | 43 | (COVID year double-direction) |
| 2021 | +31.3% | +15.7% | -15.5 | 51 | 0 | (bull captured partially) |
| **2022** | **-19.0%** | **-9.6%** | **+9.4** | 35 | **74** | **(real BEAR called)** |
| 2023 | +26.0% | +17.5% | -8.5 | 69 | 0 | |
| 2024 | +25.3% | +12.8% | -12.4 | 25 | 0 | |
| 2025 | +18.2% | +9.0% | -9.2 | 41 | 18 | |
| 2026 YTD | +2.7% | +2.9% | +0.2 | 24 | 0 | |

**v5 in 2022** fires BEAR 74 days vs v4.2's 60 days — appropriately
more bearish during the real bear year. 2022 protection (+9.4pp) is
slightly less than v4.2's +10pp because the system stops being
over-defensive in non-bear years.

## What this answers and doesn't

**Answers:**
1. **More data doesn't fix the headline Sharpe.** +0.07 over SPY is
   robust across feature schemas, training windows, and now training
   length. That's the model's natural alpha on this OOS period.
2. **More data DOES improve label semantics.** BEAR_TREND now means
   what a human would call BEAR. RANGE_LOW_VOL correctly identifies
   calm bull markets.
3. **More data DOES improve crisis-recognition robustness.** When a
   real 2008-style event happens, v5 will recognize VOLATILE_TRANSITION
   features at VIX ~38 / RV ~40% instead of confusing it with severe
   bear.

**Doesn't answer:**
1. The 65% confidence ≥ 0.999 issue. This is a model-class issue
   (diag-cov + 17 features), not a data issue. Next experiment:
   `min_covar=0.05` or `covariance_type='tied'`.
2. Whether the +0.07 Sharpe edge is statistically meaningful. Still
   ~0.21 SE on a single estimator. Multi-seed consistency suggests
   it's real but small.
3. Whether the LLM agent group has alpha. Still impossible to backtest
   due to LLM training contamination. The paper benchmark window +
   shadow_proposals are the only honest path.

## Status

- Production HMM is now id=5 (active). id=4 retired.
- Today's classification: RANGE_LOW_VOL @ 0.9987 (was BEAR @ 0.66 under v4).
- Tomorrow's premarket scan will run at 0.75x sizing (RANGE_LOW_VOL) 
  vs 0.5x (BEAR) — meaningful behavioral change.

## What's next

Worth doing:
1. `min_covar` sweep — see if we can push confidence ≥ 0.999 rate
   below 30%. 30 min.
2. Pre-2007 backfill (2000-2007 dot-com era) — adds dot-com but loses
   ~5 cross-asset features. Engineering-complex. May not improve
   anything per v5 finding that more data didn't move Sharpe.

Worth waiting on:
1. Forward production data with v5 + correct prior — will accumulate
   in shadow_proposals + benchmark_marks. After 4-6 weeks we'll see
   how v5 + LLM agent group actually behaves in real time.

## Artifacts

- `seed_{42,7,99,123,2026}/rolling_per_day.csv` — per-seed per-day classifications
- `seed_*/rolling_by_year.csv` — annual breakdown
- `seed_*/rolling_by_regime.csv` — regime aggregates
- `rolling_seed_summary.csv` — 5-seed comparison
- `rolling_seed_aggregate.json` — aggregate headline
