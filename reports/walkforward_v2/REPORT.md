# Walk-Forward Backtest v2 — Earlier Cutoff (Train 2015-2020, Test 2021-2026)

**Date run:** 2026-05-16  
**Backtest HMM:** `regime_model_versions.id=3`, status=shadow  
**Train window:** 2015-03-31 → 2020-12-31 (1451 snapshots)  
**Test window:** 2021-01-01 → 2026-05-15 (1362 OOS snapshots; 1328 with full forward)  
**SPY return over test window:** +104.96%

## Why this run

The v1 backtest (train through 2023, test 2024-2026) tested on a window
that contained no real bear market — SPY +53% with no -10% drawdowns
longer than a week. That made it impossible to evaluate whether the
classifier could detect actual bears.

v2 trains earlier so the test window contains the **complete 2022 bear
market** (SPY -19%) plus the 2024-2026 melt-up. We can now ask: does the
classifier detect real bears, and does it stay calibrated across regime
changes?

## Headline numbers

| Metric                                 | Strategy    | Passive SPY |
|----------------------------------------|-------------|-------------|
| Cumulative return over test window     | **+42.99%** | **+104.96%**|
| Sharpe (annualized)                    | 0.80        | 0.87        |
| Sharpe rough SE (n=1328)               | ±0.43       | ±0.43       |
| Max drawdown                           | -13.04%     | (~-25%)     |
| Mean size multiplier                   | 0.51        | (n/a)       |

Cumulative gap is again mostly mechanical (avg 0.51 sizing → 0.51× of SPY's
return). Risk-adjusted (Sharpe) the two are within one standard error.

## Per-regime breakdown — vs unconditional baseline

Unconditional baseline on test window (n=1328): mean fwd_20d **+1.18%**,
neg_rate **32.91%**.

| Label                | n_days | mean_fwd_20d | neg_rate | size_mult |
|----------------------|--------|--------------|----------|-----------|
| BEAR_TREND           | 1151   | +1.26%       | 31.6%    | 0.5       |
| BULL_TREND           | 151    | **+0.21%**   | **59.6%**| 1.0       |
| VOLATILE_TRANSITION  | 51     | +1.24%       | 39.2%    | 0.5       |
| CRISIS (Layer 0)     | 9      | +6.93%       | 15.7%    | 0.0       |

**Two striking findings:**

1. **BEAR_TREND fires 85% of days regardless of regime.** Across 2021-2026
   it fires 1151/1362 days. Within those days the fwd_20d distribution
   matches the unconditional baseline almost exactly — no detectable
   directional signal from BEAR_TREND.

2. **BULL_TREND is *anti-correlated* with positive returns on this
   window.** The 151 days the model said BULL_TREND had a mean fwd_20d
   of +0.21% (vs unconditional +1.18%) and a 60% negative-day rate (vs
   unconditional 33%). At n=151 this is no longer a small-sample issue —
   the BULL signal on this window is *worse than no signal*.

## Per-year breakdown — the smoking gun

| Year | unconditional fwd_20d | n_BEAR | n_BULL | n_TRANSITION | n_CRISIS |
|------|------|------|------|------|------|
| 2021 | +1.78% | 214 | 31 | 7 | 0 |
| 2022 | **-0.84%** | 231 | **12** | 5 | 3 |
| 2023 | +1.92% | 220 | 25 | 5 | 0 |
| 2024 | +1.41% | 230 | 17 | 4 | 1 |
| 2025 | +1.91% | 164 | **60** | 21 | 5 |
| 2026 YTD | +0.61% | 92 | 6 | 9 | 0 |

The 2022 BEAR-dominance looks right (231 BEAR on a -19% year), but the
classifier's BEAR firing rate is essentially constant year over year
(85-92%). Even in 2023-2024-2026 (clearly bull years), BEAR fired ≥85%
of days. **The classifier doesn't distinguish bear from bull years; it
labels almost everything BEAR all the time.**

Worth zooming in on 2022 BULL_TREND days specifically:
- n = 12
- mean fwd_20d = **-5.99%**
- neg_rate = **100%**

Every single one of the 12 days the model said BULL_TREND during the
2022 bear market was followed by a negative 20-day SPY return. That's
a contrarian signal at small n — when the classifier's "I see a bull
opportunity in the middle of a bear market" intuition fires, the next
20 days have been catastrophically wrong. (n=12 is too small to call a
real finding, but it's suggestive enough to flag.)

## What v2 confirms vs what it changes from v1

**Confirms:** the classifier's BEAR_TREND label is essentially a constant
"market down" prior, not a regime-conditional signal. The mean forward
return inside BEAR_TREND tracks the unconditional baseline within noise.

**Changes:** v1 saw BULL_TREND as "weak positive signal" (+0.98% mean,
62% hit rate at n=37). v2 with more samples (n=151) and a tougher window
sees BULL_TREND as **anti-signal** (mean +0.21%, neg_rate 60%). The v1
finding was small-sample noise.

## How this connects to production HMM

Production HMM (id=1, trained on full 2015-2026) currently labels recent
days BULL_TREND with high confidence — opposite of what v1/v2 OOS models
do. The same in-sample evaluation problem applies: we cannot test
production HMM honestly on data it was trained on.

The paper-trading benchmark window (`benchmark_windows.id=1`, started
2026-05-16) is the only way to measure production HMM forward.

## Caveats

- hmmlearn aborted training at iter 3 again on the v2 fit (same
  `spy_above_50dma` binary-feature ill-conditioning). The state means
  are sensible but the model wasn't fully optimized.
- Test window is 1362 days but contains only one real bear market
  (2022); the BULL anti-signal finding could be an artifact of one
  specific market regime, not a generalizable failure mode.
- This is still a "single fixed split" backtest. The companion
  `walkforward_v3_rolling/` runs proper rolling walk-forward (refit each
  year, 10 splits) for a fully OOS series — see that REPORT.md for the
  longest-window numbers.

## Artifacts

- `walkforward_per_day.csv` (1362 rows)
- `walkforward_by_regime.csv`
- `walkforward_headline.json`
