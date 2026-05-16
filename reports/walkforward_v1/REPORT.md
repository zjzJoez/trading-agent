# OOS Walk-Forward Backtest v1 — Honest Findings

> **Update 2026-05-16:** This v1 used a narrow 2-year test window and is
> superseded by v2 (`../walkforward_v2/REPORT.md`, 1362-day test including
> 2022 bear) and v3 (`../walkforward_v3_rolling/REPORT.md`, rolling
> walk-forward over 2017-2026, 2367 OOS days). The v3 numbers are the
> credible read; this v1 document is preserved as the first-pass record.


**Date run:** 2026-05-16  
**Backtest HMM:** `regime_model_versions.id=2`, status=shadow  
**Train window:** 2015-03-31 → 2023-12-29 (2204 snapshots, cutoff 2023-12-31)  
**Test window:** 2024-01-01 → 2026-05-15 (609 OOS snapshots; 575 with full 20-day forward)  
**SPY return over test window:** +52.97%

## What this backtest validates (and what it doesn't)

**Validates:** the *deterministic spine* — Layer 0 (crisis overlay) + Layer 1
(HMM state inference) + the gate that maps regime labels to size multipliers.

**Does NOT validate:** the LLM trader, the bull/bear analyst debate, the risk
council, options pricing/greeks, or any tradeable PnL. The "strategy" line in
the headline number is a simplified proxy:

    strategy_ret(d) = size_mult(label(d)) * SPY_next_1d_ret(d)

That isolates regime-gate alpha from everything downstream. It is *not* what
the live system actually trades.

## Headline numbers

| Metric                                 | Strategy   | Passive SPY |
|----------------------------------------|------------|-------------|
| Cumulative return over test window     | **+22.55%**| **+52.97%** |
| Sharpe (annualized)                    | 1.18       | 1.19        |
| Annualized-Sharpe rough SE (n=575)     | ±0.66      | ±0.66       |
| Max drawdown                           | -13.04%    | (not shown) |
| Mean size multiplier across test       | 0.51       | (n/a)       |

The 30-point gap in cumulative return is mostly **mechanical**, not a
classifier-specific failure. With an average size multiplier of 0.51, the
proxy strategy is essentially 51% long SPY; getting roughly half of SPY's
return is the expected output of constant 51% sizing. The classifier-specific
claim is in the per-regime conditional statistics below, not in the headline.

The annualized-Sharpe rough SE of ±0.66 means the two Sharpes (1.18 vs 1.19)
are well within one standard error of each other — but it also means *almost
any* Sharpe in [0.5, 1.8] would be statistically indistinguishable on this
sample size. This is a **low-power test**; treat the Sharpe comparison as
"no evidence of a difference" rather than "we proved they are the same."

## Per-regime breakdown — vs the unconditional baseline

Unconditional baseline on the test window (n=575): mean fwd_20d_ret = **+1.62%**,
fraction of days with negative fwd_20d_ret = **26.8%**.

| Label                | n_days | mean_fwd_20d | neg_rate_fwd_20d | size_mult | signal? |
|----------------------|--------|--------------|------------------|-----------|---------|
| **BEAR_TREND**       | 491    | +1.56%       | 26.5%            | 0.5       | none — matches unconditional |
| BULL_TREND           | 37     | +0.98%       | **37.8%**        | 1.0       | inverse — worse than unconditional |
| VOLATILE_TRANSITION  | 41     | +1.63%       | 24.4%            | 0.5       | none — matches unconditional |
| CRISIS (Layer 0)     | 6      | +9.80%       | 0.0%             | 0.0       | n too small to read |

**Honest read:** the deterministic regime labels (on this OOS window) carry
**no detectable directional signal** about forward 20-day SPY returns. The
BEAR_TREND label, despite firing 519 of 609 days (85.2%), produces an
identical forward-return distribution to the unconditional baseline. The
BULL_TREND label actually has a *higher* negative-day rate than the
unconditional (38% vs 27%), though n=37 makes that comparison fragile.

The Layer 0 CRISIS overlay flagged 6 days, all of which had positive
forward 20-day returns (mean +9.8%) — qualitatively this looks like Layer 0
caught transient pullbacks followed by mean reversion, but n=6 is too small
to call a real finding.

## Why might this have happened?

The HMM was trained 2015-2023 and learned that "low breadth +
spy_above_50dma near zero" = state 2 (mapped to BEAR_TREND in calibration).
In 2024-2025 the market featured:

- Persistently narrow leadership (Mag-7 dominance) → low
  `breadth_above_20dma_pct`
- SPY itself often above 50-DMA, but individual sectors below their 50-DMAs
- VIX moderate (~15-22) with periodic spikes

The HMM saw these features and assigned them to its 2018/2022-style "bear"
cluster from training. But underneath, SPY itself kept rallying — the
2018-2022 features stopped tracking SPY direction in 2024-2025.

This is a **regime-shift in regime-detection features** — a meta-failure
mode that no amount of in-sample fit fixes.

**Caveat on the OOS model itself:** hmmlearn aborted training at iter 3 of
200 because the log-likelihood *decreased* (`Delta is -9074.50`), driven by
the binary feature `spy_above_50dma` causing full-covariance
ill-conditioning. State separation in the Viterbi decode is still clean
(occupancies 5.3% / 59.2% / 34.4% / 1.1%; feature means are clearly
distinct), so the per-state means we reported are real. But the strict
claim is "this non-converged model detected no signal" — not "no signal
exists in these features." A properly-converged HMM with the binary
features removed or replaced (e.g. with continuous "price minus
50DMA, normalized") might separate forward returns better.

## How this connects to the LIVE production HMM

Critical context: this OOS backtest used the *shadow* HMM (`id=2`, trained
through 2023). The **production HMM (`id=1`, trained on the full
2015-2026 history)** is currently labeling recent days differently:

    as_of      | label       | confidence
    -----------|-------------|-----------
    2026-05-15 | BULL_TREND  | 1.00
    2026-05-14 | BULL_TREND  | 0.50
    2026-05-13 | BULL_TREND  | 0.50
    2026-05-12 | BULL_TREND  | 0.50
    2026-05-11 | BULL_TREND  | 0.50

So the OOS finding ("classifier sees BEAR on a bull market") **does not
transfer 1:1 to production behavior**. The production HMM has *seen* the
2024-2026 bull data and fits it — but that means we cannot use the same
2024-2026 window to honestly measure production-HMM quality. This is the
in-sample evaluation problem the backtest was designed to dodge.

The honest summary:
- **OOS HMM** (no future leakage): no detectable signal on this window
- **Production HMM** (trained through today): classifies recent days
  BULL_TREND — could be "correctly fit" or "memorizing", we can't tell
  from a self-test

The forward-looking benchmark window (Option B, started 2026-05-16) is the
only way to break that tie.

## What to do about it (not in scope for this round)

Hypotheses to test in future work:

- Add Mag-7 leadership concentration as a feature (current `breadth` proxy
  misses this)
- Retrain HMM with a rolling 5-year window (not full 2015-2026 history)
- Make `BEAR_TREND` require *both* breadth AND a price-action signal
- Add a "trend confirmation" gate: if SPY itself is up 60 days, downgrade
  BEAR_TREND to VOLATILE_TRANSITION
- Run proper rolling walk-forward (refit at each year-end) to get a fully
  out-of-sample series across the entire 2017-2026 window

None of these are blocking for the paper-trading benchmark window. That
window will produce forward data with the production HMM, against which a
future iteration of this report can be compared.

## Honesty note

This is the kind of result where it would be tempting to tweak metric
definitions, cherry-pick a sub-window, or run more seeds until something
stat-sigs. I did none of that. The CSV is the raw output of one run with
pre-registered metrics. If we want to test a different hypothesis, that's
a fresh run with its own pre-registered design — not a post-hoc filter on
this one.

## Artifacts

- `walkforward_per_day.csv` — one row per OOS day with label, confidence,
  size_mult, and forward 1d/5d/20d SPY returns (609 rows)
- `walkforward_by_regime.csv` — per-label aggregate statistics
- `walkforward_headline.json` — top-line summary

## Companion: paper-benchmark window started

Separately, `benchmark_windows.id=1` was inserted on 2026-05-16 to begin a
6-8-week forward measurement of the live system vs SPY + 60/40 SPY/TLT
baseline, using the *production* HMM (id=1). Daily marks are recorded by
the `record_benchmark_mark` cron at 21:30 UTC on trading days.

T0 state: NAV = $1,011,058.54, SPY = $739.17, TLT = $83.66.
