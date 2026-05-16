# Walk-Forward v4.1 — Calibration Fix (4-regime auto-cal matching production)

**Date run:** 2026-05-17  
**Headline:** v4 used 3-regime auto-calibration (no RANGE_LOW_VOL). That
was systematically biased against the strategy. Fixed: rolling auto-cal
now emits all 4 production regimes by splitting middle states on training-
window mean RV. Sharpe moves from 0.78 → **0.80 ± 0.011**. Gap to SPY
narrows from -0.06 → **-0.04**. Multi-seed std also tightens (0.015 → 0.011).
The defensive role (+10.4pp in 2022) is unchanged.

## What was wrong with v4

Production HMM (`regime_model_versions.id=4`) uses 4 regime labels with
different size multipliers:

| Regime | size_mult |
|---|---|
| BULL_TREND | 1.00x |
| RANGE_LOW_VOL | 0.75x |
| BEAR_TREND | 0.50x |
| VOLATILE_TRANSITION | 0.50x |
| CRISIS | 0.00x |

The v4 rolling auto-calibration only emitted 3 labels:
- Lowest training-mean spy_ret_20 → BEAR_TREND
- Highest → BULL_TREND
- **Both middle states → VOLATILE_TRANSITION** (0.50x)

So the "calm-bull state" (low return, low RV, the production RANGE_LOW_VOL)
was sized at 0.50x in the backtest but 0.75x in production. Systematic
under-sizing of the most-frequent-during-calm-markets state.

## The fix

`scripts/rolling_walkforward.py:train_hmm_for_year`:

```python
if n_states == 4:
    state_to_label[ranked[0]]  = "BEAR_TREND"                # lowest ret
    state_to_label[ranked[-1]] = "BULL_TREND"                # highest ret
    # middle 2 split by mean training-window spy_rv_20:
    mid_a, mid_b = ranked[1], ranked[2]
    if rv_a <= rv_b:
        state_to_label[mid_a] = "RANGE_LOW_VOL"      (0.75x)
        state_to_label[mid_b] = "VOLATILE_TRANSITION" (0.50x)
    else:
        state_to_label[mid_a] = "VOLATILE_TRANSITION"
        state_to_label[mid_b] = "RANGE_LOW_VOL"
```

Still uses only training-window features (zero leakage). Now matches
production's calibration semantics.

## Results — 5-seed multi-seed test

| Seed | Strategy Sharpe | Cum Strategy | Avg size_mult |
|---|---|---|---|
| 42 | 0.818 | +84.6% | 0.531 |
| 7 | 0.791 | +81.1% | 0.531 |
| 99 | 0.799 | +82.2% | 0.531 |
| 123 | 0.800 | +82.4% | 0.532 |
| 2026 | 0.793 | +78.0% | 0.512 |
| **mean ± std** | **0.800 ± 0.011** | +81.7% ± 2.6pp | 0.527 |
| **SPY (baseline)** | 0.841 | +261.8% | n/a |

| Metric | v3 (broken) | v4 (3-regime) | **v4.1 (4-regime)** | SPY |
|---|---|---|---|---|
| Strategy Sharpe | 1.02 | 0.78 ± 0.015 | **0.80 ± 0.011** | 0.84 |
| Sharpe Δ | +0.18 (false) | -0.06 | **-0.04** | — |
| Avg size_mult | 0.604 | 0.505 | **0.527** | — |
| Cum strategy | +133.4% | +74.4% | **+81.7%** | +261.8% |

The avg size_mult went from 0.505 → 0.527 — the gate is letting more
through, consistent with RANGE_LOW_VOL bumping the size up from 0.50 →
0.75 on the days that qualify.

## Per-year breakdown (seed=42)

| Year | SPY ret | Strategy ret | Δ (pp) | Notes |
|---|---|---|---|---|
| 2017 | +21.6% | +10.5% | -11.2 | All TRANSITION (3-regime cal had it the same; few RANGE_LOW_VOL days made it through the new rule) |
| 2018 | -5.1% | -5.9% | -0.7 | Mostly TRANSITION |
| 2019 | +32.3% | +17.8% | -14.5 | 35 BULL days |
| 2020 | +15.6% | +4.4% | -11.2 | COVID — gate works as intended |
| 2021 | +31.3% | +14.3% | -17.0 | Mostly TRANSITION |
| **2022** | **-19.0%** | **-8.6%** | **+10.4** | Defensive role intact |
| 2023 | +26.0% | +12.4% | -13.6 | All TRANSITION |
| 2024 | +25.3% | +12.1% | -13.2 | (-0.5pp vs v4: marginal RANGE_LOW_VOL upgrades) |
| 2025 | +18.2% | +3.2% | -15.0 | |
| 2026 YTD | +3.3% | +2.5% | -0.8 | |

The 2024 / 2025 results barely moved. The gate's "look bearish, downsize"
behavior is driven by the BEAR_TREND state (still 0.50x), not by
RANGE_LOW_VOL. So fixing the RANGE_LOW_VOL gap helps marginally but doesn't
fundamentally change the conclusion.

## What's confirmed vs what's still open

**Confirmed (consistent across v4 and v4.1):**
- No detectable Sharpe edge over 10 years OOS
- 2022 drawdown protection ~+10pp is robust across seeds
- The strategy gives up ~5-6pp/year of beta-driven return for that protection
- Multi-seed stability is tight (std 0.011)

**Still open:**
- LLM agent group has NEVER been backtested (LLM training contamination
  makes it structurally impossible to backtest historically — see v4 REPORT)
- Only forward paper benchmark or shadow-proposal log can address this
- Shadow proposal infrastructure: WIP, see `migrations/008_shadow_proposals.sql`

## Honesty note: this is the LAST regime-gate backtest

After v1, v2, v3, v4, v4.1 — the headline number has wandered between
+0.18 Sharpe edge (broken), -0.06 (proper v4), -0.04 (calibration fixed).
At this point the message is clear:

> The deterministic regime gate is a defensive overlay, not an alpha
> generator. Whether the full LLM-powered system has alpha is a separate
> question that needs forward data, not more backtest iterations.

The next round of evaluation work should focus on **LLM behavior** (via
Shadow Proposal Log + behavior consistency tests), not on retuning the
regime gate.
