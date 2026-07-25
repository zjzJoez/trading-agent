# M1-0.4 — Gate-feasibility replay + managed-payoff expectancy for `credit_vertical_index_30_45`

**Date:** 2026-07-25 · **Branch:** `impl/vertical-replay` · **Sleeve:** short put verticals, SPY/QQQ, 30–45 DTE

Every number below is reproducible from a committed artifact in this directory:

| artifact | what it is |
| --- | --- |
| `summary.json`, `entries.jsonl` | **PRIMARY** run — `mark=smile`, `liquidity_lag=prior`, `width_policy=five-first`, `credit_floor_frac=0.25` |
| `sensitivity/entries_*.jsonl` | one complete independent replay per convention (indexed from `summary.json .sensitivity.rows`) |
| `as_reported/summary.json`, `as_reported/entries.jsonl` | **DISCLOSED UPPER BOUND** — `mark=close`, `liquidity_lag=same-day`: the exact configuration whose numbers were rejected, kept auditable |
| `diagnostics.json` | the adversarial checks for both samples (`scripts/replay_vertical_diagnostics.py`) |

Regenerate with `scripts/replay_vertical_gates.py` and `scripts/replay_vertical_diagnostics.py`; both are committed, offline, and deterministic per seed.

---

## 1. VERDICT UP FRONT

### The plan's criterion was NOT EVALUATED AS SPECIFIED

`docs/REVIVAL_PLAN_2026-07-20.md` line 81 (M1-0.4):

> 验收:**允许交易的 regime 内合格 vertical 存在于 ≥60% 快照日**

The criterion is **conditional on the regimes the sleeve is allowed to trade**. This replay carries **no regime labels**. It measured availability **unconditionally** over all planned entry days. An unconditional rate is neither the criterion nor a bound on it — the conditional rate could be higher (if the allowed regimes are the high-IV ones where credit is easiest to find) or lower.
Recorded in `summary.json .verdict` (`evaluated_as_specified: false`, `regime_labels_present: false`).

**M1-0.4 therefore remains OPEN. It cannot be signed off from this artifact.**

### What WAS measured

Availability of a qualifying vertical, unconditionally, over **144 planned (entry_date, symbol) pairs** (SPY + QQQ, 2025-10 → 2026-07):

| denominator | definition | primary (`smile`/`prior`) | upper bound (`close`/`same-day`) | ≥60%? |
| --- | --- | --- | --- | --- |
| **planned = 144** | every planned entry day; data gaps charged against availability | 26/144 = **0.1806** | 44/144 = **0.3056** | **FAIL / FAIL** |
| **data-adequate = 119** | entries whose *fetched* strikes could construct ≥1 vertical | 26/119 = **0.2185** | 44/119 = **0.3697** | **FAIL / FAIL** |
| marks-present = 123 | ≥1 usable entry-day option print | 0.2114 | 0.3577 | FAIL / FAIL |
| data-adequate **and** delta-band bracketed | the only denominator where the gate was actually *asked* the question | 26/56 = 0.4643 | 44/63 = 0.6984 | FAIL / *pass* |

The floor fails on **both** headline denominators under **both** marks. It is cleared only on the narrowest denominator (63 entries) under the mark convention this report shows to be an upper bound — that is not a pass, and it is not the criterion.

**De-noised availability is 0.227, or 0.2185.** Section 3 shows the `close` mark's selected credits sit −1.23 in-band-null sd above a cross-leg-consistent estimate. Under the cross-leg-consistent mark availability is 26/119 = 0.2185 (`prior` liquidity screen) or 27/119 = 0.2269 (`same-day`, `summary.json .sensitivity.rows["liquidity-same-day"]`). **The 0.3697 previously reported is not the honest figure.**

### The retune claim collapses

`summary.json .availability.credit_floor_sensitivity` (primary): 0.25 → 0.2185, **0.225 → 0.5714**, 0.20 → 0.8235.
Under the upper-bound mark, 0.225 → 0.6807 on the 119 denominator but only 81/144 = 0.5625 on the 144 denominator. "Just lower the credit floor to 0.225" therefore reaches 60% on **neither** denominator under the honest mark. Reaching it requires 0.20 — a 20% credit floor, i.e. abandoning the spec's `credit >= width/4` — and Section 2 shows the floor margin is the same size as the measurement noise, so that retune would be fitting noise.

---

## 2. THE HEADLINE SCIENTIFIC RESULT: identification failure

**Daily trade-print aggregates cannot resolve a ±0.05R edge on \$5-wide index verticals.**

A vertical's credit is a **difference of two legs**. In a daily aggregate each leg's "close" is its own *last trade of the day*, struck at a different minute. The difference therefore carries the full leg-vs-leg timing gap — and on a \$5 width that gap is the same size as the quantity being measured.

### 2a. Mark-convention sensitivity — same data, same gates, same walk

`summary.json .sensitivity.rows`, all at the spec floor 0.25, `liquidity_lag=prior`, `width_policy=five-first`:

| mark | availability (119 / 144) | n | WR | mean R | median R | LB95 (week) | LB95 (exposure) | maxDD R | PT / 21-DTE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **smile** (PRIMARY, cross-leg consistent) | 0.2185 / 0.1806 | 26 | 0.6923 | **−0.0455** | +0.0950 | −0.1581 | −0.0911 | 2.431 | 12 / 14 |
| **close** (disclosed upper bound) | 0.3613 / 0.2986 | 43 | 0.7674 | **+0.0524** | +0.1412 | −0.0280 | −0.0174 | 2.377 | 32 / 11 |
| vw (volume-weighted average price) | 0.4538 / 0.3750 | 54 | 0.9444 | **+0.1554** | +0.1657 | +0.1185 | +0.1310 | 0.884 | 51 / 3 |
| hl2 ((high+low)/2) | 0.3025 / 0.2500 | 36 | 0.6944 | **+0.0118** | +0.1190 | −0.0702 | −0.0277 | 1.679 | 19 / 17 |

Four defensible readings of "the price that day" span **mean R from −0.0455 to +0.1554 — 0.200R** — and availability from 0.2185 to 0.4538. The spread of that column is **four times the 0.05R the sleeve claims**. The headline result is a property of the mark convention, not of the strategy.

`smile` is primary because it is the only cross-leg-consistent convention: both legs are repriced off ONE same-day IV curve (weighted quadratic in moneyness K/S, ≥5 strikes with ≥10 trades, sqrt(trades) weights). It fitted 1,897 day-curves, left 364 days unfittable, repriced 16,075 bars and kept 1,904 raw (`summary.json .mark_diagnostics.smile`). No bar was interpolated or carried forward under any convention.

### 2b. The null study — the noise IS the edge

`diagnostics.json .samples.*.null_study`. Every liquid \$5-wide adjacent put pair, on every day, in every fetched chain, **no gate applied** — raw close-difference credit vs the same-day curve value:

- **n = 11,842**, mean **+0.0014**, median +0.0005, **sd 0.2594**, p10 −0.1361, p90 +0.1335
- restricted to the sleeve's own window (DTE 30–45, short |delta| 0.20–0.35): n = 1,864, mean +0.0047, median +0.0028, sd **0.2434**

The estimator is **unbiased** — its mean is 0.5% of its own sd — so its dispersion is a clean measurement of this data source's credit noise:

> **sd 0.2594 on a \$5 width = 5.19 percentage points of width.**

The spec's credit-floor margin (0.25 versus the market's measured median credit/width of 0.188, Section 6) and the sleeve's entire claimed edge (0.03–0.05R) are **the same size as that noise**. No configuration of this data source resolves them.

### 2c. Two absurdities the raw marks produced

`diagnostics.json .samples.as_reported_close_sameday.exit_trigger_audit`. Of 34 profit-takes testable against a same-day curve, **9 (26.5%)** are exits the curve says never happened. Two of them:

1. **A profit-take booked on a flat day.** 2025-10-13 QQQ 590/585, credit 1.32, exit 2025-10-15. The underlying moved **602.01 → 602.22 (+0.03%)** — two sessions, essentially nothing. The raw close-difference mark printed **0.620**, under the 0.705 profit-take level, and the trade booked **R = +0.1371**. The same-day curve puts that spread at **1.329** — roughly *twice* the trigger level. Nothing happened; a print pair moved.

2. **A profit-take booked on a day the underlying fell 588.00 → 573.79.** 2026-03-23 QQQ 570/565, credit 1.32, exit 2026-03-26, a **−2.42%** day — the wrong direction for a short put spread. The raw mark printed **0.010**: a \$5-wide put spread marked at one cent, against a 0.660 trigger. Booked **R = +0.1093**. The same-day curve says **1.572**.

Under the primary `smile` mark, curve-denied profit-takes number **0 of 12** (`diagnostics.json .samples.primary_smile_prior.exit_trigger_audit`) — the internal-consistency check the primary convention must pass, and does.

A related but confounded test — freezing each leg's entry-day IV and revaluing at the exit-day spot — flags **27 of 34** profit-takes. That mixes in IV mean-reversion between entry and exit, so **9/34 is the defensible figure** and 27/34 is an upper bound.

---

## 3. THRESHOLD SELECTION BIAS

`diagnostics.json .samples.as_reported_close_sameday.selection_bias`. The same null statistic, evaluated on the **44 gate-selected entries** instead of on all pairs:

| sample | n | mean | median | sd |
| --- | --- | --- | --- | --- |
| ungated null (all liquid 5-wide pairs) | 11,842 | +0.0014 | +0.0005 | 0.2594 |
| in-band null (DTE 30–45, \|delta\| 0.20–0.35) | 1,864 | +0.0047 | +0.0028 | 0.2434 |
| **gate-selected entries** | 44 | **−0.2946** | −0.2835 | 0.2411 |

Shift versus the like-for-like in-band null: **−0.2993 spread-dollars = −5.99 percentage points of a \$5 width = −1.23 in-band-null sd** (−1.15 ungated sd; both are emitted so neither can be cherry-picked).

Read plainly: **the credit floor filters the noise, not the market.** A day qualifies when its raw print pair happens to be wide. Two one-directional consequences:

1. **Availability is biased UP.** The 0.3697 figure counts days that qualified because their prints were lucky — which is why availability collapses to 0.2185 the moment the marks are made cross-leg consistent.
2. **Booked credit carries ≈ +0.084R of pure measurement bias** (`implied_measurement_bias_in_credit_r = 0.0842`) — **larger than the entire reported edge** of +0.0561R. Under the primary mark the same statistic is −0.0047 dollars = −0.02 null sd, i.e. gone by construction.

---

## 4. SURVIVORSHIP

`diagnostics.json .samples.*.survivorship`. Entries dropped from the availability denominator are **not** a random subsample:

| group | n | trailing-20 RV (mean / median) | forward-10 RV (mean) |
| --- | --- | --- | --- |
| dropped (not data-adequate) | 25 | **0.1263** / 0.1322 | 0.1494 |
| kept (data-adequate) | 105 | **0.1594** / 0.1538 | 0.1595 |
| — of which qualified | 39 | 0.1728 / 0.1743 | 0.1795 |
| — of which gate-rejected | 66 | 0.1514 / 0.1423 | 0.1477 |

**Welch t (dropped − kept) = −4.445.** Dropping is strongly vol-correlated, and in the direction that flatters the result: the days whose option data was never fetched are the **low-vol** days — exactly the days a credit floor is hardest to clear. Removing them inflates availability.

**The conservative bound is therefore 44/144 = 0.3056** for the upper-bound mark and **26/144 = 0.1806** for the primary. Charging every data gap against availability is the only treatment the fetch pattern cannot bias upward.

The gaps themselves: of 74 planned (symbol, expiry) batches, **7 were never fetched** and **4 contain no bars at all**; 543 contract rows, 511 with bars; zero unparsable lines, zero bad bars, zero mis-routed call rows in what was fetched (`summary.json .data_inventory`).

---

## 5. WHAT SURVIVES AND IS USEFUL

### 5a. The look-ahead audit is clean — with exactly one exception, now fixed

Confirmed clean by re-derivation:

- **Selection reads only the entry day.** Spot, every leg mark, the IV inversion and the delta all come from the entry date's bar. This is a close-to-close model (enter at the close) — a convention, not future information.
- **The walk moves strictly forward** (`entry_date < d <= expiry`); the forced-exit date is `expiry − 21d`, known at entry.
- **No day was silently skipped.** Across all 44 upper-bound trades, trading days inside a held window with no walkable leg pair: **0**; reported `skipped_days`: **0** (`diagnostics.json .*.walk_coverage`). Nothing is interpolated, carried forward or synthesized anywhere in the pipeline; every skip has a counter.
- **`data_end` exits: 0** in both samples. No trade's result depends on the dataset ending.
- **`credit_floor_sensitivity` is exact** at and below the floor the run used, and refuses to project above it.
- **The bootstrap is deterministic** per seed (tested), and now reports two blocking schemes rather than one.

The one exception, found and fixed: **the `n >= 30` liquidity screen read the entry day's FULL-DAY trade count**, which does not exist until the close. The screen now defaults to the last session *strictly before* entry (`liquidity_lag=prior`). Measured cost of removing the look-ahead: 44 → 43 trades under `close`, 27 → 26 under `smile`; availability 0.2269 → 0.2185. **Small, and implementable** — it did not need to be documented as unfixable.

### 5b. The payoff SHAPE is clean and informative

In the upper-bound sample the shape is **perfectly separated**: 34 profit-takes, **all** winners (R +0.1093 … +0.2945); 10 forced 21-DTE closes, **all** losers (R −0.5694 … −0.0596); 0 `data_end`. Mean R by exit: PT **+0.1639**, 21-DTE **−0.3103**.

That is the mechanical signature of a managed short-premium sleeve, not a data artifact. Under the primary mark the separation loosens — 12 PTs all winners, but 14 forced closes of which 6 were still positive (max +0.1046) — which is itself informative: the perfect separation was partly the raw marks' doing.

### 5c. The −1R tail is UNSAMPLED, not absent

Worst booked R: **−0.5694** (upper bound), **−0.6157** (primary). Occupied histogram bins run −0.7 … +0.3, with **underflow 0 and overflow 0**. A full −1R requires the short strike still breached at the forced 21-DTE exit, which never occurred in this 9-month window. **Its absence is a property of the window, not of the strategy.** Any sizing that treats the observed tail as the real tail is under-reserved.

### 5d. Drawdown attribution — stated correctly

`diagnostics.json .samples.as_reported_close_sameday.payoff_attribution`. Whole sample: n = 44, mean **+0.0561R**, median +0.1420, sum +2.4697R, **max drawdown 2.3774R** — the drawdown is 96% of the total gain.

- Excluding the **5 worst** trades, the other **39 average +0.1243R** (dropped: −0.5694, −0.5123, −0.4520, −0.4256, −0.4181).
- Excluding **all of 2026-03**, the other **33 average +0.1150R** (2026-03 alone: n = 11, mean −0.1203R).

**Both exclusions are illegitimate as stated, and the second is worse:** 2026-03 has the **highest unconditional median IV of any month in the sample (0.2755)**, and `payoff_attribution` flags it `is_highest_iv_month: true`. Deleting it removes precisely the regime a short-premium sleeve exists to harvest — and the regime in which it takes its losses. It is not a neutral operation, and no "excluding X" figure may be quoted as the strategy's expectancy.

### 5e. Unconditional per-month IV, with n

`diagnostics.json .*.iv_by_month`. **Unconditional** = every liquid invertible in-band strike on every planned entry day at DTE 30–45, regardless of whether it qualified. The conditional column is qualified trades only, selected on credit and therefore on IV — it cannot be read as the month's vol level.

| month | n strikes | n entry days | **unconditional** median IV | mean IV | conditional median IV (n) |
| --- | --- | --- | --- | --- | --- |
| 2025-10 | 62 | 16 | 0.2073 | 0.2074 | 0.2142 (6) |
| 2025-11 | 61 | 15 | 0.2321 | 0.2331 | 0.2218 (8) |
| 2025-12 | 37 | 13 | 0.1736 | 0.1804 | 0.2079 (2) |
| 2026-01 | 42 | 12 | 0.1954 | 0.1892 | 0.1855 (1) |
| 2026-02 | 56 | 16 | 0.2271 | 0.2340 | 0.2514 (6) |
| **2026-03** | 71 | 16 | **0.2755** | 0.2750 | 0.2755 (11) |
| 2026-04 | 71 | 16 | 0.2293 | 0.2242 | 0.2255 (6) |
| 2026-05 | 23 | 6 | 0.1904 | 0.2040 | 0.2252 (1) |
| 2026-06 | 20 | 5 | 0.2650 | 0.2424 | 0.2646 (3) |

### 5f. The block bootstrap was too tight

`diagnostics.json .*.blocking`. Entry-ISO-week blocking gives **25 blocks** for 44 trades (sizes 1–3). But the median holding period is **5.5 calendar days, max 18** — a one-week block cannot contain an 18-day exposure, and **51 of 946 trade pairs are open simultaneously while sitting in different blocks**, i.e. resampled as independent draws.

Exposure-cluster blocking (connected components of overlapping `[entry, exit]` intervals) gives **12 blocks** for the same 44 trades (largest 11). Both are now reported:

| sample | LB95 (entry-week) | LB95 (exposure-cluster) | blocks (week / cluster) |
| --- | --- | --- | --- |
| upper bound (44 trades) | −0.0277 | −0.0174 | 25 / 12 |
| primary (26 trades) | −0.1581 | −0.0911 | 15 / 7 |

Coarser blocks happen to move LB95 *up* in this sample (fewer, larger blocks reduce resampling variance here) — but a 5th percentile drawn from 12 blocks is so granular it should not be read as a confidence statement at all. **Every LB95 in this study is negative.**

### 5g. The friction defect this study surfaced — the most actionable output

`summary.json .friction_comparison`, `as_reported/summary.json .friction_comparison`, `diagnostics.json .*.friction`. Measured on the 44 upper-bound trades at the M1-0.3 index spreads (SPY 0.49%, QQQ 0.94% of mid):

| model | friction, mean R | mean \$ |
| --- | --- | --- |
| **replay** — spread charged on all four realized leg marks | **0.0486** | 17.31 |
| `execution_costs.friction_r` **as deployed** — `4 × half_spread(net_credit)` | **0.0176** | 6.25 |
| the same 2-legs × 2-directions stack on the **actual** leg marks | **0.0514** | 18.29 |

**Ratio replay / deployed: mean 2.74×** (min 1.63×, max 5.56×).

The cause is in the deployed function's own docstring. `friction_r` proxies each leg's mark by `net_credit`, i.e. it assumes `leg-mid-sum ≈ 2 × net_credit`, exact only at wing ratio `long/short = 1/3`, which the docstring calls "the typical 30-45 DTE vertical". Measured here:

- **wing ratio `long_mark/short_mark`: mean 0.849, median 0.854** — not 0.333.
- **`leg_mid_sum / net_credit`: mean 12.64, median 12.66** — the model assumes **2.0**.

So on SPY/QQQ \$5-wides at |delta| 0.20–0.35 and 30–45 DTE the deployed model **understates the spread leg of friction by ≈ 6.3×**. This independently confirms, from daily aggregates on SPY/QQQ, the same defect measured from real IWM quotes (wing ratio 0.731, leg-sum/credit 6.43). It is a **production cost-model bug, not a replay artifact**, and it bites hardest at exactly the 0.03–0.05R margins this sleeve claims.

**Second, worse finding — the friction inputs are not traceable.** The live `data/execution_costs.json` (sha256 `a467989051373cae…`, calibrated 2026-06-12, n_samples 160) contains **no `per_underlying` entry for SPY or QQQ**. Asked today, the deployed model falls back to its single-name global `spread_pct_of_mid_median = 0.07338` for both index names. Charging that on the same booked marks (`summary.json .friction_under_calibration_file`) gives friction of **0.358R** (upper bound) / **0.389R** (primary) — at which point the sleeve is not merely unprofitable, it is unrunnable. The 0.49%/0.94% this replay used come from the M1-0.3 calibration *run*, not from the file; the engine now emits that disagreement as a warning on every invocation. **M1-0.3's per-underlying calibration must be written into `data/execution_costs.json` before any real order.**

### 5h. Width freedom does not rescue the gate

`summary.json .sensitivity.rows["width-any"]`. The \$10-wide branch previously never fired for two reasons: `five-first` breaks after a priced \$5 wing, and an inverted/stale \$5 print used to `break` out of the width loop entirely. The second is now a `continue`, so width freedom is genuinely explored. Measured effect: **none — 26 qualified, mean R −0.0455, identical to `five-first`.** The reason is arithmetic, not code (`diagnostics.json .*.width_census`): **119 of 144 entries do have a constructible \$10-wide pair — the same 119 that have a \$5-wide one** — but the \$10-wide credit/width is *lower* than the \$5-wide (**median 0.1790 over n = 359** vs **0.1880 over n = 408**), so widening makes a 25% floor harder, not easier. Only **20 of 359 (5.6%)** constructible \$10-wides clear the spec floor, against **48 of 408 (11.8%)** \$5-wides. A real degree of freedom that simply does not help.

### 5i. All 7 "no strike in delta band" rejections were data gaps

`diagnostics.json .*.band_bracket_audit`. A per-entry bracket test asks whether the fetched, liquid, invertible strike grid *straddles* [0.20, 0.35]. In the upper-bound sample: **7 rows, 7 classified `band-not-fetched` (class `data`), 0 true gate rejections** — every one with `brackets_delta_band: false`. Examples: 2025-12-26 QQQ (spot 623.89) had a single invertible strike, K=590, |delta| 0.165; 2026-01-26 QQQ (spot 625.46) had 6 invertible strikes, K 520–545, |delta| 0.025–0.046. The strikes that would have answered the question were never fetched. Charging those against the gate was wrong.

---

## 6. WHAT THIS DOES NOT VALIDATE, AND WHAT WOULD

### It does not price the sleeve

This study **cannot** support any of the following, and no re-tuning inside it will:

- an expectancy estimate, a win rate, or a breakeven — the mark convention alone moves mean R by 0.200R, four times the claimed edge;
- an availability figure to compare against the 60% bar — the criterion is regime-conditional and this replay has no regime labels;
- a credit-floor retune — the floor margin equals the credit noise (5.19pp of width), so retuning optimizes noise;
- a −1R tail assumption — the tail is unsampled;
- a friction number for sizing — the deployed cost model understates the spread leg ≈6.3× and its calibration file has no SPY/QQQ entry at all.

What it **does** establish, and these are real deliverables: the identification limit of daily trade-print aggregates for this instrument (§2); the direction and magnitude of the credit-floor selection bias (§3); the survivorship direction (§4); the payoff *shape* under the management rules (§5b); the unconditional IV census (§5e); the correct blocking scheme (§5f); the production friction defect (§5g); and the width finding (§5h).

One more gate-design finding, independent of every measurement problem above (`diagnostics.json .*.width_census`): **credit/width on real SPY/QQQ \$5-wides has a median of 0.1880** over **408** constructible in-band verticals — close to the 0.191 median measured from real IWM quotes. Only **11.8%** of them clear `credit >= width/4`. The spec's floor sits **above the median of what the market offers** in this delta/DTE window. That, not any modelling artifact, is the first-order reason availability is low, and it is the one finding here that no better data source will change.

### Three honest paths, with their time cost

**(a) Accumulate real bid/ask from the now-fixed daily snapshot feed.** `option_chain_snapshots` carries real `bid`/`ask` (2,115 of 2,178 core-ETF rows have a bid; all have an ask), which removes the identification failure at its root: a quoted spread is a same-instant, cross-leg-consistent measurement. Coverage today (read from EC2 Postgres, 2026-07-25): **SPY 5 distinct days (2026-06-22 → 07-23), QQQ 1 day (07-16), IWM 6 days (06-19 → 07-24)** — 12 symbol-days total, following the core-underlyings-first fix (commit `6b01329`).
*Time cost:* at ~21 trading days/month, SPY+QQQ accrues 42 symbol-days/month, so the 144 symbol-days this study used take **≈ 3.4 months** (≈ mid-November 2026); with IWM, 63/month → **≈ 2.3 months**. Reaching 30 trades at the measured 0.1806 planned availability: **≈ 4.0 months** (2 symbols) or **≈ 2.6 months** (3 symbols). **Cost: calendar time only — no dollars, no new code.** Recommended, and it should start accruing regardless of what else is chosen.

**(b) Upgrade the data plan for historical NBBO.** Returns 403 on the current tier. This is the only path that answers the question *retrospectively* rather than forward.
*Time cost:* days, not months — but gated on a paid tier change, and **I cannot price it**: I did not obtain a quote and I will not estimate one. It also does not remove §5g (the production friction bug) or §6's gate-design finding.

**(c) Treat live paper trading under the M1-0.2 clamp model as the measurement.** Both round-trip ends are already clamped to the touch, so every fill is a real, cross-leg-consistent, cost-honest observation and no modelling assumption survives to be argued about.
*Time cost — and this is the number that should drive the decision:* at the observed R dispersion (sd **0.2216** upper bound, **0.2651** primary), putting a one-sided LB95 above zero on a **+0.05R** mean needs `n_eff > (1.645·sd/0.05)²` = **53** / **76** independent units. Trades are **not** independent: exposure clustering gives ~3.7 trades per cluster (44 → 12) and this sample generated **1.52 clusters/month** (0.97 under the primary mark). So 53–76 independent clusters is **≈ 35 months (upper bound) to ≈ 79 months (primary)**. Reaching a mere n = 30 *trades* takes **≈ 5–8 months** — but n = 30 trades is ≈ 8 clusters and **will not resolve a 0.05R edge**; it will produce another LB95 straddling zero.

**The uncomfortable conclusion, stated plainly:** a 0.03–0.05R edge on \$5-wide index verticals is not measurable on any timescale this program has, by *any* of the three paths, unless the edge is larger than claimed or the variance is reduced. The obvious variance levers do not work: wider widths lower R per trade in dollars but not R dispersion; more concurrent positions raise cluster *size*, not cluster *count*. The only lever that helps is **a materially larger per-trade edge**, which means the gate must select on something other than a credit floor sitting below the market's median credit/width.

Path (c) may run **under the M1-0.2 clamp with real fills journalled**, but must be framed as *deployability and process validation*, not as an expectancy measurement — and **M1-0.4 must not be signed off on its basis** until a regime-labelled, quote-based availability figure exists.

---

## 7. Things I could not compute, stated as such

- **The plan's actual criterion.** No regime labels exist in this replay. I did not invent them, and the unconditional rate is not a substitute.
- **SPY/QQQ per-leg quoted spreads.** `option_chain_snapshots` has 5 SPY days and 1 QQQ day. Any per-underlying spread published from that would be a number pretending to be a measurement. The 0.49%/0.94% used here come from the M1-0.3 calibration run and are **not** in `data/execution_costs.json`.
- **The cost of the NBBO data-plan upgrade.** Not quoted, not estimated.
- **Whether the sleeve has an edge.** This artifact cannot answer that in either direction. A negative mean R under the primary mark (−0.0455, LB95 −0.0911 to −0.1581) is **not** evidence against the strategy any more than +0.0561 was evidence for it — both sit inside the same 5.19pp-of-width measurement noise.
