# M1-0.4 — Gate-feasibility replay + managed-payoff expectancy for `credit_vertical_index_30_45`

**Date:** 2026-07-25 · **Branch:** `impl/vertical-replay` · **Sleeve:** short put verticals, SPY/QQQ, 30–45 DTE

| artifact | what it is |
| --- | --- |
| `inputs/` | the daily option aggregates and batch plan every run below reads — committed, so the replay is reproducible end to end |
| `summary.json`, `entries.jsonl` | **PRIMARY** run — `mark=smile`, `liquidity_lag=prior`, `width_policy=five-first`, `credit_floor_frac=0.25` |
| `sensitivity/entries_*.jsonl` | one complete independent replay per convention (indexed from `summary.json .sensitivity.rows`) |
| `as_reported/summary.json`, `as_reported/entries.jsonl` | **DISCLOSED UPPER BOUND** — `mark=close`, `liquidity_lag=same-day`: the configuration whose numbers were rejected, kept auditable (45 trades, not the draft's 44 — see §5a) |
| `diagnostics.json` | the adversarial checks for both samples (`scripts/replay_vertical_diagnostics.py`) |
| `REGENERATE.sh` | the exact four commands, in order |

**Scope of the reproducibility claim.** Every number in §1–§5 and §6's gate-design finding is reproducible from the committed artifacts in this directory: `bash reports/vertical_replay_2026-07/REGENERATE.sh` rebuilds all of them byte for byte, and step 4 of that script cross-checks **391** pinned values two ways (`scripts/verify_replay_report.py`). The two directions have different strengths and the script prints both counts on every run: **(a)** each pinned value is matched against the artifact row it names, tuple-keyed on `(entry_date, symbol, short_strike, long_strike)` for a trade — a value that is right for a *different* row fails; **(b)** each pinned value must also still appear in this document's prose, which catches a number deleted or a pin that never matched, but **not** a substitution whose value occurs elsewhere here (verified: rewriting §2c's `1.41` to `1.32` still passes, because 1.41 occurs in other rows). 266 of the 391 pins carry no key identifying their line, so per-claim contextual matching is unavailable for most claims and is not claimed. A fourth exclusion from this paragraph's promise, therefore: the prose direction is presence-only.

Three classes of number here are **not** covered by that claim, and are labelled where they appear:

1. **The friction calibration file.** `data/execution_costs.json` is gitignored, so §5g's "what production would charge today" figures cannot be rebuilt from this repo alone. The file's sha256 (`a467989051373cae…`) is recorded in `summary.json .config.spread_pct_provenance`, so the version used is pinned even though the bytes are not committed.
2. **The `option_chain_snapshots` coverage counts in §6(a).** Live reads from EC2 Postgres on 2026-07-25. They will be different tomorrow by design — that is what accrual means. Not artifacts.
3. **The historical-NBBO 403 in §6(b).** A vendor entitlement response, not a committed file. Re-checking it means re-issuing the request.

---

## 1. VERDICT UP FRONT

### The plan's criterion was NOT EVALUATED AS SPECIFIED

`docs/REVIVAL_PLAN_2026-07-20.md` line 81 (M1-0.4):

> 验收:**允许交易的 regime 内合格 vertical 存在于 ≥60% 快照日**

The criterion is **conditional on the regimes the sleeve is allowed to trade**. This replay carries **no regime labels**. It measured availability **unconditionally** over all planned entry days. An unconditional rate is neither the criterion nor a bound on it — the conditional rate could be higher (if the allowed regimes are the high-IV ones where credit is easiest to find) or lower.
Recorded in `summary.json .verdict` (`evaluated_as_specified: false`, `regime_labels_present: false`).

**M1-0.4 therefore remains OPEN. It cannot be signed off from this artifact.**

### What WAS measured

Availability of a qualifying vertical, unconditionally, over **144 planned (entry_date, symbol) pairs** (SPY + QQQ, planned entry days 2025-10-06 → 2026-06-11):

| denominator | n | definition | primary (`smile`/`prior`) | upper bound (`close`/`same-day`) | ≥60%? |
| --- | --- | --- | --- | --- | --- |
| **planned** | 144 | every planned entry day; data gaps charged against availability | 26/144 = **0.1806** | 45/144 = **0.3125** | **FAIL / FAIL** |
| **data-adequate** | 119 | entries whose *fetched* strikes could construct ≥1 vertical | 26/119 = **0.2185** | 45/119 = **0.3782** | **FAIL / FAIL** |
| marks-present | 123 | ≥1 usable entry-day option print | 26/123 = 0.2114 | 45/123 = 0.3659 | FAIL / FAIL |
| data-adequate **and** delta-band bracketed | 56 / 63 | the only denominator where the gate was actually *asked* the question | 8/56 = **0.1429** | 23/63 = **0.3651** | **FAIL / FAIL** |

**The floor fails on every denominator under both marks.** There is no reading of this data on which the gate clears 60%.

> **Correction to the previous draft.** It reported the band-bracketed row as 26/56 = 0.4643 and 44/63 = 0.6984, and described the second as the one denominator where the floor was "cleared". Both numbers were arithmetically invalid: the numerator was *all* qualified entries while the denominator was only the band-bracketed ones, and `qualified` is **not** a subset of `brackets_delta_band`. The bracket test asks whether the fetched liquid invertible grid *straddles* [0.20, 0.35]; an entry can qualify on an in-band strike while its grid stops short of one of the band's two edges. Restricted to its own population (`summary.json .availability.overall.n_qualified_band_bracketed` = 8 and 23) the rate is 0.1429 / 0.3651 — a fail, not the study's one pass. Every denominator now ships the numerator it was computed from, and the artifact asserts each numerator is a subset of its denominator (`diagnostics.json .*.denominator_hygiene.numerator_is_subset_of_denominator`).

**De-noised availability is 0.2185.** Section 3 shows the `close` mark's selected credits sit −1.20 in-band-null sd below a cross-leg-consistent estimate. Under the cross-leg-consistent mark availability is 26/119 = 0.2185 (`prior` liquidity screen) or 27/119 = 0.2269 (`same-day`, `summary.json .sensitivity.rows["liquidity-same-day"]`). **The 0.3782 is an upper bound, not the honest figure.**

### The retune claim collapses

`summary.json .availability.credit_floor_sensitivity` (primary): 0.25 → 0.2185, **0.225 → 0.5714**, 0.20 → 0.8235.
Under the upper-bound mark, 0.225 → 0.6807 on the 119 denominator but only 0.5625 on the 144 denominator. "Just lower the credit floor to 0.225" therefore reaches 60% on **neither** denominator under the honest mark. Reaching it requires 0.20 — a 20% credit floor, i.e. abandoning the spec's `credit >= width/4` — and Section 2 shows the floor margin is the same size as the measurement noise, so that retune would be fitting noise.

---

## 2. THE HEADLINE SCIENTIFIC RESULT: identification failure

**Daily trade-print aggregates cannot resolve a ±0.05R edge on \$5-wide index verticals.**

A vertical's credit is a **difference of two legs**. In a daily aggregate each leg's "close" is its own *last trade of the day*, struck at a different minute. The difference therefore carries the full leg-vs-leg timing gap — and on a \$5 width that gap is the same size as the quantity being measured.

### 2a. Mark-convention sensitivity — same data, same gates, same walk

`summary.json .sensitivity.rows`, all at the spec floor 0.25, `liquidity_lag=prior`, `width_policy=five-first`:

| mark | availability (119 / 144) | n | WR | mean R | median R | LB95 (week) | LB95 (exposure) | maxDD R | PT / 21-DTE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **smile** (PRIMARY, cross-leg consistent) | 0.2185 / 0.1806 | 26 | 0.6923 | **−0.0455** | +0.0950 | −0.1581 | −0.0911 | 2.4309 | 12 / 14 |
| **close** (disclosed upper bound) | 0.3697 / 0.3056 | 44 | 0.7727 | **+0.0520** | +0.1396 | −0.0254 | −0.0158 | 2.1690 | 32 / 12 |
| vw (volume-weighted average price) | 0.4538 / 0.3750 | 54 | 0.9444 | **+0.1554** | +0.1657 | +0.1185 | +0.1310 | 0.8835 | 51 / 3 |
| hl2 ((high+low)/2) | 0.3025 / 0.2500 | 36 | 0.6944 | **+0.0118** | +0.1190 | −0.0702 | −0.0277 | 1.6786 | 19 / 17 |

Four defensible readings of "the price that day" span **mean R from −0.0455 to +0.1554 — 0.2009R** — and availability from 0.2185 to 0.4538. The spread of that column is **four times the 0.05R the sleeve claims**. The headline result is a property of the mark convention, not of the strategy.

(The `close` row is at `liquidity_lag=prior`, so its 44 trades differ by one from the 45 of the `as_reported` sample, which uses `same-day`. §5a gives the reason and the cost.)

`smile` is primary because it is the only cross-leg-consistent convention: both legs are repriced off ONE same-day IV curve (weighted quadratic in moneyness K/S, ≥5 strikes with ≥10 trades, sqrt(trades) weights). It fitted **1,897** day-curves, left **364** days unfittable, repriced **16,075** bars and kept **1,904** raw (`summary.json .mark_diagnostics.smile`). No bar was interpolated or carried forward under any convention.

**The cross-leg-consistency claim, made auditable rather than asserted.** The previous draft stated the 1,904 kept-raw bars without showing they could not produce a mixed leg pair. "Kept raw" now splits by cause, and only one of the two causes can break the claim:

- `bars_kept_raw_on_unfittable_days` = **1,904** — the whole day had no fittable curve, so *every* strike that day kept its raw close. A pair drawn from such a day is raw+raw: one convention, no mixing.
- `bars_kept_raw_on_fitted_days` = **0** — no strike anywhere extrapolated to an insane IV on a day that *did* have a curve. That is the only way a curve-priced leg could ever pair with a raw-priced one, so **no mixed pair is possible in this run at all**, not merely absent from the booked trades.

And per booked trade (`summary.json .mark_diagnostics.raw_mark_exposure`): across the 26 primary trades' **217** marked leg-pair days (**434** leg-days), `n_marked_leg_days_raw` = **0**, `n_trades_with_any_raw_marked_leg_day` = **0**, `n_trades_with_a_mixed_pair_day` = **0**. Not one booked trade touched a raw close on any day it was marked. The 1,904 kept-raw bars all sit on unfittable days that no booked trade ever marked against.

### 2b. The null study — the noise IS the edge

`diagnostics.json .samples.*.null_study`. Every liquid \$5-wide adjacent put pair, on every day, in every fetched chain, **no gate applied** — raw close-difference credit vs the same-day curve value:

- **n = 11,842**, mean **+0.0014**, median +0.0005, **sd 0.2594**, p10 −0.1361, p90 +0.1335
- restricted to the sleeve's own window (DTE 30–45, short |delta| 0.20–0.35): n = 1,864, mean +0.0047, median +0.0028, sd **0.2434**

**What this identifies, and what it does not.** The statistic is a **difference between two conventions** for the same quantity, so it may only be read that far:

- The mean is 0.5% of its own sd. That says there is **no systematic offset *between* the two conventions**. It does **not** say either convention is unbiased. Any bias common to both cancels exactly in a difference, and there is no ground-truth mid anywhere in daily trade aggregates to compare either one against.
- The sd is a bound on the two conventions' **combined** noise — √(var_raw + var_curve) if their errors are independent. Each convention's own noise is therefore *at most* this sd, not equal to it.

**Not identified by this study, and not claimed anywhere in this report:** either convention's own bias, and either convention's own variance. The previous draft's "the estimator is **unbiased** … so its dispersion is a clean measurement of this data source's credit noise" over-claimed on both counts. The artifact key that asserted it is renamed `no_systematic_offset_between_conventions` and now ships a `what_is_not_identified` field stating the limits above.

What the study *does* deliver is the scale on which two defensible conventions disagree about the same spread — which is the number that matters, because the claimed edge has to be resolved against exactly that disagreement:

> **sd 0.2594 on a \$5 width = 5.19 percentage points of width.**

The spec's credit-floor margin (0.25 versus the market's measured median credit/width of 0.188, Section 6) and the sleeve's entire claimed edge (0.03–0.05R) are **the same size as that disagreement**. No configuration of this data source resolves them.

### 2c. Two absurdities the raw marks produced

`diagnostics.json .samples.as_reported_close_sameday.exit_trigger_audit`. Of 34 profit-takes testable against a same-day curve, **9 (26.5%)** are exits the curve says never happened. Two of them:

1. **A profit-take booked on a flat day.** 2025-10-13 QQQ 590/585, credit **1.41**, exit 2025-10-15. The underlying moved **602.01 → 602.22 (+0.03%)** — two sessions, essentially nothing. The raw close-difference mark printed **0.620**, under the 0.705 profit-take level, and the trade booked **R = +0.1371**. The same-day curve puts that spread at **1.329** — roughly *twice* the trigger level. Nothing happened; a print pair moved.

2. **A profit-take booked on a day the underlying fell 588.00 → 573.79.** 2026-03-23 QQQ 570/565, credit **1.32**, exit 2026-03-26, a **−2.42%** day — the wrong direction for a short put spread. The raw mark printed **0.010**: a \$5-wide put spread marked at one cent, against a 0.660 trigger. Booked **R = +0.1093**. The same-day curve says **1.572**.

> **Correction to the previous draft.** It gave example 1's credit as 1.32 — which is example 2's credit. The artifact row for `(2025-10-13, QQQ, 590/585)` says **1.41**, and the draft's own quoted trigger of 0.705 = 1.41/2 already implied it. The draft's cross-check tested whether each printed number appeared *somewhere* in the artifact, which 1.32 did — on a different trade. The check is now a per-claim tuple match against the row each claim names, and a value that has migrated to another row fails it (`scripts/verify_replay_report.py`, regression-tested in `tests/test_verify_replay_report.py`). That strength applies to the ARTIFACT direction. The prose direction is presence-only and would **not** have caught this particular substitution on its own — 1.41 occurs in other rows — which is why the artifact tuple match, not the prose scan, is what closed this defect. Both are stated in §0's scope paragraph and printed on every run.

Under the primary `smile` mark, curve-denied profit-takes number **0 of 12** (`diagnostics.json .samples.primary_smile_prior.exit_trigger_audit`) — the internal-consistency check the primary convention must pass, and does.

A related but confounded test — freezing each leg's entry-day IV and revaluing at the exit-day spot — flags 27 of 34 profit-takes. That mixes in IV mean-reversion between entry and exit, so **9/34 is the defensible figure** and 27/34 is an upper bound.

---

## 3. THRESHOLD SELECTION BIAS

`diagnostics.json .samples.as_reported_close_sameday.selection_bias`. The same null statistic, evaluated on the **45 gate-selected entries** instead of on all pairs:

| sample | n | mean | median | sd |
| --- | --- | --- | --- | --- |
| ungated null (all liquid 5-wide pairs) | 11,842 | +0.0014 | +0.0005 | 0.2594 |
| in-band null (DTE 30–45, \|delta\| 0.20–0.35) | 1,864 | +0.0047 | +0.0028 | 0.2434 |
| **gate-selected entries** | 45 | **−0.2878** | −0.2792 | 0.2427 |

Shift versus the like-for-like in-band null: **−0.2925 spread-dollars = −5.85 percentage points of a \$5 width = −1.20 in-band-null sd** (−1.13 ungated sd; both are emitted so neither can be cherry-picked).

Read plainly: **the credit floor filters the noise, not the market.** A day qualifies when its raw print pair happens to be wide. Two one-directional consequences:

1. **Availability is biased UP.** The 0.3782 figure counts days that qualified because their prints were lucky — which is why availability collapses to 0.2185 the moment the marks are made cross-leg consistent.
2. **Booked credit carries ≈ +0.082R of pure measurement bias** (`implied_measurement_bias_in_credit_r = 0.0822`) — **larger than the entire reported edge** of +0.0557R. Under the primary mark the same statistic is −0.0047 dollars = −0.02 null sd (0.0013R), i.e. gone by construction.

---

## 4. SURVIVORSHIP

`diagnostics.json .samples.*.survivorship`. Entries dropped from the availability denominator are **not** a random subsample.

**This section's own denominator is 130 of the 144 planned entries.** A record enters the comparison only if BOTH realized-vol windows exist for it — trailing 20 sessions and forward 10 sessions — and **14** entries sit too close to one end of the underlying history for that. The counts are now published (`n_records` 144, `n_records_with_rv` 130, `n_records_skipped_for_rv` 14, with each skipped entry listed in `records_skipped_for_rv`). The group counts below sum to 130, not 144; the previous draft printed 25 + 105 without saying so, leaving an unexplained shortfall of 14.

| group | n | trailing-20 RV (mean / median) | forward-10 RV (mean) |
| --- | --- | --- | --- |
| dropped (not data-adequate) | 25 | **0.1263** / 0.1322 | 0.1494 |
| kept (data-adequate) | 105 | **0.1594** / 0.1538 | 0.1595 |
| — of which qualified | 40 | 0.1718 / 0.1731 | 0.1776 |
| — of which gate-rejected | 65 | 0.1517 / 0.1430 | 0.1484 |

(The 40 qualified here is 40 of the sample's 45 booked trades; the other 5 are among the 14 without a full RV window.)

**Welch t (dropped − kept) = −4.4455.** Dropping is strongly vol-correlated, and in the direction that flatters the result: the days whose option data was never fetched are the **low-vol** days — exactly the days a credit floor is hardest to clear. Removing them inflates availability.

**The conservative bound is therefore 45/144 = 0.3125** for the upper-bound mark and **26/144 = 0.1806** for the primary. Charging every data gap against availability is the only treatment the fetch pattern cannot bias upward.

The gaps themselves: of 74 planned (symbol, expiry) batches, **7 were never fetched** and **4 contain no bars at all**; 543 contract rows, 511 with bars; zero unparsable lines, zero bad bars (`summary.json .data_inventory`).

---

## 5. WHAT SURVIVES AND IS USEFUL

### 5a. ONE information set, declared and applied consistently

The previous draft was self-contradictory here. It defended entry-day leg marks as a close-to-close *convention* while simultaneously calling the entry day's own full-day trade count *look-ahead* and "fixing" it. Both quantities come off the **same completed bar**; no single information set admits one and forbids the other. Exactly one now governs this engine, declared once (`summary.json .config.information_set`, `scripts/replay_vertical_gates.py: INFORMATION_SET`):

> **ENTER-AT-THE-CLOSE.** The decision is taken on the completed entry-day daily bar and filled at that bar's close.

Applied consistently, that means:

- **Entry-day leg closes as selection marks: legal.** Spot, every leg mark, the IV inversion and the delta all come from the entry date's bar. A close-to-close model is a convention, not future information.
- **The entry day's own full-day trade count `n`: also legal.** It is part of the same completed bar the selection already prices off. So is that bar's high/low/vwap — which is what makes the `hl2` and `vw` rows in §2a legal comparisons rather than cheats.
- **Same-day full-day trade counts are therefore *not* look-ahead**, and this report no longer says they are. `summary.json .config.liquidity_screen_uses_future_information` is `false` in both samples, and the residual `liquidity_screen_is_look_ahead` key is gone.
- **Illegal under any reading, and absent everywhere here:** any bar dated after the entry day feeding the entry decision.

**So why is `prior` still the primary screen?** Not as a look-ahead fix — as a **robustness** choice. Screening on the last session strictly *before* entry is the only variant that is also valid under the stricter decide-before-the-close reading, so the primary result does not depend on which convention a reader prefers. The cost is measured and published rather than assumed:

| screen | trades under `close` | trades under `smile` | availability (`smile`, /119) |
| --- | --- | --- | --- |
| `same-day` — legal under the declared information set | 45 | 27 | 0.2269 |
| `prior` — PRIMARY; stricter, also valid decide-before-close | 44 | 26 | 0.2185 |

**One trade under each mark, and 0.0084 of availability.** That is the entire price of being valid under both readings, which is why the stricter screen is the default. Under the declared convention the `same-day` numbers are the ones in force; the primary result is deliberately the more conservative of the two.

**The asymmetry this leaves, stated rather than hidden:** the engine does *not* implement a decide-before-the-close variant of the **marks**. Under that stricter information set the entry-day closes would be unknowable too, and *nothing* in this replay would be legal. That variant is not measured here, and no result below should be read as surviving it.

Otherwise the audit is clean, by re-derivation:

- **The walk moves strictly forward** (`entry_date < d <= expiry`); the forced-exit date is `expiry − 21d`, known at entry.
- **No day was silently skipped.** In both samples, trading days inside a held window with no walkable leg pair: **0**; reported `skipped_days`: **0** (`diagnostics.json .*.walk_coverage`). Nothing is interpolated, carried forward or synthesized anywhere in the pipeline; every skip has a counter.
- **`data_end` exits: 0** in both samples. No trade's result depends on the dataset ending.
- **`credit_floor_sensitivity` is exact** at and below the floor the run used, and refuses to project above it. It compares with the same tolerance as the gate, so its count at the run's own floor reproduces the run's.
- **The bootstrap is deterministic** per seed (tested), and now reports two blocking schemes rather than one.

**One gate defect fixed here, and it is worth naming.** The credit floor was a bare float comparison on the spec boundary. Entry 2026-01-02 QQQ 600/595 has closes 8.20 and 6.95 — a credit of exactly \$1.25 on a \$5 width, i.e. exactly `width/4` — but `8.20 - 6.95` is `1.2499999999999991` in binary, so `credit >= width * 0.25` rejected it while an identical 1.25 quoted directly passed. Both the gate and the credit-floor projection now compare with a 1e-9 tolerance, tested at exactly 1.25/5.00. Effect: the upper-bound sample goes 44 → **45** trades and availability 0.3697 → 0.3782; the primary sample is **unchanged at 26** (no primary entry sits on the boundary). Every upper-bound figure in this report is the 45-trade version.

### 5b. The payoff SHAPE — half of it is a tautology

In the upper-bound sample: 34 profit-takes, **all** winners (R +0.1093 … +0.2945); **11** forced 21-DTE closes, 10 losers and **1 marginal winner (+0.0351)**; 0 `data_end`. Mean R by exit: PT **+0.1639**, 21-DTE **−0.2789**.

**State plainly which half of that carries information.** `managed_walk` books every profit-take **at the limit**: the exit debit on a profit-take day is exactly `0.5 × credit` by construction, so the gross P&L of a profit-take is exactly `+0.5 × credit`, and its R is positive for any credit whose friction does not consume half of it. **"All 34 profit-takes are winners" is therefore a tautology of the exit rule, not a finding** — it would hold on random data. The same applies to the primary sample's 12-for-12. The previous draft presented this separation as "the mechanical signature of a managed short-premium sleeve, not a data artifact"; the sign of the profit-take leg is neither, it is arithmetic.

What the split *does* carry information about:

- **The magnitudes.** PT +0.1639 vs 21-DTE −0.2789 is a ratio of about 1:1.7, and neither side is pinned to any particular size by the exit rule. That the mean loss is under 2× the mean win, with 3.1× as many wins, is a real (if unsurprising) property of these 45 trades.
- **The mix**, 34:11 — how often the 50% take fires before the 21-DTE forced exit. Nothing in the exit rule fixes that ratio.
- **What it carries NO information about:** whether the sleeve has an edge. The sign of every PT follows from the management rule, the sign of most forced exits follows from the direction of the underlying, and §2–§3 show the credits those R values are computed from are noise-dominated and selection-biased.

The draft also called the upper-bound shape "**perfectly separated**", 34 winners against 10 losers with no exceptions. That is now wrong twice over: the sample has **11** forced exits (the §5a credit-floor fix added one), and **one of them is positive**, so the separation was never perfect. Under the primary mark it is looser still — 12 PTs all winners, but 14 forced closes of which **6** were positive (max +0.1046). The "perfect" separation was partly the raw marks' doing and partly an artifact of a float comparison.

### 5c. The −1R tail is UNSAMPLED, not absent

Worst booked R: **−0.5694** (upper bound), **−0.6157** (primary). Occupied histogram bins run −0.6 … +0.2 (upper bound) and −0.7 … +0.1 (primary), with **underflow 0 and overflow 0**. A full −1R requires the short strike still breached at the forced 21-DTE exit, which never occurred in this 9-month window. **Its absence is a property of the window, not of the strategy.** Any sizing that treats the observed tail as the real tail is under-reserved.

### 5d. Drawdown attribution — and the ordering it depends on

`diagnostics.json .samples.as_reported_close_sameday.payoff_attribution`. Whole sample: n = 45, mean **+0.0557R**, median +0.1412, sum **+2.5048R**, **max drawdown 2.4748R** — the drawdown is 99% of the total gain.

**The equity curve is now walked in BOOKING order** — `(exit_date, entry_date, symbol)` — not entry order. A drawdown is a property of when P&L is *booked*: an equity curve walked in entry order credits a trade's result before the days the position was actually still open, which both mis-times and mis-measures the trough, and in this sample the two orderings genuinely differ because several trades opened early and closed late. The previous draft reported **2.3774R** from the entry-order curve; on the same 44 trades the booking-order figure is **2.4748R**, and it is 2.4748R on the corrected 45 as well. The primary sample is 2.4309R under either ordering. The sort key and its rationale are in `max_drawdown_r`'s docstring. Ties on a single booking day break by `(entry_date, symbol)` — a disclosed *convention*, not a measurement: at daily granularity the order in which same-day exits book is unobservable, and while it cannot change that day's net it can in principle move the within-day trough. The key is fixed so the figure is reproducible.

- Excluding the **5 worst** trades, the other **40 average +0.1221R** (dropped: −0.5694, −0.5123, −0.4520, −0.4256, −0.4181).
- Excluding **all of 2026-03**, the other **34 average +0.1126R** (2026-03 alone: n = 11, mean −0.1203R).

**Both exclusions are illegitimate as stated, and the second is worse:** 2026-03 has the **highest unconditional median IV of any month in the sample (0.2755)**, and `payoff_attribution` flags it `is_highest_iv_month: true`. Deleting it removes precisely the regime a short-premium sleeve exists to harvest — and the regime in which it takes its losses. It is not a neutral operation, and no "excluding X" figure may be quoted as the strategy's expectancy.

### 5e. Unconditional per-month IV, with n

`diagnostics.json .*.iv_by_month`. **Unconditional** = every liquid invertible in-band strike on every planned entry day at DTE 30–45, regardless of whether it qualified. The conditional column is qualified trades only, selected on credit and therefore on IV — it cannot be read as the month's vol level.

| month | n strikes | n entry days | **unconditional** median IV | mean IV | conditional median IV (n) |
| --- | --- | --- | --- | --- | --- |
| 2025-10 | 62 | 16 | 0.2073 | 0.2074 | 0.2142 (6) |
| 2025-11 | 61 | 15 | 0.2321 | 0.2331 | 0.2218 (8) |
| 2025-12 | 37 | 13 | 0.1736 | 0.1804 | 0.2079 (2) |
| 2026-01 | 42 | 12 | 0.1954 | 0.1892 | 0.1902 (2) |
| 2026-02 | 56 | 16 | 0.2271 | 0.2340 | 0.2514 (6) |
| **2026-03** | 71 | 16 | **0.2755** | 0.2750 | 0.2755 (11) |
| 2026-04 | 71 | 16 | 0.2293 | 0.2242 | 0.2255 (6) |
| 2026-05 | 23 | 6 | 0.1904 | 0.2040 | 0.2252 (1) |
| 2026-06 | 20 | 5 | 0.2650 | 0.2424 | 0.2646 (3) |

### 5f. The block bootstrap was too tight

`diagnostics.json .*.blocking`. Entry-ISO-week blocking gives **26 blocks** for 45 trades. But the median holding period is **6 calendar days, max 18** — a one-week block cannot contain an 18-day exposure, and **51 of 990 trade pairs are open simultaneously while sitting in different blocks**, i.e. resampled as independent draws.

Exposure-cluster blocking (connected components of overlapping `[entry, exit]` intervals) gives **13 blocks** for the same 45 trades (largest 11). Both are now reported:

| sample | LB95 (entry-week) | LB95 (exposure-cluster) | blocks (week / cluster) |
| --- | --- | --- | --- |
| upper bound (45 trades) | −0.0261 | −0.0180 | 26 / 13 |
| primary (26 trades) | −0.1581 | −0.0911 | 15 / 7 |

Coarser blocks happen to move LB95 *up* in this sample (fewer, larger blocks reduce resampling variance here) — but a 5th percentile drawn from 13 blocks is so granular it should not be read as a confidence statement at all. **Every LB95 in this study is negative under both schemes, on every mark convention except `vw`** — and `vw` is the convention that also prints a 0.9444 win rate, which is its own tell.

### 5g. The friction defect this study surfaced — the most actionable output

`summary.json .friction_comparison`, `as_reported/summary.json .friction_comparison`, `diagnostics.json .*.friction`. Measured on the 45 upper-bound trades at the M1-0.3 index spreads (SPY 0.49%, QQQ 0.94% of mid):

| model | friction, mean R | mean \$ |
| --- | --- | --- |
| **replay** — spread charged on all four realized leg marks | **0.0484** | 17.25 |
| `execution_costs.friction_r` **as deployed** — `4 × half_spread(net_credit)` | **0.0176** | 6.26 |
| the same 2-legs × 2-directions stack on the **actual** leg marks | **0.0513** | 18.25 |

(The previous draft printed the third row's mean as \$18.29; the artifact says **18.2524**.)

**Ratio replay / deployed: mean 2.73×** (min 1.63×, max 5.56×).

The cause is in the deployed function's own docstring. `friction_r` proxies each leg's mark by `net_credit`, i.e. it assumes `leg-mid-sum ≈ 2 × net_credit`, exact only at wing ratio `long/short = 1/3`, which the docstring calls "the typical 30-45 DTE vertical". Measured here:

- **wing ratio `long_mark/short_mark`: mean 0.8486, median 0.8532** — not 0.333.
- **`leg_mid_sum / net_credit`: mean 12.63, median 12.63** — the model assumes **2.0**.

So on SPY/QQQ \$5-wides at |delta| 0.20–0.35 and 30–45 DTE the deployed model **understates the spread leg of friction by ≈ 6.3×**. This independently confirms, from daily aggregates on SPY/QQQ, the same defect measured from real IWM quotes (wing ratio 0.731, leg-sum/credit 6.43). It is a **production cost-model bug, not a replay artifact**, and it bites hardest at exactly the 0.03–0.05R margins this sleeve claims.

**Second, worse finding — the friction inputs are not traceable.** The live `data/execution_costs.json` (sha256 `a467989051373cae…`, calibrated 2026-06-12, n_samples 160) contains **no `per_underlying` entry for SPY or QQQ**. Asked today, the deployed model falls back to its single-name global `spread_pct_of_mid_median = 0.07338` for both index names. Charging that on the same booked marks (`summary.json .friction_under_calibration_file`) gives friction of **0.3555R** (upper bound) / **0.3885R** (primary) — at which point the sleeve is not merely unprofitable, it is unrunnable. The 0.49%/0.94% this replay used come from the M1-0.3 calibration *run*, not from the file; the engine now emits that disagreement as a warning on every invocation. **M1-0.3's per-underlying calibration must be written into `data/execution_costs.json` before any real order.** (As flagged in the header: this file is gitignored, so this sub-section is the one part of the report a fresh clone cannot rebuild — only its sha256 is pinned.)

### 5h. Width freedom does not rescue the gate

`summary.json .sensitivity.rows["width-any"]`. The \$10-wide branch previously never fired for two reasons: `five-first` breaks after a priced \$5 wing, and an inverted/stale \$5 print used to `break` out of the width loop entirely. The second is now a `continue`, so width freedom is genuinely explored. Measured effect: **none — 26 qualified, mean R −0.0455, identical to `five-first`.** The reason is arithmetic, not code (`diagnostics.json .*.width_census`): **119 of 144 entries do have a constructible \$10-wide pair — the same 119 that have a \$5-wide one** — but the \$10-wide credit/width is *lower* than the \$5-wide (**median 0.1790 over n = 359** vs **0.1880 over n = 408**), so widening makes a 25% floor harder, not easier. Only **20 of 359 (5.6%)** constructible \$10-wides clear the spec floor, against **48 of 408 (11.8%)** \$5-wides. A real degree of freedom that simply does not help.

### 5i. All 7 "no strike in delta band" rejections were data gaps

`diagnostics.json .*.band_bracket_audit`. A per-entry bracket test asks whether the fetched, liquid, invertible strike grid *straddles* [0.20, 0.35]. In the upper-bound sample: **7 rows, 7 classified `band-not-fetched` (class `data`), 0 true gate rejections** — every one with `brackets_delta_band: false`. Charging those against the gate was wrong.

Each row now carries the two counts that are easy to conflate — `n_contracts_fetched` (put rows fetched) and `n_liquid_invertible` (rows clearing the n≥30 screen *and* inverting to a sane IV) — plus both strike ranges, so a row cannot be paraphrased into a different claim:

| entry | spot | fetched | strikes fetched | liquid+invertible | strikes surviving | \|delta\| range |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-26 QQQ | 623.89 | 1 | 590 | 1 | 590 | 0.165 |
| 2025-12-26 SPY | 690.31 | 1 | 665 | 1 | 665 | 0.187 |
| 2025-12-29 QQQ | 620.87 | 1 | 590 | 1 | 590 | 0.176 |
| 2025-12-29 SPY | 687.85 | 1 | 665 | 1 | 665 | 0.197 |
| 2026-01-20 QQQ | 608.06 | 6 | 520–545 | 4 | 520–545 | 0.065–0.115 |
| 2026-01-26 QQQ | 625.46 | 6 | 520–545 | **1** | **530** | **0.031** |
| 2026-05-11 QQQ | 713.29 | 3 | 660–670 | 3 | 660–670 | 0.147–0.187 |

> **Correction to the previous draft.** It described the 2026-01-26 QQQ row as "6 invertible strikes, K 520–545, |delta| 0.025–0.046". The artifact says **6 rows fetched** spanning K 520–545, of which exactly **one** cleared liquidity and invertibility — K=530, at |delta| **0.031**. Three misreadings in one clause: fetched conflated with surviving, a fetched range attributed to a single surviving strike, and a delta range matching no row in the file.

In every case the strikes that would have answered the question were never fetched: the nearest fetched strike sits 5–15% below spot, at |delta| 0.03–0.20, when the band needs 0.20–0.35. (The primary sample has 6 such rows rather than 7 — the stricter `prior` liquidity screen moves one entry to `short-leg-illiquid`.)

---

## 6. WHAT THIS DOES NOT VALIDATE, AND WHAT WOULD

### It does not price the sleeve

This study **cannot** support any of the following, and no re-tuning inside it will:

- an expectancy estimate, a win rate, or a breakeven — the mark convention alone moves mean R by 0.2009R, four times the claimed edge;
- an availability figure to compare against the 60% bar — the criterion is regime-conditional and this replay has no regime labels;
- a credit-floor retune — the floor margin equals the convention disagreement (5.19pp of width), so retuning optimizes noise;
- a −1R tail assumption — the tail is unsampled;
- a friction number for sizing — the deployed cost model understates the spread leg ≈6.3× and its calibration file has no SPY/QQQ entry at all.

What it **does** establish, and these are real deliverables: the identification limit of daily trade-print aggregates for this instrument (§2); the direction and magnitude of the credit-floor selection bias (§3); the survivorship direction (§4); the exit *mix* and the loss-to-win magnitude ratio under the management rules (§5b — *not* the sign of the profit-take leg, which is a tautology); the unconditional IV census (§5e); the correct blocking scheme (§5f); the production friction defect (§5g); and the width finding (§5h).

One more gate-design finding, independent of every measurement problem above (`diagnostics.json .*.width_census`): **credit/width on real SPY/QQQ \$5-wides has a median of 0.1880** over **408** constructible in-band verticals — close to the 0.191 median measured from real IWM quotes. Only **11.8%** of them clear `credit >= width/4`. The spec's floor sits **above the median of what the market offers** in this delta/DTE window. That, not any modelling artifact, is the first-order reason availability is low, and it is the one finding here that no better data source will change.

### Three honest paths, with their time cost

**(a) Accumulate real bid/ask from the now-fixed daily snapshot feed.** `option_chain_snapshots` carries real `bid`/`ask` (2,115 of 2,178 core-ETF rows have a bid; all have an ask), which removes the identification failure at its root: a quoted spread is a same-instant, cross-leg-consistent measurement. *Coverage as read live from EC2 Postgres on 2026-07-25 — a live query, not an artifact in this directory, and different by the time you read this:* **SPY 5 distinct days (2026-06-22 → 07-23), QQQ 1 day (07-16), IWM 6 days (06-19 → 07-24)** — 12 symbol-days total, following the core-underlyings-first fix (commit `6b01329`).
*Time cost:* at ~21 trading days/month, SPY+QQQ accrues 42 symbol-days/month, so the 144 symbol-days this study used take **≈ 3.4 months** (≈ mid-November 2026); with IWM, 63/month → **≈ 2.3 months**. Reaching 30 trades at the measured 0.1806 planned availability: **≈ 4.0 months** (2 symbols) or **≈ 2.6 months** (3 symbols). **Cost: calendar time only — no dollars, no new code.** Recommended, and it should start accruing regardless of what else is chosen.

**(b) Upgrade the data plan for historical NBBO.** Returns 403 on the current tier *(a live vendor entitlement response, re-checkable only by re-issuing the request — not a committed artifact)*. This is the only path that answers the question *retrospectively* rather than forward.
*Time cost:* days, not months — but gated on a paid tier change, and **I cannot price it**: I did not obtain a quote and I will not estimate one. It also does not remove §5g (the production friction bug) or §6's gate-design finding.

**(c) Treat live paper trading under the M1-0.2 clamp model as the measurement.** Both round-trip ends are already clamped to the touch, so every fill is a real, cross-leg-consistent, cost-honest observation and no modelling assumption survives to be argued about.
*Time cost — and this is the number that should drive the decision:* at the observed R dispersion (sd **0.2191** upper bound, **0.2651** primary), putting a one-sided LB95 above zero on a **+0.05R** mean needs `n_eff > (1.645·sd/0.05)²` = **52** / **76** independent units. Trades are **not** independent: exposure clustering gives ~3.5 trades per cluster (45 → 13) and this sample generated **1.64 clusters/month** (0.97 under the primary mark). So 52–76 independent clusters is **≈ 32 months (upper bound) to ≈ 79 months (primary)**. Reaching a mere n = 30 *trades* takes **≈ 5–8 months** — but n = 30 trades is ≈ 8–9 clusters and **will not resolve a 0.05R edge**; it will produce another LB95 straddling zero.

**The uncomfortable conclusion, stated plainly:** a 0.03–0.05R edge on \$5-wide index verticals is not measurable on any timescale this program has, by *any* of the three paths, unless the edge is larger than claimed or the variance is reduced. The obvious variance levers do not work: wider widths lower R per trade in dollars but not R dispersion; more concurrent positions raise cluster *size*, not cluster *count*. The only lever that helps is **a materially larger per-trade edge**, which means the gate must select on something other than a credit floor sitting below the market's median credit/width.

Path (c) may run **under the M1-0.2 clamp with real fills journalled**, but must be framed as *deployability and process validation*, not as an expectancy measurement — and **M1-0.4 must not be signed off on its basis** until a regime-labelled, quote-based availability figure exists.

---

## 7. Things I could not compute, stated as such

- **The plan's actual criterion.** No regime labels exist in this replay. I did not invent them, and the unconditional rate is not a substitute.
- **A decide-before-the-close variant of the marks.** The declared information set is enter-at-the-close (§5a). The stricter reading, under which the entry day's own closes are also unknowable, is not implemented and not measured.
- **Either mark convention's own bias or own variance.** §2b identifies the offset and the combined noise *between* two conventions. Neither is compared against a ground-truth mid, because this data source has none.
- **SPY/QQQ per-leg quoted spreads.** `option_chain_snapshots` had 5 SPY days and 1 QQQ day when read on 2026-07-25. Any per-underlying spread published from that would be a number pretending to be a measurement. The 0.49%/0.94% used here come from the M1-0.3 calibration run and are **not** in `data/execution_costs.json`.
- **The cost of the NBBO data-plan upgrade.** Not quoted, not estimated.
- **Whether the sleeve has an edge.** This artifact cannot answer that in either direction. A negative mean R under the primary mark (−0.0455, LB95 −0.0911 to −0.1581) is **not** evidence against the strategy any more than +0.0557 was evidence for it — both sit inside the same 5.19pp-of-width convention disagreement.
