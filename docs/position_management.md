# Position Management — Rules & Thresholds

Source of truth for every numeric threshold, where it's enforced, and why.
This document is canonical: if the code disagrees with what's written here,
either fix the code or open a PR to update both together.

Last updated: 2026-05-22 (Phase 4 — CSP + naked call support; aggressive-mode thresholds)

---

## 0. Big picture

```
┌──────────────────┐    R1-R7 sizing gate    ┌──────────────────┐
│  Synthesizer     │───────────────────────► │ pretool_order_   │
│  emits proposal  │   pydantic + sizing.py  │ guard.py (hook)  │
└──────────────────┘                         └────────┬─────────┘
                                                      │ allow
                                                      ▼
                                          ┌────────────────────────┐
                                          │  place_paper_order     │
                                          │  → trade_nodes.persist │
                                          │  → exit_plan JSONB     │
                                          └────────┬───────────────┘
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │ Every 15 min while market open:  │
                                  │  refresh_quotes_and_greeks       │
                                  │  update_excursions (mae/mfe)     │
                                  │  detect_exit_triggers            │
                                  │    └─► hard_exit_decision()      │
                                  │  route_exit_or_hold              │
                                  └──────────────────────────────────┘
```

- **All exit decisions are deterministic Python code, no LLM in the hot path.**
- **All R:R discipline is enforced at entry time via R7 + pydantic validator.**
- Plan is baked into `journal_trades.exit_plan JSONB` at entry. The
  executor never re-derives it. Operator can hand-edit a plan via SQL
  if needed; the next tick picks it up.

---

## 1. Entry-time discipline (sizing.py + TraderProposal validator)

| Rule | Threshold | Severity | Source |
|------|-----------|----------|--------|
| **R1** Single-trade max loss | ≤ **2.5%** equity (raised from 2%) | block | `sizing.MAX_SINGLE_RISK_PCT` |
| **R2** Concurrent open positions | ≤ **6** (raised from 5) | block | `sizing.MAX_CONCURRENT_OPENS` |
| **R3** Per-ticker exposure | ≤ **12%** equity (raised from 10%) | block | `sizing.MAX_TICKER_EXPOSURE_PCT` |
| **R4** Sector concentration | ≤ 2 open positions in same GICS sector | block | `sizing.MAX_SAME_SECTOR_OPENS` |
| **R5** Options policy | side ∈ {BUY, SELL} — long + short both allowed | block | `sizing.check` |
| **R5** Options DTE window | 14 ≤ DTE ≤ 60 | block | `sizing.OPTION_DTE_MIN/MAX` |
| **R5** Options delta band | **0.25 ≤ |Δ| ≤ 0.65** (widened from 0.30-0.55) | block | `sizing.OPTION_DELTA_MIN/MAX` |
| **R5** Options notional | ≤ **1.5%** equity per trade (raised from 1%) | block | `sizing.MAX_OPTION_NOTIONAL_PCT` |
| **R5b** Cash-secured put | `cash ≥ strike × 100 × qty − premium` for short puts | block | `sizing.R5B` |
| **R5c** Naked call requires stop | short call must have explicit `stop > entry` (R1 uses 1.5× stress buffer) | block | `sizing.R5C` + `sizing.NAKED_CALL_STRESS_MULT` |
| **R6** Earnings lock | within 2 trading days of earnings, only `earnings_*` strategies | block | `sizing.EARNINGS_LOCK_DAYS` |
| **R7** Risk:reward floor (LONG only) | reward / risk ≥ **1.3** (lowered from 1.5) — SHORT premium exempt | block | `sizing.MIN_RISK_REWARD` + `TraderProposal._validate_geometry` |

> **Max heat:** R1 × R2 = 2.5% × 6 = **15%** of equity at full max-position
> drawdown (up from 10%). Combined with R3's 12% per-ticker cap and the
> tighter delta band, this is the "moderately aggressive" mode the
> 2026-05-22 audit explicitly enabled.

> **Why 1.3 for R7?** At our ~55% baseline win rate, 1.3:1 R:R gives EV
> `0.55 × 1.3 − 0.45 = +0.27 R/trade` — still positive-EV but accepts
> more setups than 1.5:1. Anything below 1.3 needs >60% win rate to
> break even (implausible).

> **Why no R7 for shorts?** Premium-selling trades have R:R structurally
> capped (reward = premium ≤ strike, risk = up to strike × 100). A CSP
> at strike $100 selling for $2 has R:R ≈ 0.02:1 even when EV is
> clearly positive. R1 with proper max-loss math + R5b cash collateral
> bound the practical risk instead.

R7 is enforced **twice** (LONG only):
1. **`TraderProposal._validate_geometry`** (pydantic) — raises before
   the proposal ever leaves the synthesizer.
2. **`sizing.check` R7** — runs at the pretool hook, catches manual
   rescue orders that bypass the synthesizer.

---

## 1a. Direction-aware geometry

| direction | geometry constraint | who profits when mark... |
|---|---|---|
| LONG / LONG_CALL / LONG_PUT | target > entry > stop | rises (mark > entry) |
| SHORT_PUT (cash-secured) | stop > entry > target | falls (premium decays) |
| SHORT_CALL (naked) | stop > entry > target | falls (premium decays) |

The pydantic validator enforces both:
- LONG: `target > entry > stop` AND R:R ≥ 1.3
- SHORT: `stop > entry > target`, R7 exempt

---

## 1b. R1 max-loss math (varies by direction)

| Position type | max_loss formula |
|---|---|
| STK LONG (with stop) | `|entry − stop| × qty` |
| STK LONG (no stop) | `IMPLICIT_STOP_FRAC × entry × qty` (= 5%) — warn-only fallback |
| OPT LONG (buy premium) | `qty × 100 × entry_price` (capped at debit paid) |
| OPT SHORT_PUT (CSP) | `strike × 100 × qty − premium_collected` (assignment exposure) |
| OPT SHORT_CALL (naked) | `NAKED_CALL_STRESS_MULT × |stop − entry| × qty × 100` (= 1.5×, gap-risk buffer) |
| OPT SHORT_CALL (no stop) | `inf` — R1 blocks even if R5c missed it |

> **Naked-call stress buffer rationale:** unbounded theoretical risk
> can't be bounded by stop discipline alone — overnight gap-ups blow
> through stops. The 1.5× multiplier pretends the gap cost 50% more
> than the stop distance, which is generous protection at the
> portfolio level while still allowing the trade.

---

## 2. Exit plan structure (set at entry, stored as JSONB)

Every new trade gets an `ExitPlan` written into `journal_trades.exit_plan`.
If the synthesizer doesn't emit one, `default_exit_plan()` generates a
sensible default from `(entry, stop, target, asset_type)`.

```jsonc
{
  "version": 1,
  "hard_stop": 4.55,
  "hard_target": 13.50,

  "scale_out_ladder": [
    { "at_mark": 11.20, "exit_factor": 0.5, "then_engage_trail": true }
  ],

  "trail_stop": {
    "engage_at_mark": 11.20,
    "distance_pct": 0.15,
    "never_below": "break_even"      // or "entry", "original_stop"
  },

  "dte_rules": {
    "force_exit_at_dte": 2,
    "force_exit_at_dte_5_if_delta_below": 0.40,
    "switch_to_intrinsic_floor_at_dte": 5,
    "intrinsic_floor_buffer": 0.10
  },

  "regime_rules": {
    "exit_on_labels": ["CRISIS"],
    "downsize_50_on_labels": []      // reserved, not yet implemented
  },

  "time_in_trade_max_days": 30
}
```

**Defaults** (`schemas.default_exit_plan`):

| asset | scale rung | trail | time_in_trade_max |
|---|---|---|---|
| OPT | 50% at entry+0.7×(target−entry), then engage trail | 15% trail from high-water, floor break-even | 30 days |
| STK | none (whole-share atomicity) | 10% trail from high-water at entry+0.7×(target−entry), floor break-even | 60 days |

---

## 3. Hard exit executor (every 15-min tick during US hours)

`trading_agent.exits.hard_executor.hard_exit_decision` evaluates rules in
**strict priority order**. The first rule that fires wins; the rest are
skipped that tick.

### P0 — Regime kill switch

| Condition | Action |
|---|---|
| `regime_label in plan.regime_rules.exit_on_labels` (default: `["CRISIS"]`) | **EXIT_REGIME** full close |

### P1 — Hard stop (direction-aware)

| direction | Condition | Action |
|---|---|---|
| LONG | `mark ≤ plan.hard_stop` | **EXIT_STOP** full close |
| SHORT | `mark ≥ plan.hard_stop` | **EXIT_STOP** full close |

> Uses **mark** (best mid or last_price from moomoo), not intra-bar wick.
> This is intentionally permissive — a single fast trade through the
> stop won't trigger; the price has to actually settle there.

### P2 — DTE rules (options only, direction-aware)

**LONG options**:

| Condition | Action | Default threshold |
|---|---|---|
| `DTE ≤ force_exit_at_dte` | **EXIT_DTE_HARD** full close | 2 days |
| `DTE ≤ 5` AND `|Δ| < force_exit_at_dte_5_if_delta_below` | **EXIT_DTE_OTM** full close | 0.40 |
| `DTE ≤ switch_to_intrinsic_floor_at_dte` AND `|Δ| ≥ 0.40` AND `mark ≤ intrinsic − buffer` | **EXIT_INTRINSIC_FLOOR** full close | 5 days, $0.10 buffer |

**SHORT options**:

| Condition | Action |
|---|---|
| `DTE ≤ force_exit_at_dte` (assignment risk) | **EXIT_DTE_HARD** full close |

> **Why different?** For LONG options, time decay is your enemy near
> expiry, especially OTM (theta vampire) → force exit. For SHORT
> options, time decay is your FRIEND — let it cook. Only assignment
> risk on the final day(s) demands a forced exit. The intrinsic-floor
> rule is also LONG-only (different math for shorts; skipped).

### P3 — Hard target + scale-out ladder (direction-aware)

**LONG**:

| Condition | Action |
|---|---|
| `mark ≥ plan.hard_target` | **EXIT_TARGET** full close |
| `mark ≥ rung.at_mark` (for the next un-fired rung) | **EXIT_TARGET** partial close at `rung.exit_factor` |

**SHORT**:

| Condition | Action |
|---|---|
| `mark ≤ plan.hard_target` | **EXIT_TARGET** full close (buy back cheap) |
| `mark ≤ rung.at_mark` (for the next un-fired rung) | **EXIT_TARGET** partial close at `rung.exit_factor` |

Scale rungs are evaluated in declaration order. The first rung whose
`at_mark` is satisfied AND that hasn't already fired wins. "Already
fired" is tracked persistently via `journal_trades.scale_rungs_taken`,
incremented by `route_exit_or_hold` after each successful partial close.

> **Partial-close semantics:** when a scale rung fires:
> - Broker order trims position by `qty × exit_factor` contracts/shares
> - `journal_trades.scale_rungs_taken` increments by 1
> - `journal_trades` row STAYS `outcome='OPEN'` (residual qty keeps being monitored)
> - Parent thesis stays `status='open'` until a full close fires
>
> When a FULL close fires (`exit_factor == 1.0` OR any non-EXIT_TARGET
> action), close_trade + close_thesis run as usual.

> **Partial-exit safety on qty=1:** if `qty == 1` (1-contract option),
> any rung with `exit_factor < 1.0` is **demoted to HOLD** by
> `detect_exit_triggers` since `max(1, round(1 × 0.5)) = 1` would force
> a full close anyway. The next tick re-evaluates from scratch (likely
> hits the hard target if mark stays up).

### P4 — Trailing stop (direction-aware)

**LONG** trails the HIGH-water mark; exit when mark falls back:

| Condition | Action |
|---|---|
| `high_water ≥ engage_at_mark` AND `mark ≤ high_water × (1 − distance_pct)` AND `trailed_stop ≥ floor` | **EXIT_TRAIL** full close |

`high_water` is derived from `journal_trades.mfe_so_far` (MFE in
R-multiples, updated every tick by `excursion.update_excursions_once`):
```
R_unit     = |entry − hard_stop|
high_water = entry + mfe_so_far × R_unit  (LONG)
low_water  = entry + mae_so_far × R_unit  (SHORT — note mae is negative)
```

**SHORT** trails the LOW-water mark; exit when mark rises back:

| Condition | Action |
|---|---|
| `low_water ≤ engage_at_mark` AND `mark ≥ low_water × (1 + distance_pct)` AND `trailed_stop ≤ floor` | **EXIT_TRAIL** full close |

`never_below` enforces a floor:
- `"break_even"` / `"entry"` → LONG: trailed_stop ≥ `entry_price`; SHORT: trailed_stop ≤ `entry_price`
- `"original_stop"` → never gives back beyond entry stop

### P5 — Max age

| Condition | Action | Default |
|---|---|---|
| `now − opened_at ≥ time_in_trade_max_days` | **EXIT_AGE** full close | 30d (OPT) / 60d (STK) |

> Capital efficiency — a trade that hasn't worked in 30 days is
> consuming a position slot without earning return. Free the slot.

---

## 4. Safety nets in detect_exit_triggers

The executor runs inside `intraday_nodes.detect_exit_triggers`, which
adds three integration-layer protections:

| Guard | Behavior |
|---|---|
| **No plan + no legacy stop/target** | HOLD + emit sev-1 `position_no_exit_plan` event. Position is unmonitored — likely manual fill or phantom row. |
| **Partial exit on qty=1 option** | Demote to HOLD (see P3 note). Next tick re-evaluates. |
| **Plan parse failure** | Falls back to legacy `journal_trades.stop` + `.target` (loaded via `load_exit_plan`). |

---

## 5. Phantom-trade reconcile (EOD)

`eod_nodes.reconcile_phantom_trades` runs at 21:30 UTC after
`reconcile_journal`. Any `journal_trades` row that:

- has `outcome = 'OPEN'`
- is older than 24 hours (`PHANTOM_AGE_HOURS`)
- has every available fill signal (`fill_qty`, `dealt_qty`,
  `dealt_avg_price`, `avg_fill_price`) at zero

is marked `outcome='UNFILLED'`, `closed_at=NOW()`, and its parent thesis
is voided. Emits sev-1 `phantom_trades_voided` event for ops visibility.

> **Why a real ops event?** A phantom trade means the broker rejected
> something. We want to know — but we don't want to keep the imaginary
> position alive in the executor's view of the world.

---

## 6. What's intentionally NOT here

- **No LLM exit_monitor.** Removed 2026-05-22 after audit. The schema
  (`ExitMonitorOutput`) and prompt are kept as dead code in case we
  want to revive as shadow comparison later.
- **No mid-trade adjustments to stop/target via LLM.** Plan is set at
  entry, immutable except via SQL. (Trailing-stop is mechanical, not LLM.)
- **No thesis-broken auto-detection from news.** The plan's
  `event_rules` field is reserved for future use; right now nothing
  populates it.

---

## 7. Worked example — what would happen to a SPY 742C 5/29

Hypothetical: entry $10.14, hard_stop $4.55, hard_target $13.50, qty=1.

`default_exit_plan` for an option produces:

| field | value |
|---|---|
| hard_stop | 4.55 |
| hard_target | 13.50 |
| scale_out_ladder | 1 rung @ 10.14 + 0.7×(13.50−10.14) = **12.49**, factor 0.5 |
| trail_stop | engage @ 12.49, distance 15%, never_below break_even |
| dte_rules | force_exit_at_dte=2, OTM force at DTE≤5 if |Δ|<0.40, intrinsic floor at DTE≤5 |
| time_in_trade_max | 30d |

Daily trajectory (illustrative):

| Day | DTE | mark | Δ | Rule that fires | Action |
|---|---|---|---|---|---|
| 5/14 entry | 15 | 10.14 | 0.54 | (entry) | open |
| 5/22 (today, T-7) | 7 | 9.50 | 0.45 | none | HOLD |
| 5/27 (T-2) | 2 | 9.00 | 0.50 | **P2a** DTE ≤ 2 | **EXIT_DTE_HARD** |
| OR if SPY rallies to 745 on 5/26 (T-3) | 3 | 14.00 | 0.85 | **P3** mark ≥ target | **EXIT_TARGET** |
| OR if SPY drops to 738 on 5/26 (T-3) | 3 | 4.50 | 0.18 | **P1** mark ≤ stop fires first | **EXIT_STOP** |

The 1-contract scale-out rung at 12.49 would be **demoted to HOLD**
(see safety nets); the trail would then engage on subsequent ticks if
mark stays above 12.49, with stop trailing 15% behind high-water and
never below entry $10.14.

> R7 would have **rejected** this trade at proposal time: R:R =
> (13.50−10.14) / (10.14−4.55) = 3.36/5.59 = **0.60 < 1.5**. The
> synthesizer would have had to either tighten the stop to ≥ $7.90
> (R:R = 1.5) or widen target to ≥ $18.53.

---

## 8. Operator escape hatches

- **Edit a plan**: `UPDATE journal_trades SET exit_plan = '...' WHERE id=N;`
  Next intraday tick picks it up.
- **Force close**: just close at the broker. EOD `reconcile_journal`
  will flag the discrepancy (`only_in_journal`); next intraday tick
  sees no broker position so the executor's decisions don't matter.
- **Disable hard executor for a position**: set
  `journal_trades.exit_plan = NULL` AND
  `journal_trades.stop = NULL` AND
  `journal_trades.target = NULL`. The executor will return HOLD +
  emit `position_no_exit_plan`. (Use sparingly — this is unmonitored.)
- **Pause everything**: hit `/halt` endpoint → blocks new entries; open
  positions continue to be monitored by the hard executor.
