# Position Management — Rules & Thresholds

Source of truth for every numeric threshold, where it's enforced, and why.
This document is canonical: if the code disagrees with what's written here,
either fix the code or open a PR to update both together.

Last updated: 2026-05-22 (Phase 3 — deterministic exit executor)

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
| **R1** Single-trade max loss | ≤ 2% equity | block | `sizing.MAX_SINGLE_RISK_PCT` |
| **R2** Concurrent open positions | ≤ 5 | block | `sizing.MAX_CONCURRENT_OPENS` |
| **R3** Per-ticker exposure | ≤ 10% equity (stock+options combined) | block | `sizing.MAX_TICKER_EXPOSURE_PCT` |
| **R4** Sector concentration | ≤ 2 open positions in same GICS sector | block | `sizing.MAX_SAME_SECTOR_OPENS` |
| **R5** Options policy: BUY-only | side must be BUY (no naked shorts) | block | `sizing.check` |
| **R5** Options DTE window | 14 ≤ DTE ≤ 60 | block | `sizing.OPTION_DTE_MIN/MAX` |
| **R5** Options delta band | 0.30 ≤ |Δ| ≤ 0.55 | block | `sizing.OPTION_DELTA_MIN/MAX` |
| **R5** Options notional | ≤ 1% equity per trade | block | `sizing.MAX_OPTION_NOTIONAL_PCT` |
| **R6** Earnings lock | within 2 trading days of earnings, only `earnings_*` strategies | block | `sizing.EARNINGS_LOCK_DAYS` |
| **R7** Risk:reward floor | reward / risk ≥ 1.5 | block | `sizing.MIN_RISK_REWARD` + `TraderProposal._validate_risk_reward` |

> **Why 1.5?** At our ~55% baseline win rate, R:R 1.5:1 gives expected
> value `0.55 × 1.5 − 0.45 × 1 = +0.375 R` per trade. Anything below 1.5
> requires a win rate above 67% just to break even — implausible for
> directional trading. The SPY 742C 0.6:1 R:R was the trigger for R7.

R7 is enforced **twice**:
1. **`TraderProposal._validate_risk_reward`** (pydantic) — raises before
   the proposal ever leaves the synthesizer. Router retries the LLM with
   the validation error so it can either tighten stop, widen target, or
   decline the trade.
2. **`sizing.check` R7** — runs at the pretool hook, catches manual
   rescue orders that bypass the synthesizer. Warns if `target` is
   missing (hook layer may not know it), blocks otherwise.

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

### P1 — Hard stop

| Condition | Action |
|---|---|
| `mark ≤ plan.hard_stop` | **EXIT_STOP** full close |

> Uses **mark** (best mid or last_price from moomoo), not intra-bar wick.
> This is intentionally permissive — a single fast trade through the
> stop won't trigger; the price has to actually settle below.

### P2 — DTE rules (options only)

| Condition | Action | Default threshold |
|---|---|---|
| `DTE ≤ force_exit_at_dte` | **EXIT_DTE_HARD** full close | 2 days |
| `DTE ≤ 5` AND `|Δ| < force_exit_at_dte_5_if_delta_below` | **EXIT_DTE_OTM** full close | 0.40 |
| `DTE ≤ switch_to_intrinsic_floor_at_dte` AND `|Δ| ≥ 0.40` AND `mark ≤ intrinsic − buffer` | **EXIT_INTRINSIC_FLOOR** full close | 5 days, $0.10 buffer |

> **Why this split?** Theta decay near expiry kills OTM options
> (negative EV), but ITM/ATM options retain intrinsic value worth
> capturing. The intrinsic floor locks in intrinsic value − $0.10 of
> remaining premium, never letting the stop drop below the entry-time
> `hard_stop`.

### P3 — Hard target + scale-out ladder

| Condition | Action |
|---|---|
| `mark ≥ plan.hard_target` | **EXIT_TARGET** full close |
| `mark ≥ rung.at_mark` (for the next un-fired rung) | **EXIT_TARGET** partial close at `rung.exit_factor` |

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

### P4 — Trailing stop

| Condition | Action |
|---|---|
| `high_water ≥ plan.trail_stop.engage_at_mark` AND `mark ≤ high_water × (1 − distance_pct)` AND `trailed_stop ≥ never_below_floor` | **EXIT_TRAIL** full close |

`high_water` is derived from `journal_trades.mfe_so_far` (MFE in
R-multiples, updated every tick by `excursion.update_excursions_once`):
```
R_unit     = |entry − hard_stop|
high_water = entry + mfe_so_far × R_unit
```

`never_below` enforces a floor:
- `"break_even"` / `"entry"` → trailed_stop ≥ `entry_price`
- `"original_stop"` → trailed_stop ≥ `hard_stop` (never gives back below entry stop)

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
