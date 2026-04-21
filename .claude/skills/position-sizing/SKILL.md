---
name: position-sizing
description: Hard sizing rules (R1-R6) for every proposed paper order. The PreToolUse hook re-validates these numerically — if you skip them, the order is blocked.
---

# Position sizing

Every order proposal MUST pass these six rules. The `pretool_order_guard` hook re-checks them against live equity + open positions at order-time, so slop here means a blocked order, not a soft warning.

Canonical code: `src/trading_agent/sizing.py` (pure functions, no I/O). Hook: `src/trading_agent/hooks/pretool_order_guard.py`.

## The six rules

### R1 — single-trade risk ≤ 2% equity
- **Stock**: `risk = |entry - stop| × qty`. Pass `stop` into `place_paper_order` — without it the hook falls back to an implicit 5% stop and emits `R1_stop_missing` warn.
- **Long option**: `risk = debit_per_share × contracts × 100`. Max loss is the debit paid.
- No averaging into losers unless a new thesis is filed.

### R2 — concurrent open positions ≤ 5
- BUY orders blocked when already at 5 opens. SELL orders bypass (they free a slot).
- If you need a 6th, close one first.

### R3 — per-ticker exposure ≤ 10% equity
- Stock + options for the same ticker combined. Notional = `qty × entry` (stock) or `contracts × 100 × entry` (options).
- Prevents "all-in on NVDA" when the thesis is hot.

### R4 — sector concentration: max 2 opens per GICS sector
- Lookup table: `data/sectors.csv` (~90 US large-caps + common ETFs).
- If the proposed ticker isn't in the table, the hook emits `R4_sector_unknown` warn and does NOT block. Add missing tickers manually if you trade them often.

### R5 — options policy (long premium only)
- **Side** = BUY only (selling premium is out of scope for MVP).
- **DTE** ∈ [14, 60]. Parsed from the OCC expiry in the option code; caller can also pass `dte` explicitly.
- **|delta|** ∈ [0.30, 0.55] when `delta` is passed.
- **Notional** ≤ 1% equity (= `contracts × 100 × entry` for long calls/puts).
- Single-leg only at the hook level. Multi-leg requires manual journal entry.

### R6 — earnings lock
- Within 2 trading days of next earnings (`earnings_dte ∈ [0, 2]`), only strategy labels starting with `earnings_` are allowed.
- Rationale: IV crush eats directional single-leg long premium; use debit spreads (label `earnings_debit_spread`) or skip the print.
- `earnings_dte` must be passed by the caller — the hook cannot look it up itself.

## Sizing recipe

Given `equity`, `entry`, `stop`, and target exposure:

```
# Stock, risk-first
risk_budget   = 0.02 × equity                 # R1
per_share_risk = |entry - stop|
max_qty_R1    = floor(risk_budget / per_share_risk)

max_qty_R3    = floor(0.10 × equity / entry)  # R3
max_qty_R2    = 0 if opens ≥ 5 else max_qty_R1
qty           = min(max_qty_R1, max_qty_R3, max_qty_R2)
```

```
# Long option, notional-first
cap_notional  = min(0.01 × equity,            # R5
                    0.02 × equity)            # R1 (==max loss for long premium)
max_contracts = floor(cap_notional / (debit × 100))
```

## When the hook blocks

The hook stderr message names the rule + shows the numbers. Read it. Don't retry the same order — fix the inputs (smaller qty, wider stop, or different ticker). If you genuinely believe the rule is wrong for your strategy, that's a Phase-2 policy discussion — don't edit the hook.

## What the hook DOES NOT check

- Liquidity (bid/ask spread, volume, open interest). That's your job.
- Correlation beyond sector bucket. Two tech stocks = blocked by R4 once you have two opens; two banks with different exposure profiles might both be "Financials" — treat the rule as a floor, not a ceiling.
- Earnings dates. Caller must pass `earnings_dte` for R6 to engage.
