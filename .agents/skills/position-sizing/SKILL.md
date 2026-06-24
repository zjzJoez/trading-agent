---
name: position-sizing
description: Hard sizing rules (R1-R7) for every proposed paper order. The order tools re-validate these numerically inside the moomoo MCP server — if you skip them, the order is blocked.
---

# Position sizing

Every order proposal MUST pass these rules (R1-R7). The shared order guard re-checks them against live equity + open positions at order-time — once client-side in the `pretool_order_guard` hook, and again **inside `place_paper_order` / `place_paper_option_order` themselves** (the authoritative check; the autonomous graph, scripts, and direct MCP calls cannot route around it). Slop here means a blocked order, not a soft warning.

Canonical code: `src/trading_agent/sizing.py` (pure rules, no I/O) + `src/trading_agent/order_guard.py` (shared gate: thesis freshness, open/close intent, SELL-to-open block, audit). Hook: `src/trading_agent/hooks/pretool_order_guard.py`. Every evaluation is recorded in the `hook_audit_log` table; `jobs/reconcile_order_guard.py` alerts nightly on any fill with no evaluation row.

**The numbers in sizing.py are canonical** — this doc mirrors `MAX_SINGLE_RISK_PCT`, `MAX_CONCURRENT_OPENS`, `MAX_TICKER_EXPOSURE_PCT`, `MAX_OPTION_NOTIONAL_PCT`, `OPTION_DELTA_MIN/MAX`, `OPTION_DTE_MIN/MAX`, `MIN_RISK_REWARD`. If this doc and the constants ever disagree, the constants win; when thresholds change again, update the constants and just sync the numbers here. Current values include the 2026-05-22 threshold raises (see the comment block above the constants).

Closing orders (selling a long, buying back a short — `intent="close"`) reduce risk and bypass the opening-only rules (R1, R2, R4–R7); R3 stops counting the new order's notional on closes.

## The rules

### R1 — single-trade risk ≤ 2.5% equity (`MAX_SINGLE_RISK_PCT`)
- **Stock**: `risk = |entry - stop| × qty`. Pass `stop` into `place_paper_order` — without it the hook falls back to an implicit 5% stop and emits `R1_stop_missing` warn.
- **Long option**: `risk = debit_per_share × contracts × 100`. Max loss is the debit paid.
- **Short put (CSP)**: `risk = strike × 100 × qty − premium collected` (worst case = full assignment).
- **Naked short call**: `risk = 1.5 × |stop − entry| × qty × 100` (stress-buffered stop distance, `NAKED_CALL_STRESS_MULT`). No stop → max loss is infinite → blocked (R5c also blocks separately).
- No averaging into losers unless a new thesis is filed.

### R2 — concurrent open positions ≤ 6 (`MAX_CONCURRENT_OPENS`)
- Opening orders blocked when already at 6 opens. Closing orders bypass (they free a slot).
- If you need a 7th, close one first.

### R3 — per-ticker exposure ≤ 12% equity (`MAX_TICKER_EXPOSURE_PCT`)
- Stock + options for the same ticker combined. Notional = `qty × entry` (stock) or `contracts × 100 × entry` (options).
- Prevents "all-in on NVDA" when the thesis is hot.

### R4 — sector concentration: max 2 opens per GICS sector
- Lookup table: `data/sectors.csv` (~90 US large-caps + common ETFs).
- If the proposed ticker isn't in the table, the hook emits `R4_sector_unknown` warn and does NOT block. Add missing tickers manually if you trade them often.

### R5 — options policy (single-leg)
- **Side** = BUY only to open via the single-leg tool. Single-leg SELL-to-open stays hard-blocked at the order layer (`R_short_option_open_blocked`) — a legged-in short evaluated alone is a naked short. SELL-to-close an existing long is exempt. To open a short leg, use a DEFINED-RISK vertical via the combo path (R5e below), never two single-leg orders.
- **DTE** ∈ [14, 60]. Parsed from the OCC expiry in the option code; caller can also pass `dte` explicitly.
- **|delta|** ∈ [0.25, 0.65] when `delta` is passed.
- **Notional** ≤ 1.5% equity (= `contracts × 100 × entry` for the single trade).
- Single-leg only via `place_paper_option_order`. Do NOT leg into spreads with two single-leg orders — the SELL leg will be refused.

### R5b — cash-secured put collateral
- An opening short put must be fully cash-collateralized: `cash ≥ strike × 100 × qty − premium collected`.
- Missing/invalid strike → block (collateral can't be verified).
- Applies to a standalone CSP (still blocked at the single-leg tool); a vertical's defined `max_loss` supersedes the CSP collateral requirement because the long leg caps assignment.

### R5c — naked short call requires an explicit stop
- Opening a short call needs `stop > entry`, otherwise blocked outright — the stop (plus the 1.5× stress buffer in R1) is the only thing bounding the theoretically unbounded risk. Standalone naked calls stay blocked; only defined-risk verticals are unblocked (R5e).

### R5e — multi-leg defined-risk vertical (combo path)
- Open a credit vertical atomically via the `place_paper_option_combo` tool (`evaluate_combo` → `check_combo`). It PROVES the structure is defined-risk before sizing: exactly 2 legs, same underlying + expiry + right, equal contracts, **net credit > 0**, and the **long leg caps the short** (put spread: long strike below short; call spread: long strike above short).
- Sized as ONE position off `max_loss = (width − net_credit) × 100 × contracts`: that max_loss flows through R1 (≤ 2.5% equity), the combo counts once for R2/R3/R4, R5 DTE band applies on the shared expiry, and R5d liquidity is checked on BOTH legs.
- The server places the protective LONG leg first, then the SHORT leg; a short-leg rejection rolls back (closes) the long. The single-leg SELL-to-open block is untouched.

### R6 — earnings lock
- Within 2 trading days of next earnings (`earnings_dte ∈ [0, 2]`), only strategy labels starting with `earnings_` are allowed.
- Rationale: IV crush eats directional single-leg long premium; use debit spreads (label `earnings_debit_spread`) or skip the print.
- `earnings_dte` must be passed by the caller — the hook cannot look it up itself.

### R7 — risk:reward ≥ 1.3 for opening LONG trades (`MIN_RISK_REWARD`)
- `R:R = |target − entry| / |entry − stop|` must be ≥ 1.3. Fails → block; fix by tightening the stop OR widening the target.
- No `target` passed → `R7_target_missing` warn only (floor not enforced). Pass a target so it is.
- Stop or target within a penny of entry → block (degenerate geometry).
- SHORT premium opens are exempt — R:R is structurally capped when selling premium.
- The floor is mirrored in `llm.schemas.MIN_RISK_REWARD`; both must move together.

## Sizing recipe

Given `equity`, `entry`, `stop`, and target exposure:

```
# Stock, risk-first
risk_budget   = 0.025 × equity                # R1
per_share_risk = |entry - stop|
max_qty_R1    = floor(risk_budget / per_share_risk)

max_qty_R3    = floor(0.12 × equity / entry)  # R3
max_qty_R2    = 0 if opens ≥ 6 else max_qty_R1
qty           = min(max_qty_R1, max_qty_R3, max_qty_R2)

# R7 gate before submitting:
#   (target - entry) / (entry - stop) ≥ 1.3   (long trades)
```

```
# Long option, notional-first
cap_notional  = min(0.015 × equity,           # R5
                    0.025 × equity)           # R1 (==max loss for long premium)
max_contracts = floor(cap_notional / (debit × 100))
```

## When the guard blocks

The hook stderr message (or the order tool's `{"order_blocked": true, "violations": [...]}` response) names the rule + shows the numbers. Read it. Don't retry the same order — fix the inputs (smaller qty, wider stop, or different ticker). If you genuinely believe the rule is wrong for your strategy, that's a Phase-2 policy discussion — don't edit the guard.

## What the guard DOES NOT check

- Liquidity (bid/ask spread, volume, open interest). That's your job.
- Correlation beyond sector bucket. Two tech stocks = blocked by R4 once you have two opens; two banks with different exposure profiles might both be "Financials" — treat the rule as a floor, not a ceiling.
- Earnings dates. Caller must pass `earnings_dte` for R6 to engage.
