---
description: Propose a sized order (qty, stop, target) that passes R1-R7. Returns numbers, not orders.
argument-hint: SYMBOL DIRECTION
---

# /size $ARGUMENTS

Compute a sized order proposal for `$ARGUMENTS` (format: `TICKER LONG` or `TICKER LONG_CALL` etc). Returns proposed qty + stop + target + max_loss, with every rule check shown.

If the argument is malformed or missing, ask the user for `<TICKER> <DIRECTION>`.

Directions accepted: `LONG`, `SHORT`, `LONG_CALL`, `LONG_PUT`, `SHORT_PUT` (cash-secured put), `SHORT_CALL` (naked call), `VERTICAL_CALL_DEBIT`, `VERTICAL_PUT_DEBIT`. Short-premium opens must also satisfy R5b (CSP fully cash-collateralized) and R5c (naked call requires an explicit stop > entry).

## Procedure

**1. Gather current state.**

- `mcp__moomoo-mcp__get_account_info()` → equity (`total_assets`).
- `mcp__moomoo-mcp__get_positions()` → current open positions.
- `mcp__journal-mcp__get_open_positions_with_thesis()` → open trades tracked by journal (the hook uses this list too).

Reconcile any mismatch between broker positions and journal — flag to the user if they diverge.

**2. Gather proposed-ticker info.**

- `mcp__moomoo-mcp__get_quote(["US.<TICKER>"])` for spot.
- For stock: compute 14-day ATR from 30 days of daily K (use `get_historical_kline`). Propose stop at `spot - 1.5 × ATR` for LONG, `spot + 1.5 × ATR` for SHORT.
- For options (LONG_CALL / LONG_PUT / SHORT_PUT / SHORT_CALL): use the pre-researched option contract (premium, strike, delta, DTE) from `/research`. If the user hasn't run `/research` yet, ask them to, or call `get_option_chain_snapshot` ourselves.

**3. Apply R1-R7 sizing math.**

The numbers below mirror the canonical constants in `src/trading_agent/sizing.py` (`MAX_SINGLE_RISK_PCT`, `MAX_OPTION_NOTIONAL_PCT`, …) — if this doc and the constants disagree, sizing.py wins.

For stock:

```
risk_per_share = |entry - proposed_stop|
qty_R1   = floor(0.025 * equity / risk_per_share)
qty_R3   = floor(0.12 * equity / entry)
qty      = min(qty_R1, qty_R3)
```

For long option:

```
max_contracts_R5   = floor(0.015 * equity / (debit * 100))
max_contracts_R1   = floor(0.025 * equity / (debit * 100))
max_contracts      = min(max_contracts_R5, max_contracts_R1)
```

For short premium (R5 notional cap of 1.5% equity still applies):

```
# Short put (CSP): risk_R1 = strike × 100 × qty − premium collected
#                  R5b also requires cash ≥ that same amount.
# Naked short call: risk_R1 = 1.5 × (stop − entry) × qty × 100
#                  (stress-buffered; R5c blocks without an explicit stop > entry)
```

Also check:

- R2: `open_count < 6` — if ≥ 6, propose `qty=0` and tell the user to close something first.
- R4: look up proposed ticker in `data/sectors.csv` (read the file directly). Count opens in that sector. If ≥ 2, propose `qty=0`.
- R6: if within 2 trading days of earnings AND direction is long premium without `earnings_` label, propose `qty=0` or switch to `earnings_directional_debit_spread`.
- R7: opening LONG trades need `R:R = |target − entry| / |entry − stop| ≥ 1.3`. The default 2:1 target passes; if you override the target, re-check. Short-premium opens are exempt.

**4. Propose a target.**

Rough R:R 2:1 default:

```
target_price = entry + 2 * |entry - stop|   # LONG
target_price = entry - 2 * |entry - stop|   # SHORT
```

For options, target = 2 × debit (100% on debit) for single-leg. Override if the underlying chart suggests a concrete level.

## Output

```
# /size $ARGUMENTS

Equity: $XXX,XXX
Open positions: N / 6
Same-sector opens: M / 2 (sector: ...)
Earnings DTE: ... (or unknown)

Proposed order:
  symbol        US.TICKER
  side          BUY / SELL
  qty           <N>
  entry         $X.XX  (limit)
  stop          $Y.YY  (implied risk: $X per share)
  target        $Z.ZZ  (R:R 2:1)
  max loss      $X.XX  (<= 2.5% equity = $EQUITY * 0.025)
  strategy_label <label>

Rule checks:
  R1  single-trade risk: PASS (risk $X ≤ $Y budget)
  R2  concurrent opens:  PASS (N < 6)
  R3  ticker exposure:   PASS ($A < $B budget)
  R4  sector concent.:   PASS (M < 2 in sector)
  R5  option policy:     N/A  (stock trade)
  R5b CSP collateral:    N/A  (not a short put)
  R5c naked-call stop:   N/A  (not a short call)
  R6  earnings lock:     PASS (DTE > 2)
  R7  risk:reward:       PASS (2.0 ≥ 1.3)

Ready for /enter with thesis_id=<TBD>.
```

If any rule FAILS, show PASS/FAIL per rule and propose a fix (smaller qty, wider stop, different expiry, or "wait N days for earnings to clear"). Do NOT place the order.

## Constraints

- `/size` does NOT file a thesis and does NOT place an order.
- If `get_account_info` fails (OpenD down), stop and tell the user. Don't fabricate equity.
- Paper equity is typically ≈ $1M USD on Moomoo SG; use the live value, not an assumption.
