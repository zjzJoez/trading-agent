---
name: options-strategy
description: How to pick options for directional setups — single-leg long premium, plus DEFINED-RISK credit verticals via the combo path (place_paper_option_combo / R5e). Single-leg SELL-to-open stays hard-blocked; standalone naked shorts (R5b/R5c) stay blocked.
---

# Options strategy

Default is **single-leg long premium**. A short leg may ONLY be opened as part of a **defined-risk vertical** via the `place_paper_option_combo` tool (rule `R5e`): the guard proves the long leg caps the short and sizes the spread as ONE position off `max_loss = (width − net_credit) × 100 × contracts`. Single-leg SELL-to-open is still **hard-blocked inside `place_paper_option_order` itself** (`R_short_option_open_blocked`; SELL-to-close an existing long is exempt) — a leg evaluated alone is a naked short (see the 2026-06-08 AAPL 8x 300P leg, ~$235k assignment exposure at zero portfolio heat). Do NOT leg into a spread with two single-leg orders; use the combo tool so both legs are validated and placed atomically (protective long first, then short, with rollback if the short is rejected). Standalone naked shorts stay blocked: R5b (cash-secured puts fully collateralized: `cash ≥ strike × 100 × qty − premium collected`) and R5c (naked short calls need an explicit stop above entry; R1 sizes on a 1.5× stress-buffered stop distance, `NAKED_CALL_STRESS_MULT`) apply only to single legs, which the vertical's defined `max_loss` supersedes for the bounded short leg. The same guards run in the PreToolUse hook for early feedback, but the server-side check is authoritative — scripts and direct MCP calls cannot route around it. Known v1 limitations: portfolio heat over-counts a vertical's short leg at the broker-position level (conservative — blocks more, never less); a first-class combo-CLOSE path is a follow-up. Thresholds quoted in this doc mirror the constants in `src/trading_agent/sizing.py`; if they ever disagree, sizing.py wins.

## When to use options vs stock

Use **stock** when:
- Thesis timeframe is weeks-to-months. Options decay; stocks don't.
- You want pure directional exposure without vega risk.
- Liquidity of the option is thin (wide spreads eat the edge).

Use **long options** when:
- Binary event within 30 days (earnings, FDA, macro print) with a clear direction view → directional call/put.
- You want leverage but can't afford the full stock drawdown.
- Implied vol is *low* relative to realized — long vol is cheap.

Don't use options when:
- IV rank > 70 percentile → you're buying expensive vol; prefer stock.
- DTE < 14 → gamma bleeds faster than your thesis unfolds.
- You want to "average down" — one loss-making option contract is the max pain; adding another is a new thesis.

## Picking the contract — checklist

1. **Expiry**: DTE in `[14, 60]`. Sweet spot for directional plays is ~30 DTE.
2. **Delta**: `|delta|` in `[0.25, 0.65]`. Below 0.25 is a lottery ticket; above 0.65 you're paying too much intrinsic.
3. **IV**: check `list_option_expiries` and inspect chain IV. If IV rank > 70, consider stock instead.
4. **Spread**: quoted bid/ask spread ≤ 5% of mid — **enforced in code (rule R5d)**, no longer just prose. The order tools fetch a live snapshot for every OPENING option order and refuse wider spreads (`R5d_illiquid`); a contract with no obtainable bid/ask is refused outright (`R5d_unquotable` — an entry you can't price honestly is an entry you skip). Wider spreads = instant drawdown.
5. **Open interest**: ≥ 500 contracts at your strike — **enforced in code (rule R5d)**, same gate (`R5d_illiquid`). If the snapshot omits OI the order passes with an `R5d_oi_unknown` warn (spread is the primary protection). Thin OI = roach motel (easy to get in, hard to get out). Closes/SELL-to-close are always exempt from R5d — you are never blocked from exiting. Thresholds are `LIQ_MAX_SPREAD_PCT_OF_MID` / `LIQ_MIN_OPEN_INTEREST` in `src/trading_agent/order_guard.py`; if this doc ever disagrees, order_guard.py wins.
6. **Strike selection**: ATM-to-slightly-OTM on a directional play; avoid deep OTM unless the move is binary and near-term.

## MVP strategies, by label

Use these `strategy_label` values consistently — post-mortem aggregates by label:

- `directional_long_call` / `directional_long_put` — straight single-leg with a clear catalyst.
- `pullback_to_MA` — long call when underlying bounces off 50/200 DMA with volume confirmation.
- `breakout_long_call` — long call when price closes above a consolidation range on volume.
- `earnings_directional_debit_spread` — **currently unavailable.** Legging into a spread with two `place_paper_option_order` calls no longer works: the SELL leg is a SELL-to-open and the order guard refuses it (the old legging advice routed around per-order max-loss math — exactly how the MRVL/AAPL spreads slipped through in June 2026). Until multi-leg combos are first-class (atomic width-minus-credit sizing), express the view with a single long leg instead, sized so the full debit is the max loss.
- `earnings_iv_drop` — buy the call *after* earnings once IV collapses (known edge on outliers).

Any label starting with `earnings_` is the only allowed form within 2 trading days of the earnings date (R6).

## Greeks & what they mean here

- **Delta** — directional exposure. 0.40 delta call on a $100 stock ≈ $40 P&L per $1 move, at t=0.
- **Theta** — daily decay. Long premium: theta is negative. At 30 DTE ATM, typical theta ≈ 1-2% of the debit per day.
- **Vega** — IV sensitivity. On earnings plays this is what kills you: IV drops ~40% overnight, debit drops with it. Earnings-aware strategies neutralize vega with spreads.
- **Gamma** — delta change per $1 move. High near ATM + near expiry. Lots of gamma = fast delta changes, both ways.

## Pre-trade checklist (options)

Before calling `place_paper_option_order`:

1. `search_past_trades` with the underlying and the candidate strategy.
2. Check last 4 earnings reactions if the ticker is within 30 days of earnings.
3. Quote the contract: confirm mid, spread, OI, delta.
4. Compute max loss: long premium = `contracts × debit × 100`; short put = `strike × 100 × qty − premium`; naked call = `1.5 × |stop − entry| × qty × 100`. Cross-check notional ≤ 1.5% equity (R5) and max loss ≤ 2.5% equity (R1) — `MAX_OPTION_NOTIONAL_PCT` / `MAX_SINGLE_RISK_PCT` in sizing.py are canonical.
5. File thesis with concrete invalidation (e.g., "close below $250 underlying OR 30% debit loss").
6. Place the order.

If MoomooOpenD rejects the order (account tier), the tool returns `virtual_fill_suggested=True` — immediately call `record_virtual_fill` at the mid-price so the learning loop isn't broken by a paper-API gap.
