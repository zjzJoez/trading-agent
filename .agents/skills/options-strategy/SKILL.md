---
name: options-strategy
description: How to pick single-leg options for directional setups. Long premium is the default; short premium (cash-secured puts, naked calls) is allowed under R5b/R5c.
---

# Options strategy

Phase-1 is **single leg only**. Long premium is the default structure; opening short premium is also allowed — R5b requires cash-secured puts to be fully collateralized (`cash ≥ strike × 100 × qty − premium collected`), and R5c blocks a naked short call unless an explicit stop above entry is passed (R1 then sizes its risk on a 1.5× stress-buffered stop distance, `NAKED_CALL_STRESS_MULT`). Multi-leg structures (credit spreads, iron condors) still have no atomic sizing — each leg is checked standalone. The thresholds quoted in this doc mirror the constants in `src/trading_agent/sizing.py`; if they ever disagree, sizing.py wins (see its "Threshold raises (2026-05-22)" comment block).

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
4. **Spread**: bid/ask < 10% of mid. Wider spreads = instant drawdown.
5. **Open interest**: > 100 contracts at your strike. Thin OI = roach motel (easy to get in, hard to get out).
6. **Strike selection**: ATM-to-slightly-OTM on a directional play; avoid deep OTM unless the move is binary and near-term.

## MVP strategies, by label

Use these `strategy_label` values consistently — post-mortem aggregates by label:

- `directional_long_call` / `directional_long_put` — straight single-leg with a clear catalyst.
- `pullback_to_MA` — long call when underlying bounces off 50/200 DMA with volume confirmation.
- `breakout_long_call` — long call when price closes above a consolidation range on volume.
- `earnings_directional_debit_spread` — NOTE: MVP hook is single-leg; if you want a spread, leg into it manually with two `place_paper_option_order` calls **on the same thesis_id**. The SELL leg is checked standalone as an opening short (R5b/R5c apply — a short call leg needs an explicit stop). Journal will treat them as two trades; post-mortem aggregates by `strategy_label`.
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
