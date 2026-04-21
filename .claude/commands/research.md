---
description: Deep-dive on a single ticker — quote, options chain, filings, insider activity, past-trade memory.
argument-hint: TICKER
---

# /research $ARGUMENTS

Run full-stack research on `$ARGUMENTS` (a single ticker, e.g. `AAPL`). This is the mandatory step before `/size` or `/enter`.

If no argument was passed, ask the user for one ticker (bare symbol, no `US.` prefix) and stop.

## Procedure — do these in the order listed

**1. Past-trade memory (FIRST, not last).**

Before fetching any new data, call:
- `mcp__journal-mcp__search_past_trades(query="$ARGUMENTS")` — pulls both semantic matches in `notes` and lexical matches in `trades`.
- `mcp__journal-mcp__search_past_trades(query="<sector> <candidate strategy>")` — optional; run after step 5 once you know which strategy is on the table.

Render the returned `notes_semantic` entries prominently in your working notes. If any lesson or past trade is directly relevant, quote it into the thesis draft.

**2. Spot + technicals.**

- `mcp__moomoo-mcp__get_quote(["US.$ARGUMENTS"])` for last/prev_close/volume.
- `mcp__moomoo-mcp__get_historical_kline("US.$ARGUMENTS", start=<90 days ago>, end=<today>, ktype="K_DAY")` → compute 20DMA, 50DMA, 200DMA, 14-day ATR. Note where spot sits relative to each MA.
- Identify the nearest support and resistance from the last 90 days.

**3. Options context (if options are on the table).**

- `mcp__moomoo-mcp__list_option_expiries("US.$ARGUMENTS")` → pick 1-2 expiries in the 14-60 DTE window.
- `mcp__moomoo-mcp__get_option_chain_snapshot("US.$ARGUMENTS", expiry_window_days=45, strike_count_each_side=10)` → note IV levels, delta by strike, bid/ask spreads.
- Flag expensive IV (IV > ~40 for mega-caps, > ~60 for small-caps) — long premium is likely a poor EV bet at those levels.

**4. Recent filings.**

- `mcp__edgar-mcp__get_recent_filings_for_ticker("$ARGUMENTS", limit=10)` → scan for the last 10-Q, most recent 8-K, and any proxy / registration.
- If any filing in the last 30 days looks material, `mcp__edgar-mcp__get_filing_text(...)` on that one and extract the key 3-5 bullet takeaways.

**5. Insider activity.**

- `mcp__edgar-mcp__get_insider_transactions("$ARGUMENTS", lookback_days=90)` → count Form 4 buys vs sells. Cluster of insider buys in the last 30 days = meaningful tailwind.

**6. Earnings proximity.**

- From the filings index, compute trading days until next expected earnings (last 10-Q date + ~90 calendar days as a cheap heuristic if no explicit date).
- If within 5 trading days of earnings → flag it and remind: any order within 2 days requires `strategy_label` starting with `earnings_` (R6).

## Output — the "brief"

Produce a short markdown brief:

```
# Research brief: $ARGUMENTS

## Tape
Spot: ... (vs 20DMA: ..., 50DMA: ..., 200DMA: ...), ATR: ..., support: ..., resistance: ...

## Filings (last 30d)
- ...

## Insider (last 90d)
- Buys: N of $X notional; sells: M of $Y notional

## Options (if relevant)
Expiry ...: ATM IV ..., 0.40-delta call mid ..., spread ...

## Past-trade memory
- Lesson (score ...): ...
- Prior trade (ticker=..., strategy=..., outcome=...): ...

## Earnings
Next estimated: ... ; trading days out: ...

## Candidate theses (2-3 ranked by edge)
1. Strategy `pullback_to_200DMA` long stock. Invalidation: close < $X. Timeframe: 2-3 weeks. Est. RR: 2:1.
2. ...

## Recommendation
Proceed to `/size` with thesis #1, or wait for (concrete trigger: ...).
```

## Constraints

- NO orders, NO thesis filing, NO `place_paper_order` calls from `/research`. This step only gathers context.
- If the user's next turn is `/size` or `/enter`, the thesis gets recorded THEN. Keep the candidate theses as draft text for now.
- If past-trade memory contains a direct "avoid this" lesson that conflicts with all candidate theses, say so clearly and recommend skipping the ticker.
