---
description: Morning scan — macro regime (SPY/QQQ/VIX) + watchlist quotes + recent filings, ranked by actionability
---

# /scan — morning scan

Run this pre-market or at the open to survey the macro tape and a short watchlist. Output is a ranked list of candidates for deeper `/research`.

## Watchlist

Default watchlist (edit inline if the user asks for a different set):

`SPY, QQQ, IWM, VIX, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, XOM, UNH, LLY`

If the user passes tickers after `/scan` (e.g., `/scan AAPL NVDA TSM`), use that list instead plus SPY/QQQ/VIX as regime context.

## Procedure

1. **Macro regime** — call `mcp__moomoo-mcp__get_quote` on `["US.SPY", "US.QQQ", "US.VIX"]` (VIX code may be `US.VIX` or the near futures; try both). Report:
   - SPY close vs 20DMA / 50DMA (use `get_historical_kline` for the MAs — last 60 daily bars).
   - QQQ close vs 20DMA / 50DMA.
   - VIX absolute + 5-day change. >20 rising = risk-off; <15 sliding = complacent.
   - Regime tag: **RISK_ON** (SPY > 20DMA > 50DMA and VIX trending down), **RISK_OFF** (opposite), **MIXED** (anything else).

2. **Watchlist quotes** — one `get_quote` call with all watchlist symbols. For each: last, prev_close, %chg, volume.

3. **Recent filings** — for each watchlist ticker, call `mcp__edgar-mcp__get_recent_filings_for_ticker(ticker, limit=3)`. Flag anything filed in the last 7 days (especially 8-K, 10-Q, Form 4 clusters).

4. **Past-trade memory** — for the user's watchlist, `mcp__journal-mcp__search_past_trades(ticker)` to surface any relevant lessons or prior strategy outcomes.

5. **Rank candidates** by:
   - Intraday %chg × |volume / 20-day avg volume|
   - Recent filing relevance (8-K with material news = high)
   - Whether `search_past_trades` surfaced a matching successful strategy
   - Alignment with regime (longs in RISK_ON, flat/short in RISK_OFF)

## Output format

Markdown table ordered by rank:

```
| Rank | Ticker | % | Vol / 20d | Filing | Past edge | Notes |
|------|--------|---|-----------|--------|-----------|-------|
| 1    | NVDA   | +2.1 | 1.8x   | 8-K 3d ago (guide) | pullback_to_50DMA 3W/1L | above 20DMA on volume |
...
```

Then, 2-3 sentence regime summary. Then a short "worth /research-ing" list (1-5 names).

## Constraints

- Do NOT place any orders from `/scan`. This is pure reconnaissance.
- Do NOT file theses from `/scan`. Theses belong in `/research` → `/enter`.
- If any MCP tool errors, report it inline and continue — partial data > no data.
- Keep the whole scan under 3 MCP rounds if possible (batch `get_quote` calls).
