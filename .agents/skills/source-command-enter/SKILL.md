---
name: "source-command-enter"
description: "End-to-end order flow — thesis → sizing → paper order. Any step failing aborts the whole flow."
---

# source-command-enter

Use this skill when the user asks to run the migrated source command `enter`.

## Command Template

# /enter

Chain the full trade-entry flow:

1. Confirm there's a recent `/research` brief + a `/size` proposal on the table.
2. File the thesis via `mcp__journal-mcp__record_thesis`.
3. Place the order via `mcp__moomoo-mcp__place_paper_order` (stock) or `place_paper_option_order` (single-leg).
4. Confirm `posttool_fill_capture` landed the trade + snapshot in the DB.
5. Summarize.

## Preconditions

Before doing anything, check:

- Has the user run `/research <TICKER>` in this session? If no, refuse and tell them to run it first.
- Has the user run `/size <TICKER> <DIRECTION>` in this session? If no, run it now (inline) to get the sized proposal.
- Is MoomooOpenD reachable? Call `mcp__moomoo-mcp__get_account_info` first — if it errors, STOP and tell the user to check OpenD.

## Step 1 — thesis

Call `mcp__journal-mcp__record_thesis` with everything finalized in `/research` + `/size`:

```
ticker=<BARE>,                    # "AAPL", NOT "US.AAPL"
direction=<LONG|SHORT|LONG_CALL|...>,
thesis_text="<2-5 sentences, concrete catalyst>",
invalidation="<concrete trigger: price level, date, or filing event>",
timeframe="<2-4 weeks | overnight | ...>",
expected_return_pct=<number>,
max_loss_pct=<number>,
```

Capture the returned `thesis_id`. If this fails, abort — no order without a thesis.

## Step 2 — place the order

Within 10 minutes of recording the thesis (the hook's freshness window):

**Stock**:
```
mcp__moomoo-mcp__place_paper_order(
  symbol="US.<TICKER>",
  side="BUY" | "SELL",
  qty=<from /size>,
  price=<limit price>,
  thesis_id=<from step 1>,
  stop=<from /size>,
  target=<from /size>,
  strategy_label=<from /size>,
)
```

**Option (single-leg, long premium only)**:
```
mcp__moomoo-mcp__place_paper_option_order(
  option_symbol="US.<OCC>",
  side="BUY",
  contracts=<from /size>,
  price=<debit>,
  thesis_id=<from step 1>,
  strategy_label=<earnings_... or directional_long_call/...>,
  delta=<from research>,
  dte=<from research>,
)
```

### If the hook blocks:

Exit code 2 with stderr explaining which rule. Read it carefully — do NOT retry the identical order. Possible causes:

- Thesis aged out of the 10-min window → re-file.
- Sizing rule failed (R1-R6) → fix inputs. Tell the user which rule and the fix.
- `reject_real_env` caught something → audit the tool_input for `trd_env` / `trading_password` / `REAL`. This is a real anomaly; surface to user loudly.

### If the option tool returns `virtual_fill_suggested=True`:

MoomooOpenD refused the paper option order. Immediately call:

```
mcp__journal-mcp__record_virtual_fill(
  option_symbol=<same>,
  side="BUY",
  contracts=<same>,
  mid_price=<last known mid from /research>,
  thesis_id=<from step 1>,
  strategy_label=<same>,
  reasoning="virtual fill — MoomooOpenD rejected paper option order",
)
```

## Step 3 — verify capture

Call `mcp__journal-mcp__get_open_positions_with_thesis` and confirm the new trade row is present with the correct symbol, qty, entry_price, stop, and thesis_id.

If the trade isn't there: the posttool hook may have crashed (unlikely; it's fail-open). Tell the user and show the hook audit log tail.

## Output

```
# /enter — done

Thesis #<id> filed:
  <ticker> <direction>
  thesis: <text>
  invalidation: <text>
  timeframe: <...>

Order:
  broker_order_id: <id>
  status: SUBMITTING / FILLED / ...
  entry: $<price>  qty: <n>  stop: $<s>  target: $<t>

Captured in journal: trade_id=<id>.

Next: monitor for fill; when target or invalidation hits, call close_trade.
```

## Hard constraints

- Do NOT attempt to bypass the hook. If it blocks, the answer is to fix the inputs, not the hook.
- Do NOT place multi-leg orders from `/enter`. Leg in manually via two `place_paper_option_order` calls on the same thesis_id, documenting in the thesis text.
- Never pass `trd_env` — the tool doesn't accept it and `reject_real_env` hook will kill the call.
- If any step fails, REPORT THE ERROR and STOP. Do not silently proceed to the next step.
