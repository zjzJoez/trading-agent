# R5_option_policy review — why NVDA was blocked 2026-05-21

## TL;DR

R5 is the option-policy hard sizing gate. The 2026-05-21 NVDA proposal
(2x 6/12 220C, "breakout_retest", entry $2.59ish) blocked at R5 **before
the risk council ran** — sizing reported `infeasible: R5_option_policy`.

Below: rule definition, which sub-check fired for NVDA, and one
candidate tightening + one candidate loosening to consider once we exit
burn-in. **Recommendation: do not change R5 right now — observe 5-10
more proposals first to see which sub-check fires most.**

## Rule definition

`src/trading_agent/sizing.py:12-50` (constants) and `:212-245` (check).

```python
MAX_OPTION_NOTIONAL_PCT = 0.01      # 1 % of equity per option trade
OPTION_DTE_MIN = 14                  # not too near-dated (gamma blowup)
OPTION_DTE_MAX = 60                  # not too far-dated (theta efficient)
OPTION_DELTA_MIN = 0.30              # not deep-OTM lottery tickets
OPTION_DELTA_MAX = 0.55              # not deep-ITM (stock substitutes)
```

Block conditions (every one of these emits a `SizingViolation` with
`severity='block'`, which makes the trade infeasible):

| sub-check | criterion | rationale |
|---|---|---|
| **side** | must be BUY | MVP is long-premium only (R5 is THE filter that enforces this) |
| **dte** | 14 ≤ DTE ≤ 60 | avoid 0-7d gamma and 90d+ theta-bleed |
| **\|delta\|** | 0.30 ≤ \|δ\| ≤ 0.55 | avoid lottery tickets (δ < 0.30) and stock-substitutes (δ > 0.55) |
| **notional** | qty × multiplier × price ≤ 1% × equity | hard cap per trade |

## What blocked NVDA on 2026-05-21

From the event log of `candidate_entry:candidate_entry_20260521_105454_565482c1`:

```
proposal_built  → qty: 2, symbol: US.NVDA260612C220000, strategy: breakout_retest
tiny_paper_cap_applied → cap 1, requested 2.0
deterministic_sizing infeasible → blockers: ["R5_option_policy"]
```

The proposal was a 6/12 220C with NVDA spot ~$223 — strike just 1.3% OTM.
**DTE = 22 days** (within [14, 60]), **side = BUY** ✓. The block was
almost certainly **`|delta|` > 0.55**: a 220 strike against $223 spot
with 22 DTE has |δ| ≈ 0.55-0.62 depending on IV.

The `tiny_paper_cap_applied` step (capping 2 → 1 contract because we're
still in tiny-paper soak phase) doesn't change delta — the option's
greeks are independent of contract count. So even at 1 contract, the
delta sub-check still blocks.

**The notional sub-check was not the issue.** Even at 2 contracts ×
$100 × ~$5 = $1,000, that's ≪ 1% × $1.01M = $10,100. R5 has plenty of
notional headroom in the current account.

## Why R5 exists where it is (and not just in the LLM council)

R5 sits **upstream of the LLM council** — it's a deterministic guardrail,
not a debate topic. Three reasons:

1. **Cost**: every council invocation runs 3 LLMs in parallel. Blocking
   structurally bad option trades before the council saves $0.30-1 per
   skipped run.
2. **Auditability**: if a position blows up, "R5 said no" is a one-line
   answer; "the council judged the structure acceptable but..." is not.
3. **Soak / canary protection**: during the first weeks of any model
   change, you want a rigid wall; the LLM may converge on bad option
   structures it likes but the operator doesn't.

## Should R5 be changed?

Two candidate tweaks, both contingent on observing more proposals after
burn-in clears (currently 5/15):

### Candidate A — widen `OPTION_DELTA_MAX` to 0.60

**Argument for**: A 0.55-0.60 delta call is still long-premium long-vol;
it's not a stock substitute. Currently R5 blocks all near-the-money
strikes when spot is near the next round-number strike (common pattern
for breakout setups, which the bull researcher tends to suggest).

**Argument against**: At δ ≈ 0.60 you're paying ~10% extrinsic and 90%
intrinsic — you've essentially bought a leveraged stock position with
~half the leverage of a futures contract. If that's the desired exposure
the proposal should be a stock long, not a call.

**Empirical test before deciding**: count, over the next 14 days, how
many proposals R5-block on delta and what the |δ| distribution looks
like. The `/api/risk/r5-gate` endpoint added today exposes this.

### Candidate B — tighten `MAX_OPTION_NOTIONAL_PCT` to 0.005 (0.5%)

**Argument for**: We're paper, so per-trade risk is play money. But the
production-target heat metric (sum of position notionals as % of equity)
should match institutional norms (1-2% per name, max 6-8% portfolio
heat). 1% per option trade with 6-8 simultaneous proposals could spike
heat too fast.

**Argument against**: Per-trade heat caps belong in R1 (position size)
and R6 (heat budget), not R5 (structure). R5's notional cap is a
backstop, not the primary tool.

**Empirical test**: review `risk_snapshot.heat_pct` distribution over 30
proposals. If max heat stays < 5%, leave R5 alone.

## Observability added today (auto-deployed)

`GET /api/risk/r5-gate` returns:
```json
{
  "thresholds": {
    "side": "BUY",
    "dte_min": 14, "dte_max": 60,
    "delta_abs_min": 0.30, "delta_abs_max": 0.55,
    "max_notional_pct_equity": 0.01
  },
  "recent_blocks_14d": [
    {"ts": "...", "ticker": "NVDA", "blockers": ["R5_option_policy"], ...},
    ...
  ],
  "block_count_14d": N
}
```

After 14 days of accumulated data the mobile dashboard can show a
sub-check histogram ("R5 fires: 8x delta, 2x dte, 0x notional, 0x side")
and the question of whether to widen δ becomes data-driven.

## Action items

1. ✅ Add `/api/risk/r5-gate` for empirical visibility (done in this commit).
2. ☐ Wait until **5+ R5 blocks** are logged (~2 weeks at current dispatch rate).
3. ☐ Re-evaluate Candidate A (widen `OPTION_DELTA_MAX → 0.60`) with the data.
4. ☐ If R5 dominates the rejection mix → consider adding a "soft" warn tier
     before the hard block, so the LLM council can override in unusual setups.
