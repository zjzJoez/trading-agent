# Pre-registered decision rule: DISPATCH_MIN_SCORE stays 0.50

**Date:** 2026-07-20
**Status:** accepted (pre-registered; docs-only, no behavior change)
**Owner:** revival plan Week 1, Step 6c
**Code:** `DISPATCH_MIN_SCORE = 0.50` in
`src/trading_agent/graph/nodes/premarket_nodes.py`

## Decision

The premarket-scan dispatch threshold **stays at 0.50 for now**. No change
ships with this commit — this document exists to fix the evaluation
procedure and the decision rule *in advance*, so the threshold cannot be
nudged after seeing the data.

## Context

The threshold has already walked 0.7 → 0.6 → 0.55 → 0.50, each step
justified post-hoc by that day's missed candidate (see the history comment
above `DISPATCH_MIN_SCORE`). That is exactly the pattern a pre-registered
rule prevents. The scout's scoring is also about to change under it: the
rvol fix (scout prompt ranks on relative volume, not absolute share count)
alters the score distribution, so any threshold judgment made on pre-fix
scores is stale on arrival.

## Evaluation procedure

1. Wait for the rvol fix to deploy to the EC2 executing system.
2. From the rvol-fix deploy date, collect **10 trading days** of paired
   scan logs: the digest scan scores AND the executing scan scores
   (`agent_events`: `rank_candidates` / dispatch events), so digest-vs-
   executing divergence is visible alongside the dispatch rate.
3. At the end of the window, compute the executing-scan dispatch rate in
   dispatches/week (dispatches ÷ 2, given 10 trading days = 2 weeks).

## Decision rule (fixed in advance)

| Executing-scan dispatch rate | Action |
| --- | --- |
| 2–5 dispatches/week (inclusive) | **Keep 0.50** |
| < 2 dispatches/week | **Lower to 0.45** |
| > 5 dispatches/week | **Raise to 0.55** |

Any further change beyond this single adjustment requires a **new
pre-registered rule** — no post-hoc threshold moves.

## Evaluation date

10 trading days after the rvol fix deploys. As of this writing the rvol
fix has not yet deployed; when it does, record the deploy date here and
evaluate 10 trading days later (e.g. a Monday 2026-07-27 deploy →
evaluate at close on Friday 2026-08-07).

- rvol fix deployed: _(fill in on deploy)_
- evaluation due: _(deploy date + 10 trading days)_
