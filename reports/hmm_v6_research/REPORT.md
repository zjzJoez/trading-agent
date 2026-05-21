# HMM v6 research — framework + open questions

## Where we are (v5 baseline)

Per `reports/walkforward_v5_longhistory/REPORT.md`:

- Training: 2007-04 → 2026-05 (4760 obs)
- OOS rolling annual 2017-2026: **Sharpe 0.908 ± 0.029** vs SPY 0.837
- Edge: **+0.071 Sharpe over SPY-long-only baseline**
- 4 hidden states, diag-cov, 17 features (full list in `src/trading_agent/regime/features.py:102`)

The +0.071 edge is **single-OOS-window ~2σ**. Not yet robust. Three plausible
levers to push v6 forward:

1. **Feature ablation** — drop low-value features, see if Sharpe stays.
   A leaner model = less overfit + faster training.
2. **New features** — add macroeconomic variables not currently in scope
   (Fed funds rate path, term premium, etc.).
3. **Architecture changes** — Bayesian HMM (random transition matrix),
   regime-switching with duration component, hierarchical (sectors).

This research plan tackles #1 first because it's the cheapest information.

## Step 1 — Feature ablation (scripts/hmm_feature_ablation.py)

For each of 17 features, zero it out (post-standardization → no information
content) and re-run the full rolling-annual walkforward 2017-2026. The
delta vs the full-17 baseline tells us how load-bearing each feature is.

**Runtime**: ~5 min per feature × 17 features + 1 baseline = ~90 minutes
single-seed. Cheap enough to be a Saturday job, not a session task.

**Run it**:
```bash
cd /home/ubuntu/trading-agent   # or local worktree
python -m scripts.hmm_feature_ablation
# → reports/walkforward_v5_ablation/REPORT.md
```

**Quick smoke** (3 features, ~15 min):
```bash
python -m scripts.hmm_feature_ablation --features-subset 3
```

**Single feature** (~5 min):
```bash
python -m scripts.hmm_feature_ablation --only vix_level
```

### Expected output format (`reports/walkforward_v5_ablation/REPORT.md`)

```markdown
# HMM v5 feature ablation results

Baseline (v5 full 17 features): Sharpe = 0.9082

| feature                       | ablated Sharpe | Δ vs baseline | n days |
|-------------------------------|---------------:|--------------:|-------:|
| `vix_level`                   |         0.8210 |       +0.0872 |   2369 |
| `spy_ret_20`                  |         0.8540 |       +0.0542 |   2369 |
| `breadth_above_20dma_pct`     |         0.8801 |       +0.0281 |   2369 |
| `avg_pairwise_corr_20`        |         0.8956 |       +0.0126 |   2369 |
| `vix_pctl_252`                |         0.9078 |       +0.0004 |   2369 |
| `yield_10y2y_slope_proxy`     |         0.9120 |       -0.0038 |   2369 |
| ...                           |            ... |           ... |    ... |
```

(Numbers above are hypothetical — they're the expected SHAPE of the output,
not actual results. Run the script to get real numbers.)

### Interpretation rules

- **|Δ| > 0.05**: load-bearing. Keep.
- **0 < Δ < 0.05**: helpful but not critical. Keep unless we want a leaner v6.
- **Δ < 0 (Sharpe goes UP without the feature)**: feature is *noise*. Drop in v6.
- **Δ within ±0.03**: indistinguishable from single-seed noise (±0.029 from v5).
  Re-run with `--seed-sweep` to confirm before acting.

## Step 2 — Hypothesis-driven candidate adds

Hypothesis: today's 17 features capture price/vol/breadth/credit/dollar but
have ZERO macro-policy state. Recent regime-research literature (e.g., Cai et al.
2024) shows fed-policy-stance regimes are orthogonal to market regimes.

Candidates worth trying (in order of expected ROI):

1. **Fed funds rate 6-month change** — captures "tightening cycle" regimes
2. **Term premium (ACM model)** — captures recession-pricing in long rates
3. **High-yield credit spread 60-day percentile** — already have `hyg_mom_20`
   but a level-percentile carries different info than momentum
4. **VIX-of-VIX (VVIX)** — captures "vol uncertainty"
5. **SPX dispersion proxy** — sector-pair correlation spread

Data sources: FRED API (free), CBOE (free for VVIX). Backfill cost: ~1 day
of work to add to `src/trading_agent/regime/features.py`.

Each new feature: re-run ablation to confirm |Δ| > 0.03 vs no-feature model.

## Step 3 — Architecture (longer-horizon)

After Steps 1 and 2 settle: consider a **Bayesian HMM** with prior on the
transition matrix. The current MAP estimate of `transmat` gives sharp
state boundaries; a Bayesian posterior would smooth transitions in regions
of low training data (e.g. the 2020 COVID single-day collapse, which v5 saw
exactly once in training).

Implementation cost: high. Defer until Steps 1-2 show diminishing returns.

## Decision gates

- **v6 is worth promoting** if its 5-seed Sharpe is ≥ v5 + 0.04 with the
  same SPY benchmark, i.e., > 0.95 ± 0.029. That's a ~1.4σ improvement —
  the smallest signal worth shipping.
- **v6 is worth canary-ing** if its 5-seed Sharpe is ≥ v5 (no regression),
  AND it strictly removes ≥ 2 features (simpler is better at parity).
- **v6 changes only if** ablation Step 1 shows a feature with Δ ≤ -0.03
  (i.e., it's actively hurting), OR Step 2 adds a feature with Δ ≥ +0.04.

## Files added this session

- `scripts/hmm_feature_ablation.py` — the ablation runner
- `reports/hmm_v6_research/REPORT.md` — this document
- `reports/walkforward_v5_ablation/` — will be populated on first script run

## Action items

1. ☐ Run `python -m scripts.hmm_feature_ablation` on EC2 (Saturday job — won't
   interfere with weekday brain runs)
2. ☐ Review `reports/walkforward_v5_ablation/REPORT.md` once it lands
3. ☐ For any feature with |Δ| > 0.03 OR seemingly dead: confirm with 5-seed sweep
4. ☐ Decide: lean v6 (drop noise features) or fat v6 (add macro features)?
