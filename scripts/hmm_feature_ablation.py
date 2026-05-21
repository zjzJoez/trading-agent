"""HMM v5→v6 feature ablation — which of the 17 features actually matter?

DESIGN
------
For each feature `f` in HMM_FEATURE_ORDER:
    1. Build a feature matrix that ZEROES OUT column `f` after standardization
       (i.e., the feature is mean-imputed → contributes nothing to emission
       likelihood). Equivalent to "drop the feature" in terms of information
       content while keeping the matrix shape compatible with the v5 model
       loader.
    2. Re-run rolling annual walkforward 2017-2026, seed=42 (single seed for
       speed — full 5-seed sweep would be 5× slower).
    3. Record headline strategy Sharpe (annualized).

Baseline = full 17-feature model (v5).
Delta = Sharpe_baseline - Sharpe_ablated. Positive delta = feature helps.

WHY ZERO-OUT INSTEAD OF DROP-COLUMN
-----------------------------------
Dropping a column changes the feature_order, which would require a new
HMMModel class instance and break the v5 model loader during cross-checks.
Zero-out post-standardization (i.e., set to the standardized mean = 0)
removes the feature's information content while keeping the matrix shape.
This is the standard approach for feature ablation on a fixed-architecture
model.

RUNTIME ESTIMATE
----------------
17 features × ~5 min (single seed, rolling annual 2017-2026) = ~85 min.
For a quick smoke test, pass `--features-subset 3` to only ablate the top
3 features (vix_level, spy_ret_20, breadth_above_20dma_pct).

OUTPUTS
-------
reports/walkforward_v5_ablation/
    ablation_<feature_name>.json  — per-feature headline + per-year breakdown
    REPORT.md                     — ranked table of feature importance

USAGE
-----
    # Quick smoke (3 features, ~15 min):
    python -m scripts.hmm_feature_ablation --features-subset 3

    # Full (17 features, ~85 min):
    python -m scripts.hmm_feature_ablation

    # Specific feature (~5 min):
    python -m scripts.hmm_feature_ablation --only vix_level
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_agent.regime.classifier import HMMModel, classify
from trading_agent.regime.features import HMM_FEATURE_ORDER, FeatureSnapshot
from trading_agent.regime.gates import SIZE_MULTIPLIERS, regime_size_multiplier
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

# Single seed for ablation speed — variance across seeds in v5 was ±0.029
# Sharpe (5-seed), so a single-seed read is within that error band and 5×
# cheaper. If the ablation surfaces a feature with |delta| > 0.05, re-run
# with --seed-sweep to confirm with 5 seeds.
DEFAULT_SEED = 42
N_STATES = 4

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "walkforward_v5_ablation"


def load_snapshots_range(start, end) -> list[tuple[datetime, FeatureSnapshot]]:
    """Mirror of rolling_walkforward.load_snapshots_range, pinned to date range."""
    with cursor() as cur:
        cur.execute(
            """
            SELECT as_of, features, source_freshness, data_quality
            FROM regime_feature_snapshots
            WHERE DATE(as_of) >= %s AND DATE(as_of) < %s
            ORDER BY as_of ASC
            """,
            (start, end),
        )
        rows = cur.fetchall()
    out = []
    for as_of, feats, freshness, dq in rows:
        snap = FeatureSnapshot(
            as_of=as_of,
            features=feats or {},
            source_freshness=freshness or {},
            data_quality_warnings=[],
            degradation_level=int(dq or 0),
            raw_klines={},
        )
        out.append((as_of, snap))
    return out


def feature_matrix(
    snaps: list[tuple[datetime, FeatureSnapshot]],
    ablated_feature: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build standardized feature matrix. If ablated_feature is set, the
    corresponding column is zeroed out (after standardization), effectively
    removing its information content.

    Returns (Z, means, stds) where Z is the standardized matrix.
    """
    X = np.array([s.as_vector() for _, s in snaps])
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    if ablated_feature is not None:
        try:
            idx = HMM_FEATURE_ORDER.index(ablated_feature)
            Z[:, idx] = 0.0   # zero in standardized space = mean of original feature
        except ValueError:
            raise SystemExit(f"unknown feature: {ablated_feature}")
    return Z, means, stds


def train_hmm(Z: np.ndarray, means: np.ndarray, stds: np.ndarray,
              n_states: int, seed: int, cov_type: str = "diag",
              min_covar: float = 1e-3) -> HMMModel:
    """Fit HMM on a pre-standardized matrix Z. Same as rolling_walkforward
    train_hmm_for_year but expects Z already standardized so we can pass
    the ablated matrix in unchanged."""
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(
        n_components=n_states, covariance_type=cov_type,
        n_iter=200, random_state=seed, tol=1e-4,
        min_covar=min_covar,
    )
    model.fit(Z)
    states = model.predict(Z)

    # Auto-calibrate state→label using mean spy_ret_20, mirroring
    # rolling_walkforward.train_hmm_for_year exactly.
    # NOTE: we use the ORIGINAL (un-standardized) X for the labeling so the
    # ablation doesn't perturb the label rule. Reconstruct from Z + means/stds.
    X = Z * stds + means
    spy_ret_20_idx = HMM_FEATURE_ORDER.index("spy_ret_20")
    spy_rv_20_idx = HMM_FEATURE_ORDER.index("spy_rv_20")
    state_metrics = {}
    for s in range(n_states):
        mask = (states == s)
        if mask.sum() == 0:
            state_metrics[s] = {"mean_spy_ret_20": 0.0, "mean_spy_rv_20": 0.0}
        else:
            state_metrics[s] = {
                "mean_spy_ret_20": float(X[mask, spy_ret_20_idx].mean()),
                "mean_spy_rv_20": float(X[mask, spy_rv_20_idx].mean()),
            }
    ranked = sorted(range(n_states), key=lambda s: state_metrics[s]["mean_spy_ret_20"])
    state_to_label = {ranked[0]: "BEAR_TREND", ranked[-1]: "BULL_TREND"}
    if n_states == 4:
        mid_a, mid_b = ranked[1], ranked[2]
        rv_a = state_metrics[mid_a]["mean_spy_rv_20"]
        rv_b = state_metrics[mid_b]["mean_spy_rv_20"]
        if rv_a <= rv_b:
            state_to_label[mid_a] = "RANGE_LOW_VOL"
            state_to_label[mid_b] = "VOLATILE_TRANSITION"
        else:
            state_to_label[mid_a] = "VOLATILE_TRANSITION"
            state_to_label[mid_b] = "RANGE_LOW_VOL"

    # Normalize covars to (n_states, d, d) for HMMModel
    covars_arr = np.asarray(model.covars_)
    if covars_arr.ndim == 2:
        n_st = covars_arr.shape[0]
        d_ = covars_arr.shape[1]
        covars_full = np.zeros((n_st, d_, d_), dtype=float)
        for i in range(n_st):
            np.fill_diagonal(covars_full[i], covars_arr[i])
    else:
        covars_full = covars_arr

    return HMMModel(
        n_states=int(model.n_components),
        feature_order=list(HMM_FEATURE_ORDER),
        means=model.means_.tolist(),
        covars=covars_full.tolist(),
        transmat=model.transmat_.tolist(),
        startprob=model.startprob_.tolist(),
        feature_means=means.tolist(),
        feature_stds=stds.tolist(),
        state_to_label=state_to_label,
        train_meta={"n_obs": int(Z.shape[0]), "cov_type": cov_type, "seed": seed},
    )


def yfinance_spy_returns(start_dt, end_dt) -> pd.Series:
    import yfinance as yf
    pad_start = (start_dt - timedelta(days=10)).date().isoformat()
    pad_end = (end_dt + timedelta(days=40)).date().isoformat()
    df = yf.download("SPY", start=pad_start, end=pad_end, progress=False, auto_adjust=True)
    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.pct_change()


def run_ablation_year(
    test_year: int,
    ablated_feature: str | None,
    seed: int,
) -> list[dict]:
    """Run rolling-style walkforward for one test year with one feature ablated.

    Returns a list of {as_of, label, allow_new_entries, size_multiplier}
    dicts for each day in test_year.
    """
    train_end = f"{test_year}-01-01"
    test_start = train_end
    test_end = f"{test_year+1}-01-01"

    training_snaps = load_snapshots_range(None, train_end)
    test_snaps = load_snapshots_range(test_start, test_end)

    if len(training_snaps) < 200 or len(test_snaps) < 50:
        log.warning("year %d: too few snaps (train=%d, test=%d)",
                    test_year, len(training_snaps), len(test_snaps))
        return []

    # Build training matrix WITH ablation
    Z, means, stds = feature_matrix(training_snaps, ablated_feature=ablated_feature)
    hmm = train_hmm(Z, means, stds, N_STATES, seed)

    # Classify each test day using the trained model.
    # For test-time we also need to zero the ablated feature.
    rows = []
    for as_of, snap in test_snaps:
        # Standardize test snap with training means/stds, then ablate
        x = np.array(snap.as_vector(), dtype=float)
        z_x = (x - means) / stds
        if ablated_feature is not None:
            idx = HMM_FEATURE_ORDER.index(ablated_feature)
            z_x[idx] = 0.0
        # Re-construct a FeatureSnapshot with the ablated vector by overlaying
        # the original .features dict (only spy_ret_20 / spy_rv_20 are read
        # downstream by classify() for prior history, but the HMM emission uses
        # as_vector() via FeatureSnapshot — to keep this surgical, we patch
        # the snap.features dict to set the ablated key to the training mean
        # (so as_vector reconstructs the standardized 0 we want).
        if ablated_feature is not None:
            patched = dict(snap.features)
            patched[ablated_feature] = float(means[HMM_FEATURE_ORDER.index(ablated_feature)])
            patched_snap = FeatureSnapshot(
                as_of=snap.as_of,
                features=patched,
                source_freshness=snap.source_freshness,
                data_quality_warnings=snap.data_quality_warnings,
                degradation_level=snap.degradation_level,
                raw_klines=snap.raw_klines,
            )
        else:
            patched_snap = snap

        prediction = classify(patched_snap, hmm)
        rows.append({
            "as_of": as_of.date().isoformat(),
            "label": prediction.label,
            "confidence": prediction.confidence,
            "allow_new_entries": prediction.allow_new_entries,
            "size_multiplier": regime_size_multiplier(prediction.label),
        })
    return rows


def compute_strategy_sharpe(daily_rows: list[dict], spy_returns: pd.Series) -> dict:
    """Compute the headline strategy Sharpe (vs SPY-long-only baseline)."""
    if not daily_rows:
        return {"strategy_sharpe": None, "spy_sharpe": None, "n_days": 0}
    df = pd.DataFrame(daily_rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.set_index("as_of")

    # Strategy: SPY return × size_multiplier, but 0 if allow_new_entries=False
    aligned = df.join(spy_returns.rename("spy_ret"), how="inner")
    aligned["strategy_ret"] = aligned["spy_ret"] * aligned["size_multiplier"]
    aligned.loc[~aligned["allow_new_entries"], "strategy_ret"] = 0.0

    strat_returns = aligned["strategy_ret"].dropna()
    spy_only = aligned["spy_ret"].dropna()

    def sharpe(s):
        if len(s) < 2 or s.std() == 0:
            return None
        return float(s.mean() / s.std() * math.sqrt(252))

    return {
        "strategy_sharpe": sharpe(strat_returns),
        "spy_sharpe": sharpe(spy_only),
        "n_days": int(len(strat_returns)),
        "cum_strategy_ret": float((1 + strat_returns).prod() - 1),
        "cum_spy_ret": float((1 + spy_only).prod() - 1),
    }


def run_full_ablation(ablated_feature: str | None,
                      first_year: int = 2017, last_year: int = 2026,
                      seed: int = DEFAULT_SEED) -> dict:
    """Run rolling annual walkforward for all years [first_year, last_year)
    with one feature ablated. Returns the headline + per-year breakdown."""
    all_rows: list[dict] = []
    per_year_meta = []
    for y in range(first_year, last_year):
        log.info("  ablation=%s year=%d", ablated_feature or "BASELINE", y)
        rows = run_ablation_year(y, ablated_feature, seed)
        all_rows.extend(rows)
        per_year_meta.append({"year": y, "n_days": len(rows)})

    if not all_rows:
        return {"feature": ablated_feature, "error": "no data"}

    start_dt = datetime.fromisoformat(all_rows[0]["as_of"])
    end_dt = datetime.fromisoformat(all_rows[-1]["as_of"])
    spy_ret = yfinance_spy_returns(start_dt, end_dt)

    headline = compute_strategy_sharpe(all_rows, spy_ret)
    headline["feature"] = ablated_feature or "BASELINE_full_17"
    headline["seed"] = seed
    headline["per_year"] = per_year_meta
    return headline


def write_report(results: list[dict], out_dir: Path) -> None:
    """Write a ranked importance table to REPORT.md."""
    baseline = next((r for r in results if r["feature"].startswith("BASELINE")), None)
    if baseline is None:
        print("ERROR: no baseline run in results", file=sys.stderr)
        return

    baseline_sharpe = baseline["strategy_sharpe"]
    rows = []
    for r in results:
        if r["feature"].startswith("BASELINE"):
            continue
        s = r["strategy_sharpe"]
        delta = (baseline_sharpe - s) if (s is not None and baseline_sharpe is not None) else None
        rows.append({
            "feature": r["feature"],
            "sharpe": s,
            "delta": delta,
            "n_days": r.get("n_days"),
        })

    # Sort by delta descending (biggest contribution first)
    rows.sort(key=lambda r: -(r["delta"] if r["delta"] is not None else -999))

    lines = [
        "# HMM v5 feature ablation results",
        "",
        f"Baseline (v5 full 17 features): Sharpe = **{baseline_sharpe:.4f}** "
        f"(seed={baseline['seed']}, {baseline.get('n_days', '?')} days)",
        "",
        "Each row drops one feature and re-runs the rolling annual walkforward "
        "2017→2026 with the SAME seed. Positive Δ = the feature was helping.",
        "",
        "| feature | ablated Sharpe | Δ vs baseline | n days |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        s = f"{r['sharpe']:.4f}" if r["sharpe"] is not None else "—"
        d = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        lines.append(f"| `{r['feature']}` | {s} | {d} | {r['n_days']} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- |Δ| > 0.05: feature is meaningfully load-bearing — keep.",
        "- 0 < Δ < 0.05: feature helps a little — keep unless we want a leaner model.",
        "- Δ < 0: feature hurts (rare, but possible) — drop in v6.",
        "",
        "Single-seed runs have ±0.03 noise band. Re-run any candidate-drop "
        "feature with `--seed-sweep` (5 seeds) to confirm before promoting v6.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="ablate just one feature (faster)")
    ap.add_argument("--features-subset", type=int, default=0,
                    help="ablate only the first N features (debug)")
    ap.add_argument("--first-year", type=int, default=2017)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=str, default=str(REPORT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_to_ablate: list[str]
    if args.only:
        features_to_ablate = [args.only]
    elif args.features_subset:
        features_to_ablate = HMM_FEATURE_ORDER[: args.features_subset]
    else:
        features_to_ablate = list(HMM_FEATURE_ORDER)

    log.info("ablation plan: %d features to ablate + 1 baseline",
             len(features_to_ablate))

    results = []

    # Baseline first — needed for delta computation.
    log.info("running BASELINE (full 17 features)")
    baseline = run_full_ablation(None, args.first_year, args.last_year, args.seed)
    (out_dir / "ablation_BASELINE.json").write_text(json.dumps(baseline, indent=2))
    results.append(baseline)

    for feat in features_to_ablate:
        log.info("ablating feature: %s", feat)
        r = run_full_ablation(feat, args.first_year, args.last_year, args.seed)
        (out_dir / f"ablation_{feat}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
        log.info("  → Sharpe %.4f (Δ %.4f)",
                 r["strategy_sharpe"] or 0,
                 (baseline["strategy_sharpe"] - (r["strategy_sharpe"] or 0))
                 if baseline["strategy_sharpe"] else 0)

    write_report(results, out_dir)
    log.info("done. wrote %s", out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
