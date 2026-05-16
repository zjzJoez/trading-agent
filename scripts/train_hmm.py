"""Train a 4-state Gaussian HMM on backfilled regime features.

End-state: a serialized HMMModel JSON (no DB write) + a "fingerprint"
report for each hidden state showing mean feature values, mean SPY/VIX
metrics, and state occupancy. The fingerprint is what a human uses to
calibrate `state_to_label` — there is NO automatic mapping because the
typical failure mode is two states both looking "bull-ish" with subtle
differences (vol high vs low) and an auto-labeler will get this wrong.

Workflow:
    1. backfill_history.py     (populates regime_feature_snapshots)
    2. train_hmm.py            (fits HMM, dumps fingerprint, NO DB insert)
    3. human review of fingerprint, edit state_to_label dict in YAML below
    4. promote_hmm.py          (validates mapping + INSERTs ACTIVE row)

Why this split: calibration error means weeks of bad regime signals
because state assignments affect every premarket dispatch. The 5-minute
manual review step is cheap insurance.

Usage:
    python -m scripts.train_hmm --n-states 4 --out /tmp/hmm_v1.json
    # → reads ALL rows from regime_feature_snapshots, fits HMM,
    #   writes serialized model to /tmp/hmm_v1.json and prints fingerprint.

Fingerprint output format:
    Per state: 17-feature mean vector + occupancy fraction +
    [mean SPY ret_20, mean spy_rv_20, mean VIX, mean breadth] +
    SPY-return-distribution-by-state (helps see if a state captures
    "bull" days vs "bear" days even when feature means look similar).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from trading_agent.regime.classifier import HMMModel
from trading_agent.regime.features import HMM_FEATURE_ORDER
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def load_features_matrix(
    train_end_date: str | None = None,
) -> tuple[pd.DataFrame, list[datetime]]:
    """Return (features_df, as_of_list) — DF indexed by as_of, columns =
    HMM_FEATURE_ORDER. Skip rows with degradation_level≥2 (safe-mode).

    `train_end_date` (YYYY-MM-DD inclusive) caps the training window. This
    is the *only* way to produce an HMM whose `train_end < today` so the
    out-of-sample backtest is honestly out-of-sample. Without this flag,
    the function reads every snapshot up to NOW() and the resulting model
    has been fit to the entire history.
    """
    with cursor() as cur:
        if train_end_date:
            cur.execute(
                """
                SELECT as_of, features, data_quality
                FROM regime_feature_snapshots
                WHERE DATE(as_of) <= %s
                ORDER BY as_of ASC
                """,
                (train_end_date,),
            )
        else:
            cur.execute(
                """
                SELECT as_of, features, data_quality
                FROM regime_feature_snapshots
                ORDER BY as_of ASC
                """
            )
        rows = cur.fetchall()

    log.info("Loaded %d feature snapshots from DB", len(rows))
    if len(rows) < 100:
        log.error("Insufficient training data: %d rows (need ≥500 for stable HMM)",
                  len(rows))
        sys.exit(1)

    data = []
    timestamps = []
    skipped_dq = 0
    for as_of, feats_blob, dq_blob in rows:
        if (dq_blob or {}).get("degradation_level", 0) >= 2:
            skipped_dq += 1
            continue
        feats = feats_blob or {}
        # Ensure all 17 features are present; missing → 0.0 (matches
        # FeatureSnapshot.as_vector behavior).
        row = [float(feats.get(k, 0.0)) for k in HMM_FEATURE_ORDER]
        data.append(row)
        timestamps.append(as_of)

    if skipped_dq:
        log.info("Skipped %d rows with degradation_level≥2", skipped_dq)

    df = pd.DataFrame(data, index=timestamps, columns=HMM_FEATURE_ORDER)
    return df, timestamps


def fit_hmm(features_df: pd.DataFrame, n_states: int = 4, n_iter: int = 200,
            random_state: int = 42, cov_type: str = "diag",
            min_covar: float = 1e-3) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
    """Standardize features, fit HMM, return (model, feature_means, feature_stds).

    Standardization is critical: feature scales differ wildly (spy_ret_20 ∈
    [-0.3, 0.3], vix_level ∈ [10, 80]). Without it, vix_level dominates the
    Mahalanobis distance and the HMM collapses to vol-only states.

    cov_type default: 'diag' (not 'full'). Full-covariance with 17 features
    repeatedly broke EM convergence in v1/v2/v3 (log-likelihood would
    decrease after iter 2-3, abort). Diagonal covariance is standard in
    finance regime-HMM literature (Hamilton 1989, Ang & Bekaert 2002): per-
    state feature means are the dominant signal; the inter-feature
    covariance is hard to estimate with limited regime samples.

    min_covar default: 1e-3 (hmmlearn default). Larger values regularize the
    emission covariance, preventing collapse to a point mass; bump to 1e-2
    or 5e-3 if EM still doesn't converge on `diag`.
    """
    X = features_df.values
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid /0 on constant features
    Z = (X - means) / stds

    log.info("Fitting GaussianHMM: n_states=%d, n_features=%d, n_obs=%d, cov=%s, min_covar=%g",
             n_states, Z.shape[1], Z.shape[0], cov_type, min_covar)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cov_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
        min_covar=min_covar,
        verbose=False,
    )
    model.fit(Z)
    log.info("Fit converged at iteration %d, score=%.2f",
             getattr(model.monitor_, "iter", -1), model.score(Z))
    return model, means, stds


def fingerprint(model: GaussianHMM, features_df: pd.DataFrame,
                feature_means: np.ndarray, feature_stds: np.ndarray) -> dict:
    """Return per-state fingerprint for calibration.

    For each state we report:
      - mean feature vector in ORIGINAL scale (de-standardized)
      - occupancy fraction across full history
      - aggregate behavioral metrics: mean SPY 20-day return, mean RV_20,
        mean VIX, mean breadth, mean correlation — what a human reads to
        decide if state X is "bull trend" or "bear" etc.
      - SPY return distribution percentiles to detect e.g. "state with
        symmetric ±large moves" (VOLATILE) vs "state with skewed positive"
        (BULL_TREND).
    """
    X = features_df.values
    Z = (X - feature_means) / feature_stds
    states = model.predict(Z)  # Viterbi

    out = {"n_states": int(model.n_components),
           "n_obs": int(len(states)),
           "feature_order": list(HMM_FEATURE_ORDER),
           "states": {}}

    for s in range(model.n_components):
        mask = (states == s)
        n = int(mask.sum())
        if n == 0:
            out["states"][s] = {"empty": True}
            continue
        # De-standardize means back to original scale
        state_mean_z = model.means_[s]
        state_mean_orig = state_mean_z * feature_stds + feature_means
        # Pull notable behavioral features
        feat_dict = {k: float(v) for k, v in zip(HMM_FEATURE_ORDER, state_mean_orig)}

        # SPY return distribution during this state
        spy_ret20_values = features_df.loc[mask, "spy_ret_20"].values
        out["states"][s] = {
            "n_obs": n,
            "occupancy_frac": float(n / len(states)),
            "feature_means": feat_dict,
            "spy_ret_20_pct": {
                "p10": float(np.percentile(spy_ret20_values, 10)),
                "p50": float(np.percentile(spy_ret20_values, 50)),
                "p90": float(np.percentile(spy_ret20_values, 90)),
            },
            "summary": _summarize_state(feat_dict),
        }

    out["transmat"] = model.transmat_.tolist()
    out["startprob"] = model.startprob_.tolist()
    return out


def _summarize_state(feats: dict) -> str:
    """One-line English summary of a state's regime character.

    This is NOT a calibration decision — it's a hint to whoever is doing
    the human review. The mapping `state→regime` still needs explicit
    human confirmation.
    """
    spy_ret20 = feats.get("spy_ret_20", 0.0)
    vix = feats.get("vix_level", 18.0)
    rv = feats.get("spy_rv_20", 0.15)
    above_50dma = feats.get("spy_above_50dma", 0.5)
    corr = feats.get("avg_pairwise_corr_20", 0.5)

    if vix > 28 or rv > 0.25 or corr > 0.7:
        if spy_ret20 < -0.05:
            return "high-vol, falling — likely VOLATILE_TRANSITION or BEAR_TREND"
        return "high-vol, mixed direction — likely VOLATILE_TRANSITION"
    if spy_ret20 > 0.03 and above_50dma > 0.6 and vix < 20:
        return "rising, low-vol, above MAs — likely BULL_TREND"
    if spy_ret20 < -0.03 and above_50dma < 0.4:
        return "falling, below MAs — likely BEAR_TREND"
    if abs(spy_ret20) < 0.01 and rv < 0.13:
        return "flat, ultra-low vol — likely RANGE_LOW_VOL"
    return "ambiguous — needs human inspection"


def serialize_hmm(model: GaussianHMM, feature_means: np.ndarray,
                  feature_stds: np.ndarray,
                  state_to_label: dict[int, str] | None = None,
                  train_meta: dict | None = None) -> HMMModel:
    """Convert a fitted hmmlearn model into our HMMModel dataclass.

    `state_to_label` should be filled in after human calibration. If you
    pass None, this returns a model with placeholder labels (all
    VOLATILE_TRANSITION) — calling code can patch it before persisting.
    """
    if state_to_label is None:
        # Placeholder so HMMModel.to_json() doesn't break; promote_hmm.py
        # validates this dict isn't all-placeholder before INSERTing ACTIVE.
        state_to_label = {i: "VOLATILE_TRANSITION" for i in range(model.n_components)}

    # Normalize covariance to (n_states, n_features, n_features) full-matrix
    # form so the inference path (_gaussian_logpdf in classifier.py) doesn't
    # need to know whether the model was fit with diag or full covariance.
    # For 'diag', the n×d entries from covars_ become n diagonal matrices.
    covars_array = np.asarray(model.covars_)
    if covars_array.ndim == 2:
        # 'diag' shape: (n_states, n_features) → build (n_states, d, d)
        n = covars_array.shape[0]
        d = covars_array.shape[1]
        covars_full = np.zeros((n, d, d), dtype=float)
        for i in range(n):
            np.fill_diagonal(covars_full[i], covars_array[i])
    elif covars_array.ndim == 3:
        covars_full = covars_array
    else:
        raise ValueError(f"Unexpected covars_ shape: {covars_array.shape}")

    return HMMModel(
        n_states=int(model.n_components),
        feature_order=list(HMM_FEATURE_ORDER),
        means=model.means_.tolist(),
        covars=covars_full.tolist(),
        transmat=model.transmat_.tolist(),
        startprob=model.startprob_.tolist(),
        feature_means=feature_means.tolist(),
        feature_stds=feature_stds.tolist(),
        state_to_label=state_to_label,
        train_meta=train_meta or {},
    )


def main():
    p = argparse.ArgumentParser("train_hmm")
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cov-type", default="diag", choices=["diag", "full", "spherical", "tied"],
                   help="Emission covariance structure. 'diag' (default) is "
                        "standard for finance regime HMMs and avoids the "
                        "ill-conditioning that caused EM divergence in earlier runs.")
    p.add_argument("--min-covar", type=float, default=1e-3,
                   help="Lower bound on emission covariance diagonal (regularization).")
    p.add_argument("--out", default="/tmp/hmm_v1.json",
                   help="Output path for serialized HMM (NO DB write)")
    p.add_argument("--fingerprint-out", default="/tmp/hmm_fingerprint.json",
                   help="Output path for state fingerprint (for calibration review)")
    p.add_argument(
        "--train-end-date", default=None,
        help=(
            "Cap training data at this YYYY-MM-DD (inclusive). Use this to "
            "produce a backtest-only HMM that leaves the post-cutoff window "
            "for honest out-of-sample evaluation. Without this flag, the "
            "model is fit to every available snapshot — fine for production, "
            "useless for backtesting."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df, timestamps = load_features_matrix(train_end_date=args.train_end_date)
    log.info("Training window: %s → %s", df.index[0], df.index[-1])
    if args.train_end_date:
        log.info("Cutoff applied: train_end_date=%s (rows after this are reserved "
                 "for out-of-sample evaluation)", args.train_end_date)

    model, means, stds = fit_hmm(
        df, n_states=args.n_states, n_iter=args.n_iter,
        random_state=args.seed, cov_type=args.cov_type, min_covar=args.min_covar,
    )

    fp = fingerprint(model, df, means, stds)

    # Print human-readable summary
    print("\n" + "=" * 80)
    print(f"HMM training complete — {args.n_states} states, "
          f"{fp['n_obs']} obs from {df.index[0]} to {df.index[-1]}")
    print("=" * 80)
    for s in range(args.n_states):
        st = fp["states"][s]
        if st.get("empty"):
            print(f"\nState {s}: EMPTY (no observations assigned)")
            continue
        print(f"\nState {s}: occupancy={st['occupancy_frac']:.1%} "
              f"({st['n_obs']} days)")
        print(f"  → {st['summary']}")
        f = st["feature_means"]
        print(f"  spy_ret_20={f['spy_ret_20']:+.3f}  "
              f"spy_rv_20={f['spy_rv_20']:.3f}  "
              f"vix_level={f['vix_level']:.1f}")
        # Print whichever of the binary/continuous MA-distance keys are present.
        dma_50 = f.get('spy_dist_50dma', f.get('spy_above_50dma', 0.0))
        dma_200 = f.get('spy_dist_200dma', f.get('spy_above_200dma', 0.0))
        print(f"  spy_dist_50dma={dma_50:+.3f}  "
              f"spy_dist_200dma={dma_200:+.3f}  "
              f"breadth={f['breadth_above_20dma_pct']:.2f}")
        print(f"  avg_pairwise_corr_20={f['avg_pairwise_corr_20']:.2f}  "
              f"vix_chg_5={f['vix_chg_5']:+.1f}  "
              f"vix_pctl_252={f['vix_pctl_252']:.2f}")
        print(f"  hyg_mom_20={f['hyg_mom_20']:+.3f}  "
              f"tlt_mom_20={f['tlt_mom_20']:+.3f}  "
              f"gld_mom_20={f['gld_mom_20']:+.3f}")
        sp = st["spy_ret_20_pct"]
        print(f"  SPY 20d return distribution: "
              f"p10={sp['p10']:+.3f}  p50={sp['p50']:+.3f}  p90={sp['p90']:+.3f}")

    print("\n--- Transition matrix (rows = from, cols = to) ---")
    for i in range(args.n_states):
        print(f"  state {i}: " + "  ".join(f"{p:.2f}" for p in fp["transmat"][i]))
    print()

    # Serialize WITHOUT calibration (placeholder labels); promote_hmm.py
    # patches the labels after human review.
    hmm = serialize_hmm(
        model, means, stds,
        state_to_label=None,  # placeholder
        train_meta={
            "trained_at": datetime.utcnow().isoformat(),
            "train_start": str(df.index[0]),
            "train_end": str(df.index[-1]),
            "train_end_date_cutoff": args.train_end_date,  # None ↔ no cutoff
            "n_obs": int(fp["n_obs"]),
            "n_iter_done": int(getattr(model.monitor_, "iter", -1)),
            "final_log_likelihood": float(model.score((df.values - means) / stds)),
        },
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hmm.to_json(), indent=2))
    Path(args.fingerprint_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fingerprint_out).write_text(json.dumps(fp, indent=2, default=str))

    print(f"HMM model serialized to: {args.out}")
    print(f"Fingerprint (for calibration) saved to: {args.fingerprint_out}")
    print("\nNEXT STEP: review the fingerprint above, decide which state")
    print("maps to which regime (BULL_TREND / RANGE_LOW_VOL / BEAR_TREND /")
    print("VOLATILE_TRANSITION), then run:")
    print(f"  python -m scripts.promote_hmm --in {args.out} "
          "--label-0 <REGIME> --label-1 <REGIME> --label-2 <REGIME> --label-3 <REGIME>")


if __name__ == "__main__":
    main()
