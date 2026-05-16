"""Promote a calibrated HMM to ACTIVE in regime_model_versions.

Two-step process so calibration error can't silently ship:

    1. train_hmm.py     produces /tmp/hmm_v1.json with placeholder labels
                        + /tmp/hmm_fingerprint.json with state diagnostics
    2. HUMAN REVIEW of the fingerprint to decide which state is which regime
    3. promote_hmm.py   takes the labels as CLI args, patches the model JSON,
                        validates the mapping (no all-VOLATILE collapse,
                        no missing regimes), and INSERTs ACTIVE.

Why a separate script: the model JSON has 4 hidden states named 0-3.
Until a human decides "state 0 = BULL_TREND, state 1 = RANGE_LOW_VOL, ..."
the model is meaningless to the trader/risk pipeline. Doing this through
explicit CLI flags makes the calibration choice a reviewable diff.

Validation rules (refuse to promote if violated):
    - Every regime label MUST appear at most once (you can leave a state
      mapped to VOLATILE_TRANSITION as a "catch-all" for ambiguous states,
      but no two states should both be BULL_TREND — that means the HMM
      didn't separate them and the trader-level signal will collapse).
    - At least 2 distinct labels in the mapping (else the HMM adds no value
      over rule_based_fallback).
    - CRISIS is NEVER a valid HMM label (Layer 0 owns it; see classifier.py).

Usage:
    python -m scripts.promote_hmm \\
        --in /tmp/hmm_v1.json \\
        --label-0 BULL_TREND \\
        --label-1 RANGE_LOW_VOL \\
        --label-2 VOLATILE_TRANSITION \\
        --label-3 BEAR_TREND \\
        --rationale "after-review-2026-05-15: state 2 captures 2020-Q1 + 2022 selloffs"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from trading_agent.regime.classifier import HMMModel
from trading_agent.regime.persist import insert_model_version

log = logging.getLogger(__name__)

VALID_HMM_LABELS = {
    "BULL_TREND",
    "RANGE_LOW_VOL",
    "VOLATILE_TRANSITION",
    "BEAR_TREND",
    # CRISIS deliberately omitted — Layer 0 (hard rule) owns it
}


def main():
    p = argparse.ArgumentParser("promote_hmm")
    p.add_argument("--in", dest="in_path", required=True,
                   help="Serialized HMM JSON from train_hmm.py")
    p.add_argument("--label-0", required=True, choices=sorted(VALID_HMM_LABELS),
                   help="Regime label for hidden state 0")
    p.add_argument("--label-1", required=True, choices=sorted(VALID_HMM_LABELS))
    p.add_argument("--label-2", required=True, choices=sorted(VALID_HMM_LABELS))
    p.add_argument("--label-3", required=True, choices=sorted(VALID_HMM_LABELS))
    p.add_argument("--rationale", required=True,
                   help="Why this calibration — recorded in metrics_json for audit")
    p.add_argument("--retire-prior", action="store_true",
                   help="Mark any existing ACTIVE model as RETIRED first (recommended)")
    p.add_argument(
        "--status", default="active", choices=["active", "shadow", "retired"],
        help=(
            "Status to INSERT with. 'active' is the production path (one and "
            "only one row should be active at a time; combine with "
            "--retire-prior). 'shadow' is for evaluation-only models "
            "(e.g. an OOS backtest HMM) — it is NEVER picked up by "
            "get_active_model() so it cannot affect production trading."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Validate + print mapping, don't INSERT")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load
    blob = json.loads(Path(args.in_path).read_text())
    hmm = HMMModel.from_json(blob)
    log.info("Loaded HMM: n_states=%d, train_meta=%s", hmm.n_states, hmm.train_meta)

    if hmm.n_states != 4:
        log.error("Expected 4-state HMM; got %d", hmm.n_states)
        sys.exit(2)

    # Build the state→label mapping from CLI args
    mapping = {
        0: args.label_0,
        1: args.label_1,
        2: args.label_2,
        3: args.label_3,
    }

    # Validate per the rules in module docstring
    labels = list(mapping.values())
    distinct_non_volatile = {l for l in labels if l != "VOLATILE_TRANSITION"}
    if len(distinct_non_volatile) < 1:
        log.error("Refusing to promote — all states map to VOLATILE_TRANSITION; "
                  "HMM provides no signal over rule_based_fallback")
        sys.exit(3)
    # No regime appears > 1 time UNLESS it's VOLATILE_TRANSITION (catch-all OK)
    seen = {}
    for state, lbl in mapping.items():
        if lbl == "VOLATILE_TRANSITION":
            continue
        if lbl in seen:
            log.error("Refusing to promote — both states %d and %d map to %s. "
                      "Either the HMM didn't separate them (re-train with "
                      "different seed?) or one of them should be VOLATILE_TRANSITION",
                      seen[lbl], state, lbl)
            sys.exit(4)
        seen[lbl] = state

    # Patch the model
    hmm.state_to_label = mapping
    hmm.train_meta["calibrated_at"] = __import__("datetime").datetime.utcnow().isoformat()
    hmm.train_meta["calibration_rationale"] = args.rationale

    print("\nCalibrated state → regime mapping:")
    for s in sorted(mapping):
        print(f"  state {s}  →  {mapping[s]}")
    print(f"\nRationale: {args.rationale}")
    print(f"\nTrain window: {hmm.train_meta.get('train_start')} → "
          f"{hmm.train_meta.get('train_end')} ({hmm.train_meta.get('n_obs')} obs)")

    if args.dry_run:
        print("\n--dry-run: validation passed; NOT inserting to DB.")
        return

    # Retire prior active model so get_active_model() returns the new one.
    # Only meaningful when promoting status='active'; for 'shadow' (backtest
    # use only) we leave prior active alone.
    if args.retire_prior and args.status == "active":
        from trading_agent.store.postgres import cursor as pg_cursor
        with pg_cursor() as cur:
            cur.execute(
                "UPDATE regime_model_versions SET status = 'retired' "
                "WHERE status = 'active'"
            )
            log.info("Retired %d prior active model(s)", cur.rowcount)
    elif args.retire_prior and args.status != "active":
        log.info("Ignoring --retire-prior because --status=%s (shadow/retired "
                 "promotion must not retire the production HMM)", args.status)

    metrics = {
        "calibration_rationale": args.rationale,
        "state_to_label": mapping,
        "train_meta": hmm.train_meta,
    }
    new_id = insert_model_version(
        hmm,
        train_start=hmm.train_meta.get("train_start"),
        train_end=hmm.train_meta.get("train_end"),
        metrics=metrics,
        status=args.status,
    )
    print(f"\n✓ Inserted as regime_model_versions id={new_id} (status={args.status})")
    if args.status == "active":
        print("  Next premarket_scan tick will use this HMM.")
    else:
        print("  This is a NON-PRODUCTION model; only loadable explicitly by id.")


if __name__ == "__main__":
    main()
