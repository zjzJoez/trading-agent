"""Tests for scripts/replay_vertical_diagnostics.py.

Offline: a synthetic two-day chain priced off a known IV curve, plus a few
deliberately corrupted prints so each diagnostic has something to find.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

import scripts.replay_vertical_diagnostics as dg
import scripts.replay_vertical_gates as rv

S = 100.0
Q = 0.012
R = rv.RISK_FREE_RATE
EXPIRY = date(2026, 6, 19)
DAYS = [date(2026, 5, 13), date(2026, 5, 14), date(2026, 5, 15),
        date(2026, 5, 18), date(2026, 5, 19)]
ENTRY = date(2026, 5, 15)
STRIKES = [86, 88, 90, 91, 92, 94, 95, 96, 97, 98]


def _write_data(data_dir, corrupt: dict | None = None):
    """A chain flat at sigma=0.25 on every day, so the only thing that can
    move a spread mark is a corrupted print."""
    corrupt = corrupt or {}
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "underlying_SPY.json").write_text(json.dumps(
        {"results": [{"t": d.isoformat(), "c": S} for d in DAYS]}))
    rows = []
    for k in STRIKES:
        bars = []
        for d in DAYS:
            t = (EXPIRY - d).days / 365.0
            px = corrupt.get((k, d)) or rv.bs_put_price(S, k, t, R, Q, 0.25)
            bars.append({"t": d.isoformat(), "o": px, "h": px, "l": px,
                         "c": px, "vw": px, "v": 900, "n": 200})
        rows.append({"ticker": f"O:SPY260619P{int(k * 1000):08d}",
                     "strike": k, "bars": bars})
    with (data_dir / "contracts_SPY_2026-06-19.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    plan = {"batches": [{"symbol": "SPY", "expiry": EXPIRY.isoformat(),
                         "entry_dates": [ENTRY.isoformat()]}]}
    (data_dir / "batch_plan.json").write_text(json.dumps(plan))
    return data_dir / "batch_plan.json"


def _run(tmp_path, corrupt=None, **kwargs):
    data_dir = tmp_path / "data"
    plan = _write_data(data_dir, corrupt)
    out_dir = tmp_path / "out"
    # floor 0.20: real SPY/QQQ 5-wides median credit/width 0.188, so the spec's
    # 0.25 admits nothing on a textbook-priced chain either.
    kwargs.setdefault("credit_floor_frac", 0.20)
    rv.run(data_dir, plan, out_dir, {"SPY": 0.0049}, resamples=200, seed=1,
           **kwargs)
    return data_dir, plan, out_dir / "entries.jsonl"


def test_null_study_estimator_is_unbiased_on_a_clean_curve(tmp_path):
    # Every print already sits on one BS curve, so the same-day curve estimate
    # must reproduce the raw close difference to within numerical noise.
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    null = out["null_study"]["ungated"]
    assert null["n"] > 0
    assert abs(null["mean"]) < 1e-3
    assert out["null_study"]["estimator_is_unbiased"] is True
    assert out["null_study"]["credit_noise_sd_as_pct_of_5_wide_width"] == (
        pytest.approx(100.0 * null["sd"] / 5.0))


def test_selection_bias_is_zero_when_no_print_is_corrupted(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    sb = out["selection_bias"]
    assert sb["selected"]["n"] >= 1
    assert abs(sb["shift_vs_in_band_null_dollars"]) < 1e-3
    # both yardsticks are reported so neither can be cherry-picked
    assert sb["shift_in_in_band_null_sd"] is not None
    assert sb["shift_in_ungated_null_sd"] is not None


def test_selection_bias_detects_a_widened_entry_print(tmp_path):
    # Make the entry-day WING print artificially cheap: the raw credit widens
    # while the same-day curve (fitted from the other 9 strikes) does not move,
    # which is exactly the noise the credit floor selects on.
    true_wing = rv.bs_put_price(S, 92, (EXPIRY - ENTRY).days / 365.0, R, Q, 0.25)
    data_dir, plan, entries = _run(tmp_path, corrupt={(92, ENTRY): 0.01},
                                   mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    sb = out["selection_bias"]
    assert sb["shift_vs_in_band_null_dollars"] < -0.5 * true_wing
    assert sb["shift_in_in_band_null_sd"] < 0
    assert sb["implied_measurement_bias_in_credit_r"] > 0


def test_exit_trigger_audit_flags_a_profit_take_the_curve_denies(tmp_path):
    # A single stale exit-day print on the short leg drags the raw spread mark
    # under the profit-take level while the same-day curve stays above it.
    exit_day = DAYS[3]
    data_dir, plan, entries = _run(tmp_path, corrupt={(97, exit_day): 0.02},
                                   mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    audit = out["exit_trigger_audit"]
    assert audit["n_profit_takes_testable"] == 1
    assert audit["n_profit_takes_the_curve_says_never_triggered"] == 1
    assert audit["spurious_profit_take_fraction"] == pytest.approx(1.0)
    row = audit["most_extreme_spurious"][0]
    assert row["curve_exit_mark"] > row["pt_level"] > row["raw_exit_mark"]
    assert row["underlying_move_pct"] == pytest.approx(0.0, abs=1e-9)


def test_denominator_hygiene_reports_every_denominator_and_its_verdict(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    dh = out["denominator_hygiene"]
    assert dh["denominators"]["planned"] == 1
    assert dh["denominators"]["data_adequate"] == 1
    assert dh["n_qualified"] == 1
    assert dh["availability_by_denominator"]["planned"] == pytest.approx(1.0)
    assert dh["passes_60pct_by_denominator"]["planned"] is True


def test_band_bracket_audit_separates_data_gaps_from_gate_rejections(tmp_path):
    # A chain whose only strikes are far OTM never reaches the delta band.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "underlying_SPY.json").write_text(json.dumps(
        {"results": [{"t": d.isoformat(), "c": S} for d in DAYS]}))
    rows = []
    for k in (75, 80):
        bars = [{"t": d.isoformat(), "c": rv.bs_put_price(
            S, k, (EXPIRY - d).days / 365.0, R, Q, 0.25), "n": 200}
            for d in DAYS]
        rows.append({"ticker": f"O:SPY260619P{int(k * 1000):08d}",
                     "strike": k, "bars": bars})
    with (data_dir / "contracts_SPY_2026-06-19.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    plan = data_dir / "batch_plan.json"
    plan.write_text(json.dumps({"batches": [
        {"symbol": "SPY", "expiry": EXPIRY.isoformat(),
         "entry_dates": [ENTRY.isoformat()]}]}))
    out_dir = tmp_path / "out"
    rv.run(data_dir, plan, out_dir, {"SPY": 0.0049}, resamples=100, seed=1,
           mark="close", credit_floor_frac=0.20)
    out = dg.diagnose(data_dir, plan, out_dir / "entries.jsonl", ("SPY",),
                      {"SPY": 0.0049})
    audit = out["band_bracket_audit"]
    assert audit["n_rows"] == 1
    assert audit["n_data_gaps_band_not_fetched"] == 1
    assert audit["n_true_gate_rejections"] == 0
    assert audit["rows"][0]["brackets_delta_band"] is False


def test_survivorship_needs_enough_history_and_reports_a_welch_t(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    # only 5 underlying sessions exist, so no 20-session trailing window can be
    # formed: every group must be empty rather than silently short-windowed.
    groups = out["survivorship"]["groups"]
    assert all(g["trailing_rv20"]["n"] == 0 for g in groups.values())
    assert out["survivorship"]["welch_t_trailing_dropped_minus_kept"] is None


def test_blocking_and_walk_coverage(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    b = out["blocking"]
    assert b["n_trades"] == 1
    assert b["entry_week"]["n_blocks"] == 1
    assert b["exposure_cluster"]["n_blocks"] == 1
    assert b["pairs_open_simultaneously_in_different_entry_weeks"] == 0
    assert out["walk_coverage"]["unwalkable_days_total"] == 0


def test_payoff_attribution_ties_month_exclusion_to_that_months_iv(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out = dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
    pa = out["payoff_attribution"]
    assert pa["n_trades"] == 1
    assert pa["full_loss_r"] == -1.0
    assert "UNSAMPLED" in pa["tail_note"]
    month = pa["mean_r_excluding_each_month"]["2026-05"]
    assert month["n_in_month"] == 1
    assert month["mean_r_excluding_month"] is None      # nothing left
    assert month["unconditional_median_iv"] == pytest.approx(0.25, abs=0.02)
    assert month["is_highest_iv_month"] is True
    uncond = out["iv_by_month"]["unconditional"]["2026-05"]
    assert uncond["n_entry_days"] == 1
    assert uncond["n_strikes"] >= 1


def test_diagnose_restores_the_caller_s_mark_convention(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    prev = rv.set_mark_convention("vw")
    try:
        dg.diagnose(data_dir, plan, entries, ("SPY",), {"SPY": 0.0049})
        assert rv.mark_convention() == "vw"
    finally:
        rv.set_mark_convention(prev)


def test_cli_writes_one_section_per_labelled_sample(tmp_path, capsys):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    out_path = tmp_path / "diag.json"
    rc = dg.main(["--data-dir", str(data_dir), "--batch-plan", str(plan),
                  "--entries", str(entries), "--label", "sample_a",
                  "--out", str(out_path), "--symbols", "SPY",
                  "--spread-pct", "SPY=0.0049"])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert list(payload["samples"]) == ["sample_a"]
    assert payload["samples"]["sample_a"]["n_qualified"] == 1
    assert "wrote" in capsys.readouterr().out


def test_cli_rejects_mismatched_labels(tmp_path):
    data_dir, plan, entries = _run(tmp_path, mark="close")
    with pytest.raises(SystemExit):
        dg.main(["--data-dir", str(data_dir), "--batch-plan", str(plan),
                 "--entries", str(entries), "--entries", str(entries),
                 "--label", "only_one", "--out", str(tmp_path / "d.json"),
                 "--symbols", "SPY", "--spread-pct", "SPY=0.0049"])


def test_welch_t_and_describe_helpers():
    assert dg._describe([]) == {"n": 0}
    d = dg._describe([1.0, 2.0, 3.0])
    assert (d["n"], d["mean"], d["median"]) == (3, 2.0, 2.0)
    assert dg._welch_t([1.0], [2.0]) is None            # too few points
    assert dg._welch_t([1.0, 1.0], [1.0, 1.0]) is None  # zero variance
    t = dg._welch_t([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert t is not None and t < 0
    assert np.isfinite(t)


def test_module_constants_are_wired_to_the_engine():
    # the diagnostics must never carry its own copy of a gate parameter
    assert dg.E is rv
    assert dg.WORST_K == (1, 5, 10)
    assert isinstance(timedelta(days=1), timedelta)
