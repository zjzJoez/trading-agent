#!/usr/bin/env python3
"""Pin every quantitative claim in the M1-0.4 REPORT.md to its artifact row.

WHY THIS EXISTS (review finding 4). The first pass over this report was
cross-checked by *global value membership*: each number printed in the prose
was searched for somewhere in the artifact JSON. That check passes on a claim
whose value is right but whose SUBJECT is wrong — and one slipped through
exactly that way, a profit-take example quoted with credit 1.32 when the
artifact row for (2025-10-13, QQQ, 590/585) says 1.41. 1.32 existed in the
artifact, just on a different trade.

So every claim here is checked as a TUPLE MATCH: the claim names the row it is
about — (entry_date, symbol, short_strike, long_strike) for a trade,
(entry_date, symbol) for an entry, a JSON path for a scalar — the row is looked
up by that key, and only then are the values compared. A claim whose key
matches nothing is a FAILURE, not a skip; that is the whole point.

Usage:
    python scripts/verify_replay_report.py --report-dir reports/vertical_replay_2026-07

Exit status 0 = every claim matched. Non-zero = at least one mismatch, each
printed with the claim, the artifact row and the difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 5e-4          # prose rounds to 4 decimals
TOL_MONEY = 5e-3    # dollar figures are quoted to the cent


# --------------------------------------------------------------- claim types


class Claims:
    """Collects claims and their verdicts."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, section: str, what: str, expected, actual,
              tol: float = TOL, key: str | None = None) -> None:
        if expected is None or actual is None:
            ok = expected is actual
        elif isinstance(expected, bool):
            # strict: a flag the artifact publishes as 1/0 instead of a JSON
            # boolean is a schema change worth failing on, and `True == 1` in
            # Python would hide it
            ok = actual is expected
        elif isinstance(expected, (int, float)):
            ok = (not isinstance(actual, bool)
                  and abs(float(actual) - float(expected)) <= tol)
        else:
            ok = actual == expected
        self.rows.append({"section": section, "what": what, "key": key,
                          "report_says": expected, "artifact_says": actual,
                          "ok": bool(ok)})

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]


def dig(obj, path: str):
    """Fetch a dotted JSON path, returning None when any hop is missing."""
    return digk(obj, *path.split("."))


def digk(obj, *keys):
    """Same as dig() but takes the key sequence explicitly.

    Needed wherever a key CONTAINS a dot — the credit-floor grid is keyed
    "0.25", "0.225", ... and splitting those on "." silently yields None,
    which would let a wrong number pass as an absent one.
    """
    cur = obj
    for part in keys:
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError, TypeError):
                return None
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_entries(path: Path) -> list[dict]:
    """The qualified records of an entries.jsonl."""
    return [rec for rec in
            (json.loads(line) for line in path.read_text().splitlines() if line)
            if rec["qualified"]]


def rs_by_exit(entries: list[dict], reason: str) -> list[float]:
    """result_r of every booked trade that exited for `reason`."""
    return [e["result_r"] for e in entries if e["exit_reason"] == reason]


def find_row(rows: list[dict], **key) -> dict | None:
    """The single row matching every field in `key`, or None.

    Returns None when the key matches zero OR several rows — an ambiguous key
    is as bad as a missing one, because the claim then does not identify what
    it is about.
    """
    hits = [r for r in rows
            if all(r.get(k) == v for k, v in key.items())]
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------------- claims


def verify(report_dir: Path) -> Claims:
    primary = json.loads((report_dir / "summary.json").read_text())
    upper = json.loads((report_dir / "as_reported" / "summary.json").read_text())
    diag = json.loads((report_dir / "diagnostics.json").read_text())
    up = diag["samples"]["as_reported_close_sameday"]
    pr = diag["samples"]["primary_smile_prior"]
    c = Claims()

    # -- §1 availability table -------------------------------------------
    for tag, s, exp in (
        ("primary", primary, {"n_entries": 144, "n_marks_present": 123,
                              "n_data_adequate": 119, "n_band_bracketed": 56,
                              "n_qualified": 26,
                              "n_qualified_band_bracketed": 8,
                              "availability_rate_planned": 0.1806,
                              "availability_rate": 0.2185,
                              "availability_rate_strict": 0.2114,
                              "availability_rate_band_bracketed": 0.1429}),
        ("upper", upper, {"n_entries": 144, "n_marks_present": 123,
                          "n_data_adequate": 119, "n_band_bracketed": 63,
                          "n_qualified": 45,
                          "n_qualified_band_bracketed": 23,
                          "availability_rate_planned": 0.3125,
                          "availability_rate": 0.3782,
                          "availability_rate_strict": 0.3659,
                          "availability_rate_band_bracketed": 0.3651}),
    ):
        for field, want in exp.items():
            c.check("1", f"{tag} availability.{field}", want,
                    dig(s, f"availability.overall.{field}"),
                    key=f"availability.overall.{field}")
        # the floor is failed on EVERY denominator, both marks
        for field in ("passes_floor_on_planned", "passes_floor_on_data_adequate",
                      "passes_floor_on_band_bracketed"):
            c.check("1", f"{tag} availability.{field}", False,
                    dig(s, f"availability.overall.{field}"))
        # and every published rate is a real fraction
        ov = dig(s, "availability.overall")
        c.check("1", f"{tag} band-bracketed numerator <= denominator", True,
                ov["n_qualified_band_bracketed"] <= ov["n_band_bracketed"])

    # -- §1 credit-floor retune ------------------------------------------
    for tag, s, floors in (
        ("primary", primary, {"0.25": 0.2185, "0.225": 0.5714, "0.2": 0.8235}),
        ("upper", upper, {"0.25": 0.3782, "0.225": 0.6807, "0.2": 0.8319}),
    ):
        for floor, want in floors.items():
            c.check("1", f"{tag} credit_floor_sensitivity[{floor}]", want,
                    digk(s, "availability", "credit_floor_sensitivity", floor,
                         "availability_rate"))
    # the retune at 0.225 on the PLANNED denominator, which is where it fails
    n_q_225 = digk(upper, "availability", "credit_floor_sensitivity", "0.225",
                   "n_qualified")
    c.check("1", "upper: floor 0.225 qualified count", 81, n_q_225)
    c.check("1", "upper: floor 0.225 on the 144 denominator", 0.5625,
            n_q_225 / 144)

    # -- §2a mark-convention sensitivity table ---------------------------
    rows = dig(primary, "sensitivity.rows")
    for tag, exp in (
        ("mark-smile", {"n_trades": 26, "win_rate": 0.6923, "mean_r": -0.0455,
                        "median_r": 0.0950,
                        "lb95_mean_r_entry_week": -0.1581,
                        "lb95_mean_r_exposure_cluster": -0.0911,
                        "max_drawdown_r": 2.4309,
                        "availability_rate_data_adequate": 0.2185,
                        "availability_rate_planned": 0.1806,
                        "exit_profit_take": 12, "exit_dte_21": 14}),
        ("mark-close", {"n_trades": 44, "win_rate": 0.7727, "mean_r": 0.0520,
                        "median_r": 0.1396,
                        "lb95_mean_r_entry_week": -0.0254,
                        "lb95_mean_r_exposure_cluster": -0.0158,
                        "max_drawdown_r": 2.1690,
                        "availability_rate_data_adequate": 0.3697,
                        "availability_rate_planned": 0.3056,
                        "exit_profit_take": 32, "exit_dte_21": 12}),
        ("mark-vw", {"n_trades": 54, "win_rate": 0.9444, "mean_r": 0.1554,
                     "median_r": 0.1657,
                     "lb95_mean_r_entry_week": 0.1185,
                     "lb95_mean_r_exposure_cluster": 0.1310,
                     "max_drawdown_r": 0.8835,
                     "availability_rate_data_adequate": 0.4538,
                     "availability_rate_planned": 0.3750,
                     "exit_profit_take": 51, "exit_dte_21": 3}),
        ("mark-hl2", {"n_trades": 36, "win_rate": 0.6944, "mean_r": 0.0118,
                      "median_r": 0.1190,
                      "lb95_mean_r_entry_week": -0.0702,
                      "lb95_mean_r_exposure_cluster": -0.0277,
                      "max_drawdown_r": 1.6786,
                      "availability_rate_data_adequate": 0.3025,
                      "availability_rate_planned": 0.2500,
                      "exit_profit_take": 19, "exit_dte_21": 17}),
        ("liquidity-same-day", {"n_trades": 27, "mean_r": -0.0335,
                                "availability_rate_data_adequate": 0.2269,
                                "availability_rate_planned": 0.1875}),
        ("width-any", {"n_trades": 26, "mean_r": -0.0455,
                       "availability_rate_data_adequate": 0.2185}),
    ):
        row = rows.get(tag)
        c.check("2a", f"sensitivity row {tag} exists", True, row is not None,
                key=tag)
        if row is None:
            continue
        for field, want in exp.items():
            c.check("2a", f"sensitivity[{tag}].{field}", want, row.get(field),
                    key=tag)
    # the claimed SPREAD of the mark column
    means = [rows[f"mark-{m}"]["mean_r"] for m in ("smile", "close", "vw", "hl2")]
    c.check("2a", "mean_r spread across mark conventions", 0.2009,
            max(means) - min(means))

    # -- §2 smile provenance and the no-mixed-pair claim (finding 12) ----
    sm = dig(primary, "mark_diagnostics.smile")
    for field, want in (("days_fitted", 1897), ("days_unfittable", 364),
                        ("bars_repriced", 16075), ("bars_kept_raw", 1904),
                        ("bars_kept_raw_on_unfittable_days", 1904),
                        ("bars_kept_raw_on_fitted_days", 0)):
        c.check("2", f"mark_diagnostics.smile.{field}", want, sm.get(field))
    rme = dig(primary, "mark_diagnostics.raw_mark_exposure")
    for field, want in (("n_trades", 26),
                        ("n_trades_with_any_raw_marked_leg_day", 0),
                        ("n_trades_with_a_mixed_pair_day", 0),
                        ("n_marked_pair_days", 217),
                        ("n_marked_leg_days_raw", 0),
                        ("n_marked_pair_days_mixed", 0)):
        c.check("2", f"raw_mark_exposure.{field}", want, rme.get(field))
    c.check("2", "raw_mark_exposure.no_mixed_pair_anywhere", True,
            rme.get("no_mixed_pair_anywhere"))

    # -- §2b null study ---------------------------------------------------
    for field, want in (("n", 11842), ("mean", 0.0014), ("median", 0.0005),
                        ("sd", 0.2594), ("p10", -0.1361), ("p90", 0.1335)):
        c.check("2b", f"null_study.ungated.{field}", want,
                dig(up, f"null_study.ungated.{field}"))
    for field, want in (("n", 1864), ("mean", 0.0047), ("median", 0.0028),
                        ("sd", 0.2434)):
        c.check("2b", f"null_study.in_band.{field}", want,
                dig(up, f"null_study.ungated_in_dte_and_delta_band.{field}"))
    c.check("2b", "credit noise as pp of a $5 width", 5.1884,
            dig(up, "null_study.credit_noise_sd_as_pct_of_5_wide_width"))
    # finding 3: the artifact must NOT claim either estimator is unbiased
    c.check("2b", "no over-claiming 'estimator_is_unbiased' key", None,
            dig(up, "null_study.estimator_is_unbiased"))
    c.check("2b", "no_systematic_offset_between_conventions", True,
            dig(up, "null_study.no_systematic_offset_between_conventions"))

    # -- §2c the two per-trade absurdities: TUPLE-MATCHED (finding 4) -----
    ea = dig(up, "exit_trigger_audit")
    c.check("2c", "n_profit_takes_testable", 34, ea["n_profit_takes_testable"])
    c.check("2c", "n_profit_takes_curve_denies", 9,
            ea["n_profit_takes_the_curve_says_never_triggered"])
    c.check("2c", "spurious_profit_take_fraction", 0.2647,
            ea["spurious_profit_take_fraction"])
    for label, key, exp in (
        # example 1 — the flat-day profit-take. Its credit is 1.41, NOT the
        # 1.32 the first draft printed (that value belongs to example 2).
        ("example 1 (flat day)",
         {"entry_date": "2025-10-13", "symbol": "QQQ",
          "short_strike": 590.0, "long_strike": 585.0},
         {"credit": 1.41, "exit_date": "2025-10-15", "raw_exit_mark": 0.620,
          "pt_level": 0.705, "curve_exit_mark": 1.3294, "spot_entry": 602.01,
          "spot_exit": 602.22, "underlying_move_pct": 0.0349,
          "result_r": 0.1371, "exit_reason": "profit_take",
          "curve_above_pt_level": True}),
        # example 2 — the $5-wide spread marked at one cent on a -2.42% day
        ("example 2 (falling underlying)",
         {"entry_date": "2026-03-23", "symbol": "QQQ",
          "short_strike": 570.0, "long_strike": 565.0},
         {"credit": 1.32, "exit_date": "2026-03-26", "raw_exit_mark": 0.010,
          "pt_level": 0.660, "curve_exit_mark": 1.5716, "spot_entry": 588.00,
          "spot_exit": 573.79, "underlying_move_pct": -2.4167,
          "result_r": 0.1093, "exit_reason": "profit_take",
          "curve_above_pt_level": True}),
    ):
        row = find_row(ea["all_rows"], **key)
        c.check("2c", f"{label}: artifact row exists for {key}", True,
                row is not None, key=json.dumps(key, sort_keys=True))
        if row is None:
            continue
        for field, want in exp.items():
            tol = TOL_MONEY if field.startswith("spot") else TOL
            c.check("2c", f"{label}.{field}", want, row.get(field), tol=tol,
                    key=json.dumps(key, sort_keys=True))
    c.check("2c", "primary sample: curve-denied profit-takes", 0,
            dig(pr, "exit_trigger_audit."
                    "n_profit_takes_the_curve_says_never_triggered"))
    c.check("2c", "primary sample: profit-takes testable", 12,
            dig(pr, "exit_trigger_audit.n_profit_takes_testable"))

    # -- §3 selection bias ------------------------------------------------
    sb = dig(up, "selection_bias")
    for field, want in (("n", 45), ("mean", -0.2878), ("median", -0.2792),
                        ("sd", 0.2427)):
        c.check("3", f"selection_bias.selected.{field}", want,
                dig(sb, f"selected.{field}"))
    for field, want in (("shift_vs_in_band_null_dollars", -0.2925),
                        ("shift_as_pct_of_5_wide_width", -5.8492),
                        ("shift_in_in_band_null_sd", -1.2017),
                        ("shift_in_ungated_null_sd", -1.1274),
                        ("implied_measurement_bias_in_credit_r", 0.0822)):
        c.check("3", f"selection_bias.{field}", want, sb.get(field))
    c.check("3", "primary selection shift in in-band-null sd", -0.0192,
            dig(pr, "selection_bias.shift_in_in_band_null_sd"))
    c.check("3", "primary implied measurement bias in credit R", 0.0013,
            dig(pr, "selection_bias.implied_measurement_bias_in_credit_r"))

    # -- §4 survivorship, with ITS OWN denominator (finding 8) ------------
    sv = dig(up, "survivorship")
    for field, want in (("n_records", 144), ("n_records_with_rv", 130),
                        ("n_records_skipped_for_rv", 14)):
        c.check("4", f"survivorship.{field}", want, sv.get(field))
    for group, n, mean, med, fwd in (
        ("dropped_not_data_adequate", 25, 0.1263, 0.1322, 0.1494),
        ("kept_data_adequate", 105, 0.1594, 0.1538, 0.1595),
        ("kept_and_qualified", 40, 0.1718, 0.1731, 0.1776),
        ("kept_and_gate_rejected", 65, 0.1517, 0.1430, 0.1484),
    ):
        base = f"survivorship.groups.{group}"
        c.check("4", f"{group}.n", n, dig(up, f"{base}.trailing_rv20.n"))
        c.check("4", f"{group}.trailing mean", mean,
                dig(up, f"{base}.trailing_rv20.mean"))
        c.check("4", f"{group}.trailing median", med,
                dig(up, f"{base}.trailing_rv20.median"))
        c.check("4", f"{group}.forward mean", fwd,
                dig(up, f"{base}.forward_rv10.mean"))
    c.check("4", "survivorship Welch t (dropped - kept)", -4.4455,
            sv.get("welch_t_trailing_dropped_minus_kept"))
    inv = dig(primary, "data_inventory")
    for field, want in (("batches_planned", 74), ("batches_missing_file", 7),
                        ("batches_no_bars", 4), ("contract_rows", 543),
                        ("contract_rows_with_bars", 511),
                        ("skipped_bad_json_lines", 0), ("skipped_bad_bars", 0)):
        c.check("4", f"data_inventory.{field}", want, inv.get(field))

    # -- §5a look-ahead / information set (finding 2) ----------------------
    c.check("5a", "declared information set", "enter-at-the-close",
            dig(primary, "config.information_set"))
    c.check("5a", "liquidity screen uses future information", False,
            dig(primary, "config.liquidity_screen_uses_future_information"))
    c.check("5a", "primary screen is the stricter one", True,
            dig(primary, "config."
                         "liquidity_screen_stricter_than_declared_information_set"))
    c.check("5a", "upper-bound screen is the same-day one", False,
            dig(upper, "config."
                       "liquidity_screen_stricter_than_declared_information_set"))
    for tag, s in (("primary", pr), ("upper", up)):
        c.check("5a", f"{tag} unwalkable days inside a held window", 0,
                dig(s, "walk_coverage.unwalkable_days_total"))
        c.check("5a", f"{tag} reported skipped_days", 0,
                dig(s, "walk_coverage.reported_skipped_days_total"))
    for tag, s in (("primary", primary), ("upper", upper)):
        c.check("5a", f"{tag} data_end exits", 0,
                dig(s, "exit_degradation.counts.data_end"))
    # cost of the stricter screen, as the report states it
    c.check("5a", "trades under close: same-day vs prior", 45,
            dig(upper, "managed_payoff.n_trades"))
    c.check("5a", "trades under close, prior screen", 44,
            dig(primary, "sensitivity.rows.mark-close.n_trades"))
    c.check("5a", "trades under smile: same-day", 27,
            dig(primary, "sensitivity.rows.liquidity-same-day.n_trades"))
    c.check("5a", "trades under smile: prior", 26,
            dig(primary, "managed_payoff.n_trades"))
    c.check("5a", "availability cost of the stricter screen (smile)", 0.0084,
            dig(primary, "sensitivity.rows.liquidity-same-day"
                         ".availability_rate_data_adequate")
            - dig(primary, "availability.overall.availability_rate"))
    c.check("5a", "availability before the credit-floor tolerance fix", 0.3697,
            dig(primary, "sensitivity.rows.mark-close"
                         ".availability_rate_data_adequate"))

    # -- §5b payoff shape (finding 13: PT>0 is a tautology) ---------------
    for tag, s, exp in (
        ("upper", upper, {"profit_take": 34, "dte_21": 11, "data_end": 0}),
        ("primary", primary, {"profit_take": 12, "dte_21": 14, "data_end": 0}),
    ):
        for k, want in exp.items():
            c.check("5b", f"{tag} exit count {k}", want,
                    dig(s, f"exit_degradation.counts.{k}"))
    c.check("5b", "upper mean R by exit: profit_take", 0.1639,
            dig(upper, "exit_degradation.mean_r_by_exit.profit_take"))
    c.check("5b", "upper mean R by exit: dte_21", -0.2789,
            dig(upper, "exit_degradation.mean_r_by_exit.dte_21"))
    c.check("5b", "primary mean R by exit: profit_take", 0.1360,
            dig(primary, "exit_degradation.mean_r_by_exit.profit_take"))
    c.check("5b", "primary mean R by exit: dte_21", -0.2010,
            dig(primary, "exit_degradation.mean_r_by_exit.dte_21"))
    # The forced-exit leg is NOT all losers under either mark — the first
    # draft claimed it was for the upper bound. The profit-take leg being all
    # winners, by contrast, is a tautology of the exit rule (managed_walk books
    # every profit-take at exactly 0.5*credit), so it is asserted as such.
    upper_entries = load_entries(report_dir / "as_reported" / "entries.jsonl")
    primary_entries = load_entries(report_dir / "entries.jsonl")
    for tag, entries, n_pos, best_pos in (("upper", upper_entries, 1, 0.0351),
                                          ("primary", primary_entries, 6, 0.1046)):
        d21 = rs_by_exit(entries, "dte_21")
        pt = rs_by_exit(entries, "profit_take")
        c.check("5b", f"{tag}: forced 21-DTE exits that were POSITIVE", n_pos,
                sum(1 for x in d21 if x > 0))
        c.check("5b", f"{tag}: best positive forced exit", best_pos, max(d21))
        c.check("5b", f"{tag}: every profit-take is positive (tautology)", True,
                all(x > 0 for x in pt))
    # the magnitudes and the mix, which are NOT tautological
    c.check("5b", "upper: loss-to-win magnitude ratio", 1.70,
            abs(dig(upper, "exit_degradation.mean_r_by_exit.dte_21"))
            / dig(upper, "exit_degradation.mean_r_by_exit.profit_take"),
            tol=5e-3)
    c.check("5b", "upper: profit-take to forced-exit count ratio", 3.09,
            dig(upper, "exit_degradation.counts.profit_take")
            / dig(upper, "exit_degradation.counts.dte_21"), tol=5e-3)
    upper_pt = rs_by_exit(upper_entries, "profit_take")
    c.check("5b", "upper: worst profit-take R", 0.1093, min(upper_pt))
    c.check("5b", "upper: best profit-take R", 0.2945, max(upper_pt))

    # -- §5c the unsampled tail -------------------------------------------
    for tag, s, dg_s, worst in (("upper", upper, up, -0.5694),
                                ("primary", primary, pr, -0.6157)):
        h = dig(s, "managed_payoff.histogram_r")
        c.check("5c", f"{tag} histogram underflow", 0, h["underflow"])
        c.check("5c", f"{tag} histogram overflow", 0, h["overflow"])
        c.check("5c", f"{tag} worst booked R", worst,
                dig(dg_s, "payoff_attribution.min_r"))
        # a full -1R is not in any occupied bin
        occupied = [h["edges"][i] for i, n in enumerate(h["counts"]) if n]
        c.check("5c", f"{tag} lowest occupied histogram bin", -0.7 if
                tag == "primary" else -0.6, min(occupied))

    # -- §5d drawdown attribution (finding 7: booking order) --------------
    c.check("5d", "upper n_trades", 45, dig(upper, "managed_payoff.n_trades"))
    c.check("5d", "upper mean R", 0.0557, dig(upper, "managed_payoff.mean_r"))
    c.check("5d", "upper median R", 0.1412, dig(upper, "managed_payoff.median_r"))
    c.check("5d", "upper sum R", 2.5048, dig(upper, "managed_payoff.sum_r"))
    c.check("5d", "upper max drawdown R (booking order)", 2.4748,
            dig(upper, "managed_payoff.max_drawdown_r"))
    c.check("5d", "primary max drawdown R (booking order)", 2.4309,
            dig(primary, "managed_payoff.max_drawdown_r"))
    c.check("5d", "upper drawdown as a fraction of total gain", 0.988,
            dig(upper, "managed_payoff.max_drawdown_r")
            / dig(upper, "managed_payoff.sum_r"), tol=5e-3)
    pa = dig(up, "payoff_attribution")
    c.check("5d", "excluding the 5 worst: n remaining", 40,
            dig(pa, "mean_r_excluding_k_worst.5.n_remaining"))
    c.check("5d", "excluding the 5 worst: mean R", 0.1221,
            dig(pa, "mean_r_excluding_k_worst.5.mean_r"))
    c.check("5d", "2026-03: n in month", 11,
            dig(pa, "mean_r_excluding_each_month.2026-03.n_in_month"))
    c.check("5d", "2026-03: mean R in month", -0.1203,
            dig(pa, "mean_r_excluding_each_month.2026-03.mean_r_in_month"))
    c.check("5d", "excluding 2026-03: n remaining", 34,
            dig(pa, "mean_r_excluding_each_month.2026-03.n_remaining"))
    c.check("5d", "excluding 2026-03: mean R", 0.1126,
            dig(pa, "mean_r_excluding_each_month.2026-03.mean_r_excluding_month"))
    c.check("5d", "2026-03 is the highest unconditional-IV month", True,
            dig(pa, "mean_r_excluding_each_month.2026-03.is_highest_iv_month"))
    c.check("5d", "2026-03 unconditional median IV", 0.2755,
            dig(pa, "mean_r_excluding_each_month.2026-03.unconditional_median_iv"))

    # -- §5e unconditional IV census --------------------------------------
    for month, n_str, n_days, med, mean, cond, cond_n in (
        ("2025-10", 62, 16, 0.2073, 0.2074, 0.2142, 6),
        ("2025-11", 61, 15, 0.2321, 0.2331, 0.2218, 8),
        ("2025-12", 37, 13, 0.1736, 0.1804, 0.2079, 2),
        ("2026-01", 42, 12, 0.1954, 0.1892, 0.1902, 2),
        ("2026-02", 56, 16, 0.2271, 0.2340, 0.2514, 6),
        ("2026-03", 71, 16, 0.2755, 0.2750, 0.2755, 11),
        ("2026-04", 71, 16, 0.2293, 0.2242, 0.2255, 6),
        ("2026-05", 23, 6, 0.1904, 0.2040, 0.2252, 1),
        ("2026-06", 20, 5, 0.2650, 0.2424, 0.2646, 3),
    ):
        base = f"iv_by_month.unconditional.{month}"
        c.check("5e", f"{month} n_strikes", n_str, dig(up, f"{base}.n_strikes"))
        c.check("5e", f"{month} n_entry_days", n_days,
                dig(up, f"{base}.n_entry_days"))
        c.check("5e", f"{month} unconditional median IV", med,
                dig(up, f"{base}.median_iv"))
        c.check("5e", f"{month} unconditional mean IV", mean,
                dig(up, f"{base}.mean_iv"))
        c.check("5e", f"{month} conditional median IV", cond,
                dig(up, f"iv_by_month.conditional_qualified_only.{month}"
                        ".median_iv"))
        c.check("5e", f"{month} conditional n_trades", cond_n,
                dig(up, f"iv_by_month.conditional_qualified_only.{month}"
                        ".n_trades"))

    # -- §5f blocking ------------------------------------------------------
    for tag, s, n_tr, wk, cl, maxcl, cross, pairs, hold_med, hold_max in (
        ("upper", up, 45, 26, 13, 11, 51, 990, 6.0, 18.0),
        ("primary", pr, 26, 15, 7, 12, 40, 325, 11.0, 18.0),
    ):
        c.check("5f", f"{tag} blocking n_trades", n_tr, dig(s, "blocking.n_trades"))
        c.check("5f", f"{tag} entry-week blocks", wk,
                dig(s, "blocking.entry_week.n_blocks"))
        c.check("5f", f"{tag} exposure-cluster blocks", cl,
                dig(s, "blocking.exposure_cluster.n_blocks"))
        sizes = dig(s, "blocking.exposure_cluster.block_sizes")
        c.check("5f", f"{tag} largest exposure cluster", maxcl,
                max(sizes) if sizes else None)
        c.check("5f", f"{tag} pairs open together in different weeks", cross,
                dig(s, "blocking."
                       "pairs_open_simultaneously_in_different_entry_weeks"))
        c.check("5f", f"{tag} total trade pairs", pairs,
                dig(s, "blocking.total_pairs"))
        c.check("5f", f"{tag} median holding period (days)", hold_med,
                dig(s, "blocking.holding_period_days.median"))
        c.check("5f", f"{tag} max holding period (days)", hold_max,
                dig(s, "blocking.holding_period_days.max"))
    # every LB95 in the study is negative except the vw row
    for tag in ("mark-smile", "mark-close", "mark-hl2", "liquidity-same-day",
                "width-any"):
        for scheme in ("lb95_mean_r_entry_week", "lb95_mean_r_exposure_cluster"):
            val = dig(primary, f"sensitivity.rows.{tag}.{scheme}")
            c.check("5f", f"{tag}.{scheme} is negative", True,
                    val is not None and val < 0)

    # -- §5g friction -----------------------------------------------------
    fc = dig(upper, "friction_comparison")
    c.check("5g", "friction n_trades", 45, fc.get("n_trades"))
    for field, mean, money in (("replay_friction", 0.0484, 17.2527),
                               ("production_deployed_friction", 0.0176, 6.2565),
                               ("production_honest_leg_mark_friction",
                                0.0513, 18.2524)):
        c.check("5g", f"{field}_r mean", mean, dig(fc, f"{field}_r.mean"))
        c.check("5g", f"{field}_dollars mean", money,
                dig(fc, f"{field}_dollars.mean"), tol=TOL_MONEY)
    for field, want in (("mean", 2.7276), ("min", 1.6303), ("max", 5.5600)):
        c.check("5g", f"ratio_replay_over_deployed.{field}", want,
                dig(fc, f"ratio_replay_over_deployed.{field}"))
    c.check("5g", "wing ratio long/short mean", 0.8486,
            dig(fc, "wing_ratio_long_over_short.mean"))
    c.check("5g", "wing ratio long/short median", 0.8532,
            dig(fc, "wing_ratio_long_over_short.median"))
    c.check("5g", "leg_mid_sum / net_credit mean", 12.6295,
            dig(fc, "leg_mid_sum_over_net_credit.mean"))
    c.check("5g", "leg_mid_sum / net_credit median", 12.6279,
            dig(fc, "leg_mid_sum_over_net_credit.median"))
    prov = dig(primary, "config.spread_pct_provenance")
    c.check("5g", "calibration file sha256 (first 16)",
            "a467989051373cae", (prov.get("sha256") or "")[:16])
    c.check("5g", "calibration file n_samples", 160,
            dig(prov, "file_globals.n_samples"))
    c.check("5g", "calibration file global spread_pct_of_mid_median", 0.07338,
            dig(prov, "file_globals.spread_pct_of_mid_median"))
    c.check("5g", "file has no per-underlying SPY entry", "cli_override",
            dig(prov, "per_symbol.SPY.source"))
    c.check("5g", "file value SPY falls back to the global median", 0.07338,
            dig(prov, "per_symbol.SPY.overrides_file_value"))
    for tag, s, want in (("upper", upper, 0.3555), ("primary", primary, 0.3885)):
        c.check("5g", f"{tag} friction under the calibration file (mean R)",
                want, dig(s, "friction_under_calibration_file."
                             "replay_friction_r.mean"))

    # -- §5h width freedom ------------------------------------------------
    c.check("5h", "width-any qualified count equals five-first", 26,
            dig(primary, "sensitivity.rows.width-any.n_trades"))
    c.check("5h", "width-any mean R equals five-first", -0.0455,
            dig(primary, "sensitivity.rows.width-any.mean_r"))
    for width, n, med, clearing, frac, days in (
        ("5", 408, 0.1880, 48, 0.1176, 119),
        ("10", 359, 0.1790, 20, 0.0557, 119),
    ):
        base = f"width_census.per_width.{width}"
        c.check("5h", f"width {width}: n constructible", n, dig(up, f"{base}.n"))
        c.check("5h", f"width {width}: median credit/width", med,
                dig(up, f"{base}.median"))
        c.check("5h", f"width {width}: n clearing the spec floor", clearing,
                dig(up, f"{base}.n_clearing_spec_floor"))
        c.check("5h", f"width {width}: frac clearing the spec floor", frac,
                dig(up, f"{base}.frac_clearing_spec_floor"))
        c.check("5h", f"width {width}: entry days with a constructible pair",
                days, dig(up, f"{base}.n_entry_days_with_a_constructible_pair"))

    # -- §5i band-bracket audit, TUPLE-MATCHED (finding 5) ----------------
    bb = dig(up, "band_bracket_audit")
    c.check("5i", "audit rows", 7, bb["n_rows"])
    c.check("5i", "rows classified as data gaps", 7,
            bb["n_data_gaps_band_not_fetched"])
    c.check("5i", "true gate rejections", 0, bb["n_true_gate_rejections"])
    for label, key, exp in (
        ("example 1", {"entry_date": "2025-12-26", "symbol": "QQQ"},
         {"spot": 623.89, "n_contracts_fetched": 1, "n_liquid": 1,
          "n_liquid_invertible": 1,
          "strike_min_fetched": 590.0, "strike_max_fetched": 590.0,
          "strike_min_invertible": 590.0, "strike_max_invertible": 590.0,
          "delta_min": 0.1654, "delta_max": 0.1654,
          "brackets_delta_band": False}),
        # the first draft said "6 invertible strikes, K 520-545, |delta|
        # 0.025-0.046" for this row. The artifact says 6 rows FETCHED spanning
        # K 520-545, and exactly ONE clearing liquidity + invertibility, at
        # K=530 and |delta| 0.031.
        ("example 2", {"entry_date": "2026-01-26", "symbol": "QQQ"},
         {"spot": 625.46, "n_contracts_fetched": 6, "n_liquid": 1,
          "n_liquid_invertible": 1,
          "strike_min_fetched": 520.0, "strike_max_fetched": 545.0,
          "strike_min_invertible": 530.0, "strike_max_invertible": 530.0,
          "delta_min": 0.0315, "delta_max": 0.0315,
          "brackets_delta_band": False}),
    ):
        row = find_row(bb["rows"], **key)
        c.check("5i", f"{label}: artifact row exists for {key}", True,
                row is not None, key=json.dumps(key, sort_keys=True))
        if row is None:
            continue
        for field, want in exp.items():
            tol = TOL_MONEY if field == "spot" else TOL
            c.check("5i", f"{label}.{field}", want, row.get(field), tol=tol,
                    key=json.dumps(key, sort_keys=True))
    c.check("5i", "primary sample audit rows", 6, dig(pr, "band_bracket_audit"
                                                          ".n_rows"))

    # -- §6 gate design ---------------------------------------------------
    c.check("6", "median credit/width on real $5-wides", 0.1880,
            dig(up, "width_census.per_width.5.median"))
    c.check("6", "fraction of $5-wides clearing credit >= width/4", 0.1176,
            dig(up, "width_census.per_width.5.frac_clearing_spec_floor"))
    c.check("6", "criterion was NOT evaluated as specified", False,
            dig(primary, "verdict.evaluated_as_specified"))
    c.check("6", "no regime labels exist in this replay", False,
            dig(primary, "verdict.regime_labels_present"))
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true",
                    help="print every claim, not just the failures")
    args = ap.parse_args(argv)

    c = verify(args.report_dir)
    by_section: dict[str, int] = {}
    for row in c.rows:
        by_section[row["section"]] = by_section.get(row["section"], 0) + 1

    if args.verbose:
        for row in c.rows:
            flag = "ok  " if row["ok"] else "FAIL"
            print(f"[{flag}] §{row['section']:3} {row['what']}: "
                  f"report={row['report_says']!r} artifact={row['artifact_says']!r}")

    print(f"verified {len(c.rows)} REPORT.md claims against the artifacts "
          f"({', '.join(f'§{k}:{v}' for k, v in sorted(by_section.items()))})")
    if not c.failures:
        print("ALL CLAIMS MATCH THEIR ARTIFACT ROW")
        return 0
    print(f"\n{len(c.failures)} MISMATCH(ES):")
    for row in c.failures:
        print(f"  §{row['section']} {row['what']}")
        print(f"      report says   : {row['report_says']!r}")
        print(f"      artifact says : {row['artifact_says']!r}")
        if row["key"]:
            print(f"      matched on    : {row['key']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
