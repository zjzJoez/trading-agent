#!/usr/bin/env python3
"""M1-0.4 replay DIAGNOSTICS: the adversarial checks, as a committed artifact.

The replay engine (scripts/replay_vertical_gates.py) answers "what would the
gate stack have done". This script answers the harder question — "could this
data have answered that at all" — and writes every number the report cites
into one JSON file so no figure in the report is un-recomputable.

Sections, each keyed by the entries sample it was computed from:

  null_study            Over EVERY liquid 5-wide adjacent put pair on EVERY
                        day in EVERY fetched chain (no gate), compare the raw
                        close-difference credit to the same-day-curve-consistent
                        value. If the ungated gap is ~0, the curve estimator is
                        unbiased and its dispersion IS the credit noise of a
                        daily trade-print aggregate.
  selection_bias        The same statistic on the GATE-SELECTED entries,
                        expressed in ungated-null sd units. A large negative
                        shift means the credit floor is selecting days whose
                        raw print pair happened to be wide, i.e. mining noise.
  exit_trigger_audit    Per trade, the raw close-difference mark on the exit
                        day vs the same-day-curve spread. Counts profit-takes
                        the curve says never happened, and lists the most
                        extreme cases with the underlying move that day.
  survivorship          Trailing-20 and forward-10 realized vol of the entries
                        DROPPED from the availability denominator vs those
                        kept, with a Welch t. Vol-correlated dropping biases
                        availability.
  denominator_hygiene   Every unqualified reason x class x data_adequate, and
                        the availability rate under each candidate denominator.
  band_bracket_audit    For each band-related rejection, the fetched grid's
                        |delta| range and whether it brackets [0.20, 0.35].
  blocking              Bootstrap block counts under both schemes and the
                        number of trade pairs open simultaneously yet placed in
                        different entry-week blocks.
  walk_coverage         Trading days inside a held window with no walkable leg
                        pair (silently un-walked days).
  friction              Replay friction vs the deployed friction_r leg-mark
                        proxy vs the same stack on the actual leg marks.
  payoff_attribution    Drawdown/expectancy attribution: mean R excluding the
                        k worst trades and excluding each calendar month, with
                        that month's UNCONDITIONAL IV so "just delete the bad
                        month" cannot be mistaken for a neutral operation.
  iv_by_month           Unconditional short-leg IV (every liquid invertible
                        in-band strike, DTE 30-45), with n. The conditional
                        (qualified-only) figures alongside it.
  width_census          credit/width of every constructible vertical, per
                        width, with no floor applied — the gate-DESIGN
                        measurement: is the spec's credit >= width/4 above or
                        below what the market offers?

Offline: stdlib + numpy, reads only the pre-fetched aggregates.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

# Import the engine as the SAME module object the caller sees when this repo is
# on sys.path, so the mark-convention global is shared rather than duplicated.
try:
    from scripts import replay_vertical_gates as E  # noqa: E402
except ImportError:  # running the file directly: scripts/ is sys.path[0]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replay_vertical_gates as E  # noqa: E402

WORST_K = (1, 5, 10)


# ------------------------------------------------------------- helpers


def _load_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _describe(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def _welch_t(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return None
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return None if se == 0 else float((a.mean() - b.mean()) / se)


class Chains:
    """Lazily loaded chains + underlying closes, always on RAW closes."""

    def __init__(self, data_dir: Path, symbols: tuple[str, ...]):
        self.data_dir = data_dir
        self._chains: dict[tuple[str, date], dict | None] = {}
        self.closes = {s: E.load_underlying_closes(data_dir, s) for s in symbols}

    def chain(self, symbol: str, expiry: date) -> dict | None:
        key = (symbol, expiry)
        if key not in self._chains:
            self._chains[key], _ = E.load_put_contracts(self.data_dir, symbol,
                                                        expiry)
        return self._chains[key]

    def curve_spread(self, symbol: str, expiry: date, day: date,
                     short_k: float, long_k: float) -> float | None:
        """Same-day-curve-consistent value of the (short_k, long_k) spread."""
        puts = self.chain(symbol, expiry)
        spot = self.closes[symbol].get(day)
        if not puts or spot is None:
            return None
        t = (expiry - day).days / 365.0
        q = E.DIV_YIELD.get(symbol, 0.0)
        coef = E.fit_iv_curve(puts, day, spot, t, q)
        if coef is None:
            return None
        lo, hi = E.SMILE_IV_BOUNDS
        out = []
        for k in (short_k, long_k):
            m = k / spot
            iv = float(coef[0] + coef[1] * m + coef[2] * m * m)
            if not (lo < iv < hi):
                return None
            out.append(E.bs_put_price(spot, k, t, E.RISK_FREE_RATE, q, iv))
        return out[0] - out[1]

    def realized_vol(self, symbol: str, day: date, n: int = 20,
                     forward: bool = False) -> float | None:
        """Annualized close-to-close realized vol over n sessions."""
        closes = self.closes[symbol]
        days = sorted(d for d in closes if (d >= day if forward else d <= day))
        window = days[:n + 1] if forward else days[-(n + 1):]
        if len(window) < n + 1:
            return None
        rets = [math.log(closes[window[i]] / closes[window[i - 1]])
                for i in range(1, len(window))]
        return statistics.stdev(rets) * math.sqrt(252)


# ------------------------------------------------------------- sections


def null_study(ch: Chains, min_trades: int = E.MIN_SHORT_TRADES) -> dict:
    """Ungated distribution of (curve spread - raw close-difference)."""
    gaps, gaps_in_band = [], []
    for path in sorted(ch.data_dir.glob("contracts_*.jsonl")):
        _, symbol, exp = path.stem.split("_")
        expiry = date.fromisoformat(exp)
        puts = ch.chain(symbol, expiry)
        if not puts:
            continue
        q = E.DIV_YIELD.get(symbol, 0.0)
        days: set[date] = set()
        for contract in puts.values():
            days |= set(contract["bars"])
        for day in sorted(days):
            spot = ch.closes[symbol].get(day)
            t = (expiry - day).days / 365.0
            coef = E.fit_iv_curve(puts, day, spot, t, q)
            if coef is None:
                continue
            dte = (expiry - day).days
            for short_k in sorted(puts):
                long_k = short_k - 5.0
                if long_k not in puts:
                    continue
                bs = puts[short_k]["bars"].get(day)
                bl = puts[long_k]["bars"].get(day)
                if bs is None or bl is None:
                    continue
                ps, pl = E._close_of(bs), E._close_of(bl)
                if ps is None or pl is None:
                    continue
                if E._trade_count(bs) < min_trades:
                    continue
                curve = ch.curve_spread(symbol, expiry, day, short_k, long_k)
                if curve is None:
                    continue
                gap = curve - (ps - pl)
                gaps.append(gap)
                iv = E.implied_vol_put(ps, spot, short_k, t, E.RISK_FREE_RATE, q)
                if iv is None:
                    continue
                d = abs(E.bs_put_delta(spot, short_k, t, E.RISK_FREE_RATE, q, iv))
                if (E.DTE_RANGE[0] <= dte <= E.DTE_RANGE[1]
                        and E.DELTA_LO <= d <= E.DELTA_HI):
                    gaps_in_band.append(gap)
    all_stats = _describe(gaps)
    band_stats = _describe(gaps_in_band)
    return {
        "what": ("curve_spread minus raw close-difference, over every liquid "
                 "5-wide adjacent put pair on every day of every fetched "
                 "chain. No gate applied."),
        "ungated": all_stats,
        "ungated_in_dte_and_delta_band": band_stats,
        # "unbiased" = the mean gap is small next to its own dispersion (and,
        # for a synthetic chain that already sits on one curve, small outright).
        "estimator_is_unbiased": (
            abs(all_stats.get("mean", 1.0))
            < max(0.1 * (all_stats.get("sd") or 0.0), 1e-6)),
        "credit_noise_sd_as_pct_of_5_wide_width": (
            None if all_stats.get("sd") is None
            else 100.0 * all_stats["sd"] / 5.0),
    }


def selection_bias(ch: Chains, qualified: list[dict], null: dict) -> dict:
    """Where the gate-selected credits sit inside the ungated null."""
    shifts = []
    for r in qualified:
        curve = ch.curve_spread(r["symbol"], date.fromisoformat(r["expiry"]),
                                date.fromisoformat(r["entry_date"]),
                                r["short_strike"], r["long_strike"])
        if curve is not None:
            shifts.append(curve - r["credit"])
    sel = _describe(shifts)
    band = null["ungated_in_dte_and_delta_band"]
    ungated_sd = null["ungated"].get("sd")
    band_sd = band.get("sd")
    shift = (None if not sel.get("n") or not band.get("n")
             else sel["mean"] - band["mean"])
    return {
        "what": ("the null statistic evaluated on the GATE-SELECTED entries. "
                 "Negative = the booked credit is HIGHER than a same-day "
                 "cross-leg-consistent estimate, i.e. the floor selected days "
                 "whose raw print pair happened to be wide."),
        "selected": sel,
        "shift_vs_in_band_null_dollars": shift,
        "shift_as_pct_of_5_wide_width": (None if shift is None
                                         else 100.0 * shift / 5.0),
        # Two yardsticks: the in-band null is the like-for-like comparison
        # sample (same DTE and delta window as the gate), the ungated null is
        # the widest one. Both are reported so neither can be cherry-picked.
        "shift_in_in_band_null_sd": (None if shift is None or not band_sd
                                     else shift / band_sd),
        "shift_in_ungated_null_sd": (None if shift is None or not ungated_sd
                                     else shift / ungated_sd),
        "implied_measurement_bias_in_credit_r": (
            None if shift is None or not sel["n"] else
            float(np.mean([abs(shift) * 100.0 / r["risk_dollars"]
                           for r in qualified]))),
    }


def exit_trigger_audit(ch: Chains, qualified: list[dict],
                       n_examples: int = 6) -> dict:
    """Do the booked exits survive a same-day cross-leg-consistent mark?"""
    rows = []
    for r in qualified:
        expiry = date.fromisoformat(r["expiry"])
        exit_day = date.fromisoformat(r["exit_date"])
        puts = ch.chain(r["symbol"], expiry)
        if not puts:
            continue
        bs = puts[r["short_strike"]]["bars"].get(exit_day)
        bl = puts[r["long_strike"]]["bars"].get(exit_day)
        if bs is None or bl is None:
            continue
        ps, pl = E._close_of(bs), E._close_of(bl)
        if ps is None or pl is None:
            continue
        raw = ps - pl
        curve = ch.curve_spread(r["symbol"], expiry, exit_day,
                                r["short_strike"], r["long_strike"])
        spot_in = ch.closes[r["symbol"]].get(date.fromisoformat(r["entry_date"]))
        spot_out = ch.closes[r["symbol"]].get(exit_day)
        rows.append({
            "entry_date": r["entry_date"], "symbol": r["symbol"],
            "short_strike": r["short_strike"], "long_strike": r["long_strike"],
            "credit": r["credit"], "exit_date": r["exit_date"],
            "exit_reason": r["exit_reason"], "result_r": r["result_r"],
            "raw_exit_mark": raw, "curve_exit_mark": curve,
            "pt_level": E.PT_FRAC * r["credit"],
            "spot_entry": spot_in, "spot_exit": spot_out,
            "underlying_move_pct": (None if not spot_in or not spot_out
                                    else 100.0 * (spot_out / spot_in - 1.0)),
            "curve_above_pt_level": (None if curve is None
                                     else curve > E.PT_FRAC * r["credit"]),
        })
    pt = [x for x in rows if x["exit_reason"] == "profit_take"
          and x["curve_exit_mark"] is not None]
    spurious = [x for x in pt if x["curve_above_pt_level"]]
    gaps_pt = [x["curve_exit_mark"] - x["raw_exit_mark"] for x in pt]
    gaps_21 = [x["curve_exit_mark"] - x["raw_exit_mark"] for x in rows
               if x["exit_reason"] == "dte_21" and x["curve_exit_mark"] is not None]
    worst = sorted(spurious, key=lambda x: -(x["curve_exit_mark"]
                                             - x["raw_exit_mark"]))
    flattest = sorted(
        [x for x in spurious if x["underlying_move_pct"] is not None],
        key=lambda x: abs(x["underlying_move_pct"]))
    return {
        "what": ("for every booked exit, the raw close-difference mark that "
                 "triggered it vs the same-day cross-leg-consistent spread."),
        "n_profit_takes_testable": len(pt),
        "n_profit_takes_the_curve_says_never_triggered": len(spurious),
        "spurious_profit_take_fraction": (len(spurious) / len(pt)) if pt else None,
        "curve_minus_raw_on_profit_take_days": _describe(gaps_pt),
        "curve_minus_raw_on_21_dte_days": _describe(gaps_21),
        "most_extreme_spurious": worst[:n_examples],
        "spurious_on_the_flattest_underlying_days": flattest[:n_examples],
        "spurious_with_underlying_falling": sorted(
            [x for x in spurious if (x["underlying_move_pct"] or 0) < 0],
            key=lambda x: x["underlying_move_pct"])[:n_examples],
        "all_rows": rows,
    }


def survivorship(ch: Chains, records: list[dict]) -> dict:
    groups: dict[str, dict[str, list[float]]] = {
        k: {"trailing": [], "forward": []} for k in
        ("dropped_not_data_adequate", "kept_data_adequate",
         "kept_and_qualified", "kept_and_gate_rejected")}
    for r in records:
        day = date.fromisoformat(r["entry_date"])
        sym = r["symbol"]
        trailing = ch.realized_vol(sym, day, 20, forward=False)
        forward = ch.realized_vol(sym, day, 10, forward=True)
        if trailing is None or forward is None:
            continue
        keys = (["kept_data_adequate"] if r.get("data_adequate")
                else ["dropped_not_data_adequate"])
        if r.get("data_adequate"):
            keys.append("kept_and_qualified" if r["qualified"]
                        else "kept_and_gate_rejected")
        for k in keys:
            groups[k]["trailing"].append(trailing)
            groups[k]["forward"].append(forward)
    out = {k: {"trailing_rv20": _describe(v["trailing"]),
               "forward_rv10": _describe(v["forward"])}
           for k, v in groups.items()}
    return {
        "what": ("realized vol of the entries EXCLUDED from the availability "
                 "denominator vs those kept. If dropping is vol-correlated, "
                 "the surviving sample is not a random subsample and the "
                 "availability rate is biased."),
        "groups": out,
        "welch_t_trailing_dropped_minus_kept": _welch_t(
            groups["dropped_not_data_adequate"]["trailing"],
            groups["kept_data_adequate"]["trailing"]),
        "interpretation": (
            "|t| > 2 means the drop is vol-correlated; dropped days being "
            "LOWER vol biases availability UP, because low-IV days are the "
            "days a credit floor is hardest to clear."),
    }


def denominator_hygiene(records: list[dict]) -> dict:
    counts = Counter(
        (r["reason"], r.get("reason_class"), bool(r.get("data_adequate")))
        for r in records if not r["qualified"])
    n_all = len(records)
    n_adequate = sum(1 for r in records if r.get("data_adequate"))
    n_marks = sum(1 for r in records if r.get("marks_present"))
    n_q = sum(1 for r in records if r["qualified"])
    n_bracketed = sum(1 for r in records if r.get("data_adequate")
                      and r.get("brackets_delta_band"))
    excluded_gate = sum(n for (_, klass, adeq), n in counts.items()
                        if klass == "gate" and not adeq)
    denominators = {
        "planned": n_all, "marks_present": n_marks,
        "data_adequate": n_adequate, "data_adequate_and_band_bracketed": n_bracketed,
    }
    return {
        "reason_breakdown": [
            {"reason": reason, "class": klass, "data_adequate": adeq, "n": n}
            for (reason, klass, adeq), n in sorted(counts.items(),
                                                   key=lambda kv: str(kv[0]))],
        "n_qualified": n_q,
        "denominators": denominators,
        "availability_by_denominator": {
            k: (n_q / v if v else None) for k, v in denominators.items()},
        "passes_60pct_by_denominator": {
            k: (None if not v else (n_q / v) >= E.AVAILABILITY_FLOOR)
            for k, v in denominators.items()},
        "gate_classified_entries_excluded_from_data_adequate": excluded_gate,
    }


def band_bracket_audit(records: list[dict]) -> dict:
    rows = []
    for r in records:
        if r["qualified"] or r["reason"] not in ("band-not-fetched",
                                                "no-strike-in-band"):
            continue
        diag = r.get("diag") or {}
        rows.append({
            "entry_date": r["entry_date"], "symbol": r["symbol"],
            "spot": r.get("spot"), "reason": r["reason"],
            "reason_class": r.get("reason_class"),
            "expiry": r.get("reason_expiry"),
            "n_liquid_invertible": diag.get("n_iv_ok"),
            "delta_min": diag.get("delta_min"),
            "delta_max": diag.get("delta_max"),
            "brackets_delta_band": diag.get("brackets_delta_band"),
        })
    return {
        "what": ("every band-related rejection with the fetched grid's "
                 "|delta| range. brackets_delta_band=False means the strikes "
                 "that would have answered the question were never fetched."),
        "delta_band": [E.DELTA_LO, E.DELTA_HI],
        "n_rows": len(rows),
        "n_data_gaps_band_not_fetched": sum(
            1 for x in rows if x["reason"] == "band-not-fetched"),
        "n_true_gate_rejections": sum(
            1 for x in rows if x["reason"] == "no-strike-in-band"),
        "rows": rows,
    }


def blocking(qualified: list[dict]) -> dict:
    weeks = E.entry_week_blocks(qualified)
    clusters = E.exposure_cluster_blocks(qualified)
    spans = [(date.fromisoformat(r["entry_date"]),
              date.fromisoformat(r["exit_date"]),
              date.fromisoformat(r["entry_date"]).isocalendar()[:2])
             for r in qualified]
    cross = sum(1 for i in range(len(spans)) for j in range(i + 1, len(spans))
                if spans[i][2] != spans[j][2]
                and spans[i][0] <= spans[j][1] and spans[j][0] <= spans[i][1])
    holds = [(b - a).days for a, b, _ in spans]
    return {
        "n_trades": len(qualified),
        "entry_week": {"n_blocks": len(weeks),
                       "block_sizes": sorted(len(v) for v in weeks.values())},
        "exposure_cluster": {"n_blocks": len(clusters),
                             "block_sizes": sorted(len(v) for v in clusters.values())},
        "pairs_open_simultaneously_in_different_entry_weeks": cross,
        "total_pairs": len(spans) * (len(spans) - 1) // 2,
        "holding_period_days": _describe(holds) if holds else {"n": 0},
        "why_it_matters": (
            "entry-week blocks cannot contain a holding period longer than a "
            "week, so simultaneously-open trades are resampled as independent "
            "draws and the LB95 is too tight."),
    }


def walk_coverage(ch: Chains, qualified: list[dict]) -> dict:
    total_missing = 0
    worst = []
    for r in qualified:
        expiry = date.fromisoformat(r["expiry"])
        entry = date.fromisoformat(r["entry_date"])
        exit_day = date.fromisoformat(r["exit_date"])
        puts = ch.chain(r["symbol"], expiry)
        if not puts:
            continue
        short_days = set(puts[r["short_strike"]]["bars"])
        long_days = set(puts[r["long_strike"]]["bars"])
        window = {d for d in ch.closes[r["symbol"]] if entry < d <= exit_day}
        missing = [d for d in window
                   if not (d in short_days and d in long_days)]
        total_missing += len(missing)
        if missing:
            worst.append({"entry_date": r["entry_date"], "symbol": r["symbol"],
                          "n_unwalkable": len(missing),
                          "n_days_in_window": len(window)})
    return {
        "what": ("trading days inside a held window where at least one leg "
                 "has no bar, so the day was never marked."),
        "unwalkable_days_total": total_missing,
        "reported_skipped_days_total": sum(r.get("skipped_days", 0)
                                          for r in qualified),
        "trades_with_unwalkable_days": worst,
    }


def width_census(ch: Chains, batch_plan: Path) -> dict:
    """What the market actually offers, per width, before any floor is applied.

    For every planned entry day at DTE 30-45, every liquid invertible in-band
    short candidate, and every width in WIDTHS: the credit/width actually
    constructible. This is the gate-DESIGN measurement — it says whether the
    spec's credit >= width/4 is above or below the market's median — and it is
    independent of every mark-convention argument.
    """
    plan = E.load_batch_plan(batch_plan)
    grouped: dict[tuple[date, str], list[date]] = defaultdict(list)
    for row in plan:
        key = (row["entry_date"], row["symbol"])
        if row["expiry"] not in grouped[key]:
            grouped[key].append(row["expiry"])
    fracs: dict[str, list[float]] = defaultdict(list)
    constructible_entries: dict[str, set] = defaultdict(set)
    for (entry, symbol), expiries in sorted(grouped.items()):
        spot = ch.closes[symbol].get(entry)
        if spot is None:
            continue
        q = E.DIV_YIELD.get(symbol, 0.0)
        for expiry in sorted(expiries):
            dte = (expiry - entry).days
            if not (E.DTE_RANGE[0] <= dte <= E.DTE_RANGE[1]):
                continue
            puts = ch.chain(symbol, expiry)
            if not puts:
                continue
            t = dte / 365.0
            have = {k for k in puts
                    if E._entry_mark(puts[k], entry) is not None}
            for width in E.WIDTHS:
                if any((k - width) in have for k in have):
                    constructible_entries[f"{width:g}"].add((entry, symbol))
            for strike in sorted(puts):
                mark = E._entry_mark(puts[strike], entry, E.MIN_SHORT_TRADES,
                                     "same-day")
                if mark is None:
                    continue
                iv = E.implied_vol_put(mark, spot, strike, t, E.RISK_FREE_RATE, q)
                if iv is None:
                    continue
                d = abs(E.bs_put_delta(spot, strike, t, E.RISK_FREE_RATE, q, iv))
                if not (E.DELTA_LO <= d <= E.DELTA_HI):
                    continue
                for width in E.WIDTHS:
                    wing = puts.get(strike - width)
                    long_mark = E._entry_mark(wing, entry) if wing else None
                    if long_mark is None:
                        continue
                    credit = mark - long_mark
                    if 0 < credit < width:
                        fracs[f"{width:g}"].append(credit / width)
    per_width = {}
    for width, values in sorted(fracs.items()):
        stats = _describe(values)
        stats["n_clearing_spec_floor"] = sum(
            1 for v in values if v >= E.CREDIT_FLOOR_FRAC)
        stats["frac_clearing_spec_floor"] = (
            stats["n_clearing_spec_floor"] / len(values) if values else None)
        stats["n_entry_days_with_a_constructible_pair"] = len(
            constructible_entries.get(width, ()))
        per_width[width] = stats
    return {
        "what": ("credit/width of every constructible vertical on an in-band "
                 "liquid short candidate, per width, with no credit floor "
                 "applied."),
        "spec_credit_floor_frac": E.CREDIT_FLOOR_FRAC,
        "n_planned_entries": len(grouped),
        "per_width": per_width,
        "interpretation": (
            "if the spec floor sits ABOVE the median credit/width, low "
            "availability is a gate-design consequence, not a data artifact — "
            "and widening cannot help when the wider width's median is lower."),
    }


def payoff_attribution(qualified: list[dict], iv_by_month: dict) -> dict:
    if not qualified:
        return {"n_trades": 0}
    ordered = sorted(qualified, key=lambda r: r["result_r"])
    rs = np.array([r["result_r"] for r in qualified])
    drop_worst = {}
    for k in WORST_K:
        if k >= len(ordered):
            continue
        rest = [r["result_r"] for r in ordered[k:]]
        drop_worst[str(k)] = {
            "n_remaining": len(rest), "mean_r": float(np.mean(rest)),
            "dropped_r": [round(r["result_r"], 4) for r in ordered[:k]]}
    by_month = defaultdict(list)
    for r in qualified:
        by_month[r["entry_date"][:7]].append(r["result_r"])
    drop_month = {}
    for month in sorted(by_month):
        rest = [v for m, vs in by_month.items() if m != month for v in vs]
        uncond = iv_by_month["unconditional"].get(month, {})
        drop_month[month] = {
            "n_in_month": len(by_month[month]),
            "mean_r_in_month": float(np.mean(by_month[month])),
            "n_remaining": len(rest),
            "mean_r_excluding_month": float(np.mean(rest)) if rest else None,
            "unconditional_median_iv": uncond.get("median_iv"),
            "is_highest_iv_month": uncond.get("is_highest_iv_month"),
        }
    losers = [r["result_r"] for r in qualified if r["result_r"] <= 0]
    return {
        "n_trades": len(qualified),
        "mean_r": float(rs.mean()), "median_r": float(np.median(rs)),
        "min_r": float(rs.min()), "max_r": float(rs.max()),
        "n_losers": len(losers),
        "mean_r_of_losers": float(np.mean(losers)) if losers else None,
        "mean_r_excluding_k_worst": drop_worst,
        "mean_r_excluding_each_month": drop_month,
        "worst_r_observed": float(rs.min()),
        "full_loss_r": -1.0,
        "tail_note": (
            "the -1R tail is UNSAMPLED, not absent: a full -1R needs the short "
            "strike still breached at the forced 21-DTE exit, which never "
            "happened in this window. Its absence is a property of the window, "
            "not of the strategy."),
        "attribution_warning": (
            "deleting a month is not a neutral operation — check "
            "unconditional_median_iv on the month being deleted before "
            "reading any 'excluding X' figure as the strategy's expectancy."),
    }


def iv_by_month(ch: Chains, batch_plan: Path, qualified: list[dict]) -> dict:
    """UNCONDITIONAL short-leg IV: every liquid invertible in-band strike on
    every planned entry day at DTE 30-45, regardless of whether it qualified."""
    plan = E.load_batch_plan(batch_plan)
    grouped: dict[tuple[date, str], list[date]] = defaultdict(list)
    for row in plan:
        key = (row["entry_date"], row["symbol"])
        if row["expiry"] not in grouped[key]:
            grouped[key].append(row["expiry"])
    per_month: dict[str, list[float]] = defaultdict(list)
    days_seen: dict[str, set] = defaultdict(set)
    for (entry, symbol), expiries in sorted(grouped.items()):
        spot = ch.closes[symbol].get(entry)
        if spot is None:
            continue
        q = E.DIV_YIELD.get(symbol, 0.0)
        month = entry.isoformat()[:7]
        for expiry in sorted(expiries):
            dte = (expiry - entry).days
            if not (E.DTE_RANGE[0] <= dte <= E.DTE_RANGE[1]):
                continue
            puts = ch.chain(symbol, expiry)
            if not puts:
                continue
            t = dte / 365.0
            for strike in sorted(puts):
                mark = E._entry_mark(puts[strike], entry, E.MIN_SHORT_TRADES,
                                     "same-day")
                if mark is None:
                    continue
                iv = E.implied_vol_put(mark, spot, strike, t, E.RISK_FREE_RATE, q)
                if iv is None:
                    continue
                d = abs(E.bs_put_delta(spot, strike, t, E.RISK_FREE_RATE, q, iv))
                if E.DELTA_LO <= d <= E.DELTA_HI:
                    per_month[month].append(iv)
                    days_seen[month].add((entry, symbol))
    uncond = {}
    for month, ivs in sorted(per_month.items()):
        a = np.asarray(ivs)
        uncond[month] = {"n_strikes": int(a.size),
                         "n_entry_days": len(days_seen[month]),
                         "median_iv": float(np.median(a)),
                         "mean_iv": float(a.mean())}
    if uncond:
        top = max(uncond, key=lambda m: uncond[m]["median_iv"])
        for month, row in uncond.items():
            row["is_highest_iv_month"] = month == top
    cond: dict[str, dict] = {}
    by_month = defaultdict(list)
    for r in qualified:
        by_month[r["entry_date"][:7]].append(r["short_iv"])
    for month, ivs in sorted(by_month.items()):
        cond[month] = {"n_trades": len(ivs),
                       "median_iv": float(np.median(ivs))}
    return {
        "what": ("UNCONDITIONAL IV is measured over every in-band liquid "
                 "strike on every planned entry day. The CONDITIONAL figures "
                 "are the qualified trades only and are selected on credit, "
                 "hence on IV — they cannot be read as the month's vol "
                 "level."),
        "unconditional": uncond,
        "conditional_qualified_only": cond,
        "liquidity_lag_used": "same-day",
        "liquidity_lag_note": ("this section deliberately uses the same-day "
                               "count so the IV census covers the widest "
                               "strike set; it is a description of the data, "
                               "not a tradeable screen."),
    }


# ------------------------------------------------------------- driver


def diagnose(data_dir: Path, batch_plan: Path, entries_path: Path,
             symbols: tuple[str, ...], spread_pct: dict[str, float]) -> dict:
    prev = E.set_mark_convention("close")   # diagnostics always read RAW closes
    try:
        ch = Chains(data_dir, symbols)
        records = _load_entries(entries_path)
        qualified = [r for r in records if r["qualified"]]
        null = null_study(ch)
        ivs = iv_by_month(ch, batch_plan, qualified)
        return {
            "entries_file": entries_path.name,
            "n_records": len(records), "n_qualified": len(qualified),
            "mark_convention_of_entries": (
                qualified[0].get("mark_convention") if qualified else None),
            "liquidity_lag_of_entries": (
                qualified[0].get("liquidity_lag") if qualified else None),
            "null_study": null,
            "selection_bias": selection_bias(ch, qualified, null),
            "exit_trigger_audit": exit_trigger_audit(ch, qualified),
            "survivorship": survivorship(ch, records),
            "denominator_hygiene": denominator_hygiene(records),
            "band_bracket_audit": band_bracket_audit(records),
            "blocking": blocking(qualified),
            "walk_coverage": walk_coverage(ch, qualified),
            "friction": E.friction_comparison(qualified, spread_pct),
            "iv_by_month": ivs,
            "width_census": width_census(ch, batch_plan),
            "payoff_attribution": payoff_attribution(qualified, ivs),
        }
    finally:
        E.set_mark_convention(prev)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--batch-plan", required=True, type=Path)
    ap.add_argument("--entries", required=True, type=Path, action="append",
                    help="entries.jsonl to diagnose; repeat for several "
                         "samples (e.g. the primary and the upper bound)")
    ap.add_argument("--label", action="append", default=None,
                    help="label for each --entries, in the same order")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--spread-pct", default="SPY=0.0049,QQQ=0.0094")
    args = ap.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    spread_pct = E.parse_spread_pct(args.spread_pct)
    labels = args.label or [p.stem for p in args.entries]
    if len(labels) != len(args.entries):
        raise SystemExit("--label must be given once per --entries")
    out = {"data_dir": str(args.data_dir), "batch_plan": str(args.batch_plan),
           "spread_pct": spread_pct, "samples": {}}
    for label, path in zip(labels, args.entries):
        out["samples"][label] = diagnose(args.data_dir, args.batch_plan, path,
                                        symbols, spread_pct)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    for label, d in out["samples"].items():
        n = d["null_study"]["ungated"]
        sb = d["selection_bias"]
        ea = d["exit_trigger_audit"]
        print(f"[{label}] n_qualified={d['n_qualified']} "
              f"mark={d['mark_convention_of_entries']} "
              f"lag={d['liquidity_lag_of_entries']}")
        print(f"  null: n={n['n']} mean {n['mean']:+.4f} sd {n['sd']:.4f} "
              f"({d['null_study']['credit_noise_sd_as_pct_of_5_wide_width']:.2f}"
              f"pp of a $5 width)")
        if sb["shift_vs_in_band_null_dollars"] is None:
            print("  selection shift: not computable (no gate-selected entry "
                  "had a fittable same-day curve)")
        else:
            print(f"  selection shift {sb['shift_vs_in_band_null_dollars']:+.4f} "
                  f"= {sb['shift_in_in_band_null_sd']:+.2f} in-band-null sd "
                  f"({sb['shift_in_ungated_null_sd']:+.2f} ungated sd) "
                  f"= {sb['shift_as_pct_of_5_wide_width']:+.2f}pp of width")
        print(f"  spurious profit-takes "
              f"{ea['n_profit_takes_the_curve_says_never_triggered']}"
              f"/{ea['n_profit_takes_testable']}")
        wt = d["survivorship"]["welch_t_trailing_dropped_minus_kept"]
        dh = d["denominator_hygiene"]
        print(f"  availability by denominator: "
              + ", ".join(f"{k}={v:.4f}" for k, v in
                          dh["availability_by_denominator"].items() if v))
        print("  survivorship Welch t = "
              + ("n/a (insufficient underlying history)" if wt is None
                 else f"{wt:+.3f}"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
