"""Tests for scripts/replay_vertical_gates.py (M1-0.4 offline replay engine).

Pure offline: synthetic chains and tmp files only — no network, no project
services. BS hand values anchored on Hull's textbook European put example
and an independent quadrature integration of the risk-neutral payoff.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pytest

import scripts.replay_vertical_gates as rv

S = 100.0
T35 = 35 / 365
R = rv.RISK_FREE_RATE   # 0.043
Q = 0.012

ENTRY = date(2026, 5, 15)               # Friday
EXPIRY = date(2026, 6, 19)              # 35 DTE
FORCED = EXPIRY - timedelta(days=21)    # 2026-05-29


# ------------------------------------------------------------------ BS math


def test_bs_put_price_hull_hand_value():
    # Hull, Options Futures & Other Derivatives: S=42, K=40, r=10%,
    # sigma=20%, T=0.5y -> European put 0.81 (call 4.76).
    assert rv.bs_put_price(42, 40, 0.5, 0.10, 0.0, 0.2) == pytest.approx(0.8086, abs=1e-4)


def test_bs_put_delta_hull_hand_value():
    # Same Hull example: N(d1)=0.7791 -> put delta = 0.7791 - 1 = -0.2209.
    assert rv.bs_put_delta(42, 40, 0.5, 0.10, 0.0, 0.2) == pytest.approx(-0.2209, abs=1e-4)


def test_bs_put_price_matches_quadrature_with_dividends():
    # Independent check: E[e^{-rT} max(K - S_T, 0)] by dense quadrature
    # under S_T = S exp((r-q-sigma^2/2)T + sigma sqrt(T) Z).
    K, sigma = 95.0, 0.18
    z = np.linspace(-10, 10, 400_001)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    st = S * np.exp((R - Q - 0.5 * sigma**2) * T35 + sigma * math.sqrt(T35) * z)
    expected = math.exp(-R * T35) * np.trapezoid(np.maximum(K - st, 0.0) * pdf, z)
    assert rv.bs_put_price(S, K, T35, R, Q, sigma) == pytest.approx(expected, abs=1e-6)


def test_bs_put_delta_dividend_sign_and_bounds():
    d = rv.bs_put_delta(S, 95, T35, R, Q, 0.18)
    assert -1.0 < d < 0.0
    # deeper OTM => smaller magnitude
    assert abs(rv.bs_put_delta(S, 85, T35, R, Q, 0.18)) < abs(d)


def test_implied_vol_round_trip():
    for sigma in (0.12, 0.23, 0.55):
        price = rv.bs_put_price(S, 96, T35, R, Q, sigma)
        iv = rv.implied_vol_put(price, S, 96, T35, R, Q)
        assert iv == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_unattainable_prices():
    assert rv.implied_vol_put(0.0, S, 96, T35, R, Q) is None
    assert rv.implied_vol_put(-1.0, S, 96, T35, R, Q) is None
    assert rv.implied_vol_put(200.0, S, 96, T35, R, Q) is None  # above sigma=5 price
    assert rv.implied_vol_put(1.0, S, 96, 0.0, R, Q) is None    # expired


# ----------------------------------------------------- vertical selection


def _put(strike: float, close: float, n: int = 100,
         day: date = ENTRY) -> tuple[float, dict]:
    """A contract with a bar on `day` AND on the prior session.

    The prior-session bar carries the same trade count, because the liquidity
    screen defaults to the session BEFORE entry. That default is stricter than
    the engine's declared enter-at-the-close information set requires, not a
    look-ahead fix — see rv.INFORMATION_SET. The prior bar's close is
    irrelevant to selection; only the entry-day mark is ever read.
    """
    prior = day - timedelta(days=1)
    return strike, {"ticker": f"O:SPY260619P{int(strike * 1000):08d}",
                    "bars": {prior: {"c": close, "n": n},
                             day: {"c": close, "n": n}}}


def _chain(*items) -> dict[float, dict]:
    return dict(items)


def test_select_vertical_prefers_mid_band_delta_and_enforces_credit_floor():
    # sigma=0.25 closes: 95 -> |d| .2295, 96 -> .2725 (closest to .275),
    # 97 -> .3186. 96's 5-wing (91) exists but credit 1.048 < 1.25 -> per
    # spec no widening (wing HAS a bar), move to next candidate; 97's
    # 5-wing (92) gives credit 1.3697 >= 1.25 -> selected.
    chain = _chain(
        _put(95, 1.0730), _put(96, 1.3481), _put(97, 1.6697),
        _put(90, 0.2688), _put(91, 0.30), _put(92, 0.30),
    )
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel["qualified"] is True
    assert sel["short_strike"] == 97
    assert sel["long_strike"] == 92
    assert sel["width"] == 5
    assert sel["credit"] == pytest.approx(1.6697 - 0.30, abs=1e-9)
    assert rv.DELTA_LO <= abs(sel["short_delta"]) <= rv.DELTA_HI


def test_select_vertical_width_10_fallback_when_5_wing_has_no_bar():
    # Only 96 (close 2.8, |d| ~.334, in band) and 86 as the 10-wing; no 91
    # contract at all -> 5-wing missing -> width 10, credit 2.6 >= 2.5.
    chain = _chain(_put(96, 2.8), _put(86, 0.20))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel["qualified"] is True
    assert (sel["short_strike"], sel["long_strike"], sel["width"]) == (96, 86, 10)
    assert sel["credit"] == pytest.approx(2.6, abs=1e-9)


def test_band_not_fetched_is_data_when_the_grid_never_reaches_the_band():
    # |delta| ~ .015 / .002: the fetched grid tops out far below 0.35, so the
    # delta band was never OBSERVABLE. Calling that a gate rejection would
    # charge a data gap against the gate (finding 7).
    chain = _chain(_put(85, 0.0414), _put(80, 0.0035))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "band-not-fetched")
    assert rv.REASON_CLASS[sel["reason"]] == "data"
    assert sel["diag"]["brackets_delta_band"] is False
    assert sel["diag"]["delta_max"] < rv.DELTA_HI


def test_no_strike_in_band_is_a_gate_rejection_only_when_bracketed():
    # 94 -> |delta| .1902 (below the band), 98 -> .3672 (above it), nothing in
    # between: the grid straddles [0.20, 0.35] and the band is genuinely empty.
    chain = _chain(_put(94, 0.8413), _put(98, 2.0402))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "no-strike-in-band")
    assert rv.REASON_CLASS[sel["reason"]] == "gate"
    assert sel["diag"]["brackets_delta_band"] is True
    assert sel["diag"]["delta_min"] == pytest.approx(0.1902, abs=5e-4)
    assert sel["diag"]["delta_max"] == pytest.approx(0.3672, abs=5e-4)


def test_select_vertical_short_leg_liquidity_gate_n30():
    # In band, but the entry-day bar has too few trades: the liquidity screen
    # is part of the GATE stack, and it gets its own reason so it is never
    # confused with a missing-data skip.
    strike, contract = _put(96, 1.3481, n=10)
    sel = rv.select_vertical(ENTRY, EXPIRY, S, {strike: contract}, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "short-leg-illiquid")
    assert rv.REASON_CLASS[sel["reason"]] == "gate"
    assert sel["diag"]["n_entry_mark"] == 1  # the print existed...
    assert sel["diag"]["n_liquid"] == 0      # ...it just failed n >= 30


def test_select_vertical_wing_missing_is_classified_as_data_limited():
    # In band, but neither the 5- nor the 10-wide wing was ever fetched.
    # That is a data gap, not the gate rejecting a tradeable chain.
    chain = _chain(_put(96, 1.3481))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "wing-missing")
    assert rv.REASON_CLASS[sel["reason"]] == "data"
    assert sel["diag"]["wing_unfetched"] == 2      # 91 and 86 both absent
    assert sel["diag"]["n_constructible"] == 0     # no vertical constructible


def test_select_vertical_credit_below_floor():
    # Wing bar exists but credit 1.3481-0.60 = 0.7481 < 1.25.
    chain = _chain(_put(96, 1.3481), _put(91, 0.60))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "credit-below-floor")
    assert rv.REASON_CLASS[sel["reason"]] == "gate"
    assert sel["diag"]["n_constructible"] == 1     # data was adequate
    assert sel["diag"]["best_credit_frac"] == pytest.approx(0.7481 / 5.0)


def test_select_vertical_credit_floor_override_admits_the_same_chain():
    chain = _chain(_put(96, 1.3481), _put(91, 0.60))  # credit frac 0.14962
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q, credit_floor_frac=0.10)
    assert sel["qualified"] is True
    assert sel["credit"] == pytest.approx(0.7481, abs=1e-9)


def test_select_vertical_width_policy_any_widens_for_credit():
    # 96 is in band (|delta| .339); its 5-wing 91 prints but is illiquid, so it
    # is a usable WING yet never a short candidate. 5-wide credit 3.0-1.9 =
    # 1.10 < 1.25 -> five-first refuses to widen; "any" takes the 10-wide,
    # credit 3.0-0.4 = 2.60 >= 2.50.
    chain = _chain(_put(96, 3.0), _put(91, 1.9, n=5), _put(86, 0.4))
    five = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q,
                              width_policy="five-first")
    assert (five["qualified"], five["reason"]) == (False, "credit-below-floor")
    any_ = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q, width_policy="any")
    assert any_["qualified"] is True
    assert (any_["long_strike"], any_["width"]) == (86, 10)
    assert any_["credit"] == pytest.approx(2.6, abs=1e-9)


def test_stale_five_wide_print_no_longer_vetoes_the_ten_wide():
    # 96 is in band; its 5-wing 91 carries an INVERTED stale print (marks above
    # the short leg) and is itself illiquid so it can never be a short
    # candidate, while the 10-wing 86 is clean and clears the floor. The
    # original engine `break`-ed on the bad pair and never reached width 10
    # (finding 8); width freedom must actually be explored.
    chain = _chain(_put(96, 3.0), _put(91, 4.5, n=5), _put(86, 0.4))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q, width_policy="any")
    assert sel["qualified"] is True
    assert (sel["long_strike"], sel["width"]) == (86, 10)
    assert sel["diag"]["n_bad_credit"] == 1          # the stale pair is counted
    assert sel["diag"]["best_credit_frac_by_width"] == {"10": pytest.approx(0.26)}


def test_diag_records_constructible_counts_per_width():
    chain = _chain(_put(96, 1.3481), _put(86, 0.20))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel["diag"]["n_constructible_by_width"] == {"5": 0, "10": 1}


def test_select_vertical_rejects_unknown_width_policy():
    with pytest.raises(ValueError):
        rv.select_vertical(ENTRY, EXPIRY, S, _chain(_put(96, 1.3481)), Q,
                           width_policy="widest")


def test_select_vertical_inverted_credit_mark_is_data_not_gate():
    # Stale print: the further-OTM wing marks ABOVE the short leg.
    chain = _chain(_put(96, 1.3481), _put(91, 2.50))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "inverted-credit-mark")
    assert rv.REASON_CLASS[sel["reason"]] == "data"
    assert sel["diag"]["n_bad_credit"] == 1


def test_select_vertical_no_entry_day_mark_is_data():
    # Contracts exist but none printed on the entry date.
    chain = _chain(_put(96, 1.3481, day=date(2026, 5, 14)))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert (sel["qualified"], sel["reason"]) == (False, "no-entry-day-mark")
    assert rv.REASON_CLASS[sel["reason"]] == "data"


def test_count_constructible_needs_both_legs_on_the_entry_day():
    assert rv.count_constructible(ENTRY, _chain(_put(96, 1.3), _put(91, 0.3))) == 1
    assert rv.count_constructible(ENTRY, _chain(_put(96, 1.3), _put(86, 0.2))) == 1
    assert rv.count_constructible(ENTRY, _chain(_put(96, 1.3), _put(92, 0.3))) == 0
    # wing row exists but has no bar on the entry day -> not constructible
    assert rv.count_constructible(
        ENTRY, _chain(_put(96, 1.3), _put(91, 0.3, day=date(2026, 5, 14)))) == 0


# --------------------------------------------------------- managed walk


def _bars(marks: dict[date, float]) -> dict[date, dict]:
    return {d: {"c": c} for d, c in marks.items()}


def test_managed_walk_profit_take_exits_at_half_credit():
    credit = 1.4  # PT threshold 0.70
    short = _bars({date(2026, 5, 18): 1.20, date(2026, 5, 19): 0.80})
    long_ = _bars({date(2026, 5, 18): 0.15, date(2026, 5, 19): 0.15})
    walk = rv.managed_walk(ENTRY, EXPIRY, credit, short, long_)
    assert walk["exit_reason"] == "profit_take"
    assert walk["exit_date"] == date(2026, 5, 19)   # mark 0.65 <= 0.70
    assert walk["exit_debit"] == pytest.approx(0.70)  # fills AT the PT level
    assert walk["data_end"] is False


def test_managed_walk_forced_21_dte_exit_at_that_days_mark():
    credit = 1.4
    days = [date(2026, 5, 18) + timedelta(days=i) for i in range(15)]
    short = _bars({d: 1.30 for d in days})
    long_ = _bars({d: 0.20 for d in days})   # mark 1.10 > 0.70 forever
    walk = rv.managed_walk(ENTRY, EXPIRY, credit, short, long_)
    assert walk["exit_reason"] == "dte_21"
    assert walk["exit_date"] == FORCED           # first day >= expiry-21d
    assert walk["exit_debit"] == pytest.approx(1.10)
    assert walk["data_end"] is False


def test_managed_walk_profit_take_wins_over_forced_exit_same_day():
    credit = 1.4
    short = _bars({FORCED: 0.75})
    long_ = _bars({FORCED: 0.15})            # mark 0.60 <= 0.70 on the 21-DTE day
    walk = rv.managed_walk(ENTRY, EXPIRY, credit, short, long_)
    assert walk["exit_reason"] == "profit_take"
    assert walk["exit_debit"] == pytest.approx(0.70)


def test_managed_walk_data_end_flags_and_uses_last_mark():
    credit = 1.4
    short = _bars({date(2026, 5, 18): 1.30, date(2026, 5, 20): 1.25})
    long_ = _bars({date(2026, 5, 18): 0.20, date(2026, 5, 20): 0.18})
    walk = rv.managed_walk(ENTRY, EXPIRY, credit, short, long_)
    assert walk["exit_reason"] == "data_end"
    assert walk["data_end"] is True
    assert walk["exit_date"] == date(2026, 5, 20)
    assert walk["exit_debit"] == pytest.approx(1.07)


def test_managed_walk_no_post_entry_bars_flat_exit():
    walk = rv.managed_walk(ENTRY, EXPIRY, 1.4, _bars({}), _bars({}))
    assert walk["exit_reason"] == "data_end"
    assert walk["data_end"] is True
    assert walk["exit_debit"] == pytest.approx(1.4)  # flat: debit == credit
    assert walk["exit_date"] == ENTRY


# --------------------------------------------------- mark conventions


def test_mark_convention_selects_the_field_and_falls_back_to_close():
    bar = {"c": 1.50, "vw": 1.42, "h": 1.80, "l": 1.10, "n": 40}
    prev = rv.set_mark_convention("close")
    try:
        assert rv._close_of(bar) == pytest.approx(1.50)
        rv.set_mark_convention("vw")
        assert rv._close_of(bar) == pytest.approx(1.42)
        rv.set_mark_convention("hl2")
        assert rv._close_of(bar) == pytest.approx(1.45)
        # smile reads the close field; the repricing happens chain-wide
        rv.set_mark_convention("smile")
        assert rv._close_of(bar) == pytest.approx(1.50)
        # a convention whose own field is missing/absurd falls back to close
        rv.set_mark_convention("vw")
        assert rv._close_of({"c": 1.5, "vw": 0}) == pytest.approx(1.5)
        rv.set_mark_convention("hl2")
        assert rv._close_of({"c": 1.5}) == pytest.approx(1.5)
        # and an unusable close is still None, never fabricated
        rv.set_mark_convention("close")
        assert rv._close_of({"c": 0}) is None
    finally:
        rv.set_mark_convention(prev)


def test_set_mark_convention_rejects_unknown_names():
    with pytest.raises(ValueError):
        rv.set_mark_convention("mid")


def test_smile_repricing_makes_both_legs_come_off_one_curve():
    # A chain priced at a flat sigma=0.25 except that the 92 leg carries a
    # stale 0.75 print (BS says 0.4926). After repricing, EVERY leg is the BS
    # price of one quadratic-in-moneyness IV curve, so the IVs recovered from
    # the repriced grid must be exactly quadratic — third finite differences on
    # an evenly spaced strike run vanish. That property, not the level, is what
    # a spread mark needs: both legs share one curve.
    strikes = [88, 90, 92, 94, 95, 96, 97, 98]
    puts = {}
    for k in strikes:
        px = 0.75 if k == 92 else rv.bs_put_price(S, k, T35, R, Q, 0.25)
        _, contract = _put(float(k), px)
        puts[float(k)] = contract
    prev = rv.set_mark_convention("smile")
    try:
        closes = {ENTRY: S, ENTRY - timedelta(days=1): S}
        stats = rv.apply_smile_repricing(puts, EXPIRY, Q, closes)
        assert stats["days_fitted"] == 2
        assert stats["bars_repriced"] == 2 * len(strikes)
        assert stats["bars_kept_raw"] == 0
        priced = [puts[float(k)]["bars"][ENTRY]["c"] for k in strikes]
        assert all(b > a for a, b in zip(priced, priced[1:]))   # monotone in K
        # the stale 0.75 print is pulled back toward its arbitrage-free level
        assert 0.45 < priced[2] < 0.75
        ivs = [rv.implied_vol_put(puts[float(k)]["bars"][ENTRY]["c"], S, k,
                                  T35, R, Q) for k in (94, 95, 96, 97, 98)]
        d1 = np.diff(ivs)
        assert np.allclose(np.diff(d1, n=2), 0.0, atol=1e-7)
    finally:
        rv.set_mark_convention(prev)


def test_smile_repricing_keeps_raw_closes_when_no_curve_fits():
    # Fewer than SMILE_MIN_POINTS usable strikes -> the day is unfittable and
    # every bar keeps its raw close. Nothing is interpolated or invented.
    puts = dict([_put(96, 1.3481), _put(91, 0.30)])
    prev = rv.set_mark_convention("smile")
    try:
        stats = rv.apply_smile_repricing(puts, EXPIRY, Q, {ENTRY: S})
        assert stats["days_fitted"] == 0
        assert stats["days_unfittable"] == 2       # entry day and the prior one
        assert stats["bars_repriced"] == 0
        assert stats["bars_kept_raw"] == 4
        assert puts[96.0]["bars"][ENTRY]["c"] == pytest.approx(1.3481)
    finally:
        rv.set_mark_convention(prev)


# --------------------------------------- liquidity-screen information set


def test_the_information_set_is_declared_once_and_consistently():
    # finding 2: the artifact used to defend entry-day marks as a
    # close-to-close convention while calling the entry day's own full-day
    # trade count look-ahead. Both come off the SAME completed bar, so exactly
    # one information set governs both.
    assert rv.INFORMATION_SET == "enter-at-the-close"
    stmt = rv.INFORMATION_SET_STATEMENT
    assert "enter-at-the-close" in stmt.lower()
    # it must say the same-day count is legal, and why prior is still primary
    assert "'same-day' screen is legal" in stmt
    assert "robustness" in stmt
    assert "not implemented and not measured" in stmt
    # and the summary must publish it, with no residual look-ahead claim
    cfg = rv.summarize([], 10, 1, {"SPY": 0.0049}, "five-first", 0.25,
                       "same-day", "close")["config"]
    assert cfg["information_set"] == "enter-at-the-close"
    assert cfg["liquidity_screen_uses_future_information"] is False
    assert cfg["liquidity_screen_stricter_than_declared_information_set"] is False
    cfg_prior = rv.summarize([], 10, 1, {"SPY": 0.0049}, "five-first", 0.25,
                             "prior", "smile")["config"]
    assert cfg_prior["liquidity_screen_stricter_than_declared_information_set"] is True
    assert "liquidity_screen_is_look_ahead" not in cfg


def test_liquidity_screen_prior_day_vs_same_day():
    # The entry day trades 100 times; the session before it traded 5. Both
    # screens are legal under the declared information set; "prior" is the
    # stricter one, and this is the measured difference between them.
    _, contract = _put(96, 1.3481, n=100)
    contract["bars"][ENTRY - timedelta(days=1)]["n"] = 5
    assert rv.short_leg_is_liquid(contract, ENTRY, 30, "same-day") is True
    assert rv.short_leg_is_liquid(contract, ENTRY, 30, "prior") is False
    # a strike with no prior session at all fails the stricter screen
    fresh = {"bars": {ENTRY: {"c": 1.0, "n": 500}}}
    assert rv.short_leg_is_liquid(fresh, ENTRY, 30, "prior") is False
    assert rv.short_leg_is_liquid(fresh, ENTRY, 30, "same-day") is True
    with pytest.raises(ValueError):
        rv.short_leg_is_liquid(contract, ENTRY, 30, "intraday")


def test_select_vertical_honours_the_liquidity_lag():
    # Both strikes traded heavily on the entry day but barely the day before.
    # The wing needs no liquidity, so the same-day run qualifies while the
    # stricter prior-session run has no eligible short leg at all.
    chain = _chain(_put(97, 1.6697), _put(92, 0.30))
    for k in (97.0, 92.0):
        chain[k]["bars"][ENTRY - timedelta(days=1)]["n"] = 3
    assert rv.select_vertical(ENTRY, EXPIRY, S, chain, Q,
                              liquidity_lag="same-day")["qualified"] is True
    lagged = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q,
                                liquidity_lag="prior")
    assert (lagged["qualified"], lagged["reason"]) == (False, "short-leg-illiquid")


# ------------------------------------------------------------- friction


def test_friction_arithmetic_hand_value():
    # half-spread 0.0049/2 = 0.00245 on each of the four leg marks
    # (x100 multiplier) + $1 x 4 fees.
    got = rv.friction_dollars(0.0049, 1.6697, 0.30, 0.85, 0.15)
    expected = 0.00245 * (1.6697 + 0.30 + 0.85 + 0.15) * 100.0 + 4.0
    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(4.7275765, abs=1e-6)


def test_friction_fees_floor_when_marks_zero():
    assert rv.friction_dollars(0.0094, 0, 0, 0, 0) == pytest.approx(4.0)


# ------------------------------------------------------------- bootstrap


def _mk_entries(rs_by_date: dict[date, float]) -> list[dict]:
    return [{"entry_date": d.isoformat(), "symbol": "SPY", "result_r": r}
            for d, r in rs_by_date.items()]


def test_block_bootstrap_deterministic_for_fixed_seed():
    rng = np.random.default_rng(7)
    days = [date(2026, 4, 6) + timedelta(days=i) for i in range(28)]
    entries = _mk_entries({d: float(r) for d, r in zip(days, rng.normal(0.02, 0.3, 28))})
    a = rv.block_bootstrap_lb95_mean(entries, resamples=2000, seed=123)
    b = rv.block_bootstrap_lb95_mean(entries, resamples=2000, seed=123)
    c = rv.block_bootstrap_lb95_mean(entries, resamples=2000, seed=124)
    assert a == b
    assert a != c  # different seed resamples differently
    mean = float(np.mean([e["result_r"] for e in entries]))
    assert a < mean  # lower bound sits below the point estimate


def test_block_bootstrap_degenerate_cases():
    assert rv.block_bootstrap_lb95_mean([]) is None
    same = _mk_entries({date(2026, 4, 6) + timedelta(days=i): 0.1 for i in range(10)})
    assert rv.block_bootstrap_lb95_mean(same, resamples=100, seed=1) == pytest.approx(0.1)


def test_exposure_cluster_blocks_group_overlapping_holds():
    # A and B overlap; B and C overlap but A and C do not -> transitive closure
    # puts all three in ONE block. D is disjoint and stands alone.
    entries = [
        {"entry_date": "2026-04-06", "exit_date": "2026-04-10",
         "symbol": "SPY", "result_r": 0.1},
        {"entry_date": "2026-04-09", "exit_date": "2026-04-15",
         "symbol": "SPY", "result_r": 0.2},
        {"entry_date": "2026-04-14", "exit_date": "2026-04-20",
         "symbol": "QQQ", "result_r": -0.3},
        {"entry_date": "2026-05-04", "exit_date": "2026-05-06",
         "symbol": "SPY", "result_r": 0.4},
    ]
    blocks = rv.exposure_cluster_blocks(entries)
    assert len(blocks) == 2
    assert sorted(len(v) for v in blocks.values()) == [1, 3]
    # entry-week blocking splits the same overlapping trio into 2 blocks and
    # would resample them as if they were independent draws
    weeks = rv.entry_week_blocks(entries)
    assert len(weeks) == 3
    assert sorted(len(v) for v in weeks.values()) == [1, 1, 2]


def test_bootstrap_reports_both_blocking_schemes_and_rejects_unknown():
    entries = [
        {"entry_date": "2026-04-06", "exit_date": "2026-04-24",
         "symbol": "SPY", "result_r": 0.3},
        {"entry_date": "2026-04-20", "exit_date": "2026-04-30",
         "symbol": "QQQ", "result_r": -0.5},
        {"entry_date": "2026-06-01", "exit_date": "2026-06-03",
         "symbol": "SPY", "result_r": 0.2},
    ]
    week = rv.block_bootstrap_lb95_mean(entries, 2000, 5, "entry-week")
    clus = rv.block_bootstrap_lb95_mean(entries, 2000, 5, "exposure-cluster")
    assert week is not None and clus is not None
    assert week != clus            # the schemes are genuinely different
    with pytest.raises(ValueError):
        rv.block_bootstrap_lb95_mean(entries, 100, 1, "monthly")


def test_max_drawdown_r():
    entries = _mk_entries({
        date(2026, 4, 6): 0.15, date(2026, 4, 7): -1.0,
        date(2026, 4, 8): 0.15, date(2026, 4, 9): -0.5,
    })
    # path: .15, -.85, -.70, -1.20 ; peak .15 -> trough -1.20 => dd 1.35
    assert rv.max_drawdown_r(entries) == pytest.approx(1.35)
    assert rv.max_drawdown_r([]) == 0.0


def test_max_drawdown_r_orders_by_when_pnl_is_booked():
    # finding 7: a drawdown is a property of when P&L is BOOKED. The middle
    # trade here is opened second but held far longest, so it books last.
    entries = [
        {"entry_date": "2026-04-06", "exit_date": "2026-04-10",
         "symbol": "SPY", "result_r": -1.0},
        {"entry_date": "2026-04-20", "exit_date": "2026-05-29",
         "symbol": "QQQ", "result_r": -1.0},   # opened 2nd, booked LAST
        {"entry_date": "2026-05-04", "exit_date": "2026-05-08",
         "symbol": "SPY", "result_r": +0.2},
    ]
    # booking order: -1.0, -0.8, -1.8 -> peak 0.0, trough -1.8 => dd 1.8
    assert rv.max_drawdown_r(entries) == pytest.approx(1.8)
    # entry order gives -1.0, -2.0, -1.8 => dd 2.0, the old, wrong answer
    cum = peak = dd = 0.0
    for e in sorted(entries, key=lambda x: (x["entry_date"], x["symbol"])):
        cum += e["result_r"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    assert dd == pytest.approx(2.0)
    # deterministic when several trades book on the SAME day
    tie = [
        {"entry_date": "2026-04-06", "exit_date": "2026-04-30",
         "symbol": "SPY", "result_r": -0.5},
        {"entry_date": "2026-04-07", "exit_date": "2026-04-30",
         "symbol": "QQQ", "result_r": +0.3},
    ]
    assert rv.max_drawdown_r(tie) == rv.max_drawdown_r(list(reversed(tie)))
    # records without an exit_date fall back to entry_date and do not crash
    assert rv.max_drawdown_r(
        [{"entry_date": "2026-04-06", "symbol": "SPY", "result_r": -0.4}]
    ) == pytest.approx(0.4)


# -------------------------------------------------------------- CLI / e2e


def test_parse_spread_pct():
    assert rv.parse_spread_pct("SPY=0.0049,QQQ=0.0094") == {
        "SPY": 0.0049, "QQQ": 0.0094}
    with pytest.raises(ValueError):
        rv.parse_spread_pct("SPY")


# ------------------------------------------- friction provenance (B8)


def test_load_spread_calibration_prefers_per_underlying_and_records_source(tmp_path):
    path = tmp_path / "execution_costs.json"
    path.write_text(json.dumps({
        "calibrated_at": "2026-07-22T00:00:00+00:00",
        "spread_pct_of_mid_median": 0.07338,
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.0049,
                                   "n_quotes": 62, "zone": "delta 0.20-0.35"}},
    }))
    pct, prov = rv.load_spread_calibration(path, ("SPY", "QQQ"))
    assert pct["SPY"] == pytest.approx(0.0049)
    assert prov["per_symbol"]["SPY"]["source"] == "per_underlying"
    assert prov["per_symbol"]["SPY"]["n_quotes"] == 62
    assert prov["per_symbol"]["SPY"]["zone"] == "delta 0.20-0.35"
    # QQQ has no per-underlying entry: it inherits the single-name GLOBAL and
    # that inheritance must be shouted, not silently absorbed.
    assert pct["QQQ"] == pytest.approx(0.07338)
    assert prov["per_symbol"]["QQQ"]["source"] == "file_global"
    assert any("QQQ" in w and "GLOBAL" in w for w in prov["warnings"])
    assert prov["sha256"] and prov["file_present"] is True


def test_load_spread_calibration_missing_file_degrades_loudly(tmp_path):
    pct, prov = rv.load_spread_calibration(tmp_path / "nope.json", ("SPY",))
    assert pct["SPY"] == pytest.approx(rv.DEFAULT_SPREAD_PCT["SPY"])
    assert prov["file_present"] is False
    assert prov["per_symbol"]["SPY"]["source"] == "compiled_default"
    assert any("absent" in w for w in prov["warnings"])
    assert any("assumption" in w for w in prov["warnings"])


def test_load_spread_calibration_derives_full_spread_from_a_half_spread(tmp_path):
    path = tmp_path / "execution_costs.json"
    path.write_text(json.dumps({"opt_half_spread_pct_of_mark": 0.03669}))
    pct, prov = rv.load_spread_calibration(path, ("SPY",))
    assert pct["SPY"] == pytest.approx(2 * 0.03669)
    assert prov["per_symbol"]["SPY"]["key"].startswith("2 x ")


def test_friction_comparison_exposes_the_leg_mark_proxy_error():
    # short 3.00, long 2.00 -> credit 1.00, wing ratio 0.667, leg-mid-sum 5.00.
    # The deployed model charges the spread on 2 x credit = 2.00 of notional;
    # the honest model charges it on 5.00 -> 2.5x understatement.
    trade = {"symbol": "SPY", "credit": 1.0, "short_mark": 3.0,
             "long_mark": 2.0, "risk_dollars": 400.0, "friction": 12.0,
             "short_exit_mark": 1.5, "long_exit_mark": 1.0}
    fc = rv.friction_comparison([trade], {"SPY": 0.01})
    assert fc["wing_ratio_long_over_short"]["mean"] == pytest.approx(2 / 3)
    assert fc["leg_mid_sum_over_net_credit"]["mean"] == pytest.approx(5.0)
    assert fc["production_deployed_friction_dollars"]["mean"] == pytest.approx(
        2 * 0.01 * 1.0 * 100 + 4.0)                        # 6.0
    assert fc["production_honest_leg_mark_friction_dollars"]["mean"] == (
        pytest.approx(0.01 * 5.0 * 100 + 4.0))             # 9.0
    assert fc["ratio_honest_over_deployed"]["mean"] == pytest.approx(1.5)
    assert fc["replay_friction_r"]["mean"] == pytest.approx(12.0 / 400.0)
    # recompute_replay re-derives the replay column from the booked marks
    recomputed = rv.friction_comparison([trade], {"SPY": 0.01},
                                        recompute_replay=True)
    expected = rv.friction_dollars(0.01, 3.0, 2.0, 1.5, 1.0)
    assert recomputed["replay_friction_dollars"]["mean"] == pytest.approx(expected)


PRIOR = ENTRY - timedelta(days=1)


def _write_e2e_data(data_dir):
    days = [PRIOR, ENTRY, date(2026, 5, 18)]
    (data_dir / "underlying_SPY.json").write_text(json.dumps(
        {"results": [{"t": d.isoformat(), "c": 100.0} for d in days]}))

    def bars(closes):
        # Every contract also prints on the session BEFORE entry, because the
        # liquidity screen's knowable information set is that prior session.
        closes = {PRIOR: closes[ENTRY], **closes}
        return [{"t": d.isoformat(), "o": c, "h": c, "l": c, "c": c,
                 "vw": c, "v": 500, "n": 100} for d, c in closes.items()]

    rows = [
        {"ticker": "O:SPY260619P00095000", "strike": 95,
         "bars": bars({ENTRY: 1.0730})},
        {"ticker": "O:SPY260619P00096000", "strike": 96,
         "bars": bars({ENTRY: 1.3481})},
        # short 97 / long 92 qualifies; next-day mark 0.67 <= 0.5*1.3697 -> PT
        {"ticker": "O:SPY260619P00097000", "strike": 97,
         "bars": bars({ENTRY: 1.6697, date(2026, 5, 18): 0.80})},
        {"ticker": "O:SPY260619P00092000", "strike": 92,
         "bars": bars({ENTRY: 0.30, date(2026, 5, 18): 0.13})},
        {"ticker": "O:SPY260619P00091000", "strike": 91,
         "bars": bars({ENTRY: 0.30})},
        # a call: must be ignored by the put-vertical replay
        {"ticker": "O:SPY260619C00097000", "strike": 97,
         "bars": bars({ENTRY: 9.99})},
    ]
    with (data_dir / "contracts_SPY_2026-06-19.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    plan = {"batches": [{"symbol": "SPY", "expiry": "2026-06-19",
                         "entry_dates": [ENTRY.isoformat(),
                                         "2026-05-20"]}]}  # 5/20: no spot bar
    (data_dir / "batch_plan.json").write_text(json.dumps(plan))


def test_end_to_end_replay(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_e2e_data(data_dir)

    # Pinned on mark="close": the arithmetic below is hand-checked against the
    # raw prints. The primary convention (smile) reprices every leg off a
    # fitted curve, so it is exercised separately below.
    summary = rv.run(data_dir, data_dir / "batch_plan.json", out_dir,
                     {"SPY": 0.0049}, resamples=500, seed=42, mark="close")

    entries = [json.loads(line)
               for line in (out_dir / "entries.jsonl").read_text().splitlines()]
    assert len(entries) == 2
    ok = next(e for e in entries if e["entry_date"] == ENTRY.isoformat())
    bad = next(e for e in entries if e["entry_date"] == "2026-05-20")

    assert ok["qualified"] is True
    assert (ok["short_strike"], ok["long_strike"], ok["width"]) == (97, 92, 5)
    assert ok["exit_reason"] == "profit_take"
    credit = 1.6697 - 0.30
    assert ok["credit"] == pytest.approx(credit, abs=1e-9)
    assert ok["pnl_gross"] == pytest.approx(0.5 * credit * 100.0)
    expected_friction = 0.00245 * (1.6697 + 0.30 + 0.80 + 0.13) * 100 + 4.0
    assert ok["friction"] == pytest.approx(expected_friction, abs=1e-9)
    assert ok["result_r"] == pytest.approx(
        (0.5 * credit * 100 - expected_friction) / ((5 - credit) * 100))

    assert bad["qualified"] is False
    assert bad["reason"] == "no-underlying-bar"

    disk_summary = json.loads((out_dir / "summary.json").read_text())
    assert disk_summary == summary
    avail = summary["availability"]
    # 2 planned entries; 5/20 has no spot bar at all, so it is a DATA gap and
    # must NOT be charged against the gate. Coverage 1/2, availability 1/1.
    overall = avail["overall"]
    assert overall["n_entries"] == 2
    assert overall["n_data_adequate"] == 1
    assert overall["data_coverage_rate"] == pytest.approx(0.5)
    assert overall["availability_rate"] == pytest.approx(1.0)
    assert overall["availability_rate_strict"] == pytest.approx(1.0)
    assert avail["per_symbol"]["SPY"]["availability_rate"] == pytest.approx(1.0)
    assert avail["per_month"]["2026-05"]["data_coverage_rate"] == pytest.approx(0.5)
    assert avail["unqualified_reasons"] == {"no-underlying-bar": 1}
    assert avail["unqualified_reason_classes"] == {"data": 1}
    assert summary["data_inventory"]["skipped_call_rows"] == 1
    mp = summary["managed_payoff"]
    assert mp["n_trades"] == 1
    assert mp["win_rate"] == 1.0
    assert mp["mean_r"] == pytest.approx(ok["result_r"])
    assert mp["lb95_mean_r"] == pytest.approx(ok["result_r"])  # single block
    assert summary["exit_degradation"]["counts"] == {
        "profit_take": 1, "dte_21": 0, "data_end": 0}
    # The European-BS approximation must be disclosed in the report.
    assert any("EUROPEAN" in note for note in summary["approximation_notes"])
    # BOTH denominators, each with its own verdict (finding 5).
    assert overall["availability_rate_planned"] == pytest.approx(0.5)
    assert overall["passes_floor_on_data_adequate"] is True
    assert overall["passes_floor_on_planned"] is False
    # and the acceptance block must say the plan's criterion was NOT evaluated
    verdict = summary["verdict"]
    assert verdict["evaluated_as_specified"] is False
    assert verdict["regime_labels_present"] is False
    assert "60%" in verdict["criterion_text"] or "≥60%" in verdict["criterion_text"]
    assert verdict["denominators"]["planned"]["n"] == 2
    assert verdict["denominators"]["data_adequate"]["n"] == 1
    assert rv.PLAN_CRITERION_SOURCE in verdict["criterion_source"]
    # both blocking schemes are reported side by side (finding 9)
    assert set(mp["lb95_mean_r_by_blocking"]) == {"entry-week", "exposure-cluster"}
    assert mp["n_blocks"] == {"entry-week": 1, "exposure-cluster": 1}
    # friction provenance is recorded, not assumed
    prov = summary["config"]["spread_pct_provenance"]
    assert prov["warnings"]          # caller-supplied spreads are flagged


def test_end_to_end_sensitivity_table_covers_every_convention(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_e2e_data(data_dir)

    summary = rv.run(data_dir, data_dir / "batch_plan.json", out_dir,
                     {"SPY": 0.0049}, resamples=200, seed=42,
                     mark="smile", sensitivity=True)

    rows = summary["sensitivity"]["rows"]
    for m in rv.MARK_CONVENTIONS:
        assert f"mark-{m}" in rows
    assert "liquidity-same-day" in rows       # the less-strict screen variant
    assert "width-any" in rows
    assert summary["sensitivity"]["primary"]["mark"] == "smile"
    assert rows["mark-close"]["mark"] == "close"
    assert "UPPER BOUND" in rows["mark-close"]["note"]
    assert "PRIMARY" in rows["mark-smile"]["note"]
    assert rows["liquidity-same-day"]["liquidity_lag"] == "same-day"
    # finding 2: the row is labelled by the declared information set, and no
    # row may still call the same-day count look-ahead.
    assert "LEGAL UNDER THE DECLARED INFORMATION SET" in rows[
        "liquidity-same-day"]["note"]
    assert not any("LOOK-AHEAD" in r["note"] for r in rows.values())
    # every row ships the entries file it was computed from
    for tag, row in rows.items():
        assert (out_dir / row["entries_file"]).exists()
        assert row["n_entries"] == 2
    # the smile run repriced bars off a fitted curve, and says how many
    md = summary["mark_diagnostics"]
    assert md["convention"] == "smile"
    assert md["smile"]["days_fitted"] >= 1
    # finding 12: the kept-raw count is split by cause, and the two parts add
    # back to the total. Only the fitted-day part can create a MIXED pair.
    sm = md["smile"]
    assert (sm["bars_kept_raw_on_unfittable_days"]
            + sm["bars_kept_raw_on_fitted_days"]) == sm["bars_kept_raw"]
    # and the no-mixed-pair claim is a published per-trade audit, not prose
    rme = md["raw_mark_exposure"]
    assert rme["n_marked_pair_days_mixed"] == 0
    assert rme["no_mixed_pair_anywhere"] is True
    assert rme["n_trades_with_a_mixed_pair_day"] == 0
    assert 0 <= rme["n_trades_with_any_raw_marked_leg_day"] <= rme["n_trades"]
    assert rme["n_marked_leg_days"] == 2 * rme["n_marked_pair_days"]
    # a raw leg-day always shows up in one of the two pair buckets
    assert (rme["n_marked_pair_days_both_raw"] * 2
            + rme["n_marked_pair_days_mixed"]) == rme["n_marked_leg_days_raw"]
    # non-smile rows carry no raw-mark audit (nothing was repriced)
    assert rows["mark-close"]["raw_mark_exposure"] is None
    assert rows["mark-smile"]["raw_mark_exposure"] is not None


def test_records_persist_diagnostics_even_when_unqualified(tmp_path):
    # finding 11: a rejection must be re-auditable from entries.jsonl alone.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_e2e_data(data_dir)
    records, _, _ = rv.replay_records(
        data_dir, data_dir / "batch_plan.json", {"SPY": 0.0049}, mark="close",
        credit_floor_frac=0.9)                       # nothing can qualify
    rec = next(r for r in records if r["entry_date"] == ENTRY.isoformat())
    assert rec["qualified"] is False
    assert rec["reason"] == "credit-below-floor"
    assert rec["diag"] is not None
    assert rec["diag"]["n_in_band"] >= 1
    assert rec["diag"]["brackets_delta_band"] is not None
    assert rec["diag"]["mark_convention"] == "close"
    assert rec["reason_expiry"] == EXPIRY.isoformat()
    assert EXPIRY.isoformat() in rec["diag_by_expiry"]


# ------------------------------------------------- real-data edge cases


def test_load_put_contracts_missing_file_is_a_counted_gap_not_a_raise(tmp_path):
    puts, diag = rv.load_put_contracts(tmp_path, "SPY", EXPIRY)
    assert puts is None            # data gap, reported, never fabricated
    assert diag["file"] is None


def test_load_put_contracts_skips_junk_and_unions_duplicates(tmp_path):
    lines = [
        '{"ticker": "O:SPY260619P00095000", "strike": 95, "bars": '
        '[{"t": "2026-05-15", "c": 1.0, "n": 40}]}',
        "{not json",                                            # bad_json
        '{"ticker": "O:SPY260619C00095000", "strike": 95, "bars": []}',  # call
        '{"ticker": "weird", "bars": []}',                      # no strike
        '{"ticker": "O:SPY260619P00090000", "strike": 90, "bars": []}',  # empty
        # zero close, missing close, unparsable t: each skipped, never filled in
        '{"ticker": "O:SPY260619P00085000", "strike": 85, "bars": '
        '[{"t": "2026-05-15", "c": 0}, {"t": "2026-05-18"}, '
        '{"t": "not-a-date", "c": 1.0}, {"t": "2026-05-19", "c": 0.5, "n": 9}]}',
        # duplicate strike row: bars are unioned, not dropped
        '{"ticker": "O:SPY260619P00095000", "strike": 95, "bars": '
        '[{"t": "2026-05-18", "c": 0.8, "n": 40}]}',
    ]
    (tmp_path / "contracts_SPY_2026-06-19.jsonl").write_text("\n".join(lines))
    puts, diag = rv.load_put_contracts(tmp_path, "SPY", EXPIRY)

    assert sorted(puts) == [85.0, 90.0, 95.0]
    assert diag["bad_json"] == 1
    assert diag["calls"] == 1
    assert diag["no_strike"] == 1
    assert diag["empty_bars"] == 1
    assert diag["bad_bars"] == 3           # zero close, no close, bad date
    assert diag["dup_strikes"] == 1
    assert sorted(puts[95.0]["bars"]) == [date(2026, 5, 15), date(2026, 5, 18)]
    assert sorted(puts[85.0]["bars"]) == [date(2026, 5, 19)]  # only usable bar


def test_replay_entry_missing_contract_file_is_data_not_gate():
    rec = rv.replay_entry(ENTRY, "SPY", [EXPIRY], {ENTRY: 100.0},
                          {EXPIRY: None}, 0.0049, Q)
    assert rec["qualified"] is False
    assert rec["reason"] == "no-contract-file"
    assert rec["reason_class"] == "data"
    assert rec["data_adequate"] is False    # excluded from the availability denominator


def test_replay_entry_all_contracts_empty_is_data_not_gate():
    puts = {95.0: {"ticker": "O:SPY260619P00095000", "bars": {}}}
    rec = rv.replay_entry(ENTRY, "SPY", [EXPIRY], {ENTRY: 100.0},
                          {EXPIRY: puts}, 0.0049, Q)
    assert rec["reason"] == "no-contract-bars"
    assert rec["reason_class"] == "data"
    assert rec["data_adequate"] is False


def test_managed_walk_skips_unusable_days_without_filling_them():
    credit = 1.4
    short = {date(2026, 5, 18): {"c": 0.0},        # unusable print
             date(2026, 5, 19): {"c": 1.30},
             date(2026, 5, 20): {"c": 0.80}}
    long_ = {date(2026, 5, 18): {"c": 0.15},
             date(2026, 5, 19): {"c": 0.20},
             date(2026, 5, 20): {"c": 0.15}}
    walk = rv.managed_walk(ENTRY, EXPIRY, credit, short, long_)
    assert walk["skipped_days"] == 1
    assert walk["walk_days"] == 3
    assert walk["exit_reason"] == "profit_take"
    assert walk["exit_date"] == date(2026, 5, 20)  # 5/18 skipped, not marked


def test_credit_floor_sensitivity_is_exact_at_and_below_the_floor_used():
    records = [
        {"qualified": True, "data_adequate": True, "best_credit_frac": 0.30},
        {"qualified": False, "data_adequate": True, "best_credit_frac": 0.22},
        {"qualified": False, "data_adequate": True, "best_credit_frac": 0.12},
        {"qualified": False, "data_adequate": False, "best_credit_frac": None},
    ]
    sens = rv.credit_floor_sensitivity(records, 0.25, (0.10, 0.20, 0.25, 0.30))
    assert "0.3" not in sens                  # never projects ABOVE the floor used
    assert sens["0.25"]["n_data_adequate"] == 3   # the data-gap row is excluded
    assert sens["0.25"]["n_qualified"] == 1
    assert sens["0.2"]["n_qualified"] == 2
    assert sens["0.1"]["n_qualified"] == 3
    assert sens["0.2"]["availability_rate"] == pytest.approx(2 / 3)


# --------------------------------------------- credit floor on the boundary


def test_credit_floor_accepts_a_credit_exactly_on_the_spec_boundary():
    # finding 10: the credit is a DIFFERENCE of two decimal closes, so an
    # entry that is exactly on the spec boundary need not compare equal to it.
    # 8.20 - 6.95 is 1.2499999999999991, not 1.25 — this is the real
    # 2026-01-02 QQQ 600/595 entry that the bare `>=` rejected.
    assert 8.20 - 6.95 < 1.25                    # the defect, in one line
    # Same defect on an in-band synthetic pair: 2.01 - 0.76 is
    # 1.2499999999999998, i.e. credit/width 0.24999999999999997 on a $5 width.
    assert 2.01 - 0.76 < 1.25
    chain = _chain(_put(94, 2.01), _put(89, 0.76))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q, credit_floor_frac=0.25)
    assert sel["qualified"] is True, sel["reason"]
    assert (sel["short_strike"], sel["long_strike"], sel["width"]) == (
        94.0, 89.0, 5.0)
    assert sel["credit"] == pytest.approx(1.25)
    assert sel["credit"] / sel["width"] == pytest.approx(0.25)
    # a credit genuinely below the floor is still rejected: tolerance is 1e-9,
    # not slack
    below = rv.select_vertical(ENTRY, EXPIRY, S,
                               _chain(_put(94, 2.01), _put(89, 0.77)), Q,
                               credit_floor_frac=0.25)
    assert (below["qualified"], below["reason"]) == (False, "credit-below-floor")
    assert below["diag"]["best_credit_frac"] == pytest.approx(1.24 / 5.0)
    assert rv.CREDIT_FLOOR_TOL == 1e-9


def test_credit_floor_sensitivity_uses_the_same_tolerance_as_the_gate():
    # finding 10: the projection must reproduce the gate's own count at the
    # floor the run used, so it needs the identical tolerant compare.
    records = [{"qualified": False, "data_adequate": True,
                "best_credit_frac": (8.20 - 6.95) / 5.0}]
    assert records[0]["best_credit_frac"] < 0.25          # 0.24999999999999983
    sens = rv.credit_floor_sensitivity(records, 0.25, (0.25,))
    assert sens["0.25"]["n_qualified"] == 1


# ------------------------------ availability denominators (numerator hygiene)


def test_availability_band_bracketed_numerator_matches_its_denominator():
    # finding 1: qualified is NOT a subset of band-bracketed. The bracket test
    # asks whether the fetched liquid invertible grid STRADDLES [0.20, 0.35];
    # an entry can qualify on an in-band strike while the grid stops short of
    # one edge. Dividing ALL qualified entries by the bracketed count is
    # arithmetically invalid and here would exceed 1.
    records = [
        # qualified, and the grid brackets the band
        {"qualified": True, "marks_present": True, "data_adequate": True,
         "brackets_delta_band": True},
        # qualified on an in-band strike whose grid does NOT bracket the band
        {"qualified": True, "marks_present": True, "data_adequate": True,
         "brackets_delta_band": False},
        {"qualified": True, "marks_present": True, "data_adequate": True,
         "brackets_delta_band": False},
        # rejected, grid brackets the band -> a real gate rejection
        {"qualified": False, "marks_present": True, "data_adequate": True,
         "brackets_delta_band": True},
        # a pure data gap
        {"qualified": False, "marks_present": False, "data_adequate": False,
         "brackets_delta_band": None},
    ]
    av = rv._availability(records)
    assert (av["n_entries"], av["n_qualified"]) == (5, 3)
    assert av["n_band_bracketed"] == 2
    assert av["n_qualified_band_bracketed"] == 1
    # the invalid arithmetic would have been 3/2 = 1.5
    assert av["availability_rate_band_bracketed"] == pytest.approx(0.5)
    assert av["availability_rate_band_bracketed"] <= 1.0
    assert av["passes_floor_on_band_bracketed"] is False
    # every published rate has its numerator inside its denominator
    assert av["n_qualified"] <= av["n_data_adequate"] <= av["n_entries"]
    assert av["n_qualified"] <= av["n_marks_present"]
    assert av["n_qualified_band_bracketed"] <= av["n_band_bracketed"]


def test_selection_diag_publishes_the_strike_grid_behind_the_delta_range():
    # finding 5: "n invertible strikes" alone cannot be read — a band-not-
    # fetched row needs how many rows were FETCHED and WHERE the surviving
    # strikes were. Here 3 strikes are fetched, only 2 survive the n>=30
    # screen, and none reaches the band.
    chain = _chain(_put(70, 0.05), _put(75, 0.08), _put(80, 0.12, n=3))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    diag = sel["diag"]
    assert sel["reason"] == "band-not-fetched"
    assert diag["n_contracts"] == 3
    assert diag["n_entry_mark"] == 3
    assert diag["n_liquid"] == 2
    assert diag["n_iv_ok"] == 2
    assert (diag["strike_min_fetched"], diag["strike_max_fetched"]) == (70.0, 80.0)
    # the invertible grid is the SUBSET the delta range was measured on
    assert (diag["strike_min_invertible"],
            diag["strike_max_invertible"]) == (70.0, 75.0)
    assert diag["delta_max"] < rv.DELTA_LO
    assert diag["brackets_delta_band"] is False
    # a chain with nothing invertible reports None, never a bogus range
    empty = rv.select_vertical(ENTRY, EXPIRY, S, {}, Q)
    assert empty["diag"]["strike_min_fetched"] is None
    assert empty["diag"]["strike_min_invertible"] is None


# ------------------------------------- raw-mark exposure under the smile mark


def _stamped(days: dict[date, str]) -> dict:
    """A contract whose bars carry apply_smile_repricing()'s provenance stamp.

    days maps day -> "repriced" | "raw".
    """
    bars = {}
    for d, kind in days.items():
        bar = {"c": 1.0, "n": 100}
        if kind == "raw":
            bar["smile_kept_raw"] = True
        else:
            bar["smile_repriced"] = True
        bars[d] = bar
    return {"ticker": "O:SPY260619P00097000", "bars": bars}


def test_raw_mark_exposure_detects_a_mixed_leg_pair():
    # finding 12: the report's cross-leg-consistency claim must be PROVABLE
    # from the artifact. A day on which one leg was curve-priced and the other
    # kept its raw close is the only way that claim can break, so the audit
    # has to detect it.
    d1, d2 = ENTRY, ENTRY + timedelta(days=1)
    puts = {97.0: _stamped({d1: "repriced", d2: "repriced"}),
            92.0: _stamped({d1: "repriced", d2: "raw"})}   # <- mixed on d2
    trade = {"entry_date": d1.isoformat(), "exit_date": d2.isoformat(),
             "symbol": "SPY", "expiry": EXPIRY.isoformat(),
             "short_strike": 97.0, "long_strike": 92.0}
    out = rv.raw_mark_exposure([trade], {("SPY", EXPIRY): puts})
    assert out["n_marked_pair_days"] == 2
    assert out["n_marked_pair_days_mixed"] == 1
    assert out["n_marked_pair_days_both_raw"] == 0
    assert out["n_trades_with_a_mixed_pair_day"] == 1
    assert out["n_trades_with_any_raw_marked_leg_day"] == 1
    assert out["no_mixed_pair_anywhere"] is False
    assert out["trades_with_mixed_pair"] == [{
        "entry_date": d1.isoformat(), "symbol": "SPY",
        "short_strike": 97.0, "long_strike": 92.0}]

    # a day where BOTH legs kept raw is raw+raw: one convention, NOT mixed
    both = {97.0: _stamped({d1: "repriced", d2: "raw"}),
            92.0: _stamped({d1: "repriced", d2: "raw"})}
    out2 = rv.raw_mark_exposure([trade], {("SPY", EXPIRY): both})
    assert out2["n_marked_pair_days_mixed"] == 0
    assert out2["n_marked_pair_days_both_raw"] == 1
    assert out2["n_trades_with_a_mixed_pair_day"] == 0
    assert out2["n_trades_with_any_raw_marked_leg_day"] == 1   # still flagged
    assert out2["no_mixed_pair_anywhere"] is True

    # a fully repriced trade is clean on both counters
    clean = {97.0: _stamped({d1: "repriced", d2: "repriced"}),
             92.0: _stamped({d1: "repriced", d2: "repriced"})}
    out3 = rv.raw_mark_exposure([trade], {("SPY", EXPIRY): clean})
    assert out3["n_trades_with_any_raw_marked_leg_day"] == 0
    assert out3["n_marked_leg_days_raw"] == 0

    # a missing chain is skipped, never a crash or a false clean bill
    assert rv.raw_mark_exposure([trade], {})["n_marked_pair_days"] == 0


def test_smile_repricing_stamps_provenance_and_splits_kept_raw_by_cause():
    # finding 12: bars_kept_raw alone cannot distinguish "the whole day had no
    # curve" (raw+raw, consistent) from "this strike extrapolated insanely"
    # (the only mixed-pair source).
    day = ENTRY
    thin = day + timedelta(days=1)          # too few strikes to fit
    closes = {day: S, thin: S}
    puts = {}
    raw = {}
    for k, px in ((90.0, 0.30), (92.0, 0.45), (94.0, 0.70),
                  (96.0, 1.10), (98.0, 1.75)):
        raw[k] = px
        puts[k] = {"ticker": f"O:SPY260619P{int(k * 1000):08d}",
                   "bars": {day: {"c": px, "n": 100},
                            thin: {"c": px, "n": 1}}}   # n=1 -> unfittable
    stats = rv.apply_smile_repricing(puts, EXPIRY, Q, closes)
    assert stats["days_fitted"] == 1
    assert stats["days_unfittable"] == 1
    # the thin day left all 5 strikes raw, and says so by cause
    assert stats["bars_kept_raw_on_unfittable_days"] == 5
    assert (stats["bars_kept_raw_on_unfittable_days"]
            + stats["bars_kept_raw_on_fitted_days"]) == stats["bars_kept_raw"]
    for k in puts:
        assert puts[k]["bars"][thin]["smile_kept_raw"] is True
        assert puts[k]["bars"][thin]["smile_kept_raw_cause"] == "day-unfittable"
        # kept raw means kept RAW: the close is untouched, never fabricated
        assert puts[k]["bars"][thin]["c"] == pytest.approx(raw[k])
        assert "smile_repriced" not in puts[k]["bars"][thin]
    # the fitted day's bars are stamped as repriced
    assert sum(1 for k in puts
               if puts[k]["bars"][day].get("smile_repriced")) == stats[
                   "bars_repriced"]
