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
    return strike, {"ticker": f"O:SPY260619P{int(strike * 1000):08d}",
                    "bars": {day: {"c": close, "n": n}}}


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


def test_select_vertical_no_strike_in_band():
    chain = _chain(_put(85, 0.0414), _put(80, 0.0035))  # |delta| ~ .015/.002
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel == {"qualified": False, "reason": "no-strike-in-band"}


def test_select_vertical_short_leg_liquidity_gate_n30():
    strike, contract = _put(96, 1.3481, n=10)  # in band but too few trades
    sel = rv.select_vertical(ENTRY, EXPIRY, S, {strike: contract}, Q)
    assert sel == {"qualified": False, "reason": "no-strike-in-band"}


def test_select_vertical_wing_missing():
    chain = _chain(_put(96, 1.3481))  # in band, but no 91 and no 86 contract
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel == {"qualified": False, "reason": "wing-missing"}


def test_select_vertical_credit_below_floor():
    # Wing bar exists but credit 1.3481-0.60 = 0.748 < 1.25.
    chain = _chain(_put(96, 1.3481), _put(91, 0.60))
    sel = rv.select_vertical(ENTRY, EXPIRY, S, chain, Q)
    assert sel == {"qualified": False, "reason": "credit-below-floor"}


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


def test_max_drawdown_r():
    entries = _mk_entries({
        date(2026, 4, 6): 0.15, date(2026, 4, 7): -1.0,
        date(2026, 4, 8): 0.15, date(2026, 4, 9): -0.5,
    })
    # path: .15, -.85, -.70, -1.20 ; peak .15 -> trough -1.20 => dd 1.35
    assert rv.max_drawdown_r(entries) == pytest.approx(1.35)
    assert rv.max_drawdown_r([]) == 0.0


# -------------------------------------------------------------- CLI / e2e


def test_parse_spread_pct():
    assert rv.parse_spread_pct("SPY=0.0049,QQQ=0.0094") == {
        "SPY": 0.0049, "QQQ": 0.0094}
    with pytest.raises(ValueError):
        rv.parse_spread_pct("SPY")


def _write_e2e_data(data_dir):
    days = [ENTRY, date(2026, 5, 18)]
    (data_dir / "underlying_SPY.json").write_text(json.dumps(
        {"results": [{"t": d.isoformat(), "c": 100.0} for d in days]}))

    def bars(closes):
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

    summary = rv.run(data_dir, data_dir / "batch_plan.json", out_dir,
                     {"SPY": 0.0049}, resamples=500, seed=42)

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
    assert avail["overall"]["rate"] == pytest.approx(0.5)
    assert avail["per_symbol"]["SPY"]["rate"] == pytest.approx(0.5)
    assert avail["per_month"]["2026-05"]["rate"] == pytest.approx(0.5)
    assert avail["unqualified_reasons"] == {"no-underlying-bar": 1}
    mp = summary["managed_payoff"]
    assert mp["n_trades"] == 1
    assert mp["win_rate"] == 1.0
    assert mp["mean_r"] == pytest.approx(ok["result_r"])
    assert mp["lb95_mean_r"] == pytest.approx(ok["result_r"])  # single block
    assert summary["exit_degradation"]["counts"] == {
        "profit_take": 1, "dte_21": 0, "data_end": 0}
    # The European-BS approximation must be disclosed in the report.
    assert any("EUROPEAN" in note for note in summary["approximation_notes"])
