"""Unit tests for the Phase 0 execution-cost model."""
from __future__ import annotations

import json

import pytest

from trading_agent import execution_costs as ec


@pytest.fixture(autouse=True)
def _fresh_calibration(tmp_path, monkeypatch):
    """Hermetic defaults: point data_dir at an empty tmp dir so a real
    data/execution_costs.json (written by the calibration script) can't
    leak into tests that assert the built-in default constants."""
    import dataclasses

    from trading_agent import config as config_mod
    new_cfg = dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    ec.reset_calibration_cache()
    yield
    ec.reset_calibration_cache()


def test_option_fees_per_side_default():
    # 4 contracts × $1.00/contract/side
    assert ec.fees_per_side(4, "OPT") == pytest.approx(4.0)


def test_stock_fees_per_side_default():
    assert ec.fees_per_side(100, "STK") == pytest.approx(0.5)


def test_half_spread_uses_live_quote_when_present():
    # bid 1.90 / ask 2.10 → half-spread 0.10 × 2 contracts × 100 mult = $20
    cost = ec.half_spread_cost(2.00, 2, "OPT", bid=1.90, ask=2.10)
    assert cost == pytest.approx(20.0)


def test_half_spread_falls_back_to_pct_of_mark():
    # No quote → 4% of mark: 2.00 × 0.04 × 1 × 100 = $8
    cost = ec.half_spread_cost(2.00, 1, "OPT")
    assert cost == pytest.approx(8.0)


def test_half_spread_ignores_crossed_quote():
    # ask < bid is garbage — must fall back to pct-of-mark, not go negative
    cost = ec.half_spread_cost(2.00, 1, "OPT", bid=2.10, ask=1.90)
    assert cost == pytest.approx(8.0)


def test_net_pnl_dealt_prices_charge_only_fees():
    # Both sides dealt: gross (3.00-2.00)×2×100 = $200, fees 2×$2 = $4
    net, costs = ec.net_pnl(
        200.0, 2, "OPT",
        entry_price=2.0, exit_price=3.0,
        entry_is_dealt=True, exit_is_dealt=True,
    )
    assert net == pytest.approx(196.0)
    assert costs.entry_spread_cost == 0.0
    assert costs.exit_spread_cost == 0.0


def test_net_pnl_mark_exit_charges_half_spread():
    net, costs = ec.net_pnl(
        200.0, 2, "OPT",
        entry_price=2.0, exit_price=3.0,
        entry_is_dealt=True, exit_is_dealt=False,
        exit_bid=2.90, exit_ask=3.10,
    )
    # half-spread 0.10 × 2 × 100 = $20, fees $4
    assert costs.exit_spread_cost == pytest.approx(20.0)
    assert net == pytest.approx(176.0)


def test_round_trip_fee_per_unit_option():
    # $1/contract/side → $2 round trip → 0.02 price points
    assert ec.round_trip_fee_per_unit("OPT") == pytest.approx(0.02)


def test_calibration_file_overrides_defaults(tmp_path, monkeypatch):
    import dataclasses

    from trading_agent import config as config_mod
    (tmp_path / "execution_costs.json").write_text(
        json.dumps({"opt_half_spread_pct_of_mark": 0.10}))
    new_cfg = dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    ec.reset_calibration_cache()
    # 10% of mark now: 2.00 × 0.10 × 1 × 100 = $20
    assert ec.half_spread_cost(2.00, 1, "OPT") == pytest.approx(20.0)
    # Un-overridden keys keep their defaults
    assert ec.fees_per_side(1, "OPT") == pytest.approx(1.0)


def test_negative_calibration_values_rejected(tmp_path, monkeypatch):
    import dataclasses

    from trading_agent import config as config_mod
    (tmp_path / "execution_costs.json").write_text(
        json.dumps({"opt_fee_per_contract_per_side": -5}))
    new_cfg = dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    ec.reset_calibration_cache()
    assert ec.fees_per_side(1, "OPT") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# M1-0.3 — per-underlying calibration + vertical friction_r
# ---------------------------------------------------------------------------


def _write_calibration(tmp_path, payload: dict) -> None:
    (tmp_path / "execution_costs.json").write_text(json.dumps(payload))
    ec.reset_calibration_cache()


def test_per_underlying_persistence_round_trip(tmp_path):
    _write_calibration(tmp_path, {
        "opt_half_spread_pct_of_mark": 0.04,
        "per_underlying": {
            "SPY": {"median_spread_pct_mid": 0.01, "n_quotes": 62,
                    "calibrated_at": "2026-07-22T00:00:00+00:00",
                    "zone": "puts 30-45 DTE"},
        },
    })
    # SPY: full spread 1% of mid → half 0.5%: 2.00 × 0.005 × 1 × 100 = $1
    assert ec.half_spread_cost(2.00, 1, "OPT", underlying="SPY") == pytest.approx(1.0)
    # 'US.SPY' and lowercase normalize to the same entry
    assert ec.half_spread_cost(2.00, 1, "OPT", underlying="US.SPY") == pytest.approx(1.0)
    assert ec.half_spread_cost(2.00, 1, "OPT", underlying="spy") == pytest.approx(1.0)


def test_per_underlying_fallback_semantics(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {
            "SPY": {"median_spread_pct_mid": 0.01},
        },
    })
    # Uncalibrated name → global default 4% of mark: 2.00 × 0.04 × 100 = $8
    assert ec.half_spread_cost(2.00, 1, "OPT", underlying="TSLA") == pytest.approx(8.0)
    # No underlying at all → unchanged legacy behavior
    assert ec.half_spread_cost(2.00, 1, "OPT") == pytest.approx(8.0)
    # Live quote still beats the per-underlying figure
    assert ec.half_spread_cost(
        2.00, 1, "OPT", bid=1.90, ask=2.10, underlying="SPY"
    ) == pytest.approx(10.0)
    # Stocks never consult per_underlying (10 bps half-spread default)
    assert ec.half_spread_cost(
        100.0, 10, "STK", underlying="SPY") == pytest.approx(0.5)


def test_per_underlying_malformed_entries_ignored(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {
            "SPY": {"median_spread_pct_mid": 0},        # not > 0
            "QQQ": {"median_spread_pct_mid": -0.01},    # negative
            "IWM": {"median_spread_pct_mid": 7.3},      # percentage, not frac
            "DIA": "garbage",                            # not a dict
        },
    })
    for u in ("SPY", "QQQ", "IWM", "DIA"):
        assert ec.half_spread_cost(2.00, 1, "OPT", underlying=u) == pytest.approx(8.0)


def test_net_pnl_threads_underlying_to_spread(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    })
    _, costs = ec.net_pnl(
        200.0, 2, "OPT",
        entry_price=2.0, exit_price=3.0,
        entry_is_dealt=True, exit_is_dealt=False,
        underlying="SPY",
    )
    # exit mark 3.00 × half-spread 0.5% × 2 × 100 = $3
    assert costs.exit_spread_cost == pytest.approx(3.0)


def test_friction_r_hand_computed_per_underlying(tmp_path):
    """2026-07-25 friction-truth fix — this expectation CHANGED deliberately.

    It used to pin 6.50/375 = 0.01733R, which came from proxying BOTH leg
    marks by the net credit (leg-mid-sum = 2 × 1.25 = 2.50). Real quotes
    measure leg-mid-sum/credit 6.43 on a $5-wide, so the old number
    understated the spread bill ~3.2×. With no leg marks and no calibrated
    wing ratio the model falls back to the MEASURED p75 for THIS width
    (MEASURED_WING_RATIO_P75[5.0] = 0.734 — an upper quantile, because the
    bill is strictly increasing in r):
      legs   = 1.25/(1−0.734) = 4.6992 short, × 0.734 = 3.4492 long
      spread = 2 dirs × (0.01/2 × 4.6992 + 0.01/2 × 3.4492) × 100 = $8.148
      fees   = 4 fills × $1 = $4 ; R = (5 − 1.25) × 100 = $375
      friction_r = 12.148 / 375 = 0.032396…
    """
    _write_calibration(tmp_path, {
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    })
    short, long = ec.leg_marks_from_wing_ratio(
        1.25, ec.MEASURED_WING_RATIO_P75[5.0])
    expected = (4.0 + 2.0 * (0.005 * short + 0.005 * long) * 100.0) / 375.0
    assert ec.friction_r("SPY", 5.0, 1.25) == pytest.approx(expected, abs=1e-6)
    assert ec.friction_r("SPY", 5.0, 1.25) == pytest.approx(0.032396, abs=1e-6)
    # the old net_credit-proxy answer must not survive anywhere
    assert ec.friction_r("SPY", 5.0, 1.25) != pytest.approx(6.5 / 375.0, abs=1e-4)


def test_friction_r_global_fallback_default():
    """No calibration file: half-spread 4% of mark, wing ratio the measured
    p75 for this width. CHANGED 2026-07-25 from 24.0/375 = 0.064R — that
    figure charged both legs at the 1.25 credit; the honest legsum is ~8.1,
    not 2.50."""
    short, long = ec.leg_marks_from_wing_ratio(
        1.25, ec.MEASURED_WING_RATIO_P75[5.0])
    expected = (4.0 + 2.0 * 0.04 * (short + long) * 100.0) / 375.0
    assert ec.friction_r("IWM", 5.0, 1.25) == pytest.approx(expected, abs=1e-6)
    assert ec.friction_r(None, 5.0, 1.25) == pytest.approx(expected, abs=1e-6)
    # Fail-conservative: the (width, credit)-only fallback must OVERSTATE, so
    # it lands well above the old understated number, never below it.
    assert ec.friction_r("IWM", 5.0, 1.25) > 24.0 / 375.0


def test_friction_r_explicit_leg_marks_are_exact(tmp_path):
    """The live sizing path has both leg prices — no wing-ratio guess at all.
    IWM measured case: credit 0.955 on a $5 width, legs 3.550/2.595, zone
    spreads 4.24% short / 5.04% wing → ~0.079R (the number that FAILS the
    plan's 0.06R bar, vs the 0.031R the old model reported)."""
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
            "wing_ratio": 0.731,
        }},
    })
    got = ec.friction_r("IWM", 5.0, 0.955, short_mark=3.550, long_mark=2.595)
    spread = 2.0 * (0.0212 * 3.550 + 0.0252 * 2.595) * 100.0
    assert got == pytest.approx((4.0 + spread) / 404.5, abs=1e-6)
    assert got == pytest.approx(0.0794, abs=5e-4)
    assert got > 0.06                      # IWM fails the whitelist bar


def test_friction_r_wing_ratio_matches_the_algebra(tmp_path):
    """legsum = cr(1+r)/(1−r); r = 1/3 is the ONLY ratio where the old
    net_credit proxy (legsum = 2·cr) was right."""
    _write_calibration(tmp_path, {
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    })
    cr = 1.25
    # An EXPLICIT ratio is trusted as given (not floored at MIN_WING_RATIO) —
    # it is a caller asserting a known structure, not untrusted file data.
    for r in (0.01, 1.0 / 3.0, 0.53, 0.731):
        legsum = cr * (1.0 + r) / (1.0 - r)
        expected = (4.0 + 2.0 * 0.005 * legsum * 100.0) / 375.0
        assert ec.friction_r("SPY", 5.0, cr, wing_ratio=r) == pytest.approx(
            expected, abs=1e-6), r
    # r = 0 is NOT admissible: it claims the wing is free, zeroing its half of
    # the spread bill. It degrades to the measured fallback instead.
    assert ec.friction_r("SPY", 5.0, cr, wing_ratio=0.0) == pytest.approx(
        ec.friction_r("SPY", 5.0, cr), abs=1e-9)
    # r = 1/3 ⇔ legsum == 2 × credit: the falsified premise, pinned as algebra
    assert ec.leg_marks_from_wing_ratio(cr, 1.0 / 3.0) == pytest.approx(
        (1.875, 0.625))
    assert sum(ec.leg_marks_from_wing_ratio(cr, 1.0 / 3.0)) == pytest.approx(
        2.0 * cr)
    # monotone in r: a richer wing can only cost more
    assert (ec.friction_r("SPY", 5.0, cr, wing_ratio=0.8)
            > ec.friction_r("SPY", 5.0, cr, wing_ratio=0.731)
            > ec.friction_r("SPY", 5.0, cr, wing_ratio=1.0 / 3.0))


def test_friction_r_prefers_calibrated_wing_ratio_over_the_default(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {"median_spread_pct_mid": 0.04,
                                   "wing_ratio": 0.530}},
    })
    assert ec.friction_r("IWM", 10.0, 1.76) == pytest.approx(
        ec.friction_r("IWM", 10.0, 1.76, wing_ratio=0.530), abs=1e-9)
    # an uncalibrated name gets the MEASURED p75 for that width, not r = 1/3
    # and not the $5 figure (r is strongly width-dependent)
    assert ec.friction_r("DIA", 10.0, 1.76) == pytest.approx(
        ec.friction_r("DIA", 10.0, 1.76,
                      wing_ratio=ec.MEASURED_WING_RATIO_P75[10.0]),
        abs=1e-9)
    assert ec.friction_r("DIA", 10.0, 1.76) < ec.friction_r(
        "DIA", 10.0, 1.76, wing_ratio=ec.MEASURED_WING_RATIO_P75[5.0])


def test_calibrated_wing_ratio_is_width_aware_and_p75_preferring(tmp_path):
    """MEASURED_WING_RATIO used to be dead code and friction_r ignored ``width``
    when picking r, even though r moves 0.734 -> 0.536 between $5 and $10.

    The by_width block wins at the exact width; a traded width with no exact
    match takes the MAX over NARROWER measured widths (narrower => larger r =>
    overstated bill); a width narrower than anything measured falls through to
    the name-level scalar. p75 always beats the median within a block.
    """
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.04,
            "wing_ratio": 0.60,                     # name-level scalar
            "by_width": {
                "5": {"wing_ratio": 0.7308, "wing_ratio_p75": 0.734},
                "10": {"wing_ratio": 0.5305, "wing_ratio_p75": 0.536},
            },
        }},
    })
    for width, r in ((5.0, 0.734), (10.0, 0.536)):
        assert ec.friction_r("IWM", width, 1.25) == pytest.approx(
            ec.friction_r("IWM", width, 1.25, wing_ratio=r), abs=1e-9), width
    # $7.50 traded: no exact block, so the $5 (larger r) figure is used
    assert ec.friction_r("IWM", 7.5, 1.25) == pytest.approx(
        ec.friction_r("IWM", 7.5, 1.25, wing_ratio=0.734), abs=1e-9)
    # $2.50 traded: nothing measured at or below it → name-level scalar
    assert ec.friction_r("IWM", 2.5, 1.25) == pytest.approx(
        ec.friction_r("IWM", 2.5, 1.25, wing_ratio=0.60), abs=1e-9)
    # a block carrying ONLY a median is still honoured
    _write_calibration(tmp_path, {
        "per_underlying": {"QQQ": {"median_spread_pct_mid": 0.01,
                                   "by_width": {"5": {"wing_ratio": 0.62}}}},
    })
    assert ec.friction_r("QQQ", 5.0, 1.25) == pytest.approx(
        ec.friction_r("QQQ", 5.0, 1.25, wing_ratio=0.62), abs=1e-9)


def test_zero_or_tiny_file_wing_ratio_cannot_delete_the_wing_spread_bill(
        tmp_path):
    """The verified hole: wing_ratio 0 in the file passed ``0 <= r < 1`` and,
    because file values beat the measured default, priced the IWM $5 credit
    0.955 case at 0.0207R against an honest 0.0794R — a 74% discount on the
    modeled bill from one bad calibration write.

    Now: 0 (and any non-positive value) fails validation outright, so the
    resolver falls through to the width-aware measured p75 and reproduces the
    honest ~0.079R. A surviving but implausible positive value is floored at
    MIN_WING_RATIO — the smallest ratio ever measured on a real chain.

    The floor is width-BLIND by design (see MIN_WING_RATIO): it is a
    credibility bound derived from IWM alone, so it must not overwrite a
    legitimately calibrated figure for another name. RESIDUAL, stated openly: a
    corrupt-but-positive IWM $5 ratio of 0.01 therefore prices at 0.530, which
    is 0.045R and would pass the 0.06R bar. That is why the STRICT 0 < r < 1
    validator, not the floor, is what closes the verified hole.
    """
    zone = {"median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504}
    honest = 0.0794            # the leg-marks answer for this structure
    # non-positive → dropped by the validator → measured p75 for the width
    for bad in (0.0, -0.0, -0.2, "0.7", None):
        _write_calibration(tmp_path, {
            "per_underlying": {"IWM": {**zone, "wing_ratio": bad}}})
        got = ec.friction_r("IWM", 5.0, 0.955)
        assert got == pytest.approx(
            ec.friction_r("IWM", 5.0, 0.955,
                          wing_ratio=ec.MEASURED_WING_RATIO_P75[5.0]),
            abs=1e-9), bad
        assert got > 0.06, bad                  # still fails the whitelist bar
        assert got == pytest.approx(honest, abs=2e-3), bad
        # the falsified 0.0207R discount is unreachable
        assert got > 3 * 0.0207, bad
    # implausible-but-positive → floored at the measured minimum, never used raw
    for bad in (0.01, 0.2, 0.5299):
        _write_calibration(tmp_path, {
            "per_underlying": {"IWM": {**zone, "wing_ratio": bad}}})
        assert ec.friction_r("IWM", 5.0, 0.955) == pytest.approx(
            ec.friction_r("IWM", 5.0, 0.955, wing_ratio=ec.MIN_WING_RATIO),
            abs=1e-9), bad
        assert ec.friction_r("IWM", 5.0, 0.955) > 2 * 0.0207, bad
    # the floor is never applied to a plausible file value
    for good in (0.53, 0.60, 0.90):
        _write_calibration(tmp_path, {
            "per_underlying": {"IWM": {**zone, "wing_ratio": good}}})
        assert ec.friction_r("IWM", 5.0, 0.955) == pytest.approx(
            ec.friction_r("IWM", 5.0, 0.955, wing_ratio=good), abs=1e-9), good


def test_friction_r_degrades_instead_of_raising_on_garbage_marks(tmp_path):
    """A cost model must never abort a trade-decision path. Unusable marks or
    an out-of-range wing ratio fall back to the conservative default."""
    _write_calibration(tmp_path, {
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    })
    baseline = ec.friction_r("SPY", 5.0, 1.25)
    for kwargs in (
        {"short_mark": 1.0, "long_mark": 2.0},      # long > short
        {"short_mark": 0.0, "long_mark": 0.0},      # no marks
        {"short_mark": 3.0, "long_mark": None},     # only one leg
        {"wing_ratio": 1.0},                        # r >= 1 → legsum diverges
        {"wing_ratio": -0.2},                       # negative
    ):
        assert ec.friction_r("SPY", 5.0, 1.25, **kwargs) == pytest.approx(
            baseline, abs=1e-9), kwargs
    with pytest.raises(ValueError):
        ec.leg_marks_from_wing_ratio(1.25, 1.0)


def test_measured_wing_ratio_constants_are_the_measured_ones():
    """Provenance guard: these are MEASURED figures (option_chain_snapshots,
    IWM PUTS 30-37 DTE, short |delta| 0.20-0.35, 3 distinct (day,expiry)
    chains per width — 5-wide n=32, 10-wide n=25), not a textbook assumption.
    r = 1/3 must appear nowhere as a default."""
    assert ec.MEASURED_WING_RATIO[5.0] == 0.731        # medians (provenance)
    assert ec.MEASURED_WING_RATIO[10.0] == 0.530
    assert ec.MEASURED_WING_RATIO_P75[5.0] == 0.734    # consumed by the model
    assert ec.MEASURED_WING_RATIO_P75[10.0] == 0.536
    assert ec.MIN_WING_RATIO == min(ec.MEASURED_WING_RATIO.values())
    assert abs(ec.DEFAULT_WING_RATIO - 1.0 / 3.0) > 0.3


def test_the_blind_default_is_an_upper_quantile_not_a_median():
    """THE 2026-07-25 CRITICAL, pinned as arithmetic.

    The modeled spread bill legsum(r) = cr(1+r)/(1-r) has derivative
    2cr/(1-r)^2 > 0 on [0, 1), so it is STRICTLY INCREASING in r. A median r
    therefore understates friction for every pair above it — the exact
    opposite of fail-conservative. The blind fallback must be an UPPER
    quantile of the measured distribution, per width.
    """
    cr = 1.25
    # monotonicity, checked numerically across the admissible range
    prev = None
    for i in range(0, 99):
        r = i / 100.0
        legsum = sum(ec.leg_marks_from_wing_ratio(cr, r))
        assert prev is None or legsum > prev, r
        prev = legsum
    # every p75 sits at or above its own median, and the default is the max p75
    for w, med in ec.MEASURED_WING_RATIO.items():
        assert ec.MEASURED_WING_RATIO_P75[w] >= med, w
    assert ec.DEFAULT_WING_RATIO == max(ec.MEASURED_WING_RATIO_P75.values())
    # ... so blind pricing is strictly MORE expensive than the old median
    assert (ec.friction_r(None, 5.0, cr)
            > ec.friction_r(None, 5.0, cr,
                            wing_ratio=ec.MEASURED_WING_RATIO[5.0]))


def test_friction_r_contracts_invariant(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"QQQ": {"median_spread_pct_mid": 0.012}},
    })
    # abs=1e-6, not 1e-9: half_spread_cost rounds its dollar figure to 4dp, so
    # the per-contract rounding residue does not cancel exactly across
    # contracts. friction_r itself only reports 6dp.
    assert ec.friction_r("QQQ", 5.0, 1.25, contracts=1) == pytest.approx(
        ec.friction_r("QQQ", 5.0, 1.25, contracts=3), abs=1e-6)


def test_friction_r_validates_inputs():
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 0.0)          # no credit
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 5.0)          # credit >= width (no risk)
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, -1.0)         # negative credit
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 1.25, contracts=0)


# ---------------------------------------------------------------------------
# Per-zone spreads (the two legs of a vertical do not quote alike)
# ---------------------------------------------------------------------------


def test_half_spread_cost_zone_preference(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
        }},
    })
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                               zone="short") == pytest.approx(2.0 * 0.0212 * 100)
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                               zone="wing") == pytest.approx(2.0 * 0.0252 * 100)
    # unknown / absent zone → the name's pooled median
    for z in (None, "sideways"):
        assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                                   zone=z) == pytest.approx(2.0 * 0.023 * 100)
    # live bid/ask still beats every calibrated figure
    assert ec.half_spread_cost(2.0, 1, "OPT", bid=1.9, ask=2.1,
                               underlying="IWM", zone="wing") == pytest.approx(10.0)


def test_zone_falls_back_through_pooled_then_global(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {"median_spread_pct_mid": 0.046},
                           "SPY": {"wing_ratio": 0.6}},   # no spread at all
    })
    # IWM has no zone figures → pooled median for both zones
    for z in ("short", "wing"):
        assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                                   zone=z) == pytest.approx(2.0 * 0.023 * 100)
    # SPY carries only a wing ratio → global 4%-of-mark default for the spread
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="SPY",
                               zone="short") == pytest.approx(8.0)
    # ... but the ratio is still honoured by friction_r
    assert ec.friction_r("SPY", 5.0, 1.25) == pytest.approx(
        ec.friction_r("SPY", 5.0, 1.25, wing_ratio=0.6), abs=1e-9)


def test_malformed_zone_and_ratio_fields_are_dropped(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 4.24,      # percent, not a fraction
            "wing_zone_spread_pct": -0.05,      # negative
            "wing_ratio": 1.4,                  # r >= 1 → not a credit vertical
        }},
    })
    for z in ("short", "wing"):
        assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                                   zone=z) == pytest.approx(2.0 * 0.023 * 100)
    assert ec.friction_r("IWM", 5.0, 1.25) == pytest.approx(
        ec.friction_r("IWM", 5.0, 1.25, wing_ratio=ec.DEFAULT_WING_RATIO),
        abs=1e-9)


# ---------------------------------------------------------------------------
# combo_friction_r — the live wiring
# ---------------------------------------------------------------------------


def test_combo_friction_r_uses_the_real_leg_marks(tmp_path):
    from trading_agent.sizing import ComboLeg, ProposedCombo
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
            "wing_ratio": 0.731,
        }},
    })
    combo = ProposedCombo(
        ticker="IWM",
        legs=(
            ComboLeg(option_symbol="US.IWM260828P220000", side="SELL",
                     contracts=1, price=3.550, right="PUT", strike=220.0,
                     dte=37, delta=-0.28),
            ComboLeg(option_symbol="US.IWM260828P215000", side="BUY",
                     contracts=1, price=2.595, right="PUT", strike=215.0,
                     dte=37, delta=-0.21),
        ),
    )
    assert combo.width == 5.0
    assert combo.net_credit == pytest.approx(0.955)
    got = ec.combo_friction_r(combo)
    assert got == pytest.approx(
        ec.friction_r("IWM", 5.0, 0.955, short_mark=3.550, long_mark=2.595),
        abs=1e-9)
    assert got == pytest.approx(0.0794, abs=5e-4)
    # ... and strictly worse than the falsified r = 1/3 proxy it replaced
    assert got > ec.friction_r("IWM", 5.0, 0.955, wing_ratio=1.0 / 3.0)


def test_combo_friction_r_returns_none_for_unpriceable_structures():
    from trading_agent.sizing import ComboLeg, ProposedCombo

    def leg(side, price, strike):
        return ComboLeg(option_symbol=f"US.IWM260828P{int(strike) * 1000}",
                        side=side, contracts=1, price=price, right="PUT",
                        strike=strike, dte=37, delta=-0.25)

    # debit (long mark above short) → not a credit vertical
    debit = ProposedCombo(ticker="IWM", legs=(leg("SELL", 1.0, 220.0),
                                             leg("BUY", 2.0, 215.0)))
    assert ec.combo_friction_r(debit) is None
    # same strike → zero width, no defined risk
    flat = ProposedCombo(ticker="IWM", legs=(leg("SELL", 2.0, 220.0),
                                            leg("BUY", 1.0, 220.0)))
    assert ec.combo_friction_r(flat) is None
    # two shorts → no long leg to cap it
    naked = ProposedCombo(ticker="IWM", legs=(leg("SELL", 2.0, 220.0),
                                              leg("SELL", 1.0, 215.0)))
    assert ec.combo_friction_r(naked) is None
    assert ec.combo_friction_r(None) is None
    assert ec.combo_friction_r(object()) is None


def test_combo_friction_r_prices_off_the_real_leg_touches(tmp_path):
    """FIX 2026-07-25: the live guard holds fresh per-leg bid/ask, so friction
    must be charged at the ACTUAL spread the trade will cross, not at a
    percent-of-mid median measured on other days.

    IWM $5-wide, legs 3.550/2.595 (credit 0.955, R = $404.50). The calibration
    says 4.24% short / 5.04% wing of mid. Give the short leg a much WIDER real
    touch ($0.30 wide = 8.5% of mid) and the modeled bill must rise with it.
    """
    from trading_agent.sizing import ComboLeg, ProposedCombo
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
            "wing_ratio": 0.731,
        }},
    })
    combo = ProposedCombo(
        ticker="IWM",
        legs=(
            ComboLeg(option_symbol="US.IWM260828P220000", side="SELL",
                     contracts=1, price=3.550, right="PUT", strike=220.0,
                     dte=37, delta=-0.28),
            ComboLeg(option_symbol="US.IWM260828P215000", side="BUY",
                     contracts=1, price=2.595, right="PUT", strike=215.0,
                     dte=37, delta=-0.21),
        ),
    )
    calibrated = ec.combo_friction_r(combo)
    quotes = {"short": {"bid": 3.40, "ask": 3.70},      # $0.30 real spread
              "long": {"bid": 2.53, "ask": 2.66}}       # $0.13 real spread
    got = ec.combo_friction_r(combo, leg_quotes=quotes)
    # hand arithmetic: half-spreads 0.15 and 0.065, both directions, ×100
    expected = (4.0 + 2.0 * (0.15 + 0.065) * 100.0) / 404.5
    assert got == pytest.approx(expected, abs=1e-6)
    assert got > calibrated                     # the real touch is wider
    assert got == pytest.approx(
        ec.friction_r("IWM", 5.0, 0.955, short_mark=3.550, long_mark=2.595,
                      short_bid=3.40, short_ask=3.70,
                      long_bid=2.53, long_ask=2.66), abs=1e-9)
    # a TIGHTER real market is charged as tighter — the point is truth, not
    # a one-way ratchet
    tight = ec.combo_friction_r(combo, leg_quotes={
        "short": {"bid": 3.54, "ask": 3.56}, "long": {"bid": 2.59, "ask": 2.60}})
    assert tight < calibrated


def test_combo_friction_r_falls_back_per_leg_on_a_missing_quote(tmp_path):
    """One unquotable leg must not discard the other leg's real touch."""
    from trading_agent.sizing import ComboLeg, ProposedCombo
    _write_calibration(tmp_path, {
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
            "wing_ratio": 0.731,
        }},
    })
    combo = ProposedCombo(
        ticker="IWM",
        legs=(
            ComboLeg(option_symbol="US.IWM260828P220000", side="SELL",
                     contracts=1, price=3.550, right="PUT", strike=220.0,
                     dte=37, delta=-0.28),
            ComboLeg(option_symbol="US.IWM260828P215000", side="BUY",
                     contracts=1, price=2.595, right="PUT", strike=215.0,
                     dte=37, delta=-0.21),
        ),
    )
    short_only = ec.combo_friction_r(
        combo, leg_quotes={"short": {"bid": 3.40, "ask": 3.70}})
    # short leg at the real touch (0.15), wing at its calibrated 5.04%/2
    expected = (4.0 + 2.0 * (0.15 + 0.0252 * 2.595) * 100.0) / 404.5
    assert short_only == pytest.approx(expected, abs=1e-6)
    # every degenerate quote shape degrades to the pure-calibration answer
    calibrated = ec.combo_friction_r(combo)
    for bad in ({}, None, {"short": None}, {"short": {}},
                {"short": {"bid": 0, "ask": 3.7}},
                {"short": {"bid": "x", "ask": "y"}},
                {"short": {"bid": 3.4}},               # ask missing
                {"wrong_role": {"bid": 3.4, "ask": 3.7}}):
        assert ec.combo_friction_r(combo, leg_quotes=bad) == pytest.approx(
            calibrated, abs=1e-9), bad
