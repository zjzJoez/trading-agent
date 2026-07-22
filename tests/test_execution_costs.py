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
    _write_calibration(tmp_path, {
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    })
    # width 5, credit 1.25, 1 contract:
    #   R      = (5 − 1.25) × 100          = $375
    #   fees   = 4 fills × $1              = $4
    #   spread = 2 legs × 2 dirs × (0.01/2 × 1.25 × 100) = 4 × 0.625 = $2.50
    #   friction_r = 6.50 / 375 = 0.017333…
    assert ec.friction_r("SPY", 5.0, 1.25) == pytest.approx(6.5 / 375.0, abs=1e-6)


def test_friction_r_global_fallback_default():
    # No calibration file: half-spread 4% of mark.
    #   spread = 4 × (0.04 × 1.25 × 100) = $20 ; fees $4 ; R $375
    assert ec.friction_r("IWM", 5.0, 1.25) == pytest.approx(24.0 / 375.0, abs=1e-6)
    assert ec.friction_r(None, 5.0, 1.25) == pytest.approx(24.0 / 375.0, abs=1e-6)


def test_friction_r_contracts_invariant(tmp_path):
    _write_calibration(tmp_path, {
        "per_underlying": {"QQQ": {"median_spread_pct_mid": 0.012}},
    })
    assert ec.friction_r("QQQ", 5.0, 1.25, contracts=1) == pytest.approx(
        ec.friction_r("QQQ", 5.0, 1.25, contracts=3), abs=1e-9)


def test_friction_r_validates_inputs():
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 0.0)          # no credit
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 5.0)          # credit >= width (no risk)
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, -1.0)         # negative credit
    with pytest.raises(ValueError):
        ec.friction_r("SPY", 5.0, 1.25, contracts=0)
