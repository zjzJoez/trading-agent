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
