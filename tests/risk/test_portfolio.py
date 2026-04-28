"""Synthetic-portfolio tests for portfolio.build_snapshot.

Pure-Python: no DB, no moomoo. Verifies aggregate Greeks, factor exposures,
correlation matrix, heat math.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from trading_agent.risk.portfolio import (
    OpenPosition,
    PortfolioSnapshot,
    build_snapshot,
)


def _opt(symbol="US.SPY260530C00720000", underlying="US.SPY", **kw) -> OpenPosition:
    base = dict(
        symbol=symbol,
        underlying=underlying,
        asset_type="OPT",
        side="BUY",
        qty=10.0,
        entry_price=1.50,
        mark=1.50,
        stop=None,
        target=None,
        delta=0.40,
        gamma=0.02,
        vega=0.10,
        theta=-0.05,
        iv=0.18,
        notional=1500.0,
        unrealized_pnl=0.0,
        thesis_id=None,
        sector="Information Technology",
        strategy_label="long_call",
        age_minutes=10.0,
    )
    base.update(kw)
    return OpenPosition(**base)


def test_empty_portfolio_returns_zero_state():
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[])
    assert snap.n_open == 0
    assert snap.heat_pct == 0.0
    assert snap.aggregate_greeks["net_delta"] == 0.0
    assert snap.factor_exposures == {}
    assert snap.data_quality["degradation_level"] == 0


def test_aggregate_greeks_long_call():
    """One long 10x SPY call delta=0.4 → net_delta = 0.4*10*100 = 400 (per-share)."""
    p = _opt(qty=10, delta=0.4, gamma=0.02, vega=0.10)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])

    g = snap.aggregate_greeks
    assert g["net_delta"] == 400.0  # 0.4 × 10 × 100
    assert g["net_gamma"] == 20.0
    assert g["net_vega"] == 100.0
    # delta_dollar = delta × contracts × multiplier × mark
    # = 0.4 × 10 × 100 × 1.5 = 600
    assert math.isclose(g["delta_dollar"], 600.0)
    assert math.isclose(g["delta_dollar_pct_equity"], 600.0 / 100_000)


def test_short_position_subtracts_delta():
    """SELL side flips Greek sign."""
    short_p = _opt(side="SELL", qty=5, delta=0.5)
    long_p = _opt(symbol="US.QQQ_C", underlying="US.QQQ", qty=5, delta=0.5)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[short_p, long_p])
    # short cancels long
    assert math.isclose(snap.aggregate_greeks["net_delta"], 0.0)


def test_heat_metric_options_use_premium_at_risk():
    """Long premium: at-risk = entry × qty × 100."""
    p = _opt(entry_price=2.0, qty=10)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    # at_risk = 2 × 10 × 100 = 2000; heat = 2000 / 100k = 0.02
    assert math.isclose(snap.heat_metrics["heat_pct"], 0.02)
    assert math.isclose(snap.heat_metrics["max_per_underlying_pct"], 0.02)


def test_heat_aggregates_per_underlying():
    p1 = _opt(symbol="US.SPY_C", underlying="US.SPY", entry_price=1.0, qty=5)
    p2 = _opt(symbol="US.SPY_C2", underlying="US.SPY", entry_price=2.0, qty=5)
    p3 = _opt(symbol="US.QQQ_C", underlying="US.QQQ", entry_price=1.0, qty=5)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p1, p2, p3])
    # SPY at-risk = (1+2)×5×100 = 1500; QQQ = 500. Total = 2000. Max-per-und = 1500.
    assert math.isclose(snap.heat_metrics["heat_pct"], 0.02)
    assert math.isclose(snap.heat_metrics["max_per_underlying_pct"], 0.015)


def test_factor_exposure_aggregates_by_sector_bucket():
    tech = _opt(underlying="US.AAPL", sector="Information Technology", delta=0.4, qty=10, mark=2.0)
    fin = _opt(symbol="US.XLF_C", underlying="US.XLF", sector="Financials", delta=0.4, qty=10, mark=2.0)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[tech, fin])
    # Each notional = 0.4 × 1 × 10 × 100 × 2.0 = 800; tech and rates_sensitive each 0.008
    assert "tech" in snap.factor_exposures
    assert "rates_sensitive" in snap.factor_exposures
    assert math.isclose(snap.factor_exposures["tech"], 0.008)


def test_correlation_matrix_identity_on_self():
    daily = {
        "US.SPY": [0.01, -0.005, 0.002, 0.003, -0.001, 0.0, 0.004, 0.001],
        "US.QQQ": [0.012, -0.006, 0.003, 0.004, 0.0, 0.001, 0.005, 0.001],
    }
    p = _opt()
    snap = build_snapshot(
        equity=100_000, cash=100_000, open_positions=[p], daily_returns=daily
    )
    assert snap.correlations["US.SPY"]["US.SPY"] == 1.0
    # Highly correlated movements
    assert snap.correlations["US.SPY"]["US.QQQ"] > 0.9


def test_data_quality_flags_missing_greeks():
    p = _opt(delta=None, gamma=None)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    assert snap.data_quality["degradation_level"] == 2
    assert any("greeks" in m for m in snap.data_quality["missing"])


def test_to_db_payload_round_trip_friendly():
    p = _opt()
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    payload = snap.to_db_payload()
    assert "open_positions" in payload
    assert "aggregate_greeks" in payload
    assert isinstance(payload["open_positions"], list)
    assert payload["open_positions"][0]["symbol"] == p.symbol
