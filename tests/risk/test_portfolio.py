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


def test_heat_aggregates_per_expiry():
    """Per-expiry bucket sums at-risk by parsed OCC expiry (YYMMDD)."""
    p1 = _opt(symbol="US.SPY260710C00500000", underlying="US.SPY", entry_price=1.0, qty=5)
    p2 = _opt(symbol="US.SPY260710C00510000", underlying="US.SPY", entry_price=2.0, qty=5)
    p3 = _opt(symbol="US.QQQ260814C00400000", underlying="US.QQQ", entry_price=1.0, qty=5)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p1, p2, p3])
    # 260710 = (1+2)×5×100 = 1500; 260814 = 500. max-per-expiry = 1500 → 1.5%.
    assert math.isclose(snap.heat_metrics["per_expiry_usd"]["260710"], 1500.0)
    assert math.isclose(snap.heat_metrics["per_expiry_usd"]["260814"], 500.0)
    assert math.isclose(snap.heat_metrics["max_per_expiry_pct"], 0.015)


def test_heat_per_expiry_clusters_across_underlyings():
    """Event clustering: different underlyings sharing one expiry sum together."""
    p1 = _opt(symbol="US.SPY260710C00500000", underlying="US.SPY", entry_price=1.0, qty=5)
    p2 = _opt(symbol="US.QQQ260710C00400000", underlying="US.QQQ", entry_price=1.0, qty=5)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p1, p2])
    # both expire 260710 → 500 + 500 = 1000 in that one bucket → 1.0%.
    assert math.isclose(snap.heat_metrics["per_expiry_usd"]["260710"], 1000.0)
    assert math.isclose(snap.heat_metrics["max_per_expiry_pct"], 0.01)


def test_heat_short_put_uses_assignment_exposure():
    """Short put heat = strike × 100 × qty − premium collected (CSP assignment).

    Regression: short legs used to contribute ZERO heat — the open AAPL 8x
    short 300P (journal id 12) carried ~$235k assignment exposure invisible
    to every heat guardrail.
    """
    p = _opt(
        symbol="US.AAPL260710P300000", underlying="US.AAPL",
        side="SELL", qty=8, entry_price=0.60, strategy_label="csp",
    )
    snap = build_snapshot(equity=1_000_000, cash=1_000_000, open_positions=[p])
    # at_risk = 300 × 100 × 8 − 0.60 × 100 × 8 = 240,000 − 480 = 239,520
    assert math.isclose(snap.heat_metrics["total_at_risk_usd"], 239_520.0)
    assert math.isclose(snap.heat_metrics["heat_pct"], 0.23952)
    assert math.isclose(snap.heat_metrics["max_per_underlying_pct"], 0.23952)


def test_heat_short_call_uses_stress_buffered_stop_distance():
    """Short call with stop: NAKED_CALL_STRESS_MULT × (stop − entry) × 100 × qty."""
    p = _opt(
        symbol="US.SPY260710C600000", underlying="US.SPY",
        side="SELL", qty=4, entry_price=2.0, stop=5.0,
    )
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    # at_risk = 1.5 × (5.0 − 2.0) × 4 × 100 = 1,800
    assert math.isclose(snap.heat_metrics["total_at_risk_usd"], 1_800.0)
    assert math.isclose(snap.heat_metrics["heat_pct"], 0.018)


def test_heat_short_call_without_stop_falls_back_to_strike_notional():
    """No usable stop on a short call → full strike notional, not zero.

    A stop at/below entry is a take-profit, not a loss cap, so it must hit
    the same conservative fallback as a missing stop.
    """
    no_stop = _opt(
        symbol="US.SPY260710C600000", underlying="US.SPY",
        side="SELL", qty=4, entry_price=2.0, stop=None,
    )
    snap = build_snapshot(equity=1_000_000, cash=1_000_000, open_positions=[no_stop])
    # at_risk = 600 × 100 × 4 = 240,000
    assert math.isclose(snap.heat_metrics["total_at_risk_usd"], 240_000.0)

    profit_stop = _opt(
        symbol="US.SPY260710C600000", underlying="US.SPY",
        side="SELL", qty=4, entry_price=2.0, stop=1.0,
    )
    snap2 = build_snapshot(equity=1_000_000, cash=1_000_000, open_positions=[profit_stop])
    assert math.isclose(snap2.heat_metrics["total_at_risk_usd"], 240_000.0)


def test_heat_unparseable_short_option_charges_premium_multiple():
    """Short option whose symbol won't parse still contributes heat (10× premium)."""
    p = _opt(symbol="US.WEIRD_SYMBOL", underlying="US.WEIRD", side="SELL",
             qty=2, entry_price=3.0)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    # at_risk = 3.0 × 2 × 100 × 10 = 6,000
    assert math.isclose(snap.heat_metrics["total_at_risk_usd"], 6_000.0)


def test_heat_mixed_long_and_short_legs_aggregate_per_underlying():
    """Long premium and short assignment exposure sum into the same underlying."""
    long_leg = _opt(symbol="US.AAPL260710C320000", underlying="US.AAPL",
                    entry_price=2.0, qty=8)
    short_leg = _opt(symbol="US.AAPL260710P300000", underlying="US.AAPL",
                     side="SELL", qty=8, entry_price=0.60)
    snap = build_snapshot(
        equity=1_000_000, cash=1_000_000, open_positions=[long_leg, short_leg]
    )
    # long: 2.0 × 8 × 100 = 1,600; short put: 239,520. AAPL total = 241,120.
    assert math.isclose(snap.heat_metrics["total_at_risk_usd"], 241_120.0)
    assert math.isclose(snap.heat_metrics["per_underlying_usd"]["US.AAPL"], 241_120.0)


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
    # Missing greeks on a HELD position is a WARNING (level 1), not a hard
    # block: it must never DEFER new entries. A single un-greek'd holding
    # previously froze the whole entry engine. The symbol is still surfaced
    # in `missing` so exit-side monitoring can flag it.
    p = _opt(delta=None, gamma=None)
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    assert snap.data_quality["degradation_level"] == 1
    assert any("greeks" in m for m in snap.data_quality["missing"])


def test_to_db_payload_round_trip_friendly():
    p = _opt()
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[p])
    payload = snap.to_db_payload()
    assert "open_positions" in payload
    assert "aggregate_greeks" in payload
    assert isinstance(payload["open_positions"], list)
    assert payload["open_positions"][0]["symbol"] == p.symbol


def test_proposal_without_option_delta_is_not_missing_greeks():
    # Regression: a proposal that omits option_delta must yield a hypothetical
    # with delta/gamma defaulted to 0.0 (never None), so it does NOT read as
    # "missing critical greeks" → data_quality level 2 → DEFER. In candidate_entry
    # the hypothetical is the only snapshot position, so a delta-less proposal
    # previously froze every new entry.
    from trading_agent.graph.nodes.risk_nodes import _proposal_to_hypothetical

    proposal = {
        "symbol": "US.NVDA260116C00150000", "ticker": "NVDA",
        "asset_type": "OPT", "side": "BUY", "qty": 1, "entry_price": 2.5,
    }  # deliberately no option_delta / option_gamma
    hyp = _proposal_to_hypothetical(proposal)
    assert hyp.delta == 0.0
    assert hyp.gamma == 0.0
    snap = build_snapshot(equity=100_000, cash=100_000, open_positions=[hyp])
    assert snap.data_quality["degradation_level"] == 0
    assert not any("greeks" in m for m in snap.data_quality["missing"])
