"""R5e — multi-leg defined-risk credit verticals (area A).

check_combo proves a two-leg structure is a defined-risk credit vertical
(long leg caps the short) before sizing it as ONE position off its
max_loss = (width − net_credit) × 100 × contracts. Non-verticals are blocked
outright — they must never fall through to a naked-short evaluation.
"""
from __future__ import annotations

from trading_agent.sizing import (
    R1,
    R3,
    R5,
    R5E,
    ComboLeg,
    OpenPosition,
    ProposedCombo,
    SizingContext,
    blockers,
    check_combo,
)


def _ctx(equity=100_000.0, opens=(), sectors=True):
    return SizingContext(equity=equity, cash=equity, opens=tuple(opens),
                         sector_lookup_available=sectors)


def _put_credit_spread(qty=1.0, short_strike=100.0, long_strike=95.0,
                       short_price=2.0, long_price=0.8, dte=30, sector="Tech"):
    """Sell the higher-strike put, buy the lower-strike put (long caps short)."""
    return ProposedCombo(
        ticker="SPY", sector=sector, strategy_label="credit_put_spread_30_45",
        legs=(
            ComboLeg(option_symbol=f"US.SPY260717P{int(short_strike*1000):08d}",
                     side="SELL", contracts=qty, price=short_price, right="P",
                     strike=short_strike, dte=dte, delta=-0.30),
            ComboLeg(option_symbol=f"US.SPY260717P{int(long_strike*1000):08d}",
                     side="BUY", contracts=qty, price=long_price, right="P",
                     strike=long_strike, dte=dte, delta=-0.18),
        ),
    )


def _call_credit_spread(qty=1.0, short_strike=100.0, long_strike=105.0,
                        short_price=2.0, long_price=0.8, dte=30):
    return ProposedCombo(
        ticker="SPY", sector="Tech", strategy_label="credit_call_spread",
        legs=(
            ComboLeg(option_symbol="US.SPY260717C00100000", side="SELL",
                     contracts=qty, price=short_price, right="C",
                     strike=short_strike, dte=dte, delta=0.30),
            ComboLeg(option_symbol="US.SPY260717C00105000", side="BUY",
                     contracts=qty, price=long_price, right="C",
                     strike=long_strike, dte=dte, delta=0.18),
        ),
    )


# -- happy paths -------------------------------------------------------------


def test_valid_put_credit_spread_passes():
    combo = _put_credit_spread()
    assert combo.net_credit == 1.2
    assert combo.width == 5.0
    assert combo.max_loss == (5.0 - 1.2) * 100.0  # 380
    assert blockers(check_combo(_ctx(), combo)) == []


def test_valid_call_credit_spread_passes():
    assert blockers(check_combo(_ctx(), _call_credit_spread())) == []


# -- defined-risk invariants (R5e) -------------------------------------------


def test_three_legs_blocked():
    combo = _put_credit_spread()
    combo = ProposedCombo(ticker="SPY", legs=combo.legs + (combo.legs[0],))
    vs = check_combo(_ctx(), combo)
    assert any(v.rule == R5E for v in vs)


def test_two_buys_no_short_blocked():
    combo = _put_credit_spread()
    long_like = ComboLeg(option_symbol="x", side="BUY", contracts=1.0,
                         price=1.0, right="P", strike=90.0, dte=30)
    combo = ProposedCombo(ticker="SPY", legs=(combo.long_leg, long_like))
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


def test_mismatched_expiry_blocked():
    combo = _put_credit_spread()
    far = ComboLeg(option_symbol="x", side="BUY", contracts=1.0, price=0.8,
                   right="P", strike=95.0, dte=60)  # different dte
    combo = ProposedCombo(ticker="SPY", legs=(combo.short_leg, far))
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


def test_mismatched_right_blocked():
    short_put = _put_credit_spread().short_leg
    long_call = ComboLeg(option_symbol="x", side="BUY", contracts=1.0,
                         price=0.8, right="C", strike=105.0, dte=30)
    combo = ProposedCombo(ticker="SPY", legs=(short_put, long_call))
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


def test_unequal_contracts_blocked():
    combo = _put_credit_spread()
    bad_long = ComboLeg(option_symbol="x", side="BUY", contracts=2.0,
                        price=0.8, right="P", strike=95.0, dte=30)
    combo = ProposedCombo(ticker="SPY", legs=(combo.short_leg, bad_long))
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


def test_long_does_not_cap_short_put_blocked():
    """A put spread where the long strike is ABOVE the short = not capped."""
    combo = _put_credit_spread(short_strike=95.0, long_strike=100.0)  # inverted
    vs = check_combo(_ctx(), combo)
    assert any(v.rule == R5E and "does not cap" in v.message for v in vs)


def test_long_does_not_cap_short_call_blocked():
    combo = _call_credit_spread(short_strike=105.0, long_strike=100.0)  # inverted
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


def test_debit_spread_blocked():
    """Long premium > short premium → net debit → blocked."""
    combo = _put_credit_spread(short_price=0.8, long_price=2.0)
    assert any(v.rule == R5E and "credit" in v.message for v in check_combo(_ctx(), combo))


def test_zero_max_loss_blocked():
    """net_credit >= width → max_loss <= 0 (mispriced); blocked."""
    combo = _put_credit_spread(short_strike=100.0, long_strike=95.0,
                               short_price=6.0, long_price=0.5)  # credit 5.5 >= width 5
    assert any(v.rule == R5E for v in check_combo(_ctx(), combo))


# -- sizing the combo as one position ----------------------------------------


def test_r1_blocks_oversized_combo():
    """width 50, credit 1, qty 2 → max_loss $9800 > 2.5% of $100k ($2500)."""
    combo = _put_credit_spread(qty=2.0, short_strike=150.0, long_strike=100.0,
                               short_price=3.0, long_price=2.0)
    assert combo.max_loss == (50.0 - 1.0) * 100.0 * 2  # 9800
    assert any(v.rule == R1 for v in check_combo(_ctx(), combo))


def test_r3_counts_combo_max_loss_against_ticker_exposure():
    """Existing $11,800 SPY + combo max_loss $380 = $12,180 > 12% ($12,000)."""
    existing = OpenPosition(symbol="US.SPY", ticker="SPY", asset_type="STK",
                            qty=100.0, entry_price=118.0, sector="Tech")
    vs = check_combo(_ctx(opens=[existing]), _put_credit_spread())
    assert any(v.rule == R3 for v in vs)


def test_r5_dte_band_enforced_on_combo():
    combo = _put_credit_spread(dte=5)  # below OPTION_DTE_MIN (14)
    assert any(v.rule == R5 for v in check_combo(_ctx(), combo))


def test_combo_unaffected_by_single_leg_short_block():
    """check_combo must NOT emit R_short_option_open_blocked — the bounded
    short leg is governed by R5e's width-minus-credit math, not the per-leg
    naked-short path."""
    vs = check_combo(_ctx(), _put_credit_spread())
    assert all("short_option_open_blocked" not in v.rule for v in vs)


def test_open_position_risk_notional_override():
    """A combo is journaled at net_credit but its R3/R4 exposure must be its
    max_loss — the risk_notional override carries that through OpenPosition."""
    combo_row = OpenPosition(
        symbol="US.SPY260717P00100000", ticker="SPY", asset_type="OPT",
        qty=1.0, entry_price=1.2, risk_notional=380.0,
    )
    assert combo_row.notional == 380.0  # NOT 1.2 × 100 × 1 = 120
    plain = OpenPosition(
        symbol="US.SPY260717P00100000", ticker="SPY", asset_type="OPT",
        qty=1.0, entry_price=1.2,
    )
    assert plain.notional == 120.0  # no override → credit-based product
