"""R7 — Risk:reward floor at the sizing layer.

The pydantic validator on TraderProposal is the strict gate at synthesizer
time (tested in tests/llm/test_schemas.py); R7 in sizing.py is the same
rule callable from any non-schema call path (manual rescue order, future
backtester, etc.) and visible to the pretool hook's audit log.
"""
from __future__ import annotations

from trading_agent.sizing import (
    MIN_RISK_REWARD,
    ProposedTrade,
    R7,
    R_TARGET_MISSING,
    SizingContext,
    blockers,
    check,
)


def _ctx(equity: float = 100_000.0) -> SizingContext:
    return SizingContext(equity=equity, opens=(), sector_lookup_available=True)


def _trade(stop: float | None = 8.0, target: float | None = 13.0) -> ProposedTrade:
    """Default: entry=10, qty=10. Risk=$2/sh × 10 = $20 (well under R1)."""
    return ProposedTrade(
        ticker="AAPL", asset_type="STK", side="BUY",
        qty=10, entry_price=10.0,
        stop=stop, target=target,
        strategy_label="trend",
    )


# ---------------------------------------------------------------------------
# R7 blocks low R:R
# ---------------------------------------------------------------------------


def test_r7_blocks_bad_rr():
    """entry=10, stop=5, target=13 → risk=5, reward=3, R:R 0.6 < 1.5 → block.

    Mirrors the SPY 742C audit: stop too tight, target too modest.
    """
    vs = check(_ctx(), _trade(stop=5.0, target=13.0))
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1
    assert "0.60" in r7_blocks[0].message


def test_r7_accepts_exactly_15():
    """R:R exactly 1.5 passes."""
    vs = check(_ctx(), _trade(stop=8.0, target=13.0))  # R:R = 1.5
    assert not any(v.rule == R7 for v in vs)


def test_r7_blocks_zero_risk_stop_at_entry():
    vs = check(_ctx(), _trade(stop=10.0, target=15.0))  # risk = 0
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1


def test_r7_blocks_zero_reward_target_at_entry():
    vs = check(_ctx(), _trade(stop=5.0, target=10.0))  # reward = 0
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1


# ---------------------------------------------------------------------------
# Warn-only when target is missing
# ---------------------------------------------------------------------------


def test_target_missing_warns_not_blocks():
    """Hook-layer ProposedTrade may not have target; warn rather than
    block (synthesizer validator catches it earlier upstream)."""
    vs = check(_ctx(), _trade(stop=8.0, target=None))
    warns = [v for v in vs if v.rule == R_TARGET_MISSING]
    assert len(warns) == 1
    assert warns[0].severity == "warn"
    # No R7 blocker should appear in this case
    assert not blockers([v for v in vs if v.rule == R7])


def test_sell_side_skips_r7():
    """Closing orders (SELL) shouldn't be R7-checked."""
    t = ProposedTrade(
        ticker="AAPL", asset_type="STK", side="SELL",
        qty=10, entry_price=10.0, stop=9.0, target=10.5,  # bad R:R, but SELL
    )
    vs = check(_ctx(), t)
    assert not any(v.rule == R7 for v in vs)


# ---------------------------------------------------------------------------
# Sanity: constant matches schema constant
# ---------------------------------------------------------------------------


def test_min_rr_constant_matches_schema():
    """sizing.py and llm/schemas.py both hold a MIN_RISK_REWARD constant;
    they must move together. This guards against drift."""
    from trading_agent.llm.schemas import MIN_RISK_REWARD as schema_min
    assert MIN_RISK_REWARD == schema_min
