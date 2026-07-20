"""Area E — soak acceptance gate 6 now compares paper Sharpe vs a real SPY
Sharpe (it used to only check paper Sharpe > 0).

Tests patch the two fetch helpers so no Postgres / yfinance is touched.
"""
from __future__ import annotations

from unittest.mock import patch

import scripts.check_soak_acceptance as soak


def _patch(paper, spy_exc=None, spy=None, source="portfolio_marks_nav"):
    """Patch the paper + SPY return fetchers. spy_exc raises on SPY fetch."""
    p_paper = patch.object(soak, "fetch_paper_daily_returns",
                           return_value=(paper, source))
    if spy_exc is not None:
        p_spy = patch.object(soak, "fetch_spy_daily_returns", side_effect=spy_exc)
    else:
        p_spy = patch.object(soak, "fetch_spy_daily_returns", return_value=spy)
    return p_paper, p_spy


# Steady positive paper series → high Sharpe; choppy zero-mean SPY → ~0 Sharpe.
_GOOD_PAPER = [0.012, 0.008, 0.011, 0.009, 0.010, 0.012, 0.008, 0.011, 0.009, 0.010]
_CHOPPY_SPY = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02]


def test_annualized_sharpe_helper():
    assert soak._annualized_sharpe([0.01]) is None          # <2 points
    assert soak._annualized_sharpe([0.01, 0.01, 0.01]) is None  # zero variance
    s = soak._annualized_sharpe([0.012, 0.008, 0.011, 0.009])
    assert s is not None and s > 0


def test_pass_when_paper_beats_spy():
    p_paper, p_spy = _patch(_GOOD_PAPER, spy=_CHOPPY_SPY)
    with p_paper, p_spy:
        gate = soak.check_paper_sharpe_vs_spy(since="x")
    assert gate["label"] == "paper_sharpe_vs_spy"
    assert gate["pass"] is True


def test_fail_when_spy_beats_paper():
    p_paper, p_spy = _patch(_CHOPPY_SPY, spy=_GOOD_PAPER)
    with p_paper, p_spy:
        gate = soak.check_paper_sharpe_vs_spy(since="x")
    assert gate["label"] == "paper_sharpe_vs_spy"
    assert gate["pass"] is False


def test_fail_on_spy_fetch_error():
    p_paper, p_spy = _patch(_GOOD_PAPER, spy_exc=RuntimeError("no data"))
    with p_paper, p_spy:
        gate = soak.check_paper_sharpe_vs_spy(since="x")
    assert gate["pass"] is False
    assert "SPY" in gate["detail"]


def test_fail_when_no_paper_returns():
    p_paper, p_spy = _patch([], spy=_CHOPPY_SPY)
    with p_paper, p_spy:
        gate = soak.check_paper_sharpe_vs_spy(since="x")
    assert gate["pass"] is False
    assert "no paper returns" in gate["detail"]


def test_pass_label_consistent_per_trade_fallback():
    """Per-trade fallback source still labels the gate paper_sharpe_vs_spy."""
    p_paper, p_spy = _patch(_GOOD_PAPER, spy=_CHOPPY_SPY,
                            source="per_trade_realized_r")
    with p_paper, p_spy:
        gate = soak.check_paper_sharpe_vs_spy(since="x")
    assert gate["label"] == "paper_sharpe_vs_spy"
    assert "per_trade_realized_r" in gate["detail"]
