"""Shared OOS evaluation helpers for the regime layer.

These functions used to live as near-identical copies inside
scripts/rolling_walkforward.py and scripts/walkforward_backtest.py.
scripts/validate_production_hmm.py needs the same forward-return and
headline math — a third copy is how the v1-v3 evaluation bugs survived
so long (each script fixed independently). Single home, imported by all
three scripts and by tests.

No Postgres imports here on purpose: tests must be able to exercise the
statistics (Newey-West variance especially) without a DB or network.
yfinance is imported lazily inside the one function that needs it.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd


def yfinance_spy_returns(start_dt, end_dt) -> pd.Series:
    """Daily SPY simple returns from yfinance, indexed by date.

    Padded ±10/+40 calendar days so forward-return windows ending near
    `end_dt` still have data. Raises RuntimeError on an empty download so
    callers fail loudly instead of computing metrics on nothing.
    """
    import yfinance as yf

    pad_start = (start_dt - timedelta(days=10)).isoformat()
    pad_end = (end_dt + timedelta(days=40)).isoformat()
    df = yf.download("SPY", start=pad_start, end=pad_end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no SPY data for {pad_start} → {pad_end}")
    close = df["Close"]
    # When yfinance returns a MultiIndex (Ticker, Field), squeeze it.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change()
    rets.index = pd.to_datetime(rets.index).date
    return rets


def fwd_return(rets: pd.Series, day, horizon: int) -> float:
    """Compound simple returns over [day+1, day+horizon] trading days.

    NaN when the series doesn't extend `horizon` days past `day` (test
    window ends inside the forward horizon) — callers must dropna, not
    zero-fill, or they understate tail-day variance.
    """
    idx = rets.index.searchsorted(day, side="right")
    take = rets.iloc[idx : idx + horizon]
    if len(take) < horizon:
        return float("nan")
    return float((1.0 + take).prod() - 1.0)


def newey_west_variance(x, lag: int = 20) -> float:
    """Newey-West (1987) long-run variance of a series, plain numpy.

    Returns the heteroskedasticity-and-autocorrelation-consistent estimate
    of Var(x_t) accounting for serial correlation up to `lag`:

        s² = γ₀ + 2 · Σ_{k=1..lag} w_k · γ_k,   w_k = 1 − k/(lag+1)

    where γ_k is the lag-k autocovariance (divided by n, the standard
    biased estimator — guarantees s² ≥ 0 with the Bartlett kernel) and the
    triangular Bartlett weights w_k taper longer lags. Var(mean(x)) = s²/n.

    Why we need it: the overlay's daily active returns are serially
    correlated by construction — the regime label (hence size multiplier)
    persists for weeks, and per-regime stats use overlapping 20-day forward
    windows. A naive iid SE on ~2400 such days overstates the effective
    sample several-fold; lag≈20 matches both the label persistence scale
    and the forward-return overlap.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    lag = max(0, min(int(lag), n - 1))
    xc = x - x.mean()
    gamma0 = float(xc @ xc) / n
    s = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(xc[k:] @ xc[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    return float(s)


def active_return_stats(strat_daily, spy_daily, *, nw_lag: int = 20) -> dict:
    """IR of strategy vs SPY with overlap-aware (Newey-West) standard errors.

    Inputs are aligned daily simple-return series (array-like). Returns a
    dict with the annualized information ratio, the NW standard error of
    the mean daily active return, and the resulting t-stat. The t-stat —
    not the raw IR — is the number to quote when asking "is this edge
    distinguishable from zero": mean(a) / sqrt(NW_var(a)/n).
    """
    a = np.asarray(strat_daily, dtype=float) - np.asarray(spy_daily, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    mean_a = float(a.mean()) if n else float("nan")
    std_a = float(a.std(ddof=1)) if n > 1 else float("nan")
    ir_ann = mean_a / std_a * math.sqrt(252.0) if std_a and std_a > 0 else float("nan")
    nw_var = newey_west_variance(a, lag=nw_lag)
    nw_se_mean = math.sqrt(nw_var / n) if n and nw_var == nw_var else float("nan")
    naive_se_mean = std_a / math.sqrt(n) if n and std_a == std_a else float("nan")
    t_stat = mean_a / nw_se_mean if nw_se_mean and nw_se_mean > 0 else float("nan")
    return {
        "n_days": n,
        "mean_active_daily": mean_a,
        "ir_annualized": ir_ann,
        "nw_lag": nw_lag,
        "nw_se_mean_daily": nw_se_mean,
        "naive_se_mean_daily": naive_se_mean,
        "t_stat_nw": t_stat,
    }


def overlay_headline(strat_daily, spy_daily, *, nw_lag: int = 20) -> dict:
    """Headline metrics for a size-multiplier overlay vs SPY buy-and-hold.

    Same definitions as scripts/rolling_walkforward.py's headline block
    (cumulative compounded return, annualized Sharpe, max drawdown of the
    compounded curve) plus the NW-aware active-return stats. Not a
    tradeable PnL — no costs, slippage, or options pricing.
    """
    strat = pd.Series(np.asarray(strat_daily, dtype=float)).dropna()
    spy = pd.Series(np.asarray(spy_daily, dtype=float)).dropna()
    cum_strat = float((1.0 + strat).prod() - 1.0)
    cum_spy = float((1.0 + spy).prod() - 1.0)
    sr = float(strat.mean() / strat.std() * math.sqrt(252.0)) if strat.std() > 0 else float("nan")
    ssr = float(spy.mean() / spy.std() * math.sqrt(252.0)) if spy.std() > 0 else float("nan")
    cum_curve = (1.0 + strat).cumprod()
    mdd = float((cum_curve / cum_curve.cummax() - 1.0).min()) if len(cum_curve) else float("nan")
    spy_curve = (1.0 + spy).cumprod()
    spy_mdd = float((spy_curve / spy_curve.cummax() - 1.0).min()) if len(spy_curve) else float("nan")
    return {
        "cum_strategy_ret": cum_strat,
        "cum_spy_ret": cum_spy,
        "cum_gap_pp": (cum_strat - cum_spy) * 100.0,
        "strategy_sharpe_annualized": sr,
        "spy_sharpe_annualized": ssr,
        "strategy_max_drawdown": mdd,
        "spy_max_drawdown": spy_mdd,
        "active_vs_spy": active_return_stats(strat_daily, spy_daily, nw_lag=nw_lag),
    }
