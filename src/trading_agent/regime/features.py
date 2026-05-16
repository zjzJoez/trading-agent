"""Regime feature engineering.

Pulls daily bars + macro data, computes a low-dimensional standardized
feature vector. Designed to be cheap (CPU-only, <1s end-to-end) and
deterministic (same as_of date → same features).

Feature groups (§6.2 of plan):
    Equity trend        — SPY/QQQ/IWM 5/20/60d returns + price-vs-MA
    Realized vol        — 5/20/60d annualized RV + ATR%
    Vol risk            — VIX level + 5d Δ + percentile
    Cross-asset         — TLT/HYG/GLD/DXY (proxy via UUP) momentum
    Rates               — 10Y-2Y slope + 20d Δ in 10Y yield
    Credit stress       — HYG/TLT, HYG drawdown
    Correlation stress  — avg pairwise 20d corr across SPY/QQQ/IWM/sectors
    Breadth proxy       — % of watchlist > 20DMA
    Data quality        — missing/stale/outlier counts → degradation_level
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Tickers used in feature extraction. Sector ETFs let us approximate the
# S&P sector breadth without subscribing to a paid index dataset.
EQUITY_BENCHMARKS = ["US.SPY", "US.QQQ", "US.IWM"]
CROSS_ASSETS = ["US.TLT", "US.HYG", "US.GLD", "US.UUP"]   # UUP ≈ DXY proxy
SECTOR_ETFS = [
    "US.XLK", "US.XLF", "US.XLE", "US.XLV", "US.XLY",
    "US.XLP", "US.XLI", "US.XLB", "US.XLU", "US.XLRE", "US.XLC",
]
# Moomoo doesn't expose the VIX index directly (US.^VIX returns ret=-1).
# VIXY is the 1x VIX-tracking ETF; its level is a usable proxy for the VIX
# regime feature (close-to-close changes are highly correlated with VIX).
# When we eventually add a free FRED feed (Phase 2.5+), we'll replace this
# with the actual VIX close.
VIX_TICKER = "US.VIXY"
ALL_TICKERS = list(set(EQUITY_BENCHMARKS + CROSS_ASSETS + SECTOR_ETFS + [VIX_TICKER]))

# Lookback windows used in features
WIN_SHORT = 5
WIN_MID = 20
WIN_LONG = 60
WIN_VIX_PCTL = 252


# ---------------------------------------------------------------------------
# Output dataclass — what the classifier consumes and what we persist
# ---------------------------------------------------------------------------


@dataclass
class FeatureSnapshot:
    """Output of a feature build. Persisted to regime_feature_snapshots."""

    as_of: datetime
    features: dict[str, float] = field(default_factory=dict)
    source_freshness: dict[str, float] = field(default_factory=dict)  # seconds-old per source
    data_quality: dict[str, Any] = field(default_factory=dict)        # missing/stale/outlier flags
    raw_klines: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    @property
    def degradation_level(self) -> int:
        return int(self.data_quality.get("degradation_level", 0))

    def as_vector(self) -> np.ndarray:
        """Flat ordered numeric vector for the HMM. Order is fixed (see ORDER below)."""
        return np.array([self.features.get(k, 0.0) for k in HMM_FEATURE_ORDER], dtype=float)

    def to_db_payload(self) -> dict[str, Any]:
        """Plain dict suitable for Postgres jsonb columns."""
        return {
            "features": _safe_floats(self.features),
            "source_freshness": _safe_floats(self.source_freshness),
            "data_quality": self.data_quality,
        }


# Order matters — HMM emission means/covars are stored aligned with this.
# IMPORTANT: As of 2026-05-17, the binary `spy_above_*dma` features were
# replaced with continuous `spy_dist_*dma` ((price - MA) / MA). Binary
# features made the 17×17 full-covariance emission matrix ill-conditioned,
# causing hmmlearn to abort EM at iter 3-4 with log-likelihood decreasing.
# Continuous versions preserve all the information (sign(dist) == above_ma)
# and add the magnitude signal "how far above/below the MA" which is itself
# a useful regime indicator.
#
# Old HMM models (regime_model_versions id=1,2,3) still embed the OLD
# feature_order via HMMModel.feature_order; they continue to work as long
# as snapshots also contain `spy_above_*dma` keys (which features.py still
# writes). When loading an old model, FeatureSnapshot.as_vector() uses
# HMM_FEATURE_ORDER from the loaded model, not this module-level constant.
HMM_FEATURE_ORDER = [
    "spy_ret_5", "spy_ret_20", "spy_ret_60",
    "spy_dist_50dma", "spy_dist_200dma",   # was: spy_above_50dma / spy_above_200dma
    "spy_rv_20", "spy_rv_60",
    "vix_level", "vix_chg_5", "vix_pctl_252",
    "breadth_above_20dma_pct",
    "tlt_mom_20", "hyg_mom_20", "gld_mom_20", "uup_mom_20",
    "avg_pairwise_corr_20",
    "yield_10y2y_slope_proxy",  # via TLT vs SHY ratio if available; else 0
]


# ---------------------------------------------------------------------------
# Helpers — stateless feature math on a single price series
# ---------------------------------------------------------------------------


def _ret(s: pd.Series, n: int) -> float:
    """n-day simple return of the LAST point. Returns NaN-safe 0.0 on missing."""
    if s is None or len(s) <= n:
        return 0.0
    cur, prev = s.iloc[-1], s.iloc[-(n + 1)]
    if not (prev and not math.isnan(prev) and not math.isnan(cur)):
        return 0.0
    return float(cur / prev - 1.0)


def _realized_vol(s: pd.Series, n: int) -> float:
    """Annualized realized vol of last n daily log returns."""
    if s is None or len(s) <= n:
        return 0.0
    log_rets = np.log(s.iloc[-(n + 1):] / s.iloc[-(n + 1):].shift(1)).dropna()
    if len(log_rets) < 3:
        return 0.0
    return float(log_rets.std(ddof=0) * math.sqrt(252.0))


def _above_ma(s: pd.Series, n: int) -> float:
    """1.0 if last price > simple MA(n); 0.0 otherwise. -1 if insufficient data.

    Kept for backwards compat / inspection. NOT used in HMM_FEATURE_ORDER
    anymore — see `_dist_from_ma`. The binary form causes ill-conditioning
    in the full-covariance HMM emission (column variance is too small,
    EM optimizer aborts at iter 3-4).
    """
    if s is None or len(s) < n:
        return 0.0
    ma = s.iloc[-n:].mean()
    return 1.0 if s.iloc[-1] > ma else 0.0


def _dist_from_ma(s: pd.Series, n: int) -> float:
    """Continuous: (last_price - MA(n)) / MA(n). Sign matches `_above_ma`,
    but with magnitude so the HMM sees a real-valued feature instead of a
    binary one. Returns 0.0 (neutral) on insufficient data.
    """
    if s is None or len(s) < n:
        return 0.0
    ma = float(s.iloc[-n:].mean())
    if ma == 0 or math.isnan(ma):
        return 0.0
    last = float(s.iloc[-1])
    if math.isnan(last):
        return 0.0
    return (last - ma) / ma


def _percentile(value: float, history: pd.Series) -> float:
    """Percentile rank of `value` against `history` (0..1)."""
    h = history.dropna()
    if len(h) < 10:
        return 0.5
    return float((h <= value).sum() / len(h))


def _avg_pairwise_corr(prices: dict[str, pd.Series], window: int = 20) -> float:
    """Average pairwise corr of daily returns over last `window` days across given prices."""
    rets = []
    for k, s in prices.items():
        if s is None or len(s) <= window:
            continue
        r = s.iloc[-(window + 1):].pct_change().dropna().rename(k)
        rets.append(r)
    if len(rets) < 2:
        return 0.0
    df = pd.concat(rets, axis=1).dropna()
    if len(df) < 5:
        return 0.0
    corr = df.corr().values
    n = corr.shape[0]
    # average of strict upper triangle
    return float(corr[np.triu_indices(n, k=1)].mean())


def _safe_floats(d: dict) -> dict:
    """Replace NaN/Inf with None for jsonb safety."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                out[k] = None
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_feature_snapshot(
    as_of: datetime | None = None,
    *,
    klines: dict[str, pd.DataFrame] | None = None,
    vix_level: float | None = None,
    vix_history: pd.Series | None = None,
    lookback_days: int = WIN_VIX_PCTL + WIN_LONG + 5,
) -> FeatureSnapshot:
    """Compute the full feature vector.

    `klines` is a dict {ticker -> OHLCV DataFrame indexed by date, latest last}.
    Pass None and we'll fetch via the moomoo Phase-1 module (TODO Phase 2.2.1).
    For unit tests, pass synthetic DataFrames directly.

    `vix_level` overrides the latest VIX close if you have a more recent
    intraday print. `vix_history` is a Series of historical VIX closes
    used for percentile rank.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    klines = klines or {}
    snapshot = FeatureSnapshot(as_of=as_of, raw_klines=klines)
    feats: dict[str, float] = {}
    freshness: dict[str, float] = {}
    missing: list[str] = []
    stale: list[str] = []

    # ----- SPY-anchored equity features -----
    spy_close = _close(klines.get("US.SPY"))
    if spy_close is not None and len(spy_close) > WIN_LONG:
        feats["spy_ret_5"] = _ret(spy_close, WIN_SHORT)
        feats["spy_ret_20"] = _ret(spy_close, WIN_MID)
        feats["spy_ret_60"] = _ret(spy_close, WIN_LONG)
        # Binary forms kept in the JSONB for inspection / backward-compat,
        # but the HMM consumes the *continuous* `_dist_from_ma` versions
        # below (see HMM_FEATURE_ORDER). Binary features broke EM
        # convergence in v1/v2/v3 backtests.
        feats["spy_above_50dma"] = _above_ma(spy_close, 50)
        feats["spy_above_200dma"] = _above_ma(spy_close, 200) if len(spy_close) >= 200 else 0.0
        feats["spy_dist_50dma"] = _dist_from_ma(spy_close, 50)
        feats["spy_dist_200dma"] = _dist_from_ma(spy_close, 200) if len(spy_close) >= 200 else 0.0
        feats["spy_rv_20"] = _realized_vol(spy_close, WIN_MID)
        feats["spy_rv_60"] = _realized_vol(spy_close, WIN_LONG)
    else:
        for k in (
            "spy_ret_5", "spy_ret_20", "spy_ret_60",
            "spy_above_50dma", "spy_above_200dma",
            "spy_dist_50dma", "spy_dist_200dma",
            "spy_rv_20", "spy_rv_60",
        ):
            feats[k] = 0.0
        missing.append("US.SPY")

    # ----- VIX features (proxied via spy_rv_20 + VIXY momentum) -----
    # Moomoo doesn't expose ^VIX. We use SPY's 20-day realized vol scaled
    # to VIX-equivalent (typically VIX ≈ RV*100*1.1 — IV is slightly above
    # historical RV). This is more reliable than VIXY's price level (which
    # has compounding decay) but less reliable than the real VIX feed.
    # When Phase 2.5 wires FRED, this gets replaced with the actual VIX close.
    vix_series = _close(klines.get(VIX_TICKER))
    spy_rv_20 = feats.get("spy_rv_20", 0.0)

    if vix_level is None:
        if spy_rv_20 > 0:
            # Calibrate: long-run VIX ≈ 1.1 × annualized RV
            vix_level = float(spy_rv_20 * 100 * 1.1)
        else:
            vix_level = 18.0  # neutral fallback

    feats["vix_level"] = float(vix_level)

    if vix_series is not None and len(vix_series) > WIN_SHORT:
        # VIXY's daily % change is a clean signal of VIX direction (avoids
        # the level-decay issue). Multiply by typical VIX/VIXY beta ~10pts.
        vixy_pct_chg_5 = (
            float(vix_series.iloc[-1] / vix_series.iloc[-(WIN_SHORT + 1)] - 1.0)
            if vix_series.iloc[-(WIN_SHORT + 1)]
            else 0.0
        )
        feats["vix_chg_5"] = vixy_pct_chg_5 * 10.0  # ≈ VIX-point change
        feats["vix_pctl_252"] = _percentile(
            vix_series.iloc[-1], vix_series.iloc[-WIN_VIX_PCTL:]
        )
    else:
        feats["vix_chg_5"] = 0.0
        feats["vix_pctl_252"] = 0.5

    # ----- Breadth proxy: % of watchlist (sectors) above their 20DMA -----
    if SECTOR_ETFS:
        n_total = 0
        n_above = 0
        for s in SECTOR_ETFS:
            ser = _close(klines.get(s))
            if ser is None or len(ser) < 20:
                continue
            n_total += 1
            if ser.iloc[-1] > ser.iloc[-20:].mean():
                n_above += 1
        feats["breadth_above_20dma_pct"] = (
            (n_above / n_total) if n_total > 0 else 0.5
        )
        if n_total < 5:
            stale.append("breadth_proxy")
    else:
        feats["breadth_above_20dma_pct"] = 0.5

    # ----- Cross-asset 20d momentum -----
    for tkr, name in [
        ("US.TLT", "tlt_mom_20"),
        ("US.HYG", "hyg_mom_20"),
        ("US.GLD", "gld_mom_20"),
        ("US.UUP", "uup_mom_20"),
    ]:
        ser = _close(klines.get(tkr))
        feats[name] = _ret(ser, WIN_MID) if ser is not None else 0.0

    # ----- Correlation stress -----
    series_for_corr = {
        k: _close(klines.get(k))
        for k in EQUITY_BENCHMARKS + SECTOR_ETFS
        if klines.get(k) is not None
    }
    series_for_corr = {k: v for k, v in series_for_corr.items() if v is not None}
    feats["avg_pairwise_corr_20"] = _avg_pairwise_corr(series_for_corr, WIN_MID)

    # ----- Rates slope proxy: TLT/SHY ratio change as a coarse 10y-2y signal -----
    tlt = _close(klines.get("US.TLT"))
    if tlt is not None and len(tlt) > WIN_MID:
        feats["yield_10y2y_slope_proxy"] = _ret(tlt, WIN_MID)  # TLT up == long-end down == flatter
    else:
        feats["yield_10y2y_slope_proxy"] = 0.0

    # ----- Data quality assessment -----
    degradation = 0
    critical_missing = {"US.SPY", "US.VIX"} & set(missing)
    if critical_missing:
        degradation = 2
    elif missing or stale:
        degradation = 1

    snapshot.features = feats
    snapshot.source_freshness = freshness
    snapshot.data_quality = {
        "missing": missing,
        "stale": stale,
        "outlier": [],
        "degradation_level": degradation,
    }
    return snapshot


# ---------------------------------------------------------------------------
# Internal: extract close series from various OHLCV layouts
# ---------------------------------------------------------------------------


def _close(df: pd.DataFrame | None) -> pd.Series | None:
    """Extract a close-price series from common OHLCV column names."""
    if df is None or len(df) == 0:
        return None
    for col in ("close", "Close", "CLOSE", "closing", "c", "Adj Close", "adj_close"):
        if col in df.columns:
            return df[col].astype(float)
    # fallback: last numeric column
    for col in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].astype(float)
    return None
