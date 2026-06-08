"""Portfolio snapshot for the Active Risk Agent.

Aggregates the data the risk agent needs into a single dataclass that's
cheap to compute (≤200ms typically) and can be serialized to Postgres
risk_snapshots in one shot.

Inputs come from:
  - moomoo OpenD: account equity, open positions, current quotes, Greeks
  - regime cache: latest regime_states row
  - cached klines: for correlation / factor exposure math
  - journal: thesis stop levels for heat-to-stop calc
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """One open position with everything risk needs for decisions."""

    symbol: str  # e.g. US.SPY260530C00720000
    underlying: str  # e.g. US.SPY
    asset_type: str  # STK | OPT
    side: str  # BUY (long) | SELL (short)
    qty: float
    entry_price: float
    mark: float
    stop: float | None
    target: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    iv: float | None
    notional: float
    unrealized_pnl: float
    thesis_id: int | None
    sector: str | None  # GICS sector lookup from data/sectors.csv
    strategy_label: str | None
    age_minutes: float


@dataclass
class PortfolioSnapshot:
    """Output of build_snapshot — everything the risk agent reads."""

    as_of: datetime
    equity: float
    cash: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    aggregate_greeks: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    heat_metrics: dict[str, float] = field(default_factory=dict)
    liquidity_warnings: list[str] = field(default_factory=list)
    data_quality: dict[str, Any] = field(default_factory=dict)

    @property
    def n_open(self) -> int:
        return len(self.open_positions)

    @property
    def heat_pct(self) -> float:
        """Portfolio heat-to-stop as fraction of equity. 0.06 = 6%."""
        return float(self.heat_metrics.get("heat_pct", 0.0))

    def to_db_payload(self) -> dict[str, Any]:
        """Plain dict suitable for risk_snapshots jsonb columns."""
        return {
            "open_positions": [_pos_to_dict(p) for p in self.open_positions],
            "aggregate_greeks": self.aggregate_greeks,
            "factor_exposures": self.factor_exposures,
            "correlations": self.correlations,
            "heat_metrics": self.heat_metrics,
            "liquidity_warnings": self.liquidity_warnings,
            "data_quality": self.data_quality,
        }


def _pos_to_dict(p: OpenPosition) -> dict[str, Any]:
    return {k: getattr(p, k) for k in p.__dataclass_fields__}


def build_snapshot(
    equity: float,
    cash: float,
    open_positions: list[OpenPosition],
    *,
    daily_returns: dict[str, list[float]] | None = None,
    sector_lookup: dict[str, str] | None = None,
    as_of: datetime | None = None,
) -> PortfolioSnapshot:
    """Compute the full portfolio snapshot.

    `daily_returns` is {underlying -> last 60 daily returns} used for the
    correlation matrix. Caller pulls this from cached klines.

    `sector_lookup` is {underlying -> GICS sector} from data/sectors.csv.
    Used for factor exposure aggregation.
    """
    snap = PortfolioSnapshot(
        as_of=as_of or datetime.now(timezone.utc),
        equity=float(equity),
        cash=float(cash),
        open_positions=open_positions,
    )

    if not open_positions:
        snap.aggregate_greeks = _zero_greeks()
        snap.factor_exposures = {}
        snap.correlations = {}
        snap.heat_metrics = {"heat_pct": 0.0, "max_per_underlying_pct": 0.0}
        snap.data_quality = {
            "missing": [], "stale": [], "outlier": [], "degradation_level": 0,
        }
        return snap

    # ---------- Aggregate Greeks ----------
    snap.aggregate_greeks = _aggregate_greeks(open_positions, equity)

    # ---------- Factor exposures ----------
    snap.factor_exposures = _factor_exposures(
        open_positions, equity, sector_lookup or {}
    )

    # ---------- Correlation matrix ----------
    if daily_returns:
        snap.correlations = _correlation_matrix(daily_returns)

    # ---------- Heat-to-stop ----------
    snap.heat_metrics = _heat_metrics(open_positions, equity)

    # ---------- Liquidity warnings ----------
    snap.liquidity_warnings = _liquidity_warnings(open_positions)

    # ---------- Data quality ----------
    snap.data_quality = _assess_data_quality(open_positions)

    return snap


# -----------------------------------------------------------------------------
# Aggregate Greeks
# -----------------------------------------------------------------------------


def _zero_greeks() -> dict[str, float]:
    return {
        "net_delta": 0.0,
        "net_gamma": 0.0,
        "net_vega": 0.0,
        "net_theta": 0.0,
        "delta_dollar": 0.0,
        "vega_10vol_pnl": 0.0,
        "gamma_2sigma_pnl": 0.0,
    }


def _aggregate_greeks(
    positions: list[OpenPosition], equity: float
) -> dict[str, float]:
    """Sum per-position Greeks into portfolio-level totals + stress estimates."""
    g = _zero_greeks()
    for p in positions:
        sign = 1.0 if p.side == "BUY" else -1.0
        contracts_or_shares = sign * p.qty
        multiplier = 100 if p.asset_type == "OPT" else 1

        if p.delta is not None:
            g["net_delta"] += p.delta * contracts_or_shares * multiplier
            g["delta_dollar"] += (
                p.delta * contracts_or_shares * multiplier * p.mark
            )
        if p.gamma is not None:
            g["net_gamma"] += p.gamma * contracts_or_shares * multiplier
        if p.vega is not None:
            g["net_vega"] += p.vega * contracts_or_shares * multiplier
        if p.theta is not None:
            g["net_theta"] += p.theta * contracts_or_shares * multiplier

    # Stress estimates — dollar P&L impact of canonical shocks
    # Vega ±10 vol points: vega is per 1% IV change, so ×10 for 10pt move.
    g["vega_10vol_pnl"] = abs(g["net_vega"]) * 10
    # Gamma 2σ: assume ~2% underlying move; gamma is per $1 move → ×($price×0.02)²/2.
    # Hard to compute portfolio-wide without per-underlying breakdown; approximate.
    g["gamma_2sigma_pnl"] = abs(g["net_gamma"]) * (300 * 0.02) ** 2 / 2  # SPY-ish proxy

    # Express stress as % equity for easy comparison to guardrail caps
    if equity > 0:
        g["vega_10vol_pct_equity"] = g["vega_10vol_pnl"] / equity
        g["gamma_2sigma_pct_equity"] = g["gamma_2sigma_pnl"] / equity
        g["delta_dollar_pct_equity"] = g["delta_dollar"] / equity
    return g


# -----------------------------------------------------------------------------
# Factor exposures — coarse sector + rates buckets
# -----------------------------------------------------------------------------


SECTOR_TO_FACTOR = {
    "Information Technology": "tech",
    "Communication Services": "tech",
    "Consumer Discretionary": "consumer_cyclical",
    "Consumer Staples": "defensive",
    "Health Care": "defensive",
    "Utilities": "defensive",
    "Real Estate": "rates_sensitive",
    "Financials": "rates_sensitive",
    "Energy": "commodity",
    "Materials": "commodity",
    "Industrials": "cyclical",
}


def _factor_exposures(
    positions: list[OpenPosition], equity: float, sector_lookup: dict[str, str]
) -> dict[str, float]:
    """Sum dollar-delta exposure per factor bucket, normalized by equity."""
    exposures: dict[str, float] = {}
    for p in positions:
        sector = p.sector or sector_lookup.get(p.underlying, "Unknown")
        factor = SECTOR_TO_FACTOR.get(sector, "other")
        sign = 1.0 if p.side == "BUY" else -1.0
        notional = (
            (p.delta or 1.0) * sign * p.qty * (100 if p.asset_type == "OPT" else 1) * p.mark
        )
        exposures[factor] = exposures.get(factor, 0.0) + notional

    if equity > 0:
        return {k: v / equity for k, v in exposures.items()}
    return exposures


# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------


def _correlation_matrix(
    daily_returns: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation across underlyings."""
    syms = list(daily_returns.keys())
    out: dict[str, dict[str, float]] = {s: {} for s in syms}
    for i, a in enumerate(syms):
        out[a][a] = 1.0
        for b in syms[i + 1:]:
            ra, rb = daily_returns[a], daily_returns[b]
            n = min(len(ra), len(rb))
            if n < 5:
                out[a][b] = 0.0
                out[b][a] = 0.0
                continue
            c = _pearson(ra[-n:], rb[-n:])
            out[a][b] = c
            out[b][a] = c
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


# -----------------------------------------------------------------------------
# Heat-to-stop
# -----------------------------------------------------------------------------


def _heat_metrics(
    positions: list[OpenPosition], equity: float
) -> dict[str, float]:
    """Heat = aggregate $ at risk (entry → stop or option premium) / equity."""
    total_at_risk = 0.0
    per_underlying: dict[str, float] = {}

    for p in positions:
        if p.asset_type == "OPT":
            # Long premium: max loss = entry × contracts × 100
            at_risk = p.entry_price * p.qty * 100
        else:
            # Stock: at_risk = (entry - stop) × qty
            if p.stop and p.stop > 0 and p.stop < p.entry_price:
                at_risk = (p.entry_price - p.stop) * p.qty
            else:
                at_risk = p.entry_price * p.qty * 0.05  # 5% implicit stop

        if p.side == "BUY":
            total_at_risk += at_risk
            per_underlying[p.underlying] = per_underlying.get(p.underlying, 0.0) + at_risk

    heat_pct = total_at_risk / equity if equity > 0 else 0.0
    max_per_underlying_pct = (
        max(per_underlying.values()) / equity if per_underlying and equity > 0 else 0.0
    )

    return {
        "heat_pct": heat_pct,
        "max_per_underlying_pct": max_per_underlying_pct,
        "total_at_risk_usd": total_at_risk,
        "per_underlying_usd": per_underlying,
    }


# -----------------------------------------------------------------------------
# Liquidity + data quality
# -----------------------------------------------------------------------------


def _liquidity_warnings(positions: list[OpenPosition]) -> list[str]:
    warns = []
    for p in positions:
        if p.asset_type == "OPT" and p.iv is None:
            warns.append(f"{p.symbol}: missing IV")
        if p.asset_type == "OPT" and p.delta is None:
            warns.append(f"{p.symbol}: missing delta")
    return warns


def _assess_data_quality(positions: list[OpenPosition]) -> dict[str, Any]:
    missing = []
    stale = []
    for p in positions:
        if p.asset_type == "OPT" and (p.delta is None or p.gamma is None):
            missing.append(f"{p.symbol}_greeks")
        if p.age_minutes > 60:
            stale.append(f"{p.symbol}_stale_quote")
    return {
        "missing": missing,
        "stale": stale,
        "outlier": [],
        # Held-position greek/quote gaps describe our ability to *manage
        # existing* positions — NOT whether market data for a *new* candidate
        # is trustworthy. Keep them a WARNING (level 1) so they never DEFER new
        # entries; otherwise a single un-greek'd holding freezes the entire
        # entry engine (root cause of the 2026-06 candidate_entry lockup:
        # candidate_entry_graph never runs refresh_quotes_and_greeks, so any
        # held option shows missing greeks -> level 2 -> DEFER every new trade).
        # Market/candidate blindness is gated separately via the regime path.
        "degradation_level": 1 if (missing or stale) else 0,
    }
