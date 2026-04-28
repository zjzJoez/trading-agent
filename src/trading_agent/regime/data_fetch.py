"""Pull historical klines from moomoo OpenD for regime feature computation.

Uses the Phase-1 moomoo-api Python SDK. Connects to OpenD on
127.0.0.1:11111 (the systemd unit). Returns a dict {ticker -> DataFrame}
suitable for `build_feature_snapshot(klines=...)`.

Caches per-ticker per-day in `data/regime/klines_cache/<ticker>_<date>.parquet`
to avoid hitting the broker for the same data twice.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pickle

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / "trading-agent" / "data" / "regime" / "klines_cache"
CACHE_SUFFIX = ".pkl"   # pickle = no extra deps; switch to parquet later if needed

# How far back to pull for each ticker. Long enough to support 252-day VIX
# percentile + 200DMA + 60-day RV calculations.
DEFAULT_LOOKBACK_DAYS = 380


def fetch_klines(
    tickers: Iterable[str],
    *,
    end_date: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    host: str = "127.0.0.1",
    port: int = 11111,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch daily klines via moomoo. Returns {ticker -> DataFrame}.

    Falls back to disk cache when OpenD is unavailable. Each DataFrame
    has at least {date_index, close, volume} columns.
    """
    from moomoo import KLType, OpenQuoteContext, RET_OK

    end_date = end_date or datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    out: dict[str, pd.DataFrame] = {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ctx = None
    try:
        ctx = OpenQuoteContext(host=host, port=port)
        for tkr in tickers:
            cache_path = (
                CACHE_DIR / f"{tkr.replace('.', '_').replace('^', 'IDX_')}_{end_date.isoformat()}{CACHE_SUFFIX}"
            )

            if use_cache and cache_path.is_file():
                try:
                    with cache_path.open("rb") as f:
                        out[tkr] = pickle.load(f)
                    log.debug("[regime/data] cache hit %s", tkr)
                    continue
                except Exception as e:
                    log.warning("Cache read failed for %s: %s", tkr, e)

            try:
                ret, df, _ = ctx.request_history_kline(
                    tkr,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    ktype=KLType.K_DAY,
                    autype=None,
                    fields=None,
                    max_count=lookback_days + 5,
                )
            except Exception as e:
                log.warning("OpenD fetch failed for %s: %s", tkr, e)
                continue

            if ret != RET_OK or df is None or len(df) == 0:
                log.warning("No data for %s (ret=%s)", tkr, ret)
                continue

            # Normalize: index by date, expose 'close' / 'volume' columns
            df = df.copy()
            if "time_key" in df.columns:
                df["date"] = pd.to_datetime(df["time_key"]).dt.date
                df = df.set_index("date")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.set_index("date")
            # Coerce numeric columns
            for col in ("close", "volume", "high", "low", "open"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            out[tkr] = df

            try:
                with cache_path.open("wb") as f:
                    pickle.dump(df, f)
            except Exception as e:
                log.warning("Cache write failed for %s: %s", tkr, e)
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass

    log.info(
        "[regime/data] fetched %d/%d tickers (end=%s, lookback=%d)",
        len(out), len(list(tickers)), end_date, lookback_days,
    )
    return out


def fetch_for_features(end_date: date | None = None) -> dict[str, pd.DataFrame]:
    """Convenience: fetch all tickers used in regime features."""
    from trading_agent.regime.features import ALL_TICKERS
    return fetch_klines(ALL_TICKERS, end_date=end_date)
