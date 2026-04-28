"""Phase 2 learning — shared running-stats helpers.

Single source of truth for drawdown math used by ``archive`` (online,
per-cell), ``promote`` (per-canary aggregate over closed trades), and
``replay`` (offline counterfactual aggregate).  Codex review on
af480bd flagged the duplication.

Two complementary entry points:

* ``running_drawdown_step(prev_peak, prev_mdd, new_cum)`` — one-step online
  update for systems that hold the running ``(peak, mdd)`` pair in a row
  (archive's ``param_archive_cells``).

* ``series_max_drawdown(series)`` — full-series scan, used by replay /
  promote where realized R values are pulled in chronological order from
  Postgres.

Both implement the same definition: drawdown is peak-to-trough from
the cumulative-R curve, never negative, monotonically non-decreasing
within the window.
"""
from __future__ import annotations

from typing import Iterable


def running_drawdown_step(
    prev_peak: float, prev_mdd: float, new_cum: float
) -> tuple[float, float]:
    """Apply one new cumulative-R sample to the running ``(peak, mdd)``.

    Returns the updated pair.  Pure function — no DB reads.
    """
    peak = max(prev_peak, new_cum)
    mdd = max(prev_mdd, peak - new_cum)
    return peak, mdd


def series_max_drawdown(rs: Iterable[float]) -> tuple[float, float]:
    """Compute (cum_R, max_drawdown_R) over a chronological R series.

    Trades must be ordered oldest-first.  Returns (0, 0) on an empty
    iterable so callers can compose without branch-on-empty.
    """
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in rs:
        cum += r
        peak, mdd = running_drawdown_step(peak, mdd, cum)
    return cum, mdd


__all__ = ["running_drawdown_step", "series_max_drawdown"]
