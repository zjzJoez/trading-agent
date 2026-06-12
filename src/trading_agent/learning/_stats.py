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

``lcb_mean`` (Phase 2 item 10) is the expectancy statistic the promotion
gate keys on: a one-sided lower confidence bound on the mean realized R.
"""
from __future__ import annotations

import math
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


# One-sided Student-t critical values P(T > t) = 1 − confidence, indexed by
# df − 1 for df 1..30. Beyond df 30 the normal quantile is used — t_30 vs z
# differs by ~3% at 95% and the gap shrinks with n, a deliberately accepted
# approximation so the runtime image needs no scipy.
_T_ONE_SIDED: dict[float, tuple[float, ...]] = {
    0.95: (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833,
           1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734,
           1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703,
           1.701, 1.699, 1.697),
    0.99: (31.821, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821,
           2.764, 2.718, 2.681, 2.650, 2.624, 2.602, 2.583, 2.567, 2.552,
           2.539, 2.528, 2.518, 2.508, 2.500, 2.492, 2.485, 2.479, 2.473,
           2.467, 2.462, 2.457),
}
_Z_ONE_SIDED: dict[float, float] = {0.95: 1.6449, 0.99: 2.3263}


def lcb_mean(values: Iterable[float], confidence: float = 0.95) -> float | None:
    """One-sided lower confidence bound on the mean via Student-t.

    ``LB = mean − t_{confidence, n−1} · s / √n`` with s the sample standard
    deviation (ddof=1).

    Returns None for n < 2: a single sample has df = 0, the sample stdev is
    undefined, and there is NO defensible lower bound — callers must treat
    None as "no evidence", never as 0 (0 would let a one-trade canary look
    exactly breakeven). A constant series has s = 0 and returns the mean
    itself. Only the tabled confidences (0.95, 0.99) are supported;
    anything else raises rather than silently interpolating.
    """
    vs = [float(v) for v in values]
    n = len(vs)
    if n < 2:
        return None
    table = _T_ONE_SIDED.get(confidence)
    if table is None:
        raise ValueError(
            f"unsupported confidence {confidence}; "
            f"tabled: {sorted(_T_ONE_SIDED)}"
        )
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / (n - 1)
    df = n - 1
    t = table[df - 1] if df <= len(table) else _Z_ONE_SIDED[confidence]
    return mean - t * math.sqrt(var) / math.sqrt(n)


__all__ = ["lcb_mean", "running_drawdown_step", "series_max_drawdown"]
