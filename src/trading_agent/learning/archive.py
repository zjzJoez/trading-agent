"""Phase 2.6.5 — Diversity Archive (QuantEvolve cells).

Maintains a portfolio of bounded parameter candidates partitioned by
*behavior cells* — `(regime, strategy_label, DTE_bucket, IV_bucket)` — with
one champion per cell scored by `sharpe − λ · max_drawdown_R` (λ = 0.5).

Why per-cell instead of one global champion?  Because a parameter set that
wins in `BULL_TREND × earnings_iv_play × 0-7 DTE × high_IV` is almost
certainly NOT the same one that wins in `BEAR_TREND × trend_continuation ×
30-60 DTE × mid_IV`.  The archive lets the weekly_learning_graph (Phase 2.7.5
LLM Critic) propose context-specific candidates instead of a single global
mutation.

Storage: one row per `(cell_key, param_version_id)` in `param_archive_cells`.
Aggregates are updated *online* on each closed-trade enrichment — no
re-scanning of history required.  The `is_champion` flag is updated
atomically via a single transaction in `update_champion`.

Phase 2.6.5 keeps the math simple:
    sharpe        ≈ mean(R) / std(R) × sqrt(N)
    composite     = sharpe − λ · max_drawdown_R
    min_trades_for_champion = 3

The Information Ratio term is intentionally absent at cell granularity —
with low n it dominates noise.  Global IR returns in the weekly_learning
report (Phase 2.7.5).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

LAMBDA_MDD = 0.5
MIN_TRADES_FOR_CHAMPION = 3


# ---------------------------------------------------------------------------
# Cell taxonomy
# ---------------------------------------------------------------------------

DTE_BUCKETS: list[tuple[int, int, str]] = [
    (0, 7, "0-7"),
    (7, 14, "7-14"),
    (14, 30, "14-30"),
    (30, 60, "30-60"),
    (60, 365, "60+"),
]

IV_BUCKETS: list[tuple[float, float, str]] = [
    (0.0, 0.25, "low"),       # IV < 25% absolute
    (0.25, 0.40, "mid"),      # 25–40%
    (0.40, 10.0, "high"),     # > 40%
]

NA = "na"


@dataclass(frozen=True)
class CellKey:
    """A cell's coordinates.  Composes to a deterministic textual key."""

    regime: str
    strategy: str
    dte_bucket: str
    iv_bucket: str

    def to_text(self) -> str:
        return f"{self.regime}|{self.strategy}|{self.dte_bucket}|{self.iv_bucket}"

    @classmethod
    def from_text(cls, txt: str) -> "CellKey":
        parts = txt.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"malformed cell_key: {txt!r}")
        return cls(*parts)


def bucket_dte(dte: int | float | None) -> str:
    if dte is None:
        return NA
    try:
        d = int(dte)
    except (TypeError, ValueError):
        return NA
    if d < 0:
        return NA
    for lo, hi, label in DTE_BUCKETS:
        if lo <= d < hi:
            return label
    return DTE_BUCKETS[-1][2]


def bucket_iv(iv: float | None) -> str:
    if iv is None:
        return NA
    try:
        v = float(iv)
    except (TypeError, ValueError):
        return NA
    if v < 0 or not math.isfinite(v):
        return NA
    for lo, hi, label in IV_BUCKETS:
        if lo <= v < hi:
            return label
    return IV_BUCKETS[-1][2]


def cell_for_trade(
    *,
    regime_label: str | None,
    strategy_label: str | None,
    option_dte: int | float | None,
    option_iv: float | None,
) -> CellKey:
    """Compute cell coordinates from a closed trade's audit fields.

    All inputs may be ``None``: missing fields collapse to the ``na`` bucket
    so unattributable trades still land in a stable cell instead of being
    silently dropped.
    """
    return CellKey(
        regime=str(regime_label or NA),
        strategy=str(strategy_label or NA),
        dte_bucket=bucket_dte(option_dte),
        iv_bucket=bucket_iv(option_iv),
    )


# ---------------------------------------------------------------------------
# Aggregate math (pure)
# ---------------------------------------------------------------------------

def _composite_from_running(
    n: int, cum_R: float, sum_R_sq: float, mdd: float
) -> float | None:
    """Compute `sharpe − λ·MDD` from running aggregates.

    Returns ``None`` until at least 2 trades exist (variance undefined).
    """
    if n < 2:
        return None
    mean = cum_R / n
    var = (sum_R_sq / n) - (mean * mean)
    if var <= 0:
        # Zero variance — can't compute SR; degrade to mean-only proxy
        return float(mean - LAMBDA_MDD * mdd)
    sd = math.sqrt(var)
    sharpe = (mean / sd) * math.sqrt(n)
    return float(sharpe - LAMBDA_MDD * mdd)


def _next_drawdown(prev_peak: float, prev_mdd: float, new_cum: float) -> tuple[float, float]:
    """Update peak and max-drawdown after appending a trade with new cum_R."""
    peak = max(prev_peak, new_cum)
    drop = peak - new_cum
    mdd = max(prev_mdd, drop)
    return peak, mdd


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@dataclass
class ArchiveRow:
    cell_key: str
    param_version_id: int
    n_trades: int
    cum_R: float
    sum_R_sq: float
    peak_cum_R: float
    max_drawdown_R: float
    composite_score: float | None
    is_champion: bool


def get_champion(cell_key: str) -> int | None:
    """Return the param_version_id of the champion in this cell, or None."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT param_version_id FROM param_archive_cells
                WHERE cell_key = %s AND is_champion = TRUE
                LIMIT 1
                """,
                (cell_key,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None
    except Exception as e:
        log.warning("get_champion(%s) failed: %s", cell_key, e)
        return None


def archive_summary(limit: int = 200) -> list[dict[str, Any]]:
    """List cell rows ordered by composite_score desc, for diagnostics."""
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT cell_key, param_version_id, n_trades, cum_R,
                       max_drawdown_R, composite_score, is_champion,
                       last_trade_at
                FROM param_archive_cells
                ORDER BY composite_score DESC NULLS LAST, n_trades DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("archive_summary failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        ck, pvid, n, cR, mdd, score, champ, last_at = r
        out.append({
            "cell_key": ck,
            "param_version_id": int(pvid),
            "n_trades": int(n),
            "cum_R": float(cR),
            "max_drawdown_R": float(mdd),
            "composite_score": float(score) if score is not None else None,
            "is_champion": bool(champ),
            "last_trade_at": last_at,
        })
    return out


def update_champion(
    cell: CellKey,
    param_version_id: int,
    realized_R: float,
    *,
    trade_at: datetime | None = None,
) -> dict[str, Any]:
    """Append one realized R to (cell, param_version) and re-elect champion.

    Atomic via a single transaction:
      1. UPSERT the row with updated running aggregates.
      2. Recompute composite_score for the affected row.
      3. If new score > current champion AND n_trades ≥ MIN_TRADES_FOR_CHAMPION,
         demote old champion + promote this row.

    Returns a small dict describing the outcome (for logging/audit).  Never
    raises — DB issues are logged and the dict reports the error.
    """
    cell_key = cell.to_text()
    result: dict[str, Any] = {
        "cell_key": cell_key,
        "param_version_id": int(param_version_id),
        "realized_R": float(realized_R),
        "champion_changed": False,
        "promoted": False,
        "demoted_from": None,
        "score_before": None,
        "score_after": None,
        "n_trades": 0,
    }
    try:
        with cursor() as cur:
            # 1. Lock the row for update — or insert a new one
            cur.execute(
                """
                SELECT id, n_trades, cum_R, sum_R_sq, peak_cum_R,
                       max_drawdown_R, composite_score
                FROM param_archive_cells
                WHERE cell_key = %s AND param_version_id = %s
                FOR UPDATE
                """,
                (cell_key, param_version_id),
            )
            row = cur.fetchone()

            if row is None:
                row_id = None
                n = 0
                cum_R = 0.0
                sum_R_sq = 0.0
                peak = 0.0
                mdd = 0.0
                old_score = None
            else:
                row_id, n, cum_R, sum_R_sq, peak, mdd, old_score = row
                n = int(n); cum_R = float(cum_R); sum_R_sq = float(sum_R_sq)
                peak = float(peak); mdd = float(mdd)
                old_score = float(old_score) if old_score is not None else None

            # 2. Update aggregates with the new realized R
            n_new = n + 1
            cum_new = cum_R + float(realized_R)
            sum_sq_new = sum_R_sq + float(realized_R) ** 2
            peak_new, mdd_new = _next_drawdown(peak, mdd, cum_new)
            score_new = _composite_from_running(n_new, cum_new, sum_sq_new, mdd_new)

            result["score_before"] = old_score
            result["score_after"] = score_new
            result["n_trades"] = n_new

            now = trade_at  # may be None — DB default fills last_updated_at

            if row_id is None:
                cur.execute(
                    """
                    INSERT INTO param_archive_cells
                        (cell_key, cell_regime, cell_strategy,
                         cell_dte_bucket, cell_iv_bucket,
                         param_version_id,
                         n_trades, cum_R, sum_R_sq,
                         peak_cum_R, max_drawdown_R, composite_score,
                         is_champion, first_trade_at, last_trade_at)
                    VALUES (%s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            FALSE, %s, %s)
                    RETURNING id
                    """,
                    (
                        cell_key, cell.regime, cell.strategy,
                        cell.dte_bucket, cell.iv_bucket,
                        param_version_id,
                        n_new, cum_new, sum_sq_new,
                        peak_new, mdd_new, score_new,
                        now, now,
                    ),
                )
                row_id = int(cur.fetchone()[0])
            else:
                cur.execute(
                    """
                    UPDATE param_archive_cells
                    SET n_trades = %s, cum_R = %s, sum_R_sq = %s,
                        peak_cum_R = %s, max_drawdown_R = %s,
                        composite_score = %s,
                        last_trade_at = COALESCE(%s, last_trade_at),
                        first_trade_at = COALESCE(first_trade_at, %s),
                        last_updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        n_new, cum_new, sum_sq_new,
                        peak_new, mdd_new, score_new,
                        now, now, row_id,
                    ),
                )

            # 3. Champion election — only consider once we have enough samples
            if n_new < MIN_TRADES_FOR_CHAMPION or score_new is None:
                return result

            cur.execute(
                """
                SELECT id, param_version_id, composite_score
                FROM param_archive_cells
                WHERE cell_key = %s AND is_champion = TRUE
                """,
                (cell_key,),
            )
            champ = cur.fetchone()

            if champ is None:
                # No champion yet → this row becomes one
                cur.execute(
                    "UPDATE param_archive_cells SET is_champion = TRUE WHERE id = %s",
                    (row_id,),
                )
                result["champion_changed"] = True
                result["promoted"] = True
                return result

            champ_id, champ_pvid, champ_score = champ
            champ_score = float(champ_score) if champ_score is not None else float("-inf")
            if int(champ_id) == row_id:
                # We are already the champion — nothing to do
                return result
            if score_new > champ_score:
                # Demote old, promote new — both updates inside the same txn
                cur.execute(
                    "UPDATE param_archive_cells SET is_champion = FALSE WHERE id = %s",
                    (champ_id,),
                )
                cur.execute(
                    "UPDATE param_archive_cells SET is_champion = TRUE WHERE id = %s",
                    (row_id,),
                )
                result["champion_changed"] = True
                result["promoted"] = True
                result["demoted_from"] = int(champ_pvid)
            return result
    except Exception as e:
        log.warning(
            "update_champion(%s, pv=%s, R=%.4f) failed: %s",
            cell_key, param_version_id, realized_R, e,
        )
        result["error"] = str(e)[:200]
        return result


__all__ = [
    "CellKey",
    "DTE_BUCKETS",
    "IV_BUCKETS",
    "LAMBDA_MDD",
    "MIN_TRADES_FOR_CHAMPION",
    "ArchiveRow",
    "archive_summary",
    "bucket_dte",
    "bucket_iv",
    "cell_for_trade",
    "get_champion",
    "update_champion",
]
