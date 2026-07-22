"""Calibrate the execution-cost model from live option-chain quotes.

Measures quoted bid/ask spreads for the kind of contracts the system actually
trades (14–60 DTE, |delta| 0.25–0.65 — the R5 windows) across a basket of
underlyings, then writes ``data/execution_costs.json`` which
``trading_agent.execution_costs`` reads over its conservative defaults.

Requires MoomooOpenD running. Re-run periodically (spreads are regime-
dependent) — the output file records the calibration timestamp and sample.

Usage:
    .venv/bin/python scripts/calibrate_execution_costs.py \
        --tickers AAPL QQQ SPY MRVL GLD [--write]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path

DTE_MIN, DTE_MAX = 14, 60
DELTA_MIN, DELTA_MAX = 0.25, 0.65


def _collect_spreads(tickers: list[str]) -> list[dict]:
    """Compose list_option_expiries → get_option_chain → get_quote (which
    populates bid/ask + greeks for option codes) per the moomoo server docs."""
    from trading_agent.mcp_servers.moomoo.server import (
        get_option_chain,
        get_quote,
        list_option_expiries,
    )
    samples: list[dict] = []
    today = date.today()
    for t in tickers:
        code = f"US.{t}"
        try:
            spot_rows = get_quote([code]).get("rows") or []
            spot = float(spot_rows[0].get("last_price") or 0) if spot_rows else 0.0
            expiries = []
            for row in (list_option_expiries(code).get("rows") or []):
                st = str(row.get("strike_time") or "")[:10]
                try:
                    dte = (date.fromisoformat(st) - today).days
                except ValueError:
                    continue
                if DTE_MIN <= dte <= DTE_MAX:
                    expiries.append(st)
        except Exception as e:
            print(f"  {t}: expiry/spot fetch failed ({e}) — skipped", file=sys.stderr)
            continue
        if spot <= 0:
            print(f"  {t}: no spot price — skipped", file=sys.stderr)
            continue
        for exp in expiries[:2]:  # two expiries per name keeps this fast
            try:
                chain = get_option_chain(code, exp).get("rows") or []
            except Exception as e:
                print(f"  {t} {exp}: chain failed ({e})", file=sys.stderr)
                continue
            # ~8 strikes nearest spot per side — the band the system trades.
            chain = [c for c in chain if c.get("code") and c.get("strike_price")]
            chain.sort(key=lambda c: abs(float(c["strike_price"]) - spot))
            codes = [c["code"] for c in chain[:16]]
            if not codes:
                continue
            try:
                quotes = get_quote(codes).get("rows") or []
            except Exception as e:
                print(f"  {t} {exp}: option quotes failed ({e})", file=sys.stderr)
                continue
            for r in quotes:
                try:
                    bid = float(r.get("bid_price") or r.get("bid") or 0)
                    ask = float(r.get("ask_price") or r.get("ask") or 0)
                    # Real moomoo get_market_snapshot rows carry the greek as
                    # ``option_delta``; ``delta`` is the legacy fallback. (The
                    # 2026-06 global calibration ran with the dead ``delta``
                    # read, so its delta gate never fired — the strike
                    # prefilter alone bounded that sample.)
                    delta = abs(float(r.get("option_delta") or r.get("delta") or 0))
                except (TypeError, ValueError):
                    continue
                if bid <= 0 or ask <= bid:
                    continue
                if delta and not (DELTA_MIN <= delta <= DELTA_MAX):
                    continue
                mid = (bid + ask) / 2
                samples.append({
                    "ticker": t, "expiry": exp,
                    "spread_pct_of_mid": (ask - bid) / mid,
                    "mid": mid,
                })
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "QQQ", "SPY", "MRVL", "GLD"])
    p.add_argument("--write", action="store_true",
                   help="write data/execution_costs.json (default: report only)")
    args = p.parse_args()

    print(f"sampling chains: {', '.join(args.tickers)} "
          f"(DTE {DTE_MIN}-{DTE_MAX}, |delta| {DELTA_MIN}-{DELTA_MAX})")
    samples = _collect_spreads(args.tickers)
    if len(samples) < 20:
        print(f"only {len(samples)} usable quotes — OpenD down or market "
              "closed? Not calibrating; defaults stay in force.")
        return 1

    spreads = sorted(s["spread_pct_of_mid"] for s in samples)
    med = statistics.median(spreads)
    p75 = spreads[int(0.75 * (len(spreads) - 1))]
    print(f"n={len(samples)}  spread/mid: median={med:.1%}  p75={p75:.1%}  "
          f"min={spreads[0]:.1%}  max={spreads[-1]:.1%}")

    # Half-spread at the MEDIAN quoted spread; the p75 figure is reported so
    # the operator can see how heavy the tail is.
    payload = {
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "tickers": args.tickers,
        "spread_pct_of_mid_median": round(med, 5),
        "spread_pct_of_mid_p75": round(p75, 5),
        "opt_half_spread_pct_of_mark": round(med / 2, 5),
    }
    print(json.dumps(payload, indent=2))

    if args.write:
        from trading_agent.config import CONFIG
        out = Path(CONFIG.data_dir) / "execution_costs.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"written: {out}")
    else:
        print("(report only — pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
