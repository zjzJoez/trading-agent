"""M1-0.3 — per-underlying spread calibration for the index-vertical sleeve.

Samples live option quotes for the sleeve-1 zone (PUTS, 30-45 DTE, short-leg
|delta| 0.15-0.40 plus the further-OTM long-wing zone) on each requested
underlying, computes the per-name median quoted spread as a fraction of mid,
and writes it under the ``per_underlying`` key of ``data/execution_costs.json``
— the GLOBAL watchlist-calibrated figures in that file are left untouched.
``trading_agent.execution_costs.half_spread_cost(..., underlying=...)`` and
``friction_r(...)`` prefer these per-name figures over the global median.

Also prints the plan's IWM decision line: modeled ``friction_r`` for a $5-wide
vertical at the measured spread vs the 0.06R whitelist bar. REPORT ONLY — the
spec whitelist change is the operator's call.

Requires MoomooOpenD (run on EC2 where trading-agent-opend.service is live);
fails fast with a clear message when OpenD is unreachable.

Usage:
    .venv/bin/python scripts/calibrate_index_options.py \
        --underlyings SPY,QQQ,IWM --min-quotes 40 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path

# Sleeve-1 zone (REVIVAL_PLAN M1-0.3 / M1-1): puts only, 30-45 DTE. Delta
# window unions the short-leg band (0.15-0.40) with the further-OTM long
# wing (a $5-10 wide vertical's wing sits a few delta points below the
# short leg; 0.03 floors out sub-penny junk strikes).
DTE_MIN, DTE_MAX = 30, 45
DELTA_MIN, DELTA_MAX = 0.03, 0.40
# When the quote carries no delta, fall back to a strike prefilter: OTM puts
# from spot down to 20% below — generously covers both zones at 30-45 DTE.
STRIKE_MIN_FRAC_OF_SPOT = 0.80
QUOTE_BATCH = 16          # codes per get_quote call (matches the 2026-06 run)
MAX_CODES_PER_EXPIRY = 32
# futu OpenAPI limits get_option_chain to ~10 requests per 30 s. Cap expiries
# per name (4 × 32 strikes = 128 candidates, ample for --min-quotes 40) and
# space the chain calls so a default SPY,QQQ,IWM run never breaches the limit
# mid-run and starves the last name (IWM — the decision line) below
# --min-quotes. Tests set CHAIN_SLEEP_S to 0.
MAX_EXPIRIES_PER_NAME = 4
CHAIN_SLEEP_S = 3.5

# IWM decision-line parameters (M1-1 spec): $5-wide vertical at the gate
# floor credit = width/4; whitelist bar friction_r <= 0.06R.
DECISION_WIDTH = 5.0
DECISION_CREDIT = DECISION_WIDTH / 4.0
DECISION_BAR_R = 0.06

ZONE_DESC = (f"PUTS {DTE_MIN}-{DTE_MAX} DTE, |delta| {DELTA_MIN}-{DELTA_MAX} "
             f"(short-leg + long-wing zones)")


def _load_moomoo():
    """Import the in-repo moomoo quote functions. Separated for test mocking."""
    from trading_agent.mcp_servers.moomoo.server import (
        get_option_chain,
        get_quote,
        list_option_expiries,
    )
    return get_quote, list_option_expiries, get_option_chain


def collect_spreads(underlyings: list[str], quote_fns) -> dict[str, list[float]]:
    """Per-underlying list of quoted spread/mid samples inside the zone."""
    get_quote, list_option_expiries, get_option_chain = quote_fns
    out: dict[str, list[float]] = {}
    today = date.today()
    first_chain_call = True
    for t in underlyings:
        code = f"US.{t}"
        samples: list[float] = []
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
            print(f"  {t}: spot/expiry fetch failed ({e}) — skipped", file=sys.stderr)
            out[t] = samples
            continue
        if spot <= 0:
            print(f"  {t}: no spot price — skipped", file=sys.stderr)
            out[t] = samples
            continue
        for exp in expiries[:MAX_EXPIRIES_PER_NAME]:
            if not first_chain_call and CHAIN_SLEEP_S:
                time.sleep(CHAIN_SLEEP_S)  # futu chain-request rate limit
            first_chain_call = False
            try:
                chain = get_option_chain(code, exp, option_type="PUT").get("rows") or []
            except Exception as e:
                print(f"  {t} {exp}: chain failed ({e})", file=sys.stderr)
                continue
            strikes = []
            for c in chain:
                try:
                    k = float(c.get("strike_price") or 0)
                except (TypeError, ValueError):
                    continue
                if c.get("code") and STRIKE_MIN_FRAC_OF_SPOT * spot <= k <= spot:
                    strikes.append((k, c["code"]))
            strikes.sort(reverse=True)  # nearest-the-money first
            codes = [c for _, c in strikes[:MAX_CODES_PER_EXPIRY]]
            for i in range(0, len(codes), QUOTE_BATCH):
                try:
                    quotes = get_quote(codes[i:i + QUOTE_BATCH]).get("rows") or []
                except Exception as e:
                    print(f"  {t} {exp}: option quotes failed ({e})", file=sys.stderr)
                    continue
                for r in quotes:
                    try:
                        bid = float(r.get("bid_price") or r.get("bid") or 0)
                        ask = float(r.get("ask_price") or r.get("ask") or 0)
                        # Real moomoo get_market_snapshot rows carry the greek
                        # as ``option_delta``; ``delta`` is the legacy fallback.
                        delta = abs(float(r.get("option_delta") or r.get("delta") or 0))
                    except (TypeError, ValueError):
                        continue
                    if bid <= 0 or ask <= bid:
                        continue
                    # Delta gate only when delta is populated; otherwise the
                    # strike prefilter above already bounds the zone.
                    if delta and not (DELTA_MIN <= delta <= DELTA_MAX):
                        continue
                    samples.append((ask - bid) / ((bid + ask) / 2))
        out[t] = samples
    return out


def _decision_friction_r(median_spread_pct_mid: float) -> float:
    """friction_r for the decision-line vertical AT THE MEASURED spread.

    Mirrors ``execution_costs.friction_r`` (fees on 4 fills + half-spread on
    2 legs × 2 directions at the net_credit mark) but takes the freshly
    measured median directly, so the printed decision is identical with or
    without --dry-run. Cross-checked against ``execution_costs.friction_r``
    in tests/test_calibrate_index_options.py.
    """
    from trading_agent.execution_costs import fees_per_side
    fees = 4.0 * fees_per_side(1, "OPT")
    spread = 4.0 * (median_spread_pct_mid / 2.0) * DECISION_CREDIT * 100.0
    return (fees + spread) / ((DECISION_WIDTH - DECISION_CREDIT) * 100.0)


def _write_atomic(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".execution_costs.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _emit_calibration_event(per_underlying: dict) -> None:
    """Best-effort agent_event — only when a Postgres env is present, so a
    laptop dry-run never spawns the ~/agent_events.fallback.jsonl file."""
    has_pg = bool(os.environ.get("POSTGRES_DSN")) or (
        Path.home() / ".env.postgres").exists()
    if not has_pg:
        return
    try:
        from trading_agent.events import emit, new_run_id
        emit(run_id=new_run_id("calib"), trigger="manual",
             agent="calibrate_index_options", event_type="calibration_updated",
             payload={"per_underlying": per_underlying, "zone": ZONE_DESC})
    except Exception as e:
        print(f"agent_event emit failed (non-fatal): {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--underlyings", default="SPY,QQQ,IWM",
                   help="comma-separated, e.g. SPY,QQQ,IWM")
    p.add_argument("--min-quotes", type=int, default=40,
                   help="minimum usable quotes per underlying to calibrate it")
    p.add_argument("--dry-run", action="store_true",
                   help="report medians + decision line only; no file write")
    args = p.parse_args(argv)
    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]
    if not underlyings:
        print("no underlyings given", file=sys.stderr)
        return 1

    # --- fail fast when OpenD is unreachable (this runs on EC2 next to
    # trading-agent-opend.service; anywhere else it should die loudly, not
    # write a garbage calibration). ---------------------------------------
    from trading_agent.config import CONFIG
    try:
        quote_fns = _load_moomoo()
        preflight = quote_fns[0]([f"US.{underlyings[0]}"]).get("rows") or []
        if not preflight:
            raise RuntimeError("empty snapshot for preflight symbol")
    except Exception as e:
        print(
            f"FATAL: MoomooOpenD unreachable or not serving quotes at "
            f"{CONFIG.opend_host}:{CONFIG.opend_port} ({e}).\n"
            f"Run this on EC2 with trading-agent-opend.service active "
            f"during US market hours, then retry.",
            file=sys.stderr,
        )
        return 2

    print(f"sampling {', '.join(underlyings)} — zone: {ZONE_DESC}")
    by_name = collect_spreads(underlyings, quote_fns)

    now = datetime.now(UTC).isoformat()
    calibrated: dict[str, dict] = {}
    for t in underlyings:
        spreads = sorted(by_name.get(t) or [])
        if len(spreads) < args.min_quotes:
            print(f"  {t}: only {len(spreads)} usable quotes "
                  f"(< {args.min_quotes}) — NOT calibrated (market closed? "
                  f"illiquid chain?)")
            continue
        med = statistics.median(spreads)
        p75 = spreads[int(0.75 * (len(spreads) - 1))]
        print(f"  {t}: n={len(spreads)}  spread/mid median={med:.2%}  "
              f"p75={p75:.2%}  min={spreads[0]:.2%}  max={spreads[-1]:.2%}")
        calibrated[t] = {
            "median_spread_pct_mid": round(med, 5),
            "spread_pct_of_mid_p75": round(p75, 5),
            "n_quotes": len(spreads),
            "calibrated_at": now,
            "zone": ZONE_DESC,
        }

    # --- per-underlying friction report + the IWM decision line ----------
    for t, entry in calibrated.items():
        fr = _decision_friction_r(entry["median_spread_pct_mid"])
        print(f"  {t}: modeled friction_r ${DECISION_WIDTH:.0f}-wide vertical "
              f"(credit {DECISION_CREDIT:.2f}) = {fr:.3f}R")
    if "IWM" in underlyings:
        if "IWM" in calibrated:
            fr = _decision_friction_r(calibrated["IWM"]["median_spread_pct_mid"])
            if fr > DECISION_BAR_R:
                print(f"IWM EXCLUDED (friction_r {fr:.3f}R > {DECISION_BAR_R}R)")
            else:
                print(f"IWM OK (friction_r {fr:.3f}R <= {DECISION_BAR_R}R)")
            print("(report only — the whitelist change in the spec is the "
                  "operator's call)")
        else:
            print("IWM NOT ASSESSED — insufficient quotes this run")

    if not calibrated:
        print("no underlying reached --min-quotes; nothing written.")
        return 1
    if args.dry_run:
        print("(dry-run — data/execution_costs.json untouched)")
        return 0

    # --- merge into data/execution_costs.json, GLOBAL keys untouched -----
    out_path = Path(CONFIG.data_dir) / "execution_costs.json"
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"FATAL: refusing to overwrite unparseable {out_path}: {e}",
                  file=sys.stderr)
            return 1
        if not isinstance(existing, dict):
            print(f"FATAL: refusing to overwrite non-object JSON in {out_path} "
                  f"(top-level {type(existing).__name__})", file=sys.stderr)
            return 1
    per = dict(existing.get("per_underlying") or {})
    per.update(calibrated)
    existing["per_underlying"] = per
    _write_atomic(out_path, existing)
    print(f"written: {out_path} (per_underlying: {', '.join(sorted(per))})")

    from trading_agent import execution_costs as ec
    ec.reset_calibration_cache()  # this process sees the new figures at once
    _emit_calibration_event(calibrated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
