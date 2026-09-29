"""Public market data tools (no account, no signing)."""
from __future__ import annotations

from typing import Any, Literal

from trading_agent.mcp_servers.hyperliquid.core import (
    READ,
    address,
    cap,
    client,
    iso,
    mcp,
    mids,
    now_ms,
    num,
    pct,
    resolve,
    time_range,
)
from trading_agent.mcp_servers.hyperliquid.universe import Instrument

HOURS_PER_YEAR = 24 * 365
CANDLE_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d",
                    "1w", "1M")


def _next_hour_ms(ms: int) -> int:
    return (ms // 3_600_000 + 1) * 3_600_000


def _perp_row(inst: Instrument, ctx: dict) -> dict:
    funding = num(ctx.get("funding"))
    mark = num(ctx.get("markPx"))
    oi = num(ctx.get("openInterest"))
    impact = ctx.get("impactPxs") or [None, None]
    return {
        "coin": inst.coin, "dex": inst.dex, "mark_px": mark, "mid_px": num(ctx.get("midPx")),
        "oracle_px": num(ctx.get("oraclePx")), "prev_day_px": num(ctx.get("prevDayPx")),
        "change_24h_pct": pct(ctx.get("markPx"), ctx.get("prevDayPx")),
        "volume_24h_usd": num(ctx.get("dayNtlVlm")),
        # Hyperliquid pays funding every hour; `funding` is the hourly rate.
        "funding_rate_1h": funding,
        "funding_apr_pct": None if funding is None else round(funding * HOURS_PER_YEAR * 100, 4),
        "premium": num(ctx.get("premium")),
        "open_interest": oi,
        "open_interest_usd": None if oi is None or mark is None else round(oi * mark, 2),
        "impact_bid_px": num(impact[0]), "impact_ask_px": num(impact[1]),
        "max_leverage": inst.max_leverage, "quote": inst.quote,
        **({"delisted": True} if inst.delisted else {}),
    }


def _perp_ctxs(dex: str) -> dict[str, dict]:
    meta, ctxs = client().info({"type": "metaAndAssetCtxs", "dex": dex})
    return {a["name"]: ctx for a, ctx in zip(meta["universe"], ctxs, strict=False)}


def _spot_ctxs() -> dict[str, dict]:
    _meta, ctxs = client().info({"type": "spotMetaAndAssetCtxs"})
    return {ctx["coin"]: ctx for ctx in ctxs}


def _spot_row(inst: Instrument, ctx: dict) -> dict:
    return {
        "coin": inst.coin, "name": inst.name, "mark_px": num(ctx.get("markPx")),
        "mid_px": num(ctx.get("midPx")), "prev_day_px": num(ctx.get("prevDayPx")),
        "change_24h_pct": pct(ctx.get("markPx"), ctx.get("prevDayPx")),
        "volume_24h": num(ctx.get("dayNtlVlm")), "quote": inst.quote,
        "circulating_supply": num(ctx.get("circulatingSupply")),
    }


def _book_top(coin: str) -> dict:
    book = client().info({"type": "l2Book", "coin": coin}) or {}
    levels = book.get("levels") or [[], []]
    bid = levels[0][0] if levels and levels[0] else None
    ask = levels[1][0] if len(levels) > 1 and levels[1] else None
    return {"best_bid": num(bid["px"]) if bid else None,
            "best_ask": num(ask["px"]) if ask else None,
            "book_time_utc": iso(book.get("time"))}


@mcp.tool(annotations=READ)
def market_search(query: str, kind: Literal["any", "perp", "spot", "outcome"] = "any",
                  limit: int = 20, include_delisted: bool = False) -> dict:
    """Find instruments by name across perps (every HIP-3 dex), spot and outcome markets.

    Returns the exact `coin` to pass to other tools plus asset id, size decimals
    and tick rules. Examples: "BTC", "TSLA" (finds xyz:TSLA etc.), "HYPE/USDC",
    "priceBinary BTC" (outcomes match on their description).
    """
    kinds = None if kind == "any" else [kind]
    hits = client().universe.search(query, kinds=kinds, limit=limit,
                                    include_delisted=include_delisted)
    return {"query": query, "matches": [h.summary() for h in hits]}


@mcp.tool(annotations=READ)
def market_get_ticker(coin: str) -> dict:
    """Live snapshot for one instrument: prices, 24h change and volume, funding, open interest.

    `coin`: perp "BTC", HIP-3 perp "xyz:TSLA", spot "HYPE/USDC" or "@107",
    outcome "#12090". Funding on Hyperliquid is settled hourly:
    `funding_rate_1h` is the current hourly rate, `next_funding_utc` the next
    settlement. All times are UTC.
    """
    inst = resolve(coin)
    fetched = now_ms()
    if inst.is_perp:
        ctx = _perp_ctxs(inst.dex).get(inst.coin)
        if ctx is None:
            raise ValueError(f"no market context for {inst.coin}")
        row = _perp_row(inst, ctx)
        row["next_funding_utc"] = iso(_next_hour_ms(fetched))
    elif inst.kind == "spot":
        ctx = _spot_ctxs().get(inst.coin)
        if ctx is None:
            raise ValueError(f"no market context for {inst.coin}")
        row = _spot_row(inst, ctx)
    else:
        q = client().universe.question_for(inst.outcome or -1)
        row = {"coin": inst.coin, "name": inst.name, "outcome": inst.outcome, "side": inst.side,
               "description": inst.description, "mid_px": num(mids("").get(inst.coin)),
               "question": q.get("name") if q else None, **_book_top(inst.coin)}
    row.update(kind=inst.kind, fetched_at_utc=iso(fetched), network=client().settings.network)
    return row


@mcp.tool(annotations=READ)
def market_get_perp_markets(dex: str = "",
                            sort_by: Literal["volume", "open_interest", "funding", "change",
                                             "name"] = "volume",
                            limit: int = 50, include_delisted: bool = False) -> dict:
    """All perpetual markets on one perp dex, with price, 24h change, volume, funding and OI.

    `dex`: "" for the main Hyperliquid perps, or a HIP-3 dex name (see
    market_get_perp_dexs). Sorted descending by `sort_by` (except `name`).
    """
    ctxs = _perp_ctxs(dex)
    rows = []
    for inst in client().universe.instruments(kind="perp", dex=dex):
        if inst.delisted and not include_delisted:
            continue
        if inst.coin in ctxs:
            rows.append(_perp_row(inst, ctxs[inst.coin]))
    key = {"volume": "volume_24h_usd", "open_interest": "open_interest_usd",
           "funding": "funding_rate_1h", "change": "change_24h_pct"}.get(sort_by)
    if key:
        rows.sort(key=lambda r: (r[key] is not None, r[key] or 0), reverse=True)
    else:
        rows.sort(key=lambda r: r["coin"])
    rows, truncated = cap(rows, limit)
    return {"dex": dex, "count": len(rows), "truncated": truncated, "markets": rows,
            "fetched_at_utc": iso(now_ms())}


@mcp.tool(annotations=READ)
def market_get_spot_markets(sort_by: Literal["volume", "change", "name"] = "volume",
                            quote: str | None = None, limit: int = 50) -> dict:
    """Spot pairs with price, 24h change and volume. Optional `quote` filter (e.g. "USDC")."""
    ctxs = _spot_ctxs()
    rows = []
    for inst in client().universe.instruments(kind="spot"):
        if quote and inst.quote.upper() != quote.upper():
            continue
        if inst.coin in ctxs:
            rows.append(_spot_row(inst, ctxs[inst.coin]))
    if sort_by == "name":
        rows.sort(key=lambda r: r["name"])
    else:
        key = "volume_24h" if sort_by == "volume" else "change_24h_pct"
        rows.sort(key=lambda r: (r[key] is not None, r[key] or 0), reverse=True)
    rows, truncated = cap(rows, limit)
    return {"count": len(rows), "truncated": truncated, "markets": rows,
            "fetched_at_utc": iso(now_ms())}


@mcp.tool(annotations=READ)
def market_get_outcome_markets(query: str | None = None, limit: int = 50) -> dict:
    """HIP-4 outcome (prediction) markets: each outcome's sides, their coins and mid prices.

    A side's price is its implied probability (0-1). Buy a side with the
    order tools using its coin, e.g. "#12090". `query` filters on name or
    description.
    """
    all_mids = mids("")
    grouped: dict[int, dict] = {}
    for inst in client().universe.instruments(kind="outcome"):
        text = f"{inst.name} {inst.description or ''}".lower()
        if query and query.lower() not in text:
            continue
        o = grouped.setdefault(inst.outcome, {
            "outcome": inst.outcome, "name": inst.name.rsplit(" [", 1)[0],
            "description": inst.description, "sides": [],
        })
        o["sides"].append({"side": inst.side, "coin": inst.coin,
                           "label": inst.name.rsplit(" [", 1)[-1].rstrip("]"),
                           "mid_px": num(all_mids.get(inst.coin))})
        q = client().universe.question_for(inst.outcome)
        if q and "question" not in o:
            o["question"] = {"question": q.get("question"), "name": q.get("name")}
    rows, truncated = cap(list(grouped.values()), limit)
    return {"count": len(rows), "truncated": truncated, "outcomes": rows,
            "fetched_at_utc": iso(now_ms())}


@mcp.tool(annotations=READ)
def market_get_mids(coins: list[str] | None = None, dex: str = "",
                    include_spot_and_outcomes: bool = False) -> dict:
    """Mid prices. Pass `coins` for specific instruments (any kind, any dex);
    otherwise returns every perp mid on `dex` ("" = main perps)."""
    if coins:
        out: dict[str, Any] = {}
        cache: dict[str, dict] = {}
        for name in coins:
            inst = resolve(name)
            d = inst.dex if inst.is_perp else ""
            if d not in cache:
                cache[d] = mids(d)
            out[name] = num(cache[d].get(inst.coin))
        return {"mids": out, "fetched_at_utc": iso(now_ms())}
    raw = mids(dex)
    if not include_spot_and_outcomes:
        raw = {k: v for k, v in raw.items() if not k.startswith(("@", "#")) and "/" not in k}
    return {"dex": dex, "count": len(raw), "mids": {k: num(v) for k, v in raw.items()},
            "fetched_at_utc": iso(now_ms())}


@mcp.tool(annotations=READ)
def market_get_orderbook(coin: str, depth: int = 10, n_sig_figs: Literal[2, 3, 4, 5] | None = None,
                         mantissa: Literal[1, 2, 5] | None = None) -> dict:
    """L2 order book (up to 20 levels per side), with best bid/ask, mid and spread.

    `n_sig_figs` aggregates price levels (2-5 significant figures); `mantissa`
    (1, 2 or 5) is only valid with n_sig_figs=5.
    """
    inst = resolve(coin)
    req: dict[str, Any] = {"type": "l2Book", "coin": inst.coin}
    if n_sig_figs is not None:
        req["nSigFigs"] = n_sig_figs
    if mantissa is not None:
        if n_sig_figs != 5:
            raise ValueError("mantissa is only allowed with n_sig_figs=5")
        req["mantissa"] = mantissa
    book = client().info(req) or {}
    levels = book.get("levels") or [[], []]
    d = max(1, min(int(depth), 20))
    bids = [[num(x["px"]), num(x["sz"]), x.get("n")] for x in levels[0][:d]]
    asks = [[num(x["px"]), num(x["sz"]), x.get("n")] for x in levels[1][:d]]
    out: dict[str, Any] = {"coin": inst.coin, "time_utc": iso(book.get("time")),
                           "columns": ["px", "sz", "n_orders"], "bids": bids, "asks": asks}
    if bids and asks:
        bb, ba = bids[0][0], asks[0][0]
        m = (bb + ba) / 2
        out.update(best_bid=bb, best_ask=ba, mid=m, spread=round(ba - bb, 12),
                   spread_bps=round((ba - bb) / m * 1e4, 3) if m else None)
    return out


@mcp.tool(annotations=READ)
def market_get_candles(coin: str, interval: str = "1h", start_time: int | str | None = None,
                       end_time: int | str | None = None, lookback_hours: float | None = None,
                       limit: int = 300) -> dict:
    """OHLCV candles, oldest first. Only the latest 5000 candles exist per coin/interval.

    `interval`: 1m 3m 5m 15m 30m 1h 2h 4h 8h 12h 1d 3d 1w 1M. Times are epoch ms
    or ISO-8601; with no start_time the window is the last `lookback_hours`
    (default 24h, or 30 days for 1d and longer intervals).
    """
    if interval not in CANDLE_INTERVALS:
        raise ValueError(f"interval must be one of {CANDLE_INTERVALS}")
    inst = resolve(coin)
    default_h = 24 * 30 if interval in ("1d", "3d", "1w", "1M") else 24
    start, end = time_range(start_time, end_time, lookback_hours, default_h)
    rows = client().info({"type": "candleSnapshot", "req": {
        "coin": inst.coin, "interval": interval, "startTime": start, "endTime": end or now_ms()}})
    rows = rows or []
    out_rows = [{"t_utc": iso(r["t"]), "o": num(r["o"]), "h": num(r["h"]), "l": num(r["l"]),
                 "c": num(r["c"]), "v": num(r["v"]), "n": r.get("n")} for r in rows]
    truncated = len(out_rows) > limit
    if truncated:
        out_rows = out_rows[-limit:]  # keep the most recent
    return {"coin": inst.coin, "interval": interval, "count": len(out_rows),
            "truncated_oldest": truncated, "candles": out_rows}


@mcp.tool(annotations=READ)
def market_get_recent_trades(coin: str, limit: int = 50) -> dict:
    """Most recent public trades. `side` is the aggressor: buy = taker bought."""
    inst = resolve(coin)
    trades = client().info({"type": "recentTrades", "coin": inst.coin}) or []
    rows = [{"time_utc": iso(t["time"]), "side": "buy" if t["side"] == "B" else "sell",
             "px": num(t["px"]), "sz": num(t["sz"]), "tid": t.get("tid")} for t in trades]
    rows.sort(key=lambda r: r["time_utc"] or "", reverse=True)
    rows, truncated = cap(rows, limit)
    return {"coin": inst.coin, "count": len(rows), "truncated": truncated, "trades": rows}


@mcp.tool(annotations=READ)
def market_get_funding_history(coin: str, start_time: int | str | None = None,
                               end_time: int | str | None = None,
                               lookback_hours: float | None = None, limit: int = 500) -> dict:
    """Hourly funding rate history for a perp (default: last 24h), with average and APR."""
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError(f"{inst.coin} is not a perp; funding only applies to perps")
    start, end = time_range(start_time, end_time, lookback_hours, 24)
    req: dict[str, Any] = {"type": "fundingHistory", "coin": inst.coin, "startTime": start}
    if end is not None:
        req["endTime"] = end
    rows = client().info(req) or []
    out = [{"time_utc": iso(r["time"]), "funding_rate_1h": num(r["fundingRate"]),
            "premium": num(r.get("premium"))} for r in rows]
    rates = [r["funding_rate_1h"] for r in out if r["funding_rate_1h"] is not None]
    avg = sum(rates) / len(rates) if rates else None
    out, truncated = cap(out[::-1], limit)
    return {"coin": inst.coin, "count": len(out), "truncated": truncated,
            "avg_rate_1h": avg, "sum_rate": sum(rates) if rates else None,
            "avg_apr_pct": None if avg is None else round(avg * HOURS_PER_YEAR * 100, 4),
            "history_newest_first": out}


@mcp.tool(annotations=READ)
def market_get_predicted_fundings(coins: list[str] | None = None, limit: int = 50) -> dict:
    """Predicted next funding on Hyperliquid vs Binance and Bybit (main perp dex only).

    Rates are per each venue's own interval; `rate_1h_equiv` normalises them
    so venues compare directly (useful for funding-arbitrage screens).
    """
    raw = client().info({"type": "predictedFundings"}) or []
    wanted = {resolve(c).coin for c in coins} if coins else None
    rows = []
    for coin, venues in raw:
        if wanted is not None and coin not in wanted:
            continue
        v_out = {}
        for venue, info in venues or []:
            if not info:
                continue
            rate = num(info.get("fundingRate"))
            hours = info.get("fundingIntervalHours") or (1 if venue == "HlPerp" else 8)
            v_out[venue] = {"rate": rate, "interval_hours": hours,
                            "rate_1h_equiv": None if rate is None else rate / hours,
                            "next_funding_utc": iso(info.get("nextFundingTime"))}
        rows.append({"coin": coin, "venues": v_out})
    rows, truncated = cap(rows, limit)
    return {"count": len(rows), "truncated": truncated, "predictions": rows}


@mcp.tool(annotations=READ)
def market_get_perp_dexs() -> dict:
    """List perp dexs: the main one ("") plus builder-deployed HIP-3 dexs, with asset counts."""
    u = client().universe
    counts: dict[str, int] = {}
    for inst in u.instruments(kind="perp"):
        counts[inst.dex] = counts.get(inst.dex, 0) + 1
    out = []
    for i, d in enumerate(u.dexs()):
        name = "" if i == 0 else (d or {}).get("name", "")
        row: dict[str, Any] = {"dex": name, "index": i, "n_assets": counts.get(name, 0)}
        if d:
            row.update(full_name=d.get("fullName"), deployer=d.get("deployer"),
                       fee_recipient=d.get("feeRecipient"), oracle_updater=d.get("oracleUpdater"))
        else:
            row["full_name"] = "Hyperliquid (main perps)"
        out.append(row)
    return {"count": len(out), "dexs": out}


@mcp.tool(annotations=READ)
def market_get_perp_dex_details(dex: str) -> dict:
    """Limits and status for one perp dex: OI caps, transfer limit, net deposits,
    assets currently at their open-interest cap. `dex` "" = main perps."""
    c = client()
    out: dict[str, Any] = {"dex": dex}
    out["perps_at_open_interest_cap"] = c.info({"type": "perpsAtOpenInterestCap", "dex": dex})
    out["status"] = c.info({"type": "perpDexStatus", "dex": dex})
    if dex:
        out["limits"] = c.info({"type": "perpDexLimits", "dex": dex})
    return out


@mcp.tool(annotations=READ)
def market_get_trading_limits(dex: str = "") -> dict:
    """Exchange-wide order limits: max market-order notional by leverage tier,
    and the perps currently at their open-interest cap on `dex`."""
    c = client()
    tiers = c.info({"type": "maxMarketOrderNtls"}) or []
    return {
        "max_market_order_notional_by_leverage": [
            {"max_leverage": lev, "max_notional_usd": num(ntl)} for lev, ntl in tiers],
        "perps_at_open_interest_cap": c.info({"type": "perpsAtOpenInterestCap", "dex": dex}),
    }


@mcp.tool(annotations=READ)
def market_get_token_info(token: str, include_genesis: bool = False) -> dict:
    """Spot token details: supply, prices, deployer, decimals and token id.
    `token` is a name ("HYPE"), index ("150") or "NAME:0xtokenId"."""
    u = client().universe
    t = u.token(token)
    details = client().info({"type": "tokenDetails", "tokenId": t["tokenId"]}) or {}
    if not include_genesis:
        details.pop("genesis", None)
        details.pop("nonCirculatingUserBalances", None)
    return {"token": t["name"], "index": t["index"], "token_id": t["tokenId"],
            "wire_name": u.token_wire(t), "sz_decimals": t["szDecimals"],
            "wei_decimals": t["weiDecimals"], "evm_contract": t.get("evmContract"),
            "full_name": t.get("fullName"), "details": details}


@mcp.tool(annotations=READ)
def market_get_exchange_status() -> dict:
    """Exchange status (special statuses such as upgrades or halts) and server time."""
    s = client().info({"type": "exchangeStatus"}) or {}
    return {**s, "time_utc": iso(s.get("time"))}


@mcp.tool(annotations=READ)
def market_get_vault_details(vault_address: str, user: str | None = None,
                             top_followers: int = 10) -> dict:
    """A vault's leader, APR, TVL, lock-ups, recent performance and largest depositors.
    Pass `user` to include that depositor's own state (followerState)."""
    req: dict[str, Any] = {"type": "vaultDetails", "vaultAddress": address(vault_address)}
    if user:
        req["user"] = address(user, "user")
    d = client().info(req) or {}
    followers = d.pop("followers", None) or []
    followers.sort(key=lambda f: num(f.get("vaultEquity")) or 0, reverse=True)
    portfolio = {}
    for period, data in d.pop("portfolio", None) or []:
        av = (data or {}).get("accountValueHistory") or []
        pnl = (data or {}).get("pnlHistory") or []
        portfolio[period] = {"account_value": num(av[-1][1]) if av else None,
                             "pnl": num(pnl[-1][1]) if pnl else None,
                             "volume": num((data or {}).get("vlm"))}
    d["n_followers"] = len(followers)
    d["tvl_usd"] = round(sum(num(f.get("vaultEquity")) or 0 for f in followers), 2)
    d["top_followers"] = followers[: max(int(top_followers), 0)]
    d["performance"] = portfolio
    return d


@mcp.tool(annotations=READ)
def market_get_validators(limit: int = 50) -> dict:
    """Staking validators: stake (HYPE), commission, active/jailed state and uptime."""
    vals = client().info({"type": "validatorSummaries"}) or []
    rows = []
    for v in vals:
        stats = dict(v.get("stats") or [])
        day = stats.get("day") or {}
        rows.append({
            "validator": v.get("validator"), "name": v.get("name"), "signer": v.get("signer"),
            "stake_hype": (v.get("stake") or 0) / 1e8, "commission": num(v.get("commission")),
            "is_active": v.get("isActive"), "is_jailed": v.get("isJailed"),
            "uptime_day": num(day.get("uptimeFraction")),
            "predicted_apr": num(day.get("predictedApr")),
        })
    rows.sort(key=lambda r: r["stake_hype"], reverse=True)
    rows, truncated = cap(rows, limit)
    return {"count": len(rows), "truncated": truncated, "validators": rows}


@mcp.tool(annotations=READ)
def market_get_borrow_lend_reserves(token: str | None = None) -> dict:
    """Borrow/lend reserves (portfolio-margin money market): rates, utilisation, LTV.
    `token` optional ("USDC", "HYPE" or an index)."""
    u = client().universe
    if token is not None:
        t = u.token(token)
        r = client().info({"type": "borrowLendReserveState", "token": t["index"]})
        return {"token": t["name"], "index": t["index"], **(r or {})}
    rows = []
    for idx, state in client().info({"type": "allBorrowLendReserveStates"}) or []:
        try:
            name = u.token(int(idx))["name"]
        except LookupError:
            name = None
        rows.append({"token": name, "index": idx, **state})
    return {"reserves": rows}


@mcp.tool(annotations=READ)
def market_get_margin_table(margin_table_id: int, dex: str = "") -> dict:
    """Margin tiers (max leverage by position size) for a margin table id from a perp's meta."""
    req: dict[str, Any] = {"type": "marginTable", "id": int(margin_table_id)}
    if dex:
        req["dex"] = dex
    return {"id": margin_table_id, "table": client().info(req)}


@mcp.tool(annotations=READ)
def market_get_perp_categories(coin: str | None = None, category: str | None = None,
                               limit: int = 200) -> dict:
    """Perp categories (crypto, stocks, commodities, indices, preipo...). With `coin`,
    that perp's annotation; with `category`, only perps in it."""
    c = client()
    if coin:
        inst = resolve(coin)
        return {"coin": inst.coin, "annotation": c.info({"type": "perpAnnotation",
                                                          "coin": inst.coin})}
    all_rows = [{"coin": name, "category": cat}
                for name, cat in c.info({"type": "perpCategories"}) or []]
    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    rows = [r for r in all_rows if category is None or r["category"] == category]
    rows, truncated = cap(rows, limit)
    return {"count_by_category": counts, "truncated": truncated, "perps": rows}


@mcp.tool(annotations=READ)
def market_get_deploy_auctions() -> dict:
    """Status of the HIP-3 perp-deploy, spot-pair-deploy and gossip-priority Dutch auctions."""
    c = client()
    out: dict[str, Any] = {}
    for key, typ in (("perp_deploy", "perpDeployAuctionStatus"),
                     ("spot_pair_deploy", "spotPairDeployAuctionStatus"),
                     ("gossip_priority", "gossipPriorityAuctionStatus")):
        try:
            out[key] = c.info({"type": typ})
        except Exception as e:
            out[key] = {"error": str(e)}
    return out


@mcp.tool(annotations=READ)
def market_get_settled_outcome(outcome: int) -> dict:
    """Settlement of an outcome market: spec, settle fraction (1 = Yes won) and details."""
    return client().info({"type": "settledOutcome", "outcome": int(outcome)}) or {}


@mcp.tool(annotations=READ)
def info_raw(request: dict, max_items: int = 200) -> dict:
    """Escape hatch: send any request body to the read-only POST /info endpoint.

    Use for info types without a dedicated tool, e.g. {"type": "webData2",
    "user": "0x..."} or {"type": "outcomeTemplates"}. Nothing is signed; this
    cannot trade. Top-level lists longer than `max_items` are truncated.
    """
    if not isinstance(request, dict) or not isinstance(request.get("type"), str):
        raise ValueError('request must be an object with a string "type"')
    res = client().info(request)
    if isinstance(res, list) and len(res) > max_items:
        return {"result": res[:max_items], "truncated": True, "total_items": len(res)}
    return {"result": res, "truncated": False}
