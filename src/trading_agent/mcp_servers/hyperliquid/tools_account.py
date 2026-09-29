"""Read-only account tools. `user` defaults to the configured account.

Hyperliquid account data is public: any address can be queried. Pass the
MASTER account (or sub-account / vault) address, never an API wallet's —
an agent address returns empty results.
"""
from __future__ import annotations

from importlib import metadata
from typing import Any, Literal

from trading_agent.mcp_servers.hyperliquid.core import (
    READ,
    address,
    cap,
    client,
    iso,
    mcp,
    mids,
    num,
    resolve,
    time_range,
    user_or_default,
)
from trading_agent.mcp_servers.hyperliquid.settings import WRITE_MODULES


def _positions(state: dict) -> list[dict]:
    out = []
    for ap in state.get("assetPositions") or []:
        p = ap.get("position") or {}
        szi = num(p.get("szi")) or 0.0
        if szi == 0:
            continue
        lev = p.get("leverage") or {}
        out.append({
            "coin": p.get("coin"), "side": "long" if szi > 0 else "short", "size": abs(szi),
            "entry_px": num(p.get("entryPx")), "position_value": num(p.get("positionValue")),
            "unrealized_pnl": num(p.get("unrealizedPnl")),
            "return_on_equity": num(p.get("returnOnEquity")),
            "liquidation_px": num(p.get("liquidationPx")), "margin_used": num(p.get("marginUsed")),
            "leverage": lev.get("value"), "margin_mode": lev.get("type"),
            "max_leverage": p.get("maxLeverage"),
            "funding_since_open": num((p.get("cumFunding") or {}).get("sinceOpen")),
        })
    return out


def _margin(state: dict) -> dict:
    ms = state.get("marginSummary") or {}
    cms = state.get("crossMarginSummary") or {}
    av = num(ms.get("accountValue"))
    ntl = num(ms.get("totalNtlPos"))
    return {
        "account_value": av, "total_notional": ntl,
        "total_margin_used": num(ms.get("totalMarginUsed")),
        "withdrawable": num(state.get("withdrawable")),
        "cross_account_value": num(cms.get("accountValue")),
        "cross_maintenance_margin_used": num(state.get("crossMaintenanceMarginUsed")),
        "effective_leverage": round(ntl / av, 4) if av and ntl is not None else None,
        "time_utc": iso(state.get("time")),
    }


def _dex_list(dex: str | None, all_dexs: bool) -> list[str]:
    if all_dexs:
        return client().universe.dex_names()
    return [dex or ""]


@mcp.tool(annotations=READ)
def server_status() -> dict:
    """What this server can and cannot do right now: network, signer, account, which
    write modules are enabled, limits, and whether live writes are possible.

    Call this first. Nothing here is secret: the private key is never shown.
    """
    c = client()
    s = c.settings
    role = None
    if c.wallet is not None:
        role = c.signer_role()
    live_blockers = []
    if s.read_only:
        live_blockers.append("HL_READ_ONLY=true")
    if c.wallet is None:
        live_blockers.append("no HL_PRIVATE_KEY")
    if s.is_mainnet and not s.allow_mainnet_writes:
        live_blockers.append("mainnet without HL_ALLOW_MAINNET_WRITES=true")
    if s.dry_run:
        live_blockers.append("HL_DRY_RUN=true (every write is a dry run)")
    try:
        sdk = metadata.version("hyperliquid-python-sdk")
    except metadata.PackageNotFoundError:  # pragma: no cover
        sdk = None
    return {
        "network": s.network, "api_url": s.base_url,
        "signer_address": c.signer_address,
        "signer_role": role,
        "signer_is_api_wallet": c.signer_is_agent() if c.wallet else None,
        "account_address": c.account_address(),
        "default_vault_address": s.vault_address,
        "read_only": s.read_only, "dry_run": s.dry_run,
        "mainnet_writes_allowed": s.allow_mainnet_writes if s.is_mainnet else None,
        "write_modules": {m: (m in s.write_modules) for m in WRITE_MODULES},
        "live_writes_possible": not live_blockers and bool(s.write_modules),
        "live_write_blockers": live_blockers,
        "limits": {
            "max_order_notional_usd": s.max_order_notional_usd,
            "max_leverage": s.max_leverage,
            "default_slippage": s.default_slippage, "max_slippage": s.max_slippage,
            "allowed_coins": sorted(s.allowed_coins) if s.allowed_coins else "any",
            "withdraw_allowlist": sorted(s.withdraw_allowlist),
        },
        "audit_log": str(s.audit_path),
        "sdk": {"hyperliquid-python-sdk": sdk},
    }


@mcp.tool(annotations=READ)
def account_get_summary(user: str | None = None, include_all_dexs: bool = False) -> dict:
    """Account overview: perp margin summary and open positions, plus non-zero spot balances.

    `include_all_dexs` also scans every HIP-3 perp dex (one request each).
    Under unified-account / portfolio-margin modes, spot balances are the
    source of truth for the trading balance.
    """
    u = user_or_default(user)
    c = client()
    perps = []
    for d in _dex_list("", include_all_dexs):
        st = c.info({"type": "clearinghouseState", "user": u, "dex": d}) or {}
        pos = _positions(st)
        m = _margin(st)
        if d == "" or pos or (m["account_value"] or 0) > 0:
            perps.append({"dex": d, "margin": m, "positions": pos})
    spot = c.info({"type": "spotClearinghouseState", "user": u}) or {}
    balances = [b for b in spot.get("balances") or [] if (num(b.get("total")) or 0) != 0]
    return {"user": u, "perp": perps, "spot_balances": [
        {"coin": b.get("coin"), "total": num(b.get("total")), "hold": num(b.get("hold")),
         "entry_notional": num(b.get("entryNtl"))} for b in balances]}


@mcp.tool(annotations=READ)
def account_get_positions(user: str | None = None, dex: str | None = None,
                          all_dexs: bool = False) -> dict:
    """Open perp positions with entry, uPnL, liquidation price, leverage and margin mode.
    `dex` "" (default) = main perps; `all_dexs` scans every HIP-3 dex too."""
    u = user_or_default(user)
    out = []
    for d in _dex_list(dex, all_dexs):
        st = client().info({"type": "clearinghouseState", "user": u, "dex": d}) or {}
        for p in _positions(st):
            p["dex"] = d
            out.append(p)
    return {"user": u, "count": len(out), "positions": out}


@mcp.tool(annotations=READ)
def account_get_spot_balances(user: str | None = None, include_zero: bool = False) -> dict:
    """Spot token balances (incl. outcome shares like "+12090"), with a USDC value
    estimate from current mids where a USDC market exists."""
    u = user_or_default(user)
    c = client()
    state = c.info({"type": "spotClearinghouseState", "user": u}) or {}
    all_mids = mids("")
    rows = []
    for b in state.get("balances") or []:
        total = num(b.get("total")) or 0.0
        if total == 0 and not include_zero:
            continue
        coin = b.get("coin") or ""
        px = None
        if coin.upper() in ("USDC",):
            px = 1.0
        elif coin.startswith("+"):
            px = num(all_mids.get("#" + coin[1:]))
        else:
            try:
                px = num(all_mids.get(c.universe.resolve(f"{coin}/USDC").coin))
            except LookupError:
                px = None
        rows.append({"coin": coin, "token": b.get("token"), "total": total,
                     "hold": num(b.get("hold")), "entry_notional": num(b.get("entryNtl")),
                     "mid_usdc": px, "value_usdc": None if px is None else round(total * px, 6)})
    rows.sort(key=lambda r: r["value_usdc"] or 0, reverse=True)
    return {"user": u, "balances": rows,
            "total_value_usdc": round(sum(r["value_usdc"] or 0 for r in rows), 6)}


@mcp.tool(annotations=READ)
def account_get_open_orders(user: str | None = None, coin: str | None = None,
                            dex: str | None = None, all_dexs: bool = False) -> dict:
    """Open orders incl. trigger (TP/SL) details. Spot and outcome orders are
    listed with the main dex (""); pass `all_dexs` for HIP-3 perps too."""
    u = user_or_default(user)
    want = resolve(coin).coin if coin else None
    dexs = _dex_list(dex, all_dexs)
    if want and not all_dexs and dex is None:
        inst = resolve(coin)
        dexs = [inst.dex if inst.is_perp else ""]
    rows = []
    for d in dexs:
        for o in client().info({"type": "frontendOpenOrders", "user": u, "dex": d}) or []:
            if want and o.get("coin") != want:
                continue
            rows.append({
                "coin": o.get("coin"), "oid": o.get("oid"), "cloid": o.get("cloid"),
                "side": "buy" if o.get("side") == "B" else "sell",
                "limit_px": num(o.get("limitPx")), "size": num(o.get("sz")),
                "orig_size": num(o.get("origSz")), "order_type": o.get("orderType"),
                "tif": o.get("tif"), "reduce_only": o.get("reduceOnly"),
                "is_trigger": o.get("isTrigger"), "trigger_px": num(o.get("triggerPx")),
                "trigger_condition": o.get("triggerCondition"),
                "is_position_tpsl": o.get("isPositionTpsl"),
                "children": o.get("children") or [], "time_utc": iso(o.get("timestamp")),
            })
    return {"user": u, "count": len(rows), "orders": rows}


@mcp.tool(annotations=READ)
def account_get_order_status(oid: int | None = None, cloid: str | None = None,
                             user: str | None = None) -> dict:
    """Status of one order by exchange id (`oid`) or client id (`cloid`, 0x + 32 hex).

    Statuses include open, filled, canceled, triggered, rejected and the
    specific reject/cancel reasons (e.g. perpMarginRejected, reduceOnlyCanceled).
    """
    if (oid is None) == (cloid is None):
        raise ValueError("pass exactly one of oid or cloid")
    u = user_or_default(user)
    ref = oid if oid is not None else cloid.strip().lower()
    res = client().info({"type": "orderStatus", "user": u, "oid": ref})
    if isinstance(res, dict) and isinstance(res.get("order"), dict):
        res["order"]["status_time_utc"] = iso(res["order"].get("statusTimestamp"))
    return {"user": u, "result": res}


def _fill_row(f: dict) -> dict:
    return {
        "time_utc": iso(f.get("time")), "coin": f.get("coin"),
        "side": "buy" if f.get("side") == "B" else "sell", "dir": f.get("dir"),
        "px": num(f.get("px")), "sz": num(f.get("sz")), "fee": num(f.get("fee")),
        "fee_token": f.get("feeToken"), "builder_fee": num(f.get("builderFee")),
        "closed_pnl": num(f.get("closedPnl")), "start_position": num(f.get("startPosition")),
        "crossed": f.get("crossed"), "oid": f.get("oid"), "tid": f.get("tid"),
        "hash": f.get("hash"),
    }


@mcp.tool(annotations=READ)
def account_get_fills(user: str | None = None, start_time: int | str | None = None,
                      end_time: int | str | None = None, lookback_hours: float | None = None,
                      coin: str | None = None, aggregate_by_time: bool = False,
                      limit: int = 100) -> dict:
    """Trade fills, newest first, with fee and closed-PnL totals.

    Without a time window: the latest fills (up to 2000). With start_time or
    lookback_hours: fills in that window (max 2000 per call; only the 10000
    most recent are retrievable).
    """
    u = user_or_default(user)
    if start_time is None and lookback_hours is None:
        req: dict[str, Any] = {"type": "userFills", "user": u}
        if aggregate_by_time:
            req["aggregateByTime"] = True
    else:
        start, end = time_range(start_time, end_time, lookback_hours, 24)
        req = {"type": "userFillsByTime", "user": u, "startTime": start, "endTime": end,
               "aggregateByTime": aggregate_by_time}
    fills = client().info(req) or []
    want = resolve(coin).coin if coin else None
    rows = [_fill_row(f) for f in fills if not want or f.get("coin") == want]
    rows.sort(key=lambda r: r["time_utc"] or "", reverse=True)
    totals = {"fees": round(sum(r["fee"] or 0 for r in rows), 8),
              "closed_pnl": round(sum(r["closed_pnl"] or 0 for r in rows), 8),
              "volume": round(sum((r["px"] or 0) * (r["sz"] or 0) for r in rows), 4)}
    rows, truncated = cap(rows, limit)
    return {"user": u, "count": len(rows), "truncated": truncated,
            "totals_over_all_matching": totals, "fills": rows}


@mcp.tool(annotations=READ)
def account_get_order_history(user: str | None = None, coin: str | None = None,
                              limit: int = 100) -> dict:
    """Historical orders (latest 2000) with final status: filled, canceled, rejected..."""
    u = user_or_default(user)
    want = resolve(coin).coin if coin else None
    rows = []
    for h in client().info({"type": "historicalOrders", "user": u}) or []:
        o = h.get("order") or {}
        if want and o.get("coin") != want:
            continue
        rows.append({"coin": o.get("coin"), "oid": o.get("oid"), "cloid": o.get("cloid"),
                     "side": "buy" if o.get("side") == "B" else "sell",
                     "order_type": o.get("orderType"), "tif": o.get("tif"),
                     "limit_px": num(o.get("limitPx")), "orig_size": num(o.get("origSz")),
                     "remaining_size": num(o.get("sz")), "reduce_only": o.get("reduceOnly"),
                     "trigger_px": num(o.get("triggerPx")) if o.get("isTrigger") else None,
                     "status": h.get("status"), "status_time_utc": iso(h.get("statusTimestamp")),
                     "placed_utc": iso(o.get("timestamp"))})
    rows, truncated = cap(rows, limit)
    return {"user": u, "count": len(rows), "truncated": truncated, "orders": rows}


@mcp.tool(annotations=READ)
def account_get_funding_payments(user: str | None = None, start_time: int | str | None = None,
                                 end_time: int | str | None = None,
                                 lookback_hours: float | None = None, coin: str | None = None,
                                 limit: int = 200) -> dict:
    """Funding paid/received per position (default: last 7 days). Positive usdc = received."""
    u = user_or_default(user)
    start, end = time_range(start_time, end_time, lookback_hours, 24 * 7)
    req: dict[str, Any] = {"type": "userFunding", "user": u, "startTime": start}
    if end is not None:
        req["endTime"] = end
    want = resolve(coin).coin if coin else None
    rows = []
    for r in client().info(req) or []:
        d = r.get("delta") or {}
        if want and d.get("coin") != want:
            continue
        rows.append({"time_utc": iso(r.get("time")), "coin": d.get("coin"),
                     "usdc": num(d.get("usdc")), "position_size": num(d.get("szi")),
                     "funding_rate": num(d.get("fundingRate"))})
    total = round(sum(r["usdc"] or 0 for r in rows), 6)
    rows, truncated = cap(rows[::-1], limit)
    return {"user": u, "count": len(rows), "truncated": truncated, "total_usdc": total,
            "payments_newest_first": rows}


@mcp.tool(annotations=READ)
def account_get_ledger(user: str | None = None, start_time: int | str | None = None,
                       end_time: int | str | None = None, lookback_hours: float | None = None,
                       limit: int = 200) -> dict:
    """Non-funding ledger: deposits, withdrawals, transfers, liquidations, vault and
    staking movements, rewards (default: last 30 days)."""
    u = user_or_default(user)
    start, end = time_range(start_time, end_time, lookback_hours, 24 * 30)
    req: dict[str, Any] = {"type": "userNonFundingLedgerUpdates", "user": u, "startTime": start}
    if end is not None:
        req["endTime"] = end
    rows = [{"time_utc": iso(r.get("time")), "hash": r.get("hash"), "delta": r.get("delta")}
            for r in client().info(req) or []]
    rows, truncated = cap(rows[::-1], limit)
    return {"user": u, "count": len(rows), "truncated": truncated, "updates_newest_first": rows}


@mcp.tool(annotations=READ)
def account_get_twaps(user: str | None = None, include_slice_fills: bool = False,
                      limit: int = 50) -> dict:
    """TWAP orders (running and finished) and, optionally, their individual slice fills."""
    u = user_or_default(user)
    hist = client().info({"type": "twapHistory", "user": u}) or []
    hist, truncated = cap(hist[::-1], limit)
    out: dict[str, Any] = {"user": u, "count": len(hist), "truncated": truncated, "twaps": hist}
    if include_slice_fills:
        fills = client().info({"type": "userTwapSliceFills", "user": u}) or []
        out["slice_fills"] = [{"twap_id": f.get("twapId"), **_fill_row(f.get("fill") or {})}
                              for f in fills[:limit]]
    return out


@mcp.tool(annotations=READ)
def account_get_fees(user: str | None = None, recent_days: int = 7) -> dict:
    """The account's current maker/taker rates (perp and spot), discounts and recent volume."""
    u = user_or_default(user)
    f = client().info({"type": "userFees", "user": u}) or {}
    sched = f.get("feeSchedule") or {}
    return {
        "user": u,
        "perp_taker": num(f.get("userCrossRate")), "perp_maker": num(f.get("userAddRate")),
        "spot_taker": num(f.get("userSpotCrossRate")), "spot_maker": num(f.get("userSpotAddRate")),
        "base_schedule": {k: sched.get(k) for k in ("cross", "add", "spotCross", "spotAdd")},
        "active_referral_discount": num(f.get("activeReferralDiscount")),
        "active_staking_discount": f.get("activeStakingDiscount"),
        "staking_link": f.get("stakingLink"),
        "daily_volume_recent": (f.get("dailyUserVlm") or [])[-max(int(recent_days), 0):],
    }


@mcp.tool(annotations=READ)
def account_get_rate_limit(user: str | None = None) -> dict:
    """Address-based action budget: requests used vs cap (1 request per 1 USDC traded,
    10k initial buffer). Info requests do not count."""
    u = user_or_default(user)
    return {"user": u, **(client().info({"type": "userRateLimit", "user": u}) or {})}


@mcp.tool(annotations=READ)
def account_get_portfolio(user: str | None = None,
                          period: Literal["day", "week", "month", "allTime", "perpDay",
                                          "perpWeek", "perpMonth", "perpAllTime"] = "month",
                          points: int = 30) -> dict:
    """Account value and PnL history for a period (last `points` samples) and its volume."""
    u = user_or_default(user)
    data = dict(client().info({"type": "portfolio", "user": u}) or [])
    p = data.get(period) or {}
    av = p.get("accountValueHistory") or []
    pnl = p.get("pnlHistory") or []
    n = max(int(points), 1)
    return {"user": u, "period": period, "volume": num(p.get("vlm")),
            "account_value": [[iso(t), num(v)] for t, v in av[-n:]],
            "pnl": [[iso(t), num(v)] for t, v in pnl[-n:]],
            "available_periods": sorted(data)}


@mcp.tool(annotations=READ)
def account_get_asset_state(coin: str, user: str | None = None) -> dict:
    """Per-coin trading state: current leverage and margin mode, max trade sizes
    (buy, sell) and available-to-trade amounts at the current mark price."""
    u = user_or_default(user)
    inst = resolve(coin)
    return {"user": u, **(client().info({"type": "activeAssetData", "user": u,
                                         "coin": inst.coin}) or {})}


@mcp.tool(annotations=READ)
def account_get_sub_accounts(user: str | None = None) -> dict:
    """Sub-accounts of a master account, each with its perp margin summary and spot balances."""
    u = user_or_default(user)
    subs = client().info({"type": "subAccounts", "user": u}) or []
    rows = []
    for s in subs:
        st = s.get("clearinghouseState") or {}
        rows.append({
            "name": s.get("name"), "address": s.get("subAccountUser"), "master": s.get("master"),
            "margin": _margin(st), "positions": _positions(st),
            "spot_balances": [b for b in (s.get("spotState") or {}).get("balances") or []
                              if (num(b.get("total")) or 0) != 0],
        })
    return {"user": u, "count": len(rows), "sub_accounts": rows}


@mcp.tool(annotations=READ)
def account_get_vault_equities(user: str | None = None) -> dict:
    """The user's deposits in vaults (equity per vault, lock-ups)."""
    u = user_or_default(user)
    return {"user": u, "vaults": client().info({"type": "userVaultEquities", "user": u}) or [],
            "leading_vaults": client().info({"type": "leadingVaults", "user": u}) or []}


@mcp.tool(annotations=READ)
def account_get_staking(user: str | None = None, include_rewards: bool = False,
                        include_history: bool = False, limit: int = 50) -> dict:
    """HYPE staking: delegated / undelegated / pending withdrawals, delegations per
    validator, and optionally rewards and delegation history."""
    u = user_or_default(user)
    c = client()
    out: dict[str, Any] = {
        "user": u,
        "summary": c.info({"type": "delegatorSummary", "user": u}),
        "delegations": [{**d, "locked_until_utc": iso(d.get("lockedUntilTimestamp"))}
                        for d in c.info({"type": "delegations", "user": u}) or []],
    }
    if include_rewards:
        out["rewards"] = (c.info({"type": "delegatorRewards", "user": u}) or [])[-limit:]
    if include_history:
        out["history"] = (c.info({"type": "delegatorHistory", "user": u}) or [])[:limit]
    return out


@mcp.tool(annotations=READ)
def account_get_borrow_lend_state(user: str | None = None) -> dict:
    """Supplied and borrowed balances per token, health and health factor."""
    u = user_or_default(user)
    st = client().info({"type": "borrowLendUserState", "user": u}) or {}
    rows = []
    for idx, s in st.get("tokenToState") or []:
        try:
            name = client().universe.token(int(idx))["name"]
        except LookupError:
            name = None
        rows.append({"token": name, "index": idx, **s})
    return {"user": u, "health": st.get("health"), "health_factor": st.get("healthFactor"),
            "tokens": rows}


@mcp.tool(annotations=READ)
def account_get_role(user: str | None = None) -> dict:
    """Account type (user / agent / vault / subAccount / missing) and margining mode
    (default, unifiedAccount, portfolioMargin, dexAbstraction...)."""
    u = user_or_default(user)
    c = client()
    out: dict[str, Any] = {"user": u, "role": c.info({"type": "userRole", "user": u})}
    for key, typ in (("abstraction", "userAbstraction"),
                     ("hip3_dex_abstraction", "userDexAbstraction")):
        try:
            out[key] = c.info({"type": typ, "user": u})
        except Exception as e:
            out[key] = {"error": str(e)}
    return out


@mcp.tool(annotations=READ)
def account_get_api_wallets(user: str | None = None) -> dict:
    """API (agent) wallets approved by this account, with names and expiry."""
    u = user_or_default(user)
    agents = client().info({"type": "extraAgents", "user": u}) or []
    return {"user": u, "api_wallets": [{**a, "valid_until_utc": iso(a.get("validUntil"))}
                                       for a in agents]}


@mcp.tool(annotations=READ)
def account_get_referral(user: str | None = None) -> dict:
    """Referral state: who referred the user, rewards, and the user's own referral code."""
    u = user_or_default(user)
    r = client().info({"type": "referral", "user": u}) or {}
    r.pop("rewardHistory", None)  # legacy; claimed rewards live in the ledger
    return {"user": u, **r}


@mcp.tool(annotations=READ)
def account_get_builder_fee_approvals(user: str | None = None, builder: str | None = None) -> dict:
    """Builders this account approved to charge fees; with `builder`, that builder's
    approved max fee (in tenths of a basis point)."""
    u = user_or_default(user)
    out: dict[str, Any] = {"user": u,
                           "approved_builders": client().info({"type": "approvedBuilders",
                                                                "user": u})}
    if builder:
        b = address(builder, "builder")
        out["max_builder_fee_tenths_bp"] = client().info(
            {"type": "maxBuilderFee", "user": u, "builder": b})
    return out


@mcp.tool(annotations=READ)
def account_get_multisig_signers(user: str | None = None) -> dict:
    """Authorized signers and threshold if the account was converted to multi-sig."""
    u = user_or_default(user)
    return {"user": u, "signers": client().info({"type": "userToMultiSigSigners", "user": u})}
