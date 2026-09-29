"""Trading tools (write module `trade`): orders, cancels, positions, leverage, TWAP.

One order action serves perps (main + HIP-3), spot and outcomes; the coin
name picks the instrument. Every tool accepts `dry_run=True` to build and
guard the exact action without signing it.

Guards on top of the module/network gates (see guard.py):
  * sizes are floored to the lot size, prices snapped to a valid tick in the
    trader's favour (buys down, sells up) unless round_to_tick=False;
  * opening orders are capped by HL_MAX_ORDER_NOTIONAL_USD (per order and per
    batch) and HL_MAX_DAILY_NOTIONAL_USD (per UTC day), and need the coin's
    current leverage <= HL_MAX_LEVERAGE. Exposure is valued conservatively:
    buys at their limit, sells at max(limit, mid) since a low sell limit
    still fills at the bid;
  * market orders are IOC limits at mid ± slippage, capped by HL_MAX_SLIPPAGE;
    limit orders may not sit more than HL_MAX_SLIPPAGE through the mid;
  * market TP/SL triggers carry a worst-fill limit of trigger ± slippage:
    10% for exits (Hyperliquid's own default — a stop-loss must fill),
    HL_DEFAULT_SLIPPAGE for entries (capped by HL_MAX_SLIPPAGE);
  * reduce-only PERP orders skip the notional caps and coin allowlist (exits
    first). reduce_only means nothing on spot/outcomes and is cleared there.
Every order carries a client order id (generated when not given) so that an
order whose send timed out can still be found with account_get_order_status.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from hyperliquid.utils.signing import float_to_usd_int

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid import guard
from trading_agent.mcp_servers.hyperliquid.core import (
    WRITE,
    address,
    client,
    destination_reasons,
    execute,
    leverage_reasons,
    mcp,
    mid,
    notional_usd,
    now_ms,
    num,
    require_target,
    resolve,
    trade_target,
)
from trading_agent.mcp_servers.hyperliquid.universe import (
    Instrument,
    fmt_decimal,
    is_valid_price,
    round_price,
    round_size,
    to_decimal,
)

# Hyperliquid's frontend sends market TP/SL with a worst-fill limit 10% past
# the trigger; TWAP slices are capped at 3% by the exchange.
TRIGGER_EXIT_SLIPPAGE = 0.10
TWAP_SLICE_SLIPPAGE = Decimal("0.03")
OUTCOME_MIN_PX = Decimal("0.00001")
OUTCOME_MAX_PX = Decimal("0.99999")
ORDER_KEYS = {"coin", "side", "size", "type", "price", "tif", "reduce_only", "trigger_price",
              "trigger_type", "cloid", "slippage", "round_to_tick"}


@dataclass
class Prepared:
    inst: Instrument
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    order_type: dict
    reduce_only: bool          # effective flag: only ever True on perps
    cloid: str
    notional: Decimal | None   # conservative USD exposure if it fills
    slippage: float | None = None      # slippage this order was priced with...
    slippage_cap: float | None = None  # ...and the most it may use
    band_ref: Decimal | None = None    # price the limit must stay within HL_MAX_SLIPPAGE of
    band_what: str | None = None       # "mid" / "trigger"; None = no band check
    notes: list[str] = field(default_factory=list)

    @property
    def is_exit(self) -> bool:
        return guard.is_exit(self.inst, self.reduce_only)

    def wire(self) -> dict:
        return A.order_wire(self.inst.asset, self.is_buy, float(self.size), float(self.limit_px),
                            self.order_type, self.reduce_only, self.cloid)

    def summary(self) -> dict:
        d: dict[str, Any] = {
            "coin": self.inst.coin, "kind": self.inst.kind,
            "side": "buy" if self.is_buy else "sell",
            "size": fmt_decimal(self.size), "limit_px": fmt_decimal(self.limit_px),
            "order_type": self.order_type, "reduce_only": self.reduce_only, "cloid": self.cloid,
            "notional_usd": None if self.notional is None else round(float(self.notional), 2),
        }
        if self.notes:
            d["notes"] = self.notes
        return d


def _is_buy(side: str) -> bool:
    s = side.strip().lower()
    if s in ("buy", "long", "b"):
        return True
    if s in ("sell", "short", "s", "a"):
        return False
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def _cloid(cloid: str | None) -> str:
    if cloid is None:
        return "0x" + secrets.token_hex(16)
    c = cloid.strip().lower()
    if not (c.startswith("0x") and len(c) == 34 and all(ch in "0123456789abcdef" for ch in c[2:])):
        raise ValueError("cloid must be 0x followed by 32 hex characters (16 bytes)")
    return c


def _flag(value: Any, name: str) -> bool:
    """Strict boolean: bool("false") is True, so strings must say true/false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _reduce_only(inst: Instrument, requested: bool, notes: list[str]) -> bool:
    """reduce_only only exists for perps; on spot/outcomes it is cleared so it
    can neither be rejected by the exchange nor exempt the order from guards."""
    if requested and not inst.is_perp:
        notes.append(f"reduce_only ignored: {inst.kind} has no positions to reduce")
        return False
    return requested


def _limit_px(inst: Instrument, price: Any, is_buy: bool, round_to_tick: bool,
              notes: list[str]) -> Decimal:
    raw = to_decimal(price, "price")
    if round_to_tick:
        px = round_price(raw, inst, "down" if is_buy else "up")
        if px != raw:
            notes.append(f"price {fmt_decimal(raw)} snapped to tick {fmt_decimal(px)}")
        return px
    if not is_valid_price(raw, inst):
        lo, hi = round_price(raw, inst, "down"), round_price(raw, inst, "up")
        raise ValueError(
            f"price {price} is not a valid tick for {inst.coin} (max 5 significant figures and "
            f"{inst.price_max_decimals} decimals); nearest valid: {fmt_decimal(lo)} / "
            f"{fmt_decimal(hi)}"
        )
    return raw


def _size(inst: Instrument, size: Any, notes: list[str]) -> Decimal:
    raw = to_decimal(size, "size")
    sz = round_size(raw, inst)
    if sz != raw:
        notes.append(f"size {fmt_decimal(raw)} floored to lot size {fmt_decimal(sz)}")
    return sz


def _market_px(inst: Instrument, is_buy: bool, slippage: float,
               ref: Decimal | None = None) -> Decimal:
    """IOC limit at ref × (1 ± slippage); ref defaults to the live mid."""
    ref = ref if ref is not None else mid(inst)
    if ref is None or ref <= 0:
        raise ValueError(f"no mid price for {inst.coin}; use a limit order instead")
    slip = Decimal(str(slippage))
    raw = ref * (1 + slip) if is_buy else ref * (1 - slip)
    if inst.kind == "outcome":
        raw = min(max(raw, OUTCOME_MIN_PX), OUTCOME_MAX_PX)
    # Buys round down, sells up: the IOC limit never exceeds the slippage bound.
    return round_price(raw, inst, "down" if is_buy else "up")


def _slip_cap(exit_: bool) -> float:
    """Entries are bounded by HL_MAX_SLIPPAGE; perp exits may use at least 10%
    so a stop can still fill in a fast market."""
    s = client().settings
    return max(s.max_slippage, TRIGGER_EXIT_SLIPPAGE) if exit_ else s.max_slippage


def _slippage(value: float | None, default: float | None = None) -> float:
    fallback = client().settings.default_slippage if default is None else default
    s = fallback if value is None else float(value)
    if not 0 < s < 1:
        raise ValueError("slippage must be a fraction between 0 and 1 (0.01 = 1%)")
    return s


def _exposure(inst: Instrument, is_buy: bool, size: Decimal, price: Decimal,
              floor: Decimal | None = None) -> Decimal | None:
    """Conservative USD value if the order fills. A buy fills at or below its
    limit; a sell fills at or ABOVE it — up to the bid — so a sell is valued at
    max(price, floor) where floor is the mid (or the trigger)."""
    ref = price if is_buy or floor is None else max(price, floor)
    return notional_usd(inst, size, ref)


def prepare_limit(inst: Instrument, is_buy: bool, size: Any, price: Any, tif: str,
                  reduce_only: bool, cloid: str | None, round_to_tick: bool = True) -> Prepared:
    notes: list[str] = []
    sz = _size(inst, size, notes)
    px = _limit_px(inst, price, is_buy, round_to_tick, notes)
    ro = _reduce_only(inst, reduce_only, notes)
    ref = mid(inst)
    return Prepared(inst, is_buy, sz, px, {"limit": {"tif": tif}}, ro, _cloid(cloid),
                    _exposure(inst, is_buy, sz, px, ref), band_ref=ref, band_what="mid",
                    notes=notes)


def prepare_market(inst: Instrument, is_buy: bool, size: Any, slippage: float | None,
                   reduce_only: bool, cloid: str | None, ref: Decimal | None = None) -> Prepared:
    notes: list[str] = []
    sz = _size(inst, size, notes)
    slip = _slippage(slippage)
    ro = _reduce_only(inst, reduce_only, notes)
    ref = ref if ref is not None else mid(inst)
    px = _market_px(inst, is_buy, slip, ref)
    notes.append(f"market order = IOC limit at {fmt_decimal(px)} (reference ± {slip:.2%})")
    return Prepared(inst, is_buy, sz, px, {"limit": {"tif": "Ioc"}}, ro, _cloid(cloid),
                    _exposure(inst, is_buy, sz, px, ref), slippage=slip,
                    slippage_cap=_slip_cap(guard.is_exit(inst, ro)), notes=notes)


def prepare_trigger(inst: Instrument, is_buy: bool, size: Any, trigger_price: Any,
                    trigger_type: str, limit_price: Any | None, reduce_only: bool,
                    cloid: str | None, round_to_tick: bool = True,
                    slippage: float | None = None, whole_position: bool = False) -> Prepared:
    """TP/SL trigger on the mark price. Market triggers still carry a limit:
    the worst fill, trigger ± slippage (10% for perp exits, like Hyperliquid's
    frontend; HL_DEFAULT_SLIPPAGE for entries). `whole_position` sends size 0,
    which the exchange reads as "the entire position, resizing with it"."""
    if trigger_type not in ("stop", "take_profit"):
        raise ValueError("trigger_type must be 'stop' or 'take_profit'")
    notes: list[str] = []
    ro = _reduce_only(inst, reduce_only, notes)
    if whole_position:
        if not (inst.is_perp and ro):
            raise ValueError("whole-position TP/SL is only for reduce-only perp orders")
        sz = Decimal(0)
        notes.append("size 0 = the whole position; resizes with it")
    else:
        sz = _size(inst, size, notes)
    raw_trig = to_decimal(trigger_price, "trigger_price")
    if not round_to_tick and not is_valid_price(raw_trig, inst):
        raise ValueError(f"trigger_price {trigger_price} is not a valid tick for {inst.coin}; "
                         f"nearest: {fmt_decimal(round_price(raw_trig, inst, 'nearest'))}")
    trig = round_price(raw_trig, inst, "nearest") if round_to_tick else raw_trig
    if trig != raw_trig:
        notes.append(f"trigger {fmt_decimal(raw_trig)} snapped to tick {fmt_decimal(trig)}")
    order_type = {"trigger": {"triggerPx": float(trig), "isMarket": limit_price is None,
                              "tpsl": "sl" if trigger_type == "stop" else "tp"}}
    if limit_price is None:
        exit_ = guard.is_exit(inst, ro)
        slip = _slippage(slippage, TRIGGER_EXIT_SLIPPAGE if exit_
                         else client().settings.default_slippage)
        px = _market_px(inst, is_buy, slip, trig)
        notes.append(f"market trigger: worst fill {fmt_decimal(px)} (trigger ± {slip:.2%})")
        return Prepared(inst, is_buy, sz, px, order_type, ro, _cloid(cloid),
                        _exposure(inst, is_buy, sz, px, trig), slippage=slip,
                        slippage_cap=_slip_cap(exit_), notes=notes)
    px = _limit_px(inst, limit_price, is_buy, round_to_tick, notes)
    return Prepared(inst, is_buy, sz, px, order_type, ro, _cloid(cloid),
                    _exposure(inst, is_buy, sz, px, trig), band_ref=trig,
                    band_what="trigger", notes=notes)


def _order_reasons(preps: list[Prepared], target: str | None) -> list[str]:
    s = client().settings
    reasons: list[str] = []
    for p in preps:
        reasons += guard.order_block_reasons(s, p.inst, reduce_only=p.reduce_only,
                                             notional_usd=p.notional)
        cap = p.slippage_cap if p.slippage_cap is not None else _slip_cap(p.is_exit)
        if p.slippage is not None and p.slippage > cap:
            reasons.append(f"{p.inst.coin}: slippage {p.slippage:.4f} exceeds the allowed "
                           f"{cap} (HL_MAX_SLIPPAGE)")
        if p.band_what is not None:
            reasons += guard.price_band_reasons(s, p.inst, is_buy=p.is_buy, limit_px=p.limit_px,
                                                reference=p.band_ref, what=p.band_what,
                                                band=_slip_cap(p.is_exit))
    opening = [p for p in preps if not p.is_exit]
    total = _opening_total(preps)
    if len(opening) > 1 and total > Decimal(str(s.max_order_notional_usd)):
        reasons.append(f"batch opening notional ${total:,.2f} exceeds "
                       f"HL_MAX_ORDER_NOTIONAL_USD ${s.max_order_notional_usd:,.2f}")
    checked: set[str] = set()
    for p in opening:
        if p.inst.coin not in checked:
            checked.add(p.inst.coin)
            reasons += leverage_reasons(p.inst, target, reduce_only=False)
    return list(dict.fromkeys(reasons))


def _opening_total(preps: list[Prepared]) -> Decimal:
    return sum((p.notional for p in preps if not p.is_exit and p.notional is not None),
               Decimal(0))


def _describe_statuses(out: dict, preps: list[Prepared]) -> dict:
    if out.get("status") != "sent":
        return out
    results = []
    for i, st in enumerate(out.get("statuses") or []):
        r: dict[str, Any] = {"cloid": preps[i].cloid if i < len(preps) else None}
        if isinstance(st, dict) and "resting" in st:
            r.update(result="resting", oid=st["resting"].get("oid"))
        elif isinstance(st, dict) and "filled" in st:
            f = st["filled"]
            r.update(result="filled", oid=f.get("oid"), filled_size=num(f.get("totalSz")),
                     avg_px=num(f.get("avgPx")))
        elif isinstance(st, dict) and "error" in st:
            r.update(result="rejected", error=st["error"])
        else:
            r.update(result=st)  # e.g. "waitingForFill" / "waitingForTrigger" for TP/SL legs
        results.append(r)
    out["order_results"] = results
    return out


def _send_orders(tool: str, preps: list[Prepared], vault: str | None, target: str | None, *,
                 grouping: Any = "na", builder: dict | None = None, dry_run: bool) -> dict:
    req = A.orders([p.wire() for p in preps], client().next_nonce(), grouping=grouping,
                   builder=builder, vault_address=vault)
    summary = {"orders": [p.summary() for p in preps], "grouping": grouping,
               "account": target, **({"vault_address": vault} if vault else {})}
    reasons = _order_reasons(preps, target)
    if builder:
        # A builder code pays the builder a fee on every fill: a payee like any other.
        reasons += destination_reasons(builder["b"], "builder")
    out = execute(tool, "trade", req, dry_run=dry_run, reasons=reasons, summary=summary,
                  opening_notional_usd=_opening_total(preps))
    return _describe_statuses(out, preps)


# --------------------------------------------------------------------------
# Order placement
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def order_place_limit(coin: str, side: Literal["buy", "sell"], size: float, price: float,
                      tif: Literal["Gtc", "Alo", "Ioc"] = "Gtc", reduce_only: bool = False,
                      cloid: str | None = None, round_to_tick: bool = True,
                      vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Place a limit order on a perp (e.g. "BTC", "xyz:TSLA"), spot pair ("HYPE/USDC")
    or outcome side ("#12090").

    `size` is in base units (coins / shares), `price` in the quote currency.
    `tif`: Gtc rests until cancelled, Alo is post-only, Ioc fills-or-cancels.
    `vault_address` trades for a vault or sub-account you manage (defaults to
    HL_VAULT_ADDRESS). Use dry_run=True to preview without signing.
    """
    inst = resolve(coin)
    vault, target = trade_target(vault_address)
    p = prepare_limit(inst, _is_buy(side), size, price, tif, reduce_only, cloid, round_to_tick)
    return _send_orders("order_place_limit", [p], vault, target, dry_run=dry_run)


@mcp.tool(annotations=WRITE)
def order_place_market(coin: str, side: Literal["buy", "sell"], size: float,
                       slippage: float | None = None, reduce_only: bool = False,
                       cloid: str | None = None, vault_address: str | None = None,
                       dry_run: bool = False) -> dict:
    """Market order: an immediate-or-cancel limit at mid × (1 ± slippage).

    `slippage` is a fraction (0.01 = 1%; default HL_DEFAULT_SLIPPAGE, capped by
    HL_MAX_SLIPPAGE). Any unfilled remainder is cancelled, never left resting.
    """
    inst = resolve(coin)
    vault, target = trade_target(vault_address)
    p = prepare_market(inst, _is_buy(side), size, slippage, reduce_only, cloid)
    return _send_orders("order_place_market", [p], vault, target, dry_run=dry_run)


@mcp.tool(annotations=WRITE)
def order_place_trigger(coin: str, side: Literal["buy", "sell"], size: float,
                        trigger_price: float, trigger_type: Literal["stop", "take_profit"],
                        limit_price: float | None = None, reduce_only: bool = True,
                        slippage: float | None = None, cloid: str | None = None,
                        round_to_tick: bool = True, vault_address: str | None = None,
                        dry_run: bool = False) -> dict:
    """Stop / take-profit order triggered by the MARK price.

    `trigger_type="stop"` fires when price moves through the trigger in the
    direction of `side` (a sell stop fires on a fall, a buy stop on a rise);
    "take_profit" is the opposite. With a `limit_price` it becomes a limit
    order once triggered. Without one it executes as a market order no worse
    than trigger ± `slippage`: 10% by default for perp exits (as on
    Hyperliquid's own frontend), HL_DEFAULT_SLIPPAGE for entries.
    `reduce_only` defaults to True (a protective perp exit); set False for stop
    entries. It has no effect on spot/outcomes. For a TP/SL on an existing
    position prefer position_set_tpsl.
    """
    inst = resolve(coin)
    vault, target = trade_target(vault_address)
    p = prepare_trigger(inst, _is_buy(side), size, trigger_price, trigger_type, limit_price,
                        reduce_only, cloid, round_to_tick, slippage=slippage)
    return _send_orders("order_place_trigger", [p], vault, target, dry_run=dry_run)


def _check_bracket(is_buy: bool, ref: Decimal, tp: Any | None, sl: Any | None) -> None:
    if tp is not None:
        tpd = to_decimal(tp, "take_profit_price")
        if (is_buy and tpd <= ref) or (not is_buy and tpd >= ref):
            raise ValueError(f"take_profit_price {tp} must be {'above' if is_buy else 'below'} "
                             f"the entry reference {fmt_decimal(ref)} for a "
                             f"{'long' if is_buy else 'short'}")
    if sl is not None:
        sld = to_decimal(sl, "stop_loss_price")
        if (is_buy and sld >= ref) or (not is_buy and sld <= ref):
            raise ValueError(f"stop_loss_price {sl} must be {'below' if is_buy else 'above'} "
                             f"the entry reference {fmt_decimal(ref)} for a "
                             f"{'long' if is_buy else 'short'}")


@mcp.tool(annotations=WRITE)
def order_place_bracket(coin: str, side: Literal["buy", "sell"], size: float,
                        entry_price: float | None = None, take_profit_price: float | None = None,
                        stop_loss_price: float | None = None,
                        tif: Literal["Gtc", "Alo", "Ioc"] = "Gtc", slippage: float | None = None,
                        cloid: str | None = None, vault_address: str | None = None,
                        dry_run: bool = False) -> dict:
    """Entry order with attached take-profit and/or stop-loss (one-cancels-other).

    Entry is a limit at `entry_price`, or a market (IOC) order when omitted.
    The TP/SL legs are reduce-only market triggers for the same size and only
    go live once the entry fully fills. Perps only.
    """
    if take_profit_price is None and stop_loss_price is None:
        raise ValueError("give take_profit_price and/or stop_loss_price")
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError("brackets are for perps; for spot place the legs separately")
    vault, target = trade_target(vault_address)
    is_buy = _is_buy(side)
    if entry_price is None:
        entry = prepare_market(inst, is_buy, size, slippage, False, cloid)
        ref = mid(inst) or entry.limit_px
    else:
        entry = prepare_limit(inst, is_buy, size, entry_price, tif, False, cloid)
        ref = entry.limit_px
    _check_bracket(is_buy, ref, take_profit_price, stop_loss_price)
    legs = [entry]
    if take_profit_price is not None:
        legs.append(prepare_trigger(inst, not is_buy, entry.size, take_profit_price,
                                    "take_profit", None, True, None))
    if stop_loss_price is not None:
        legs.append(prepare_trigger(inst, not is_buy, entry.size, stop_loss_price,
                                    "stop", None, True, None))
    return _send_orders("order_place_bracket", legs, vault, target, grouping="normalTpsl",
                        dry_run=dry_run)


@mcp.tool(annotations=WRITE)
def order_place_batch(orders: list[dict],
                      grouping: Literal["na", "normalTpsl", "positionTpsl"] = "na",
                      builder_address: str | None = None,
                      builder_fee_tenths_bp: int | None = None,
                      vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Place several orders in one signed action (one nonce, one round trip).

    Each order: {"coin", "side": "buy"|"sell", "size", "type": "limit"|"market"|"trigger",
    "price" (limit, or trigger limit), "tif", "reduce_only", "trigger_price",
    "trigger_type": "stop"|"take_profit", "cloid", "slippage", "round_to_tick"}.
    The notional cap applies to each order and to the batch's opening total.
    `builder_address` + `builder_fee_tenths_bp` attach a builder code: the
    builder is paid a fee on every fill, so it must be approved first
    (builder_fee_approve) and be your own address or in HL_WITHDRAW_ALLOWLIST.
    """
    if not orders:
        raise ValueError("orders is empty")
    vault, target = trade_target(vault_address)
    preps: list[Prepared] = []
    for i, o in enumerate(orders):
        unknown = set(o) - ORDER_KEYS
        if unknown:
            raise ValueError(f"order[{i}] has unknown keys {sorted(unknown)}")
        missing = {"coin", "side", "size"} - set(o)
        if missing:
            raise ValueError(f"order[{i}] is missing {sorted(missing)}")
        inst = resolve(o["coin"])
        is_buy = _is_buy(o["side"])
        kind = o.get("type", "limit")
        ro = _flag(o.get("reduce_only", False), f"order[{i}].reduce_only")
        rt = _flag(o.get("round_to_tick", True), f"order[{i}].round_to_tick")
        if kind == "limit":
            if "price" not in o:
                raise ValueError(f"order[{i}] is a limit order without a price")
            preps.append(prepare_limit(inst, is_buy, o["size"], o["price"], o.get("tif", "Gtc"),
                                       ro, o.get("cloid"), rt))
        elif kind == "market":
            preps.append(prepare_market(inst, is_buy, o["size"], o.get("slippage"), ro,
                                        o.get("cloid")))
        elif kind == "trigger":
            if "trigger_price" not in o:
                raise ValueError(f"order[{i}] is a trigger order without a trigger_price")
            preps.append(prepare_trigger(inst, is_buy, o["size"], o["trigger_price"],
                                         o.get("trigger_type", "stop"), o.get("price"), ro,
                                         o.get("cloid"), rt, slippage=o.get("slippage")))
        else:
            raise ValueError(f"order[{i}].type must be limit, market or trigger")
    builder = None
    if builder_address:
        if builder_fee_tenths_bp is None or not 0 <= int(builder_fee_tenths_bp) <= 1000:
            raise ValueError("builder_fee_tenths_bp must be 0-1000 (100 max for perps)")
        builder = {"b": address(builder_address, "builder_address"),
                   "f": int(builder_fee_tenths_bp)}
    return _send_orders("order_place_batch", preps, vault, target, grouping=grouping,
                        builder=builder, dry_run=dry_run)


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

def _position_rows(target: str, dex: str) -> list[dict]:
    st = client().info({"type": "clearinghouseState", "user": target, "dex": dex}) or {}
    return [ap.get("position") or {} for ap in st.get("assetPositions") or []]


def _position(inst: Instrument, target: str) -> tuple[Decimal, Decimal | None]:
    """(signed size szi, reference price) of a perp for `target`; (0, None) if flat."""
    for p in _position_rows(target, inst.dex):
        if p.get("coin") == inst.coin:
            return to_decimal(p.get("szi", "0"), "szi"), _exit_ref(inst, p)
    return Decimal(0), None


def _exit_ref(inst: Instrument, pos: dict) -> Decimal | None:
    """Price to close against: the live mid, else the mark implied by the
    position itself (positionValue / |szi|), so an exit never depends on a
    mid being available."""
    m = mid(inst)
    if m is not None and m > 0:
        return m
    try:
        value = to_decimal(pos.get("positionValue"), "positionValue")
        szi = abs(to_decimal(pos.get("szi"), "szi"))
        return value / szi if szi > 0 and value > 0 else None
    except ValueError:
        return None


@mcp.tool(annotations=WRITE)
def position_close(coin: str, size: float | None = None, slippage: float | None = None,
                   vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Close a perp position (all of it, or `size`) with a reduce-only market order.

    Reduce-only means it can only shrink the position, so it is never blocked
    by the notional caps or coin allowlist. `slippage` may go up to
    max(HL_MAX_SLIPPAGE, 10%) for exits.
    """
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError(f"{inst.coin} is not a perp; sell spot/outcome balances with "
                         "order_place_market")
    vault, target = trade_target(vault_address)
    szi, ref = _position(inst, require_target(target))
    if szi == 0:
        return {"tool": "position_close", "status": "nothing_to_close", "sent": False,
                "coin": inst.coin, "account": target}
    qty = abs(szi) if size is None else min(to_decimal(size, "size"), abs(szi))
    p = prepare_market(inst, szi < 0, qty, slippage, True, None, ref=ref)
    return _send_orders("position_close", [p], vault, target, dry_run=dry_run)


@mcp.tool(annotations=WRITE)
def position_close_all(all_dexs: bool = True, slippage: float | None = None,
                       vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Close every open perp position (main dex and, by default, every HIP-3 dex)
    with reduce-only market orders in a single action. A position that cannot
    be priced or resolved is reported in `not_closed` instead of stopping the
    others from closing."""
    vault, target = trade_target(vault_address)
    target = require_target(target)
    dexs = client().universe.dex_names() if all_dexs else [""]
    preps: list[Prepared] = []
    not_closed: list[dict] = []
    for d in dexs:
        try:
            rows = _position_rows(target, d)
        except Exception as e:
            not_closed.append({"dex": d, "error": f"cannot read positions: {e}"})
            continue
        for pos in rows:
            try:
                szi = to_decimal(pos.get("szi", "0"), "szi")
                if szi == 0:
                    continue
                inst = resolve(pos["coin"])
                preps.append(prepare_market(inst, szi < 0, abs(szi), slippage, True, None,
                                            ref=_exit_ref(inst, pos)))
            except Exception as e:
                not_closed.append({"coin": pos.get("coin"), "dex": d, "error": str(e)})
    if not preps:
        return {"tool": "position_close_all", "status": "nothing_to_close" if not not_closed
                else "error", "sent": False, "account": target, "not_closed": not_closed}
    out = _send_orders("position_close_all", preps, vault, target, dry_run=dry_run)
    if not_closed:
        out["not_closed"] = not_closed
    return out


@mcp.tool(annotations=WRITE)
def position_set_tpsl(coin: str, take_profit_price: float | None = None,
                      stop_loss_price: float | None = None, size: float | None = None,
                      take_profit_limit_price: float | None = None,
                      stop_loss_limit_price: float | None = None,
                      vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Attach take-profit and/or stop-loss orders to an EXISTING perp position.

    Without `size` they cover the whole position and resize with it (sent as
    size 0, like Hyperliquid's own position TP/SL); with `size` they are fixed.
    Legs are reduce-only triggers on the mark price: market (worst fill 10%
    past the trigger) unless a *_limit_price is given. TP must be on the
    profit side of the current mark, SL on the loss side.
    """
    if take_profit_price is None and stop_loss_price is None:
        raise ValueError("give take_profit_price and/or stop_loss_price")
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError("position TP/SL is for perps")
    vault, target = trade_target(vault_address)
    szi, ref = _position(inst, require_target(target))
    if szi == 0:
        raise ValueError(f"no open {inst.coin} position for {target}")
    is_long = szi > 0
    whole = size is None
    qty = Decimal(0) if whole else to_decimal(size, "size")
    if qty > abs(szi):
        raise ValueError(f"size {size} exceeds the position size {fmt_decimal(abs(szi))}")
    if ref is not None:
        _check_bracket(is_long, ref, take_profit_price, stop_loss_price)
    legs = []
    if take_profit_price is not None:
        legs.append(prepare_trigger(inst, not is_long, qty, take_profit_price, "take_profit",
                                    take_profit_limit_price, True, None, whole_position=whole))
    if stop_loss_price is not None:
        legs.append(prepare_trigger(inst, not is_long, qty, stop_loss_price, "stop",
                                    stop_loss_limit_price, True, None, whole_position=whole))
    return _send_orders("position_set_tpsl", legs, vault, target, grouping="positionTpsl",
                        dry_run=dry_run)


# --------------------------------------------------------------------------
# Modify / cancel
# --------------------------------------------------------------------------

def _open_order(target: str, oid: int | None, cloid: str | None) -> dict:
    if (oid is None) == (cloid is None):
        raise ValueError("pass exactly one of oid or cloid")
    ref: Any = oid if oid is not None else _cloid(cloid)
    res = client().info({"type": "orderStatus", "user": target, "oid": ref}) or {}
    if res.get("status") != "order":
        raise ValueError(f"order {ref} not found for {target} ({res.get('status')})")
    wrapped = res.get("order") or {}
    if wrapped.get("status") != "open":
        raise ValueError(f"order {ref} is {wrapped.get('status')}, not open")
    return wrapped.get("order") or {}


@mcp.tool(annotations=WRITE)
def order_modify(oid: int | None = None, cloid: str | None = None, price: float | None = None,
                 size: float | None = None, trigger_price: float | None = None,
                 tif: Literal["Gtc", "Alo", "Ioc"] | None = None, round_to_tick: bool = True,
                 always_place: bool = False, vault_address: str | None = None,
                 dry_run: bool = False) -> dict:
    """Change an open order's price, size, trigger price or TIF, keeping everything else.

    Identify it by `oid` or `cloid`; the current order is read from the
    exchange and only the fields you pass are changed.

    Default (always_place=False) is the exchange's safe mode: the replacement
    is placed only if cancelling the original succeeded, and it must be a
    resting (non-marketable) limit order. Trigger (TP/SL) orders and moves to a
    marketable price need always_place=True, which places the replacement
    EVEN IF the original already filled — check fills afterwards.
    """
    vault, target = trade_target(vault_address)
    o = _open_order(require_target(target), oid, cloid)
    if o.get("isTrigger") and not always_place:
        raise ValueError("trigger (TP/SL) orders can only be modified with always_place=True, "
                         "or cancelled and placed again")
    inst = resolve(o["coin"])
    is_buy = o.get("side") == "B"
    reduce_only = bool(o.get("reduceOnly"))
    new_size = size if size is not None else o.get("sz")
    keep_cloid = o.get("cloid") or None
    # A whole-position TP/SL reports size 0 and must keep it (it resizes with the position).
    whole = (bool(o.get("isPositionTpsl")) and size is None
             and to_decimal(o.get("sz") or "0", "sz") == 0)
    if o.get("isTrigger"):
        otype = str(o.get("orderType") or "")
        p = prepare_trigger(inst, is_buy, new_size,
                            trigger_price if trigger_price is not None else o.get("triggerPx"),
                            "take_profit" if "Take Profit" in otype else "stop",
                            None if "Market" in otype else (price if price is not None
                                                            else o.get("limitPx")),
                            reduce_only, keep_cloid, round_to_tick, whole_position=whole)
    else:
        cur_tif = o.get("tif") if o.get("tif") in ("Gtc", "Alo", "Ioc") else "Gtc"
        p = prepare_limit(inst, is_buy, new_size, price if price is not None else o.get("limitPx"),
                          tif or cur_tif, reduce_only, keep_cloid, round_to_tick)
    # Only the increase over the order being replaced counts towards the daily budget.
    old = None
    try:
        old = _exposure(inst, is_buy, to_decimal(o.get("sz") or "0", "sz"),
                        to_decimal(o.get("limitPx") or "0", "limitPx"), mid(inst))
    except ValueError:
        pass
    added = _opening_total([p]) - (old or Decimal(0))
    req = A.batch_modify([(int(o["oid"]), p.wire())], client().next_nonce(),
                         always_place=always_place, vault_address=vault)
    out = execute("order_modify", "trade", req, dry_run=dry_run,
                  reasons=_order_reasons([p], target),
                  summary={"oid": o["oid"], "from": {"limit_px": o.get("limitPx"),
                                                     "size": o.get("sz"),
                                                     "trigger_px": o.get("triggerPx")},
                           "to": p.summary()},
                  opening_notional_usd=max(added, Decimal(0)))
    return _describe_statuses(out, [p])


@mcp.tool(annotations=WRITE)
def order_cancel(coin: str, oid: int | None = None, cloid: str | None = None,
                 vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Cancel one order by exchange id (`oid`) or client id (`cloid`)."""
    if (oid is None) == (cloid is None):
        raise ValueError("pass exactly one of oid or cloid")
    inst = resolve(coin)
    vault, _target = trade_target(vault_address)
    nonce = client().next_nonce()
    req = (A.cancel([(inst.asset, int(oid))], nonce, vault_address=vault) if oid is not None
           else A.cancel_by_cloid([(inst.asset, _cloid(cloid))], nonce, vault_address=vault))
    return execute("order_cancel", "trade", req, dry_run=dry_run,
                   summary={"coin": inst.coin, "oid": oid, "cloid": cloid})


@mcp.tool(annotations=WRITE)
def order_cancel_batch(cancels: list[dict], vault_address: str | None = None,
                       dry_run: bool = False) -> dict:
    """Cancel several orders in one action: [{"coin": "BTC", "oid": 123}, ...] or all
    by cloid: [{"coin": "BTC", "cloid": "0x..."}]. Do not mix oid and cloid."""
    if not cancels:
        raise ValueError("cancels is empty")
    by_oid = [c for c in cancels if c.get("oid") is not None]
    by_cloid = [c for c in cancels if c.get("cloid") is not None]
    if by_oid and by_cloid:
        raise ValueError("do not mix oid and cloid cancels in one batch")
    if len(by_oid) + len(by_cloid) != len(cancels):
        raise ValueError("every cancel needs a coin and an oid or cloid")
    vault, _target = trade_target(vault_address)
    nonce = client().next_nonce()
    if by_oid:
        req = A.cancel([(resolve(c["coin"]).asset, int(c["oid"])) for c in by_oid], nonce,
                       vault_address=vault)
    else:
        req = A.cancel_by_cloid([(resolve(c["coin"]).asset, _cloid(c["cloid"])) for c in by_cloid],
                                nonce, vault_address=vault)
    return execute("order_cancel_batch", "trade", req, dry_run=dry_run,
                   summary={"n": len(cancels)})


@mcp.tool(annotations=WRITE)
def order_cancel_all(coin: str | None = None, all_dexs: bool = True,
                     vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Cancel every open order (optionally only for `coin`), across all perp dexs,
    spot and outcomes, including TP/SL triggers. Orders that cannot be mapped
    to an asset are listed in `skipped` rather than blocking the rest."""
    vault, target = trade_target(vault_address)
    target = require_target(target)
    want = resolve(coin) if coin else None
    if want is not None:
        dexs = [want.dex if want.is_perp else ""]
    else:
        dexs = client().universe.dex_names() if all_dexs else [""]
    pairs: list[tuple[int, int]] = []
    skipped: list[dict] = []
    for d in dexs:
        try:
            # frontendOpenOrders includes untriggered TP/SL orders.
            orders = client().info({"type": "frontendOpenOrders", "user": target, "dex": d}) or []
        except Exception as e:
            skipped.append({"dex": d, "error": f"cannot read open orders: {e}"})
            continue
        for o in orders:
            if want is not None and o.get("coin") != want.coin:
                continue
            try:
                pairs.append((resolve(o["coin"]).asset, int(o["oid"])))
            except (ValueError, KeyError) as e:
                skipped.append({"coin": o.get("coin"), "oid": o.get("oid"), "error": str(e)})
    if not pairs:
        return {"tool": "order_cancel_all", "sent": False, "account": target,
                "status": "error" if skipped else "nothing_to_cancel", "skipped": skipped}
    req = A.cancel(pairs, client().next_nonce(), vault_address=vault)
    out = execute("order_cancel_all", "trade", req, dry_run=dry_run,
                  summary={"n_orders": len(pairs), "coin": want.coin if want else None})
    if skipped:
        out["skipped"] = skipped
    return out


@mcp.tool(annotations=WRITE)
def order_schedule_cancel_all(delay_seconds: int | None = None, clear: bool = False,
                              vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Dead man's switch: cancel ALL open orders at now + `delay_seconds` (>= 5) unless
    this is called again first. `clear=True` removes the schedule. Max 10
    triggers per day (resets 00:00 UTC)."""
    if clear == (delay_seconds is not None):
        raise ValueError("pass delay_seconds, or clear=True")
    t = None
    if delay_seconds is not None:
        if int(delay_seconds) < 5:
            raise ValueError("delay_seconds must be at least 5")
        t = now_ms() + int(delay_seconds) * 1000
    vault, _target = trade_target(vault_address)
    req = A.schedule_cancel(t, client().next_nonce(), vault_address=vault)
    return execute("order_schedule_cancel_all", "trade", req, dry_run=dry_run,
                   summary={"cancel_at_ms": t, "clear": clear})


# --------------------------------------------------------------------------
# Leverage and margin
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def leverage_update(coin: str, leverage: int, margin_mode: Literal["cross", "isolated"] = "cross",
                    vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Set a perp's leverage and margin mode (cross or isolated).
    Capped by HL_MAX_LEVERAGE and the asset's own maximum."""
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError("leverage applies to perps only")
    lev = int(leverage)
    if lev < 1:
        raise ValueError("leverage must be >= 1")
    reasons = guard.leverage_block_reasons(client().settings, lev, inst.coin)
    if inst.max_leverage and lev > inst.max_leverage:
        reasons.append(f"{inst.coin} allows at most {inst.max_leverage}x")
    if margin_mode == "cross" and inst.only_isolated:
        reasons.append(f"{inst.coin} is isolated-only")
    vault, _target = trade_target(vault_address)
    req = A.update_leverage(inst.asset, margin_mode == "cross", lev, client().next_nonce(),
                            vault_address=vault)
    return execute("leverage_update", "trade", req, dry_run=dry_run, reasons=reasons,
                   summary={"coin": inst.coin, "leverage": lev, "margin_mode": margin_mode})


@mcp.tool(annotations=WRITE)
def margin_adjust_isolated(coin: str, amount_usd: float, vault_address: str | None = None,
                           dry_run: bool = False) -> dict:
    """Add (positive) or remove (negative) USD margin on an isolated perp position.
    Removing margin raises leverage; it is refused if the result would exceed
    HL_MAX_LEVERAGE."""
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError("isolated margin applies to perps only")
    amount = to_decimal(amount_usd, "amount_usd")
    if amount == 0:
        raise ValueError("amount_usd must be non-zero")
    vault, target = trade_target(vault_address)
    reasons: list[str] = []
    if amount < 0:
        reasons = _margin_removal_reasons(inst, target, amount)
    req = A.update_isolated_margin(inst.asset, float_to_usd_int(float(amount_usd)),
                                   client().next_nonce(), vault_address=vault)
    return execute("margin_adjust_isolated", "trade", req, dry_run=dry_run, reasons=reasons,
                   summary={"coin": inst.coin, "amount_usd": amount_usd})


def _margin_removal_reasons(inst: Instrument, target: str | None, amount: Decimal) -> list[str]:
    """Leverage after removing |amount| of margin = position value / new margin."""
    cap = client().settings.max_leverage
    if not target:
        return ["cannot verify the resulting leverage: no account address configured"]
    try:
        for p in _position_rows(target, inst.dex):
            if p.get("coin") != inst.coin:
                continue
            value = to_decimal(p.get("positionValue"), "positionValue")
            margin = to_decimal(p.get("marginUsed"), "marginUsed") + amount
            if margin <= 0:
                return [f"removing ${-amount} would leave no margin on {inst.coin}"]
            lev = value / margin
            if lev > cap:
                return [f"removing ${-amount} would put {inst.coin} at {lev:.2f}x, above "
                        f"HL_MAX_LEVERAGE {cap}x"]
            return []
    except Exception as e:
        return [f"cannot verify the resulting leverage ({e}); refusing fail-closed"]
    return [f"no open {inst.coin} position to remove margin from"]


@mcp.tool(annotations=WRITE)
def margin_set_isolated_leverage(coin: str, leverage: float, vault_address: str | None = None,
                                 dry_run: bool = False) -> dict:
    """Top up or release isolated margin so the position sits at a target `leverage`
    (e.g. 3.5), instead of specifying a USD amount."""
    inst = resolve(coin)
    if not inst.is_perp:
        raise ValueError("isolated margin applies to perps only")
    lev = to_decimal(leverage, "leverage")
    if lev < 1:
        raise ValueError("leverage must be >= 1")
    reasons = []
    if lev > client().settings.max_leverage:
        reasons.append(f"leverage {leverage} exceeds HL_MAX_LEVERAGE "
                       f"{client().settings.max_leverage}x")
    vault, _target = trade_target(vault_address)
    req = A.top_up_isolated_only_margin(inst.asset, fmt_decimal(lev), client().next_nonce(),
                                        vault_address=vault)
    return execute("margin_set_isolated_leverage", "trade", req, dry_run=dry_run,
                   reasons=reasons, summary={"coin": inst.coin, "leverage": fmt_decimal(lev)})


# --------------------------------------------------------------------------
# TWAP
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def twap_place(coin: str, side: Literal["buy", "sell"], size: float, minutes: int,
               randomize: bool = False, reduce_only: bool = False,
               vault_address: str | None = None, dry_run: bool = False) -> dict:
    """Exchange-native TWAP: `size` executed in slices every ~30s over `minutes`
    (5-1440). The exchange lets each slice slip up to 3%, so an opening TWAP
    needs HL_MAX_SLIPPAGE >= 0.03."""
    inst = resolve(coin)
    m = int(minutes)
    if not 5 <= m <= 1440:
        raise ValueError("minutes must be between 5 and 1440")
    is_buy = _is_buy(side)
    notes: list[str] = []
    sz = _size(inst, size, notes)
    ro = _reduce_only(inst, reduce_only, notes)
    exit_ = guard.is_exit(inst, ro)
    ref = mid(inst)
    notional = None
    if ref is not None:
        notional = (notional_usd(inst, sz, ref * (1 + TWAP_SLICE_SLIPPAGE)) if is_buy
                    else notional_usd(inst, sz, ref))
    vault, target = trade_target(vault_address)
    s = client().settings
    reasons = guard.order_block_reasons(s, inst, reduce_only=ro, notional_usd=notional)
    if not exit_ and Decimal(str(s.max_slippage)) < TWAP_SLICE_SLIPPAGE:
        reasons.append(f"TWAP slices may slip up to {TWAP_SLICE_SLIPPAGE:.0%}, above "
                       f"HL_MAX_SLIPPAGE {s.max_slippage}")
    reasons += leverage_reasons(inst, target, ro)
    req = A.twap_order(inst.asset, is_buy, fmt_decimal(sz), ro, m, randomize,
                       client().next_nonce(), vault_address=vault)
    return execute("twap_place", "trade", req, dry_run=dry_run, reasons=reasons,
                   summary={"coin": inst.coin, "side": side, "size": fmt_decimal(sz),
                            "minutes": m, "notional_usd": None if notional is None
                            else round(float(notional), 2), "notes": notes},
                   opening_notional_usd=None if exit_ else notional)


@mcp.tool(annotations=WRITE)
def twap_cancel(coin: str, twap_id: int, vault_address: str | None = None,
                dry_run: bool = False) -> dict:
    """Cancel a running TWAP by its id (see account_get_twaps)."""
    inst = resolve(coin)
    vault, _target = trade_target(vault_address)
    req = A.twap_cancel(inst.asset, int(twap_id), client().next_nonce(), vault_address=vault)
    return execute("twap_cancel", "trade", req, dry_run=dry_run,
                   summary={"coin": inst.coin, "twap_id": twap_id})


@mcp.tool(annotations=WRITE)
def nonce_invalidate(nonce: int, vault_address: str | None = None,
                     dry_run: bool = False) -> dict:
    """Burn a nonce with a no-op so an in-flight action signed with that nonce can
    no longer land. Use the `nonce` reported by a write tool whose send ended in
    status "error" (outcome unknown). If the original already landed, the no-op
    is rejected and nothing changes."""
    n = int(nonce)
    now = now_ms()
    if not now - 2 * 86_400_000 < n < now + 86_400_000:
        raise ValueError("nonce must be within the last 2 days (exchange validity window)")
    vault, _target = trade_target(vault_address)
    return execute("nonce_invalidate", "trade", A.noop(n, vault_address=vault), dry_run=dry_run,
                   summary={"nonce": n})
