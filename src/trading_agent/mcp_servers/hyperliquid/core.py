"""Shared plumbing for the hyperliquid-mcp tool modules.

Holds the FastMCP instance, the lazily-built client, input normalisation
(addresses, time ranges, instruments) and `execute`, the single choke point
every signed action goes through:

    guards → (dry run | blocked | sign + send) → normalised result → audit

Tool modules only build ActionRequests and extra guard reasons; they never
sign or post themselves.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from trading_agent.mcp_servers.hyperliquid import guard
from trading_agent.mcp_servers.hyperliquid.actions import ActionRequest
from trading_agent.mcp_servers.hyperliquid.client import (
    HLClient,
    normalize_exchange_response,
)
from trading_agent.mcp_servers.hyperliquid.settings import (
    SettingsError,
    load_settings,
    normalize_address,
)
from trading_agent.mcp_servers.hyperliquid.universe import (
    USD_STABLES,
    Instrument,
    UnknownInstrument,
    to_decimal,
)

mcp = FastMCP("hyperliquid-mcp")

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                       openWorldHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False,
                        openWorldHint=True)

_client: HLClient | None = None


def client() -> HLClient:
    global _client
    if _client is None:
        _client = HLClient(load_settings())
    return _client


def set_client(c: HLClient | None) -> None:
    """Swap the client (tests, or a settings reload)."""
    global _client
    _client = c


# --------------------------------------------------------------------------
# Input normalisation
# --------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def iso(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(float(ms) / 1000, tz=UTC).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def to_ms(value: int | float | str | None, what: str) -> int | None:
    """Epoch milliseconds from int/float ms, digit strings, or ISO-8601."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a timestamp, not a boolean")
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(
            f"{what} {value!r} is neither epoch milliseconds nor ISO-8601 "
            "(e.g. 2026-09-01T00:00:00Z)"
        ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def time_range(start_time: Any, end_time: Any, lookback_hours: float | None,
               default_hours: float) -> tuple[int, int | None]:
    start = to_ms(start_time, "start_time")
    end = to_ms(end_time, "end_time")
    if start is None:
        hours = default_hours if lookback_hours is None else float(lookback_hours)
        if hours <= 0:
            raise ValueError("lookback_hours must be positive")
        start = (end or now_ms()) - int(hours * 3_600_000)
    if end is not None and end < start:
        raise ValueError("end_time is before start_time")
    return start, end


def address(value: str, what: str = "address") -> str:
    try:
        return normalize_address(value, what)
    except SettingsError as e:
        raise ValueError(str(e)) from e


def user_or_default(user: str | None) -> str:
    """The queried account: explicit `user`, else the configured account."""
    if user:
        return address(user, "user")
    acct = client().account_address()
    if not acct:
        raise ValueError(
            "no account configured: pass `user` (a 0x address) or set "
            "HL_ACCOUNT_ADDRESS / HL_PRIVATE_KEY"
        )
    return acct


def trade_target(vault_address: str | None) -> tuple[str | None, str | None]:
    """(vault address to sign with, address whose orders/positions are affected).

    The target is None only when nothing is configured; guards that need it
    then refuse, while a dry run can still show the action."""
    c = client()
    vault = address(vault_address, "vault_address") if vault_address else c.settings.vault_address
    return vault, vault or c.account_address()


def require_target(target: str | None) -> str:
    if not target:
        raise ValueError("no account configured: set HL_ACCOUNT_ADDRESS or HL_PRIVATE_KEY")
    return target


def resolve(coin: str) -> Instrument:
    try:
        return client().universe.resolve(coin)
    except UnknownInstrument as e:
        raise ValueError(str(e)) from e


def num(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def pct(a: Any, b: Any) -> float | None:
    fa, fb = num(a), num(b)
    if fa is None or not fb:
        return None
    return round((fa / fb - 1) * 100, 4)


def cap(rows: list, limit: int) -> tuple[list, bool]:
    limit = max(int(limit), 0)
    return rows[:limit], len(rows) > limit


# --------------------------------------------------------------------------
# Prices and valuation
# --------------------------------------------------------------------------

def mids(dex: str = "") -> dict[str, str]:
    return client().info({"type": "allMids", "dex": dex}) or {}


def mid(inst: Instrument) -> Decimal | None:
    m = mids(inst.dex if inst.is_perp else "").get(inst.coin)
    return to_decimal(m, "mid") if m is not None else None


def quote_usd_rate(quote: str) -> Decimal | None:
    """USD value of one unit of a quote/collateral token; None if unknown."""
    if quote.upper() in USD_STABLES:
        return Decimal(1)
    try:
        pair = client().universe.resolve(f"{quote}/USDC")
    except UnknownInstrument:
        return None
    return mid(pair)


def notional_usd(inst: Instrument, size: Decimal, price: Decimal) -> Decimal | None:
    rate = quote_usd_rate(inst.quote)
    return None if rate is None else size * price * rate


def own_addresses() -> set[str]:
    """Addresses that are unambiguously the user's: the account and the signer."""
    c = client()
    return {a for a in (c.account_address(), c.signer_address) if a}


def destination_reasons(destination: str, what: str = "destination") -> list[str]:
    """Anything that hands funds or control to `destination` needs it to be the
    user's own address or listed in HL_WITHDRAW_ALLOWLIST."""
    return guard.destination_block_reasons(client().settings, destination, own_addresses(), what)


def leverage_reasons(inst: Instrument, target: str | None, reduce_only: bool) -> list[str]:
    """Live check that the coin's current leverage is within HL_MAX_LEVERAGE.

    Only for perp orders that can add exposure. Fails closed: if the leverage
    cannot be read, the order is refused rather than sent unchecked."""
    if not inst.is_perp or reduce_only:
        return []
    if not target:
        return ["cannot verify leverage: no account address configured"]
    try:
        data = client().info({"type": "activeAssetData", "user": target, "coin": inst.coin})
        lev = int((data.get("leverage") or {}).get("value"))
    except Exception as e:
        return [f"cannot verify current {inst.coin} leverage ({e}); refusing fail-closed"]
    return guard.leverage_block_reasons(client().settings, lev, inst.coin)


# --------------------------------------------------------------------------
# The choke point
# --------------------------------------------------------------------------

def _display_action(req: ActionRequest) -> dict:
    """The action as it will be posted (user-signed ones gain two fields)."""
    action = dict(req.action)
    if req.scheme == "user":
        action["signatureChainId"] = "0x66eee"
        action["hyperliquidChain"] = "Mainnet" if client().settings.is_mainnet else "Testnet"
        for k in req.drop_after_sign:
            action.pop(k, None)
    return action


def execute(tool: str, module: str, req: ActionRequest, *, dry_run: bool = False,
            reasons: list[str] | None = None, needs_owner_key: bool = False,
            summary: dict | None = None,
            opening_notional_usd: Decimal | None = None) -> dict:
    """Guard, then dry-run / refuse / sign-and-send one action, and audit it.

    `opening_notional_usd` is the exposure this action can add; it is checked
    against HL_MAX_DAILY_NOTIONAL_USD and, once sent, counted towards it.

    Returns a dict whose `status` is one of:
      dry_run  – nothing signed or sent; `blocked_reasons` lists what would
                 stop a live send.
      blocked  – a guard refused; nothing signed or sent.
      error    – the request failed in transit; the outcome is UNKNOWN.
      sent     – the exchange answered; `ok` is False if any part was rejected.
    """
    c = client()
    s = c.settings
    has_signer = c.wallet is not None
    all_reasons = guard.write_block_reasons(
        s, module, has_signer=has_signer,
        signer_is_agent=c.signer_is_agent() if (needs_owner_key and has_signer) else False,
        needs_owner_key=needs_owner_key,
    ) + list(reasons or [])
    if opening_notional_usd:
        all_reasons += guard.daily_budget_reasons(s, opening_notional_usd)
    head: dict[str, Any] = {"tool": tool, "network": s.network, "action_type": req.action_type,
                            "nonce": req.nonce}
    if summary:
        head["summary"] = summary
    record = {"tool": tool, "module": module, "signer": c.signer_address, "nonce": req.nonce,
              "vault": req.vault_address, "action": req.action, "summary": summary}
    if opening_notional_usd:
        # Counted against the daily budget on "sent" and "error" records:
        # conservatively, even when the exchange rejects part of the batch or
        # the outcome of the send is unknown.
        record[guard.BUDGET_FIELD] = float(round(opening_notional_usd, 2))

    if dry_run or s.dry_run:
        guard.audit(s, {**record, "decision": "dry_run", "reasons": all_reasons})
        return {
            **head, "status": "dry_run", "sent": False,
            "live_send_would_be_blocked": bool(all_reasons),
            "blocked_reasons": all_reasons,
            "signing_scheme": req.scheme, "vault_address": req.vault_address,
            "action": _display_action(req),
            "note": "Nothing was signed or sent."
                    + (" HL_DRY_RUN=true forces dry runs." if s.dry_run else ""),
        }
    if all_reasons:
        guard.audit(s, {**record, "decision": "blocked", "reasons": all_reasons})
        return {**head, "status": "blocked", "sent": False, "reasons": all_reasons}

    t0 = time.monotonic()
    try:
        resp = c.send(req)
    except Exception as e:
        guard.audit(s, {**record, "decision": "error", "error": repr(e)})
        return {
            **head, "status": "error", "sent": "unknown", "error": str(e),
            "hint": "The request may or may not have reached the exchange. Check "
                    "account_get_open_orders / account_get_order_status (by cloid when "
                    "one was set) before retrying, to avoid a duplicate; "
                    "nonce_invalidate(nonce) stops it from landing later.",
        }
    norm = normalize_exchange_response(resp)
    guard.audit(s, {**record, "decision": "sent", "ok": norm["ok"], "errors": norm["errors"],
                    "response": resp, "latency_ms": round((time.monotonic() - t0) * 1000)})
    out = {**head, "status": "sent", "sent": True, "ok": norm["ok"], "errors": norm["errors"],
           "response": resp}
    if "statuses" in norm:
        out["statuses"] = norm["statuses"]
    return out
