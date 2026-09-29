"""Deterministic write gates + audit log for hyperliquid-mcp.

Every signed action passes `write_block_reasons` (module switch, read-only,
key present, key type, mainnet: code switch + env opt-in). Orders additionally pass
`order_block_reasons` (coin allowlist, per-order notional cap, slippage cap),
`price_band_reasons` (no limit priced through the market by more than
HL_MAX_SLIPPAGE), `daily_budget_reasons` (HL_MAX_DAILY_NOTIONAL_USD) and a
live leverage check in the server. Anything that hands funds or control to
another address passes `destination_block_reasons` (HL_WITHDRAW_ALLOWLIST).
All of these return reason strings instead of raising, so a dry run can
report every gate that would block the live send at once.

Reduce-only PERP orders skip the allowlist and notional caps on purpose: an
exit must never be blocked by the limits that exist to stop entries. Only
perps qualify — the exchange enforces reduce-only against a position there,
while on spot and outcomes the flag is meaningless (the server clears it).
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from trading_agent.mcp_servers.hyperliquid.settings import Settings
from trading_agent.mcp_servers.hyperliquid.universe import Instrument

# README design principle 9: graduating to real money takes a code edit and
# review, not a config flip. Mainnet writes need this set to True in a
# reviewed commit, AND HL_NETWORK=mainnet, AND HL_ALLOW_MAINNET_WRITES=true.
# tests/hl_mcp/test_execute.py pins it to False, so flipping it also means
# editing that test on purpose.
MAINNET_WRITES_ENABLED_IN_CODE = False


def mainnet_write_blockers(settings: Settings) -> list[str]:
    """Why mainnet writes are off ([] on testnet or when fully enabled)."""
    if not settings.is_mainnet:
        return []
    out = []
    if not MAINNET_WRITES_ENABLED_IN_CODE:
        out.append("mainnet writes are disabled in code "
                   "(guard.MAINNET_WRITES_ENABLED_IN_CODE = False); enabling them is a "
                   "reviewed code change, not a setting")
    if not settings.allow_mainnet_writes:
        out.append("mainnet writes are disabled; HL_ALLOW_MAINNET_WRITES=true is also required")
    return out


MODULE_HINTS = {
    "trade": "orders, cancels, TWAP, leverage and margin",
    "transfer": "moving funds between your own balances, sub-accounts, vaults, staking, "
                "lending and outcome shares",
    "withdraw": "sending funds to another address or withdrawing to the bridge",
    "admin": "API-wallet approval, builder fees, sub-account/vault creation, referral and "
             "account-mode settings",
    "advanced": "raw signed L1 actions (deployer / validator operations)",
}


def write_block_reasons(settings: Settings, module: str, *, has_signer: bool,
                        signer_is_agent: bool, needs_owner_key: bool) -> list[str]:
    reasons: list[str] = []
    if module not in settings.write_modules:
        reasons.append(
            f"write module '{module}' ({MODULE_HINTS.get(module, module)}) is disabled; "
            f"add it to HL_WRITE_MODULES to enable"
        )
    if settings.read_only:
        reasons.append("server is read-only (HL_READ_ONLY=true)")
    if not has_signer:
        reasons.append("no signing key configured (HL_PRIVATE_KEY)")
    reasons += mainnet_write_blockers(settings)
    if needs_owner_key and signer_is_agent:
        reasons.append(
            "this action must be signed by the account owner's key; the configured key is an "
            "API (agent) wallet, which can only sign trading actions"
        )
    return reasons


def coin_allowed(settings: Settings, inst: Instrument) -> bool:
    allowed = settings.allowed_coins
    if allowed is None:
        return True
    return inst.coin.lower() in allowed or inst.name.lower() in allowed


def is_exit(inst: Instrument, reduce_only: bool) -> bool:
    """A reduce-only perp order: the exchange guarantees it only shrinks a position."""
    return reduce_only and inst.is_perp


def order_block_reasons(settings: Settings, inst: Instrument, *, reduce_only: bool,
                        notional_usd: Decimal | None) -> list[str]:
    reasons: list[str] = []
    exit_ = is_exit(inst, reduce_only)
    if inst.delisted and not exit_:
        reasons.append(f"{inst.coin} is delisted; only reduce-only orders are allowed")
    if exit_:
        return reasons
    if not coin_allowed(settings, inst):
        reasons.append(f"{inst.coin} is not in HL_ALLOWED_COINS")
    if notional_usd is None:
        reasons.append(
            f"cannot value this order in USD ({inst.coin} is quoted in {inst.quote}); "
            "refusing fail-closed"
        )
    elif notional_usd > Decimal(str(settings.max_order_notional_usd)):
        reasons.append(
            f"order notional ${notional_usd:,.2f} exceeds HL_MAX_ORDER_NOTIONAL_USD "
            f"${settings.max_order_notional_usd:,.2f}"
        )
    return reasons


def price_band_reasons(settings: Settings, inst: Instrument, *, is_buy: bool, limit_px: Decimal,
                       reference: Decimal | None, what: str = "mid",
                       band: float | None = None) -> list[str]:
    """Refuse a limit priced through the market by more than the slippage cap
    (HL_MAX_SLIPPAGE unless `band` is given — exits get a wider one).

    A buy limit far above mid (or a sell far below) fills against whatever is
    on the book — on a thin book, possibly someone's resting order placed to
    catch exactly that. Market orders are already capped by slippage; this
    gives marketable limits the same bound."""
    if reference is None or reference <= 0:
        return [f"no {what} price for {inst.coin} to bound the limit price against; "
                "refusing fail-closed"]
    b = Decimal(str(settings.max_slippage if band is None else band))
    if is_buy and limit_px > reference * (1 + b):
        return [f"buy limit {limit_px} is more than {b:.2%} above the {what} {reference} "
                "(HL_MAX_SLIPPAGE); it would fill through the book"]
    if not is_buy and limit_px < reference * (1 - b):
        return [f"sell limit {limit_px} is more than {b:.2%} below the {what} {reference} "
                "(HL_MAX_SLIPPAGE); it would fill through the book"]
    return []


BUDGET_FIELD = "opening_notional_usd"


def daily_opening_notional(settings: Settings) -> Decimal:
    """USD notional of opening orders sent today (UTC) on this network,
    including sends whose outcome is unknown ("error": they may have landed).

    Read from the audit log, so every server process shares one budget and a
    restart does not reset it. Records are appended in time order, so the
    scan stops at the first record from an earlier day."""
    path = settings.audit_path
    if not path.exists():
        return Decimal(0)
    today = datetime.now(UTC).date().isoformat()
    total = Decimal(0)
    with path.open() as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        day = str(rec.get("ts", ""))[:10]
        if day < today:
            break
        if (day == today and rec.get("decision") in ("sent", "error")
                and rec.get("network") == settings.network and rec.get(BUDGET_FIELD)):
            total += Decimal(str(rec[BUDGET_FIELD]))
    return total


def daily_budget_reasons(settings: Settings, opening_usd: Decimal) -> list[str]:
    if opening_usd <= 0:
        return []
    try:
        used = daily_opening_notional(settings)
    except OSError as e:
        return [f"cannot read the audit log to check HL_MAX_DAILY_NOTIONAL_USD ({e}); "
                "refusing fail-closed"]
    cap = Decimal(str(settings.max_daily_notional_usd))
    if used + opening_usd > cap:
        return [f"today's opening notional would reach ${used + opening_usd:,.2f} "
                f"(${used:,.2f} already sent since 00:00 UTC), above "
                f"HL_MAX_DAILY_NOTIONAL_USD ${cap:,.2f}"]
    return []


def leverage_block_reasons(settings: Settings, leverage: int | None, coin: str) -> list[str]:
    if leverage is not None and leverage > settings.max_leverage:
        return [
            f"{coin} leverage is {leverage}x, above HL_MAX_LEVERAGE {settings.max_leverage}x; "
            "lower it with leverage_update first"
        ]
    return []


def destination_block_reasons(settings: Settings, destination: str,
                              own_addresses: set[str], what: str = "destination") -> list[str]:
    d = destination.lower()
    if d in own_addresses or d in settings.withdraw_allowlist:
        return []
    return [f"{what} {destination} is not your account and not in HL_WITHDRAW_ALLOWLIST"]


def audit(settings: Settings, record: dict[str, Any]) -> None:
    """Append one JSONL record. Best effort: a full disk must not block an
    exit order, so failures go to stderr instead of raising."""
    rec = {"ts": datetime.now(UTC).isoformat(), "network": settings.network, **record}
    try:
        settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.audit_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:  # pragma: no cover - disk failure path
        print(f"[hyperliquid-mcp] audit write failed: {e!r}", file=sys.stderr)
