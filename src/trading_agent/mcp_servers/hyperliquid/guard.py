"""Deterministic write gates + audit log for hyperliquid-mcp.

Every signed action passes `write_block_reasons` (module switch, read-only,
key present, key type, mainnet opt-in). Orders additionally pass
`order_block_reasons` (coin allowlist, per-order notional cap, slippage cap)
and a live leverage check in the server. Funds leaving the account pass
`destination_block_reasons` (HL_WITHDRAW_ALLOWLIST). All of these return
reason strings instead of raising, so a dry run can report every gate that
would block the live send at once.

Reduce-only orders skip the allowlist and notional cap on purpose: an exit
must never be blocked by the limits that exist to stop entries.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from trading_agent.mcp_servers.hyperliquid.settings import Settings
from trading_agent.mcp_servers.hyperliquid.universe import Instrument

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
    if settings.is_mainnet and not settings.allow_mainnet_writes:
        reasons.append(
            "mainnet writes are disabled; set HL_ALLOW_MAINNET_WRITES=true to trade real funds"
        )
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


def order_block_reasons(settings: Settings, inst: Instrument, *, reduce_only: bool,
                        notional_usd: Decimal | None, slippage: float | None = None) -> list[str]:
    reasons: list[str] = []
    if inst.delisted and not reduce_only:
        reasons.append(f"{inst.coin} is delisted; only reduce-only orders are allowed")
    if slippage is not None and slippage > settings.max_slippage:
        reasons.append(f"slippage {slippage:.4f} exceeds HL_MAX_SLIPPAGE {settings.max_slippage}")
    if reduce_only:
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


def leverage_block_reasons(settings: Settings, leverage: int | None, coin: str) -> list[str]:
    if leverage is not None and leverage > settings.max_leverage:
        return [
            f"{coin} leverage is {leverage}x, above HL_MAX_LEVERAGE {settings.max_leverage}x; "
            "lower it with leverage_update first"
        ]
    return []


def destination_block_reasons(settings: Settings, destination: str,
                              own_addresses: set[str]) -> list[str]:
    d = destination.lower()
    if d in own_addresses or d in settings.withdraw_allowlist:
        return []
    return [
        f"destination {destination} is not your account and not in HL_WITHDRAW_ALLOWLIST"
    ]


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
