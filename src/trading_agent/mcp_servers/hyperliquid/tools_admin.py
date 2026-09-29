"""Account administration (write module `admin`) and raw actions (`advanced`).

Both modules are off by default. `admin` covers API-wallet approval, builder
fees, sub-accounts, referral codes, display name, margining mode, vault
management and similar settings. `advanced` is the escape hatch for L1
actions without a dedicated tool (HIP-1/2/3/4 deployer and validator
operations) plus multi-sig conversion.

API-wallet private keys are written to a 0600 file under HL_AGENT_KEY_DIR and
never returned through MCP: a key in a tool result ends up in model context
and transcripts.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime
from typing import Literal

import eth_account

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid.core import (
    WRITE,
    address,
    client,
    destination_reasons,
    execute,
    mcp,
    now_ms,
    trade_target,
    user_or_default,
)
from trading_agent.mcp_servers.hyperliquid.tools_funds import _micro_usd

MAX_AGENT_VALID_DAYS = 180
# 0.0005 USDC per unit: one call can buy at most $50 of request budget.
MAX_RESERVE_WEIGHT = 100_000


def _write_agent_key(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=2)


@mcp.tool(annotations=WRITE)
def agent_approve(name: str | None = None, valid_days: int | None = None,
                  dry_run: bool = False) -> dict:
    """Create and approve a new API (agent) wallet that can trade for this account
    but cannot withdraw or transfer funds.

    The new private key is saved to a 0600 file under HL_AGENT_KEY_DIR and is
    NOT returned. To use it, set HL_PRIVATE_KEY to the key in that file and
    HL_ACCOUNT_ADDRESS to this account. An account has 1 unnamed + up to 3
    named agents; approving an unnamed agent (or reusing a name) replaces the
    previous one. `valid_days` (<= 180, needs a name) sets an expiry.
    Must be signed by the account owner's key.
    """
    if name is not None and not 1 <= len(name) <= 16:
        raise ValueError("name must be 1-16 characters")
    agent_name = name
    valid_until = None
    if valid_days is not None:
        if name is None:
            raise ValueError("valid_days needs a name")
        if not 1 <= int(valid_days) <= MAX_AGENT_VALID_DAYS:
            raise ValueError(f"valid_days must be 1-{MAX_AGENT_VALID_DAYS}")
        valid_until = now_ms() + int(valid_days) * 86_400_000
        agent_name = f"{name} valid_until {valid_until}"
    c = client()
    key = "0x" + secrets.token_hex(32)
    agent = eth_account.Account.from_key(key)
    agent_addr = agent.address.lower()
    req = A.approve_agent(agent_addr, agent_name, c.next_nonce())
    summary = {"agent_address": agent_addr, "name": name,
               "valid_until_ms": valid_until, "account": c.account_address()}
    if dry_run or c.settings.dry_run:
        out = execute("agent_approve", "admin", req, dry_run=True, needs_owner_key=True,
                      summary=summary)
        out["note"] += " The agent key shown by address was discarded."
        return out
    path = c.settings.agent_key_dir / f"{c.settings.network}-{agent_addr}.json"
    _write_agent_key(path, {
        "network": c.settings.network, "agent_address": agent_addr, "private_key": key,
        "account_address": c.account_address(), "name": name, "valid_until_ms": valid_until,
        "created_utc": datetime.now(UTC).isoformat(), "status": "pending",
    })
    out = execute("agent_approve", "admin", req, needs_owner_key=True, summary=summary)
    if out.get("status") == "sent" and out.get("ok"):
        rec = json.loads(path.read_text())
        rec["status"] = "approved"
        path.write_text(json.dumps(rec, indent=2))
        out["agent_key_file"] = str(path)
        out["next_steps"] = (f"Set HL_PRIVATE_KEY from {path} and "
                             f"HL_ACCOUNT_ADDRESS={c.account_address()} to trade via this agent.")
    elif out.get("status") == "error":
        out["agent_key_file"] = str(path)
        out["note"] = "Outcome unknown; the key file is kept in case the approval landed."
    else:
        path.unlink(missing_ok=True)
    return out


@mcp.tool(annotations=WRITE)
def builder_fee_approve(builder: str, max_fee_rate_percent: float, dry_run: bool = False) -> dict:
    """Allow a builder (front-end / bot operator) to charge up to `max_fee_rate_percent`
    per order (e.g. 0.01 = 0.01%; max 0.1% on perps, 1% on spot). 0 revokes.
    The builder must be in HL_WITHDRAW_ALLOWLIST (revoking is always allowed).
    Must be signed by the account owner's key."""
    b = address(builder, "builder")
    rate = float(max_fee_rate_percent)
    if not 0 <= rate <= 1:
        raise ValueError("max_fee_rate_percent must be between 0 and 1 (percent)")
    rate_str = f"{rate:.4f}".rstrip("0").rstrip(".") + "%"
    req = A.approve_builder_fee(b, rate_str, client().next_nonce())
    return execute("builder_fee_approve", "admin", req, dry_run=dry_run, needs_owner_key=True,
                   reasons=destination_reasons(b, "builder") if rate > 0 else [],
                   summary={"builder": b, "max_fee_rate": rate_str})


@mcp.tool(annotations=WRITE)
def sub_account_create(name: str, dry_run: bool = False) -> dict:
    """Create a sub-account (its own margin and positions, traded by the master key
    via vault_address). Requires enough trading volume on the master account."""
    if not 1 <= len(name) <= 16:
        raise ValueError("name must be 1-16 characters")
    return execute("sub_account_create", "admin", A.create_sub_account(name, client().next_nonce()),
                   dry_run=dry_run, summary={"name": name})


@mcp.tool(annotations=WRITE)
def sub_account_rename(sub_account: str, name: str, dry_run: bool = False) -> dict:
    """Rename a sub-account."""
    if not 1 <= len(name) <= 16:
        raise ValueError("name must be 1-16 characters")
    sub = address(sub_account, "sub_account")
    return execute("sub_account_rename", "admin",
                   A.sub_account_modify(sub, name, client().next_nonce()), dry_run=dry_run,
                   summary={"sub_account": sub, "name": name})


@mcp.tool(annotations=WRITE)
def referral_use_code(code: str, dry_run: bool = False) -> dict:
    """Register under someone's referral code (fee discount). Only possible once."""
    return execute("referral_use_code", "admin", A.set_referrer(code, client().next_nonce()),
                   dry_run=dry_run, summary={"code": code})


@mcp.tool(annotations=WRITE)
def referral_create_code(code: str, dry_run: bool = False) -> dict:
    """Create your own referral code (1-20 characters)."""
    if not 1 <= len(code) <= 20:
        raise ValueError("code must be 1-20 characters")
    return execute("referral_create_code", "admin",
                   A.register_referrer(code, client().next_nonce()), dry_run=dry_run,
                   summary={"code": code})


@mcp.tool(annotations=WRITE)
def account_set_display_name(name: str, dry_run: bool = False) -> dict:
    """Set the public display name (leaderboard, vaults). Empty string removes it."""
    if len(name) > 20:
        raise ValueError("name must be at most 20 characters")
    return execute("account_set_display_name", "admin",
                   A.set_display_name(name, client().next_nonce()), dry_run=dry_run,
                   summary={"display_name": name})


@mcp.tool(annotations=WRITE)
def account_set_abstraction(mode: Literal["disabled", "unifiedAccount", "portfolioMargin"],
                            user: str | None = None, vault_address: str | None = None,
                            dry_run: bool = False) -> dict:
    """Switch margining mode: "disabled" (separate spot/perp balances),
    "unifiedAccount" (one balance for spot, perps and HIP-3), or
    "portfolioMargin" (unified + borrowing against collateral; eligibility applies).
    `user` may be a sub-account. With an API-wallet key this is sent as
    agentSetAbstraction."""
    c = client()
    if c.wallet is not None and c.signer_is_agent():
        vault, _t = trade_target(vault_address)
        req = A.agent_set_abstraction(mode, c.next_nonce(), vault_address=vault)
        return execute("account_set_abstraction", "admin", req, dry_run=dry_run,
                       summary={"mode": mode, "via": "agentSetAbstraction"})
    u = user_or_default(user)
    req = A.user_set_abstraction(u, mode, c.next_nonce())
    return execute("account_set_abstraction", "admin", req, dry_run=dry_run, needs_owner_key=True,
                   summary={"user": u, "mode": mode})


@mcp.tool(annotations=WRITE)
def account_set_portfolio_margin(enabled: bool, user: str | None = None,
                                 dry_run: bool = False) -> dict:
    """Enable or disable portfolio margin for the account (or a sub-account `user`)."""
    u = user_or_default(user)
    req = A.user_portfolio_margin(u, enabled, client().next_nonce())
    return execute("account_set_portfolio_margin", "admin", req, dry_run=dry_run,
                   needs_owner_key=True, summary={"user": u, "enabled": enabled})


@mcp.tool(annotations=WRITE)
def account_set_spot_dusting(opt_out: bool, dry_run: bool = False) -> dict:
    """Opt out of (or back into) automatic conversion of tiny spot balances ("dust")."""
    return execute("account_set_spot_dusting", "admin",
                   A.spot_dusting(opt_out, client().next_nonce()), dry_run=dry_run,
                   summary={"opt_out": opt_out})


@mcp.tool(annotations=WRITE)
def account_set_evm_big_blocks(enabled: bool, dry_run: bool = False) -> dict:
    """Route this address's HyperEVM transactions to large (slow, high-gas) blocks,
    e.g. for contract deployment, or back to small fast blocks."""
    return execute("account_set_evm_big_blocks", "admin",
                   A.evm_user_modify(enabled, client().next_nonce()), dry_run=dry_run,
                   summary={"using_big_blocks": enabled})


@mcp.tool(annotations=WRITE)
def account_reserve_request_weight(weight: int, destination: str | None = None,
                                   dry_run: bool = False) -> dict:
    """Buy extra address-based action budget (0.0005 USDC per request, paid from the
    perp balance; at most 100,000 = 50 USDC per call), optionally for another
    user (who must be your own or allowlisted address)."""
    if not 1 <= int(weight) <= MAX_RESERVE_WEIGHT:
        raise ValueError(f"weight must be between 1 and {MAX_RESERVE_WEIGHT}")
    dest = address(destination, "destination") if destination else None
    req = A.reserve_request_weight(int(weight), client().next_nonce(), destination=dest)
    return execute("account_reserve_request_weight", "admin", req, dry_run=dry_run,
                   reasons=destination_reasons(dest) if dest else [],
                   summary={"weight": int(weight), "cost_usdc": int(weight) * 0.0005,
                            "destination": dest})


@mcp.tool(annotations=WRITE)
def staking_link_user(user: str, is_finalize: bool, dry_run: bool = False) -> dict:
    """Link a trading account to a staking account so the trading account gets the
    staking fee discount. The trading user initiates (is_finalize=False, `user` =
    staking account); the staking user finalizes (is_finalize=True, `user` =
    trading account). The link is permanent."""
    u = address(user, "user")
    req = A.link_staking_user(u, is_finalize, client().next_nonce())
    return execute("staking_link_user", "admin", req, dry_run=dry_run, needs_owner_key=True,
                   summary={"user": u, "is_finalize": is_finalize})


@mcp.tool(annotations=WRITE)
def staking_unlink_trading_user(trading_user: str, dry_run: bool = False) -> dict:
    """As a staking account, disable the fee-discount link of a trading user."""
    u = address(trading_user, "trading_user")
    req = A.staking_link_disable_trading_user(u, client().next_nonce())
    return execute("staking_unlink_trading_user", "admin", req, dry_run=dry_run,
                   needs_owner_key=True, summary={"trading_user": u})


@mcp.tool(annotations=WRITE)
def vault_create(name: str, description: str, initial_usd: float, dry_run: bool = False) -> dict:
    """Create a vault you lead, seeded with `initial_usd` (minimum 100 USDC)."""
    if not 3 <= len(name) <= 50:
        raise ValueError("name must be 3-50 characters")
    if not 10 <= len(description) <= 250:
        raise ValueError("description must be 10-250 characters")
    if float(initial_usd) < 100:
        raise ValueError("initial_usd must be at least 100")
    c = client()
    req = A.create_vault(name, description, _micro_usd(initial_usd, "initial_usd"),
                         c.next_nonce())
    return execute("vault_create", "admin", req, dry_run=dry_run,
                   summary={"name": name, "initial_usd": initial_usd})


@mcp.tool(annotations=WRITE)
def vault_modify(vault_address: str, allow_deposits: bool | None = None,
                 always_close_on_withdraw: bool | None = None, dry_run: bool = False) -> dict:
    """Change a vault you lead: accept new deposits or not, and whether follower
    withdrawals always close positions proportionally."""
    if allow_deposits is None and always_close_on_withdraw is None:
        raise ValueError("pass allow_deposits and/or always_close_on_withdraw")
    v = address(vault_address, "vault_address")
    req = A.vault_modify(v, allow_deposits, always_close_on_withdraw, client().next_nonce())
    return execute("vault_modify", "admin", req, dry_run=dry_run,
                   summary={"vault": v, "allow_deposits": allow_deposits,
                            "always_close_on_withdraw": always_close_on_withdraw})


@mcp.tool(annotations=WRITE)
def vault_distribute(vault_address: str, amount_usd: float, dry_run: bool = False) -> dict:
    """Distribute USDC from a vault you lead to its depositors pro rata.
    amount_usd=0 CLOSES the vault."""
    v = address(vault_address, "vault_address")
    usd = 0 if float(amount_usd) == 0 else _micro_usd(amount_usd)
    req = A.vault_distribute(v, usd, client().next_nonce())
    return execute("vault_distribute", "admin", req, dry_run=dry_run,
                   summary={"vault": v, "amount_usd": amount_usd, "closes_vault": usd == 0})


# --------------------------------------------------------------------------
# advanced
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def advanced_send_l1_action(action: dict, vault_address: str | None = None,
                            dry_run: bool = True) -> dict:
    """Sign and send a raw L1 action that has no dedicated tool: HIP-1/2 spotDeploy,
    HIP-3 perpDeploy, HIP-4 outcome deployment, CValidatorAction / CSignerAction,
    gossipPriorityBid, validatorL1Stream, authorizeAqav2Role,
    hip3LiquidatorTransfer, finalizeEvmContract...

    KEY ORDER MATTERS: the action is msgpack-hashed exactly as given, so it
    must match the canonical field order from the API docs. Types that have a
    dedicated tool (orders, cancels, transfers, withdrawals, approvals...) are
    refused so their guards cannot be bypassed. Defaults to dry_run=True.
    """
    vault = address(vault_address, "vault_address") if vault_address else None
    req = A.raw_l1(action, client().next_nonce(), vault_address=vault)
    return execute("advanced_send_l1_action", "advanced", req, dry_run=dry_run,
                   summary={"type": req.action_type})


@mcp.tool(annotations=WRITE)
def advanced_convert_to_multisig(authorized_users: list[str], threshold: int,
                                 dry_run: bool = True) -> dict:
    """Convert this account into a multi-sig account controlled by `authorized_users`
    with `threshold` required signatures. Afterwards this key alone can no
    longer act for the account. Every authorized user gains control of the
    funds, so each must be your own address or in HL_WITHDRAW_ALLOWLIST.
    Owner key only. Defaults to dry_run=True."""
    users = [address(u, "authorized user") for u in authorized_users]
    if not users or not 1 <= int(threshold) <= len(users):
        raise ValueError("threshold must be between 1 and len(authorized_users)")
    reasons = [r for u in users for r in destination_reasons(u, "authorized user")]
    req = A.convert_to_multi_sig_user(users, int(threshold), client().next_nonce())
    return execute("advanced_convert_to_multisig", "advanced", req, dry_run=dry_run,
                   needs_owner_key=True, reasons=reasons,
                   summary={"authorized_users": users, "threshold": int(threshold)})
