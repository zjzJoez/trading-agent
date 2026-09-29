"""Pure builders for every signed Hyperliquid action. No I/O, no signing.

Two signing schemes exist (see the Signing docs):
  * L1 actions: msgpack(action) is hashed, so KEY ORDER IS PART OF THE
    SIGNATURE. Every dict below is written in the exact key order the
    official Python SDK uses (exchange.py), or, for actions the Python SDK
    does not implement, the order of the nktkas TypeScript SDK schemas.
    tests/test_hyperliquid_mcp.py pins the SDK-equivalent ones against
    hyperliquid.exchange.Exchange itself.
  * User-signed actions: EIP-712 typed data; the field list in `types` is
    what gets signed. Only the account owner's key can sign these — an API
    (agent) wallet cannot.

Builders return an ActionRequest describing what to sign and send, so the
server can dry-run, audit and guard an action before any key touches it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from hyperliquid.utils.signing import (
    CONVERT_TO_MULTI_SIG_USER_SIGN_TYPES,
    SEND_ASSET_SIGN_TYPES,
    SPOT_TRANSFER_SIGN_TYPES,
    TOKEN_DELEGATE_TYPES,
    USD_CLASS_TRANSFER_SIGN_TYPES,
    USD_SEND_SIGN_TYPES,
    USER_DEX_ABSTRACTION_SIGN_TYPES,
    USER_SET_ABSTRACTION_SIGN_TYPES,
    WITHDRAW_SIGN_TYPES,
    order_request_to_order_wire,
    order_wires_to_order_action,
)
from hyperliquid.utils.types import Cloid

# EIP-712 field lists the Python SDK inlines (sign_agent / sign_approve_builder_fee)
# or does not have at all (taken from the TS SDK).
APPROVE_AGENT_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "agentAddress", "type": "address"},
    {"name": "agentName", "type": "string"},
    {"name": "nonce", "type": "uint64"},
]
APPROVE_BUILDER_FEE_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "maxFeeRate", "type": "string"},
    {"name": "builder", "type": "address"},
    {"name": "nonce", "type": "uint64"},
]
C_TRANSFER_SIGN_TYPES = [  # cDeposit / cWithdraw
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "wei", "type": "uint64"},
    {"name": "nonce", "type": "uint64"},
]
USER_PORTFOLIO_MARGIN_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "user", "type": "address"},
    {"name": "enabled", "type": "bool"},
    {"name": "nonce", "type": "uint64"},
]
LINK_STAKING_USER_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "user", "type": "address"},
    {"name": "isFinalize", "type": "bool"},
    {"name": "nonce", "type": "uint64"},
]
STAKING_LINK_DISABLE_TRADING_USER_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "tradingUser", "type": "address"},
    {"name": "nonce", "type": "uint64"},
]
SEND_TO_EVM_WITH_DATA_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "token", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "sourceDex", "type": "string"},
    {"name": "destinationRecipient", "type": "string"},
    {"name": "addressEncoding", "type": "string"},
    {"name": "destinationChainId", "type": "uint32"},
    {"name": "gasLimit", "type": "uint64"},
    {"name": "data", "type": "bytes"},
    {"name": "nonce", "type": "uint64"},
]

# L1 action types the SDK signs with vault_address=None even when trading for
# a vault/sub-account; everything else carries the vault address.
L1_NO_VAULT_TYPES = frozenset({
    "createSubAccount", "subAccountModify", "subAccountTransfer", "subAccountSpotTransfer",
    "vaultTransfer", "setReferrer", "registerReferrer", "setDisplayName", "createVault",
    "vaultModify", "vaultDistribute", "spotDeploy", "perpDeploy", "CSignerAction",
    "CValidatorAction", "evmUserModify", "claimRewards", "borrowLend", "userOutcome",
    "spotUser", "reserveRequestWeight",
})


@dataclass
class ActionRequest:
    """Everything needed to sign and send one action (or to dry-run it)."""
    action: dict
    nonce: int
    scheme: str  # "l1" | "user"
    vault_address: str | None = None
    sign_types: list[dict] | None = None
    primary_type: str | None = None
    # approveAgent signs agentName="" but posts the action without the key
    # when the agent is unnamed (SDK behaviour); list keys to drop post-sign.
    drop_after_sign: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def action_type(self) -> str:
        return str(self.action.get("type"))


def _l1(action: dict, nonce: int, vault_address: str | None = None) -> ActionRequest:
    vault = None if action["type"] in L1_NO_VAULT_TYPES else vault_address
    return ActionRequest(action=action, nonce=nonce, scheme="l1",
                         vault_address=vault.lower() if vault else None)


def _user(action: dict, nonce: int, types: list[dict], primary: str) -> ActionRequest:
    return ActionRequest(action=action, nonce=nonce, scheme="user", sign_types=types,
                         primary_type=f"HyperliquidTransaction:{primary}")


# --------------------------------------------------------------------------
# Orders (L1)
# --------------------------------------------------------------------------

def order_wire(asset: int, is_buy: bool, sz: float, limit_px: float, order_type: dict,
               reduce_only: bool, cloid: str | None) -> dict:
    """One order in wire form, via the SDK's own converter."""
    req: dict[str, Any] = {
        "coin": "", "is_buy": is_buy, "sz": sz, "limit_px": limit_px,
        "order_type": order_type, "reduce_only": reduce_only,
    }
    if cloid:
        req["cloid"] = Cloid.from_str(cloid)
    return order_request_to_order_wire(req, asset)


def orders(wires: list[dict], nonce: int, *, grouping: Any = "na", builder: dict | None = None,
           vault_address: str | None = None) -> ActionRequest:
    if builder:
        builder = {"b": builder["b"].lower(), "f": int(builder["f"])}
    return _l1(order_wires_to_order_action(wires, builder, grouping), nonce, vault_address)


def cancel(cancels: list[tuple[int, int]], nonce: int, *,
           vault_address: str | None = None) -> ActionRequest:
    action = {"type": "cancel", "cancels": [{"a": a, "o": o} for a, o in cancels]}
    return _l1(action, nonce, vault_address)


def cancel_by_cloid(cancels: list[tuple[int, str]], nonce: int, *,
                    vault_address: str | None = None) -> ActionRequest:
    action = {"type": "cancelByCloid",
              "cancels": [{"asset": a, "cloid": c} for a, c in cancels]}
    return _l1(action, nonce, vault_address)


def batch_modify(modifies: list[tuple[int | str, dict]], nonce: int, *,
                 always_place: bool = False, vault_address: str | None = None) -> ActionRequest:
    action: dict[str, Any] = {"type": "batchModify",
                              "modifies": [{"oid": oid, "order": wire} for oid, wire in modifies]}
    if always_place:
        # Must be omitted when false: actions hashed with a=false are rejected.
        action["a"] = True
    return _l1(action, nonce, vault_address)


def schedule_cancel(time_ms: int | None, nonce: int, *,
                    vault_address: str | None = None) -> ActionRequest:
    action: dict[str, Any] = {"type": "scheduleCancel"}
    if time_ms is not None:
        action["time"] = time_ms
    return _l1(action, nonce, vault_address)


def twap_order(asset: int, is_buy: bool, sz: str, reduce_only: bool, minutes: int,
               randomize: bool, nonce: int, *, vault_address: str | None = None) -> ActionRequest:
    action = {"type": "twapOrder",
              "twap": {"a": asset, "b": is_buy, "s": sz, "r": reduce_only,
                       "m": minutes, "t": randomize}}
    return _l1(action, nonce, vault_address)


def twap_cancel(asset: int, twap_id: int, nonce: int, *,
                vault_address: str | None = None) -> ActionRequest:
    return _l1({"type": "twapCancel", "a": asset, "t": twap_id}, nonce, vault_address)


def update_leverage(asset: int, is_cross: bool, leverage: int, nonce: int, *,
                    vault_address: str | None = None) -> ActionRequest:
    action = {"type": "updateLeverage", "asset": asset, "isCross": is_cross,
              "leverage": leverage}
    return _l1(action, nonce, vault_address)


def update_isolated_margin(asset: int, ntli: int, nonce: int, *,
                           vault_address: str | None = None) -> ActionRequest:
    # isBuy has no effect until hedge mode exists; the SDK always sends True.
    action = {"type": "updateIsolatedMargin", "asset": asset, "isBuy": True, "ntli": ntli}
    return _l1(action, nonce, vault_address)


def top_up_isolated_only_margin(asset: int, leverage: str, nonce: int, *,
                                vault_address: str | None = None) -> ActionRequest:
    action = {"type": "topUpIsolatedOnlyMargin", "asset": asset, "leverage": leverage}
    return _l1(action, nonce, vault_address)


def noop(nonce: int, *, vault_address: str | None = None) -> ActionRequest:
    return _l1({"type": "noop"}, nonce, vault_address)


# --------------------------------------------------------------------------
# Own-fund movements
# --------------------------------------------------------------------------

def usd_class_transfer(amount: str, to_perp: bool, nonce: int, *,
                       sub_account: str | None = None) -> ActionRequest:
    if sub_account:
        amount = f"{amount} subaccount:{sub_account.lower()}"
    action = {"type": "usdClassTransfer", "amount": amount, "toPerp": to_perp, "nonce": nonce}
    return _user(action, nonce, USD_CLASS_TRANSFER_SIGN_TYPES, "UsdClassTransfer")


def send_asset(destination: str, source_dex: str, destination_dex: str, token: str,
               amount: str, nonce: int, *, from_sub_account: str = "") -> ActionRequest:
    action = {
        "type": "sendAsset", "destination": destination.lower(), "sourceDex": source_dex,
        "destinationDex": destination_dex, "token": token, "amount": amount,
        "fromSubAccount": from_sub_account.lower(), "nonce": nonce,
    }
    return _user(action, nonce, SEND_ASSET_SIGN_TYPES, "SendAsset")


def agent_send_asset(destination: str, source_dex: str, destination_dex: str, token: str,
                     amount: str, nonce: int, *, from_sub_account: str = "") -> ActionRequest:
    """sendAsset for API wallets: L1-signed, destination must be the account itself."""
    action = {
        "type": "agentSendAsset", "destination": destination.lower(), "sourceDex": source_dex,
        "destinationDex": destination_dex, "token": token, "amount": amount,
        "fromSubAccount": from_sub_account.lower(), "nonce": nonce,
    }
    return _l1(action, nonce)


def sub_account_transfer(sub_account: str, is_deposit: bool, usd: int, nonce: int) -> ActionRequest:
    action = {"type": "subAccountTransfer", "subAccountUser": sub_account.lower(),
              "isDeposit": is_deposit, "usd": usd}
    return _l1(action, nonce)


def sub_account_spot_transfer(sub_account: str, is_deposit: bool, token: str, amount: str,
                              nonce: int) -> ActionRequest:
    action = {"type": "subAccountSpotTransfer", "subAccountUser": sub_account.lower(),
              "isDeposit": is_deposit, "token": token, "amount": amount}
    return _l1(action, nonce)


def vault_transfer(vault: str, is_deposit: bool, usd: int, nonce: int) -> ActionRequest:
    action = {"type": "vaultTransfer", "vaultAddress": vault.lower(),
              "isDeposit": is_deposit, "usd": usd}
    return _l1(action, nonce)


def c_deposit(wei: int, nonce: int) -> ActionRequest:
    return _user({"type": "cDeposit", "wei": wei, "nonce": nonce}, nonce,
                 C_TRANSFER_SIGN_TYPES, "CDeposit")


def c_withdraw(wei: int, nonce: int) -> ActionRequest:
    return _user({"type": "cWithdraw", "wei": wei, "nonce": nonce}, nonce,
                 C_TRANSFER_SIGN_TYPES, "CWithdraw")


def token_delegate(validator: str, wei: int, is_undelegate: bool, nonce: int) -> ActionRequest:
    action = {"validator": validator.lower(), "wei": wei, "isUndelegate": is_undelegate,
              "nonce": nonce, "type": "tokenDelegate"}
    return _user(action, nonce, TOKEN_DELEGATE_TYPES, "TokenDelegate")


def claim_rewards(nonce: int) -> ActionRequest:
    return _l1({"type": "claimRewards"}, nonce)


def borrow_lend(operation: str, token: int, amount: str | None, nonce: int) -> ActionRequest:
    action = {"type": "borrowLend", "operation": operation, "token": token, "amount": amount}
    return _l1(action, nonce)


def user_outcome(operation: str, nonce: int, *, outcome: int | None = None,
                 question: int | None = None, amount: str | None = None) -> ActionRequest:
    if operation == "split":
        body = {"splitOutcome": {"outcome": outcome, "amount": amount}}
    elif operation == "merge":
        body = {"mergeOutcome": {"outcome": outcome, "amount": amount}}
    elif operation == "merge_question":
        body = {"mergeQuestion": {"question": question, "amount": amount}}
    elif operation == "negate":
        body = {"negateOutcome": {"question": question, "outcome": outcome, "amount": amount}}
    else:
        raise ValueError(f"unknown outcome operation {operation!r}")
    return _l1({"type": "userOutcome", **body}, nonce)


# --------------------------------------------------------------------------
# Funds leaving the account
# --------------------------------------------------------------------------

def usd_send(destination: str, amount: str, nonce: int) -> ActionRequest:
    action = {"destination": destination.lower(), "amount": amount, "time": nonce,
              "type": "usdSend"}
    return _user(action, nonce, USD_SEND_SIGN_TYPES, "UsdSend")


def spot_send(destination: str, token: str, amount: str, nonce: int) -> ActionRequest:
    action = {"destination": destination.lower(), "amount": amount, "token": token,
              "time": nonce, "type": "spotSend"}
    return _user(action, nonce, SPOT_TRANSFER_SIGN_TYPES, "SpotSend")


def withdraw3(destination: str, amount: str, nonce: int) -> ActionRequest:
    action = {"destination": destination.lower(), "amount": amount, "time": nonce,
              "type": "withdraw3"}
    return _user(action, nonce, WITHDRAW_SIGN_TYPES, "Withdraw")


def send_to_evm_with_data(token: str, amount: str, source_dex: str, destination_recipient: str,
                          address_encoding: str, destination_chain_id: int, gas_limit: int,
                          data: str, nonce: int) -> ActionRequest:
    action = {
        "type": "sendToEvmWithData", "token": token, "amount": amount, "sourceDex": source_dex,
        "destinationRecipient": destination_recipient, "addressEncoding": address_encoding,
        "destinationChainId": destination_chain_id, "gasLimit": gas_limit, "data": data,
        "nonce": nonce,
    }
    return _user(action, nonce, SEND_TO_EVM_WITH_DATA_SIGN_TYPES, "SendToEvmWithData")


# --------------------------------------------------------------------------
# Account administration
# --------------------------------------------------------------------------

def approve_agent(agent_address: str, name: str | None, nonce: int) -> ActionRequest:
    action = {"type": "approveAgent", "agentAddress": agent_address, "agentName": name or "",
              "nonce": nonce}
    req = _user(action, nonce, APPROVE_AGENT_SIGN_TYPES, "ApproveAgent")
    if name is None:
        req.drop_after_sign = ("agentName",)
    return req


def approve_builder_fee(builder: str, max_fee_rate: str, nonce: int) -> ActionRequest:
    action = {"maxFeeRate": max_fee_rate, "builder": builder.lower(), "nonce": nonce,
              "type": "approveBuilderFee"}
    return _user(action, nonce, APPROVE_BUILDER_FEE_SIGN_TYPES, "ApproveBuilderFee")


def create_sub_account(name: str, nonce: int) -> ActionRequest:
    return _l1({"type": "createSubAccount", "name": name}, nonce)


def sub_account_modify(sub_account: str, name: str, nonce: int) -> ActionRequest:
    return _l1({"type": "subAccountModify", "subAccountUser": sub_account.lower(),
                "name": name}, nonce)


def set_referrer(code: str, nonce: int) -> ActionRequest:
    return _l1({"type": "setReferrer", "code": code}, nonce)


def register_referrer(code: str, nonce: int) -> ActionRequest:
    return _l1({"type": "registerReferrer", "code": code}, nonce)


def set_display_name(name: str, nonce: int) -> ActionRequest:
    return _l1({"type": "setDisplayName", "displayName": name}, nonce)


def user_set_abstraction(user: str, abstraction: str, nonce: int) -> ActionRequest:
    action = {"type": "userSetAbstraction", "user": user.lower(), "abstraction": abstraction,
              "nonce": nonce}
    return _user(action, nonce, USER_SET_ABSTRACTION_SIGN_TYPES, "UserSetAbstraction")


AGENT_ABSTRACTION_CODES = {"disabled": "i", "unifiedAccount": "u", "portfolioMargin": "p"}


def agent_set_abstraction(abstraction: str, nonce: int, *,
                          vault_address: str | None = None) -> ActionRequest:
    action = {"type": "agentSetAbstraction",
              "abstraction": AGENT_ABSTRACTION_CODES[abstraction]}
    return _l1(action, nonce, vault_address)


def user_dex_abstraction(user: str, enabled: bool, nonce: int) -> ActionRequest:
    action = {"type": "userDexAbstraction", "user": user.lower(), "enabled": enabled,
              "nonce": nonce}
    return _user(action, nonce, USER_DEX_ABSTRACTION_SIGN_TYPES, "UserDexAbstraction")


def user_portfolio_margin(user: str, enabled: bool, nonce: int) -> ActionRequest:
    action = {"type": "userPortfolioMargin", "user": user.lower(), "enabled": enabled,
              "nonce": nonce}
    return _user(action, nonce, USER_PORTFOLIO_MARGIN_SIGN_TYPES, "UserPortfolioMargin")


def spot_dusting(opt_out: bool, nonce: int) -> ActionRequest:
    return _l1({"type": "spotUser", "toggleSpotDusting": {"optOut": opt_out}}, nonce)


def evm_user_modify(using_big_blocks: bool, nonce: int) -> ActionRequest:
    return _l1({"type": "evmUserModify", "usingBigBlocks": using_big_blocks}, nonce)


def reserve_request_weight(weight: int, nonce: int, *,
                           destination: str | None = None) -> ActionRequest:
    action: dict[str, Any] = {"type": "reserveRequestWeight", "weight": weight}
    if destination:  # skipped in hashing when unset
        action["destination"] = destination.lower()
    return _l1(action, nonce)


def link_staking_user(user: str, is_finalize: bool, nonce: int) -> ActionRequest:
    action = {"type": "linkStakingUser", "user": user.lower(), "isFinalize": is_finalize,
              "nonce": nonce}
    return _user(action, nonce, LINK_STAKING_USER_SIGN_TYPES, "LinkStakingUser")


def staking_link_disable_trading_user(trading_user: str, nonce: int) -> ActionRequest:
    action = {"type": "stakingLinkDisableTradingUser", "tradingUser": trading_user.lower(),
              "nonce": nonce}
    return _user(action, nonce, STAKING_LINK_DISABLE_TRADING_USER_SIGN_TYPES,
                 "StakingLinkDisableTradingUser")


def create_vault(name: str, description: str, initial_usd: int, nonce: int) -> ActionRequest:
    action = {"type": "createVault", "name": name, "description": description,
              "initialUsd": initial_usd, "nonce": nonce}
    return _l1(action, nonce)


def vault_modify(vault: str, allow_deposits: bool | None, always_close_on_withdraw: bool | None,
                 nonce: int) -> ActionRequest:
    action = {"type": "vaultModify", "vaultAddress": vault.lower(),
              "allowDeposits": allow_deposits, "alwaysCloseOnWithdraw": always_close_on_withdraw}
    return _l1(action, nonce)


def vault_distribute(vault: str, usd: int, nonce: int) -> ActionRequest:
    return _l1({"type": "vaultDistribute", "vaultAddress": vault.lower(), "usd": usd}, nonce)


def convert_to_multi_sig_user(authorized_users: list[str], threshold: int,
                              nonce: int) -> ActionRequest:
    signers = {"authorizedUsers": sorted(u.lower() for u in authorized_users),
               "threshold": threshold}
    action = {"type": "convertToMultiSigUser", "signers": json.dumps(signers), "nonce": nonce}
    return _user(action, nonce, CONVERT_TO_MULTI_SIG_USER_SIGN_TYPES, "ConvertToMultiSigUser")


# --------------------------------------------------------------------------
# Raw L1 actions (advanced module)
# --------------------------------------------------------------------------

# Types with a dedicated, guarded tool (or a different signing scheme). The
# raw escape hatch refuses them so it can never be used to skip a guard.
RAW_L1_FORBIDDEN_TYPES = frozenset({
    "order", "modify", "batchModify", "cancel", "cancelByCloid", "scheduleCancel",
    "twapOrder", "twapCancel", "updateLeverage", "updateIsolatedMargin",
    "topUpIsolatedOnlyMargin", "subAccountTransfer", "subAccountSpotTransfer", "vaultTransfer",
    "agentSendAsset", "borrowLend", "userOutcome", "claimRewards", "createVault",
    "vaultModify", "vaultDistribute", "createSubAccount", "subAccountModify", "setReferrer",
    "registerReferrer", "setDisplayName", "spotUser", "evmUserModify", "reserveRequestWeight",
    "noop", "agentSetAbstraction", "agentEnableDexAbstraction", "multiSig",
    # user-signed (EIP-712) actions never go through the L1 path
    "usdSend", "spotSend", "withdraw3", "usdClassTransfer", "sendAsset", "sendToEvmWithData",
    "approveAgent", "approveBuilderFee", "tokenDelegate", "cDeposit", "cWithdraw",
    "userSetAbstraction", "userDexAbstraction", "userPortfolioMargin", "linkStakingUser",
    "stakingLinkDisableTradingUser", "convertToMultiSigUser",
})


def raw_l1(action: dict, nonce: int, *, vault_address: str | None = None) -> ActionRequest:
    t = action.get("type") if isinstance(action, dict) else None
    if not isinstance(t, str) or not t:
        raise ValueError("action must be an object with a string 'type'")
    if t in RAW_L1_FORBIDDEN_TYPES:
        raise ValueError(
            f"action type {t!r} has a dedicated tool with its own guards; use that tool"
        )
    return _l1(dict(action), nonce, vault_address)
