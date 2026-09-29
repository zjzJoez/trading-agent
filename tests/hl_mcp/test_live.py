"""Live checks against the real Hyperliquid API. Marked `integration`, so the
deploy gate (-m 'not integration') skips them.

* mainnet: public reads only, no key.
* testnet: the signature oracle. A fresh random key has no account, so the
  exchange rejects every action — but the rejection names the address it
  recovered from our signature. recovered == our wallet means the action
  encoding (including msgpack key order) and the signature are exactly
  right. This covers every action type the server can send, including the
  ones the Python SDK does not implement (encoded from the TS SDK schemas).
"""
from __future__ import annotations

import re
import secrets

import eth_account
import pytest

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid import (
    core,
    server,  # noqa: F401  (register tools)
)
from trading_agent.mcp_servers.hyperliquid import tools_account as AC
from trading_agent.mcp_servers.hyperliquid import tools_admin as AD
from trading_agent.mcp_servers.hyperliquid import tools_funds as F
from trading_agent.mcp_servers.hyperliquid import tools_market as M
from trading_agent.mcp_servers.hyperliquid import tools_trade as T
from trading_agent.mcp_servers.hyperliquid.client import HLClient, normalize_exchange_response
from trading_agent.mcp_servers.hyperliquid.settings import WRITE_MODULES, Settings

pytestmark = pytest.mark.integration

DEST = "0x" + "22" * 20
COSIGNER = "0x" + "33" * 20
MASTER = "0x" + "44" * 20
CLOID = "0x" + "cd" * 16


def test_mainnet_public_reads(tmp_path):
    core.set_client(HLClient(Settings(network="mainnet", audit_path=tmp_path / "a.jsonl")))
    try:
        assert AC.server_status()["live_writes_possible"] is False
        t = M.market_get_ticker("BTC")
        assert t["mark_px"] > 0 and t["funding_rate_1h"] is not None
        assert M.market_get_orderbook("ETH", depth=2)["best_ask"] > 0
        hip3 = M.market_search("TSLA", kind="perp")["matches"]
        assert hip3 and hip3[0]["asset_id"] >= 110000
        assert M.market_get_ticker(hip3[0]["coin"])["mark_px"] > 0
        assert M.market_get_ticker("HYPE/USDC")["mark_px"] > 0
    finally:
        core.set_client(None)


@pytest.fixture(scope="module")
def wallet(tmp_path_factory):
    key = "0x" + secrets.token_hex(32)
    return key, eth_account.Account.from_key(key).address.lower(), tmp_path_factory.mktemp("hl")


def _client(wallet, **kw) -> HLClient:
    key, _me, tmp = wallet
    return HLClient(Settings(
        network="testnet", private_key=key, write_modules=frozenset(WRITE_MODULES),
        withdraw_allowlist=frozenset({DEST, COSIGNER}), max_leverage=100,
        audit_path=tmp / "audit.jsonl", agent_key_dir=tmp / "agents", **kw))


def _btc(frac: float) -> int:
    return round(float(core.mids("")["BTC"]) * frac)


def _raw(req_fn):
    """Send a builder's action directly (for encodings no tool reaches offline)."""
    c = core.client()
    return normalize_exchange_response(c.send(req_fn(c.next_nonce())))


def _pretend(fn, **answers):
    """The oracle wallet is empty. Answer some /info lookups as if it were not,
    so the tool's guards pass and its signed action reaches the exchange."""
    def run():
        c = core.client()
        real = c.info
        c.info = lambda p: answers[p["type"]](p) if p.get("type") in answers else real(p)
        return fn()
    return run


def _long_btc(p):
    size = 0.001
    return {"assetPositions": [{"type": "oneWay", "position": {
        "coin": "BTC", "szi": str(size), "positionValue": str(_btc(1) * size),
        "marginUsed": "1", "leverage": {"type": "cross", "value": 5}}}] if not p.get("dex") else []}


_WIRE = A.order_wire(0, True, 0.001, 50000.0, {"limit": {"tif": "Alo"}}, False, CLOID)
_MY_SUB = {"subAccounts": lambda p: [{"subAccountUser": DEST}]}

OWNER_CASES = {
    "order limit": lambda: T.order_place_limit("BTC", "buy", 0.0002, _btc(0.8)),
    "order market": lambda: T.order_place_market("ETH", "sell", 0.005),
    "order trigger": lambda: T.order_place_trigger("BTC", "sell", 0.0002, _btc(0.7), "stop",
                                                   reduce_only=False),
    "order priority grouping": lambda: _raw(lambda n: A.orders([_WIRE], n, grouping={"p": 1000})),
    "whole-position TP/SL (size 0)": _pretend(
        lambda: T.position_set_tpsl("BTC", take_profit_price=_btc(1.3), stop_loss_price=_btc(0.7)),
        clearinghouseState=_long_btc),
    "cancel": lambda: T.order_cancel("BTC", oid=1),
    "cancelByCloid": lambda: T.order_cancel("BTC", cloid=CLOID),
    "batchModify": lambda: _raw(lambda n: A.batch_modify([(123, _WIRE)], n)),
    "batchModify always_place": lambda: _raw(
        lambda n: A.batch_modify([(123, _WIRE)], n, always_place=True)),
    "scheduleCancel": lambda: T.order_schedule_cancel_all(delay_seconds=30),
    "scheduleCancel clear": lambda: T.order_schedule_cancel_all(clear=True),
    "updateLeverage": lambda: T.leverage_update("BTC", 3),
    "updateIsolatedMargin": lambda: T.margin_adjust_isolated("BTC", 1),
    "topUpIsolatedOnlyMargin": lambda: T.margin_set_isolated_leverage("BTC", 2.5),
    "twapOrder": lambda: T.twap_place("BTC", "buy", 0.0002, 10),
    "twapCancel": lambda: T.twap_cancel("BTC", 1),
    "noop": lambda: T.nonce_invalidate(core.client().next_nonce()),
    "usdClassTransfer": lambda: F.transfer_usdc_perp_spot(1, to="spot"),
    "sendAsset (self)": lambda: F.transfer_between_dexs(1, "", "spot"),
    "subAccountTransfer": _pretend(lambda: F.transfer_sub_account_usdc(DEST, 1, "deposit"),
                                   **_MY_SUB),
    "subAccountSpotTransfer": _pretend(
        lambda: F.transfer_sub_account_spot(DEST, "USDC", 1, "deposit"), **_MY_SUB),
    "vaultTransfer": lambda: F.vault_transfer(DEST, 5, "deposit"),
    "cDeposit": lambda: F.staking_deposit(1),
    "cWithdraw": lambda: F.staking_withdraw(1),
    "tokenDelegate": lambda: F.staking_delegate(DEST, 1),
    "claimRewards": lambda: F.staking_claim_rewards(),
    "borrowLend": lambda: F.lending_update("supply", "USDC", 1),
    "borrowLend null amount": lambda: F.lending_update("withdraw", "USDC"),
    "userOutcome": lambda: F.outcome_convert("split", outcome=1, amount=1),
    "withdraw3": lambda: F.withdraw_to_arbitrum(2, destination=DEST),
    "usdSend": lambda: F.send_usdc(DEST, 1),
    "spotSend": lambda: F.send_spot_token(DEST, "USDC", 1),
    "sendAsset (other)": lambda: F.send_asset(DEST, "USDC", 1, "", "spot"),
    "sendToEvmWithData": lambda: F.send_to_evm_with_data("USDC", 1, DEST, 1, 100000),
    "approveAgent named": lambda: AD.agent_approve(name="probe"),
    "approveAgent unnamed": lambda: AD.agent_approve(),
    "approveBuilderFee": lambda: AD.builder_fee_approve(DEST, 0.01),
    "createSubAccount": lambda: AD.sub_account_create("probe"),
    "subAccountModify": lambda: AD.sub_account_rename(DEST, "probe2"),
    "setReferrer": lambda: AD.referral_use_code("TESTNET"),
    "registerReferrer": lambda: AD.referral_create_code("PROBECODE"),
    "setDisplayName": lambda: AD.account_set_display_name("probe"),
    "userSetAbstraction": lambda: AD.account_set_abstraction("unifiedAccount"),
    "userDexAbstraction": lambda: _raw(
        lambda n: A.user_dex_abstraction(core.client().signer_address, True, n)),
    "userPortfolioMargin": lambda: AD.account_set_portfolio_margin(True),
    "spotUser": lambda: AD.account_set_spot_dusting(True),
    "evmUserModify": lambda: AD.account_set_evm_big_blocks(True),
    "reserveRequestWeight": lambda: AD.account_reserve_request_weight(10),
    "linkStakingUser": lambda: AD.staking_link_user(DEST, False),
    "stakingLinkDisableTradingUser": lambda: AD.staking_unlink_trading_user(DEST),
    "createVault": lambda: AD.vault_create("Probe Vault", "signature probe vault", 100),
    "vaultModify": lambda: AD.vault_modify(DEST, allow_deposits=False),
    "vaultDistribute": lambda: AD.vault_distribute(DEST, 1),
    "raw L1 (gossipPriorityBid)": lambda: AD.advanced_send_l1_action(
        {"type": "gossipPriorityBid", "slotId": 0, "ip": "1.2.3.4", "maxGas": 1}, dry_run=False),
    "convertToMultiSigUser": lambda: AD.advanced_convert_to_multisig(
        [DEST, COSIGNER], 1, dry_run=False),
}
# With an API-wallet key (account != signer) these go out as agent variants.
AGENT_CASES = {
    "agentSendAsset": lambda: F.transfer_between_dexs(1, "", "spot"),
    "agentSetAbstraction": lambda: AD.account_set_abstraction("unifiedAccount"),
}


def _recovered(out: dict) -> set[str]:
    text = str(out.get("errors"))
    return {a.lower() for a in re.findall(r"0x[0-9a-fA-F]{40}", text)}


@pytest.mark.parametrize("name", list(OWNER_CASES))
def test_testnet_signature_oracle_owner_key(wallet, name):
    core.set_client(_client(wallet))
    try:
        out = OWNER_CASES[name]()
    finally:
        core.set_client(None)
    assert out.get("status", "sent") == "sent" and out["ok"] is False, out
    assert _recovered(out) == {wallet[1]}, out["errors"]


@pytest.mark.parametrize("name", list(AGENT_CASES))
def test_testnet_signature_oracle_api_wallet(wallet, name):
    core.set_client(_client(wallet, account_address=MASTER))
    try:
        out = AGENT_CASES[name]()
    finally:
        core.set_client(None)
    assert out["status"] == "sent" and out["action_type"].startswith("agent"), out
    assert _recovered(out) == {wallet[1]}, out["errors"]
