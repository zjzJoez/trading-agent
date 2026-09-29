"""Every action the official SDK can build must come out of actions.py
byte-identical: same keys in the same order (L1 actions are msgpack-hashed,
so order is part of the signature) and the same signature.

The server builds actions itself (for dry-run, guards and per-call vaults)
instead of calling hyperliquid.exchange.Exchange; this file is what keeps
that honest. If an SDK upgrade changes an encoding, these fail.
"""
from __future__ import annotations

import eth_account
import pytest
from hyperliquid import exchange as sdk_exchange
from hyperliquid.exchange import Exchange
from hyperliquid.utils.constants import TESTNET_API_URL
from hyperliquid.utils.signing import float_to_usd_int
from hyperliquid.utils.types import Cloid

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid.client import HLClient
from trading_agent.mcp_servers.hyperliquid.settings import Settings

from .conftest import ALL_PERP_METAS, SPOT_META, TEST_KEY, FakeAPI

NONCE = 1_790_000_000_123
VAULT = "0x" + "5a" * 20
DEST = "0x" + "22" * 20
SUB = "0x" + "33" * 20
CLOID = "0x" + "12" * 16
USDC = "USDC:0x6d1e7cde53ba9467b783cb7c530ce054"
HYPE = "HYPE:0x0d01dc56dcaaca66ad901c959b4011ec"
AGENT_HEX = "ab" * 32


def ordered(x):
    """Structure with dict key order made significant."""
    if isinstance(x, dict):
        return [(k, ordered(v)) for k, v in x.items()]
    if isinstance(x, list | tuple):
        return [ordered(v) for v in x]
    return x


@pytest.fixture()
def sdk(monkeypatch):
    """Factory for SDK Exchanges whose _post_action captures instead of posting."""
    captured: list[tuple[dict, dict, int]] = []
    monkeypatch.setattr(sdk_exchange, "get_timestamp_ms", lambda: NONCE)
    monkeypatch.setattr(sdk_exchange.secrets, "token_hex", lambda n: AGENT_HEX)

    def post(self, action, signature, nonce):
        captured.append((action, signature, nonce))
        return {"status": "ok"}

    monkeypatch.setattr(Exchange, "_post_action", post)
    wallet = eth_account.Account.from_key(TEST_KEY)

    def make(vault_address=None):
        return Exchange(wallet, TESTNET_API_URL, meta=ALL_PERP_METAS[0], spot_meta=SPOT_META,
                        vault_address=vault_address)

    return make, captured


@pytest.fixture()
def ours():
    return HLClient(Settings(network="testnet", private_key=TEST_KEY), api=FakeAPI())


def assert_same(ours_client, req, captured, *, l1: bool):
    sdk_action, sdk_sig, sdk_nonce = captured[-1]
    sig = ours_client.sign(req)
    assert req.nonce == sdk_nonce
    if l1:
        assert ordered(req.action) == ordered(sdk_action)
    else:
        assert req.action == sdk_action
    assert sig == sdk_sig


LIMIT = {"limit": {"tif": "Gtc"}}


def test_limit_order_with_cloid(sdk, ours):
    make, cap = sdk
    make().order("BTC", True, 0.001, 80000.0, LIMIT, cloid=Cloid(CLOID))
    req = A.orders([A.order_wire(0, True, 0.001, 80000.0, LIMIT, False, CLOID)], NONCE)
    assert_same(ours, req, cap, l1=True)


def test_trigger_order_reduce_only(sdk, ours):
    make, cap = sdk
    ot = {"trigger": {"triggerPx": 2450.0, "isMarket": True, "tpsl": "sl"}}
    make().order("ETH", False, 0.5, 2400.0, ot, reduce_only=True)
    req = A.orders([A.order_wire(1, False, 0.5, 2400.0, ot, True, None)], NONCE)
    assert_same(ours, req, cap, l1=True)


def test_bulk_orders_grouping_and_builder(sdk, ours):
    make, cap = sdk
    reqs = [
        {"coin": "ETH", "is_buy": True, "sz": 0.1, "limit_px": 2500.0, "order_type": LIMIT,
         "reduce_only": False},
        {"coin": "ETH", "is_buy": False, "sz": 0.1, "limit_px": 3000.0, "reduce_only": True,
         "order_type": {"trigger": {"triggerPx": 3000.0, "isMarket": True, "tpsl": "tp"}}},
    ]
    make().bulk_orders(reqs, builder={"b": "0x" + "AB" * 20, "f": 10}, grouping="normalTpsl")
    wires = [A.order_wire(1, r["is_buy"], r["sz"], r["limit_px"], r["order_type"],
                          r["reduce_only"], None) for r in reqs]
    req = A.orders(wires, NONCE, grouping="normalTpsl", builder={"b": "0x" + "AB" * 20, "f": 10})
    assert_same(ours, req, cap, l1=True)


def test_spot_order(sdk, ours):
    make, cap = sdk
    make().order("@1", True, 1.5, 40.0, {"limit": {"tif": "Ioc"}})
    req = A.orders([A.order_wire(10001, True, 1.5, 40.0, {"limit": {"tif": "Ioc"}}, False,
                                 None)], NONCE)
    assert_same(ours, req, cap, l1=True)


def test_order_for_a_vault_signs_with_the_vault(sdk, ours):
    make, cap = sdk
    make(vault_address=VAULT).order("BTC", False, 0.002, 81000.0, LIMIT)
    req = A.orders([A.order_wire(0, False, 0.002, 81000.0, LIMIT, False, None)], NONCE,
                   vault_address=VAULT)
    assert req.vault_address == VAULT
    assert_same(ours, req, cap, l1=True)


def test_cancels(sdk, ours):
    make, cap = sdk
    make(vault_address=VAULT).cancel("BTC", 123)
    assert_same(ours, A.cancel([(0, 123)], NONCE, vault_address=VAULT), cap, l1=True)
    make().cancel_by_cloid("ETH", Cloid(CLOID))
    assert_same(ours, A.cancel_by_cloid([(1, CLOID)], NONCE), cap, l1=True)


@pytest.mark.parametrize("oid", [123, CLOID])
def test_modify(sdk, ours, oid):
    make, cap = sdk
    alo = {"limit": {"tif": "Alo"}}
    make().modify_order(Cloid(oid) if isinstance(oid, str) else oid, "BTC", True, 0.002,
                        79000.0, alo, cloid=Cloid(CLOID))
    req = A.batch_modify([(oid, A.order_wire(0, True, 0.002, 79000.0, alo, False, CLOID))],
                         NONCE)
    assert_same(ours, req, cap, l1=True)


def test_schedule_cancel(sdk, ours):
    make, cap = sdk
    make().schedule_cancel(NONCE + 60_000)
    assert_same(ours, A.schedule_cancel(NONCE + 60_000, NONCE), cap, l1=True)
    make().schedule_cancel(None)
    assert_same(ours, A.schedule_cancel(None, NONCE), cap, l1=True)


def test_leverage_and_isolated_margin(sdk, ours):
    make, cap = sdk
    make().update_leverage(5, "BTC", is_cross=False)
    assert_same(ours, A.update_leverage(0, False, 5, NONCE), cap, l1=True)
    make().update_isolated_margin(-12.5, "ETH")
    assert_same(ours, A.update_isolated_margin(1, float_to_usd_int(-12.5), NONCE), cap, l1=True)


def test_internal_transfers(sdk, ours):
    make, cap = sdk
    make().usd_class_transfer(1.5, True)
    assert_same(ours, A.usd_class_transfer("1.5", True, NONCE), cap, l1=False)
    make(vault_address=SUB).usd_class_transfer(1.5, False)
    assert_same(ours, A.usd_class_transfer("1.5", False, NONCE, sub_account=SUB), cap, l1=False)
    make().send_asset(DEST, "", "spot", USDC, 2.5)
    assert_same(ours, A.send_asset(DEST, "", "spot", USDC, "2.5", NONCE), cap, l1=False)
    make().sub_account_transfer(SUB, True, 5_000_000)
    assert_same(ours, A.sub_account_transfer(SUB, True, 5_000_000, NONCE), cap, l1=True)
    make().sub_account_spot_transfer(SUB, False, HYPE, 1.25)
    assert_same(ours, A.sub_account_spot_transfer(SUB, False, HYPE, "1.25", NONCE), cap, l1=True)
    make().vault_usd_transfer(VAULT, True, 100_000_000)
    assert_same(ours, A.vault_transfer(VAULT, True, 100_000_000, NONCE), cap, l1=True)
    make().token_delegate(DEST, 100_000_000, False)
    assert_same(ours, A.token_delegate(DEST, 100_000_000, False, NONCE), cap, l1=False)


def test_outbound_transfers(sdk, ours):
    make, cap = sdk
    make().usd_transfer(1.5, DEST)
    assert_same(ours, A.usd_send(DEST, "1.5", NONCE), cap, l1=False)
    make().spot_transfer(1.5, DEST, HYPE)
    assert_same(ours, A.spot_send(DEST, HYPE, "1.5", NONCE), cap, l1=False)
    make().withdraw_from_bridge(2.5, DEST)
    assert_same(ours, A.withdraw3(DEST, "2.5", NONCE), cap, l1=False)


@pytest.mark.parametrize("name", ["bot", None])
def test_approve_agent(sdk, ours, name):
    make, cap = sdk
    _resp, key = make().approve_agent(name)
    agent = eth_account.Account.from_key(key).address
    sdk_action, sdk_sig, _ = cap[-1]
    req = A.approve_agent(agent.lower(), name, NONCE)
    sig = ours.sign(req)
    assert sig == sdk_sig  # EIP-712 address fields are case-insensitive
    assert {**req.action, "agentAddress": agent} == sdk_action
    assert ("agentName" in req.action) == (name is not None)


def test_account_settings(sdk, ours):
    make, cap = sdk
    make().approve_builder_fee(DEST, "0.01%")
    assert_same(ours, A.approve_builder_fee(DEST, "0.01%", NONCE), cap, l1=False)
    make().convert_to_multi_sig_user([SUB, DEST], 2)
    assert_same(ours, A.convert_to_multi_sig_user([SUB, DEST], 2, NONCE), cap, l1=False)
    make().create_sub_account("alpha")
    assert_same(ours, A.create_sub_account("alpha", NONCE), cap, l1=True)
    make().set_referrer("CODE")
    assert_same(ours, A.set_referrer("CODE", NONCE), cap, l1=True)
    make().use_big_blocks(True)
    assert_same(ours, A.evm_user_modify(True, NONCE), cap, l1=True)
    make().user_dex_abstraction(DEST, True)
    assert_same(ours, A.user_dex_abstraction(DEST, True, NONCE), cap, l1=False)
    make().user_set_abstraction(DEST, "unifiedAccount")
    assert_same(ours, A.user_set_abstraction(DEST, "unifiedAccount", NONCE), cap, l1=False)
    make(vault_address=VAULT).agent_set_abstraction("p")
    assert_same(ours, A.agent_set_abstraction("portfolioMargin", NONCE, vault_address=VAULT),
                cap, l1=True)
    make(vault_address=VAULT).noop(NONCE)
    assert_same(ours, A.noop(NONCE, vault_address=VAULT), cap, l1=True)


def test_raw_l1_refuses_guarded_types():
    for t in ("order", "cancel", "usdSend", "withdraw3", "approveAgent", "vaultTransfer"):
        with pytest.raises(ValueError):
            A.raw_l1({"type": t}, NONCE)
    req = A.raw_l1({"type": "perpDeploy", "setOracle": {"dex": "x"}}, NONCE, vault_address=VAULT)
    assert req.vault_address is None  # deployer actions never carry a vault
