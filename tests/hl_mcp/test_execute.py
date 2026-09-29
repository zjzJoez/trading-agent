"""core.execute — the single choke point every signed action goes through —
and the exchange-response normaliser."""
from __future__ import annotations

import json

import eth_account
import pytest
from hyperliquid.utils.signing import (
    recover_agent_or_user_from_l1_action,
    recover_user_from_user_signed_action,
)

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid import tools_funds as F
from trading_agent.mcp_servers.hyperliquid import tools_trade as T
from trading_agent.mcp_servers.hyperliquid.client import (
    HyperliquidAPIError,
    normalize_exchange_response,
)

from .conftest import OTHER, TEST_KEY

ME = eth_account.Account.from_key(TEST_KEY).address.lower()


# ---------------------------------------------------------------- normaliser

@pytest.mark.parametrize("resp,ok,n_err", [
    ({"status": "ok", "response": {"type": "default"}}, True, 0),
    ({"status": "ok", "response": {"type": "order", "data": {"statuses": [
        {"resting": {"oid": 1}}, {"error": "Order must have minimum value of $10."},
        {"filled": {"totalSz": "1", "avgPx": "2", "oid": 3}}]}}}, False, 1),
    ({"status": "ok", "response": {"type": "cancel", "data": {"statuses": [
        "success", {"error": "Order was never placed, already canceled, or filled."}]}}}, False, 1),
    ({"status": "ok", "response": {"type": "twapOrder",
                                   "data": {"status": {"running": {"twapId": 7}}}}}, True, 0),
    ({"status": "ok", "response": {"type": "twapOrder",
                                   "data": {"status": {"error": "Invalid TWAP duration"}}}},
     False, 1),
    ({"status": "err", "response": "User or API Wallet 0x1 does not exist."}, False, 1),
    ("garbage", False, 1),
])
def test_normalize_flags_every_rejection(resp, ok, n_err):
    out = normalize_exchange_response(resp)
    assert out["ok"] is ok and len(out["errors"]) == n_err


# ---------------------------------------------------------------- gates

def test_dry_run_signs_and_sends_nothing(hl):
    out = T.leverage_update("BTC", 3, dry_run=True)
    assert out["status"] == "dry_run" and out["sent"] is False
    assert out["live_send_would_be_blocked"] is False
    assert out["action"] == {"type": "updateLeverage", "asset": 0, "isCross": True, "leverage": 3}
    assert hl.sent == []
    assert json.loads(hl.audit_lines()[-1])["decision"] == "dry_run"


def test_global_dry_run_overrides_the_call(hl):
    hl.configure(dry_run=True)
    out = T.leverage_update("BTC", 3)
    assert out["status"] == "dry_run" and hl.sent == []


@pytest.mark.parametrize("overrides,needle", [
    ({"write_modules": frozenset()}, "write module 'trade'"),
    ({"read_only": True}, "read-only"),
    ({"private_key": None}, "no signing key"),
    ({"network": "mainnet"}, "HL_ALLOW_MAINNET_WRITES"),
])
def test_each_gate_blocks_before_signing(hl, overrides, needle):
    hl.configure(**overrides)
    out = T.leverage_update("BTC", 3)
    assert out["status"] == "blocked" and out["sent"] is False
    assert any(needle in r for r in out["reasons"])
    assert hl.sent == []
    assert json.loads(hl.audit_lines()[-1])["decision"] == "blocked"


def test_dry_run_still_reports_what_would_block(hl):
    hl.configure(write_modules=frozenset(), network="mainnet")
    out = T.leverage_update("BTC", 3, dry_run=True)
    assert out["status"] == "dry_run" and out["live_send_would_be_blocked"] is True
    assert len(out["blocked_reasons"]) == 2


def test_live_l1_send_carries_a_signature_that_recovers_to_the_signer(hl):
    out = T.leverage_update("BTC", 3)
    assert out["status"] == "sent" and out["ok"] is True
    p = hl.sent[-1]
    assert p["nonce"] == out["nonce"] and p["vaultAddress"] is None and p["expiresAfter"] is None
    signer = recover_agent_or_user_from_l1_action(p["action"], p["signature"], None, p["nonce"],
                                                  None, False)
    assert signer.lower() == ME


def test_mainnet_opt_in_signs_for_mainnet(hl):
    hl.configure(network="mainnet", allow_mainnet_writes=True)
    T.leverage_update("BTC", 3)
    p = hl.sent[-1]
    assert recover_agent_or_user_from_l1_action(p["action"], p["signature"], None, p["nonce"],
                                                None, True).lower() == ME
    # ...and the same signature is NOT valid on testnet (no cross-network replay).
    assert recover_agent_or_user_from_l1_action(p["action"], p["signature"], None, p["nonce"],
                                                None, False).lower() != ME


def test_user_signed_send_recovers_to_the_signer(hl):
    hl.configure(withdraw_allowlist=frozenset({OTHER}))
    out = F.send_usdc(OTHER, 1.5)
    assert out["status"] == "sent"
    p = hl.sent[-1]
    assert p["action"]["hyperliquidChain"] == "Testnet" and p["vaultAddress"] is None
    action = dict(p["action"])
    signer = recover_user_from_user_signed_action(action, p["signature"], A.USD_SEND_SIGN_TYPES,
                                                  "HyperliquidTransaction:UsdSend", False)
    assert signer.lower() == ME


def test_transport_failure_reports_unknown_outcome_with_the_nonce(hl):
    hl.api.exchange_response = HyperliquidAPIError("timeout")
    out = T.leverage_update("BTC", 3)
    assert out["status"] == "error" and out["sent"] == "unknown"
    assert isinstance(out["nonce"], int) and "nonce_invalidate" in out["hint"]
    assert json.loads(hl.audit_lines()[-1])["decision"] == "error"


def test_rejected_order_is_not_reported_as_success(hl):
    hl.api.exchange_response = {"status": "ok", "response": {"type": "order", "data": {
        "statuses": [{"error": "Insufficient margin to place order."}]}}}
    out = T.order_place_limit("BTC", "buy", 0.001, 80000)
    assert out["status"] == "sent" and out["ok"] is False
    assert out["order_results"][0]["result"] == "rejected"


def test_nonces_strictly_increase(hl):
    nonces = [hl.client.next_nonce() for _ in range(200)]
    assert nonces == sorted(set(nonces))


def test_audit_log_never_contains_the_private_key(hl):
    hl.configure(withdraw_allowlist=frozenset({OTHER}))
    T.leverage_update("BTC", 3)
    T.leverage_update("BTC", 3, dry_run=True)
    F.send_usdc(OTHER, 1)
    text = "\n".join(hl.audit_lines())
    assert text and TEST_KEY[2:] not in text


def test_api_wallet_cannot_sign_owner_actions_but_can_trade(hl):
    hl.configure(account_address=OTHER)  # signer != account → API wallet
    blocked = F.transfer_usdc_perp_spot(5, to="spot")
    assert blocked["status"] == "blocked"
    assert any("owner" in r for r in blocked["reasons"])
    assert T.leverage_update("BTC", 3)["status"] == "sent"
