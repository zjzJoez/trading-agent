"""Regression tests for the adversarial review of hyperliquid-mcp.

Each test is one way a guard could be bypassed or an exit could fail before
the fix. Mid prices in the fake market: BTC 80000.5, ETH 2500.25,
HYPE/USDC 40.5 (spot "@1").
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from trading_agent.mcp_servers.hyperliquid import tools_admin as AD
from trading_agent.mcp_servers.hyperliquid import tools_funds as F
from trading_agent.mcp_servers.hyperliquid import tools_trade as T
from trading_agent.mcp_servers.hyperliquid.settings import SettingsError, load_settings

from .conftest import OTHER, position

ATTACKER = "0x" + "66" * 20


def only_order(out: dict) -> dict:
    (o,) = out["action"]["orders"]
    return o


# ---- reduce_only is a perp-only exemption ----------------------------------

def test_reduce_only_does_not_exempt_spot_or_outcome_buys(hl):
    hl.configure(allowed_coins=frozenset({"eth"}))
    # Was: sent a $49.5k HYPE buy past the allowlist and the $1k cap.
    out = T.order_place_trigger("HYPE/USDC", "buy", 1000, 45, "stop")
    assert out["status"] == "blocked"
    assert any("HL_ALLOWED_COINS" in r for r in out["reasons"])
    assert any("HL_MAX_ORDER_NOTIONAL_USD" in r for r in out["reasons"])
    o = only_order(T.order_place_limit("HYPE/USDC", "sell", 1, 40, reduce_only=True,
                                       dry_run=True))
    assert o["r"] is False  # cleared: meaningless on spot
    twap = T.twap_place("#12090", "buy", 5000, 30, reduce_only=True)
    assert twap["status"] == "blocked"
    assert twap["action_type"] == "twapOrder" and hl.sent == []


def test_batch_flags_are_parsed_strictly(hl):
    out = T.order_place_batch([{"coin": "ETH", "side": "buy", "size": 0.01, "price": 2400,
                                "reduce_only": "false"}], dry_run=True)
    assert only_order(out)["r"] is False  # bool("false") would have been True
    with pytest.raises(ValueError, match="reduce_only"):
        T.order_place_batch([{"coin": "ETH", "side": "buy", "size": 0.01, "price": 2400,
                              "reduce_only": "yes"}])


# ---- exposure is valued where the order actually fills ---------------------

def test_low_sell_limit_is_valued_at_the_bid_not_the_limit(hl):
    # Was: valued at 0.06 × 16001 = $960 and sent; it fills at ~80000 → ~$4.8k short.
    out = T.order_place_limit("BTC", "sell", 0.06, 16001)
    assert out["status"] == "blocked"
    notional = out["summary"]["orders"][0]["notional_usd"]
    assert notional == pytest.approx(0.06 * 80000.5, rel=1e-6)
    assert any("HL_MAX_ORDER_NOTIONAL_USD" in r for r in out["reasons"])


def test_marketable_limits_are_held_to_the_slippage_band(hl):
    # Was: an IOC buy at 2× mid passed with no reasons at all.
    far = T.order_place_limit("ETH", "buy", 0.01, 5000, tif="Ioc")
    assert far["status"] == "blocked"
    assert any("through the book" in r for r in far["reasons"])
    assert T.order_place_limit("ETH", "buy", 0.01, 2300)["status"] == "sent"  # resting
    assert T.order_place_limit("ETH", "sell", 0.01, 2600)["status"] == "sent"  # resting
    # Exits get a wider band (10%) but not an unlimited one.
    assert T.order_place_limit("ETH", "sell", 1, 2300, reduce_only=True)["status"] == "sent"
    assert T.order_place_limit("ETH", "sell", 1, 1000, reduce_only=True)["status"] == "blocked"


def test_opening_twap_needs_slippage_headroom(hl):
    hl.configure(max_slippage=0.02, default_slippage=0.01)
    out = T.twap_place("ETH", "buy", 0.1, 30)
    assert out["status"] == "blocked" and any("TWAP" in r for r in out["reasons"])
    hl.api.positions[""] = [position("ETH", "0.1", value="250", margin="50")]
    assert T.twap_place("ETH", "sell", 0.1, 30, reduce_only=True)["status"] == "sent"


# ---- daily budget ----------------------------------------------------------

def _write_audit(hl, **rec):
    with (hl.tmp_path / "audit.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def test_daily_budget_counts_sent_opening_orders(hl):
    hl.configure(max_order_notional_usd=1000, max_daily_notional_usd=1500)
    assert T.order_place_limit("ETH", "buy", 0.3, 2400)["status"] == "sent"      # $720
    assert T.order_place_limit("ETH", "buy", 0.3, 2400, dry_run=True)["status"] == "dry_run"
    second = T.order_place_limit("ETH", "buy", 0.35, 2400)                         # +$840
    assert second["status"] == "blocked"
    assert any("HL_MAX_DAILY_NOTIONAL_USD" in r for r in second["reasons"])
    # Exits never count and are never blocked by it.
    assert T.order_place_limit("ETH", "sell", 0.35, 2600, reduce_only=True)["status"] == "sent"


def test_daily_budget_counts_sends_with_unknown_outcome(hl):
    hl.configure(max_daily_notional_usd=1000)
    hl.api.exchange_response = TimeoutError("read timed out")
    # $720 whose send timed out: it may have landed, so it counts.
    assert T.order_place_limit("ETH", "buy", 0.3, 2400)["status"] == "error"
    hl.api.exchange_response = {"status": "ok", "response": {"type": "default"}}
    assert T.order_place_limit("ETH", "buy", 0.2, 2400)["status"] == "blocked"  # +$480


def test_daily_budget_ignores_other_days_and_networks(hl):
    hl.configure(max_daily_notional_usd=1000)
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    _write_audit(hl, ts=yesterday, network="testnet", decision="sent",
                 opening_notional_usd=5000)
    _write_audit(hl, ts=datetime.now(UTC).isoformat(), network="mainnet", decision="sent",
                 opening_notional_usd=5000)
    _write_audit(hl, ts=datetime.now(UTC).isoformat(), network="testnet", decision="blocked",
                 opening_notional_usd=5000)
    assert T.order_place_limit("ETH", "buy", 0.3, 2400)["status"] == "sent"


def test_daily_budget_is_shared_through_the_audit_log(hl):
    hl.configure(max_daily_notional_usd=1000)
    _write_audit(hl, ts=datetime.now(UTC).isoformat(), network="testnet", decision="sent",
                 opening_notional_usd=900)  # e.g. another server process
    out = T.order_place_limit("ETH", "buy", 0.1, 2400)
    assert out["status"] == "blocked"


# ---- payees and controllers must be allowlisted ----------------------------

def test_builder_code_must_be_allowlisted(hl):
    order = [{"coin": "ETH", "side": "buy", "size": 0.01, "price": 2400}]
    out = T.order_place_batch(order, builder_address=ATTACKER, builder_fee_tenths_bp=10)
    assert out["status"] == "blocked" and any("builder" in r for r in out["reasons"])
    hl.configure(withdraw_allowlist=frozenset({ATTACKER}))
    assert T.order_place_batch(order, builder_address=ATTACKER,
                               builder_fee_tenths_bp=10)["status"] == "sent"


def test_builder_fee_approval_needs_an_allowlisted_builder(hl):
    out = AD.builder_fee_approve(ATTACKER, 0.1)
    assert out["status"] == "blocked" and any("builder" in r for r in out["reasons"])
    assert AD.builder_fee_approve(ATTACKER, 0)["status"] == "sent"  # revoking is always allowed


def test_multisig_signers_must_be_allowlisted(hl):
    # Was: sent with an empty allowlist, handing the account to any key.
    out = AD.advanced_convert_to_multisig([ATTACKER], 1, dry_run=False)
    assert out["status"] == "blocked" and hl.sent == []
    me = hl.client.account_address()
    ok = AD.advanced_convert_to_multisig([me], 1, dry_run=True)
    assert ok["live_send_would_be_blocked"] is False


def test_vault_deposits_only_into_own_or_allowlisted_vaults(hl):
    me = hl.client.account_address()
    hl.api.overrides["vaultDetails"] = lambda p: {"leader": ATTACKER}
    out = F.vault_transfer(OTHER, 50_000, "deposit")
    assert out["status"] == "blocked" and any("led by" in r for r in out["reasons"])
    assert F.vault_transfer(OTHER, 50_000, "withdraw")["status"] == "sent"  # money comes back
    hl.api.overrides["vaultDetails"] = lambda p: {"leader": me}
    assert F.vault_transfer(OTHER, 100, "deposit")["status"] == "sent"


def test_sub_account_transfers_need_a_real_sub_account(hl):
    hl.api.overrides["subAccounts"] = []
    out = F.transfer_sub_account_usdc(ATTACKER, 100, "deposit")
    assert out["status"] == "blocked"
    hl.api.overrides["subAccounts"] = [{"subAccountUser": ATTACKER}]
    assert F.transfer_sub_account_spot(ATTACKER, "HYPE", 1, "deposit")["status"] == "sent"


def test_reserve_weight_is_capped(hl):
    with pytest.raises(ValueError):
        AD.account_reserve_request_weight(1_000_000_000)
    out = AD.account_reserve_request_weight(10, destination=ATTACKER)
    assert out["status"] == "blocked"


# ---- emergency exits degrade per position ----------------------------------

def test_close_all_closes_what_it_can(hl):
    hl.api.positions[""] = [position("BTC", "0.01", value="800"),
                            position("NOT_LISTED", "5", value="10")]
    out = T.position_close_all(dry_run=True)
    assert [o["a"] for o in out["action"]["orders"]] == [0]
    assert out["not_closed"][0]["coin"] == "NOT_LISTED"


def test_close_without_a_mid_uses_the_position_mark(hl):
    hl.api.overrides["allMids"] = lambda p: {}
    hl.api.positions[""] = [position("ETH", "-2", value="5000")]  # mark 2500
    o = only_order(T.position_close("ETH", dry_run=True))
    assert o["b"] is True and o["s"] == "2"
    assert 2500 < float(o["p"]) <= 2500 * 1.01


def test_cancel_all_skips_orders_it_cannot_map(hl):
    hl.api.open_orders[""] = [{"coin": "BTC", "oid": 1}, {"coin": "NOT_LISTED", "oid": 2}]
    out = T.order_cancel_all(dry_run=True)
    assert out["action"]["cancels"] == [{"a": 0, "o": 1}]
    assert out["skipped"][0]["oid"] == 2


def test_modify_keeps_a_whole_position_tpsl_at_size_zero(hl):
    hl.api.order_status = {"status": "order", "order": {"status": "open", "order": {
        "coin": "BTC", "side": "A", "limitPx": "63000", "sz": "0.0", "oid": 9,
        "reduceOnly": True, "orderType": "Stop Market", "isTrigger": True,
        "isPositionTpsl": True, "triggerPx": "70000"}}}
    out = T.order_modify(oid=9, trigger_price=71000, always_place=True, dry_run=True)
    order = out["action"]["modifies"][0]["order"]
    assert order["s"] == "0" and order["t"]["trigger"]["triggerPx"] == "71000"


def test_margin_removal_cannot_exceed_max_leverage(hl):
    hl.api.positions[""] = [position("ETH", "0.4", value="1000", margin="200")]  # 5x
    too_much = T.margin_adjust_isolated("ETH", -150)  # → $50 margin → 20x
    assert too_much["status"] == "blocked" and any("20.00x" in r for r in too_much["reasons"])
    assert T.margin_adjust_isolated("ETH", -50)["status"] == "sent"  # → 6.67x
    assert T.margin_adjust_isolated("ETH", 25)["status"] == "sent"


# ---- settings hygiene ------------------------------------------------------

@pytest.mark.parametrize("var,value", [("HL_MAX_SLIPPAGE", "nan"),
                                       ("HL_MAX_ORDER_NOTIONAL_USD", "inf"),
                                       ("HL_MAX_DAILY_NOTIONAL_USD", "nan")])
def test_non_finite_limits_are_rejected(clean_hl_env, monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(SettingsError, match="finite"):
        load_settings()


def test_address_errors_do_not_echo_a_pasted_key(clean_hl_env, monkeypatch):
    key_like = "0x" + "ab" * 32
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", key_like)
    with pytest.raises(SettingsError) as ei:
        load_settings()
    assert "ab" * 8 not in str(ei.value)
