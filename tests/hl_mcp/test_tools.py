"""Tool behaviour: what each tool builds, and every guard that can refuse it.

Mid prices in the fake market: BTC 80000.5, ETH 2500.25, xyz:TSLA 350.5
(USDH collateral), HYPE/USDC 40.5, FOO/HYPE 0.5, outcome #12090 0.3.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from trading_agent.mcp_servers.hyperliquid import tools_account as AC
from trading_agent.mcp_servers.hyperliquid import tools_admin as AD
from trading_agent.mcp_servers.hyperliquid import tools_funds as F
from trading_agent.mcp_servers.hyperliquid import tools_market as M
from trading_agent.mcp_servers.hyperliquid import tools_trade as T

from .conftest import OTHER, TEST_KEY, position


def only_order(out: dict) -> dict:
    orders = out["action"]["orders"]
    assert len(orders) == 1
    return orders[0]


# ---------------------------------------------------------------- orders

def test_limit_rounds_in_the_traders_favour_and_floors_size(hl):
    buy = only_order(T.order_place_limit("BTC", "buy", 0.00123456, 80000.37, dry_run=True))
    assert (buy["p"], buy["s"], buy["b"], buy["t"]) == ("80000", "0.00123", True,
                                                        {"limit": {"tif": "Gtc"}})
    sell = only_order(T.order_place_limit("xyz:TSLA", "sell", 1.5, 400.12345, dry_run=True))
    assert (sell["a"], sell["p"]) == (110000, "400.13")


def test_every_order_gets_a_client_order_id(hl):
    o = only_order(T.order_place_limit("BTC", "buy", 0.001, 80000, dry_run=True))
    assert o["c"].startswith("0x") and len(o["c"]) == 34
    mine = "0x" + "ab" * 16
    assert only_order(T.order_place_limit("BTC", "buy", 0.001, 80000, cloid=mine,
                                          dry_run=True))["c"] == mine
    with pytest.raises(ValueError):
        T.order_place_limit("BTC", "buy", 0.001, 80000, cloid="0x123")


def test_unrounded_invalid_price_is_rejected_with_nearest_ticks(hl):
    with pytest.raises(ValueError, match="80000 / 80001"):
        T.order_place_limit("BTC", "buy", 0.001, 80000.37, round_to_tick=False)


def test_notional_cap_blocks_entries_but_not_exits(hl):
    big = T.order_place_limit("BTC", "buy", 0.1, 80000)  # $8,000 > $1,000
    assert big["status"] == "blocked"
    assert any("HL_MAX_ORDER_NOTIONAL_USD" in r for r in big["reasons"])
    assert hl.sent == []
    exit_ = T.order_place_limit("BTC", "sell", 0.1, 80000, reduce_only=True)
    assert exit_["status"] == "sent"


def test_coin_allowlist(hl):
    hl.configure(allowed_coins=frozenset({"eth"}))
    assert T.order_place_limit("BTC", "buy", 0.001, 80000)["status"] == "blocked"
    assert T.order_place_limit("ETH", "buy", 0.01, 2500)["status"] == "sent"
    assert T.order_place_limit("BTC", "sell", 0.001, 80000, reduce_only=True)["status"] == "sent"


def test_leverage_above_cap_blocks_perp_entries(hl):
    hl.api.leverage = 20
    out = T.order_place_limit("BTC", "buy", 0.001, 80000)
    assert out["status"] == "blocked"
    assert any("HL_MAX_LEVERAGE" in r for r in out["reasons"])
    # spot has no leverage; exits are never blocked by it
    assert T.order_place_limit("HYPE/USDC", "buy", 1, 40)["status"] == "sent"
    assert T.position_close_all()["status"] == "nothing_to_close"


def test_leverage_lookup_failure_fails_closed(hl):
    def boom(_):
        raise RuntimeError("down")
    hl.api.overrides["activeAssetData"] = boom
    out = T.order_place_limit("BTC", "buy", 0.001, 80000)
    assert out["status"] == "blocked" and any("fail-closed" in r for r in out["reasons"])


def test_non_usd_quote_is_valued_through_its_usdc_pair(hl):
    out = T.order_place_limit("@3", "buy", 10, 0.5, dry_run=True)  # FOO/HYPE, HYPE = 40.5
    assert out["summary"]["orders"][0]["notional_usd"] == pytest.approx(202.5)
    hl.api.overrides["allMids"] = lambda p: {}  # no HYPE price → cannot value → refuse
    out = T.order_place_limit("@3", "buy", 10, 0.5)
    assert out["status"] == "blocked" and any("cannot value" in r for r in out["reasons"])


def test_market_order_is_an_ioc_limit_inside_the_slippage_bound(hl):
    buy = only_order(T.order_place_market("ETH", "buy", 0.01, slippage=0.01, dry_run=True))
    assert buy["t"] == {"limit": {"tif": "Ioc"}}
    assert float(buy["p"]) <= 2500.25 * 1.01 and float(buy["p"]) > 2500.25
    sell = only_order(T.order_place_market("ETH", "sell", 0.01, slippage=0.01, dry_run=True))
    assert 2500.25 * 0.99 <= float(sell["p"]) < 2500.25
    too_wide = T.order_place_market("ETH", "buy", 0.01, slippage=0.2)
    assert too_wide["status"] == "blocked"
    assert any("HL_MAX_SLIPPAGE" in r for r in too_wide["reasons"])


def test_outcome_orders_use_outcome_asset_ids(hl):
    o = only_order(T.order_place_market("#12090", "buy", 10.7, dry_run=True))
    assert o["a"] == 100_012_090 and o["s"] == "10" and float(o["p"]) < 1


def test_trigger_orders(hl):
    sl = only_order(T.order_place_trigger("BTC", "sell", 0.001, 70000, "stop", dry_run=True))
    assert sl["t"] == {"trigger": {"isMarket": True, "triggerPx": "70000", "tpsl": "sl"}}
    # A market stop-loss carries a worst-fill limit 10% past the trigger, like
    # Hyperliquid's frontend — limit == trigger could leave it unfilled on a gap.
    assert sl["r"] is True and sl["p"] == "63000"
    tp = only_order(T.order_place_trigger("BTC", "sell", 0.001, 95000, "take_profit",
                                          limit_price=94900, dry_run=True))
    assert tp["t"]["trigger"]["tpsl"] == "tp" and tp["t"]["trigger"]["isMarket"] is False
    assert tp["p"] == "94900"
    with pytest.raises(ValueError, match="trigger_price"):
        T.order_place_trigger("BTC", "sell", 0.001, 70000.55, "stop", round_to_tick=False)


def test_trigger_entries_are_held_to_the_slippage_cap(hl):
    entry = only_order(T.order_place_trigger("BTC", "buy", 0.001, 85000, "stop",
                                             reduce_only=False, dry_run=True))
    assert entry["p"] == "85850"  # trigger + HL_DEFAULT_SLIPPAGE (1%)
    wide = T.order_place_trigger("BTC", "buy", 0.001, 85000, "stop", reduce_only=False,
                                 slippage=0.2)
    assert wide["status"] == "blocked" and any("slippage" in r for r in wide["reasons"])
    exit_wide = T.order_place_trigger("BTC", "sell", 0.001, 70000, "stop", slippage=0.2)
    assert exit_wide["status"] == "blocked"  # exits get 10%, not unlimited
    far_limit = T.order_place_trigger("BTC", "buy", 0.001, 85000, "stop", reduce_only=False,
                                      limit_price=95000)
    assert any("trigger" in r and "through the book" in r for r in far_limit["reasons"])


def test_bracket_links_reduce_only_legs(hl):
    out = T.order_place_bracket("ETH", "buy", 0.1, entry_price=2400, take_profit_price=3000,
                                stop_loss_price=2200, dry_run=True)
    entry, tp, sl = out["action"]["orders"]
    assert out["action"]["grouping"] == "normalTpsl"
    assert entry["b"] is True and entry["r"] is False
    assert (tp["b"], tp["r"], tp["t"]["trigger"]["tpsl"]) == (False, True, "tp")
    assert (sl["b"], sl["r"], sl["t"]["trigger"]["tpsl"]) == (False, True, "sl")
    assert tp["s"] == sl["s"] == entry["s"]
    with pytest.raises(ValueError, match="stop_loss_price"):
        T.order_place_bracket("ETH", "buy", 0.1, entry_price=2400, stop_loss_price=2500)
    with pytest.raises(ValueError):
        T.order_place_bracket("HYPE/USDC", "buy", 1, entry_price=40, take_profit_price=50)


def test_batch_cap_applies_to_the_opening_total(hl):
    orders = [{"coin": "ETH", "side": "buy", "size": 0.15, "price": 2500},   # $375 each
              {"coin": "ETH", "side": "buy", "size": 0.15, "price": 2490},
              {"coin": "ETH", "side": "buy", "size": 0.15, "price": 2480}]
    out = T.order_place_batch(orders)
    assert out["status"] == "blocked" and any("batch" in r for r in out["reasons"])
    assert T.order_place_batch(orders[:2])["status"] == "sent"
    with pytest.raises(ValueError, match="unknown keys"):
        T.order_place_batch([{"coin": "ETH", "side": "buy", "size": 1, "px": 1}])
    with pytest.raises(ValueError, match="missing"):
        T.order_place_batch([{"coin": "ETH", "side": "buy", "price": 1}])


def test_batch_builder_code(hl):
    out = T.order_place_batch([{"coin": "ETH", "side": "buy", "size": 0.01, "price": 2500}],
                              builder_address="0x" + "AB" * 20, builder_fee_tenths_bp=10,
                              dry_run=True)
    assert out["action"]["builder"] == {"b": "0x" + "ab" * 20, "f": 10}


def test_sent_orders_report_per_order_results(hl):
    hl.api.exchange_response = {"status": "ok", "response": {"type": "order", "data": {
        "statuses": [{"resting": {"oid": 11}}, {"filled": {"totalSz": "0.01", "avgPx": "2501",
                                                            "oid": 12}}]}}}
    out = T.order_place_batch([{"coin": "ETH", "side": "buy", "size": 0.01, "price": 2400},
                               {"coin": "ETH", "side": "buy", "size": 0.01, "type": "market"}])
    assert out["ok"] is True
    assert [r["result"] for r in out["order_results"]] == ["resting", "filled"]
    assert out["order_results"][1]["avg_px"] == 2501.0


# ---------------------------------------------------------------- positions

def test_close_uses_the_live_position_and_is_reduce_only(hl):
    hl.api.positions[""] = [position("ETH", "-0.3")]
    o = only_order(T.position_close("ETH", dry_run=True))
    assert (o["b"], o["r"], o["s"]) == (True, True, "0.3")
    part = only_order(T.position_close("ETH", size=0.1, dry_run=True))
    assert part["s"] == "0.1"
    assert T.position_close("BTC")["status"] == "nothing_to_close"
    with pytest.raises(ValueError):
        T.position_close("HYPE/USDC")


def test_close_all_spans_every_dex(hl):
    hl.api.positions[""] = [position("BTC", "0.01")]
    hl.api.positions["xyz"] = [position("xyz:TSLA", "2.5")]
    out = T.position_close_all(dry_run=True)
    assert {o["a"] for o in out["action"]["orders"]} == {0, 110000}
    assert all(o["r"] and not o["b"] for o in out["action"]["orders"])


def test_position_tpsl(hl):
    hl.api.positions[""] = [position("BTC", "0.02")]
    out = T.position_set_tpsl("BTC", take_profit_price=90000, stop_loss_price=75000, dry_run=True)
    assert out["action"]["grouping"] == "positionTpsl"
    # size 0 = the whole position, resizing with it (what the frontend sends)
    assert all(o["s"] == "0" and o["b"] is False and o["r"] is True
               for o in out["action"]["orders"])
    fixed = T.position_set_tpsl("BTC", stop_loss_price=75000, size=0.01, dry_run=True)
    assert only_order(fixed)["s"] == "0.01"
    with pytest.raises(ValueError):
        T.position_set_tpsl("BTC", take_profit_price=70000)  # below mark for a long
    with pytest.raises(ValueError):
        T.position_set_tpsl("ETH", stop_loss_price=2000)  # no ETH position


# ---------------------------------------------------------------- modify / cancel

def _open(order: dict) -> dict:
    return {"status": "order", "order": {"order": order, "status": "open",
                                         "statusTimestamp": 1}}


def test_modify_changes_only_what_was_passed(hl):
    hl.api.order_status = _open({"coin": "BTC", "side": "B", "limitPx": "79000", "sz": "0.002",
                                 "oid": 77, "reduceOnly": False, "orderType": "Limit",
                                 "tif": "Alo", "isTrigger": False, "cloid": "0x" + "cd" * 16})
    out = T.order_modify(oid=77, price=79500, dry_run=True)
    mod = out["action"]["modifies"][0]
    assert mod["oid"] == 77 and "a" not in out["action"]
    assert mod["order"]["p"] == "79500" and mod["order"]["s"] == "0.002"
    assert mod["order"]["t"] == {"limit": {"tif": "Alo"}} and mod["order"]["c"] == "0x" + "cd" * 16


def test_modifying_a_trigger_needs_always_place(hl):
    hl.api.order_status = _open({"coin": "BTC", "side": "A", "limitPx": "70000", "sz": "0.001",
                                 "oid": 5, "reduceOnly": True, "orderType": "Stop Market",
                                 "isTrigger": True, "triggerPx": "70000"})
    with pytest.raises(ValueError, match="always_place"):
        T.order_modify(oid=5, trigger_price=71000)
    out = T.order_modify(oid=5, trigger_price=71000, always_place=True, dry_run=True)
    assert list(out["action"])[-1] == "a" and out["action"]["a"] is True
    trig = out["action"]["modifies"][0]["order"]["t"]["trigger"]
    assert trig == {"isMarket": True, "triggerPx": "71000", "tpsl": "sl"}


def test_modify_refuses_closed_orders(hl):
    hl.api.order_status = {"status": "order", "order": {"order": {"coin": "BTC"},
                                                        "status": "filled"}}
    with pytest.raises(ValueError, match="filled"):
        T.order_modify(oid=1, price=1)


def test_cancel_all_collects_orders_from_every_dex(hl):
    hl.api.open_orders[""] = [{"coin": "BTC", "oid": 1}, {"coin": "@1", "oid": 2}]
    hl.api.open_orders["xyz"] = [{"coin": "xyz:TSLA", "oid": 3}]
    out = T.order_cancel_all(dry_run=True)
    assert out["action"]["cancels"] == [{"a": 0, "o": 1}, {"a": 10001, "o": 2},
                                        {"a": 110000, "o": 3}]
    only_btc = T.order_cancel_all(coin="BTC", dry_run=True)
    assert only_btc["action"]["cancels"] == [{"a": 0, "o": 1}]
    hl.api.open_orders.clear()
    assert T.order_cancel_all()["status"] == "nothing_to_cancel"


def test_cancel_batch_and_schedule(hl):
    with pytest.raises(ValueError):
        T.order_cancel_batch([{"coin": "BTC", "oid": 1}, {"coin": "BTC", "cloid": "0x" + "1" * 32}])
    with pytest.raises(ValueError):
        T.order_schedule_cancel_all(delay_seconds=2)
    assert "time" not in T.order_schedule_cancel_all(clear=True, dry_run=True)["action"]


def test_leverage_guards(hl):
    assert T.leverage_update("BTC", 25)["status"] == "blocked"          # > HL_MAX_LEVERAGE 10
    hl.configure(max_leverage=50)
    assert T.leverage_update("BTC", 45)["status"] == "blocked"          # > asset max 40
    assert T.leverage_update("xyz:TSLA", 5)["status"] == "blocked"      # isolated-only
    assert T.leverage_update("xyz:TSLA", 5, margin_mode="isolated")["status"] == "sent"


def test_twap_and_nonce_invalidate(hl):
    out = T.twap_place("BTC", "buy", 0.005, 30, randomize=True, dry_run=True)
    assert out["action"] == {"type": "twapOrder", "twap": {"a": 0, "b": True, "s": "0.005",
                                                           "r": False, "m": 30, "t": True}}
    with pytest.raises(ValueError):
        T.twap_place("BTC", "buy", 0.005, 2)
    n = hl.client.next_nonce() - 1000
    assert T.nonce_invalidate(n, dry_run=True)["nonce"] == n
    with pytest.raises(ValueError):
        T.nonce_invalidate(1)


# ---------------------------------------------------------------- funds

def test_transfer_between_dexs_owner_vs_api_wallet(hl):
    owner = F.transfer_between_dexs(10, "", "xyz", dry_run=True)
    assert owner["action"]["type"] == "sendAsset" and owner["signing_scheme"] == "user"
    hl.configure(account_address=OTHER)
    agent = F.transfer_between_dexs(10, "", "spot", dry_run=True)
    assert agent["action"]["type"] == "agentSendAsset" and agent["signing_scheme"] == "l1"
    assert agent["action"]["destination"] == OTHER
    with pytest.raises(ValueError, match="unknown perp dex"):
        F.transfer_between_dexs(10, "", "nope")


def test_withdrawals_need_the_allowlist(hl):
    me = hl.client.account_address()
    assert F.withdraw_to_arbitrum(10)["status"] == "sent"  # own address
    assert hl.sent[-1]["action"]["destination"] == me
    blocked = F.send_usdc(OTHER, 1)
    assert blocked["status"] == "blocked"
    assert any("HL_WITHDRAW_ALLOWLIST" in r for r in blocked["reasons"])
    hl.configure(withdraw_allowlist=frozenset({OTHER}))
    assert F.send_usdc(OTHER, 1)["status"] == "sent"
    b58 = F.send_to_evm_with_data("USDC", 1, "So1anaAddr", 101, 1, address_encoding="base58")
    assert b58["status"] == "blocked"


def test_withdraw_module_is_separate_from_transfer(hl):
    hl.configure(write_modules=frozenset({"trade", "transfer"}),
                 withdraw_allowlist=frozenset({OTHER}))
    assert F.transfer_usdc_perp_spot(5, to="spot")["status"] == "sent"
    assert F.send_usdc(OTHER, 1)["status"] == "blocked"


def test_amount_units(hl):
    assert F.staking_deposit(1.25, dry_run=True)["action"]["wei"] == 125_000_000
    assert F.vault_transfer(OTHER, 100.5, "deposit", dry_run=True)["action"]["usd"] == 100_500_000
    with pytest.raises(ValueError, match="decimals"):
        F.staking_deposit(1.123456789)
    with pytest.raises(ValueError):
        F.send_usdc(OTHER, 0)
    assert F.lending_update("repay", "USDC", dry_run=True)["action"]["amount"] is None
    with pytest.raises(ValueError):
        F.lending_update("supply", "USDC")


# ---------------------------------------------------------------- admin

def test_agent_key_goes_to_a_private_file_never_to_the_caller(hl):
    out = AD.agent_approve(name="bot", valid_days=30)
    assert out["status"] == "sent" and out["ok"] is True
    assert TEST_KEY not in json.dumps(out)
    path = out["agent_key_file"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    rec = json.loads(open(path).read())
    assert rec["status"] == "approved" and rec["private_key"] not in json.dumps(out)
    assert rec["agent_address"] == hl.sent[-1]["action"]["agentAddress"]
    assert hl.sent[-1]["action"]["agentName"].startswith("bot valid_until ")


def test_agent_key_file_removed_when_not_approved(hl):
    hl.configure(write_modules=frozenset({"trade"}))
    out = AD.agent_approve(name="bot")
    assert out["status"] == "blocked"
    assert not (hl.tmp_path / "agents").exists() or not list((hl.tmp_path / "agents").iterdir())
    dry = AD.agent_approve(name="bot", dry_run=True)
    assert dry["status"] == "dry_run" and "agent_key_file" not in dry


def test_advanced_raw_action_defaults_to_dry_run(hl):
    out = AD.advanced_send_l1_action({"type": "gossipPriorityBid", "slotId": 0, "ip": "1.2.3.4",
                                      "maxGas": 1})
    assert out["status"] == "dry_run" and hl.sent == []
    with pytest.raises(ValueError, match="dedicated tool"):
        AD.advanced_send_l1_action({"type": "usdSend"}, dry_run=False)


# ---------------------------------------------------------------- reads

def test_ticker_and_mids(hl):
    hl.api.overrides["metaAndAssetCtxs"] = [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "OLD"}]},
        [{"markPx": "80000", "midPx": "80000.5", "oraclePx": "80010", "prevDayPx": "78000",
          "dayNtlVlm": "1e9", "funding": "0.00001", "openInterest": "100", "premium": "0.0001",
          "impactPxs": ["79999", "80001"]}, {}, {}]]
    t = M.market_get_ticker("BTC")
    assert t["funding_rate_1h"] == 1e-05 and t["funding_apr_pct"] == pytest.approx(8.76)
    assert t["change_24h_pct"] == pytest.approx(2.5641, rel=1e-4)
    assert t["open_interest_usd"] == 8_000_000
    m = M.market_get_mids(coins=["BTC", "xyz:TSLA", "HYPE/USDC", "#12090"])["mids"]
    assert m == {"BTC": 80000.5, "xyz:TSLA": 350.5, "HYPE/USDC": 40.5, "#12090": 0.3}


def test_positions_and_status(hl):
    hl.api.positions[""] = [position("BTC", "-0.5"), position("ETH", "0")]
    out = AC.account_get_positions()
    assert [(p["coin"], p["side"], p["size"]) for p in out["positions"]] == [("BTC", "short", 0.5)]
    st = AC.server_status()
    assert st["network"] == "testnet" and st["live_writes_possible"] is True
    assert TEST_KEY[2:] not in json.dumps(st)
