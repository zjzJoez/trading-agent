"""HL_* settings: safe defaults, loud failures, and the key never leaking."""
from __future__ import annotations

import pytest

from trading_agent.mcp_servers.hyperliquid.settings import SettingsError, load_settings

from .conftest import TEST_KEY


@pytest.fixture(autouse=True)
def _clean_env(clean_hl_env):
    pass


def test_defaults_are_the_safe_side_of_every_switch():
    s = load_settings()
    assert s.network == "testnet" and not s.is_mainnet
    assert s.base_url == "https://api.hyperliquid-testnet.xyz"
    assert s.private_key is None
    assert s.allow_mainnet_writes is False
    assert s.write_modules == frozenset({"trade"})
    assert s.withdraw_allowlist == frozenset()
    assert s.max_order_notional_usd == 1000.0 and s.max_leverage == 10
    assert s.default_slippage <= s.max_slippage


def test_mainnet_needs_its_own_flag(monkeypatch):
    monkeypatch.setenv("HL_NETWORK", "MAINNET")
    s = load_settings()
    assert s.is_mainnet and s.base_url == "https://api.hyperliquid.xyz"
    assert s.allow_mainnet_writes is False
    monkeypatch.setenv("HL_ALLOW_MAINNET_WRITES", "true")
    assert load_settings().allow_mainnet_writes is True


@pytest.mark.parametrize("var,value", [
    ("HL_NETWORK", "devnet"),
    ("HL_WRITE_MODULES", "trade,yolo"),
    ("HL_ACCOUNT_ADDRESS", "0x1234"),
    ("HL_WITHDRAW_ALLOWLIST", "0xnothex"),
    ("HL_MAX_LEVERAGE", "0"),
    ("HL_MAX_ORDER_NOTIONAL_USD", "-5"),
    ("HL_MAX_SLIPPAGE", "1.5"),
    ("HL_READ_ONLY", "maybe"),
])
def test_bad_values_fail_loudly(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(SettingsError):
        load_settings()


def test_default_slippage_cannot_exceed_max(monkeypatch):
    monkeypatch.setenv("HL_DEFAULT_SLIPPAGE", "0.1")
    monkeypatch.setenv("HL_MAX_SLIPPAGE", "0.05")
    with pytest.raises(SettingsError):
        load_settings()


def test_bad_key_error_does_not_echo_the_key(monkeypatch):
    almost = TEST_KEY[:-1] + "z"
    monkeypatch.setenv("HL_PRIVATE_KEY", almost)
    with pytest.raises(SettingsError) as ei:
        load_settings()
    assert almost[4:20] not in str(ei.value)


def test_key_is_normalised_and_hidden_from_repr(monkeypatch):
    monkeypatch.setenv("HL_PRIVATE_KEY", TEST_KEY[2:])
    s = load_settings()
    assert s.private_key == TEST_KEY
    assert TEST_KEY[2:] not in repr(s)


def test_modules_allowlists_and_addresses(monkeypatch):
    monkeypatch.setenv("HL_WRITE_MODULES", "Trade, withdraw")
    monkeypatch.setenv("HL_ALLOWED_COINS", "BTC, HYPE/USDC")
    monkeypatch.setenv("HL_WITHDRAW_ALLOWLIST", "0x" + "AB" * 20)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "CD" * 20)
    s = load_settings()
    assert s.write_modules == frozenset({"trade", "withdraw"})
    assert s.allowed_coins == frozenset({"btc", "hype/usdc"})
    assert s.withdraw_allowlist == frozenset({"0x" + "ab" * 20})
    assert s.account_address == "0x" + "cd" * 20


def test_modules_none_disables_every_write(monkeypatch):
    monkeypatch.setenv("HL_WRITE_MODULES", "none")
    assert load_settings().write_modules == frozenset()
