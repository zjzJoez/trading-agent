"""Hermetic fixtures for hyperliquid-mcp: a fake /info + /exchange transport.

No test in this package may reach api.hyperliquid.xyz: the fake API raises on
any request it does not model, and `hl` installs a client that uses it.
Live checks live in test_live.py behind @pytest.mark.integration.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

# eth-account docs test vector; never funded.
TEST_KEY = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
OTHER = "0x" + "22" * 20

PERP_DEXS = [None, {"name": "xyz", "fullName": "XYZ", "deployer": "0x" + "88" * 20,
                    "oracleUpdater": None, "feeRecipient": None}]
ALL_PERP_METAS = [
    {"universe": [
        {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
        {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        {"name": "OLD", "szDecimals": 1, "maxLeverage": 3, "isDelisted": True},
    ], "marginTables": [], "collateralToken": 0},
    {"universe": [
        {"name": "xyz:TSLA", "szDecimals": 3, "maxLeverage": 10, "onlyIsolated": True,
         "marginMode": "strictIsolated"},
        {"name": "xyz:PENNY", "szDecimals": 1, "maxLeverage": 5},
    ], "marginTables": [], "collateralToken": 4},
]


def _tok(name, idx, szd, wei, tid):
    return {"name": name, "szDecimals": szd, "weiDecimals": wei, "index": idx, "tokenId": tid,
            "isCanonical": True, "evmContract": None, "fullName": None}


SPOT_META = {
    "tokens": [
        _tok("USDC", 0, 8, 8, "0x6d1e7cde53ba9467b783cb7c530ce054"),
        _tok("PURR", 1, 0, 5, "0xc1fb593aeffbeb02f85e0308e9956a90"),
        _tok("HYPE", 2, 2, 8, "0x0d01dc56dcaaca66ad901c959b4011ec"),
        _tok("UBTC", 3, 5, 10, "0x8f254b963e8468305d409b33aa137c67"),
        _tok("USDH", 4, 2, 8, "0x54e00a5988577cb0b0c9ab0cb6ef7f4b"),
        _tok("FOO", 5, 1, 6, "0x" + "ab" * 16),
    ],
    "universe": [
        {"name": "PURR/USDC", "tokens": [1, 0], "index": 0, "isCanonical": True},
        {"name": "@1", "tokens": [2, 0], "index": 1, "isCanonical": False},
        {"name": "@2", "tokens": [3, 0], "index": 2, "isCanonical": False},
        {"name": "@3", "tokens": [5, 2], "index": 3, "isCanonical": False},  # FOO/HYPE
    ],
}
OUTCOME_META = {
    "outcomes": [{
        "outcome": 1209, "name": "Recurring",
        "description": "class:priceBinary|underlying:BTC|expiry:20261001-0600|targetPrice:80000",
        "sideSpecs": [{"name": "Yes"}, {"name": "No"}], "quoteToken": "USDC",
    }],
    "questions": [],
}
MIDS = {
    "": {"BTC": "80000.5", "ETH": "2500.25", "OLD": "1.0", "PURR/USDC": "0.2", "@1": "40.5",
         "@2": "80010", "@3": "0.5", "#12090": "0.3", "#12091": "0.7"},
    "xyz": {"xyz:TSLA": "350.5", "xyz:PENNY": "0.0123"},
}


class FakeAPI:
    """Stands in for hyperliquid.api.API. Configure state, then inspect calls."""

    def __init__(self):
        self.info_calls: list[dict] = []
        self.exchange_calls: list[dict] = []
        self.exchange_response: Any = {"status": "ok", "response": {"type": "default"}}
        self.leverage = 5
        self.role: dict = {"role": "user"}
        self.positions: dict[str, list[dict]] = {}   # dex -> assetPositions
        self.open_orders: dict[str, list[dict]] = {}  # dex -> frontendOpenOrders
        self.order_status: dict | None = None
        self.overrides: dict[str, Any] = {}
        self.all_perp_metas: Any = copy.deepcopy(ALL_PERP_METAS)

    def post(self, path: str, payload: dict) -> Any:
        if path == "/exchange":
            self.exchange_calls.append(copy.deepcopy(payload))
            if isinstance(self.exchange_response, Exception):
                raise self.exchange_response
            return self.exchange_response
        assert path == "/info", path
        self.info_calls.append(payload)
        t = payload["type"]
        if t in self.overrides:
            v = self.overrides[t]
            return v(payload) if callable(v) else copy.deepcopy(v)
        if t == "perpDexs":
            return copy.deepcopy(PERP_DEXS)
        if t == "allPerpMetas":
            return copy.deepcopy(self.all_perp_metas)
        if t == "meta":
            i = 0 if not payload.get("dex") else 1
            return copy.deepcopy(ALL_PERP_METAS[i])
        if t == "spotMeta":
            return copy.deepcopy(SPOT_META)
        if t == "outcomeMeta":
            return copy.deepcopy(OUTCOME_META)
        if t == "allMids":
            return dict(MIDS.get(payload.get("dex", ""), {}))
        if t == "userRole":
            return dict(self.role)
        if t == "activeAssetData":
            return {"user": payload["user"], "coin": payload["coin"],
                    "leverage": {"type": "cross", "value": self.leverage}}
        if t == "clearinghouseState":
            return {"assetPositions": copy.deepcopy(self.positions.get(payload.get("dex", ""), [])),
                    "marginSummary": {"accountValue": "1000", "totalNtlPos": "0",
                                      "totalMarginUsed": "0"},
                    "crossMarginSummary": {"accountValue": "1000"}, "withdrawable": "1000",
                    "time": 1790000000000}
        if t in ("frontendOpenOrders", "openOrders"):
            return copy.deepcopy(self.open_orders.get(payload.get("dex", ""), []))
        if t == "orderStatus":
            return copy.deepcopy(self.order_status) or {"status": "unknownOid"}
        raise AssertionError(f"FakeAPI: unmodelled /info request {payload}")


def position(coin: str, szi: str, value: str | None = None, margin: str = "0") -> dict:
    return {"type": "oneWay", "position": {"coin": coin, "szi": szi, "entryPx": "1",
                                           "positionValue": value, "marginUsed": margin,
                                           "leverage": {"type": "cross", "value": 5}}}


class HL:
    """Handle returned by the `hl` fixture."""

    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.api = FakeAPI()
        self.client = None

    def configure(self, **overrides):
        from trading_agent.mcp_servers.hyperliquid import core
        from trading_agent.mcp_servers.hyperliquid.client import HLClient
        from trading_agent.mcp_servers.hyperliquid.settings import Settings

        kw: dict[str, Any] = dict(
            network="testnet", private_key=TEST_KEY,
            write_modules=frozenset({"trade", "transfer", "withdraw", "admin", "advanced"}),
            audit_path=self.tmp_path / "audit.jsonl",
            agent_key_dir=self.tmp_path / "agents",
        )
        kw.update(overrides)
        self.client = HLClient(Settings(**kw), api=self.api)
        core.set_client(self.client)
        return self

    @property
    def sent(self) -> list[dict]:
        return self.api.exchange_calls

    def audit_lines(self) -> list[str]:
        p = self.tmp_path / "audit.jsonl"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture()
def clean_hl_env(monkeypatch):
    """Strip every HL_* variable (trading_agent.config loads the user's .env
    into os.environ at import) so load_settings() tests see only their own."""
    import os
    for var in [v for v in os.environ if v.startswith("HL_")]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def hl(monkeypatch, tmp_path):
    from trading_agent.mcp_servers.hyperliquid import (
        core,
        server,  # noqa: F401  (register tools)
    )

    h = HL(monkeypatch, tmp_path).configure()
    yield h
    core.set_client(None)
