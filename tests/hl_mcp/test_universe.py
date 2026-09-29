"""Instrument resolution (asset ids for every kind) and tick/lot rounding."""
from __future__ import annotations

import random
from decimal import Decimal

import pytest

from trading_agent.mcp_servers.hyperliquid.universe import (
    Instrument,
    Universe,
    UnknownInstrument,
    is_valid_price,
    price_quantum,
    round_price,
    round_size,
)

from .conftest import FakeAPI


@pytest.fixture()
def uni():
    api = FakeAPI()
    return Universe(lambda p: api.post("/info", p)), api


def test_asset_ids_for_every_instrument_kind(uni):
    u, _ = uni
    assert u.resolve("BTC").asset == 0
    assert u.resolve("ETH").asset == 1
    # HIP-3: 100000 + dex_index * 10000 + index_in_meta
    tsla = u.resolve("xyz:TSLA")
    assert (tsla.asset, tsla.dex, tsla.quote, tsla.only_isolated) == (110000, "xyz", "USDH", True)
    assert u.resolve("xyz:PENNY").asset == 110001
    # spot: 10000 + spotMeta.universe index; the API coin is "@index"
    hype = u.resolve("HYPE/USDC")
    assert (hype.asset, hype.coin, hype.sz_decimals) == (10001, "@1", 2)
    assert u.resolve("@1") is hype
    assert u.resolve("PURR/USDC").asset == 10000
    # outcomes: 100_000_000 + 10 * outcome + side
    yes, no = u.resolve("#12090"), u.resolve("#12091")
    assert (yes.asset, yes.outcome, yes.side, yes.kind) == (100_012_090, 1209, 0, "outcome")
    assert no.asset == 100_012_091 and "No" in no.name


def test_aliases_case_and_cex_suffixes(uni):
    u, _ = uni
    assert u.resolve("btc").coin == "BTC"
    assert u.resolve("XYZ:tsla").coin == "xyz:TSLA"
    assert u.resolve("hype/usdc").coin == "@1"
    # mainnet remaps BTC/USDC to UBTC/USDC on HyperCore
    assert u.resolve("BTC/USDC").name == "UBTC/USDC"
    for alias in ("BTC-PERP", "BTC-SWAP", "BTC-USDT-SWAP", "BTCPERP"):
        assert u.resolve(alias).coin == "BTC"


@pytest.mark.parametrize("name", ["BTC-USDC", "BTC-USDT", "HYPE-USD"])
def test_dash_pairs_are_ambiguous_not_guessed(uni, name):
    """On OKX / Coinbase "BTC-USDT" is SPOT; guessing the perp could flip exposure."""
    u, _ = uni
    with pytest.raises(UnknownInstrument, match="ambiguous"):
        u.resolve(name)


def test_unknown_names_suggest_matches(uni):
    u, _ = uni
    with pytest.raises(UnknownInstrument) as ei:
        u.resolve("TSL")
    assert "xyz:TSLA" in str(ei.value)
    with pytest.raises(UnknownInstrument):
        u.resolve("")


def test_misaligned_all_perp_metas_falls_back_to_per_dex_meta(uni):
    u, api = uni
    api.all_perp_metas = list(reversed(api.all_perp_metas))  # dex order scrambled
    assert u.resolve("xyz:TSLA").asset == 110000
    assert any(c.get("type") == "meta" for c in api.info_calls)


def test_search_and_tokens(uni):
    u, _ = uni
    assert [i.coin for i in u.search("tsla")] == ["xyz:TSLA"]
    assert all(i.kind == "outcome" for i in u.search("priceBinary", kinds=["outcome"]))
    assert "OLD" not in [i.coin for i in u.search("OLD")]
    assert "OLD" in [i.coin for i in u.search("OLD", include_delisted=True)]
    t = u.token("HYPE")
    assert u.token_wire(t) == "HYPE:0x0d01dc56dcaaca66ad901c959b4011ec"
    assert u.token(2) is t and u.token("HYPE:0xwhatever") is t


def _inst(kind: str, szd: int) -> Instrument:
    return Instrument(coin="X", name="X", kind=kind, asset=0, sz_decimals=szd)


@pytest.mark.parametrize("kind,szd,px,mode,expected", [
    ("perp", 5, "83826.47", "down", "83826"),      # BTC: 5 sig figs → integer
    ("perp", 5, "83826.47", "up", "83827"),
    ("perp", 0, "123456.7", "down", "123456"),     # integers always allowed
    ("perp", 1, "0.0123456", "nearest", "0.01235"),  # 5 decimals max (6 - 1)
    ("perp", 3, "400.12345", "up", "400.13"),
    ("spot", 2, "80.123456", "down", "80.123"),
    ("spot", 0, "0.000123456", "down", "0.00012345"),  # 8 decimals for spot
    ("perp", 4, "9.99995", "up", "10"),            # crossing a power of ten
])
def test_round_price(kind, szd, px, mode, expected):
    out = round_price(px, _inst(kind, szd), mode)
    assert out == Decimal(expected)
    assert is_valid_price(out, _inst(kind, szd))


def test_round_price_never_worsens_the_limit():
    inst = _inst("perp", 3)
    assert round_price("400.12345", inst, "down") <= Decimal("400.12345")
    assert round_price("400.12345", inst, "up") >= Decimal("400.12345")


def test_hip3_uses_perp_decimals_not_spot():
    """The SDK's _slippage_price treats asset ids >= 10000 as spot, which gives
    HIP-3 perps (ids >= 110000) up to 8 - szDecimals decimals. They are perps:
    6 - szDecimals."""
    hip3 = Instrument(coin="xyz:PENNY", name="xyz:PENNY", kind="perp", asset=110001,
                      sz_decimals=1, dex="xyz")
    assert hip3.price_max_decimals == 5
    assert not is_valid_price("0.012345", hip3)
    assert round_price("0.012345", hip3, "down") == Decimal("0.01234")


def test_nearest_rounding_agrees_with_the_sdk_tick_rule():
    """hyperliquid.exchange.Exchange._slippage_price rounds with
    round(float(f"{px:.5g}"), max_decimals). Its output must be a valid tick
    under our rule, and ours must be the nearest valid tick, at most one step
    from the SDK's (they differ only on double-rounding half-points and above
    1e5, where integers beat 5 significant figures)."""
    rng = random.Random(7)
    for _ in range(3000):
        kind = rng.choice(["perp", "spot"])
        szd = rng.randint(0, 5)
        inst = _inst(kind, szd)
        px = Decimal(f"{rng.uniform(0.0001, 200_000):.10f}")
        sdk = Decimal(repr(round(float(f"{float(px):.5g}"), inst.price_max_decimals)))
        if sdk <= 0:
            continue
        ours = round_price(px, inst, "nearest")
        step = price_quantum(ours, inst.price_max_decimals)
        sdk_step = Decimal(1).scaleb(px.adjusted() - 4)  # 5 significant figures only
        assert is_valid_price(sdk, inst), (kind, szd, px, sdk)
        assert abs(ours - px) <= price_quantum(px, inst.price_max_decimals) / 2, (px, ours)
        assert abs(ours - sdk) <= max(step, sdk_step), (kind, szd, px, sdk, ours)


def test_round_size_floors_to_lot():
    assert round_size("0.123456789", _inst("perp", 5)) == Decimal("0.12345")
    assert round_size("10.9", _inst("outcome", 0)) == Decimal("10")
    with pytest.raises(ValueError):
        round_size("0.000001", _inst("perp", 5))
    with pytest.raises(ValueError):
        round_size("-1", _inst("perp", 5))
