"""Instrument universe: names → asset ids, size decimals and tick rules.

Hyperliquid trades three instrument kinds through one order action; only the
asset id and the price/size rules differ:

  kind     coin (info API)   asset id                              example
  perp     "BTC"             index in meta                          BTC → 0
  perp     "xyz:TSLA"        100000 + dex_index*10000 + index       (HIP-3 dex)
  spot     "@107"/"PURR/USDC" 10000 + spotMeta.universe index       HYPE/USDC → 10107
  outcome  "#12090"          100_000_000 + 10*outcome + side        (HIP-4)

The SDK's Info only maps the first perp dex and spot, so this module builds
the full map itself: one `allPerpMetas` call covers every HIP-3 dex, plus
`spotMeta` and `outcomeMeta`. It also fixes the SDK's tick rule for HIP-3
perps (Exchange._slippage_price treats any asset id >= 10000 as spot, which
gives HIP-3 perps two decimals too many).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal
from typing import Any

PERP_MAX_DECIMALS = 6
SPOT_MAX_DECIMALS = 8
SPOT_ASSET_OFFSET = 10_000
HIP3_ASSET_OFFSET = 100_000
HIP3_DEX_STRIDE = 10_000
OUTCOME_ASSET_OFFSET = 100_000_000
# Observed on mainnet 2026-09: every outcome fill and book level has a whole
# share size, and outcomes are absent from szDecimals metadata.
OUTCOME_SZ_DECIMALS = 0
# Quote/collateral tokens valued at 1 USD for the notional cap.
USD_STABLES = frozenset({"USDC", "USDT", "USDT0", "USDH", "USDE"})

UNIVERSE_TTL_S = 300.0
# A lookup miss triggers a reload (new listings), at most this often, so a
# stream of bad names cannot turn into a stream of 4-request reloads.
MISS_RELOAD_MIN_INTERVAL_S = 30.0


class UnknownInstrument(LookupError):
    pass


@dataclass(frozen=True)
class Instrument:
    coin: str            # name the info API and fills use
    name: str            # human name ("HYPE/USDC" for spot "@107")
    kind: str            # "perp" | "spot" | "outcome"
    asset: int           # asset id for exchange actions
    sz_decimals: int
    dex: str = ""        # perp dex ("" = first dex)
    quote: str = "USDC"  # quote token (spot/outcome) or collateral (perp)
    max_leverage: int | None = None
    only_isolated: bool = False
    margin_mode: str | None = None
    delisted: bool = False
    outcome: int | None = None
    side: int | None = None
    description: str | None = None

    @property
    def is_perp(self) -> bool:
        return self.kind == "perp"

    @property
    def price_max_decimals(self) -> int:
        base = PERP_MAX_DECIMALS if self.is_perp else SPOT_MAX_DECIMALS
        return max(base - self.sz_decimals, 0)

    def summary(self) -> dict:
        d = {
            "coin": self.coin, "name": self.name, "kind": self.kind,
            "asset_id": self.asset, "sz_decimals": self.sz_decimals,
            "price_max_decimals": self.price_max_decimals, "quote": self.quote,
        }
        if self.kind == "perp":
            d.update(dex=self.dex, max_leverage=self.max_leverage,
                     only_isolated=self.only_isolated, margin_mode=self.margin_mode)
        if self.delisted:
            d["delisted"] = True
        if self.kind == "outcome":
            d.update(outcome=self.outcome, side=self.side, description=self.description)
        return d


# --------------------------------------------------------------------------
# Tick / lot rules
# --------------------------------------------------------------------------

def to_decimal(x: Any, what: str) -> Decimal:
    try:
        d = Decimal(str(x))
    except Exception as e:
        raise ValueError(f"{what} {x!r} is not a number") from e
    if not d.is_finite():
        raise ValueError(f"{what} {x!r} is not finite")
    return d


def price_quantum(px: Decimal, max_decimals: int) -> Decimal:
    """Smallest valid price step at `px`: 5 significant figures, at most
    `max_decimals` decimals, and integers are always allowed."""
    sig = Decimal(1).scaleb(px.adjusted() - 4)
    step = min(sig, Decimal(1))
    return max(step, Decimal(1).scaleb(-max_decimals))


def round_price(px: Any, inst: Instrument, mode: str = "nearest") -> Decimal:
    """Round to a valid tick. `down` for buys and `up` for sells never make a
    limit price worse for the trader than what was asked."""
    d = to_decimal(px, "price")
    if d <= 0:
        raise ValueError(f"price must be positive, got {px!r}")
    rounding = {"down": ROUND_DOWN, "up": ROUND_UP, "nearest": ROUND_HALF_EVEN}[mode]
    q = price_quantum(d, inst.price_max_decimals)
    out = (d / q).to_integral_value(rounding=rounding) * q
    if out <= 0:
        raise ValueError(f"price {px!r} rounds to zero for {inst.coin}")
    # Rounding up can cross a power of ten, which coarsens the quantum; one
    # more pass lands on the new grid.
    q2 = price_quantum(out, inst.price_max_decimals)
    if q2 != q:
        out = (out / q2).to_integral_value(rounding=rounding) * q2
    return out.normalize() if out != out.to_integral_value() else out.quantize(Decimal(1))


def is_valid_price(px: Any, inst: Instrument) -> bool:
    d = to_decimal(px, "price")
    if d <= 0:
        return False
    q = price_quantum(d, inst.price_max_decimals)
    return (d / q) == (d / q).to_integral_value()


def round_size(sz: Any, inst: Instrument) -> Decimal:
    """Floor to the lot size: never trade more than asked."""
    d = to_decimal(sz, "size")
    if d <= 0:
        raise ValueError(f"size must be positive, got {sz!r}")
    out = d.quantize(Decimal(1).scaleb(-inst.sz_decimals), rounding=ROUND_DOWN)
    if out <= 0:
        raise ValueError(
            f"size {sz!r} is below the lot size of {inst.coin} "
            f"({inst.sz_decimals} decimals)"
        )
    return out


def fmt_decimal(d: Decimal) -> str:
    """Wire format: no exponent, no trailing zeros (see the Signing docs)."""
    s = format(d.normalize(), "f")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

_PERP_SUFFIXES = ("-USDT-SWAP", "-USD-SWAP", "-SWAP", "-PERP", "PERP", "-USDC", "-USDT", "-USD")


class Universe:
    """Lazily loaded, TTL-cached map of every tradable instrument."""

    def __init__(self, info: Callable[[dict], Any], ttl: float = UNIVERSE_TTL_S):
        self._info = info
        self._ttl = ttl
        self._loaded_at = 0.0
        self._by_key: dict[str, Instrument] = {}
        self._by_asset: dict[int, Instrument] = {}
        self._instruments: list[Instrument] = []
        self._tokens_by_index: dict[int, dict] = {}
        self._tokens_by_name: dict[str, dict] = {}
        self._dexs: list[dict] = []
        self._questions: dict[int, dict] = {}

    # ---- loading ----

    def _stale(self) -> bool:
        return not self._instruments or (time.monotonic() - self._loaded_at) > self._ttl

    def ensure(self) -> None:
        if self._stale():
            self.reload()

    def reload(self) -> None:
        spot_meta = self._info({"type": "spotMeta"})
        dexs = self._info({"type": "perpDexs"}) or [None]
        perp_metas = self._load_perp_metas(dexs)
        try:
            outcome_meta = self._info({"type": "outcomeMeta"})
        except Exception:
            outcome_meta = None  # older networks: no HIP-4

        tokens_by_index = {t["index"]: t for t in spot_meta.get("tokens", [])}
        instruments: list[Instrument] = []

        for i, meta in enumerate(perp_metas):
            dex = "" if i == 0 else (dexs[i] or {}).get("name", "")
            offset = 0 if i == 0 else HIP3_ASSET_OFFSET + i * HIP3_DEX_STRIDE
            collateral = tokens_by_index.get(meta.get("collateralToken", 0), {}).get("name", "USDC")
            for idx, a in enumerate(meta.get("universe", [])):
                instruments.append(Instrument(
                    coin=a["name"], name=a["name"], kind="perp", asset=offset + idx,
                    sz_decimals=int(a["szDecimals"]), dex=dex, quote=collateral,
                    max_leverage=a.get("maxLeverage"),
                    only_isolated=bool(a.get("onlyIsolated", False)),
                    margin_mode=a.get("marginMode"),
                    delisted=bool(a.get("isDelisted", False)),
                ))

        for pair in spot_meta.get("universe", []):
            base_i, quote_i = pair["tokens"]
            base, quote = tokens_by_index.get(base_i), tokens_by_index.get(quote_i)
            if base is None or quote is None:
                continue
            instruments.append(Instrument(
                coin=pair["name"], name=f"{base['name']}/{quote['name']}", kind="spot",
                asset=SPOT_ASSET_OFFSET + pair["index"], sz_decimals=int(base["szDecimals"]),
                quote=quote["name"],
            ))

        questions: dict[int, dict] = {}
        if outcome_meta:
            for q in outcome_meta.get("questions", []) or []:
                for oid in q.get("namedOutcomes", []) or []:
                    questions[oid] = q
            for o in outcome_meta.get("outcomes", []) or []:
                for side, spec in enumerate((o.get("sideSpecs") or [])[:2]):
                    enc = 10 * int(o["outcome"]) + side
                    instruments.append(Instrument(
                        coin=f"#{enc}", name=f"{o.get('name', '')} [{spec.get('name', side)}]",
                        kind="outcome", asset=OUTCOME_ASSET_OFFSET + enc,
                        sz_decimals=OUTCOME_SZ_DECIMALS, quote=o.get("quoteToken", "USDC"),
                        outcome=int(o["outcome"]), side=side, description=o.get("description"),
                    ))

        by_key: dict[str, Instrument] = {}
        for inst in instruments:
            # Wire coin first; human names never shadow a wire coin.
            by_key.setdefault(inst.coin.lower(), inst)
        for inst in instruments:
            if inst.kind == "spot":
                by_key.setdefault(inst.name.lower(), inst)

        self._instruments = instruments
        self._by_key = by_key
        self._by_asset = {i.asset: i for i in instruments}
        self._tokens_by_index = tokens_by_index
        self._tokens_by_name = {}
        for t in spot_meta.get("tokens", []):
            self._tokens_by_name.setdefault(t["name"].lower(), t)
        self._dexs = dexs
        self._questions = questions
        self._loaded_at = time.monotonic()

    def _load_perp_metas(self, dexs: list) -> list[dict]:
        names = ["" if i == 0 else (d or {}).get("name", "") for i, d in enumerate(dexs)]
        try:
            metas = self._info({"type": "allPerpMetas"})
            if isinstance(metas, list) and len(metas) == len(names) and self._aligned(metas, names):
                return metas
        except Exception:
            pass
        # Fallback: one meta call per dex (always correct, just slower).
        return [self._info({"type": "meta", "dex": n}) for n in names]

    @staticmethod
    def _aligned(metas: list, names: list[str]) -> bool:
        for meta, name in zip(metas, names, strict=True):
            if not isinstance(meta, dict) or "universe" not in meta:
                return False
            if name and any(not a["name"].startswith(f"{name}:") for a in meta["universe"]):
                return False
        return True

    # ---- lookups ----

    def resolve(self, query: str) -> Instrument:
        """Resolve a user-facing name to an instrument.

        Accepts wire coins ("BTC", "xyz:TSLA", "@107", "#12090"), spot pairs
        ("HYPE/USDC", case-insensitive), the mainnet "U"-prefixed spot remaps
        ("BTC/USDC" → "UBTC/USDC"), and common CEX suffixes ("BTC-PERP").
        """
        if not isinstance(query, str) or not query.strip():
            raise UnknownInstrument("coin must be a non-empty string")
        just_loaded = self._stale()
        if just_loaded:
            self.reload()
        hit = self._lookup(query)
        age = time.monotonic() - self._loaded_at
        if hit is None and not just_loaded and age > MISS_RELOAD_MIN_INTERVAL_S:
            # New listings appear between reloads; retry once on fresh data.
            self.reload()
            hit = self._lookup(query)
        if hit is None:
            raise UnknownInstrument(self._not_found(query))
        return hit

    def _lookup(self, query: str) -> Instrument | None:
        q = query.strip()
        hit = self._by_key.get(q.lower())
        if hit:
            return hit
        if "/" in q:
            base, _, quote = q.partition("/")
            return self._by_key.get(f"u{base}/{quote}".lower())
        up = q.upper()
        for suffix in _PERP_SUFFIXES:
            if up.endswith(suffix) and len(up) > len(suffix):
                hit = self._by_key.get(q[: -len(suffix)].lower())
                if hit and hit.kind == "perp":
                    return hit
        return None

    def _not_found(self, query: str) -> str:
        near = [i.coin for i in self.search(query, limit=5)]
        hint = f" Close matches: {near}." if near else ""
        return (
            f"Unknown instrument {query!r}.{hint} Perps are 'BTC' or 'dex:COIN' (HIP-3), "
            "spot is 'BASE/QUOTE' or '@index', outcomes are '#encoding'. "
            "Use market_search to look names up."
        )

    def by_asset(self, asset: int) -> Instrument | None:
        self.ensure()
        return self._by_asset.get(asset)

    def search(self, query: str, kinds: list[str] | None = None, limit: int = 20,
               include_delisted: bool = False) -> list[Instrument]:
        self.ensure()
        q = query.strip().lower()
        exact, prefix, contains = [], [], []
        for inst in self._instruments:
            if kinds and inst.kind not in kinds:
                continue
            if inst.delisted and not include_delisted:
                continue
            hay = [inst.coin.lower(), inst.name.lower()]
            if inst.description:
                hay.append(inst.description.lower())
            if q in (inst.coin.lower(), inst.name.lower()):
                exact.append(inst)
            elif any(h.startswith(q) or h.split(":")[-1].startswith(q) for h in hay):
                prefix.append(inst)
            elif any(q in h for h in hay):
                contains.append(inst)
        return (exact + prefix + contains)[: max(limit, 0)]

    def instruments(self, kind: str | None = None, dex: str | None = None) -> list[Instrument]:
        self.ensure()
        return [i for i in self._instruments
                if (kind is None or i.kind == kind) and (dex is None or i.dex == dex)]

    def dex_names(self) -> list[str]:
        self.ensure()
        return ["" if i == 0 else (d or {}).get("name", "") for i, d in enumerate(self._dexs)]

    def dexs(self) -> list[dict | None]:
        self.ensure()
        return list(self._dexs)

    def question_for(self, outcome: int) -> dict | None:
        self.ensure()
        return self._questions.get(outcome)

    def token(self, ref: str | int) -> dict:
        """Spot token record by name ("HYPE"), index (150) or 'NAME:0xid'."""
        self.ensure()
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
            t = self._tokens_by_index.get(int(ref))
        else:
            name = ref.strip().split(":")[0].lower()
            t = self._tokens_by_name.get(name)
        if t is None:
            raise UnknownInstrument(f"Unknown spot token {ref!r}")
        return t

    @staticmethod
    def token_wire(t: dict) -> str:
        """spotSend / sendAsset token string: NAME:tokenId."""
        return f"{t['name']}:{t['tokenId']}"
