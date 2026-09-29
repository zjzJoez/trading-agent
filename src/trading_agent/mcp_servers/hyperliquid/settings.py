"""Environment-driven settings for hyperliquid-mcp.

Every knob that decides whether money can move lives here, is read once at
startup, and is frozen. No tool accepts the network, the signing key, or any
of these limits as a parameter — the same invariant moomoo-mcp keeps for
trd_env — so a prompt can never talk the server onto mainnet or past a cap.

Defaults are the safe side of every switch:
  * HL_NETWORK defaults to testnet.
  * Mainnet writes need BOTH HL_NETWORK=mainnet AND HL_ALLOW_MAINNET_WRITES=true.
  * Only the `trade` write module is on by default; moving funds
    (`transfer`), sending them to another address (`withdraw`), account
    administration (`admin`) and raw signed actions (`advanced`) are opt-in.
  * `withdraw` additionally needs the destination in HL_WITHDRAW_ALLOWLIST.
  * Opening orders are capped by HL_MAX_ORDER_NOTIONAL_USD and HL_MAX_LEVERAGE.
Invalid values raise SettingsError at startup instead of silently falling back.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL

from trading_agent.config import CONFIG  # importing loads <repo>/.env

NETWORKS = ("testnet", "mainnet")
# Write modules, from least to most dangerous. Market data and account reads
# never sign anything and are always available.
WRITE_MODULES = ("trade", "transfer", "withdraw", "admin", "advanced")
DEFAULT_WRITE_MODULES = ("trade",)

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


class SettingsError(ValueError):
    """A HL_* environment variable is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Settings:
    network: str = "testnet"
    # repr=False keeps the key out of logs, tracebacks and tool output.
    private_key: str | None = field(default=None, repr=False)
    account_address: str | None = None
    vault_address: str | None = None
    read_only: bool = False
    allow_mainnet_writes: bool = False
    dry_run: bool = False
    write_modules: frozenset[str] = frozenset(DEFAULT_WRITE_MODULES)
    max_order_notional_usd: float = 1000.0
    max_leverage: int = 10
    default_slippage: float = 0.01
    max_slippage: float = 0.05
    allowed_coins: frozenset[str] | None = None
    withdraw_allowlist: frozenset[str] = frozenset()
    audit_path: Path = CONFIG.log_dir / "hyperliquid_audit.jsonl"
    agent_key_dir: Path = CONFIG.data_dir / "hyperliquid_agents"
    timeout: float = 15.0

    @property
    def is_mainnet(self) -> bool:
        return self.network == "mainnet"

    @property
    def base_url(self) -> str:
        return MAINNET_API_URL if self.is_mainnet else TESTNET_API_URL


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    lv = v.lower()
    if lv in ("1", "true", "yes", "on"):
        return True
    if lv in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{name}={v!r} is not a boolean (use true/false)")


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    v = _env(name)
    if v is None:
        return default
    try:
        f = float(v)
    except ValueError as e:
        raise SettingsError(f"{name}={v!r} is not a number") from e
    if f < minimum:
        raise SettingsError(f"{name}={v!r} must be >= {minimum}")
    return f


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    v = _env(name)
    if v is None:
        return default
    try:
        i = int(v)
    except ValueError as e:
        raise SettingsError(f"{name}={v!r} is not an integer") from e
    if i < minimum:
        raise SettingsError(f"{name}={v!r} must be >= {minimum}")
    return i


def normalize_address(value: str, what: str = "address") -> str:
    """Lowercased 0x-address; the docs recommend lowercase before signing."""
    v = value.strip()
    if not _ADDRESS_RE.match(v):
        raise SettingsError(f"{what} {value!r} is not a 42-character 0x address")
    return v.lower()


def _address(name: str) -> str | None:
    v = _env(name)
    return None if v is None else normalize_address(v, name)


def _csv(name: str) -> list[str]:
    v = _env(name)
    if v is None:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


def load_settings() -> Settings:
    network = (_env("HL_NETWORK") or "testnet").lower()
    if network not in NETWORKS:
        raise SettingsError(f"HL_NETWORK={network!r}; expected one of {NETWORKS}")

    key = _env("HL_PRIVATE_KEY")
    if key is not None and not _KEY_RE.match(key):
        # Never echo the value: a near-miss key is still mostly a secret.
        raise SettingsError("HL_PRIVATE_KEY is not a 32-byte hex private key")
    if key is not None and not key.startswith("0x"):
        key = "0x" + key

    modules_raw = [m.lower() for m in _csv("HL_WRITE_MODULES")]
    if not modules_raw:
        modules = frozenset(DEFAULT_WRITE_MODULES)
    elif modules_raw == ["none"]:
        modules = frozenset()
    else:
        unknown = sorted(set(modules_raw) - set(WRITE_MODULES))
        if unknown:
            raise SettingsError(
                f"HL_WRITE_MODULES has unknown module(s) {unknown}; "
                f"choose from {list(WRITE_MODULES)} or 'none'"
            )
        modules = frozenset(modules_raw)

    allowed = [c.lower() for c in _csv("HL_ALLOWED_COINS")]
    allowlist = frozenset(
        normalize_address(a, "HL_WITHDRAW_ALLOWLIST entry") for a in _csv("HL_WITHDRAW_ALLOWLIST")
    )

    default_slippage = _float("HL_DEFAULT_SLIPPAGE", 0.01)
    max_slippage = _float("HL_MAX_SLIPPAGE", 0.05)
    if max_slippage >= 1:
        raise SettingsError("HL_MAX_SLIPPAGE must be a fraction below 1 (0.05 = 5%)")
    if default_slippage > max_slippage:
        raise SettingsError("HL_DEFAULT_SLIPPAGE must not exceed HL_MAX_SLIPPAGE")

    audit = _env("HL_AUDIT_LOG")
    agent_dir = _env("HL_AGENT_KEY_DIR")
    return Settings(
        network=network,
        private_key=key,
        account_address=_address("HL_ACCOUNT_ADDRESS"),
        vault_address=_address("HL_VAULT_ADDRESS"),
        read_only=_bool("HL_READ_ONLY", False),
        allow_mainnet_writes=_bool("HL_ALLOW_MAINNET_WRITES", False),
        dry_run=_bool("HL_DRY_RUN", False),
        write_modules=modules,
        max_order_notional_usd=_float("HL_MAX_ORDER_NOTIONAL_USD", 1000.0),
        max_leverage=_int("HL_MAX_LEVERAGE", 10),
        default_slippage=default_slippage,
        max_slippage=max_slippage,
        allowed_coins=frozenset(allowed) if allowed else None,
        withdraw_allowlist=allowlist,
        audit_path=Path(audit) if audit else Settings.audit_path,
        agent_key_dir=Path(agent_dir) if agent_dir else Settings.agent_key_dir,
        timeout=_float("HL_TIMEOUT", 15.0, minimum=1.0),
    )
