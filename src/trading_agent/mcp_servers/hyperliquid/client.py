"""Transport + signing for hyperliquid-mcp, on top of the official SDK.

The SDK supplies the HTTP layer (hyperliquid.api.API), both signing schemes
(hyperliquid.utils.signing.sign_l1_action / sign_user_signed_action) and the
error types. This module adds what the server needs around them: a
monotonic nonce, signer-role detection (master key vs API wallet), and one
place that turns an ActionRequest into a POST /exchange.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.api import API
from hyperliquid.utils.error import ClientError, ServerError
from hyperliquid.utils.signing import sign_l1_action, sign_user_signed_action

from trading_agent.mcp_servers.hyperliquid.actions import ActionRequest
from trading_agent.mcp_servers.hyperliquid.settings import Settings
from trading_agent.mcp_servers.hyperliquid.universe import Universe


class HyperliquidAPIError(RuntimeError):
    """An /info or /exchange request failed at the HTTP layer."""


def _describe(e: Exception) -> str:
    if isinstance(e, ClientError):
        msg = e.error_message if e.error_message is not None else ""
        return f"HTTP {e.status_code}: {msg}".strip()
    if isinstance(e, ServerError):
        return f"HTTP {e.status_code}: {e.message}"
    return repr(e)


class HLClient:
    def __init__(self, settings: Settings, api: Any | None = None):
        self.settings = settings
        self.api = api if api is not None else API(settings.base_url, settings.timeout)
        self.wallet: LocalAccount | None = (
            eth_account.Account.from_key(settings.private_key) if settings.private_key else None
        )
        self.universe = Universe(self.info)
        self._nonce_lock = threading.Lock()
        self._last_nonce = 0
        self._role: dict | None = None

    # ---- transport ----

    def info(self, payload: dict) -> Any:
        try:
            return self.api.post("/info", payload)
        except (ClientError, ServerError) as e:
            raise HyperliquidAPIError(f"/info {payload.get('type')} failed: {_describe(e)}") from e

    def next_nonce(self) -> int:
        """Millisecond timestamp, strictly increasing within this process.

        Hyperliquid rejects a reused nonce per signer; two actions in the same
        millisecond would otherwise collide."""
        with self._nonce_lock:
            n = max(int(time.time() * 1000), self._last_nonce + 1)
            self._last_nonce = n
            return n

    # ---- identity ----

    @property
    def signer_address(self) -> str | None:
        return self.wallet.address.lower() if self.wallet else None

    def signer_role(self) -> dict | None:
        """userRole of the signing key, cached. None when no key or lookup fails."""
        if self.wallet is None:
            return None
        if self._role is None:
            try:
                self._role = self.info({"type": "userRole", "user": self.signer_address})
            except HyperliquidAPIError:
                return None
        return self._role

    def signer_is_agent(self) -> bool:
        """True when the key is an API (agent) wallet rather than the account owner.

        Agents can sign L1 actions (orders, cancels, leverage...) but not
        user-signed ones (transfers, withdrawals, approvals)."""
        if self.wallet is None:
            return False
        acct = self.settings.account_address
        if acct and acct != self.signer_address:
            return True
        role = self.signer_role() or {}
        return role.get("role") == "agent"

    def account_address(self) -> str | None:
        """The account whose balances and positions the tools act on."""
        if self.settings.account_address:
            return self.settings.account_address
        if self.wallet is None:
            return None
        role = self.signer_role() or {}
        if role.get("role") == "agent":
            master = (role.get("data") or {}).get("user")
            if master:
                return master.lower()
        return self.signer_address

    # ---- signing + sending ----

    def sign(self, req: ActionRequest) -> dict:
        if self.wallet is None:
            raise RuntimeError("no signing key configured (HL_PRIVATE_KEY)")
        mainnet = self.settings.is_mainnet
        if req.scheme == "l1":
            return sign_l1_action(self.wallet, req.action, req.vault_address, req.nonce,
                                  None, mainnet)
        if req.scheme == "user":
            # Mutates req.action: adds signatureChainId + hyperliquidChain,
            # which are part of the posted action.
            sig = sign_user_signed_action(self.wallet, req.action, req.sign_types,
                                          req.primary_type, mainnet)
            for k in req.drop_after_sign:
                req.action.pop(k, None)
            return sig
        raise ValueError(f"unknown signing scheme {req.scheme!r}")

    def send(self, req: ActionRequest) -> Any:
        signature = self.sign(req)
        payload = {
            "action": req.action,
            "nonce": req.nonce,
            "signature": signature,
            "vaultAddress": req.vault_address if req.scheme == "l1" else None,
            "expiresAfter": None,
        }
        try:
            return self.api.post("/exchange", payload)
        except (ClientError, ServerError) as e:
            raise HyperliquidAPIError(f"/exchange {req.action_type} failed: {_describe(e)}") from e


def normalize_exchange_response(resp: Any) -> dict:
    """Flatten an /exchange response and surface per-order errors.

    The exchange answers {"status": "ok"} even when every order in the batch
    was rejected — the rejections live in response.data.statuses[i].error.
    Reporting that as success is the classic way trading bots lose money, so
    `ok` here is False if the top level OR any status carries an error.
    """
    out: dict[str, Any] = {"ok": False, "errors": [], "raw": resp}
    if not isinstance(resp, dict):
        out["errors"].append(f"unexpected response: {resp!r}")
        return out
    if resp.get("status") != "ok":
        out["errors"].append(str(resp.get("response", resp)))
        return out
    body = resp.get("response")
    data = body.get("data") if isinstance(body, dict) else None
    statuses: list = []
    if isinstance(data, dict):
        if isinstance(data.get("statuses"), list):
            statuses = data["statuses"]
        elif "status" in data:
            statuses = [data["status"]]
    for i, st in enumerate(statuses):
        if isinstance(st, dict) and "error" in st:
            out["errors"].append(f"[{i}] {st['error']}")
    out["statuses"] = statuses
    out["ok"] = not out["errors"]
    return out
