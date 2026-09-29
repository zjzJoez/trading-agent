"""Fund movement tools.

Write module `transfer` (off by default): moves between the account's OWN
balances — perp ↔ spot, between perp dexs, to/from its sub-accounts and
vaults, staking, borrow/lend, outcome split/merge.

Write module `withdraw` (off by default): funds leaving the account — to
another Hyperliquid address, to Arbitrum via the bridge, or to another chain.
Every destination must be the account itself or listed in
HL_WITHDRAW_ALLOWLIST; there is no per-call override.

Most of these are user-signed (EIP-712) actions: only the account owner's key
can sign them, an API wallet cannot.
Deposits into a vault or a sub-account hand the funds to whoever controls it,
so they are checked too: a vault must be led by this account or be in
HL_WITHDRAW_ALLOWLIST, and a sub-account must belong to this account.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from trading_agent.mcp_servers.hyperliquid import actions as A
from trading_agent.mcp_servers.hyperliquid.core import (
    WRITE,
    address,
    client,
    destination_reasons,
    execute,
    mcp,
    own_addresses,
)
from trading_agent.mcp_servers.hyperliquid.universe import fmt_decimal, to_decimal

USD_DECIMALS = 6
HYPE_WEI_DECIMALS = 8


def _amount(amount: float | str, what: str, max_decimals: int) -> Decimal:
    d = to_decimal(amount, what)
    if d <= 0:
        raise ValueError(f"{what} must be positive")
    if d != d.quantize(Decimal(1).scaleb(-max_decimals)):
        raise ValueError(f"{what} {amount} has more than {max_decimals} decimals")
    return d


def _usd(amount: float | str, what: str = "amount_usd") -> str:
    return fmt_decimal(_amount(amount, what, USD_DECIMALS))


def _micro_usd(amount: float | str, what: str = "amount_usd") -> int:
    return int(_amount(amount, what, USD_DECIMALS).scaleb(USD_DECIMALS))


def _wei(amount: float | str, what: str, decimals: int = HYPE_WEI_DECIMALS) -> int:
    return int(_amount(amount, what, decimals).scaleb(decimals))


def _token(token: str) -> tuple[dict, str]:
    try:
        t = client().universe.token(token)
    except LookupError as e:
        raise ValueError(str(e)) from e
    return t, client().universe.token_wire(t)


def _dex(name: str) -> str:
    n = (name or "").strip()
    if n == "spot":
        return n
    if n not in client().universe.dex_names():
        raise ValueError(f"unknown perp dex {name!r}; use '' (main perps), 'spot' or a name "
                         "from market_get_perp_dexs")
    return n


def _dest_reasons(destination: str) -> list[str]:
    return destination_reasons(destination)


def _sub_account_reasons(sub: str) -> list[str]:
    """The exchange should refuse a transfer to someone else's sub-account, but
    this checks it first and fails closed if it cannot."""
    me = client().account_address()
    if not me:
        return ["cannot verify the sub-account: no account address configured"]
    try:
        subs = client().info({"type": "subAccounts", "user": me}) or []
    except Exception as e:
        return [f"cannot verify {sub} is your sub-account ({e}); refusing fail-closed"]
    if sub.lower() not in {str(s.get("subAccountUser", "")).lower() for s in subs}:
        return [f"{sub} is not a sub-account of {me}"]
    return []


def _vault_deposit_reasons(vault: str) -> list[str]:
    """A vault's leader trades its deposits, so depositing into someone else's
    vault hands them the funds: allow only your own vaults and allowlisted ones."""
    if vault in own_addresses() or vault in client().settings.withdraw_allowlist:
        return []
    try:
        details = client().info({"type": "vaultDetails", "vaultAddress": vault}) or {}
    except Exception as e:
        return [f"cannot verify vault {vault} ({e}); refusing fail-closed"]
    leader = str(details.get("leader", "")).lower()
    if leader and leader in own_addresses():
        return []
    return [f"vault {vault} is led by {leader or 'unknown'}, not this account; add it to "
            "HL_WITHDRAW_ALLOWLIST to allow deposits"]


# --------------------------------------------------------------------------
# transfer: the account's own funds
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def transfer_usdc_perp_spot(amount_usd: float, to: Literal["perp", "spot"],
                            sub_account: str | None = None, dry_run: bool = False) -> dict:
    """Move USDC between the perp (main dex) and spot balances of the account,
    or of one of its sub-accounts."""
    sub = address(sub_account, "sub_account") if sub_account else None
    req = A.usd_class_transfer(_usd(amount_usd), to == "perp", client().next_nonce(),
                               sub_account=sub)
    return execute("transfer_usdc_perp_spot", "transfer", req, dry_run=dry_run,
                   needs_owner_key=True,
                   summary={"amount_usd": amount_usd, "to": to, "sub_account": sub})


@mcp.tool(annotations=WRITE)
def transfer_between_dexs(amount: float, source_dex: str, destination_dex: str,
                          token: str = "USDC", dry_run: bool = False) -> dict:
    """Move collateral between the account's own balances: main perps (""), "spot",
    or a HIP-3 dex (e.g. "xyz"). Only a dex's collateral token can enter or
    leave it. Works with an API wallet too (as agentSendAsset)."""
    src, dst = _dex(source_dex), _dex(destination_dex)
    if src == dst:
        raise ValueError("source_dex and destination_dex are the same")
    t, wire = _token(token)
    amt = fmt_decimal(_amount(amount, "amount", int(t["weiDecimals"])))
    c = client()
    me = c.account_address()
    if not me:
        raise ValueError("no account configured (HL_ACCOUNT_ADDRESS / HL_PRIVATE_KEY)")
    agent = c.wallet is not None and c.signer_is_agent()
    build = A.agent_send_asset if agent else A.send_asset
    req = build(me, src, dst, wire, amt, c.next_nonce())
    return execute("transfer_between_dexs", "transfer", req, dry_run=dry_run,
                   needs_owner_key=not agent,
                   summary={"token": wire, "amount": amt, "from": src or "main perps",
                            "to": dst or "main perps"})


@mcp.tool(annotations=WRITE)
def transfer_sub_account_usdc(sub_account: str, amount_usd: float,
                              direction: Literal["deposit", "withdraw"],
                              dry_run: bool = False) -> dict:
    """Move perp USDC between the master account and one of its sub-accounts."""
    sub = address(sub_account, "sub_account")
    req = A.sub_account_transfer(sub, direction == "deposit", _micro_usd(amount_usd),
                                 client().next_nonce())
    return execute("transfer_sub_account_usdc", "transfer", req, dry_run=dry_run,
                   reasons=_sub_account_reasons(sub),
                   summary={"sub_account": sub, "amount_usd": amount_usd, "direction": direction})


@mcp.tool(annotations=WRITE)
def transfer_sub_account_spot(sub_account: str, token: str, amount: float,
                              direction: Literal["deposit", "withdraw"],
                              dry_run: bool = False) -> dict:
    """Move a spot token between the master account and one of its sub-accounts."""
    sub = address(sub_account, "sub_account")
    t, wire = _token(token)
    amt = fmt_decimal(_amount(amount, "amount", int(t["weiDecimals"])))
    req = A.sub_account_spot_transfer(sub, direction == "deposit", wire, amt,
                                      client().next_nonce())
    return execute("transfer_sub_account_spot", "transfer", req, dry_run=dry_run,
                   reasons=_sub_account_reasons(sub),
                   summary={"sub_account": sub, "token": wire, "amount": amt,
                            "direction": direction})


@mcp.tool(annotations=WRITE)
def vault_transfer(vault_address: str, amount_usd: float,
                   direction: Literal["deposit", "withdraw"], dry_run: bool = False) -> dict:
    """Deposit USDC into, or withdraw from, a vault (check lock-ups with
    market_get_vault_details first). Deposits are allowed only into vaults this
    account leads or that are in HL_WITHDRAW_ALLOWLIST: a vault's leader
    controls the funds."""
    vault = address(vault_address, "vault_address")
    req = A.vault_transfer(vault, direction == "deposit", _micro_usd(amount_usd),
                           client().next_nonce())
    reasons = _vault_deposit_reasons(vault) if direction == "deposit" else []
    return execute("vault_transfer", "transfer", req, dry_run=dry_run, reasons=reasons,
                   summary={"vault": vault, "amount_usd": amount_usd, "direction": direction})


@mcp.tool(annotations=WRITE)
def staking_deposit(amount_hype: float, dry_run: bool = False) -> dict:
    """Move HYPE from the spot balance into the staking balance (then delegate it)."""
    req = A.c_deposit(_wei(amount_hype, "amount_hype"), client().next_nonce())
    return execute("staking_deposit", "transfer", req, dry_run=dry_run, needs_owner_key=True,
                   summary={"amount_hype": amount_hype})


@mcp.tool(annotations=WRITE)
def staking_withdraw(amount_hype: float, dry_run: bool = False) -> dict:
    """Move undelegated HYPE from staking back to spot. Goes through a 7-day queue."""
    req = A.c_withdraw(_wei(amount_hype, "amount_hype"), client().next_nonce())
    return execute("staking_withdraw", "transfer", req, dry_run=dry_run, needs_owner_key=True,
                   summary={"amount_hype": amount_hype})


@mcp.tool(annotations=WRITE)
def staking_delegate(validator: str, amount_hype: float, undelegate: bool = False,
                     dry_run: bool = False) -> dict:
    """Delegate (or undelegate) staked HYPE to a validator (see market_get_validators).
    Delegations are locked for 1 day."""
    v = address(validator, "validator")
    req = A.token_delegate(v, _wei(amount_hype, "amount_hype"), undelegate,
                           client().next_nonce())
    return execute("staking_delegate", "transfer", req, dry_run=dry_run, needs_owner_key=True,
                   summary={"validator": v, "amount_hype": amount_hype, "undelegate": undelegate})


@mcp.tool(annotations=WRITE)
def staking_claim_rewards(dry_run: bool = False) -> dict:
    """Claim accrued rewards."""
    return execute("staking_claim_rewards", "transfer", A.claim_rewards(client().next_nonce()),
                   dry_run=dry_run)


@mcp.tool(annotations=WRITE)
def lending_update(operation: Literal["supply", "withdraw", "borrow", "repay"], token: str,
                   amount: float | None = None, dry_run: bool = False) -> dict:
    """HyperCore borrow/lend: supply or withdraw a token, borrow or repay USDC/USDT
    against supplied HYPE/BTC collateral. `amount=None` means the full balance
    (withdraw / repay only). Check health with account_get_borrow_lend_state."""
    t, _wire = _token(token)
    if amount is None and operation in ("supply", "borrow"):
        raise ValueError(f"{operation} needs an amount")
    amt = None if amount is None else fmt_decimal(_amount(amount, "amount",
                                                          int(t["weiDecimals"])))
    req = A.borrow_lend(operation, int(t["index"]), amt, client().next_nonce())
    return execute("lending_update", "transfer", req, dry_run=dry_run,
                   summary={"operation": operation, "token": t["name"], "amount": amt or "max"})


@mcp.tool(annotations=WRITE)
def outcome_convert(operation: Literal["split", "merge", "merge_question", "negate"],
                    outcome: int | None = None, question: int | None = None,
                    amount: float | None = None, dry_run: bool = False) -> dict:
    """Convert between quote tokens and outcome shares without trading:
    split (X quote → X Yes + X No), merge (X Yes + X No → X quote; amount None = max),
    merge_question (X Yes of every outcome in a question → X quote),
    negate (X No of one outcome → X Yes of each other outcome in its question)."""
    needs = {"split": ("outcome",), "merge": ("outcome",), "merge_question": ("question",),
             "negate": ("question", "outcome")}[operation]
    given = {"outcome": outcome, "question": question}
    missing = [k for k in needs if given[k] is None]
    if missing:
        raise ValueError(f"{operation} needs {missing}")
    if amount is None and operation in ("split", "negate"):
        raise ValueError(f"{operation} needs an amount")
    amt = None if amount is None else fmt_decimal(_amount(amount, "amount", USD_DECIMALS))
    req = A.user_outcome(operation, client().next_nonce(), outcome=outcome, question=question,
                         amount=amt)
    return execute("outcome_convert", "transfer", req, dry_run=dry_run,
                   summary={"operation": operation, "outcome": outcome, "question": question,
                            "amount": amt or "max"})


# --------------------------------------------------------------------------
# withdraw: funds leaving the account
# --------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def withdraw_to_arbitrum(amount_usdc: float, destination: str | None = None,
                         dry_run: bool = False) -> dict:
    """Withdraw USDC to Arbitrum through the bridge (about 5 minutes, $1 fee).
    `destination` defaults to the account's own address."""
    me = client().account_address()
    dest = address(destination, "destination") if destination else me
    if not dest:
        raise ValueError("no destination and no account configured")
    req = A.withdraw3(dest, _usd(amount_usdc, "amount_usdc"), client().next_nonce())
    return execute("withdraw_to_arbitrum", "withdraw", req, dry_run=dry_run, needs_owner_key=True,
                   reasons=_dest_reasons(dest),
                   summary={"destination": dest, "amount_usdc": amount_usdc})


@mcp.tool(annotations=WRITE)
def send_usdc(destination: str, amount: float, dry_run: bool = False) -> dict:
    """Send perp USDC to another Hyperliquid address (no bridge; new accounts pay a
    1 USDC activation fee)."""
    dest = address(destination, "destination")
    req = A.usd_send(dest, _usd(amount, "amount"), client().next_nonce())
    return execute("send_usdc", "withdraw", req, dry_run=dry_run, needs_owner_key=True,
                   reasons=_dest_reasons(dest), summary={"destination": dest, "amount": amount})


@mcp.tool(annotations=WRITE)
def send_spot_token(destination: str, token: str, amount: float, dry_run: bool = False) -> dict:
    """Send a spot token (e.g. "HYPE", "PURR") to another Hyperliquid address."""
    dest = address(destination, "destination")
    t, wire = _token(token)
    amt = fmt_decimal(_amount(amount, "amount", int(t["weiDecimals"])))
    req = A.spot_send(dest, wire, amt, client().next_nonce())
    return execute("send_spot_token", "withdraw", req, dry_run=dry_run, needs_owner_key=True,
                   reasons=_dest_reasons(dest),
                   summary={"destination": dest, "token": wire, "amount": amt})


@mcp.tool(annotations=WRITE)
def send_asset(destination: str, token: str, amount: float, source_dex: str = "",
               destination_dex: str = "", dry_run: bool = False) -> dict:
    """Generalised transfer to another address, between any of: main perps (""),
    "spot" and HIP-3 dexs. For moves within your own account use
    transfer_between_dexs."""
    dest = address(destination, "destination")
    src, dst = _dex(source_dex), _dex(destination_dex)
    t, wire = _token(token)
    amt = fmt_decimal(_amount(amount, "amount", int(t["weiDecimals"])))
    req = A.send_asset(dest, src, dst, wire, amt, client().next_nonce())
    return execute("send_asset", "withdraw", req, dry_run=dry_run, needs_owner_key=True,
                   reasons=_dest_reasons(dest),
                   summary={"destination": dest, "token": wire, "amount": amt,
                            "from": src, "to": dst})


@mcp.tool(annotations=WRITE)
def send_to_evm_with_data(token: str, amount: float, destination_recipient: str,
                          destination_chain_id: int, gas_limit: int, data: str = "0x",
                          address_encoding: Literal["hex", "base58"] = "hex",
                          source_dex: str = "", dry_run: bool = False) -> dict:
    """Core → EVM / cross-chain transfer that calls coreReceiveWithData on the token's
    linked contract. The token must be linked and the contract must implement
    the interface; get either wrong and funds can be stuck. The recipient must
    be a 0x address in HL_WITHDRAW_ALLOWLIST."""
    if not data.startswith(("0x", "0X")):
        raise ValueError("data must be 0x-prefixed hex")
    t, wire = _token(token)
    amt = fmt_decimal(_amount(amount, "amount", int(t["weiDecimals"])))
    src = _dex(source_dex)
    recipient = destination_recipient.strip()
    reasons = (_dest_reasons(recipient) if address_encoding == "hex"
               else ["base58 recipients cannot be allowlisted; refusing"])
    req = A.send_to_evm_with_data(wire, amt, src, recipient, address_encoding,
                                  int(destination_chain_id), int(gas_limit), data,
                                  client().next_nonce())
    return execute("send_to_evm_with_data", "withdraw", req, dry_run=dry_run,
                   needs_owner_key=True, reasons=reasons,
                   summary={"token": wire, "amount": amt, "recipient": recipient,
                            "chain_id": destination_chain_id})
