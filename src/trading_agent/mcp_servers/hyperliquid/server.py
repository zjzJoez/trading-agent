"""hyperliquid-mcp: MCP server for the whole Hyperliquid (HyperCore) API,
built on the official hyperliquid-python-sdk.

Design invariants:
- Network, signing key, write modules and every risk limit come from HL_*
  environment variables read once at startup (settings.py). No tool takes
  them as parameters, so a prompt cannot move the server to mainnet, swap
  the key or lift a cap.
- Testnet by default. Mainnet writes need a reviewed code change
  (guard.MAINNET_WRITES_ENABLED_IN_CODE = True) AND HL_NETWORK=mainnet AND
  HL_ALLOW_MAINNET_WRITES=true.
- Reads (market data, any account's public state) never sign. Writes are
  grouped into modules — trade (on by default), transfer, withdraw, admin,
  advanced (off) — and every one goes through core.execute: guards, then
  dry-run / refuse / sign-and-send, then a JSONL audit record.
- Actions are built in-house (actions.py) instead of via
  hyperliquid.exchange.Exchange so that each can be dry-run, guarded and
  sent for a vault or sub-account per call; they are signed with the SDK's
  own sign_l1_action / sign_user_signed_action, and the tests pin every
  SDK-covered action to what Exchange itself would sign.
"""
from __future__ import annotations

import sys

from trading_agent.mcp_servers.hyperliquid import (  # noqa: F401  (registers tools)
    guard,
    tools_account,
    tools_admin,
    tools_funds,
    tools_market,
    tools_trade,
)
from trading_agent.mcp_servers.hyperliquid.core import client, mcp


def main() -> None:
    c = client()  # parses HL_* env now so a bad config fails at startup
    s = c.settings
    writes = ",".join(sorted(s.write_modules)) or "none"
    print(
        f"[hyperliquid-mcp] network={s.network} signer={c.signer_address or 'none'} "
        f"read_only={s.read_only} dry_run={s.dry_run} write_modules={writes} "
        f"mainnet_writes={'blocked' if guard.mainnet_write_blockers(s) else 'allowed'}",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
