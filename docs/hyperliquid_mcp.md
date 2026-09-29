# hyperliquid-mcp

An MCP server that exposes the whole Hyperliquid (HyperCore) API to an agent: market data, account state, trading, fund movements, staking, vaults, lending, outcome markets and account administration.

It is built on the official [`hyperliquid-python-sdk`](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) (0.24). The SDK does the HTTP transport, both signing schemes and the wire formats. The server adds four things around it: guards, dry runs, an audit log, and every endpoint the SDK does not wrap yet.

Hyperliquid has no official agent toolkit or MCP server as of 2026-09. The SDKs are the official integration surface.

Source: `src/trading_agent/mcp_servers/hyperliquid/`. Tests: `tests/hl_mcp/`.

## Safety model

**This repo is paper-only by design, and this server is the one exception.** It can sign real mainnet transactions if you set two flags. Everything defaults to the safe side, and nothing a prompt says can change these settings:

| Layer | Default | To loosen |
|---|---|---|
| Network | testnet | `HL_NETWORK=mainnet` |
| Mainnet writes | refused | also `HL_ALLOW_MAINNET_WRITES=true` |
| Signing | none (read-only) | `HL_PRIVATE_KEY` (use an API-wallet key) |
| Write modules | `trade` only | `HL_WRITE_MODULES=trade,transfer,withdraw,admin,advanced` |
| Sending funds to other addresses | refused | module `withdraw` **and** destination in `HL_WITHDRAW_ALLOWLIST` |
| Order size | ≤ $1,000 notional per order and per batch | `HL_MAX_ORDER_NOTIONAL_USD` |
| Leverage | ≤ 10x, checked live before every perp entry | `HL_MAX_LEVERAGE` |
| Market-order slippage | 1% default, 5% cap | `HL_DEFAULT_SLIPPAGE`, `HL_MAX_SLIPPAGE` |
| Instruments | any | `HL_ALLOWED_COINS` allowlist |
| Kill switch | off | `HL_READ_ONLY=true`, or `HL_DRY_RUN=true` to preview every write |

Properties that hold for every write tool:

- **Settings are fixed at startup.** No tool takes the network, the key or a limit as a parameter. This is the same invariant moomoo-mcp keeps for `trd_env`.
- **One code path.** Every signed action goes through `core.execute`. It checks the gates, then either returns a dry run, refuses, or signs and sends. It then appends a JSONL record to `data/logs/hyperliquid_audit.jsonl`. The private key never appears in results or in the audit log.
- **Dry runs everywhere.** Every write tool takes `dry_run=true`. A dry run returns the exact action that would be signed, plus every gate that would block it. It works without a key.
- **Exits are never blocked by entry limits.** Reduce-only orders and position closes skip the notional cap and the coin allowlist.
- **Guards fail closed.** An order that cannot be valued in USD, or whose current leverage cannot be read, is refused.
- **Rejections are not reported as success.** The exchange answers `"status": "ok"` even when it rejected an order. `ok` is `false` whenever any order in a batch carries an error.
- **Lost sends can be found and stopped.** Every order gets a client order id (cloid). If a send ends in `status: "error"`, the outcome is unknown. Look the order up with `account_get_order_status(cloid=…)`, or stop it from landing with `nonce_invalidate(nonce)`.
- **API-wallet keys never pass through the model.** `agent_approve` writes the new key to a `0600` file under `data/hyperliquid_agents/` and returns only the path and address.

### Which key to use

Create an **API wallet** (app.hyperliquid.xyz → API, or `agent_approve`). Put its key in `HL_PRIVATE_KEY` and your main address in `HL_ACCOUNT_ADDRESS`.

An API wallet can:
- place, cancel and modify orders, and change leverage and margin;
- move collateral between your own perp dexs and spot (`transfer_between_dexs`).

It cannot withdraw, send funds to another address, or approve anything. The server detects an API-wallet key and refuses owner-only actions up front.

Only use the account owner's key if you need the rest of the `transfer`, `withdraw` or `admin` modules.

## Setup

Fresh clone:

```bash
bash scripts/setup.sh    # uv sync + .env, .mcp.json and .claude/settings.json from the templates
```

Existing checkout (setup.sh leaves an existing `.env`, `.mcp.json` and `.claude/settings.json` alone):

1. `uv sync` installs `hyperliquid-python-sdk` and the `hyperliquid-mcp` command.
2. Copy the `# --- Hyperliquid` block from `.env.example` into `.env`.
3. Add a `hyperliquid-mcp` entry to `.mcp.json`: `{"command": "<repo>/.venv/bin/hyperliquid-mcp", "args": [], "env": {}}`.

Testnet funds: <https://app.hyperliquid-testnet.xyz/drip>. The faucet needs the same address to have deposited on mainnet.

In Claude Code, start with `server_status`. It reports the network, the signer and whether it is an API wallet, the enabled modules, the limits, and whether live writes are possible at all.

## Naming instruments

One order action serves every instrument. The coin name decides which one:

| Kind | Example `coin` | Asset id |
|---|---|---|
| Perp (main dex) | `BTC` | index in `meta` (BTC = 0) |
| HIP-3 perp | `xyz:TSLA` | 100000 + dex_index × 10000 + index |
| Spot | `HYPE/USDC` or `@107` | 10000 + spot index |
| HIP-4 outcome side | `#12090` | 100000000 + 10 × outcome + side |

Also accepted: any casing, the mainnet `U`-prefixed spot remaps (`BTC/USDC` → `UBTC/USDC`), and CEX-style suffixes (`BTC-PERP`, `BTC-USDT-SWAP`). Use `market_search` to find anything else.

Prices snap to valid ticks in the trader's favour: buys round down and sells round up. The tick rule is at most 5 significant figures and at most 6 (perp) or 8 (spot) minus `szDecimals` decimals. Pass `round_to_tick=false` to get an error instead. Sizes are floored to the lot size.

## Tools (102)

Read-only tools never sign anything. Write tools are grouped by module. All write tools accept `dry_run`, and trading tools accept `vault_address` to act for a vault or sub-account.

**Status**
- `server_status`: network, signer, enabled modules, limits, and whether live writes are possible.

**Market data (read-only, 24)**
- `market_search`: find instruments across all perp dexs, spot and outcomes.
- `market_get_ticker`: price, 24h change and volume, funding (hourly) and open interest for any instrument.
- `market_get_perp_markets`, `market_get_spot_markets`, `market_get_outcome_markets`: full market tables.
- `market_get_mids`: mid prices.
- `market_get_orderbook`: L2 book with optional aggregation.
- `market_get_candles`: 1m to 1M candles.
- `market_get_recent_trades`: latest public trades.
- `market_get_funding_history`: hourly funding rates.
- `market_get_predicted_fundings`: predicted funding on Hyperliquid vs Binance and Bybit.
- `market_get_perp_dexs`, `market_get_perp_dex_details`: HIP-3 dexs, OI caps and limits.
- `market_get_trading_limits`: max market-order notional by leverage, perps at their OI cap.
- `market_get_token_info`: spot token supply, decimals and ids.
- `market_get_exchange_status`: upgrades, halts and server time.
- `market_get_vault_details`: a vault's leader, APR, TVL and performance.
- `market_get_validators`: staking validators.
- `market_get_borrow_lend_reserves`: borrow/lend rates and utilisation.
- `market_get_margin_table`: margin tiers.
- `market_get_perp_categories`: perp categories and annotations.
- `market_get_deploy_auctions`: HIP-3 and spot-pair deploy auctions, gossip-priority auction.
- `market_get_settled_outcome`: how an outcome market settled.
- `info_raw`: any `/info` request body. It cannot trade.

**Account (read-only, any address, 23)**
- `account_get_summary`, `account_get_positions`, `account_get_spot_balances`: balances and positions.
- `account_get_open_orders`, `account_get_order_status`, `account_get_order_history`: orders.
- `account_get_fills`, `account_get_funding_payments`, `account_get_ledger`: fills, funding and deposits/withdrawals.
- `account_get_twaps`: TWAP orders and slice fills.
- `account_get_fees`, `account_get_rate_limit`: fee rates and action budget.
- `account_get_portfolio`: account value and PnL history.
- `account_get_asset_state`: per-coin leverage and max trade size.
- `account_get_sub_accounts`, `account_get_vault_equities`, `account_get_staking`, `account_get_borrow_lend_state`.
- `account_get_role`, `account_get_api_wallets`, `account_get_referral`, `account_get_builder_fee_approvals`, `account_get_multisig_signers`.

**trade (on by default, 19)**
- Placing orders:
  - `order_place_limit`: limit order with TIF Gtc, Alo (post-only) or Ioc.
  - `order_place_market`: IOC order at mid ± slippage.
  - `order_place_trigger`: stop or take-profit, market or limit.
  - `order_place_bracket`: entry plus TP/SL legs.
  - `order_place_batch`: several orders in one action, with an optional builder code.
- Positions: `position_close`, `position_close_all`, `position_set_tpsl`.
- Changing and cancelling: `order_modify`, `order_cancel`, `order_cancel_batch`, `order_cancel_all`, `order_schedule_cancel_all` (dead man's switch).
- Leverage and margin: `leverage_update`, `margin_adjust_isolated`, `margin_set_isolated_leverage`.
- TWAP: `twap_place`, `twap_cancel`.
- `nonce_invalidate`: stops an action whose send failed from landing later.

**transfer (own funds, off by default, 11)**
- `transfer_usdc_perp_spot`: USDC between perp and spot.
- `transfer_between_dexs`: collateral between main perps, spot and HIP-3 dexs. Works with an API wallet.
- `transfer_sub_account_usdc`, `transfer_sub_account_spot`: to and from sub-accounts.
- `vault_transfer`: deposit into or withdraw from a vault.
- `staking_deposit`, `staking_withdraw`, `staking_delegate`, `staking_claim_rewards`: HYPE staking.
- `lending_update`: supply, withdraw, borrow or repay.
- `outcome_convert`: split, merge, merge_question or negate outcome shares.

**withdraw (funds leave the account, off by default, allowlisted destinations, 5)**
- `withdraw_to_arbitrum`: through the bridge.
- `send_usdc`, `send_spot_token`, `send_asset`: to another Hyperliquid address.
- `send_to_evm_with_data`: to another chain.

**admin (off by default, 17)**
- `agent_approve`: create an API wallet.
- `builder_fee_approve`: allow a builder to charge fees.
- `sub_account_create`, `sub_account_rename`.
- `referral_use_code`, `referral_create_code`.
- `account_set_display_name`.
- `account_set_abstraction`: unified account or portfolio margin.
- `account_set_portfolio_margin`, `account_set_spot_dusting`, `account_set_evm_big_blocks`.
- `account_reserve_request_weight`: buy extra action budget.
- `staking_link_user`, `staking_unlink_trading_user`.
- `vault_create`, `vault_modify`, `vault_distribute`.

**advanced (off by default, dry-run by default, 2)**
- `advanced_send_l1_action`: any L1 action without a dedicated tool, such as HIP-1/2/3/4 deployer actions, validator actions or gossip-priority bids. It refuses every action type that has a guarded tool.
- `advanced_convert_to_multisig`: convert the account to a multi-sig.

## How it was verified

- **SDK equivalence** (`tests/hl_mcp/test_sdk_equivalence.py`). Every action the SDK can build is built by both `hyperliquid.exchange.Exchange` and this server. Both must produce the same keys in the same order and the same signature. This matters because L1 actions are msgpack-hashed, so key order is part of the signature.
- **Signature oracle on testnet** (`tests/hl_mcp/test_live.py`, run with `-m integration`).
  - A fresh random key has no account, so the exchange rejects every action. The rejection names the address the exchange recovered from the signature.
  - 56 signed-action checks, covering every action type the server can send, all recovered to our wallet. That includes the ones the Python SDK does not implement: TWAP, borrow/lend, cDeposit/cWithdraw, userOutcome, portfolio margin, staking links, vault create/modify/distribute, sendToEvmWithData, agentSendAsset, and batchModify with `always_place`.
  - Their field order follows the [nktkas TypeScript SDK](https://github.com/nktkas/hyperliquid) schemas.
- **Live reads.** Every read tool was run against mainnet.
- **Guards** (`tests/hl_mcp/test_execute.py`, `test_tools.py`). Each gate and each tool's refusal paths are tested against a fake API. None of the 104 unit tests touches the network.

## Known limits

- Snapshots only: websocket streams are not exposed. Poll the read tools instead.
- Multi-sig accounts cannot act through this server, because that needs several co-signers. Converting an account to multi-sig is supported.
- Outcome-market sizes are assumed to be whole shares (`szDecimals = 0`). This was observed on mainnet in 2026-09 and is not documented.
- Not exposed as tools:
  - Order-priority grouping (`{"p": …}`) and the deprecated `userDexAbstraction` action. Their builders exist in `actions.py` and are signature-verified.
  - The TWAP trigger/stop `details` option and `expiresAfter`. These are not implemented.
- The notional cap is per order and per batch. It does not cap total exposure across many calls.
- This repo's `reject_real_env` PreToolUse hook blocks any tool input whose value is exactly `live`, `real` or `production`, for example `market_search("live")`. Normal trading calls pass it.
