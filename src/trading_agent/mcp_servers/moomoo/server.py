"""moomoo-mcp: MCP server wrapping moomoo-api (Moomoo SG OpenD).

Design invariants:
- Trading env is hard-coded to SIMULATE (paper). Real trading is disabled
  at the code level; no tool exposes trd_env as a parameter.
- Read-only quote tools accept stock or option symbols in Moomoo format:
  US.AAPL (stock), US.AAPL250117C00200000 (option). Same code format as Futu.
- Security firm = FUTUSG (Moomoo SG's internal firm code; unchanged since
  the SDK was forked from futu-api).
- Context objects are lazily initialized and reused for the server lifetime.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any, Literal

from moomoo import (
    KLType,
    ModifyOrderOp,
    OpenQuoteContext,
    OpenSecTradeContext,
    OptionType,
    OrderType,
    RET_OK,
    SecurityFirm,
    TrdEnv,
    TrdMarket,
    TrdSide,
)
from mcp.server.fastmcp import FastMCP

from trading_agent.config import CONFIG

# ---------------------------------------------------------------------------
# Silence moomoo-api SDK stdout pollution.
#
# moomoo/common/ft_logger.py:102 hard-codes `logging.StreamHandler(sys.stdout)`
# for its FTConsoleLog singleton. The SDK writes a Python-logging record to
# stdout on every OpenQuoteContext / OpenSecTradeContext connect/disconnect —
# lines like:
#   "2026-04-22 04:26:43,889 | 89449 | ... | [open_context_base.py:409]
#    _init_connect_sync: New connect ready: conn=..."
# That corrupts the MCP JSON-RPC stdio stream (the client's pydantic parser
# raises "Invalid JSON: trailing characters"). The MCP client recovers, but
# any contamination on stdout is a latent bug.
#
# Fix: re-point the SDK's console StreamHandler at stderr. File logging under
# ~/futu_cache/py_YYYY_MM_DD.log is unaffected; interactive/harness stderr
# still shows the line.
# ---------------------------------------------------------------------------
try:
    from moomoo.common.ft_logger import FTLog as _FTLog  # type: ignore
    _ftlog = _FTLog()  # singleton; __init__ is idempotent (hasattr guards)
    _h = getattr(_ftlog, "consoleHandler", None)
    if _h is not None and getattr(_h, "stream", None) is sys.stdout:
        _h.setStream(sys.stderr)
except Exception:
    # Never block MCP startup on a logging tweak — worst case is the
    # historical "tolerated" behavior.
    pass

PAPER_ENV: TrdEnv = TrdEnv.SIMULATE  # MVP: paper only, never real.

# Read-only view of the real Moomoo account. Consumed *exclusively* by the
# get_real_* query tools below and by get_watchlist* (watchlists live on the
# real login even though the API is on the quote ctx, which is env-less).
# INVARIANT (grep-able): REAL_READONLY_ENV must never appear as an argument to
# place_order / modify_order — the write tools hard-code PAPER_ENV and assert
# it with _assert_paper(). Real-account writes also require unlock_trade(),
# which this server never calls; no password handling exists in this module.
REAL_READONLY_ENV: TrdEnv = TrdEnv.REAL

mcp = FastMCP("moomoo-mcp")

_quote_ctx: OpenQuoteContext | None = None
_trade_ctx: OpenSecTradeContext | None = None


def _quote() -> OpenQuoteContext:
    global _quote_ctx
    if _quote_ctx is None:
        _quote_ctx = OpenQuoteContext(host=CONFIG.opend_host, port=CONFIG.opend_port)
    return _quote_ctx


def _trade() -> OpenSecTradeContext:
    global _trade_ctx
    if _trade_ctx is None:
        _trade_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=CONFIG.opend_host,
            port=CONFIG.opend_port,
            security_firm=SecurityFirm.FUTUSG,
        )
    return _trade_ctx


def shutdown() -> None:
    """Close the singleton quote + trade contexts. Idempotent.

    Both ``OpenQuoteContext`` and ``OpenSecTradeContext`` start non-daemon
    background threads (a callback_executor + a select-loop network_manager
    socket reader). Without ``close()`` these threads keep the interpreter
    alive after ``main()`` returns — verified via py-spy stack dump on a
    hanging orchestrator process. Errors are swallowed: the goal is fast
    exit, not perfect cleanup.
    """
    global _quote_ctx, _trade_ctx
    if _quote_ctx is not None:
        try:
            _quote_ctx.close()
        except Exception as e:
            print(f"moomoo quote_ctx close failed: {e}", file=sys.stderr)
        _quote_ctx = None
    if _trade_ctx is not None:
        try:
            _trade_ctx.close()
        except Exception as e:
            print(f"moomoo trade_ctx close failed: {e}", file=sys.stderr)
        _trade_ctx = None


def _require_ok(ret: int, data: Any, op: str) -> Any:
    if ret != RET_OK:
        raise RuntimeError(f"{op} failed: {data}")
    return data


def _df_records(df, cols: list[str] | None = None) -> list[dict]:
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.to_dict(orient="records")


def _assert_paper() -> None:
    """Belt-and-suspenders — reject if module constant got tampered with."""
    if PAPER_ENV != TrdEnv.SIMULATE:
        raise RuntimeError("REFUSED: moomoo-mcp is paper-only; real trading is disabled.")


# ------------------------ Quote / market data ------------------------

@mcp.tool()
def get_quote(symbols: list[str]) -> dict:
    """Market snapshot for US stocks or options.

    Symbols use Moomoo format: "US.AAPL" for stocks, "US.AAPL250117C00200000"
    for options (OCC-like: US.<UNDERLYING><YYMMDD><C|P><STRIKE*1000 padded>).
    Returns a list of row dicts with last_price, prev_close, bid/ask, volume,
    and (for options) greeks, iv, strike, expiry.
    """
    df = _require_ok(*_quote().get_market_snapshot(symbols), op="get_market_snapshot")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_historical_kline(
    symbol: str,
    start: str,
    end: str,
    ktype: Literal["K_DAY", "K_60M", "K_30M", "K_15M", "K_5M", "K_1M", "K_WEEK", "K_MON"] = "K_DAY",
    max_count: int = 1000,
) -> dict:
    """Historical candlesticks for a US stock or ETF.

    `start`/`end` in YYYY-MM-DD. `ktype` defaults to daily. `max_count` caps
    rows returned to protect context size. Use this for structure reads,
    not for training data — no tick-level access on the free tier.
    """
    kl_enum = getattr(KLType, ktype)
    ret, df, page_key = _quote().request_history_kline(
        symbol, start=start, end=end, ktype=kl_enum, max_count=max_count
    )
    _require_ok(ret, df, op="request_history_kline")
    cols = [
        "code", "time_key", "open", "close", "high", "low",
        "volume", "turnover", "change_rate", "last_close",
    ]
    return {"rows": _df_records(df, cols), "has_more": bool(page_key)}


@mcp.tool()
def list_option_expiries(underlying: str) -> dict:
    """Return all available option expiries for a US underlying.

    `underlying` format: "US.AAPL". Returns list of {strike_time, option_expiry_date_distance}.
    """
    df = _require_ok(*_quote().get_option_expiration_date(underlying), op="get_option_expiration_date")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_option_chain(
    underlying: str,
    expiry: str,
    option_type: Literal["ALL", "CALL", "PUT"] = "ALL",
) -> dict:
    """List all option contracts for a US underlying expiring on `expiry` (YYYY-MM-DD).

    Returns rows with option code, strike, type, and basic contract info. To
    get prices/greeks, pass the resulting codes into `get_quote` (which
    populates strike/delta/gamma/vega/theta/iv for option codes).
    """
    opt_enum = {"ALL": OptionType.ALL, "CALL": OptionType.CALL, "PUT": OptionType.PUT}[option_type]
    df = _require_ok(
        *_quote().get_option_chain(code=underlying, start=expiry, end=expiry, option_type=opt_enum),
        op="get_option_chain",
    )
    return {"rows": _df_records(df)}


@mcp.tool()
def get_option_chain_snapshot(
    underlying: str,
    expiry_window_days: int = 45,
    strike_count_each_side: int = 10,
) -> dict:
    """One-shot convenience: pick the nearest expiry within `expiry_window_days`,
    grab ~`strike_count_each_side` strikes above and below spot, and return
    snapshots with greeks.

    Prefer this for a quick look at a name. For fine-grained control compose
    `list_option_expiries` + `get_option_chain` + `get_quote` yourself.
    """
    # 1) pick expiry
    exp_df = _require_ok(
        *_quote().get_option_expiration_date(underlying), op="get_option_expiration_date"
    )
    today = datetime.utcnow().date()
    horizon = today + timedelta(days=expiry_window_days)
    candidates = []
    for _, row in exp_df.iterrows():
        try:
            d = datetime.strptime(row["strike_time"], "%Y-%m-%d").date()
        except Exception:
            continue
        if today <= d <= horizon:
            candidates.append((d, row["strike_time"]))
    if not candidates:
        return {"expiry": None, "rows": [], "note": "no expiries within window"}
    expiry = min(candidates)[1]

    # 2) spot price of underlying
    snap_df = _require_ok(*_quote().get_market_snapshot([underlying]), op="get_market_snapshot")
    spot = float(snap_df.iloc[0]["last_price"])

    # 3) chain for that expiry
    chain_df = _require_ok(
        *_quote().get_option_chain(code=underlying, start=expiry, end=expiry, option_type=OptionType.ALL),
        op="get_option_chain",
    )
    if chain_df.empty:
        return {"expiry": expiry, "spot": spot, "rows": []}

    chain_df["strike_dist"] = (chain_df["strike_price"] - spot).abs()
    keep_codes: set[str] = set()
    for side in ("CALL", "PUT"):
        side_df = chain_df[chain_df["option_type"] == side].nsmallest(
            strike_count_each_side, "strike_dist"
        )
        keep_codes.update(side_df["code"].tolist())

    # 4) snapshot those codes
    codes = sorted(keep_codes)
    snapshots = []
    # Moomoo allows up to 400 per snapshot call; we stay well under.
    if codes:
        opt_snap_df = _require_ok(*_quote().get_market_snapshot(codes), op="get_market_snapshot")
        snapshots = _df_records(opt_snap_df)
    return {"expiry": expiry, "spot": spot, "underlying": underlying, "rows": snapshots}


# ------------------------ Watchlist (real-account, read-only) ------------------------
#
# Watchlists live on the user's Moomoo login and are exposed via the quote
# context (env-less). These tools are pure reads — no modify_user_security
# wrapper is exposed here.

@mcp.tool()
def get_watchlist_groups(group_type: Literal["ALL", "CUSTOM", "SYSTEM"] = "ALL") -> dict:
    """List the user's watchlist groups (自选分组).

    `group_type`: "ALL" (default), "CUSTOM" (user-created), or "SYSTEM"
    (built-in groups like 全部/港股/美股).
    Returns rows with `group_name` and `group_type`. Feed a `group_name`
    into `get_watchlist` to read its tickers.
    """
    from moomoo import UserSecurityGroupType
    gt_enum = {
        "ALL": UserSecurityGroupType.ALL,
        "CUSTOM": UserSecurityGroupType.CUSTOM,
        "SYSTEM": UserSecurityGroupType.SYSTEM,
    }[group_type]
    df = _require_ok(*_quote().get_user_security_group(group_type=gt_enum), op="get_user_security_group")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_watchlist(group_name: str) -> dict:
    """Return the symbols in a watchlist group (自选股列表).

    `group_name` is a string from `get_watchlist_groups().rows[*].group_name`.
    The rows include code (e.g. "US.AAPL"), stock_name, stock_type, and basic
    last_price snapshot. For richer fields pass the codes to `get_quote`.
    """
    df = _require_ok(*_quote().get_user_security(group_name), op="get_user_security")
    return {"rows": _df_records(df)}


# ------------------------ Account / positions (paper) ------------------------

@mcp.tool()
def get_account_info() -> dict:
    """Paper-account summary: cash, total assets, market value, buying power.

    MVP is paper-only — this always queries the SIMULATE env. Real accounts
    are unreachable from this server by design.
    """
    _assert_paper()
    df = _require_ok(*_trade().accinfo_query(trd_env=PAPER_ENV), op="accinfo_query")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_positions() -> dict:
    """Paper-account positions (stocks + options).

    Returns qty, cost, current price, unrealized P&L. Options positions show
    the full option code; cross-reference with `get_quote` for live greeks.
    """
    _assert_paper()
    df = _require_ok(*_trade().position_list_query(trd_env=PAPER_ENV), op="position_list_query")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_orders(status_filter: list[str] | None = None) -> dict:
    """Paper-account orders. Optional `status_filter`: e.g. ["SUBMITTED","FILLED_ALL"]."""
    _assert_paper()
    df = _require_ok(
        *_trade().order_list_query(trd_env=PAPER_ENV, status_filter_list=status_filter or []),
        op="order_list_query",
    )
    return {"rows": _df_records(df)}


# ------------------------ Real account (read-only) ------------------------
#
# Query-only tools that hit the user's live Moomoo account via REAL_READONLY_ENV.
# No write tool in this module accepts trd_env or references REAL_READONLY_ENV.
# Moomoo's real-account write operations additionally require unlock_trade()
# with the trading password; this module never calls unlock_trade and never
# accepts a password parameter — the real-env write path is physically
# unreachable from MCP.

@mcp.tool()
def get_real_account_list() -> dict:
    """List the real Moomoo trading accounts visible to OpenD (US market filter).

    Call this first to discover the `acc_index` to pass into other `get_real_*`
    tools. Rows include acc_id, acc_index, trd_env (REAL/SIMULATE), acc_type
    (CASH / MARGIN), card_num, security_firm. A typical user has a REAL cash
    or margin account plus a SIMULATE paper account.
    """
    df = _require_ok(*_trade().get_acc_list(), op="get_acc_list")
    return {"rows": _df_records(df)}


@mcp.tool()
def get_real_account_info(acc_index: int = 0, currency: Literal["USD", "HKD", "CNH"] = "USD") -> dict:
    """Real-account cash / asset summary (read-only).

    `acc_index` 0-based position in `get_real_account_list().rows`. Defaults
    to 0 which is usually the primary real account — call the list tool first
    if the user has multiple. `currency` controls the display currency of
    cash balances (Moomoo reports per-currency). Returns power/cash/total
    assets, market value, and margin usage.

    refresh_cache=True so we get a fresh snapshot — stale positions mislead
    analysis more than a slow call hurts.
    """
    df = _require_ok(
        *_trade().accinfo_query(
            trd_env=REAL_READONLY_ENV,
            acc_index=acc_index,
            currency=currency,
            refresh_cache=True,
        ),
        op="accinfo_query[real]",
    )
    return {"rows": _df_records(df)}


@mcp.tool()
def get_real_positions(acc_index: int = 0) -> dict:
    """Real-account positions: stocks, ETFs, options (read-only).

    Returns qty, cost, current price, market value, unrealized P&L. Option
    positions show the full option code — pass into `get_quote` for live greeks.
    """
    df = _require_ok(
        *_trade().position_list_query(
            trd_env=REAL_READONLY_ENV,
            acc_index=acc_index,
            refresh_cache=True,
        ),
        op="position_list_query[real]",
    )
    return {"rows": _df_records(df)}


@mcp.tool()
def get_real_orders(
    acc_index: int = 0,
    status_filter: list[str] | None = None,
) -> dict:
    """Real-account open/recent orders (read-only).

    `status_filter`: e.g. ["SUBMITTED", "FILLED_ALL"]. Empty list returns all
    today's orders. For closed orders beyond today use `get_real_history_orders`.
    """
    df = _require_ok(
        *_trade().order_list_query(
            trd_env=REAL_READONLY_ENV,
            acc_index=acc_index,
            status_filter_list=status_filter or [],
            refresh_cache=True,
        ),
        op="order_list_query[real]",
    )
    return {"rows": _df_records(df)}


@mcp.tool()
def get_real_history_orders(
    acc_index: int = 0,
    start: str | None = None,
    end: str | None = None,
    code: str = "",
    status_filter: list[str] | None = None,
) -> dict:
    """Real-account historical orders (read-only).

    `start` / `end` in YYYY-MM-DD; omit for the broker's default window
    (usually last 30–90 days). `code` optionally filters to one symbol
    (e.g. "US.AAPL"). Useful for reconstructing past trades when building a
    thesis or reviewing realized P&L.
    """
    df = _require_ok(
        *_trade().history_order_list_query(
            trd_env=REAL_READONLY_ENV,
            acc_index=acc_index,
            start=start or "",
            end=end or "",
            code=code,
            status_filter_list=status_filter or [],
        ),
        op="history_order_list_query[real]",
    )
    return {"rows": _df_records(df)}


# ------------------------ Paper-only order placement (Day 3) ------------------------
#
# These tools do NOT accept trd_env and always route through PAPER_ENV.
# A separate PreToolUse hook enforces thesis + sizing gates before any of
# these can run — this module only contains the transport layer.

@mcp.tool()
def place_paper_order(
    symbol: str,
    side: Literal["BUY", "SELL"],
    qty: int,
    price: float,
    thesis_id: int,
    order_type: Literal["NORMAL", "MARKET"] = "NORMAL",
    stop: float | None = None,
    target: float | None = None,
    strategy_label: str | None = None,
) -> dict:
    """Place a single US stock/ETF order on the PAPER account.

    `symbol` is "US.AAPL". `price` is the limit price (ignored for MARKET).
    `thesis_id` is the id returned from `record_thesis` in trade-journal-mcp;
    the hook blocks the call if no matching open thesis exists.

    `stop` / `target` / `strategy_label` are *not* sent to the broker; the
    Moomoo paper API only takes a single order leg. They are consumed by
    the PreToolUse sizing hook (stop → R1 risk calc) and by the PostToolUse
    fill-capture hook (persisted into trades for future post-mortem). A
    BUY order without a stop still clears R1 via the 5% implicit-stop
    fallback, but gets a warn entry in the audit log.

    Returns the broker order id and the fill status. The extra params
    are echoed back so the post-hook can recover them from tool_response.
    """
    _assert_paper()
    otype = OrderType.NORMAL if order_type == "NORMAL" else OrderType.MARKET
    tside = TrdSide.BUY if side == "BUY" else TrdSide.SELL
    ret, df = _trade().place_order(
        price=price,
        qty=qty,
        code=symbol,
        trd_side=tside,
        order_type=otype,
        trd_env=PAPER_ENV,
    )
    _require_ok(ret, df, op="place_order[stock]")
    return {
        "thesis_id": thesis_id,
        "stop": stop,
        "target": target,
        "strategy_label": strategy_label,
        "rows": _df_records(df),
    }


@mcp.tool()
def place_paper_option_order(
    option_symbol: str,
    side: Literal["BUY", "SELL"],
    contracts: int,
    price: float,
    thesis_id: int,
    strategy_label: str | None = None,
    delta: float | None = None,
    dte: int | None = None,
) -> dict:
    """Place a single-leg US options order on the PAPER account.

    `option_symbol` is "US.AAPL250117C00200000" format. `contracts` is the
    number of contracts (each = 100 shares underlying). MVP is long-premium
    only and limit-price only — market orders on options are rejected to
    avoid wide-spread fills on illiquid strikes.

    `strategy_label` / `delta` / `dte` are consumed by the PreToolUse
    sizing hook (R5 policy) and the PostToolUse fill-capture hook. They
    are not sent to the broker.

    Falls back to a virtual fill (recorded directly in the journal) if
    MoomooOpenD rejects paper option orders for this account tier.
    """
    _assert_paper()
    tside = TrdSide.BUY if side == "BUY" else TrdSide.SELL
    ret, df = _trade().place_order(
        price=price,
        qty=contracts,
        code=option_symbol,
        trd_side=tside,
        order_type=OrderType.NORMAL,
        trd_env=PAPER_ENV,
    )
    if ret != RET_OK:
        return {
            "thesis_id": thesis_id,
            "strategy_label": strategy_label,
            "delta": delta,
            "dte": dte,
            "virtual_fill_suggested": True,
            "reason": str(df),
            "hint": "MoomooOpenD rejected paper option order; caller should invoke trade-journal-mcp.record_virtual_fill",
        }
    return {
        "thesis_id": thesis_id,
        "strategy_label": strategy_label,
        "delta": delta,
        "dte": dte,
        "rows": _df_records(df),
    }


@mcp.tool()
def cancel_paper_order(order_id: str) -> dict:
    """Cancel an open paper order."""
    _assert_paper()
    ret, df = _trade().modify_order(
        modify_order_op=ModifyOrderOp.CANCEL,
        order_id=order_id,
        qty=0,
        price=0,
        trd_env=PAPER_ENV,
    )
    _require_ok(ret, df, op="modify_order[cancel]")
    return {"rows": _df_records(df)}


# ------------------------ Entry point ------------------------

def main() -> None:
    # Surface config at startup so a misconfigured OpenD host is obvious in logs.
    print(
        f"[moomoo-mcp] connecting to MoomooOpenD at {CONFIG.opend_host}:{CONFIG.opend_port} "
        f"(env={PAPER_ENV}, firm=FUTUSG, paper-only)",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
