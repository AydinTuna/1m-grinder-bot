"""
ATR-based 1-minute grinder (body trigger, ATR-based TP/SL)

Rules:
- Compute ATR (Wilder) on 1m candles.
- If candle body >= thr1*ATR: enter opposite direction at next candle open.
- If candle body >= thr2*ATR: enter opposite direction at next candle open.
- Take profit when price moves tp_atr_mult * ATR from entry.
- Stop loss when price moves sl_atr_mult * ATR from entry.

Implementation details:
- Body = abs(close - open)
- Entry at next candle open (prevents lookahead bias)
- TP/SL are evaluated within each candle using high/low.
- TP/SL ATR uses the signal candle's ATR.
- Fees and slippage are modeled.
- Signals are ignored for the first atr_warmup_bars (defaults to atr_len).
- Margin per trade is capped by initial_capital/margin_usd; leverage may be adjusted from targets to keep TP/SL $ consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from os import getenv
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from config import BacktestConfig, LiveConfig, LIVE_TRADE_FIELDS

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Optional: ccxt for fetching Binance candles
# pip install ccxt
try:
    import ccxt
except Exception:
    ccxt = None

# Optional: binance-futures-connector for live trading
# pip install binance-futures-connector
try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
except Exception:
    UMFutures = None
    ClientError = Exception


# -----------------------------
# Data fetching (Binance via ccxt)
# -----------------------------
def fetch_ohlcv_binance(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    start_utc: Optional[str] = None,  # e.g. "2025-12-01 00:00:00"
    end_utc: Optional[str] = None,    # e.g. "2025-12-02 00:00:00"
    limit_per_call: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance using ccxt, between start_utc and end_utc (UTC strings).
    If start/end not provided, fetches the latest limit_per_call candles.

    Returns a DataFrame indexed by UTC datetime with columns:
    open, high, low, close, volume
    """
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Run: pip install ccxt")

    ex = ccxt.binance({"enableRateLimit": True})

    if start_utc is None and end_utc is None:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=None, limit=limit_per_call)
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("dt").drop(columns=["ts"])
        return df

    start_ms = int(pd.Timestamp(start_utc, tz="UTC").timestamp() * 1000) if start_utc else None
    end_ms = int(pd.Timestamp(end_utc, tz="UTC").timestamp() * 1000) if end_utc else None

    all_rows = []
    since = start_ms

    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not chunk:
            break

        all_rows.extend(chunk)
        last_ts = chunk[-1][0]

        # Advance since by 1ms to avoid repeating last candle
        since = last_ts + 1

        if end_ms is not None and last_ts >= end_ms:
            break

        # Safety: if returned less than limit, likely reached current end
        if len(chunk) < limit_per_call:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["ts"])

    # Trim to end_utc if provided
    if end_utc is not None:
        df = df[df.index <= pd.Timestamp(end_utc, tz="UTC")]

    # Drop duplicates and sort
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# -----------------------------
# Indicators
# -----------------------------
def atr_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Wilder ATR using EMA with alpha=1/length.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    return atr


# -----------------------------
# Strategy signals
# -----------------------------
def build_signals_body_opposite(
    df: pd.DataFrame,
    atr: pd.Series,
    thr1: float = 1.5,
    thr2: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      signal: +1 for long entry, -1 for short entry, 0 for none (signal generated on the candle close)
      signal_atr: ATR value on the signal candle, NaN if none

    Rule:
      body = abs(close-open)
      candle_dir = sign(close-open)
      entry_dir = -candle_dir (opposite)
      if body >= thr1*ATR => signal
      if body >= thr2*ATR => signal (overwrites thr1 if both)
    """
    body = (df["close"] - df["open"]).abs()
    candle_dir = np.sign(df["close"] - df["open"]).astype(int)  # -1,0,+1

    entry_dir = -candle_dir  # opposite direction
    signal = pd.Series(0, index=df.index, dtype=int)
    signal_atr = pd.Series(np.nan, index=df.index, dtype=float)

    cond1 = body >= thr1 * atr
    cond2 = body >= thr2 * atr

    # 1.5x
    signal[cond1] = entry_dir[cond1]
    signal_atr[cond1] = atr[cond1]

    # 2x (overwrite)
    signal[cond2] = entry_dir[cond2]
    signal_atr[cond2] = atr[cond2]

    # If candle_dir is 0 (doji), entry_dir is 0 -> ignore
    signal[signal == 0] = 0
    signal_atr[signal == 0] = np.nan

    return signal, signal_atr


# -----------------------------
# Backtest engine
# -----------------------------
def backtest_atr_grinder(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      trades_df: executed trades
      df_bt: dataframe with indicators/signals/equity
      stats: summary dict
    """
    df = df.copy().sort_index()
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["atr"] = atr_wilder(df, cfg.atr_len)

    df["signal"], df["signal_atr"] = build_signals_body_opposite(
        df, df["atr"],
        thr1=cfg.thr1, thr2=cfg.thr2,
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "signal_atr"] = np.nan

    # Equity curve (USD)
    equity = cfg.initial_capital
    equity_series = pd.Series(np.nan, index=df.index, dtype=float)
    equity_series.iloc[0] = equity

    active_margin = cfg.initial_capital
    active_leverage = cfg.leverage
    active_notional = active_margin * active_leverage

    position = 0  # +1 long, -1 short, 0 flat
    entry_price = None
    target_price = None
    stop_price = None
    entry_time = None
    entry_index = None
    used_signal_atr = None

    trades: List[Dict] = []

    idx = df.index.to_list()

    def apply_slippage(price: float, side: int, is_entry: bool) -> float:
        """
        For buys (long entry) we pay higher, for sells lower.
        For exits: long exit is a sell, short exit is a buy.
        We'll model as:
          long entry (buy): price*(1+slip)
          short entry (sell): price*(1-slip)
          long exit (sell): price*(1-slip)
          short exit (buy): price*(1+slip)
        """
        slip = cfg.slippage
        if is_entry:
            return price * (1 + slip) if side == 1 else price * (1 - slip)
        else:
            # exit action is opposite of position
            return price * (1 - slip) if side == 1 else price * (1 + slip)

    def net_roi_from_prices(entry_p: float, exit_p: float, side: int, leverage: float) -> float:
        """
        Compute net leveraged ROI after fees.
        Underlying return = side*(exit-entry)/entry
        Leveraged ROI approx = underlying_return * leverage
        Fees: charged per side on notional; approximate as fee_rate * leverage per side => 2*fee_rate*leverage
        """
        underlying_ret = side * ((exit_p - entry_p) / entry_p)
        gross_roi = underlying_ret * leverage
        fee_cost = 2.0 * cfg.fee_rate * leverage
        return gross_roi - fee_cost

    for i in range(1, len(df)):
        t = idx[i]
        prev_t = idx[i - 1]

        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # --- EXIT logic (TP/SL) ---
        if position != 0 and entry_price is not None and target_price is not None and stop_price is not None:
            tp_hit = (h >= target_price) if position == 1 else (l <= target_price)
            sl_hit = (l <= stop_price) if position == 1 else (h >= stop_price)

            exited = False
            exit_reason = None
            exit_price = None

            if sl_hit and tp_hit:
                # Conservative assumption when both hit in the same candle.
                exit_reason = "SL"
                exit_price = apply_slippage(stop_price, position, is_entry=False)
                exited = True
            elif tp_hit:
                exit_reason = "TP"
                exit_price = apply_slippage(target_price, position, is_entry=False)
                exited = True
            elif sl_hit:
                exit_reason = "SL"
                exit_price = apply_slippage(stop_price, position, is_entry=False)
                exited = True

            if exited:
                roi_net = net_roi_from_prices(entry_price, exit_price, position, active_leverage)
                pnl_net = roi_net * active_margin
                equity += pnl_net

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": t,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "signal_atr": used_signal_atr,
                    "tp_atr_mult": cfg.tp_atr_mult,
                    "sl_atr_mult": cfg.sl_atr_mult,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "exit_reason": exit_reason,
                    "roi_net": roi_net,
                    "pnl_net": pnl_net,
                    "margin_used": active_margin,
                    "notional": active_notional,
                    "equity_after": equity,
                    "bars_held": i - (entry_index if entry_index is not None else i),
                })

                # flat
                position = 0
                entry_price = None
                target_price = None
                stop_price = None
                entry_time = None
                entry_index = None
                used_signal_atr = None
                active_margin = cfg.initial_capital
                active_leverage = cfg.leverage
                active_notional = active_margin * active_leverage

        # --- ENTRY logic (signal on prev candle, enter at this open) ---
        if position == 0:
            prev_signal = int(df.at[prev_t, "signal"])
            prev_signal_atr = df.at[prev_t, "signal_atr"]

            if prev_signal != 0 and not (isinstance(prev_signal_atr, float) and math.isnan(prev_signal_atr)):
                side = prev_signal
                signal_atr_value = float(prev_signal_atr)

                entry_price = apply_slippage(o, side, is_entry=True)

                trade_margin, trade_leverage, _ = resolve_trade_sizing(
                    entry_price=entry_price,
                    atr_value=signal_atr_value,
                    tp_atr_mult=cfg.tp_atr_mult,
                    sl_atr_mult=cfg.sl_atr_mult,
                    margin_cap=cfg.initial_capital,
                    max_leverage=cfg.leverage,
                    min_leverage=cfg.min_leverage,
                    target_profit_usd=cfg.target_profit_usd,
                    target_loss_usd=cfg.target_loss_usd,
                )
                if trade_margin is None or trade_leverage is None:
                    continue

                target_price, stop_price = compute_tp_sl_prices(
                    entry_price,
                    side,
                    signal_atr_value,
                    cfg.tp_atr_mult,
                    cfg.sl_atr_mult,
                )

                position = side
                entry_time = t
                entry_index = i
                used_signal_atr = signal_atr_value
                active_margin = trade_margin
                active_leverage = trade_leverage
                active_notional = active_margin * active_leverage

        equity_series.at[t] = equity

    df["equity"] = equity_series.ffill()

    trades_df = pd.DataFrame(trades)
    stats = compute_stats(trades_df, df)

    return trades_df, df, stats


def compute_stats(trades_df: pd.DataFrame, df_bt: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "trades": 0,
            "final_equity": float(df_bt["equity"].iloc[-1]),
            "win_rate": 0.0,
            "avg_roi_net": 0.0,
            "total_roi_net": 0.0,
            "total_pnl_net": 0.0,
            "max_drawdown": float(max_drawdown(df_bt["equity"].values)),
        }

    rois = trades_df["roi_net"].astype(float)
    wins = (rois > 0).sum()
    trades = len(trades_df)

    return {
        "trades": trades,
        "final_equity": float(trades_df["equity_after"].iloc[-1]),
        "win_rate": float(wins / trades),
        "avg_roi_net": float(rois.mean()),
        "median_roi_net": float(rois.median()),
        "total_roi_net": float(rois.sum()),
        "total_pnl_net": float(trades_df["pnl_net"].sum()),
        "max_drawdown": float(max_drawdown(df_bt["equity"].values)),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
    }


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Max drawdown as a fraction (e.g., 0.12 = 12%).
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity_curve)
    dd = (peaks - equity_curve) / peaks
    return float(np.nanmax(dd))


# -----------------------------
# Live trading (Binance Futures)
# -----------------------------
@dataclass
class LiveState:
    last_close_ms: Optional[int] = None
    entry_order_id: Optional[int] = None
    entry_order_time: Optional[float] = None
    pending_atr: Optional[float] = None
    pending_side: Optional[int] = None
    active_atr: Optional[float] = None
    tp_algo_id: Optional[int] = None
    sl_algo_id: Optional[int] = None
    tp_client_algo_id: Optional[str] = None
    sl_client_algo_id: Optional[str] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    entry_close_ms: Optional[int] = None
    entry_price: Optional[float] = None
    entry_qty: Optional[float] = None
    entry_margin_usd: Optional[float] = None
    entry_leverage: Optional[int] = None
    current_leverage: Optional[int] = None
    last_atr: Optional[float] = None
    entry_side: Optional[int] = None
    entry_time_iso: Optional[str] = None
    entry_signal_ms: Optional[int] = None
    had_position: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_event(log_path: str, event: Dict[str, object]) -> None:
    payload = dict(event)
    payload["ts"] = _utc_now_iso()
    with open(log_path, "a", encoding="ascii") as f:
        json.dump(payload, f, ensure_ascii=True)
        f.write("\n")


def format_float_1(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def format_float_2(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def format_float_2_str(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def interval_to_ms(interval: str) -> Optional[int]:
    if not interval or len(interval) < 2:
        return None
    num_part = interval[:-1]
    unit = interval[-1]
    if not num_part.isdigit():
        return None
    value = int(num_part)
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    if unit == "w":
        return value * 604_800_000
    return None


def format_to_step(value: float, step: float) -> str:
    step_dec = Decimal(str(step))
    if step_dec == 0:
        return format(Decimal(str(value)), "f")
    quant = (Decimal(str(value)) / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
    return format(quant, "f")


def get_um_futures_client(cfg: LiveConfig) -> UMFutures:
    if UMFutures is None:
        raise RuntimeError("binance-futures-connector not installed.")
    if load_dotenv is not None:
        load_dotenv()
    api_key = getenv("BINANCE_API_KEY")
    api_secret = getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_API_SECRET in environment.")
    base_url = "https://fapi.binance.com"
    if cfg.use_testnet:
        base_url = "https://testnet.binancefuture.com"
    return UMFutures(key=api_key, secret=api_secret, base_url=base_url)


def get_symbol_filters(client: UMFutures, symbol: str) -> Dict[str, float]:
    info = client.exchange_info()
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            tick_size = None
            step_size = None
            min_qty = None
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                elif f.get("filterType") == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    min_qty = float(f["minQty"])
            if tick_size is None or step_size is None or min_qty is None:
                raise RuntimeError(f"Missing filters for symbol: {symbol}")
            return {
                "tick_size": tick_size,
                "step_size": step_size,
                "min_qty": min_qty,
            }
    raise RuntimeError(f"Symbol not found: {symbol}")


def get_book_ticker(client: UMFutures, symbol: str) -> Tuple[float, float]:
    book = client.book_ticker(symbol=symbol)
    return float(book["bidPrice"]), float(book["askPrice"])


def get_position_info(client: UMFutures, symbol: str) -> Dict[str, str]:
    info = client.get_position_risk(symbol=symbol)
    if isinstance(info, list):
        for pos in info:
            if pos.get("symbol") == symbol:
                return pos
        return info[0] if info else {}
    return info


def limit_order(
    symbol: str,
    side: str,
    quantity: str,
    price: str,
    client: UMFutures,
    time_in_force: str = "GTC",
    reduce_only: bool = False,
    client_order_id: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """
    Places a limit order on Binance Futures.
    """
    try:
        params: Dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "timeInForce": time_in_force,
            "price": price,
        }
        if reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return client.new_order(**params)
    except ClientError as error:
        if getattr(error, "error_code", None) == -5022:
            logging.info(
                "Post-only order rejected. status: %s, code: %s, message: %s",
                getattr(error, "status_code", None),
                getattr(error, "error_code", None),
                getattr(error, "error_message", None),
            )
            return None
        logging.error(
            "Order error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def stop_limit_order(
    symbol: str,
    side: str,
    quantity: str,
    stop_price: str,
    price: str,
    client: UMFutures,
    reduce_only: bool = True,
    client_order_id: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    try:
        params: Dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP",
            "quantity": quantity,
            "timeInForce": "GTC",
            "price": price,
            "stopPrice": stop_price,
        }
        if reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return client.new_order(**params)
    except ClientError as error:
        logging.error(
            "Stop order error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def algo_order(
    symbol: str,
    side: str,
    order_type: str,
    quantity: Optional[str],
    trigger_price: str,
    client: UMFutures,
    price: Optional[str] = None,
    time_in_force: str = "GTC",
    reduce_only: bool = True,
    close_position: bool = False,
    client_algo_id: Optional[str] = None,
    working_type: Optional[str] = None,
    price_protect: Optional[bool] = None,
) -> Optional[Dict[str, object]]:
    """
    Places a conditional algo order (STOP/TAKE_PROFIT/STOP_MARKET/TAKE_PROFIT_MARKET).
    """
    try:
        params: Dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "algoType": "CONDITIONAL",
            "type": order_type,
            "triggerPrice": trigger_price,
        }
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = True
        if close_position:
            params["closePosition"] = True
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        if working_type:
            params["workingType"] = working_type
        if price_protect is not None:
            params["priceProtect"] = "TRUE" if price_protect else "FALSE"
        return client.sign_request("POST", "/fapi/v1/algoOrder", params)
    except ClientError as error:
        logging.error(
            "Algo order error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def query_algo_order(
    client: UMFutures,
    symbol: str,
    algo_id: Optional[int] = None,
    client_algo_id: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    try:
        params: Dict[str, object] = {"symbol": symbol}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        if "algoId" not in params and "clientAlgoId" not in params:
            return None
        return client.sign_request("GET", "/fapi/v1/algoOrder", params)
    except ClientError as error:
        logging.error(
            "Algo query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def cancel_algo_order(
    client: UMFutures,
    symbol: str,
    algo_id: Optional[int] = None,
    client_algo_id: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    try:
        params: Dict[str, object] = {"symbol": symbol}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        if "algoId" not in params and "clientAlgoId" not in params:
            return None
        return client.sign_request("DELETE", "/fapi/v1/algoOrder", params)
    except ClientError as error:
        logging.error(
            "Algo cancel error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def algo_order_triggered(order: Optional[Dict[str, object]]) -> bool:
    if not order:
        return False
    status = str(order.get("algoStatus") or "").upper()
    if status in {"TRIGGERED", "ORDER", "FILLED"}:
        return True
    try:
        return int(order.get("triggerTime") or 0) > 0
    except (TypeError, ValueError):
        return False


def algo_order_inactive(order: Optional[Dict[str, object]]) -> bool:
    if not order:
        return False
    status = str(order.get("algoStatus") or "").upper()
    return status in {"CANCELED", "EXPIRED", "REJECTED"}


def append_live_trade(csv_path: str, trade: Dict[str, object]) -> None:
    if not csv_path:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_TRADE_FIELDS)
        if not file_exists:
            writer.writeheader()
        row = {}
        for field in LIVE_TRADE_FIELDS:
            value = trade.get(field)
            row[field] = "" if value is None else value
        writer.writerow(row)


def klines_to_df(klines: List[List[object]]) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append({
            "dt": pd.to_datetime(k[0], unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time_ms": int(k[6]),
        })
    df = pd.DataFrame(rows).set_index("dt")
    return df


def compute_live_signal(df: pd.DataFrame, cfg: LiveConfig) -> Tuple[int, Optional[float], Optional[float]]:
    warmup = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if len(df) <= warmup:
        return 0, None, None

    atr = atr_wilder(df, cfg.atr_len)
    signal, signal_atr = build_signals_body_opposite(
        df,
        atr,
        thr1=cfg.thr1,
        thr2=cfg.thr2,
    )

    signal_val = int(signal.iloc[-1])
    atr_val = float(atr.iloc[-1])
    signal_atr_val = signal_atr.iloc[-1]
    if signal_val == 0 or pd.isna(signal_atr_val) or math.isnan(atr_val):
        return 0, None, atr_val if not math.isnan(atr_val) else None
    return signal_val, float(signal_atr_val), atr_val


def compute_tp_sl_prices(
    entry_price: float,
    side: int,
    atr_value: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
) -> Tuple[float, float]:
    move_tp = atr_value * tp_atr_mult
    move_sl = atr_value * sl_atr_mult
    if side == 1:
        target_price = entry_price + move_tp
        stop_price = entry_price - move_sl
    else:
        target_price = entry_price - move_tp
        stop_price = entry_price + move_sl
    return target_price, stop_price


def compute_margin_from_targets(
    entry_price: float,
    atr_value: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    leverage: float,
    target_profit_usd: Optional[float],
    target_loss_usd: Optional[float],
) -> Optional[float]:
    if entry_price <= 0 or atr_value <= 0 or leverage <= 0:
        return None
    move_tp = atr_value * tp_atr_mult
    move_sl = atr_value * sl_atr_mult
    if move_tp <= 0 or move_sl <= 0:
        return None
    roi_tp = (move_tp / entry_price) * leverage
    roi_sl = (move_sl / entry_price) * leverage
    margin_from_loss = None
    if target_loss_usd is not None and target_loss_usd > 0 and roi_sl > 0:
        margin_from_loss = target_loss_usd / roi_sl
    if margin_from_loss is not None:
        return margin_from_loss
    if target_profit_usd is not None and target_profit_usd > 0 and roi_tp > 0:
        return target_profit_usd / roi_tp
    return None


def compute_required_leverage(
    entry_price: float,
    atr_value: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    margin_usd: float,
    target_profit_usd: Optional[float],
    target_loss_usd: Optional[float],
) -> Optional[float]:
    if entry_price <= 0 or atr_value <= 0 or margin_usd <= 0:
        return None
    move_tp = atr_value * tp_atr_mult
    move_sl = atr_value * sl_atr_mult
    if target_loss_usd is not None and target_loss_usd > 0 and move_sl > 0:
        return (target_loss_usd * entry_price) / (move_sl * margin_usd)
    if target_profit_usd is not None and target_profit_usd > 0 and move_tp > 0:
        return (target_profit_usd * entry_price) / (move_tp * margin_usd)
    return None


def resolve_trade_sizing(
    entry_price: float,
    atr_value: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    margin_cap: float,
    max_leverage: float,
    min_leverage: float,
    target_profit_usd: Optional[float],
    target_loss_usd: Optional[float],
    leverage_step: float = 0.0,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if margin_cap <= 0 or max_leverage <= 0:
        return None, None, "invalid_config"
    req_leverage = compute_required_leverage(
        entry_price=entry_price,
        atr_value=atr_value,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        margin_usd=margin_cap,
        target_profit_usd=target_profit_usd,
        target_loss_usd=target_loss_usd,
    )
    if req_leverage is None:
        return margin_cap, max_leverage, None
    leverage_used = max(req_leverage, min_leverage)
    if leverage_step and leverage_step > 0:
        leverage_used = math.ceil(leverage_used / leverage_step) * leverage_step
    if leverage_used > max_leverage:
        return None, None, "leverage_exceeds_max"
    margin_required = compute_margin_from_targets(
        entry_price=entry_price,
        atr_value=atr_value,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        leverage=leverage_used,
        target_profit_usd=target_profit_usd,
        target_loss_usd=target_loss_usd,
    )
    if margin_required is None or margin_required <= 0:
        return margin_cap, leverage_used, None
    if margin_required > margin_cap:
        return None, None, "margin_exceeds_cap"
    return margin_required, leverage_used, None



def run_live(cfg: LiveConfig) -> None:
    if load_dotenv is not None:
        load_dotenv()
    symbol_override = getenv("LIVE_SYMBOL")
    if symbol_override:
        cfg.symbol = symbol_override
    if getenv("LIVE_TRADING") != "1":
        raise RuntimeError("Set LIVE_TRADING=1 to enable live trading.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = get_um_futures_client(cfg)
    filters = get_symbol_filters(client, cfg.symbol)

    try:
        client.change_leverage(symbol=cfg.symbol, leverage=cfg.leverage)
    except ClientError:
        pass

    state = LiveState()
    state.current_leverage = cfg.leverage
    interval_ms = interval_to_ms(cfg.interval)
    log_event(cfg.log_path, {
        "event": "startup",
        "symbol": cfg.symbol,
        "leverage": cfg.leverage,
        "margin_usd": format_float_2(cfg.margin_usd),
        "target_profit_usd": format_float_2(cfg.target_profit_usd),
        "target_loss_usd": format_float_2(cfg.target_loss_usd),
    })

    while True:
        try:
            position = get_position_info(client, cfg.symbol)
            position_amt = float(position.get("positionAmt", 0.0))
            exit_filled = False
            in_position = position_amt != 0.0
            if in_position:
                state.had_position = True

            # Manage open entry order
            if state.entry_order_id is not None:
                try:
                    order = client.query_order(symbol=cfg.symbol, orderId=state.entry_order_id)
                except ClientError:
                    log_event(cfg.log_path, {"event": "entry_order_query_error", "order_id": state.entry_order_id})
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.pending_atr = None
                    state.pending_side = None
                    state.entry_close_ms = None
                    state.entry_margin_usd = None
                    state.entry_leverage = None
                    state.entry_side = None
                    state.entry_time_iso = None
                    state.entry_signal_ms = None
                    order = None
                if order is None:
                    time.sleep(cfg.poll_interval_seconds)
                    continue
                status = order.get("status")
                if status == "FILLED":
                    entry_price = float(order.get("avgPrice") or order.get("price"))
                    qty = float(order.get("executedQty") or order.get("origQty"))
                    side = state.pending_side or (1 if position_amt > 0 else -1)
                    entry_close_ms = state.entry_close_ms
                    state.active_atr = state.pending_atr
                    state.pending_atr = None
                    state.pending_side = None
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.entry_price = entry_price
                    state.entry_qty = qty
                    if state.entry_margin_usd is None:
                        state.entry_margin_usd = cfg.margin_usd
                    if state.entry_leverage is None:
                        state.entry_leverage = state.current_leverage or cfg.leverage
                    state.entry_side = side
                    state.entry_time_iso = _utc_now_iso()
                    if state.last_close_ms is not None:
                        state.entry_signal_ms = state.last_close_ms
                    else:
                        state.entry_signal_ms = entry_close_ms

                    entry_atr = state.active_atr if state.active_atr is not None else state.last_atr
                    if entry_atr is None:
                        log_event(cfg.log_path, {"event": "missing_atr", "stage": "entry_filled"})
                        entry_atr = 0.0
                    state.active_atr = entry_atr
                    tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                    tp_price_str = format_to_step(tp_price, filters["tick_size"])
                    sl_price_str = format_to_step(sl_price, filters["tick_size"])
                    qty_str = format_to_step(qty, filters["step_size"])

                    tp_side = "SELL" if side == 1 else "BUY"
                    sl_side = "SELL" if side == 1 else "BUY"
                    tp_client_id = f"ATR_TP_{entry_close_ms or ''}"
                    sl_client_id = f"ATR_SL_{entry_close_ms or ''}"
                    tp_order = algo_order(
                        cfg.symbol,
                        tp_side,
                        "TAKE_PROFIT",
                        qty_str,
                        tp_price_str,
                        client,
                        price=tp_price_str,
                        time_in_force="GTC",
                        reduce_only=True,
                        client_algo_id=tp_client_id,
                        working_type=cfg.algo_working_type,
                        price_protect=cfg.algo_price_protect,
                    )
                    sl_order = algo_order(
                        cfg.symbol,
                        sl_side,
                        "STOP",
                        qty_str,
                        sl_price_str,
                        client,
                        price=sl_price_str,
                        time_in_force="GTC",
                        reduce_only=True,
                        client_algo_id=sl_client_id,
                        working_type=cfg.algo_working_type,
                        price_protect=cfg.algo_price_protect,
                    )
                    state.tp_algo_id = tp_order.get("algoId") if tp_order else None
                    state.sl_algo_id = sl_order.get("algoId") if sl_order else None
                    state.tp_client_algo_id = (tp_order.get("clientAlgoId") if tp_order else None) or (tp_client_id if tp_order else None)
                    state.sl_client_algo_id = (sl_order.get("clientAlgoId") if sl_order else None) or (sl_client_id if sl_order else None)
                    state.tp_price = tp_price
                    state.sl_price = sl_price
                    state.entry_close_ms = None

                    log_event(cfg.log_path, {
                        "event": "entry_filled",
                        "order_id": order.get("orderId"),
                        "side": "LONG" if side == 1 else "SHORT",
                        "entry_price": format_float_2(entry_price),
                        "quantity": format_float_2(qty),
                        "entry_atr": format_float_2(state.active_atr),
                        "margin_usd": format_float_2(state.entry_margin_usd),
                        "leverage": state.entry_leverage,
                        "tp_price": format_float_2(tp_price),
                        "sl_price": format_float_2(sl_price),
                        "tp_algo_id": state.tp_algo_id,
                        "sl_algo_id": state.sl_algo_id,
                    })
                elif status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    log_event(cfg.log_path, {"event": "entry_order_closed", "order_id": state.entry_order_id, "status": status})
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.pending_atr = None
                    state.pending_side = None
                    state.entry_close_ms = None
                    state.entry_margin_usd = None
                    state.entry_leverage = None
                else:
                    if state.entry_order_time and (time.time() - state.entry_order_time) > cfg.entry_order_timeout_seconds:
                        try:
                            client.cancel_order(symbol=cfg.symbol, orderId=state.entry_order_id)
                        except ClientError:
                            pass
                        log_event(cfg.log_path, {"event": "entry_order_timeout", "order_id": state.entry_order_id})
                        state.entry_order_id = None
                        state.entry_order_time = None
                        state.pending_atr = None
                        state.pending_side = None
                        state.entry_close_ms = None
                        state.entry_margin_usd = None
                        state.entry_leverage = None
                        state.entry_side = None
                        state.entry_time_iso = None
                        state.entry_signal_ms = None

            if not in_position and state.had_position:
                tp_order = query_algo_order(
                    client,
                    cfg.symbol,
                    algo_id=state.tp_algo_id,
                    client_algo_id=state.tp_client_algo_id,
                ) if (state.tp_algo_id or state.tp_client_algo_id) else None
                sl_order = query_algo_order(
                    client,
                    cfg.symbol,
                    algo_id=state.sl_algo_id,
                    client_algo_id=state.sl_client_algo_id,
                ) if (state.sl_algo_id or state.sl_client_algo_id) else None
                tp_triggered = algo_order_triggered(tp_order)
                sl_triggered = algo_order_triggered(sl_order)

                if tp_triggered and not sl_triggered:
                    exit_reason = "TP"
                    exit_price = state.tp_price
                elif sl_triggered and not tp_triggered:
                    exit_reason = "SL"
                    exit_price = state.sl_price
                else:
                    exit_reason = "EXIT"
                    exit_price = None

                side_sign = state.entry_side or 0
                pnl_net = None
                roi_net = None
                notional = None
                margin_used = state.entry_margin_usd or cfg.margin_usd
                bars_held = None
                if exit_price is not None and state.entry_price is not None and state.entry_qty is not None and side_sign != 0:
                    pnl_net = (exit_price - state.entry_price) * state.entry_qty * side_sign
                    notional = state.entry_price * state.entry_qty
                    if margin_used:
                        roi_net = pnl_net / margin_used
                if interval_ms and state.entry_signal_ms and state.last_close_ms:
                    bars_held = max(0, int((state.last_close_ms - state.entry_signal_ms) / interval_ms))

                log_event(cfg.log_path, {
                    "event": "exit_filled",
                    "exit_reason": exit_reason,
                    "exit_price": format_float_2(exit_price),
                    "entry_price": format_float_2(state.entry_price),
                    "quantity": format_float_2(state.entry_qty),
                    "entry_atr": format_float_2(state.active_atr),
                    "margin_usd": format_float_2(margin_used),
                    "leverage": state.entry_leverage or state.current_leverage or cfg.leverage,
                    "tp_algo_id": state.tp_algo_id,
                    "sl_algo_id": state.sl_algo_id,
                })

                append_live_trade(cfg.live_trades_csv, {
                    "entry_time": state.entry_time_iso,
                    "exit_time": _utc_now_iso(),
                    "side": "LONG" if side_sign == 1 else ("SHORT" if side_sign == -1 else ""),
                    "entry_price": format_float_2_str(state.entry_price),
                    "exit_price": format_float_2_str(exit_price),
                    "signal_atr": format_float_2_str(state.active_atr),
                    "tp_atr_mult": format_float_2_str(cfg.tp_atr_mult),
                    "sl_atr_mult": format_float_2_str(cfg.sl_atr_mult),
                    "target_price": format_float_2_str(state.tp_price),
                    "stop_price": format_float_2_str(state.sl_price),
                    "exit_reason": exit_reason,
                    "roi_net": format_float_2_str(roi_net),
                    "pnl_net": format_float_2_str(pnl_net),
                    "margin_used": format_float_2_str(margin_used),
                    "notional": format_float_2_str(notional),
                    "equity_after": "",
                    "bars_held": "" if bars_held is None else str(bars_held),
                })

                if state.tp_algo_id or state.tp_client_algo_id:
                    cancel_algo_order(client, cfg.symbol, algo_id=state.tp_algo_id, client_algo_id=state.tp_client_algo_id)
                if state.sl_algo_id or state.sl_client_algo_id:
                    cancel_algo_order(client, cfg.symbol, algo_id=state.sl_algo_id, client_algo_id=state.sl_client_algo_id)

                state.tp_algo_id = None
                state.sl_algo_id = None
                state.tp_client_algo_id = None
                state.sl_client_algo_id = None
                state.tp_price = None
                state.sl_price = None
                state.active_atr = None
                state.pending_atr = None
                state.entry_price = None
                state.entry_qty = None
                state.entry_margin_usd = None
                state.entry_leverage = None
                state.entry_side = None
                state.entry_time_iso = None
                state.entry_signal_ms = None
                state.had_position = False
                exit_filled = True

            if exit_filled:
                time.sleep(cfg.poll_interval_seconds)
                continue

            if not in_position and state.entry_price is None and (state.tp_algo_id or state.sl_algo_id):
                if state.tp_algo_id or state.tp_client_algo_id:
                    cancel_algo_order(client, cfg.symbol, algo_id=state.tp_algo_id, client_algo_id=state.tp_client_algo_id)
                if state.sl_algo_id or state.sl_client_algo_id:
                    cancel_algo_order(client, cfg.symbol, algo_id=state.sl_algo_id, client_algo_id=state.sl_client_algo_id)
                log_event(cfg.log_path, {"event": "position_closed"})
                state.tp_algo_id = None
                state.sl_algo_id = None
                state.tp_client_algo_id = None
                state.sl_client_algo_id = None
                state.tp_price = None
                state.sl_price = None
                state.active_atr = None
                state.pending_atr = None
                state.entry_price = None
                state.entry_qty = None
                state.entry_margin_usd = None
                state.entry_leverage = None
                state.entry_side = None
                state.entry_time_iso = None
                state.entry_signal_ms = None
                state.had_position = False

            if in_position:
                if state.tp_algo_id or state.tp_client_algo_id:
                    tp_order = query_algo_order(
                        client,
                        cfg.symbol,
                        algo_id=state.tp_algo_id,
                        client_algo_id=state.tp_client_algo_id,
                    )
                    if algo_order_inactive(tp_order):
                        state.tp_algo_id = None
                        state.tp_client_algo_id = None
                if state.sl_algo_id or state.sl_client_algo_id:
                    sl_order = query_algo_order(
                        client,
                        cfg.symbol,
                        algo_id=state.sl_algo_id,
                        client_algo_id=state.sl_client_algo_id,
                    )
                    if algo_order_inactive(sl_order):
                        state.sl_algo_id = None
                        state.sl_client_algo_id = None

            # If in position and no exits placed, place them using current entryPrice
            if in_position and (state.tp_algo_id is None or state.sl_algo_id is None):
                side = 1 if position_amt > 0 else -1
                entry_price = float(position.get("entryPrice", 0.0))
                qty = abs(position_amt)
                if state.entry_side is None:
                    state.entry_side = side
                entry_atr = state.active_atr or state.pending_atr or state.last_atr
                if entry_atr is None:
                    log_event(cfg.log_path, {"event": "missing_atr", "stage": "exit_recover"})
                    entry_atr = 0.0
                state.active_atr = entry_atr
                state.entry_price = entry_price
                state.entry_qty = qty

                tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                tp_price_str = format_to_step(tp_price, filters["tick_size"])
                sl_price_str = format_to_step(sl_price, filters["tick_size"])
                qty_str = format_to_step(qty, filters["step_size"])
                tp_side = "SELL" if side == 1 else "BUY"
                sl_side = "SELL" if side == 1 else "BUY"

                if state.tp_algo_id is None:
                    tp_client_id = f"ATR_TP_RECOVER_{int(time.time())}"
                    tp_order = algo_order(
                        cfg.symbol,
                        tp_side,
                        "TAKE_PROFIT",
                        qty_str,
                        tp_price_str,
                        client,
                        price=tp_price_str,
                        time_in_force="GTC",
                        reduce_only=True,
                        client_algo_id=tp_client_id,
                        working_type=cfg.algo_working_type,
                        price_protect=cfg.algo_price_protect,
                    )
                    state.tp_algo_id = tp_order.get("algoId") if tp_order else None
                    state.tp_client_algo_id = (tp_order.get("clientAlgoId") if tp_order else None) or (tp_client_id if tp_order else None)
                    state.tp_price = tp_price
                if state.sl_algo_id is None:
                    sl_client_id = f"ATR_SL_RECOVER_{int(time.time())}"
                    sl_order = algo_order(
                        cfg.symbol,
                        sl_side,
                        "STOP",
                        qty_str,
                        sl_price_str,
                        client,
                        price=sl_price_str,
                        time_in_force="GTC",
                        reduce_only=True,
                        client_algo_id=sl_client_id,
                        working_type=cfg.algo_working_type,
                        price_protect=cfg.algo_price_protect,
                    )
                    state.sl_algo_id = sl_order.get("algoId") if sl_order else None
                    state.sl_client_algo_id = (sl_order.get("clientAlgoId") if sl_order else None) or (sl_client_id if sl_order else None)
                    state.sl_price = sl_price

            # New candle check
            klines = client.klines(symbol=cfg.symbol, interval=cfg.interval, limit=cfg.atr_len + 2)
            if len(klines) >= 2:
                closed_klines = klines[:-1]
                close_time_ms = int(closed_klines[-1][6])
                if close_time_ms != state.last_close_ms:
                    state.last_close_ms = close_time_ms
                    df = klines_to_df(closed_klines)
                    signal, signal_atr, atr_value = compute_live_signal(df, cfg)
                    state.last_atr = atr_value
                    candle = df.iloc[-1]
                    body = abs(float(candle["close"]) - float(candle["open"]))
                    log_event(cfg.log_path, {
                        "event": "candle_close",
                        "close_time_ms": close_time_ms,
                        "open": format_float_1(candle["open"]),
                        "high": format_float_1(candle["high"]),
                        "low": format_float_1(candle["low"]),
                        "close": format_float_1(candle["close"]),
                        "body": format_float_1(body),
                        "atr": format_float_1(atr_value),
                        "signal": signal,
                        "signal_atr": format_float_1(signal_atr),
                    })

                    if position_amt == 0.0 and state.entry_order_id is None and signal != 0 and signal_atr is not None and atr_value is not None:
                        delay = cfg.entry_delay_min_seconds
                        if cfg.entry_delay_max_seconds > cfg.entry_delay_min_seconds:
                            delay = random.uniform(cfg.entry_delay_min_seconds, cfg.entry_delay_max_seconds)
                        time.sleep(delay)

                        bid, ask = get_book_ticker(client, cfg.symbol)
                        mid = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid if mid else 1.0
                        if spread_pct > cfg.spread_max_pct:
                            log_event(cfg.log_path, {"event": "skip_spread", "spread_pct": format_float_2(spread_pct)})
                        else:
                            offset = atr_value * cfg.atr_offset_mult
                            if signal == 1:
                                limit_price = bid - offset
                                side = "BUY"
                            else:
                                limit_price = ask + offset
                                side = "SELL"

                            if cfg.post_only:
                                tick = filters["tick_size"]
                                if tick > 0:
                                    if side == "BUY":
                                        limit_price = min(limit_price, bid - tick)
                                    else:
                                        limit_price = max(limit_price, ask + tick)

                            if limit_price <= 0:
                                log_event(cfg.log_path, {"event": "skip_price", "limit_price": format_float_2(limit_price)})
                            else:
                                limit_price_str = format_to_step(limit_price, filters["tick_size"])
                                margin_usd, leverage_used, skip_reason = resolve_trade_sizing(
                                    entry_price=limit_price,
                                    atr_value=signal_atr,
                                    tp_atr_mult=cfg.tp_atr_mult,
                                    sl_atr_mult=cfg.sl_atr_mult,
                                    margin_cap=cfg.margin_usd,
                                    max_leverage=cfg.leverage,
                                    min_leverage=cfg.min_leverage,
                                    target_profit_usd=cfg.target_profit_usd,
                                    target_loss_usd=cfg.target_loss_usd,
                                    leverage_step=1.0,
                                )
                                if margin_usd is None or leverage_used is None:
                                    log_event(cfg.log_path, {
                                        "event": "skip_leverage",
                                        "reason": skip_reason,
                                        "margin_cap": format_float_2(cfg.margin_usd),
                                        "max_leverage": cfg.leverage,
                                    })
                                    continue
                                leverage_int = int(round(leverage_used))
                                if state.current_leverage != leverage_int:
                                    try:
                                        client.change_leverage(symbol=cfg.symbol, leverage=leverage_int)
                                        state.current_leverage = leverage_int
                                    except ClientError:
                                        log_event(cfg.log_path, {"event": "leverage_change_error", "leverage": leverage_int})
                                        continue
                                notional = margin_usd * leverage_int
                                qty = notional / limit_price
                                qty_str = format_to_step(qty, filters["step_size"])
                                if float(qty_str) < filters["min_qty"]:
                                    log_event(cfg.log_path, {"event": "skip_qty", "quantity": format_float_2(qty_str)})
                                else:
                                    tif = "GTX" if cfg.post_only else "GTC"
                                    entry_order = limit_order(
                                        cfg.symbol,
                                        side,
                                        qty_str,
                                        limit_price_str,
                                        client,
                                        time_in_force=tif,
                                        reduce_only=False,
                                        client_order_id=f"ATR_E_{close_time_ms}",
                                    )
                                    if entry_order:
                                        state.entry_order_id = entry_order.get("orderId")
                                        state.entry_order_time = time.time()
                                        state.pending_atr = signal_atr
                                        state.pending_side = 1 if side == "BUY" else -1
                                        state.entry_close_ms = close_time_ms
                                        state.entry_margin_usd = margin_usd
                                        state.entry_leverage = leverage_int
                                        log_event(cfg.log_path, {
                                            "event": "entry_order",
                                            "order_id": state.entry_order_id,
                                            "side": side,
                                            "limit_price": format_float_2(limit_price_str),
                                            "quantity": format_float_2(qty_str),
                                            "signal_atr": format_float_2(signal_atr),
                                            "margin_usd": format_float_2(margin_usd),
                                            "leverage": leverage_int,
                                            "spread_pct": format_float_2(spread_pct),
                                            "offset": format_float_2(offset),
                                        })
        except Exception as exc:
            logging.exception("Live loop error")
            log_event(cfg.log_path, {"event": "error", "message": str(exc)})

        time.sleep(cfg.poll_interval_seconds)

# -----------------------------
# Example usage
# -----------------------------
def run_backtest() -> None:
    # 1) Get data
    # Option A: Use Binance via ccxt (recommended)
    df = fetch_ohlcv_binance(
        symbol="BTC/USDT",
        timeframe="1m",
        start_utc="2025-12-30 00:00:00",
        end_utc="2025-12-31 00:00:00",
    )
    #
    # Option B: Load your own CSV (must contain datetime + OHLCV)
    # df = pd.read_csv("btc_1m.csv", parse_dates=["dt"])
    # df = df.set_index(pd.DatetimeIndex(df["dt"], tz="UTC")).drop(columns=["dt"])

    # For demo without fetching, you must uncomment one option above.
    # raise SystemExit("Uncomment a data source (ccxt fetch or CSV load) and run again.")

    # 2) Run backtest
    cfg = BacktestConfig(
        atr_len=14,
        leverage=20.0,
        initial_capital=1000.0,
        fee_rate=0.0000,
        slippage=0.0000,
        sl_atr_mult=1.00,
        thr1=2.0,
        thr2=2.0,
        tp_atr_mult=2.00,
    )

    trades, df_bt, stats = backtest_atr_grinder(df, cfg)

    # 3) Output
    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== LAST 10 TRADES ===")
    print(trades.tail(10).to_string(index=False))

    # Save results
    trades.to_csv("trades.csv", index=False)
    df_bt.to_csv("backtest_series.csv")
    print("\nSaved: trades.csv, backtest_series.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATR 1m strategy")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    args = parser.parse_args()

    if args.mode == "live":
        live_cfg = LiveConfig()
        run_live(live_cfg)
    else:
        run_backtest()


if __name__ == "__main__":
    main()
