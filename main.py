"""
ATR-based 1-minute grinder (body trigger, leveraged ROI TP)

Rules:
- Compute ATR (Wilder) on 1m candles.
- If candle body >= thr1*ATR: enter opposite direction at next candle open.
  Take profit when leveraged ROI reaches tp1_roi.
- If candle body >= thr2*ATR: enter opposite direction at next candle open.
  Take profit when leveraged ROI reaches tp2_roi.
- Stop loss when leveraged ROI reaches -sl_roi.

Implementation details:
- Body = abs(close - open)
- Entry at next candle open (prevents lookahead bias)
- TP/SL are evaluated within each candle using high/low.
- Underlying move needed = ROI_target / leverage (applies to TP and SL).
- Fees and slippage are modeled.
- Signals are ignored for the first atr_warmup_bars (defaults to atr_len).
- Fixed margin per trade uses initial_capital (notional = leverage * margin).
- Stop loss is driven by sl_roi (leveraged ROI).
"""

from __future__ import annotations

import argparse
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
    tp1_roi: float = 0.005,   # 0.5% leveraged ROI target
    tp2_roi: float = 0.010,   # 1.0% leveraged ROI target
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      signal: +1 for long entry, -1 for short entry, 0 for none (signal generated on the candle close)
      tp_roi: ROI target for that signal (leveraged ROI), NaN if none

    Rule:
      body = abs(close-open)
      candle_dir = sign(close-open)
      entry_dir = -candle_dir (opposite)
      if body >= thr1*ATR => signal + tp1
      if body >= thr2*ATR => signal + tp2 (overwrites thr1 if both)
    """
    body = (df["close"] - df["open"]).abs()
    candle_dir = np.sign(df["close"] - df["open"]).astype(int)  # -1,0,+1

    entry_dir = -candle_dir  # opposite direction
    signal = pd.Series(0, index=df.index, dtype=int)
    tp_roi = pd.Series(np.nan, index=df.index, dtype=float)

    cond1 = body >= thr1 * atr
    cond2 = body >= thr2 * atr

    # 1.5x
    signal[cond1] = entry_dir[cond1]
    tp_roi[cond1] = tp1_roi

    # 2x (overwrite)
    signal[cond2] = entry_dir[cond2]
    tp_roi[cond2] = tp2_roi

    # If candle_dir is 0 (doji), entry_dir is 0 -> ignore
    signal[signal == 0] = 0
    tp_roi[signal == 0] = np.nan

    return signal, tp_roi


# -----------------------------
# Backtest engine
# -----------------------------
@dataclass
class BacktestConfig:
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    leverage: float = 10.0
    initial_capital: float = 400.0  # USD margin per trade

    # Costs
    fee_rate: float = 0.0000    # per side (0.04% typical maker/taker varies)
    slippage: float = 0.0000    # price impact fraction applied on entry/exit

    # Strategy thresholds
    thr1: float = 1.5
    thr2: float = 2.0
    tp1_roi: float = 0.0025
    tp2_roi: float = 0.0050

    # Risk/exit controls
    sl_roi: float = 0.010       # 1.0% leveraged ROI stop loss


@dataclass
class LiveConfig:
    symbol: str = "BTCUSDC"
    interval: str = "1m"
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    leverage: int = 20
    margin_usd: float = 35.0

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0
    tp1_roi: float = 0.010
    tp2_roi: float = 0.010
    sl_roi: float = 0.010

    # Execution controls
    entry_delay_min_seconds: float = 3.0
    entry_delay_max_seconds: float = 3.0
    spread_max_pct: float = 0.0001  # 0.01%
    atr_offset_mult: float = 0.02
    poll_interval_seconds: float = 1.0
    entry_order_timeout_seconds: float = 55.0
    log_path: str = "trade_log.jsonl"
    post_only: bool = True
    use_testnet: bool = False


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

    df["signal"], df["tp_roi"] = build_signals_body_opposite(
        df, df["atr"],
        thr1=cfg.thr1, thr2=cfg.thr2,
        tp1_roi=cfg.tp1_roi, tp2_roi=cfg.tp2_roi
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "tp_roi"] = np.nan

    # Equity curve (USD)
    equity = cfg.initial_capital
    equity_series = pd.Series(np.nan, index=df.index, dtype=float)
    equity_series.iloc[0] = equity

    margin_per_trade = cfg.initial_capital
    notional_per_trade = margin_per_trade * cfg.leverage

    position = 0  # +1 long, -1 short, 0 flat
    entry_price = None
    target_price = None
    stop_price = None
    entry_time = None
    entry_index = None
    used_tp_roi = None

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

    def net_roi_from_prices(entry_p: float, exit_p: float, side: int) -> float:
        """
        Compute net leveraged ROI after fees.
        Underlying return = side*(exit-entry)/entry
        Leveraged ROI approx = underlying_return * leverage
        Fees: charged per side on notional; approximate as fee_rate * leverage per side => 2*fee_rate*leverage
        """
        underlying_ret = side * ((exit_p - entry_p) / entry_p)
        gross_roi = underlying_ret * cfg.leverage
        fee_cost = 2.0 * cfg.fee_rate * cfg.leverage
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
                roi_net = net_roi_from_prices(entry_price, exit_price, position)
                pnl_net = roi_net * margin_per_trade
                equity += pnl_net

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": t,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "tp_roi_target": used_tp_roi,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "exit_reason": exit_reason,
                    "roi_net": roi_net,
                    "pnl_net": pnl_net,
                    "margin_used": margin_per_trade,
                    "notional": notional_per_trade,
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
                used_tp_roi = None

        # --- ENTRY logic (signal on prev candle, enter at this open) ---
        if position == 0:
            prev_signal = int(df.at[prev_t, "signal"])
            prev_tp_roi = df.at[prev_t, "tp_roi"]

            if prev_signal != 0 and not (isinstance(prev_tp_roi, float) and math.isnan(prev_tp_roi)):
                position = prev_signal
                entry_time = t
                entry_index = i
                used_tp_roi = float(prev_tp_roi)

                entry_price = apply_slippage(o, position, is_entry=True)

                # leveraged ROI targets -> underlying move required
                underlying_move_tp = used_tp_roi / cfg.leverage
                underlying_move_sl = cfg.sl_roi / cfg.leverage

                if position == 1:
                    target_price = entry_price * (1.0 + underlying_move_tp)
                    stop_price = entry_price * (1.0 - underlying_move_sl)
                else:
                    target_price = entry_price * (1.0 - underlying_move_tp)
                    stop_price = entry_price * (1.0 + underlying_move_sl)

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
    pending_tp_roi: Optional[float] = None
    pending_side: Optional[int] = None
    active_tp_roi: Optional[float] = None
    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    entry_close_ms: Optional[int] = None
    entry_price: Optional[float] = None
    entry_qty: Optional[float] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_event(log_path: str, event: Dict[str, object]) -> None:
    payload = dict(event)
    payload["ts"] = _utc_now_iso()
    with open(log_path, "a", encoding="ascii") as f:
        json.dump(payload, f, ensure_ascii=True)
        f.write("\n")


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
    signal, tp_roi = build_signals_body_opposite(
        df,
        atr,
        thr1=cfg.thr1,
        thr2=cfg.thr2,
        tp1_roi=cfg.tp1_roi,
        tp2_roi=cfg.tp2_roi,
    )

    signal_val = int(signal.iloc[-1])
    atr_val = float(atr.iloc[-1])
    tp_val = tp_roi.iloc[-1]
    if signal_val == 0 or pd.isna(tp_val) or math.isnan(atr_val):
        return 0, None, atr_val if not math.isnan(atr_val) else None
    return signal_val, float(tp_val), atr_val


def compute_tp_sl_prices(
    entry_price: float,
    side: int,
    tp_roi: float,
    sl_roi: float,
    leverage: float,
) -> Tuple[float, float]:
    move_tp = tp_roi / leverage
    move_sl = sl_roi / leverage
    if side == 1:
        target_price = entry_price * (1.0 + move_tp)
        stop_price = entry_price * (1.0 - move_sl)
    else:
        target_price = entry_price * (1.0 - move_tp)
        stop_price = entry_price * (1.0 + move_sl)
    return target_price, stop_price



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
    log_event(cfg.log_path, {"event": "startup", "symbol": cfg.symbol, "leverage": cfg.leverage, "margin_usd": cfg.margin_usd})

    while True:
        try:
            position = get_position_info(client, cfg.symbol)
            position_amt = float(position.get("positionAmt", 0.0))
            exit_filled = False

            # Manage open entry order
            if state.entry_order_id is not None:
                try:
                    order = client.query_order(symbol=cfg.symbol, orderId=state.entry_order_id)
                except ClientError:
                    log_event(cfg.log_path, {"event": "entry_order_query_error", "order_id": state.entry_order_id})
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.pending_tp_roi = None
                    state.pending_side = None
                    state.entry_close_ms = None
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
                    state.active_tp_roi = state.pending_tp_roi
                    state.pending_tp_roi = None
                    state.pending_side = None
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.entry_price = entry_price
                    state.entry_qty = qty

                    tp_price, sl_price = compute_tp_sl_prices(entry_price, side, state.active_tp_roi or cfg.tp1_roi, cfg.sl_roi, cfg.leverage)
                    tp_price_str = format_to_step(tp_price, filters["tick_size"])
                    sl_price_str = format_to_step(sl_price, filters["tick_size"])
                    qty_str = format_to_step(qty, filters["step_size"])

                    tp_side = "SELL" if side == 1 else "BUY"
                    sl_side = "SELL" if side == 1 else "BUY"
                    tp_order = limit_order(
                        cfg.symbol,
                        tp_side,
                        qty_str,
                        tp_price_str,
                        client,
                        time_in_force="GTC",
                        reduce_only=True,
                        client_order_id=f"ATR_TP_{entry_close_ms or ''}",
                    )
                    sl_order = stop_limit_order(
                        cfg.symbol,
                        sl_side,
                        qty_str,
                        sl_price_str,
                        sl_price_str,
                        client,
                        reduce_only=True,
                        client_order_id=f"ATR_SL_{entry_close_ms or ''}",
                    )
                    state.tp_order_id = tp_order.get("orderId") if tp_order else None
                    state.sl_order_id = sl_order.get("orderId") if sl_order else None
                    state.entry_close_ms = None

                    log_event(cfg.log_path, {
                        "event": "entry_filled",
                        "order_id": order.get("orderId"),
                        "side": "LONG" if side == 1 else "SHORT",
                        "entry_price": entry_price,
                        "quantity": qty,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                    })
                elif status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    log_event(cfg.log_path, {"event": "entry_order_closed", "order_id": state.entry_order_id, "status": status})
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.pending_tp_roi = None
                    state.pending_side = None
                    state.entry_close_ms = None
                else:
                    if state.entry_order_time and (time.time() - state.entry_order_time) > cfg.entry_order_timeout_seconds:
                        try:
                            client.cancel_order(symbol=cfg.symbol, orderId=state.entry_order_id)
                        except ClientError:
                            pass
                        log_event(cfg.log_path, {"event": "entry_order_timeout", "order_id": state.entry_order_id})
                        state.entry_order_id = None
                        state.entry_order_time = None
                        state.pending_tp_roi = None
                        state.pending_side = None
                        state.entry_close_ms = None

            # Check exit orders
            if state.tp_order_id is not None:
                try:
                    tp_order = client.query_order(symbol=cfg.symbol, orderId=state.tp_order_id)
                    tp_status = tp_order.get("status")
                    if tp_status == "FILLED":
                        exit_price = float(tp_order.get("avgPrice") or tp_order.get("price"))
                        log_event(cfg.log_path, {
                            "event": "exit_filled",
                            "exit_reason": "TP",
                            "order_id": state.tp_order_id,
                            "exit_price": exit_price,
                            "entry_price": state.entry_price,
                            "quantity": state.entry_qty,
                        })
                        if state.sl_order_id:
                            try:
                                client.cancel_order(symbol=cfg.symbol, orderId=state.sl_order_id)
                            except ClientError:
                                pass
                        state.tp_order_id = None
                        state.sl_order_id = None
                        state.active_tp_roi = None
                        state.entry_price = None
                        state.entry_qty = None
                        exit_filled = True
                    elif tp_status in {"CANCELED", "REJECTED", "EXPIRED"}:
                        state.tp_order_id = None
                except ClientError:
                    pass

            if state.sl_order_id is not None:
                try:
                    sl_order = client.query_order(symbol=cfg.symbol, orderId=state.sl_order_id)
                    sl_status = sl_order.get("status")
                    if sl_status == "FILLED":
                        exit_price = float(sl_order.get("avgPrice") or sl_order.get("price"))
                        log_event(cfg.log_path, {
                            "event": "exit_filled",
                            "exit_reason": "SL",
                            "order_id": state.sl_order_id,
                            "exit_price": exit_price,
                            "entry_price": state.entry_price,
                            "quantity": state.entry_qty,
                        })
                        if state.tp_order_id:
                            try:
                                client.cancel_order(symbol=cfg.symbol, orderId=state.tp_order_id)
                            except ClientError:
                                pass
                        state.tp_order_id = None
                        state.sl_order_id = None
                        state.active_tp_roi = None
                        state.entry_price = None
                        state.entry_qty = None
                        exit_filled = True
                    elif sl_status in {"CANCELED", "REJECTED", "EXPIRED"}:
                        state.sl_order_id = None
                except ClientError:
                    pass

            if exit_filled:
                time.sleep(cfg.poll_interval_seconds)
                continue

            # If position closed, cancel leftover exit orders
            if position_amt == 0.0 and (state.tp_order_id or state.sl_order_id):
                if state.tp_order_id:
                    try:
                        client.cancel_order(symbol=cfg.symbol, orderId=state.tp_order_id)
                    except ClientError:
                        pass
                if state.sl_order_id:
                    try:
                        client.cancel_order(symbol=cfg.symbol, orderId=state.sl_order_id)
                    except ClientError:
                        pass
                log_event(cfg.log_path, {"event": "position_closed"})
                state.tp_order_id = None
                state.sl_order_id = None
                state.active_tp_roi = None
                state.entry_price = None
                state.entry_qty = None

            # If in position and no exits placed, place them using current entryPrice
            if position_amt != 0.0 and (state.tp_order_id is None or state.sl_order_id is None):
                side = 1 if position_amt > 0 else -1
                entry_price = float(position.get("entryPrice", 0.0))
                qty = abs(position_amt)
                active_tp_roi = state.active_tp_roi or state.pending_tp_roi or cfg.tp1_roi
                state.active_tp_roi = active_tp_roi
                state.entry_price = entry_price
                state.entry_qty = qty

                tp_price, sl_price = compute_tp_sl_prices(entry_price, side, active_tp_roi, cfg.sl_roi, cfg.leverage)
                tp_price_str = format_to_step(tp_price, filters["tick_size"])
                sl_price_str = format_to_step(sl_price, filters["tick_size"])
                qty_str = format_to_step(qty, filters["step_size"])
                tp_side = "SELL" if side == 1 else "BUY"
                sl_side = "SELL" if side == 1 else "BUY"

                if state.tp_order_id is None:
                    tp_order = limit_order(cfg.symbol, tp_side, qty_str, tp_price_str, client, time_in_force="GTC", reduce_only=True)
                    state.tp_order_id = tp_order.get("orderId") if tp_order else None
                if state.sl_order_id is None:
                    sl_order = stop_limit_order(cfg.symbol, sl_side, qty_str, sl_price_str, sl_price_str, client, reduce_only=True)
                    state.sl_order_id = sl_order.get("orderId") if sl_order else None

            # New candle check
            klines = client.klines(symbol=cfg.symbol, interval=cfg.interval, limit=cfg.atr_len + 2)
            if len(klines) >= 2:
                closed_klines = klines[:-1]
                close_time_ms = int(closed_klines[-1][6])
                if close_time_ms != state.last_close_ms:
                    state.last_close_ms = close_time_ms
                    df = klines_to_df(closed_klines)
                    signal, tp_roi, atr_value = compute_live_signal(df, cfg)
                    candle = df.iloc[-1]
                    body = abs(float(candle["close"]) - float(candle["open"]))
                    log_event(cfg.log_path, {
                        "event": "candle_close",
                        "close_time_ms": close_time_ms,
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "body": body,
                        "atr": atr_value,
                        "signal": signal,
                        "tp_roi": tp_roi,
                    })

                    if position_amt == 0.0 and state.entry_order_id is None and signal != 0 and tp_roi is not None and atr_value is not None:
                        delay = cfg.entry_delay_min_seconds
                        if cfg.entry_delay_max_seconds > cfg.entry_delay_min_seconds:
                            delay = random.uniform(cfg.entry_delay_min_seconds, cfg.entry_delay_max_seconds)
                        time.sleep(delay)

                        bid, ask = get_book_ticker(client, cfg.symbol)
                        mid = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid if mid else 1.0
                        if spread_pct > cfg.spread_max_pct:
                            log_event(cfg.log_path, {"event": "skip_spread", "spread_pct": spread_pct})
                        else:
                            offset = atr_value * cfg.atr_offset_mult
                            if signal == 1:
                                limit_price = bid - offset
                                side = "BUY"
                            else:
                                limit_price = ask + offset
                                side = "SELL"

                            if limit_price <= 0:
                                log_event(cfg.log_path, {"event": "skip_price", "limit_price": limit_price})
                            else:
                                limit_price_str = format_to_step(limit_price, filters["tick_size"])
                                notional = cfg.margin_usd * cfg.leverage
                                qty = notional / limit_price
                                qty_str = format_to_step(qty, filters["step_size"])
                                if float(qty_str) < filters["min_qty"]:
                                    log_event(cfg.log_path, {"event": "skip_qty", "quantity": qty_str})
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
                                        state.pending_tp_roi = tp_roi
                                        state.pending_side = 1 if side == "BUY" else -1
                                        state.entry_close_ms = close_time_ms
                                        log_event(cfg.log_path, {
                                            "event": "entry_order",
                                            "order_id": state.entry_order_id,
                                            "side": side,
                                            "limit_price": float(limit_price_str),
                                            "quantity": float(qty_str),
                                            "tp_roi": tp_roi,
                                            "spread_pct": spread_pct,
                                            "offset": offset,
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
        start_utc="2025-10-01 00:00:00",
        end_utc="2025-12-30 00:00:00",
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
        initial_capital=40.0,
        fee_rate=0.0000,
        slippage=0.0001,
        sl_roi=0.005,
        thr1=2.0,
        thr2=2.0,
        tp1_roi=0.010,
        tp2_roi=0.010,
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
