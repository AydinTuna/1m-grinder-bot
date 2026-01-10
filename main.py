"""
ATR-based strategy (swing context + large body trigger, ATR-based SL + optional trailing stop)

Signal timeframe is configurable via signal_interval (default "1m", can be "15m", "1h", etc.)
Execution uses 1s candles for precise entry/exit simulation in backtest.

Rules:
- Compute ATR (EMA) on signal_interval candles.
- Compute swing highs/lows (default: 15m resample, fractal/pivot left/right).
- If candle body >= thr2*ATR and price is near a swing high/low:
  - Swing high: red => SHORT; green close below => SHORT; green close above => LONG (limit at mid-body)
  - Swing low: green => LONG; red close above => LONG; red close below => SHORT (limit at mid-body)
- If candle body >= thr2*ATR and price is not near swing levels: enter in the same direction as the candle.
- If use_trailing_stop is False:
  - Take profit when price moves tp_atr_mult * ATR from entry.
  - Stop loss when price moves sl_atr_mult * ATR from entry.
- If use_trailing_stop is True:
  - No take profit.
  - Initial stop is 1R where R = sl_atr_mult * ATR_at_entry.
  - Trailing stop is a ladder based on candle close in R units (gap = trail_gap_r, buffer = trail_buffer_r).

Implementation details:
- Body = abs(close - open)
- Market entries are at next candle open; mid-body signals use a limit order (midpoint of signal candle body).
- Stops are evaluated within each candle using high/low; trailing stop updates occur on candle close.
- ATR uses the signal candle's ATR (frozen at entry for R-based trailing).
- Fees and slippage are modeled.
- Signals are ignored for the first atr_warmup_bars (defaults to atr_len).
- Margin per trade is capped by initial_capital/margin_usd; leverage may be adjusted from target loss to keep TP/SL $ consistency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import http.client
import json
import logging
import math
import os
from os import getenv
import random
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from config import BacktestConfig, LiveConfig, LIVE_TRADE_FIELDS
from swing_levels import build_swing_atr_signals, compute_confirmed_swing_levels

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
def atr_ema(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    ATR using EMA smoothing (alpha=2/(length+1)) with SMA seed.
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

    atr = pd.Series(np.nan, index=tr.index, dtype=float)
    if length <= 0 or len(tr) < length:
        return atr

    alpha = 2.0 / (length + 1.0)
    atr.iloc[length - 1] = tr.iloc[:length].mean()
    for i in range(length, len(tr)):
        atr.iloc[i] = (tr.iloc[i] - atr.iloc[i - 1]) * alpha + atr.iloc[i - 1]

    return atr


# -----------------------------
# Strategy signals
# -----------------------------
def build_signals_body_opposite(
    df: pd.DataFrame,
    atr: pd.Series,
    thr1: float = 1.5,
    thr2: float = 2.0,
    tolerance_pct: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      signal: +1 for long entry, -1 for short entry, 0 for none (signal generated on the candle close)
      signal_atr: ATR value on the signal candle, NaN if none

    Rule:
      body = abs(close-open)
      candle_dir = sign(close-open)
      entry_dir = -candle_dir (opposite)
      if tolerance_pct > 0:
        body >= thr*ATR*(1-tol)
      else:
        body >= thr*ATR
      if body meets thr1 => signal
      if body meets thr2 => signal (overwrites thr1 if both)
    """
    body = (df["close"] - df["open"]).abs()
    candle_dir = np.sign(df["close"] - df["open"]).astype(int)  # -1,0,+1

    entry_dir = -candle_dir  # opposite direction
    signal = pd.Series(0, index=df.index, dtype=int)
    signal_atr = pd.Series(np.nan, index=df.index, dtype=float)

    tol = max(0.0, float(tolerance_pct or 0.0))
    target1 = thr1 * atr
    target2 = thr2 * atr
    if tol > 0.0:
        target1 = target1 * (1.0 - tol)
        target2 = target2 * (1.0 - tol)
    cond1 = body >= target1
    cond2 = body >= target2

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

    df["atr"] = atr_ema(df, cfg.atr_len)

    swing_high, swing_low = compute_confirmed_swing_levels(
        df,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["swing_high_level"] = swing_high
    df["swing_low_level"] = swing_low

    df["signal"], df["signal_atr"], df["signal_entry_price"] = build_swing_atr_signals(
        df,
        df["atr"],
        df["swing_high_level"],
        df["swing_low_level"],
        body_atr_mult=cfg.thr2,
        swing_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "signal_atr"] = np.nan
        df.loc[warmup_idx, "signal_entry_price"] = np.nan

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
    trail_r_value: Optional[float] = None
    trail_best_close_r: Optional[float] = None
    trail_stop_r: Optional[int] = None
    entry_time = None
    entry_index = None
    used_signal_atr = None
    pending_side = 0  # +1 long limit, -1 short limit, 0 none
    pending_limit_price = None
    pending_signal_atr = None
    pending_created_index = None

    trades: List[Dict] = []
    trailing_stop_updates: List[Dict] = []

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

    def maybe_fill_limit(side: int, limit_price: float, o_: float, h_: float, l_: float) -> Optional[float]:
        if side == 1:
            if o_ <= limit_price:
                return o_
            if l_ <= limit_price <= h_:
                return limit_price
        elif side == -1:
            if o_ >= limit_price:
                return o_
            if l_ <= limit_price <= h_:
                return limit_price
        return None

    limit_timeout_bars = max(1, int(cfg.entry_limit_timeout_bars))

    for i in range(1, len(df)):
        t = idx[i]
        prev_t = idx[i - 1]

        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # --- EXIT logic ---
        if position != 0 and entry_price is not None and stop_price is not None:
            sl_hit = (l <= stop_price) if position == 1 else (h >= stop_price)

            exited = False
            exit_reason = None
            exit_price = None

            if cfg.use_trailing_stop:
                if sl_hit:
                    exit_reason = "SL"
                    exit_price = apply_slippage(stop_price, position, is_entry=False)
                    exited = True
            else:
                if target_price is not None:
                    tp_hit = (h >= target_price) if position == 1 else (l <= target_price)
                else:
                    tp_hit = False

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
                # Cap SL loss at target_loss_usd to match live exchange behavior
                if exit_reason == "SL" and cfg.target_loss_usd is not None and pnl_net < -cfg.target_loss_usd:
                    pnl_net = -cfg.target_loss_usd
                    roi_net = pnl_net / active_margin if active_margin else roi_net
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
                trail_r_value = None
                trail_best_close_r = None
                trail_stop_r = None
                entry_time = None
                entry_index = None
                used_signal_atr = None
                active_margin = cfg.initial_capital
                active_leverage = cfg.leverage
                active_notional = active_margin * active_leverage

        # --- Trailing stop update (candle close) ---
        if (
            cfg.use_trailing_stop
            and position != 0
            and entry_price is not None
            and stop_price is not None
            and trail_r_value is not None
            and trail_best_close_r is not None
            and trail_stop_r is not None
        ):
            c = float(df.at[t, "close"])
            close_r = compute_close_r(entry_price, c, position, trail_r_value)
            if close_r is not None:
                side_str = "LONG" if position == 1 else "SHORT"
                prev_best_r = trail_best_close_r
                prev_stop_r = trail_stop_r
                prev_stop_price = stop_price
                
                trail_best_close_r = max(trail_best_close_r, close_r)
                next_stop_r = compute_trailing_stop_r(trail_best_close_r, trail_stop_r, cfg.trail_gap_r)
                
                # Log trailing stop evaluation
                if next_stop_r > trail_stop_r:
                    # Stop is moving - calculate new price
                    new_stop_price = compute_trailing_sl_price(
                        entry_price,
                        position,
                        trail_r_value,
                        next_stop_r,
                        cfg.trail_buffer_r,
                    )
                    
                    # Log to CSV
                    trailing_stop_updates.append({
                        "timestamp": t,
                        "entry_time": entry_time,
                        "side": side_str,
                        "entry_price": entry_price,
                        "close_price": c,
                        "close_r": close_r,
                        "prev_best_r": prev_best_r,
                        "new_best_r": trail_best_close_r,
                        "prev_stop_r": prev_stop_r,
                        "new_stop_r": next_stop_r,
                        "prev_stop_price": prev_stop_price,
                        "new_stop_price": new_stop_price,
                        "r_value": trail_r_value,
                        "trail_gap_r": cfg.trail_gap_r,
                        "trail_buffer_r": cfg.trail_buffer_r,
                        "stop_moved": True,
                    })
                    
                    trail_stop_r = next_stop_r
                    stop_price = new_stop_price
                else:
                    # Log to CSV
                    trailing_stop_updates.append({
                        "timestamp": t,
                        "entry_time": entry_time,
                        "side": side_str,
                        "entry_price": entry_price,
                        "close_price": c,
                        "close_r": close_r,
                        "prev_best_r": trail_best_close_r,
                        "new_best_r": trail_best_close_r,
                        "prev_stop_r": trail_stop_r,
                        "new_stop_r": trail_stop_r,
                        "prev_stop_price": stop_price,
                        "new_stop_price": stop_price,
                        "r_value": trail_r_value,
                        "trail_gap_r": cfg.trail_gap_r,
                        "trail_buffer_r": cfg.trail_buffer_r,
                        "stop_moved": False,
                    })

        # --- ENTRY logic (signal on prev candle; optional mid-body limit) ---
        if position == 0:
            # Cancel expired pending limit order before attempting fills.
            if pending_side != 0 and pending_created_index is not None:
                if (i - pending_created_index) >= limit_timeout_bars:
                    pending_side = 0
                    pending_limit_price = None
                    pending_signal_atr = None
                    pending_created_index = None

            # Try to fill an existing pending limit order first.
            if pending_side != 0 and pending_limit_price is not None and pending_signal_atr is not None:
                fill_raw = maybe_fill_limit(int(pending_side), float(pending_limit_price), o, h, l)
                if fill_raw is not None:
                    side = int(pending_side)
                    signal_atr_value = float(pending_signal_atr)
                    entry_price = apply_slippage(float(fill_raw), side, is_entry=True)

                    trade_margin, trade_leverage, _ = resolve_trade_sizing(
                        entry_price=entry_price,
                        atr_value=signal_atr_value,
                        sl_atr_mult=cfg.sl_atr_mult,
                        margin_cap=cfg.initial_capital,
                        max_leverage=cfg.leverage,
                        min_leverage=cfg.min_leverage,
                        target_loss_usd=cfg.target_loss_usd,
                    )
                    if trade_margin is None or trade_leverage is None:
                        pending_side = 0
                        pending_limit_price = None
                        pending_signal_atr = None
                        pending_created_index = None
                    else:
                        if cfg.use_trailing_stop:
                            target_price = None
                            trail_r_value = signal_atr_value * cfg.sl_atr_mult
                            trail_best_close_r = 0.0
                            trail_stop_r = -1
                            stop_price = compute_trailing_sl_price(
                                entry_price,
                                side,
                                trail_r_value,
                                trail_stop_r,
                                cfg.trail_buffer_r,
                            )
                        else:
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

                        pending_side = 0
                        pending_limit_price = None
                        pending_signal_atr = None
                        pending_created_index = None

            # No pending order (or not filled yet): check the previous candle signal.
            if position == 0 and pending_side == 0:
                prev_signal = int(df.at[prev_t, "signal"])
                prev_signal_atr = df.at[prev_t, "signal_atr"]
                prev_entry_price = df.at[prev_t, "signal_entry_price"]

                if prev_signal != 0 and not (isinstance(prev_signal_atr, float) and math.isnan(prev_signal_atr)):
                    side = prev_signal
                    signal_atr_value = float(prev_signal_atr)

                    if prev_entry_price is not None and not (
                        isinstance(prev_entry_price, float) and math.isnan(prev_entry_price)
                    ):
                        pending_side = side
                        pending_limit_price = float(prev_entry_price)
                        pending_signal_atr = signal_atr_value
                        pending_created_index = i

                        fill_raw = maybe_fill_limit(side, float(prev_entry_price), o, h, l)
                        if fill_raw is None:
                            pass
                        else:
                            entry_price = apply_slippage(float(fill_raw), side, is_entry=True)

                            trade_margin, trade_leverage, _ = resolve_trade_sizing(
                                entry_price=entry_price,
                                atr_value=signal_atr_value,
                                sl_atr_mult=cfg.sl_atr_mult,
                                margin_cap=cfg.initial_capital,
                                max_leverage=cfg.leverage,
                                min_leverage=cfg.min_leverage,
                                target_loss_usd=cfg.target_loss_usd,
                            )
                            if trade_margin is None or trade_leverage is None:
                                pending_side = 0
                                pending_limit_price = None
                                pending_signal_atr = None
                                pending_created_index = None
                            else:
                                if cfg.use_trailing_stop:
                                    target_price = None
                                    trail_r_value = signal_atr_value * cfg.sl_atr_mult
                                    trail_best_close_r = 0.0
                                    trail_stop_r = -1
                                    stop_price = compute_trailing_sl_price(
                                        entry_price,
                                        side,
                                        trail_r_value,
                                        trail_stop_r,
                                        cfg.trail_buffer_r,
                                    )
                                else:
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

                                pending_side = 0
                                pending_limit_price = None
                                pending_signal_atr = None
                                pending_created_index = None

                    else:
                        entry_price = apply_slippage(o, side, is_entry=True)

                        trade_margin, trade_leverage, _ = resolve_trade_sizing(
                            entry_price=entry_price,
                            atr_value=signal_atr_value,
                            sl_atr_mult=cfg.sl_atr_mult,
                            margin_cap=cfg.initial_capital,
                            max_leverage=cfg.leverage,
                            min_leverage=cfg.min_leverage,
                            target_loss_usd=cfg.target_loss_usd,
                        )
                        if trade_margin is None or trade_leverage is None:
                            continue

                        if cfg.use_trailing_stop:
                            target_price = None
                            trail_r_value = signal_atr_value * cfg.sl_atr_mult
                            trail_best_close_r = 0.0
                            trail_stop_r = -1
                            stop_price = compute_trailing_sl_price(
                                entry_price,
                                side,
                                trail_r_value,
                                trail_stop_r,
                                cfg.trail_buffer_r,
                            )
                        else:
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
    trailing_df = pd.DataFrame(trailing_stop_updates)
    stats = compute_stats(trades_df, df, cfg)

    return trades_df, df, stats, trailing_df


def compute_stats(trades_df: pd.DataFrame, df_bt: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "initial_capital": float(cfg.initial_capital),
            "target_loss_usd": float(cfg.target_loss_usd) if cfg.target_loss_usd is not None else 0.0,
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
        "initial_capital": float(cfg.initial_capital),
        "target_loss_usd": float(cfg.target_loss_usd) if cfg.target_loss_usd is not None else 0.0,
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
    active_symbol: Optional[str] = None
    last_close_ms: Optional[int] = None
    entry_order_id: Optional[int] = None
    entry_order_time: Optional[float] = None
    pending_atr: Optional[float] = None
    pending_side: Optional[int] = None
    active_atr: Optional[float] = None
    tp_order_id: Optional[int] = None
    tp_client_order_id: Optional[str] = None
    sl_order_id: Optional[int] = None
    sl_client_order_id: Optional[str] = None
    sl_algo_id: Optional[int] = None
    sl_algo_client_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    trail_r_value: Optional[float] = None
    trail_best_close_r: Optional[float] = None
    trail_stop_r: Optional[int] = None
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
    sl_triggered: bool = False
    sl_order_time: Optional[float] = None


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


def is_btc_symbol(symbol: str) -> bool:
    return symbol.upper().startswith("BTC")


def format_float_by_symbol(value: Optional[float], symbol: str) -> Optional[float]:
    if is_btc_symbol(symbol):
        return format_float_1(value)
    return format_float_2(value)


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


def format_to_step(value: float, step: float, rounding=ROUND_DOWN) -> str:
    step_dec = Decimal(str(step))
    if step_dec == 0:
        return format(Decimal(str(value)), "f")
    quant = (Decimal(str(value)) / step_dec).to_integral_value(rounding=rounding) * step_dec
    return format(quant, "f")


def format_price_to_tick(value: float, tick_size: float, side: str, post_only: bool) -> str:
    rounding = ROUND_DOWN
    if post_only and str(side).upper() == "SELL":
        rounding = ROUND_UP
    return format_to_step(value, tick_size, rounding=rounding)


def reprice_post_only(price: float, side: str, bid: float, ask: float, tick_size: float) -> float:
    """
    Keeps a post-only LIMIT order outside the spread by at least 1 tick.
    BUY: price <= bid - tick
    SELL: price >= ask + tick
    """
    if tick_size <= 0:
        return price
    side_u = str(side).upper()
    if side_u == "BUY":
        return min(price, bid - tick_size)
    if side_u == "SELL":
        return max(price, ask + tick_size)
    return price


class ClientError(Exception):
    def __init__(
        self,
        status_code: Optional[int],
        error_code: Optional[int],
        error_message: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_message = error_message
        self.headers = headers or {}
        super().__init__(
            f"Binance API error. status: {status_code}, code: {error_code}, message: {error_message}"
        )


class BinanceUMFuturesREST:
    def __init__(
        self,
        key: str,
        secret: str,
        base_url: str = "https://fapi.binance.com",
        timeout_seconds: float = 10.0,
        *,
        recv_window_ms: int = 60_000,
        time_sync_interval_seconds: float = 30.0 * 60.0,
        ssl_verify: bool = True,
        ca_bundle_path: Optional[str] = None,
    ) -> None:
        self.key = key
        self.secret = secret
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.recv_window_ms = int(recv_window_ms)
        self.time_sync_interval_seconds = float(time_sync_interval_seconds)
        self.ssl_verify = bool(ssl_verify)
        self.ca_bundle_path = str(ca_bundle_path) if ca_bundle_path else None
        self.time_offset_ms = 0
        self._last_time_sync_monotonic: Optional[float] = None
        self._last_time_sync_attempt_monotonic: Optional[float] = None

    def _build_ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        Build an SSL context for HTTPS requests.

        Why: some environments (notably macOS + certain Python builds, or corporate proxies)
        can fail certificate verification unless a CA bundle is explicitly provided.
        """
        if not self.ssl_verify:
            return ssl._create_unverified_context()

        # Explicit CA bundle wins.
        if self.ca_bundle_path:
            return ssl.create_default_context(cafile=self.ca_bundle_path)

        # Otherwise, try certifi if installed.
        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            # Fall back to system defaults.
            return ssl.create_default_context()

    @staticmethod
    def _clean_params(params: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in (params or {}).items() if v is not None}

    @staticmethod
    def _encode_params(params: Dict[str, object], special: bool = False) -> str:
        encoded = urllib.parse.urlencode(params, doseq=True)
        if special:
            encoded = encoded.replace("%40", "@").replace("%27", "%22")
        else:
            encoded = encoded.replace("%40", "@")
        return encoded

    def _sign(self, payload: str) -> str:
        return hmac.new(
            str(self.secret).encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _signed_timestamp_ms(self) -> int:
        return self._now_ms() + int(self.time_offset_ms)

    def get_server_time(self) -> Any:
        return self._request("GET", "/fapi/v1/time")

    def sync_time(self) -> None:
        """
        Fetch Binance server time and compute a local offset used for signed requests.
        """
        start_ms = self._now_ms()
        server = self.get_server_time()
        end_ms = self._now_ms()
        try:
            server_ms = int(server["serverTime"])
        except Exception as exc:
            raise RuntimeError(f"Unexpected server time response: {server!r}") from exc
        local_mid_ms = (start_ms + end_ms) // 2
        self.time_offset_ms = server_ms - local_mid_ms
        self._last_time_sync_monotonic = time.monotonic()

    def _maybe_sync_time(self) -> None:
        if self.time_sync_interval_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_time_sync_monotonic is None:
            # Avoid spamming retries if the network is down.
            if (
                self._last_time_sync_attempt_monotonic is not None
                and now - self._last_time_sync_attempt_monotonic < 5.0
            ):
                return
            self._last_time_sync_attempt_monotonic = now
            self.sync_time()
            return
        if now - self._last_time_sync_monotonic >= self.time_sync_interval_seconds:
            self._last_time_sync_attempt_monotonic = now
            self.sync_time()

    def _request(
        self,
        http_method: str,
        url_path: str,
        payload: Optional[Dict[str, object]] = None,
        *,
        signed: bool = False,
        special: bool = False,
    ) -> Any:
        def do_call(url: str) -> str:
            request = urllib.request.Request(url, method=str(http_method).upper())
            if self.key:
                request.add_header("X-MBX-APIKEY", self.key)
            request.add_header("Content-Type", "application/json;charset=utf-8")

            try:
                ctx = self._build_ssl_context()
                with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=ctx) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as error:
                try:
                    raw_error = error.read().decode("utf-8")
                except Exception:
                    raw_error = str(error)
                error_code = None
                error_message = raw_error
                try:
                    parsed = json.loads(raw_error)
                    if isinstance(parsed, dict):
                        error_code = parsed.get("code")
                        error_message = parsed.get("msg") or error_message
                except Exception:
                    pass
                raise ClientError(
                    status_code=getattr(error, "code", None),
                    error_code=error_code,
                    error_message=str(error_message),
                    headers=dict(getattr(error, "headers", {}) or {}),
                ) from None
            except urllib.error.URLError as error:
                raise ClientError(None, None, str(error)) from None
            except http.client.IncompleteRead as error:
                partial_len = 0
                try:
                    partial_len = len(error.partial or b"")
                except Exception:
                    partial_len = 0
                expected = getattr(error, "expected", None)
                raise ClientError(
                    None,
                    None,
                    f"Incomplete response read (partial={partial_len} expected={expected})",
                ) from None
            except (socket.timeout, TimeoutError) as error:
                raise ClientError(None, None, f"Request timeout: {error}") from None
            except ssl.SSLError as error:
                raise ClientError(None, None, f"SSL read error: {error}") from None
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as error:
                raise ClientError(None, None, f"Connection dropped: {error}") from None
            except OSError as error:
                raise ClientError(None, None, f"Network error: {type(error).__name__}: {error}") from None

        http_method_upper = str(http_method).upper()
        retryable_method = http_method_upper in {"GET", "HEAD"}
        max_attempts = 2 if signed else 1
        if retryable_method:
            max_attempts = max(max_attempts, 3)
        last_error: Optional[ClientError] = None

        for attempt in range(1, max_attempts + 1):
            params = self._clean_params(payload or {})
            query_string = self._encode_params(params, special=special) if params else ""
            if signed:
                try:
                    self._maybe_sync_time()
                except Exception:
                    # Continue without a time offset; we'll retry on -1021 if needed.
                    pass

                params = dict(params)
                if "recvWindow" not in params and self.recv_window_ms > 0:
                    params["recvWindow"] = max(1, min(int(self.recv_window_ms), 60_000))
                params["timestamp"] = self._signed_timestamp_ms()
                unsigned = self._encode_params(params, special=special)
                signature = self._sign(unsigned)
                query_string = f"{unsigned}&signature={signature}"

            url = f"{self.base_url}{url_path}"
            if query_string:
                url = f"{url}?{query_string}"

            try:
                raw = do_call(url)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            except ClientError as error:
                last_error = error
                if (
                    signed
                    and attempt < max_attempts
                    and getattr(error, "error_code", None) == -1021
                ):
                    try:
                        self.sync_time()
                    except Exception:
                        pass
                    continue
                if retryable_method and attempt < max_attempts:
                    status_code = getattr(error, "status_code", None)
                    error_code = getattr(error, "error_code", None)
                    retryable_status = status_code is None or status_code in {418, 429, 500, 502, 503, 504}
                    retryable_code = error_code in {-1001, -1003, -1015}
                    if retryable_status or retryable_code:
                        base_backoff = 0.25
                        sleep_seconds = min(2.0, base_backoff * (2 ** (attempt - 1)))
                        retry_after = None
                        if status_code in {418, 429}:
                            retry_after_value = (error.headers or {}).get("Retry-After") or (error.headers or {}).get(
                                "retry-after"
                            )
                            if retry_after_value is not None:
                                try:
                                    retry_after = float(retry_after_value)
                                except (TypeError, ValueError):
                                    retry_after = None
                        if retry_after is not None:
                            sleep_seconds = max(sleep_seconds, min(60.0, retry_after))
                        time.sleep(sleep_seconds + random.uniform(0.0, 0.15))
                        continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected request failure without an error.")

    def exchange_info(self, **kwargs: object) -> Any:
        _ = kwargs
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/klines", payload=dict(kwargs))

    def book_ticker(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/ticker/bookTicker", payload=dict(kwargs))

    def get_position_risk(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v3/positionRisk", payload=dict(kwargs), signed=True)

    def change_leverage(self, **kwargs: object) -> Any:
        return self._request("POST", "/fapi/v1/leverage", payload=dict(kwargs), signed=True)

    def new_order(self, **kwargs: object) -> Any:
        return self._request("POST", "/fapi/v1/order", payload=dict(kwargs), signed=True)

    def query_order(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/order", payload=dict(kwargs), signed=True)

    def cancel_order(self, **kwargs: object) -> Any:
        return self._request("DELETE", "/fapi/v1/order", payload=dict(kwargs), signed=True)

    def get_orders(self, **kwargs: object) -> Any:
        # GET /fapi/v1/openOrders (no orderId required)
        return self._request("GET", "/fapi/v1/openOrders", payload=dict(kwargs), signed=True)

    def cancel_open_orders(self, **kwargs: object) -> Any:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", payload=dict(kwargs), signed=True)

    # Algo orders (STOP/TP conditional style endpoints)
    def new_algo_order(self, **kwargs: object) -> Any:
        return self._request("POST", "/fapi/v1/algoOrder", payload=dict(kwargs), signed=True)

    def cancel_algo_order(self, **kwargs: object) -> Any:
        return self._request("DELETE", "/fapi/v1/algoOrder", payload=dict(kwargs), signed=True)

    def cancel_open_algo_orders(self, **kwargs: object) -> Any:
        return self._request("DELETE", "/fapi/v1/algoOpenOrders", payload=dict(kwargs), signed=True)

    def query_algo_order(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/algoOrder", payload=dict(kwargs), signed=True)

    def open_algo_orders(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/openAlgoOrders", payload=dict(kwargs), signed=True)

    def all_algo_orders(self, **kwargs: object) -> Any:
        return self._request("GET", "/fapi/v1/allAlgoOrders", payload=dict(kwargs), signed=True)


def get_um_futures_client(cfg: LiveConfig) -> BinanceUMFuturesREST:
    if load_dotenv is not None:
        load_dotenv()
    api_key = getenv("BINANCE_API_KEY")
    api_secret = getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_API_SECRET in environment.")
    base_url = "https://fapi.binance.com"
    if cfg.use_testnet:
        base_url = "https://testnet.binancefuture.com"
    ca_bundle_path = getenv("BINANCE_CA_BUNDLE") or getenv("SSL_CERT_FILE") or getenv("REQUESTS_CA_BUNDLE")
    ssl_verify_env = (getenv("BINANCE_SSL_VERIFY") or "").strip().lower()
    ssl_verify = True
    if ssl_verify_env in {"0", "false", "no", "off"}:
        ssl_verify = False
    recv_window_ms = getenv("BINANCE_RECV_WINDOW_MS")
    time_sync_interval_seconds = getenv("BINANCE_TIME_SYNC_INTERVAL_SECONDS")
    recv_window_value = 60_000
    if recv_window_ms:
        try:
            recv_window_value = int(recv_window_ms)
        except ValueError:
            logging.warning("Invalid BINANCE_RECV_WINDOW_MS=%r; using default.", recv_window_ms)
    time_sync_interval_value = 30.0 * 60.0
    if time_sync_interval_seconds:
        try:
            time_sync_interval_value = float(time_sync_interval_seconds)
        except ValueError:
            logging.warning(
                "Invalid BINANCE_TIME_SYNC_INTERVAL_SECONDS=%r; using default.",
                time_sync_interval_seconds,
            )
    client = BinanceUMFuturesREST(
        key=api_key,
        secret=api_secret,
        base_url=base_url,
        recv_window_ms=recv_window_value,
        time_sync_interval_seconds=time_sync_interval_value,
        ssl_verify=ssl_verify,
        ca_bundle_path=ca_bundle_path,
    )
    try:
        client.sync_time()
    except Exception as exc:
        logging.warning("Binance time sync failed; signed requests may fail (-1021). error=%s", exc)
    return client


def get_um_futures_public_client(base_url: str = "https://fapi.binance.com") -> BinanceUMFuturesREST:
    """
    Return an unauthenticated UM Futures REST client for public endpoints (e.g. /fapi/v1/klines).

    This intentionally does NOT require API keys, because public market data endpoints do not
    need signed requests. Signed endpoints (orders/positions/etc) must still use
    `get_um_futures_client()` in live trading.
    """
    ca_bundle_path = getenv("BINANCE_CA_BUNDLE") or getenv("SSL_CERT_FILE") or getenv("REQUESTS_CA_BUNDLE")
    ssl_verify_env = (getenv("BINANCE_SSL_VERIFY") or "").strip().lower()
    ssl_verify = True
    if ssl_verify_env in {"0", "false", "no", "off"}:
        ssl_verify = False
    client = BinanceUMFuturesREST(key="", secret="", base_url=base_url, ssl_verify=ssl_verify, ca_bundle_path=ca_bundle_path)
    return client


def get_symbol_filters(client: BinanceUMFuturesREST, symbol: str) -> Dict[str, float]:
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


def get_book_ticker(client: BinanceUMFuturesREST, symbol: str) -> Tuple[float, float]:
    book = client.book_ticker(symbol=symbol)
    return float(book["bidPrice"]), float(book["askPrice"])


def get_position_info(client: BinanceUMFuturesREST, symbol: str) -> Dict[str, str]:
    info = client.get_position_risk(symbol=symbol)
    if isinstance(info, list):
        for pos in info:
            if pos.get("symbol") == symbol:
                return pos
        return info[0] if info else {}
    return info


def has_open_position(client: BinanceUMFuturesREST, symbol: str) -> bool:
    """Check if there's an open position for the symbol."""
    try:
        pos = get_position_info(client, symbol)
        amt = float((pos or {}).get("positionAmt", 0.0))
        return amt != 0.0
    except Exception:
        return False


def limit_order(
    symbol: str,
    side: str,
    quantity: str,
    price: str,
    client: BinanceUMFuturesREST,
    client_order_id: Optional[str] = None,
    tick_size: Optional[float] = None,
    post_only_max_reprices: int = 3,
    reduce_only: bool = False,
) -> Optional[Dict[str, object]]:
    """
    Places a post-only limit order on Binance Futures.
    """
    params: Dict[str, object] = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "quantity": quantity,
        "timeInForce": "GTX",
        "price": price,
    }
    if client_order_id:
        params["newClientOrderId"] = client_order_id
    if reduce_only:
        params["reduceOnly"] = True

    post_only_reprices = 0
    reduce_only_retries = 0
    reduce_only_max_retries = 8

    while True:
        try:
            return client.new_order(**params)
        except ClientError as error:
            error_code = getattr(error, "error_code", None)
            status_code = getattr(error, "status_code", None)
            error_message = getattr(error, "error_message", None)

            if reduce_only and error_code == -2022:
                reduce_only_retries += 1
                if reduce_only_retries > reduce_only_max_retries:
                    logging.error(
                        "Reduce-only order rejected (max retries). status: %s, code: %s, message: %s",
                        status_code,
                        error_code,
                        error_message,
                    )
                    return None

                try:
                    refreshed_position = get_position_info(client, symbol)
                except ClientError as pos_error:
                    logging.error(
                        "Reduce-only order rejected and position refresh failed. status: %s, code: %s, message: %s",
                        getattr(pos_error, "status_code", None),
                        getattr(pos_error, "error_code", None),
                        getattr(pos_error, "error_message", None),
                    )
                    return None

                try:
                    position_amt = float((refreshed_position or {}).get("positionAmt", 0.0))
                except (TypeError, ValueError):
                    position_amt = 0.0

                if position_amt == 0.0:
                    time.sleep(0.2)
                    continue

                expected_side = "SELL" if position_amt > 0 else "BUY"
                if str(side).upper() != expected_side:
                    logging.error(
                        "Reduce-only order rejected; side would not reduce position. symbol=%s order_side=%s position_amt=%s",
                        symbol,
                        side,
                        position_amt,
                    )
                    return None

                try:
                    requested_qty = float(params.get("quantity") or 0.0)
                except (TypeError, ValueError):
                    requested_qty = 0.0
                max_qty = abs(position_amt)

                if requested_qty > max_qty:
                    try:
                        filters = get_symbol_filters(client, symbol)
                        step_size = float(filters["step_size"])
                        min_qty = float(filters["min_qty"])
                    except Exception:
                        step_size = 0.0
                        min_qty = 0.0
                    if step_size > 0:
                        new_qty_str = format_to_step(max_qty, step_size, rounding=ROUND_DOWN)
                    else:
                        new_qty_str = str(max_qty)
                    try:
                        new_qty_num = float(new_qty_str)
                    except (TypeError, ValueError):
                        new_qty_num = 0.0
                    if min_qty > 0 and new_qty_num < min_qty:
                        logging.info(
                            "Reduce-only order rejected; position below min qty. symbol=%s position_amt=%s min_qty=%s",
                            symbol,
                            position_amt,
                            min_qty,
                        )
                        return None
                    if new_qty_str != params.get("quantity"):
                        logging.info(
                            "Reduce-only order rejected; clamping quantity and retrying. symbol=%s qty=%s -> %s position_amt=%s",
                            symbol,
                            params.get("quantity"),
                            new_qty_str,
                            position_amt,
                        )
                        params["quantity"] = new_qty_str

                time.sleep(0.2)
                continue

            if error_code != -5022:
                logging.error(
                    "Order error. status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            # Post-only: reprice to stay maker and retry.
            post_only_reprices += 1
            if post_only_reprices > max(0, int(post_only_max_reprices or 0)):
                logging.info(
                    "Post-only order rejected (max reprices). status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            resolved_tick = tick_size
            if resolved_tick is None:
                try:
                    resolved_tick = get_symbol_filters(client, symbol).get("tick_size")
                except Exception:
                    resolved_tick = None
            if resolved_tick is None or resolved_tick <= 0:
                logging.info(
                    "Post-only order rejected (missing tick_size). status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            try:
                bid, ask = get_book_ticker(client, symbol)
            except Exception:
                logging.info(
                    "Post-only order rejected (book ticker unavailable). status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            try:
                current_price = float(params.get("price") or 0.0)
            except (TypeError, ValueError):
                current_price = 0.0
            if current_price <= 0:
                logging.info(
                    "Post-only order rejected (invalid price). status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            new_price = reprice_post_only(current_price, side, bid, ask, float(resolved_tick))
            if new_price <= 0:
                logging.info(
                    "Post-only order rejected (invalid reprice). status: %s, code: %s, message: %s",
                    status_code,
                    error_code,
                    error_message,
                )
                return None

            rounding = ROUND_DOWN if str(side).upper() == "BUY" else ROUND_UP
            new_price_str = format_to_step(new_price, float(resolved_tick), rounding=rounding)
            if new_price_str == params.get("price"):
                nudge = float(resolved_tick)
                if str(side).upper() == "BUY":
                    new_price_str = format_to_step(max(0.0, new_price - nudge), nudge, rounding=ROUND_DOWN)
                else:
                    new_price_str = format_to_step(new_price + nudge, nudge, rounding=ROUND_UP)

            logging.info(
                "Post-only order rejected, repricing and retrying. symbol=%s side=%s price=%s -> %s bid=%s ask=%s attempt=%s",
                symbol,
                side,
                params.get("price"),
                new_price_str,
                bid,
                ask,
                post_only_reprices,
            )

            params["price"] = new_price_str
            continue


def query_limit_order(
    client: BinanceUMFuturesREST,
    symbol: str,
    order_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if order_id is None:
        return None
    try:
        return client.query_order(symbol=symbol, orderId=order_id)
    except ClientError as error:
        logging.error(
            "Order query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def cancel_limit_order(
    client: BinanceUMFuturesREST,
    symbol: str,
    order_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if order_id is None:
        return None
    try:
        return client.cancel_order(symbol=symbol, orderId=order_id)
    except ClientError as error:
        error_code = getattr(error, "error_code", None)
        if error_code in {-2011, -2013}:
            return {"status": "UNKNOWN"}
        logging.error(
            "Order cancel error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def is_order_open(
    client: BinanceUMFuturesREST,
    symbol: str,
    order_id: Optional[int],
) -> Optional[bool]:
    if order_id is None:
        return None
    if not symbol:
        return None
    try:
        open_orders = client.get_orders(symbol=symbol)
    except ClientError as error:
        logging.error(
            "Open orders query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None
    for order in open_orders or []:
        try:
            if int(order.get("orderId")) == int(order_id):
                return True
        except (TypeError, ValueError):
            continue
    return False


def cancel_order_safely(
    client: BinanceUMFuturesREST,
    symbol: str,
    order_id: Optional[int],
) -> bool:
    if order_id is None:
        return True
    cancel_resp = cancel_limit_order(client, symbol, order_id=order_id)
    if cancel_resp is not None:
        return True
    order = query_limit_order(client, symbol, order_id=order_id)
    if limit_order_filled(order) or limit_order_inactive(order):
        return True
    open_flag = is_order_open(client, symbol, order_id=order_id)
    if open_flag is False:
        return True
    return False


def _bool_to_binance_flag(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def _extract_int(payload: Optional[Dict[str, object]], keys: Tuple[str, ...]) -> Optional[int]:
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_str(payload: Optional[Dict[str, object]], keys: Tuple[str, ...]) -> Optional[str]:
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        return str(value)
    return None


def algo_stop_limit_order(
    symbol: str,
    side: str,
    quantity: str,
    trigger_price: str,
    price: str,
    client: BinanceUMFuturesREST,
    *,
    client_order_id: Optional[str] = None,
    algo_type: str = "CONDITIONAL",
    working_type: str = "CONTRACT_PRICE",
    price_protect: bool = False,
    reduce_only: bool = True,
) -> Optional[Dict[str, object]]:
    base_params: Dict[str, object] = {
        "symbol": symbol,
        "side": side,
        "algoType": algo_type,
        "type": "STOP",
        "quantity": quantity,
        "triggerPrice": trigger_price,
        "price": price,
        "timeInForce": "GTC",
        "workingType": working_type,
        "priceProtect": _bool_to_binance_flag(price_protect),
    }
    if reduce_only:
        base_params["reduceOnly"] = True

    last_error: Optional[ClientError] = None
    if client_order_id:
        for client_key in ("newClientAlgoOrderId", "newClientOrderId"):
            params = dict(base_params)
            params[client_key] = client_order_id
            reduce_only_retries = 0
            reduce_only_max_retries = 8
            try:
                while True:
                    try:
                        return client.new_algo_order(**params)
                    except ClientError as error:
                        last_error = error
                        error_code = getattr(error, "error_code", None)
                        if error_code == -1104:
                            raise
                        if reduce_only and error_code == -2022:
                            reduce_only_retries += 1
                            if reduce_only_retries > reduce_only_max_retries:
                                logging.error(
                                    "Algo STOP reduce-only order rejected (max retries). status: %s, code: %s, message: %s",
                                    getattr(error, "status_code", None),
                                    error_code,
                                    getattr(error, "error_message", None),
                                )
                                return None

                            try:
                                refreshed_position = get_position_info(client, symbol)
                            except ClientError as pos_error:
                                logging.error(
                                    "Algo STOP reduce-only rejected and position refresh failed. status: %s, code: %s, message: %s",
                                    getattr(pos_error, "status_code", None),
                                    getattr(pos_error, "error_code", None),
                                    getattr(pos_error, "error_message", None),
                                )
                                return None

                            try:
                                position_amt = float((refreshed_position or {}).get("positionAmt", 0.0))
                            except (TypeError, ValueError):
                                position_amt = 0.0

                            if position_amt == 0.0:
                                time.sleep(0.2)
                                continue

                            expected_side = "SELL" if position_amt > 0 else "BUY"
                            if str(side).upper() != expected_side:
                                logging.error(
                                    "Algo STOP reduce-only rejected; side would not reduce position. symbol=%s order_side=%s position_amt=%s",
                                    symbol,
                                    side,
                                    position_amt,
                                )
                                return None

                            try:
                                requested_qty = float(params.get("quantity") or 0.0)
                            except (TypeError, ValueError):
                                requested_qty = 0.0
                            max_qty = abs(position_amt)

                            if requested_qty > max_qty:
                                try:
                                    filters = get_symbol_filters(client, symbol)
                                    step_size = float(filters["step_size"])
                                    min_qty = float(filters["min_qty"])
                                except Exception:
                                    step_size = 0.0
                                    min_qty = 0.0
                                if step_size > 0:
                                    new_qty_str = format_to_step(max_qty, step_size, rounding=ROUND_DOWN)
                                else:
                                    new_qty_str = str(max_qty)
                                try:
                                    new_qty_num = float(new_qty_str)
                                except (TypeError, ValueError):
                                    new_qty_num = 0.0
                                if min_qty > 0 and new_qty_num < min_qty:
                                    logging.info(
                                        "Algo STOP reduce-only rejected; position below min qty. symbol=%s position_amt=%s min_qty=%s",
                                        symbol,
                                        position_amt,
                                        min_qty,
                                    )
                                    return None
                                if new_qty_str != params.get("quantity"):
                                    logging.info(
                                        "Algo STOP reduce-only rejected; clamping quantity and retrying. symbol=%s qty=%s -> %s position_amt=%s",
                                        symbol,
                                        params.get("quantity"),
                                        new_qty_str,
                                        position_amt,
                                    )
                                    params["quantity"] = new_qty_str

                            time.sleep(0.2)
                            continue

                        raise
            except ClientError as error:
                last_error = error
                if getattr(error, "error_code", None) == -1104:
                    continue
                logging.error(
                    "Algo STOP order error. status: %s, code: %s, message: %s",
                    getattr(error, "status_code", None),
                    getattr(error, "error_code", None),
                    getattr(error, "error_message", None),
                )
                return None

    try:
        reduce_only_retries = 0
        reduce_only_max_retries = 8
        while True:
            try:
                return client.new_algo_order(**base_params)
            except ClientError as error:
                last_error = error
                if reduce_only and getattr(error, "error_code", None) == -2022:
                    reduce_only_retries += 1
                    if reduce_only_retries > reduce_only_max_retries:
                        break

                    try:
                        refreshed_position = get_position_info(client, symbol)
                    except ClientError as pos_error:
                        logging.error(
                            "Algo STOP reduce-only rejected and position refresh failed. status: %s, code: %s, message: %s",
                            getattr(pos_error, "status_code", None),
                            getattr(pos_error, "error_code", None),
                            getattr(pos_error, "error_message", None),
                        )
                        return None

                    try:
                        position_amt = float((refreshed_position or {}).get("positionAmt", 0.0))
                    except (TypeError, ValueError):
                        position_amt = 0.0

                    if position_amt == 0.0:
                        time.sleep(0.2)
                        continue

                    expected_side = "SELL" if position_amt > 0 else "BUY"
                    if str(side).upper() != expected_side:
                        logging.error(
                            "Algo STOP reduce-only rejected; side would not reduce position. symbol=%s order_side=%s position_amt=%s",
                            symbol,
                            side,
                            position_amt,
                        )
                        return None

                    try:
                        requested_qty = float(base_params.get("quantity") or 0.0)
                    except (TypeError, ValueError):
                        requested_qty = 0.0
                    max_qty = abs(position_amt)

                    if requested_qty > max_qty:
                        try:
                            filters = get_symbol_filters(client, symbol)
                            step_size = float(filters["step_size"])
                            min_qty = float(filters["min_qty"])
                        except Exception:
                            step_size = 0.0
                            min_qty = 0.0
                        if step_size > 0:
                            new_qty_str = format_to_step(max_qty, step_size, rounding=ROUND_DOWN)
                        else:
                            new_qty_str = str(max_qty)
                        try:
                            new_qty_num = float(new_qty_str)
                        except (TypeError, ValueError):
                            new_qty_num = 0.0
                        if min_qty > 0 and new_qty_num < min_qty:
                            logging.info(
                                "Algo STOP reduce-only rejected; position below min qty. symbol=%s position_amt=%s min_qty=%s",
                                symbol,
                                position_amt,
                                min_qty,
                            )
                            return None
                        if new_qty_str != base_params.get("quantity"):
                            logging.info(
                                "Algo STOP reduce-only rejected; clamping quantity and retrying. symbol=%s qty=%s -> %s position_amt=%s",
                                symbol,
                                base_params.get("quantity"),
                                new_qty_str,
                                position_amt,
                            )
                            base_params["quantity"] = new_qty_str

                    time.sleep(0.2)
                    continue

                raise
    except ClientError as error:
        last_error = error

    logging.error(
        "Algo STOP order error. status: %s, code: %s, message: %s",
        getattr(last_error, "status_code", None),
        getattr(last_error, "error_code", None),
        getattr(last_error, "error_message", None),
    )
    return None


def query_algo_order(
    client: BinanceUMFuturesREST,
    symbol: str,
    algo_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if algo_id is None:
        return None
    try:
        return client.query_algo_order(symbol=symbol, algoId=algo_id)
    except ClientError as error:
        logging.error(
            "Algo order query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def cancel_algo_order(
    client: BinanceUMFuturesREST,
    symbol: str,
    algo_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if algo_id is None:
        return None
    try:
        return client.cancel_algo_order(symbol=symbol, algoId=algo_id)
    except ClientError as error:
        error_code = getattr(error, "error_code", None)
        if error_code in {-2011, -2013}:
            return {"status": "UNKNOWN"}
        logging.error(
            "Algo order cancel error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return None


def cancel_algo_order_safely(
    client: BinanceUMFuturesREST,
    symbol: str,
    algo_id: Optional[int],
) -> bool:
    if algo_id is None:
        return True
    cancel_resp = cancel_algo_order(client, symbol, algo_id=algo_id)
    if cancel_resp is not None:
        return True
    order = query_algo_order(client, symbol, algo_id=algo_id)
    if order and str(order.get("status") or "").upper() in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}:
        return True
    return False


def algo_order_status(order: Optional[Dict[str, object]]) -> str:
    if not order:
        return ""
    for key in ("status", "algoStatus", "state", "orderStatus"):
        value = order.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str.upper()
    return ""


def algo_order_inactive(order: Optional[Dict[str, object]]) -> bool:
    return algo_order_status(order) in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}


def algo_order_executed(order: Optional[Dict[str, object]]) -> bool:
    if not order:
        return False
    status = algo_order_status(order)
    if status in {"FILLED", "EXECUTED", "TRIGGERED", "COMPLETED", "SUCCESS"}:
        return True
    for key in ("executedQty", "executedQuantity", "executedVolume"):
        value = order.get(key)
        if value is None:
            continue
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False
def cancel_open_strategy_orders(
    client: BinanceUMFuturesREST,
    symbol: str,
    client_id_prefixes: Tuple[str, ...] = ("ATR_",),
) -> None:
    if not symbol:
        return
    try:
        open_orders = client.get_orders(symbol=symbol)
    except ClientError as error:
        logging.error(
            "Open orders query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return
    for order in open_orders or []:
        client_id = str(order.get("clientOrderId") or "")
        if client_id_prefixes and not any(client_id.startswith(p) for p in client_id_prefixes):
            continue
        try:
            order_id = int(order.get("orderId"))
        except (TypeError, ValueError):
            continue
        cancel_limit_order(client, symbol, order_id=order_id)

    try:
        open_algo = client.open_algo_orders(symbol=symbol)
    except ClientError as error:
        logging.error(
            "Open algo orders query error. status: %s, code: %s, message: %s",
            getattr(error, "status_code", None),
            getattr(error, "error_code", None),
            getattr(error, "error_message", None),
        )
        return

    algo_orders: Optional[List[Dict[str, object]]] = None
    if isinstance(open_algo, list):
        algo_orders = open_algo
    elif isinstance(open_algo, dict):
        for key in ("orders", "data", "rows", "items"):
            candidate = open_algo.get(key)
            if isinstance(candidate, list):
                algo_orders = candidate
                break

    if not algo_orders:
        return

    for order in algo_orders:
        client_id = str(
            order.get("clientAlgoOrderId")
            or order.get("clientAlgoId")
            or order.get("clientOrderId")
            or order.get("clientId")
            or ""
        )
        if client_id_prefixes and not any(client_id.startswith(p) for p in client_id_prefixes):
            continue
        algo_id = _extract_int(order, ("algoId", "algoOrderId", "orderId", "id"))
        if algo_id is None:
            continue
        cancel_algo_order(client, symbol, algo_id=algo_id)


def limit_order_filled(order: Optional[Dict[str, object]]) -> bool:
    if not order:
        return False
    status = str(order.get("status") or "").upper()
    return status == "FILLED"


def limit_order_inactive(order: Optional[Dict[str, object]]) -> bool:
    if not order:
        return False
    status = str(order.get("status") or "").upper()
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


def fetch_klines_um_futures(
    symbol: str,
    interval: str,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    *,
    limit_per_call: int = 1500,
    base_url: str = "https://fapi.binance.com",
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance UM Futures public klines endpoint (/fapi/v1/klines).

    - `symbol` example: "BTCUSDC"
    - `interval` example: "1m"
    - `start_utc`/`end_utc` are UTC strings, e.g. "2026-01-09 06:30:00"

    Returns a DataFrame indexed by UTC datetime with columns:
      open, high, low, close, volume, close_time_ms
    """
    client = get_um_futures_public_client(base_url=base_url)

    if start_utc is None and end_utc is None:
        klines = client.klines(symbol=symbol, interval=interval, limit=int(limit_per_call))
        df = klines_to_df(klines)
        return df[~df.index.duplicated(keep="first")].sort_index()

    start_ms = int(pd.Timestamp(start_utc, tz="UTC").timestamp() * 1000) if start_utc else None
    end_ms = int(pd.Timestamp(end_utc, tz="UTC").timestamp() * 1000) if end_utc else None

    if start_ms is not None and end_ms is not None and end_ms < start_ms:
        raise ValueError(f"end_utc must be >= start_utc. start_utc={start_utc!r} end_utc={end_utc!r}")

    all_rows: List[List[object]] = []
    since = start_ms
    max_iters = 20000  # guardrail for unexpected pagination issues
    iters = 0

    while True:
        iters += 1
        if iters > max_iters:
            raise RuntimeError("Exceeded max_iters while fetching UM futures klines; aborting to prevent infinite loop.")

        payload: Dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit_per_call),
        }
        if since is not None:
            payload["startTime"] = int(since)
        if end_ms is not None:
            payload["endTime"] = int(end_ms)

        chunk = client.klines(**payload)
        if not chunk:
            break

        all_rows.extend(chunk)
        last_open_time_ms = int(chunk[-1][0])
        since = last_open_time_ms + 1  # avoid repeating the last candle

        if end_ms is not None and last_open_time_ms >= end_ms:
            break

        # If fewer than requested, we're likely at the end of available data for the query.
        if len(chunk) < int(limit_per_call):
            break

    df = klines_to_df(all_rows)

    # Trim strictly to end_utc if provided (by dt index).
    if end_utc is not None:
        df = df[df.index <= pd.Timestamp(end_utc, tz="UTC")]

    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def fetch_klines_spot(
    symbol: str,
    interval: str,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    *,
    limit_per_call: int = 1000,  # Spot API max is 1000
    base_url: str = "https://api.binance.com",
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance Spot public klines endpoint (GET /api/v3/klines).

    Supports 1s interval which is NOT available on Futures!

    - `symbol` example: "BTCUSDC"
    - `interval` example: "1s", "1m", "15m"
    - `start_utc`/`end_utc` are UTC strings, e.g. "2026-01-09 06:30:00"

    Returns a DataFrame indexed by UTC datetime with columns:
      open, high, low, close, volume, close_time_ms
    """
    import urllib.request
    import json

    ca_bundle_path = getenv("BINANCE_CA_BUNDLE") or getenv("SSL_CERT_FILE") or getenv("REQUESTS_CA_BUNDLE")
    ssl_verify_env = (getenv("BINANCE_SSL_VERIFY") or "").strip().lower()
    ssl_verify = ssl_verify_env not in {"0", "false", "no", "off"}

    def build_ssl_context() -> ssl.SSLContext:
        if not ssl_verify:
            return ssl._create_unverified_context()
        if ca_bundle_path:
            return ssl.create_default_context(cafile=ca_bundle_path)
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    ssl_ctx = build_ssl_context()

    def fetch_chunk(params: Dict[str, object]) -> List[List[object]]:
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{base_url}/api/v3/klines?{query}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    start_ms = int(pd.Timestamp(start_utc, tz="UTC").timestamp() * 1000) if start_utc else None
    end_ms = int(pd.Timestamp(end_utc, tz="UTC").timestamp() * 1000) if end_utc else None

    if start_ms is not None and end_ms is not None and end_ms < start_ms:
        raise ValueError(f"end_utc must be >= start_utc. start_utc={start_utc!r} end_utc={end_utc!r}")

    all_rows: List[List[object]] = []
    since = start_ms
    max_iters = 500000  # 1s data can have many iterations
    iters = 0

    while True:
        iters += 1
        if iters > max_iters:
            raise RuntimeError("Exceeded max_iters while fetching Spot klines; aborting.")

        params: Dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit_per_call),
        }
        if since is not None:
            params["startTime"] = int(since)
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        chunk = fetch_chunk(params)
        if not chunk:
            break

        all_rows.extend(chunk)
        last_open_time_ms = int(chunk[-1][0])
        since = last_open_time_ms + 1

        if end_ms is not None and last_open_time_ms >= end_ms:
            break
        if len(chunk) < int(limit_per_call):
            break

        # Progress logging for large fetches
        if iters % 100 == 0:
            logging.info(f"Fetching {interval} klines from Spot... {len(all_rows):,} candles so far")

    df = klines_to_df(all_rows)

    if end_utc is not None:
        df = df[df.index <= pd.Timestamp(end_utc, tz="UTC")]

    df = df[~df.index.duplicated(keep="first")].sort_index()
    logging.info(f"Fetched {len(df):,} {interval} candles from Spot API")
    return df


def fetch_klines_spot_parallel(
    symbol: str,
    interval: str,
    start_utc: str,
    end_utc: str,
    *,
    limit_per_call: int = 1000,
    base_url: str = "https://api.binance.com",
    max_workers: int = 15,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance Spot in PARALLEL for much faster downloads.

    For 1s data:
    - Sequential: ~70 seconds per day
    - Parallel (10 workers): ~7 seconds per day (~10x faster)

    Args:
        symbol: e.g. "BTCUSDC"
        interval: e.g. "1s", "1m"
        start_utc/end_utc: UTC strings, e.g. "2026-01-01 00:00:00"
        max_workers: Number of parallel threads (default 10, safe for Binance rate limits)

    Returns:
        DataFrame indexed by UTC datetime with columns: open, high, low, close, volume, close_time_ms
    """
    import concurrent.futures
    import urllib.request
    import json

    ca_bundle_path = getenv("BINANCE_CA_BUNDLE") or getenv("SSL_CERT_FILE") or getenv("REQUESTS_CA_BUNDLE")
    ssl_verify_env = (getenv("BINANCE_SSL_VERIFY") or "").strip().lower()
    ssl_verify = ssl_verify_env not in {"0", "false", "no", "off"}

    def build_ssl_context() -> ssl.SSLContext:
        if not ssl_verify:
            return ssl._create_unverified_context()
        if ca_bundle_path:
            return ssl.create_default_context(cafile=ca_bundle_path)
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    ssl_ctx = build_ssl_context()

    start_ms = int(pd.Timestamp(start_utc, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_utc, tz="UTC").timestamp() * 1000)

    if end_ms < start_ms:
        raise ValueError(f"end_utc must be >= start_utc. start_utc={start_utc!r} end_utc={end_utc!r}")

    # Determine interval in milliseconds
    interval_ms_map = {
        "1s": 1000,
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    interval_ms = interval_ms_map.get(interval, 1000)
    chunk_duration_ms = limit_per_call * interval_ms

    # Create time chunks
    chunks: List[Tuple[int, int]] = []
    current_start = start_ms
    while current_start < end_ms:
        chunk_end = min(current_start + chunk_duration_ms - interval_ms, end_ms)
        chunks.append((current_start, chunk_end))
        current_start = chunk_end + interval_ms

    logging.info(f"Fetching {interval} candles in parallel: {len(chunks)} chunks with {max_workers} workers")

    def fetch_chunk(chunk_range: Tuple[int, int]) -> List[List[object]]:
        chunk_start, chunk_end = chunk_range
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": chunk_start,
            "endTime": chunk_end,
            "limit": limit_per_call,
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{base_url}/api/v3/klines?{query}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Backoff
                else:
                    logging.warning(f"Failed to fetch chunk {chunk_start}-{chunk_end}: {e}")
                    return []

    all_rows: List[List[object]] = []
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            result = future.result()
            if result:
                all_rows.extend(result)
            completed += 1
            if completed % 50 == 0 or completed == len(chunks):
                logging.info(f"Progress: {completed}/{len(chunks)} chunks ({len(all_rows):,} candles)")

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_time_ms"])

    df = klines_to_df(all_rows)
    df = df[df.index <= pd.Timestamp(end_utc, tz="UTC")]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    logging.info(f"Fetched {len(df):,} {interval} candles from Spot API (parallel)")
    return df


def get_1s_candles_for_minute(
    df_1s: pd.DataFrame,
    minute_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Get the ~60 1s candles that fall within a 1-minute bar.

    Args:
        df_1s: DataFrame with 1s candles indexed by open time (UTC)
        minute_timestamp: The open time of the 1m candle

    Returns:
        DataFrame slice with 1s candles for that minute
    """
    minute_end = minute_timestamp + pd.Timedelta(seconds=59, milliseconds=999)
    mask = (df_1s.index >= minute_timestamp) & (df_1s.index <= minute_end)
    return df_1s.loc[mask]


def backtest_atr_grinder_lib(
    df: pd.DataFrame,
    df_1s: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Look-Inside-Bar (LIB) Backtest using signal_interval candles for signals and 1s candles for execution.

    This provides more realistic backtesting by:
    - Checking SL/TP hits in chronological order using 1s data
    - Updating trailing stops on each 1s close (not just signal candle close)
    - Better determining limit order fills

    Returns:
      trades_df: executed trades
      df_bt: dataframe with indicators/signals/equity
      stats: summary dict
      trailing_df: trailing stop updates
    """
    df = df.copy().sort_index()
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["atr"] = atr_ema(df, cfg.atr_len)

    swing_high, swing_low = compute_confirmed_swing_levels(
        df,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["swing_high_level"] = swing_high
    df["swing_low_level"] = swing_low

    df["signal"], df["signal_atr"], df["signal_entry_price"] = build_swing_atr_signals(
        df,
        df["atr"],
        df["swing_high_level"],
        df["swing_low_level"],
        body_atr_mult=cfg.thr2,
        swing_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "signal_atr"] = np.nan
        df.loc[warmup_idx, "signal_entry_price"] = np.nan

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
    trail_r_value: Optional[float] = None
    trail_best_close_r: Optional[float] = None
    trail_stop_r: Optional[int] = None
    entry_time = None
    entry_index = None
    used_signal_atr = None
    pending_side = 0  # +1 long limit, -1 short limit, 0 none
    pending_limit_price = None
    pending_signal_atr = None
    pending_created_index = None

    trades: List[Dict] = []
    trailing_stop_updates: List[Dict] = []

    idx = df.index.to_list()

    def apply_slippage(price: float, side: int, is_entry: bool) -> float:
        slip = cfg.slippage
        if is_entry:
            return price * (1 + slip) if side == 1 else price * (1 - slip)
        else:
            return price * (1 - slip) if side == 1 else price * (1 + slip)

    def net_roi_from_prices(entry_p: float, exit_p: float, side: int, leverage: float) -> float:
        underlying_ret = side * ((exit_p - entry_p) / entry_p)
        gross_roi = underlying_ret * leverage
        fee_cost = 2.0 * cfg.fee_rate * leverage
        return gross_roi - fee_cost

    def maybe_fill_limit_1s(side: int, limit_price: float, df_1s_chunk: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
        """Check limit fill using 1s candles, returns (fill_price, fill_time)."""
        for t_1s, candle_1s in df_1s_chunk.iterrows():
            o_ = float(candle_1s["open"])
            h_ = float(candle_1s["high"])
            l_ = float(candle_1s["low"])
            if side == 1:
                if o_ <= limit_price:
                    return o_, t_1s
                if l_ <= limit_price <= h_:
                    return limit_price, t_1s
            elif side == -1:
                if o_ >= limit_price:
                    return o_, t_1s
                if l_ <= limit_price <= h_:
                    return limit_price, t_1s
        return None, None

    def check_exit_lib(
        pos: int,
        sl_price: float,
        tp_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        use_trailing: bool,
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[pd.Timestamp]]:
        """Check for exit using 1s candles in chronological order."""
        for t_1s, candle_1s in df_1s_chunk.iterrows():
            o = float(candle_1s["open"])
            h = float(candle_1s["high"])
            l = float(candle_1s["low"])

            if pos == 1:  # LONG
                # Gap down through stop
                if o <= sl_price:
                    exit_price = apply_slippage(o, pos, is_entry=False)
                    return True, "SL", exit_price, t_1s

                sl_hit = l <= sl_price
                tp_hit = (not use_trailing) and tp_price is not None and h >= tp_price

                if sl_hit and tp_hit:
                    # Both could be hit - use proximity to open
                    dist_to_sl = o - sl_price
                    dist_to_tp = tp_price - o
                    if dist_to_sl <= dist_to_tp:
                        exit_price = apply_slippage(sl_price, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    else:
                        exit_price = apply_slippage(tp_price, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s
                elif sl_hit:
                    exit_price = apply_slippage(sl_price, pos, is_entry=False)
                    return True, "SL", exit_price, t_1s
                elif tp_hit:
                    exit_price = apply_slippage(tp_price, pos, is_entry=False)
                    return True, "TP", exit_price, t_1s

            else:  # SHORT
                # Gap up through stop
                if o >= sl_price:
                    exit_price = apply_slippage(o, pos, is_entry=False)
                    return True, "SL", exit_price, t_1s

                sl_hit = h >= sl_price
                tp_hit = (not use_trailing) and tp_price is not None and l <= tp_price

                if sl_hit and tp_hit:
                    dist_to_sl = sl_price - o
                    dist_to_tp = o - tp_price
                    if dist_to_sl <= dist_to_tp:
                        exit_price = apply_slippage(sl_price, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    else:
                        exit_price = apply_slippage(tp_price, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s
                elif sl_hit:
                    exit_price = apply_slippage(sl_price, pos, is_entry=False)
                    return True, "SL", exit_price, t_1s
                elif tp_hit:
                    exit_price = apply_slippage(tp_price, pos, is_entry=False)
                    return True, "TP", exit_price, t_1s

        return False, None, None, None

    def update_trailing_stop_lib(
        pos: int,
        entry_p: float,
        r_value: float,
        best_close_r: float,
        stop_r: int,
        sl_price: float,
        df_1s_chunk: pd.DataFrame,
        entry_t: pd.Timestamp,
        side_str: str,
    ) -> Tuple[float, float, int]:
        """Update trailing stop using 1s closes."""
        nonlocal trailing_stop_updates
        current_best_r = best_close_r
        current_stop_r = stop_r
        current_sl_price = sl_price

        for t_1s, candle_1s in df_1s_chunk.iterrows():
            c = float(candle_1s["close"])
            close_r = compute_close_r(entry_p, c, pos, r_value)

            if close_r is not None and close_r > current_best_r:
                prev_best_r = current_best_r
                prev_stop_r = current_stop_r
                prev_sl_price = current_sl_price

                current_best_r = close_r
                next_stop_r = compute_trailing_stop_r(current_best_r, current_stop_r, cfg.trail_gap_r)

                if next_stop_r > current_stop_r:
                    new_sl_price = compute_trailing_sl_price(
                        entry_p, pos, r_value, next_stop_r, cfg.trail_buffer_r
                    )
                    trailing_stop_updates.append({
                        "timestamp": t_1s,
                        "entry_time": entry_t,
                        "side": side_str,
                        "entry_price": entry_p,
                        "close_price": c,
                        "close_r": close_r,
                        "prev_best_r": prev_best_r,
                        "new_best_r": current_best_r,
                        "prev_stop_r": prev_stop_r,
                        "new_stop_r": next_stop_r,
                        "prev_stop_price": prev_sl_price,
                        "new_stop_price": new_sl_price,
                        "r_value": r_value,
                        "trail_gap_r": cfg.trail_gap_r,
                        "trail_buffer_r": cfg.trail_buffer_r,
                        "stop_moved": True,
                        "lib_mode": True,
                    })
                    current_stop_r = next_stop_r
                    current_sl_price = new_sl_price

        return current_sl_price, current_best_r, current_stop_r

    limit_timeout_bars = max(1, int(cfg.entry_limit_timeout_bars))

    for i in range(1, len(df)):
        t = idx[i]
        prev_t = idx[i - 1]

        # Get 1s candles for this minute
        df_1s_minute = get_1s_candles_for_minute(df_1s, t)
        has_1s_data = len(df_1s_minute) > 0

        # Fallback OHLC from 1m if no 1s data
        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # --- EXIT logic using 1s data ---
        if position != 0 and entry_price is not None and stop_price is not None:
            exited = False
            exit_reason = None
            exit_price = None
            exit_time = t

            if has_1s_data:
                # Update trailing stop using 1s closes FIRST
                if cfg.use_trailing_stop and trail_r_value is not None and trail_best_close_r is not None and trail_stop_r is not None:
                    side_str = "LONG" if position == 1 else "SHORT"
                    stop_price, trail_best_close_r, trail_stop_r = update_trailing_stop_lib(
                        position, entry_price, trail_r_value, trail_best_close_r, trail_stop_r,
                        stop_price, df_1s_minute, entry_time, side_str
                    )

                # Check exit using 1s candles
                exited, exit_reason, exit_price, exit_time = check_exit_lib(
                    position, stop_price, target_price, df_1s_minute, cfg.use_trailing_stop
                )
            else:
                # Fallback to 1m OHLC (original logic)
                sl_hit = (l <= stop_price) if position == 1 else (h >= stop_price)

                if cfg.use_trailing_stop:
                    if sl_hit:
                        exit_reason = "SL"
                        exit_price = apply_slippage(stop_price, position, is_entry=False)
                        exited = True
                else:
                    if target_price is not None:
                        tp_hit = (h >= target_price) if position == 1 else (l <= target_price)
                    else:
                        tp_hit = False

                    if sl_hit and tp_hit:
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
                # Cap SL loss at target_loss_usd to match live exchange behavior
                if exit_reason == "SL" and cfg.target_loss_usd is not None and pnl_net < -cfg.target_loss_usd:
                    pnl_net = -cfg.target_loss_usd
                    roi_net = pnl_net / active_margin if active_margin else roi_net
                equity += pnl_net

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
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
                    "lib_mode": has_1s_data,
                })

                position = 0
                entry_price = None
                target_price = None
                stop_price = None
                trail_r_value = None
                trail_best_close_r = None
                trail_stop_r = None
                entry_time = None
                entry_index = None
                used_signal_atr = None
                active_margin = cfg.initial_capital
                active_leverage = cfg.leverage
                active_notional = active_margin * active_leverage

        # --- Trailing stop update for 1m close (if no 1s data or using 1m fallback) ---
        if (
            not has_1s_data
            and cfg.use_trailing_stop
            and position != 0
            and entry_price is not None
            and stop_price is not None
            and trail_r_value is not None
            and trail_best_close_r is not None
            and trail_stop_r is not None
        ):
            c = float(df.at[t, "close"])
            close_r = compute_close_r(entry_price, c, position, trail_r_value)
            if close_r is not None:
                side_str = "LONG" if position == 1 else "SHORT"
                prev_best_r = trail_best_close_r
                prev_stop_r = trail_stop_r
                prev_stop_price = stop_price

                trail_best_close_r = max(trail_best_close_r, close_r)
                next_stop_r = compute_trailing_stop_r(trail_best_close_r, trail_stop_r, cfg.trail_gap_r)

                if next_stop_r > trail_stop_r:
                    new_stop_price = compute_trailing_sl_price(
                        entry_price, position, trail_r_value, next_stop_r, cfg.trail_buffer_r
                    )
                    trailing_stop_updates.append({
                        "timestamp": t,
                        "entry_time": entry_time,
                        "side": side_str,
                        "entry_price": entry_price,
                        "close_price": c,
                        "close_r": close_r,
                        "prev_best_r": prev_best_r,
                        "new_best_r": trail_best_close_r,
                        "prev_stop_r": prev_stop_r,
                        "new_stop_r": next_stop_r,
                        "prev_stop_price": prev_stop_price,
                        "new_stop_price": new_stop_price,
                        "r_value": trail_r_value,
                        "trail_gap_r": cfg.trail_gap_r,
                        "trail_buffer_r": cfg.trail_buffer_r,
                        "stop_moved": True,
                        "lib_mode": False,
                    })
                    trail_stop_r = next_stop_r
                    stop_price = new_stop_price

        # --- ENTRY logic ---
        if position == 0:
            if pending_side != 0 and pending_created_index is not None:
                if (i - pending_created_index) >= limit_timeout_bars:
                    pending_side = 0
                    pending_limit_price = None
                    pending_signal_atr = None
                    pending_created_index = None

            # Try to fill pending limit order using 1s data
            if pending_side != 0 and pending_limit_price is not None and pending_signal_atr is not None:
                fill_raw = None
                fill_time = t

                if has_1s_data:
                    fill_raw, fill_time = maybe_fill_limit_1s(int(pending_side), float(pending_limit_price), df_1s_minute)
                else:
                    # Fallback to 1m OHLC
                    side = int(pending_side)
                    if side == 1:
                        if o <= pending_limit_price:
                            fill_raw = o
                        elif l <= pending_limit_price <= h:
                            fill_raw = pending_limit_price
                    elif side == -1:
                        if o >= pending_limit_price:
                            fill_raw = o
                        elif l <= pending_limit_price <= h:
                            fill_raw = pending_limit_price

                if fill_raw is not None:
                    side = int(pending_side)
                    signal_atr_value = float(pending_signal_atr)
                    entry_price = apply_slippage(float(fill_raw), side, is_entry=True)

                    trade_margin, trade_leverage, _ = resolve_trade_sizing(
                        entry_price=entry_price,
                        atr_value=signal_atr_value,
                        sl_atr_mult=cfg.sl_atr_mult,
                        margin_cap=cfg.initial_capital,
                        max_leverage=cfg.leverage,
                        min_leverage=cfg.min_leverage,
                        target_loss_usd=cfg.target_loss_usd,
                    )
                    if trade_margin is None or trade_leverage is None:
                        pending_side = 0
                        pending_limit_price = None
                        pending_signal_atr = None
                        pending_created_index = None
                    else:
                        if cfg.use_trailing_stop:
                            target_price = None
                            trail_r_value = signal_atr_value * cfg.sl_atr_mult
                            trail_best_close_r = 0.0
                            trail_stop_r = -1
                            stop_price = compute_trailing_sl_price(
                                entry_price, side, trail_r_value, trail_stop_r, cfg.trail_buffer_r
                            )
                        else:
                            target_price, stop_price = compute_tp_sl_prices(
                                entry_price, side, signal_atr_value, cfg.tp_atr_mult, cfg.sl_atr_mult
                            )

                        position = side
                        entry_time = fill_time if has_1s_data else t
                        entry_index = i
                        used_signal_atr = signal_atr_value
                        active_margin = trade_margin
                        active_leverage = trade_leverage
                        active_notional = active_margin * active_leverage

                        pending_side = 0
                        pending_limit_price = None
                        pending_signal_atr = None
                        pending_created_index = None

            # Check for new signal
            if position == 0 and pending_side == 0:
                prev_signal = int(df.at[prev_t, "signal"])
                prev_signal_atr = df.at[prev_t, "signal_atr"]
                prev_entry_price = df.at[prev_t, "signal_entry_price"]

                if prev_signal != 0 and not (isinstance(prev_signal_atr, float) and math.isnan(prev_signal_atr)):
                    side = prev_signal
                    signal_atr_value = float(prev_signal_atr)

                    if prev_entry_price is not None and not (isinstance(prev_entry_price, float) and math.isnan(prev_entry_price)):
                        pending_side = side
                        pending_limit_price = float(prev_entry_price)
                        pending_signal_atr = signal_atr_value
                        pending_created_index = i

                        fill_raw = None
                        fill_time = t

                        if has_1s_data:
                            fill_raw, fill_time = maybe_fill_limit_1s(side, float(prev_entry_price), df_1s_minute)
                        else:
                            if side == 1:
                                if o <= prev_entry_price:
                                    fill_raw = o
                                elif l <= prev_entry_price <= h:
                                    fill_raw = prev_entry_price
                            elif side == -1:
                                if o >= prev_entry_price:
                                    fill_raw = o
                                elif l <= prev_entry_price <= h:
                                    fill_raw = prev_entry_price

                        if fill_raw is not None:
                            entry_price = apply_slippage(float(fill_raw), side, is_entry=True)

                            trade_margin, trade_leverage, _ = resolve_trade_sizing(
                                entry_price=entry_price,
                                atr_value=signal_atr_value,
                                sl_atr_mult=cfg.sl_atr_mult,
                                margin_cap=cfg.initial_capital,
                                max_leverage=cfg.leverage,
                                min_leverage=cfg.min_leverage,
                                target_loss_usd=cfg.target_loss_usd,
                            )
                            if trade_margin is None or trade_leverage is None:
                                pending_side = 0
                                pending_limit_price = None
                                pending_signal_atr = None
                                pending_created_index = None
                            else:
                                if cfg.use_trailing_stop:
                                    target_price = None
                                    trail_r_value = signal_atr_value * cfg.sl_atr_mult
                                    trail_best_close_r = 0.0
                                    trail_stop_r = -1
                                    stop_price = compute_trailing_sl_price(
                                        entry_price, side, trail_r_value, trail_stop_r, cfg.trail_buffer_r
                                    )
                                else:
                                    target_price, stop_price = compute_tp_sl_prices(
                                        entry_price, side, signal_atr_value, cfg.tp_atr_mult, cfg.sl_atr_mult
                                    )

                                position = side
                                entry_time = fill_time if has_1s_data else t
                                entry_index = i
                                used_signal_atr = signal_atr_value
                                active_margin = trade_margin
                                active_leverage = trade_leverage
                                active_notional = active_margin * active_leverage

                                pending_side = 0
                                pending_limit_price = None
                                pending_signal_atr = None
                                pending_created_index = None
                    else:
                        entry_price = apply_slippage(o, side, is_entry=True)

                        trade_margin, trade_leverage, _ = resolve_trade_sizing(
                            entry_price=entry_price,
                            atr_value=signal_atr_value,
                            sl_atr_mult=cfg.sl_atr_mult,
                            margin_cap=cfg.initial_capital,
                            max_leverage=cfg.leverage,
                            min_leverage=cfg.min_leverage,
                            target_loss_usd=cfg.target_loss_usd,
                        )
                        if trade_margin is None or trade_leverage is None:
                            continue

                        if cfg.use_trailing_stop:
                            target_price = None
                            trail_r_value = signal_atr_value * cfg.sl_atr_mult
                            trail_best_close_r = 0.0
                            trail_stop_r = -1
                            stop_price = compute_trailing_sl_price(
                                entry_price, side, trail_r_value, trail_stop_r, cfg.trail_buffer_r
                            )
                        else:
                            target_price, stop_price = compute_tp_sl_prices(
                                entry_price, side, signal_atr_value, cfg.tp_atr_mult, cfg.sl_atr_mult
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
    trailing_df = pd.DataFrame(trailing_stop_updates)
    stats = compute_stats(trades_df, df, cfg)

    return trades_df, df, stats, trailing_df


def compute_live_signal(df: pd.DataFrame, cfg: LiveConfig) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    warmup = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if len(df) <= warmup:
        return 0, None, None, None

    atr = atr_ema(df, cfg.atr_len)
    swing_high, swing_low = compute_confirmed_swing_levels(
        df,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    signal, signal_atr, signal_entry_price = build_swing_atr_signals(
        df,
        atr,
        swing_high,
        swing_low,
        body_atr_mult=cfg.thr2,
        swing_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    signal_val = int(signal.iloc[-1])
    atr_val = float(atr.iloc[-1])
    signal_atr_val = signal_atr.iloc[-1]
    entry_price_val = signal_entry_price.iloc[-1]
    if signal_val == 0 or pd.isna(signal_atr_val) or math.isnan(atr_val):
        return 0, None, atr_val if not math.isnan(atr_val) else None, None
    entry_price: Optional[float]
    if entry_price_val is not None and not pd.isna(entry_price_val):
        entry_price = float(entry_price_val)
    else:
        entry_price = None
    return signal_val, float(signal_atr_val), atr_val, entry_price


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


def compute_sl_price(entry_price: float, side: int, atr_value: float, sl_atr_mult: float) -> float:
    move_sl = atr_value * sl_atr_mult
    if side == 1:
        return entry_price - move_sl
    return entry_price + move_sl


def compute_close_r(entry_price: float, close_price: float, side: int, r_value: float) -> Optional[float]:
    if r_value <= 0:
        return None
    if side == 1:
        return (close_price - entry_price) / r_value
    return (entry_price - close_price) / r_value


def compute_trailing_stop_r(best_close_r: float, current_stop_r: int, gap_r: float) -> int:
    candidate = int(math.floor(best_close_r - gap_r))
    return max(current_stop_r, candidate)


def compute_trailing_sl_price(entry_price: float, side: int, r_value: float, stop_r: int, buffer_r: float) -> float:
    price = entry_price + (stop_r * r_value) if side == 1 else entry_price - (stop_r * r_value)
    if stop_r >= 0 and buffer_r:
        buffer_price = r_value * float(buffer_r)
        price = price + buffer_price if side == 1 else price - buffer_price
    return price


def compute_sl_limit_price(trigger_price: float, side: int, atr_value: float, offset_mult: float) -> float:
    """
    Compute limit price for stop loss order with offset to ensure maker execution.
    
    For SELL (closing LONG, side=1): price is HIGHER than trigger to sit on ask side.
    For BUY (closing SHORT, side=-1): price is LOWER than trigger to sit on bid side.
    """
    offset = atr_value * offset_mult
    if side == 1:
        # Closing LONG with SELL: limit price above trigger
        return trigger_price + offset
    else:
        # Closing SHORT with BUY: limit price below trigger
        return trigger_price - offset


def compute_margin_from_targets(
    entry_price: float,
    atr_value: float,
    sl_atr_mult: float,
    leverage: float,
    target_loss_usd: Optional[float],
) -> Optional[float]:
    if entry_price <= 0 or atr_value <= 0 or leverage <= 0:
        return None
    move_sl = atr_value * sl_atr_mult
    if move_sl <= 0:
        return None
    roi_sl = (move_sl / entry_price) * leverage
    if target_loss_usd is not None and target_loss_usd > 0 and roi_sl > 0:
        return target_loss_usd / roi_sl
    return None


def compute_required_leverage(
    entry_price: float,
    atr_value: float,
    sl_atr_mult: float,
    margin_usd: float,
    target_loss_usd: Optional[float],
) -> Optional[float]:
    if entry_price <= 0 or atr_value <= 0 or margin_usd <= 0:
        return None
    move_sl = atr_value * sl_atr_mult
    if target_loss_usd is not None and target_loss_usd > 0 and move_sl > 0:
        return (target_loss_usd * entry_price) / (move_sl * margin_usd)
    return None


def resolve_trade_sizing(
    entry_price: float,
    atr_value: float,
    sl_atr_mult: float,
    margin_cap: float,
    max_leverage: float,
    min_leverage: float,
    target_loss_usd: Optional[float],
    leverage_step: float = 0.0,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if margin_cap <= 0 or max_leverage <= 0:
        return None, None, "invalid_config"
    req_leverage = compute_required_leverage(
        entry_price=entry_price,
        atr_value=atr_value,
        sl_atr_mult=sl_atr_mult,
        margin_usd=margin_cap,
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
        sl_atr_mult=sl_atr_mult,
        leverage=leverage_used,
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
    symbols = list(dict.fromkeys(cfg.symbols or []))
    if cfg.symbol and cfg.symbol not in symbols:
        symbols.insert(0, cfg.symbol)
    if symbol_override:
        symbols = [symbol_override]
    if not symbols:
        raise RuntimeError("No live symbols configured.")
    cfg.symbols = symbols
    if getenv("LIVE_TRADING") != "1":
        raise RuntimeError("Set LIVE_TRADING=1 to enable live trading.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = get_um_futures_client(cfg)
    filters_by_symbol = {sym: get_symbol_filters(client, sym) for sym in symbols}
    current_leverage_by_symbol: Dict[str, Optional[int]] = {}
    for sym in symbols:
        try:
            client.change_leverage(symbol=sym, leverage=cfg.leverage)
            current_leverage_by_symbol[sym] = cfg.leverage
        except ClientError:
            current_leverage_by_symbol[sym] = None

    state = LiveState()
    interval_ms = interval_to_ms(cfg.signal_interval)
    last_close_ms_by_symbol: Dict[str, Optional[int]] = {sym: None for sym in symbols}
    last_atr_by_symbol: Dict[str, Optional[float]] = {sym: None for sym in symbols}
    log_event(cfg.log_path, {
        "event": "startup",
        "symbols": symbols,
        "signal_interval": cfg.signal_interval,
        "leverage": cfg.leverage,
        "margin_usd": format_float_2(cfg.margin_usd),
        "target_loss_usd": format_float_2(cfg.target_loss_usd),
    })

    def process_symbol_candle(symbol: str, allow_entry: bool) -> bool:
        filters = filters_by_symbol[symbol]
        history = max(cfg.atr_history_bars, cfg.atr_len + 2)
        if history > 1000:
            history = 1000
        try:
            klines = client.klines(symbol=symbol, interval=cfg.signal_interval, limit=history)
        except ClientError as exc:
            log_event(
                cfg.log_path,
                {
                    "event": "klines_error",
                    "symbol": symbol,
                    "status_code": getattr(exc, "status_code", None),
                    "error_code": getattr(exc, "error_code", None),
                    "message": str(getattr(exc, "error_message", str(exc))),
                },
            )
            logging.warning("Klines fetch failed. symbol=%s error=%s", symbol, exc)
            return False
        if len(klines) < 2:
            return False
        closed_klines = klines[:-1]
        close_time_ms = int(closed_klines[-1][6])
        if close_time_ms == last_close_ms_by_symbol.get(symbol):
            return False
        last_close_ms_by_symbol[symbol] = close_time_ms
        df = klines_to_df(closed_klines)
        signal, signal_atr, atr_value, signal_entry_price = compute_live_signal(df, cfg)
        last_atr_by_symbol[symbol] = atr_value
        candle = df.iloc[-1]
        body = abs(float(candle["close"]) - float(candle["open"]))
        log_event(cfg.log_path, {
            "event": "candle_close",
            "symbol": symbol,
            "close_time_ms": close_time_ms,
            "open": format_float_by_symbol(candle["open"], symbol),
            "high": format_float_by_symbol(candle["high"], symbol),
            "low": format_float_by_symbol(candle["low"], symbol),
            "close": format_float_by_symbol(candle["close"], symbol),
            "body": format_float_by_symbol(body, symbol),
            "atr": format_float_by_symbol(atr_value, symbol),
            "signal": signal,
            "signal_atr": format_float_by_symbol(signal_atr, symbol),
            "signal_entry_price": format_float_by_symbol(signal_entry_price, symbol),
        })

        if (
            cfg.use_trailing_stop
            and state.active_symbol == symbol
            and state.entry_price is not None
            and state.entry_side is not None
            and not state.sl_triggered
        ):
            pos = positions.get(symbol) or {}
            try:
                position_amt = float(pos.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                position_amt = 0.0
            if position_amt != 0.0:
                try:
                    entry_price = float(state.entry_price or 0.0)
                except (TypeError, ValueError):
                    entry_price = 0.0
                side_sign = int(state.entry_side or (1 if position_amt > 0 else -1))
                r_value = state.trail_r_value
                if r_value is None and state.active_atr is not None:
                    r_value = state.active_atr * cfg.sl_atr_mult
                    state.trail_r_value = r_value
                if entry_price > 0 and r_value is not None and r_value > 0:
                    close_r = compute_close_r(entry_price, float(candle["close"]), side_sign, r_value)
                    if close_r is not None:
                        if state.trail_best_close_r is None:
                            state.trail_best_close_r = 0.0
                        state.trail_best_close_r = max(state.trail_best_close_r, close_r)
                        if state.trail_stop_r is None:
                            state.trail_stop_r = -1
                        candidate_stop_r = compute_trailing_stop_r(
                            state.trail_best_close_r,
                            state.trail_stop_r,
                            cfg.trail_gap_r,
                        )
                        if candidate_stop_r > state.trail_stop_r:
                            next_sl_price = compute_trailing_sl_price(
                                entry_price,
                                side_sign,
                                r_value,
                                candidate_stop_r,
                                cfg.trail_buffer_r,
                            )
                            next_sl_price_str = format_to_step(next_sl_price, filters["tick_size"])
                            try:
                                next_sl_price_rounded = float(next_sl_price_str)
                            except (TypeError, ValueError):
                                next_sl_price_rounded = 0.0

                            improves = True
                            if state.sl_price is not None:
                                try:
                                    current_sl_price = float(state.sl_price)
                                except (TypeError, ValueError):
                                    current_sl_price = None
                                if current_sl_price is not None:
                                    improves = next_sl_price_rounded > current_sl_price if side_sign == 1 else next_sl_price_rounded < current_sl_price

                            qty = abs(position_amt)
                            qty_str = format_to_step(qty, filters["step_size"])
                            try:
                                qty_num = float(qty_str)
                            except (TypeError, ValueError):
                                qty_num = 0.0
                            if improves and next_sl_price_rounded > 0 and qty_num >= filters["min_qty"]:
                                if state.sl_algo_id:
                                    cancel_algo_order_safely(client, symbol, algo_id=state.sl_algo_id)
                                state.sl_algo_id = None
                                state.sl_algo_client_order_id = None
                                sl_side = "SELL" if side_sign == 1 else "BUY"
                                sl_client_id = f"ATR_SL_T_{close_time_ms}"
                                # Compute limit price with offset for maker execution
                                sl_limit_price = compute_sl_limit_price(
                                    next_sl_price_rounded,
                                    side_sign,
                                    state.active_atr or r_value,
                                    cfg.sl_maker_offset_atr_mult,
                                )
                                sl_limit_price_str = format_to_step(sl_limit_price, filters["tick_size"])
                                sl_algo_order = algo_stop_limit_order(
                                    symbol=symbol,
                                    side=sl_side,
                                    quantity=qty_str,
                                    trigger_price=next_sl_price_str,
                                    price=sl_limit_price_str,
                                    client=client,
                                    client_order_id=sl_client_id,
                                    algo_type=cfg.algo_type,
                                    working_type=cfg.algo_working_type,
                                    price_protect=cfg.algo_price_protect,
                                    reduce_only=True,
                                )
                                if sl_algo_order is None:
                                    log_event(cfg.log_path, {
                                        "event": "sl_trail_rejected",
                                        "symbol": symbol,
                                        "side": "LONG" if side_sign == 1 else "SHORT",
                                        "sl_price": format_float_by_symbol(next_sl_price_rounded, symbol),
                                        "stop_r": candidate_stop_r,
                                        "best_close_r": format_float_2(state.trail_best_close_r),
                                    })
                                else:
                                    state.sl_algo_id = _extract_int(sl_algo_order, ("algoId", "algoOrderId", "orderId", "id"))
                                    state.sl_algo_client_order_id = (
                                        _extract_str(sl_algo_order, ("clientAlgoOrderId", "clientAlgoId", "clientOrderId", "clientId"))
                                        or sl_client_id
                                    )
                                    state.sl_price = next_sl_price_rounded
                                    state.trail_stop_r = candidate_stop_r
                                    log_event(cfg.log_path, {
                                        "event": "sl_trail_update",
                                        "symbol": symbol,
                                        "side": "LONG" if side_sign == 1 else "SHORT",
                                        "sl_price": format_float_by_symbol(next_sl_price_rounded, symbol),
                                        "stop_r": candidate_stop_r,
                                        "best_close_r": format_float_2(state.trail_best_close_r),
                                        "algo_id": state.sl_algo_id,
                                    })

        if not allow_entry:
            return False
        if signal == 0 or signal_atr is None or atr_value is None:
            return False

        delay = cfg.entry_delay_min_seconds
        if cfg.entry_delay_max_seconds > cfg.entry_delay_min_seconds:
            delay = random.uniform(cfg.entry_delay_min_seconds, cfg.entry_delay_max_seconds)
        time.sleep(delay)

        try:
            bid, ask = get_book_ticker(client, symbol)
        except ClientError as exc:
            log_event(
                cfg.log_path,
                {
                    "event": "book_ticker_error",
                    "symbol": symbol,
                    "status_code": getattr(exc, "status_code", None),
                    "error_code": getattr(exc, "error_code", None),
                    "message": str(getattr(exc, "error_message", str(exc))),
                },
            )
            logging.warning("Book ticker fetch failed. symbol=%s error=%s", symbol, exc)
            return False
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid else 1.0
        if spread_pct > cfg.spread_max_pct:
            log_event(cfg.log_path, {"event": "skip_spread", "symbol": symbol, "spread_pct": format_float_2(spread_pct)})
            return False

        side = "BUY" if signal == 1 else "SELL"
        offset: Optional[float] = None
        if signal_entry_price is not None:
            limit_price = float(signal_entry_price)
        else:
            offset = atr_value * cfg.atr_offset_mult
            if signal == 1:
                limit_price = bid - offset
            else:
                limit_price = ask + offset

        tick = filters["tick_size"]
        if tick > 0:
            limit_price = reprice_post_only(limit_price, side, bid, ask, tick)

        limit_price_str = format_price_to_tick(limit_price, filters["tick_size"], side=side, post_only=True)
        try:
            limit_price = float(limit_price_str)
        except (TypeError, ValueError):
            limit_price = 0.0
        if limit_price <= 0:
            log_event(cfg.log_path, {"event": "skip_price", "symbol": symbol, "limit_price": format_float_2(limit_price)})
            return False

        margin_usd, leverage_used, skip_reason = resolve_trade_sizing(
            entry_price=limit_price,
            atr_value=signal_atr,
            sl_atr_mult=cfg.sl_atr_mult,
            margin_cap=cfg.margin_usd,
            max_leverage=cfg.leverage,
            min_leverage=cfg.min_leverage,
            target_loss_usd=cfg.target_loss_usd,
            leverage_step=1.0,
        )
        if margin_usd is None or leverage_used is None:
            log_event(cfg.log_path, {
                "event": "skip_leverage",
                "symbol": symbol,
                "reason": skip_reason,
                "margin_cap": format_float_2(cfg.margin_usd),
                "max_leverage": cfg.leverage,
            })
            return False
        leverage_int = int(round(leverage_used))
        current_leverage = current_leverage_by_symbol.get(symbol)
        if current_leverage != leverage_int:
            try:
                client.change_leverage(symbol=symbol, leverage=leverage_int)
                current_leverage_by_symbol[symbol] = leverage_int
            except ClientError:
                log_event(cfg.log_path, {"event": "leverage_change_error", "symbol": symbol, "leverage": leverage_int})
                return False
        notional = margin_usd * leverage_int
        qty = notional / limit_price
        qty_str = format_to_step(qty, filters["step_size"])
        if float(qty_str) < filters["min_qty"]:
            log_event(cfg.log_path, {"event": "skip_qty", "symbol": symbol, "quantity": format_float_2(qty_str)})
            return False
        entry_order = limit_order(
            symbol,
            side,
            qty_str,
            limit_price_str,
            client,
            client_order_id=f"ATR_E_{close_time_ms}",
            tick_size=filters["tick_size"],
        )
        if entry_order:
            state.entry_order_id = entry_order.get("orderId")
            state.entry_order_time = time.time()
            state.pending_atr = signal_atr
            state.pending_side = 1 if side == "BUY" else -1
            state.entry_close_ms = close_time_ms
            state.entry_margin_usd = margin_usd
            state.entry_leverage = leverage_int
            state.active_symbol = symbol
            log_event(cfg.log_path, {
                "event": "entry_order",
                "symbol": symbol,
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
            return True
        return False

    while True:
        try:
            positions: Dict[str, Dict[str, str]] = {}
            open_symbols: List[str] = []
            for sym in symbols:
                pos = get_position_info(client, sym)
                positions[sym] = pos
                try:
                    amt = float(pos.get("positionAmt", 0.0))
                except (TypeError, ValueError):
                    amt = 0.0
                if amt != 0.0:
                    open_symbols.append(sym)

            multi_position = len(open_symbols) > 1
            if multi_position:
                log_event(cfg.log_path, {
                    "event": "multi_position_detected",
                    "symbols": open_symbols,
                })

            active_symbol = state.active_symbol
            if active_symbol and active_symbol not in symbols:
                log_event(cfg.log_path, {
                    "event": "active_symbol_missing",
                    "symbol": active_symbol,
                })
                active_symbol = None
                state.active_symbol = None

            if active_symbol is None and open_symbols:
                active_symbol = open_symbols[0]
                state.active_symbol = active_symbol

            if state.entry_order_id is not None and active_symbol is None:
                log_event(cfg.log_path, {
                    "event": "entry_order_symbol_missing",
                    "order_id": state.entry_order_id,
                })
                time.sleep(cfg.poll_interval_seconds)
                continue

            position = positions.get(active_symbol) if active_symbol else {}
            try:
                position_amt = float(position.get("positionAmt", 0.0)) if position else 0.0
            except (TypeError, ValueError):
                position_amt = 0.0
            exit_filled = False
            in_position = position_amt != 0.0
            if in_position:
                state.had_position = True

            # Manage open entry order
            if state.entry_order_id is not None and active_symbol is not None:
                try:
                    order = client.query_order(symbol=active_symbol, orderId=state.entry_order_id)
                except ClientError:
                    log_event(cfg.log_path, {
                        "event": "entry_order_query_error",
                        "symbol": active_symbol,
                        "order_id": state.entry_order_id,
                    })
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
                    state.trail_r_value = None
                    state.trail_best_close_r = None
                    state.trail_stop_r = None
                    if not in_position:
                        state.active_symbol = None
                    order = None
                if order is None:
                    time.sleep(cfg.poll_interval_seconds)
                    continue
                status = order.get("status")
                if status == "FILLED":
                    filters = filters_by_symbol[active_symbol]
                    refreshed_position = get_position_info(client, active_symbol)
                    try:
                        refreshed_position_amt = float(refreshed_position.get("positionAmt", 0.0)) if refreshed_position else 0.0
                    except (TypeError, ValueError):
                        refreshed_position_amt = 0.0

                    entry_price = 0.0
                    if refreshed_position:
                        try:
                            entry_price = float(refreshed_position.get("entryPrice", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            entry_price = 0.0
                    if entry_price <= 0:
                        entry_price = float(order.get("avgPrice") or order.get("price") or 0.0)

                    if refreshed_position_amt != 0.0:
                        qty = abs(refreshed_position_amt)
                        side = 1 if refreshed_position_amt > 0 else -1
                    else:
                        qty = float(order.get("executedQty") or order.get("origQty") or 0.0)
                        side_from_order = str(order.get("side") or "").upper()
                        if side_from_order == "BUY":
                            side = 1
                        elif side_from_order == "SELL":
                            side = -1
                        else:
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
                        state.entry_leverage = current_leverage_by_symbol.get(active_symbol) or cfg.leverage
                    state.entry_side = side
                    state.entry_time_iso = _utc_now_iso()
                    last_close_ms = last_close_ms_by_symbol.get(active_symbol)
                    if last_close_ms is not None:
                        state.entry_signal_ms = last_close_ms
                    else:
                        state.entry_signal_ms = entry_close_ms

                    entry_atr = state.active_atr if state.active_atr is not None else last_atr_by_symbol.get(active_symbol)
                    if entry_atr is None:
                        log_event(cfg.log_path, {"event": "missing_atr", "stage": "entry_filled", "symbol": active_symbol})
                        entry_atr = 0.0
                    state.active_atr = entry_atr
                    qty_str = format_to_step(qty, filters["step_size"])

                    if cfg.use_trailing_stop:
                        state.tp_order_id = None
                        state.tp_client_order_id = None
                        state.tp_price = None

                        state.trail_r_value = entry_atr * cfg.sl_atr_mult
                        state.trail_best_close_r = 0.0
                        state.trail_stop_r = -1

                        sl_price = compute_trailing_sl_price(
                            entry_price,
                            side,
                            state.trail_r_value,
                            state.trail_stop_r,
                            cfg.trail_buffer_r,
                        )
                        sl_price_str = format_to_step(sl_price, filters["tick_size"])
                        state.sl_price = sl_price

                        sl_side = "SELL" if side == 1 else "BUY"
                        sl_client_id = f"ATR_SL_{entry_close_ms or ''}"
                        # Compute limit price with offset for maker execution
                        sl_limit_price = compute_sl_limit_price(
                            sl_price,
                            side,
                            entry_atr,
                            cfg.sl_maker_offset_atr_mult,
                        )
                        sl_limit_price_str = format_to_step(sl_limit_price, filters["tick_size"])
                        sl_algo_order = algo_stop_limit_order(
                            symbol=active_symbol,
                            side=sl_side,
                            quantity=qty_str,
                            trigger_price=sl_price_str,
                            price=sl_limit_price_str,
                            client=client,
                            client_order_id=sl_client_id,
                            algo_type=cfg.algo_type,
                            working_type=cfg.algo_working_type,
                            price_protect=cfg.algo_price_protect,
                            reduce_only=True,
                        )
                        if sl_algo_order is None:
                            log_event(cfg.log_path, {
                                "event": "sl_algo_order_rejected",
                                "symbol": active_symbol,
                                "sl_price": format_float_2(sl_price),
                            })
                        state.sl_algo_id = _extract_int(sl_algo_order, ("algoId", "algoOrderId", "orderId", "id"))
                        state.sl_algo_client_order_id = (
                            _extract_str(sl_algo_order, ("clientAlgoOrderId", "clientAlgoId", "clientOrderId", "clientId"))
                            or (sl_client_id if sl_algo_order else None)
                        )
                        state.sl_order_id = None
                        state.sl_client_order_id = None
                        state.sl_triggered = False
                        state.sl_order_time = None
                        state.entry_close_ms = None

                        log_event(cfg.log_path, {
                            "event": "entry_filled",
                            "symbol": active_symbol,
                            "order_id": order.get("orderId"),
                            "side": "LONG" if side == 1 else "SHORT",
                            "entry_price": format_float_2(entry_price),
                            "quantity": format_float_2(qty),
                            "entry_atr": format_float_2(state.active_atr),
                            "margin_usd": format_float_2(state.entry_margin_usd),
                            "leverage": state.entry_leverage,
                            "sl_price": format_float_2(sl_price),
                            "sl_algo_id": state.sl_algo_id,
                            "trail_gap_r": cfg.trail_gap_r,
                            "trail_buffer_r": cfg.trail_buffer_r,
                        })
                    else:
                        tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                        tp_price_str = format_to_step(tp_price, filters["tick_size"])
                        sl_price_str = format_to_step(sl_price, filters["tick_size"])

                        tp_side = "SELL" if side == 1 else "BUY"
                        tp_client_id = f"ATR_TP_{entry_close_ms or ''}"
                        # Place TP as a resting limit to avoid taker fees when possible.
                        # Verify position still exists before placing reduce-only order
                        if has_open_position(client, active_symbol):
                            tp_order = limit_order(
                                symbol=active_symbol,
                                side=tp_side,
                                quantity=qty_str,
                                price=tp_price_str,
                                client=client,
                                client_order_id=tp_client_id,
                                tick_size=filters["tick_size"],
                                reduce_only=True,
                            )
                        else:
                            tp_order = None
                            log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "entry_fill_tp"})
                        if tp_order is None:
                            log_event(cfg.log_path, {
                                "event": "tp_order_rejected",
                                "symbol": active_symbol,
                                "tp_price": format_float_2(tp_price),
                            })
                        state.tp_order_id = tp_order.get("orderId") if tp_order else None
                        state.tp_client_order_id = (tp_order.get("clientOrderId") if tp_order else None) or (tp_client_id if tp_order else None)
                        state.tp_price = tp_price
                        state.sl_price = sl_price

                        sl_side = "SELL" if side == 1 else "BUY"
                        sl_client_id = f"ATR_SL_{entry_close_ms or ''}"
                        # Compute limit price with offset for maker execution
                        sl_limit_price = compute_sl_limit_price(
                            sl_price,
                            side,
                            entry_atr,
                            cfg.sl_maker_offset_atr_mult,
                        )
                        sl_limit_price_str = format_to_step(sl_limit_price, filters["tick_size"])
                        # Verify position still exists before placing reduce-only order
                        if has_open_position(client, active_symbol):
                            sl_algo_order = algo_stop_limit_order(
                                symbol=active_symbol,
                                side=sl_side,
                                quantity=qty_str,
                                trigger_price=sl_price_str,
                                price=sl_limit_price_str,
                                client=client,
                                client_order_id=sl_client_id,
                                algo_type=cfg.algo_type,
                                working_type=cfg.algo_working_type,
                                price_protect=cfg.algo_price_protect,
                                reduce_only=True,
                            )
                        else:
                            sl_algo_order = None
                            log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "entry_fill_sl"})
                        if sl_algo_order is None:
                            log_event(cfg.log_path, {
                                "event": "sl_algo_order_rejected",
                                "symbol": active_symbol,
                                "sl_price": format_float_2(sl_price),
                            })
                        state.sl_algo_id = _extract_int(sl_algo_order, ("algoId", "algoOrderId", "orderId", "id"))
                        state.sl_algo_client_order_id = (
                            _extract_str(sl_algo_order, ("clientAlgoOrderId", "clientAlgoId", "clientOrderId", "clientId"))
                            or (sl_client_id if sl_algo_order else None)
                        )
                        state.sl_order_id = None
                        state.sl_client_order_id = None
                        state.sl_triggered = False
                        state.sl_order_time = None
                        state.entry_close_ms = None

                        log_event(cfg.log_path, {
                            "event": "entry_filled",
                            "symbol": active_symbol,
                            "order_id": order.get("orderId"),
                            "side": "LONG" if side == 1 else "SHORT",
                            "entry_price": format_float_2(entry_price),
                            "quantity": format_float_2(qty),
                            "entry_atr": format_float_2(state.active_atr),
                            "margin_usd": format_float_2(state.entry_margin_usd),
                            "leverage": state.entry_leverage,
                            "tp_price": format_float_2(tp_price),
                            "sl_price": format_float_2(sl_price),
                            "tp_order_id": state.tp_order_id,
                            "sl_algo_id": state.sl_algo_id,
                        })
                elif status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    log_event(cfg.log_path, {
                        "event": "entry_order_closed",
                        "symbol": active_symbol,
                        "order_id": state.entry_order_id,
                        "status": status,
                    })
                    state.entry_order_id = None
                    state.entry_order_time = None
                    state.pending_atr = None
                    state.pending_side = None
                    state.entry_close_ms = None
                    state.entry_margin_usd = None
                    state.entry_leverage = None
                    state.sl_algo_id = None
                    state.sl_algo_client_order_id = None
                    state.trail_r_value = None
                    state.trail_best_close_r = None
                    state.trail_stop_r = None
                    state.active_symbol = None
                else:
                    if state.entry_order_time and (time.time() - state.entry_order_time) > cfg.entry_order_timeout_seconds:
                        try:
                            client.cancel_order(symbol=active_symbol, orderId=state.entry_order_id)
                        except ClientError:
                            pass
                        log_event(cfg.log_path, {
                            "event": "entry_order_timeout",
                            "symbol": active_symbol,
                            "order_id": state.entry_order_id,
                        })
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
                        state.sl_algo_id = None
                        state.sl_algo_client_order_id = None
                        state.trail_r_value = None
                        state.trail_best_close_r = None
                        state.trail_stop_r = None
                        state.active_symbol = None

            tp_filled = False
            if active_symbol and not in_position and state.had_position:
                if cfg.use_trailing_stop:
                    sl_algo_order = query_algo_order(
                        client,
                        active_symbol,
                        algo_id=state.sl_algo_id,
                    ) if state.sl_algo_id else None
                    sl_order = query_limit_order(
                        client,
                        active_symbol,
                        order_id=state.sl_order_id,
                    ) if state.sl_order_id else None
                    sl_algo_executed = algo_order_executed(sl_algo_order)
                    sl_filled = limit_order_filled(sl_order)
                    sl_hit = sl_algo_executed or sl_filled

                    if sl_hit:
                        exit_reason = "SL"
                        exit_price = None
                        if sl_filled and sl_order:
                            try:
                                exit_price = float(sl_order.get("avgPrice") or sl_order.get("price") or 0.0)
                            except (TypeError, ValueError):
                                exit_price = None
                        if not exit_price and sl_algo_executed and sl_algo_order:
                            try:
                                exit_price = float(
                                    sl_algo_order.get("avgPrice")
                                    or sl_algo_order.get("price")
                                    or sl_algo_order.get("triggerPrice")
                                    or 0.0
                                )
                            except (TypeError, ValueError):
                                exit_price = None
                        if not exit_price:
                            exit_price = state.sl_price
                    else:
                        exit_reason = "EXIT"
                        exit_price = None
                else:
                    tp_order = query_limit_order(
                        client,
                        active_symbol,
                        order_id=state.tp_order_id,
                    ) if state.tp_order_id else None
                    sl_algo_order = query_algo_order(
                        client,
                        active_symbol,
                        algo_id=state.sl_algo_id,
                    ) if state.sl_algo_id else None
                    sl_order = query_limit_order(
                        client,
                        active_symbol,
                        order_id=state.sl_order_id,
                    ) if state.sl_order_id else None
                    tp_filled = limit_order_filled(tp_order)
                    sl_algo_executed = algo_order_executed(sl_algo_order)
                    sl_filled = limit_order_filled(sl_order)
                    sl_hit = sl_algo_executed or sl_filled

                    if tp_filled and not sl_hit:
                        exit_reason = "TP"
                        exit_price = None
                        if tp_order:
                            try:
                                exit_price = float(tp_order.get("avgPrice") or tp_order.get("price") or 0.0)
                            except (TypeError, ValueError):
                                exit_price = None
                        if not exit_price:
                            exit_price = state.tp_price
                    elif sl_hit and not tp_filled:
                        exit_reason = "SL"
                        exit_price = None
                        if sl_filled and sl_order:
                            try:
                                exit_price = float(sl_order.get("avgPrice") or sl_order.get("price") or 0.0)
                            except (TypeError, ValueError):
                                exit_price = None
                        if not exit_price and sl_algo_executed and sl_algo_order:
                            try:
                                exit_price = float(
                                    sl_algo_order.get("avgPrice")
                                    or sl_algo_order.get("price")
                                    or sl_algo_order.get("triggerPrice")
                                    or 0.0
                                )
                            except (TypeError, ValueError):
                                exit_price = None
                        if not exit_price:
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
                last_close_ms = last_close_ms_by_symbol.get(active_symbol)
                if interval_ms and state.entry_signal_ms and last_close_ms:
                    bars_held = max(0, int((last_close_ms - state.entry_signal_ms) / interval_ms))

                log_event(cfg.log_path, {
                    "event": "exit_filled",
                    "symbol": active_symbol,
                    "exit_reason": exit_reason,
                    "exit_price": format_float_2(exit_price),
                    "entry_price": format_float_2(state.entry_price),
                    "quantity": format_float_2(state.entry_qty),
                    "entry_atr": format_float_2(state.active_atr),
                    "margin_usd": format_float_2(margin_used),
                    "leverage": state.entry_leverage or current_leverage_by_symbol.get(active_symbol) or cfg.leverage,
                    "tp_order_id": state.tp_order_id,
                    "sl_order_id": state.sl_order_id,
                    "sl_algo_id": state.sl_algo_id,
                })

                cancel_open_strategy_orders(client, active_symbol)
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

                if state.tp_order_id and not tp_filled:
                    cancel_order_safely(client, active_symbol, order_id=state.tp_order_id)
                if state.sl_algo_id:
                    cancel_algo_order_safely(client, active_symbol, algo_id=state.sl_algo_id)
                if state.sl_order_id:
                    cancel_order_safely(client, active_symbol, order_id=state.sl_order_id)

                state.tp_order_id = None
                state.sl_order_id = None
                state.tp_client_order_id = None
                state.sl_client_order_id = None
                state.sl_algo_id = None
                state.sl_algo_client_order_id = None
                state.tp_price = None
                state.sl_price = None
                state.trail_r_value = None
                state.trail_best_close_r = None
                state.trail_stop_r = None
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
                state.active_symbol = None
                state.sl_triggered = False
                state.sl_order_time = None
                exit_filled = True

            if exit_filled:
                time.sleep(cfg.poll_interval_seconds)
                continue

            if active_symbol and not in_position and state.entry_price is None and (state.tp_order_id or state.sl_order_id or state.sl_algo_id):
                cancel_open_strategy_orders(client, active_symbol)
                log_event(cfg.log_path, {"event": "position_closed", "symbol": active_symbol})
                state.tp_order_id = None
                state.sl_order_id = None
                state.tp_client_order_id = None
                state.sl_client_order_id = None
                state.sl_algo_id = None
                state.sl_algo_client_order_id = None
                state.tp_price = None
                state.sl_price = None
                state.trail_r_value = None
                state.trail_best_close_r = None
                state.trail_stop_r = None
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
                state.active_symbol = None
                state.sl_triggered = False
                state.sl_order_time = None

            if in_position and active_symbol:
                filters = filters_by_symbol[active_symbol]
                tick_size = filters["tick_size"]

                try:
                    bid, ask = get_book_ticker(client, active_symbol)
                except Exception:
                    bid, ask = 0.0, 0.0

                tp_order = None
                if state.tp_order_id:
                    tp_order = query_limit_order(client, active_symbol, order_id=state.tp_order_id)
                    if limit_order_inactive(tp_order):
                        state.tp_order_id = None
                        state.tp_client_order_id = None

                if state.sl_algo_id is None:
                    sl_order = None
                    if state.sl_order_id:
                        sl_order = query_limit_order(client, active_symbol, order_id=state.sl_order_id)
                        if limit_order_inactive(sl_order):
                            state.sl_order_id = None
                            state.sl_client_order_id = None
                            state.sl_order_time = None
                            sl_order = None

                    if not state.sl_triggered and state.sl_price is not None and bid > 0 and ask > 0:
                        stop_price = float(state.sl_price)
                        stop_hit = (bid <= stop_price) if position_amt > 0 else (ask >= stop_price)
                        if stop_hit:
                            state.sl_triggered = True
                            log_event(cfg.log_path, {
                                "event": "sl_triggered",
                                "symbol": active_symbol,
                                "side": "LONG" if position_amt > 0 else "SHORT",
                                "stop_price": format_float_by_symbol(stop_price, active_symbol),
                                "bid": format_float_by_symbol(bid, active_symbol),
                                "ask": format_float_by_symbol(ask, active_symbol),
                            })

                    if state.sl_triggered:
                        # Ensure TP is canceled so we don't have two opposing LIMITs without reduceOnly.
                        if state.tp_order_id:
                            cancel_order_safely(client, active_symbol, order_id=state.tp_order_id)
                            state.tp_order_id = None
                            state.tp_client_order_id = None

                        # Track executed and remaining quantity for partial fill handling
                        executed_qty = 0.0
                        remaining_qty = abs(position_amt)

                        if state.sl_order_id is not None and sl_order:
                            status = str(sl_order.get("status", "")).upper()
                            try:
                                executed_qty = float(sl_order.get("executedQty", 0) or 0)
                            except (TypeError, ValueError):
                                executed_qty = 0.0
                            try:
                                original_qty = float(sl_order.get("origQty", 0) or 0)
                            except (TypeError, ValueError):
                                original_qty = 0.0
                            remaining_qty = original_qty - executed_qty if original_qty > 0 else abs(position_amt)

                            # Chase unfilled orders after interval (maker price chasing)
                            if status not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                                if remaining_qty > 0 and state.sl_order_time is not None:
                                    elapsed = time.time() - state.sl_order_time
                                    if elapsed >= cfg.sl_chase_interval_seconds:
                                        # Get current order price to log the chase
                                        try:
                                            old_price = float(sl_order.get("price", 0) or 0)
                                        except (TypeError, ValueError):
                                            old_price = 0.0

                                        # Cancel stale order to reprice
                                        if cancel_order_safely(client, active_symbol, order_id=state.sl_order_id):
                                            log_event(cfg.log_path, {
                                                "event": "sl_chase_cancel",
                                                "symbol": active_symbol,
                                                "old_price": format_float_by_symbol(old_price, active_symbol),
                                                "executed_qty": format_float_2(executed_qty),
                                                "remaining_qty": format_float_2(remaining_qty),
                                                "elapsed_seconds": round(elapsed, 2),
                                            })
                                            state.sl_order_id = None
                                            state.sl_client_order_id = None
                                            state.sl_order_time = None
                                            sl_order = None

                        # Place new SL chase order if needed (for remaining unfilled quantity)
                        if state.sl_order_id is None and remaining_qty > 0:
                            close_side = "SELL" if position_amt > 0 else "BUY"
                            close_qty = remaining_qty
                            close_qty_str = format_to_step(close_qty, filters["step_size"])
                            try:
                                close_qty_num = float(close_qty_str)
                            except (TypeError, ValueError):
                                close_qty_num = 0.0
                            if close_qty_num >= filters["min_qty"] and bid > 0 and ask > 0:
                                # Price at best bid/ask + 1 tick to ensure maker
                                close_price = ask if close_side == "SELL" else bid
                                close_price_str = format_price_to_tick(close_price, tick_size, side=close_side, post_only=True)
                                sl_client_id = f"ATR_SL_CHASE_{int(time.time() * 1000)}"
                                # Verify position still exists before placing reduce-only order
                                if has_open_position(client, active_symbol):
                                    new_sl_order = limit_order(
                                        symbol=active_symbol,
                                        side=close_side,
                                        quantity=close_qty_str,
                                        price=close_price_str,
                                        client=client,
                                        client_order_id=sl_client_id,
                                        tick_size=tick_size,
                                        reduce_only=True,
                                    )
                                else:
                                    new_sl_order = None
                                    log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "sl_chase"})
                                state.sl_order_id = new_sl_order.get("orderId") if new_sl_order else None
                                state.sl_client_order_id = (
                                    (new_sl_order.get("clientOrderId") if new_sl_order else None)
                                    or (sl_client_id if new_sl_order else None)
                                )
                                state.sl_order_time = time.time() if new_sl_order else None
                                log_event(cfg.log_path, {
                                    "event": "sl_chase_order",
                                    "symbol": active_symbol,
                                    "side": close_side,
                                    "quantity": format_float_2(close_qty_num),
                                    "price": format_float_by_symbol(float(close_price_str), active_symbol),
                                    "order_id": state.sl_order_id,
                                    "bid": format_float_by_symbol(bid, active_symbol),
                                    "ask": format_float_by_symbol(ask, active_symbol),
                                })
                            else:
                                log_event(cfg.log_path, {
                                    "event": "sl_chase_skip_qty",
                                    "symbol": active_symbol,
                                    "position_amt": format_float_2(position_amt),
                                    "remaining_qty": format_float_2(remaining_qty),
                                    "quantity": close_qty_str,
                                })

            # If in position and no exits placed, place them using current entryPrice
            if in_position and active_symbol and (
                (cfg.use_trailing_stop and (state.sl_algo_id is None or state.sl_price is None))
                or (not cfg.use_trailing_stop and (state.tp_order_id is None or state.sl_algo_id is None or state.sl_price is None))
            ):
                filters = filters_by_symbol[active_symbol]
                side = 1 if position_amt > 0 else -1
                entry_price = float(position.get("entryPrice", 0.0))
                qty = abs(position_amt)
                if state.entry_side is None:
                    state.entry_side = side
                entry_atr = state.active_atr or state.pending_atr or last_atr_by_symbol.get(active_symbol)
                if entry_atr is None:
                    log_event(cfg.log_path, {"event": "missing_atr", "stage": "exit_recover", "symbol": active_symbol})
                    entry_atr = 0.0
                state.active_atr = entry_atr
                state.entry_price = entry_price
                state.entry_qty = qty

                qty_str = format_to_step(qty, filters["step_size"])

                if cfg.use_trailing_stop:
                    state.tp_order_id = None
                    state.tp_client_order_id = None
                    state.tp_price = None
                    if state.trail_r_value is None:
                        state.trail_r_value = entry_atr * cfg.sl_atr_mult
                    if state.trail_best_close_r is None:
                        state.trail_best_close_r = 0.0
                    if state.trail_stop_r is None:
                        state.trail_stop_r = -1
                    sl_price = compute_trailing_sl_price(
                        entry_price,
                        side,
                        state.trail_r_value,
                        state.trail_stop_r,
                        cfg.trail_buffer_r,
                    )
                    sl_price_str = format_to_step(sl_price, filters["tick_size"])
                    if state.sl_price is None:
                        state.sl_price = sl_price
                    if state.sl_algo_id is None and not state.sl_triggered:
                        sl_side = "SELL" if side == 1 else "BUY"
                        sl_client_id = f"ATR_SL_RECOVER_{int(time.time())}"
                        # Compute limit price with offset for maker execution
                        sl_limit_price = compute_sl_limit_price(
                            sl_price,
                            side,
                            entry_atr,
                            cfg.sl_maker_offset_atr_mult,
                        )
                        sl_limit_price_str = format_to_step(sl_limit_price, filters["tick_size"])
                        # Verify position still exists before placing reduce-only order
                        if has_open_position(client, active_symbol):
                            sl_algo_order = algo_stop_limit_order(
                                symbol=active_symbol,
                                side=sl_side,
                                quantity=qty_str,
                                trigger_price=sl_price_str,
                                price=sl_limit_price_str,
                                client=client,
                                client_order_id=sl_client_id,
                                algo_type=cfg.algo_type,
                                working_type=cfg.algo_working_type,
                                price_protect=cfg.algo_price_protect,
                                reduce_only=True,
                            )
                        else:
                            sl_algo_order = None
                            log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "sl_algo_recover_trailing"})
                        if sl_algo_order is None:
                            log_event(cfg.log_path, {
                                "event": "sl_algo_recover_rejected",
                                "symbol": active_symbol,
                                "sl_price": format_float_2(sl_price),
                            })
                        state.sl_algo_id = _extract_int(sl_algo_order, ("algoId", "algoOrderId", "orderId", "id"))
                        state.sl_algo_client_order_id = (
                            _extract_str(sl_algo_order, ("clientAlgoOrderId", "clientAlgoId", "clientOrderId", "clientId"))
                            or (sl_client_id if sl_algo_order else None)
                        )
                else:
                    tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                    tp_price_str = format_to_step(tp_price, filters["tick_size"])
                    sl_price_str = format_to_step(sl_price, filters["tick_size"])
                    tp_side = "SELL" if side == 1 else "BUY"

                    if state.sl_price is None:
                        state.sl_price = sl_price

                    if state.sl_algo_id is None and not state.sl_triggered:
                        sl_side = "SELL" if side == 1 else "BUY"
                        sl_client_id = f"ATR_SL_RECOVER_{int(time.time())}"
                        # Compute limit price with offset for maker execution
                        sl_limit_price = compute_sl_limit_price(
                            sl_price,
                            side,
                            entry_atr,
                            cfg.sl_maker_offset_atr_mult,
                        )
                        sl_limit_price_str = format_to_step(sl_limit_price, filters["tick_size"])
                        # Verify position still exists before placing reduce-only order
                        if has_open_position(client, active_symbol):
                            sl_algo_order = algo_stop_limit_order(
                                symbol=active_symbol,
                                side=sl_side,
                                quantity=qty_str,
                                trigger_price=sl_price_str,
                                price=sl_limit_price_str,
                                client=client,
                                client_order_id=sl_client_id,
                                algo_type=cfg.algo_type,
                                working_type=cfg.algo_working_type,
                                price_protect=cfg.algo_price_protect,
                                reduce_only=True,
                            )
                        else:
                            sl_algo_order = None
                            log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "sl_algo_recover"})
                        if sl_algo_order is None:
                            log_event(cfg.log_path, {
                                "event": "sl_algo_recover_rejected",
                                "symbol": active_symbol,
                                "sl_price": format_float_2(sl_price),
                            })
                        state.sl_algo_id = _extract_int(sl_algo_order, ("algoId", "algoOrderId", "orderId", "id"))
                        state.sl_algo_client_order_id = (
                            _extract_str(sl_algo_order, ("clientAlgoOrderId", "clientAlgoId", "clientOrderId", "clientId"))
                            or (sl_client_id if sl_algo_order else None)
                        )

                    if state.tp_order_id is None and not state.sl_triggered:
                        tp_client_id = f"ATR_TP_RECOVER_{int(time.time())}"
                        # Verify position still exists before placing reduce-only order
                        if has_open_position(client, active_symbol):
                            tp_order = limit_order(
                                symbol=active_symbol,
                                side=tp_side,
                                quantity=qty_str,
                                price=tp_price_str,
                                client=client,
                                client_order_id=tp_client_id,
                                tick_size=filters["tick_size"],
                                reduce_only=True,
                            )
                        else:
                            tp_order = None
                            log_event(cfg.log_path, {"event": "skip_reduce_only_no_position", "symbol": active_symbol, "stage": "tp_recover"})
                        if tp_order is None:
                            log_event(cfg.log_path, {
                                "event": "tp_order_rejected",
                                "symbol": active_symbol,
                                "tp_price": format_float_2(tp_price),
                            })
                        state.tp_order_id = tp_order.get("orderId") if tp_order else None
                        state.tp_client_order_id = (tp_order.get("clientOrderId") if tp_order else None) or (tp_client_id if tp_order else None)
                        state.tp_price = tp_price

            active_symbol = state.active_symbol
            # New candle check / entry scan
            if active_symbol:
                process_symbol_candle(active_symbol, allow_entry=False)
            elif not multi_position and state.entry_order_id is None and not open_symbols:
                for symbol in symbols:
                    if process_symbol_candle(symbol, allow_entry=True):
                        active_symbol = symbol
                        break
        except ClientError as exc:
            status_code = getattr(exc, "status_code", None)
            error_code = getattr(exc, "error_code", None)
            error_message = getattr(exc, "error_message", str(exc))
            if status_code is None or status_code in {418, 429} or (isinstance(status_code, int) and status_code >= 500):
                logging.warning(
                    "Live loop Binance transient error. status=%s code=%s message=%s",
                    status_code,
                    error_code,
                    error_message,
                )
            else:
                logging.error(
                    "Live loop Binance client error. status=%s code=%s message=%s",
                    status_code,
                    error_code,
                    error_message,
                )
            log_event(
                cfg.log_path,
                {
                    "event": "client_error",
                    "status_code": status_code,
                    "error_code": error_code,
                    "message": str(error_message),
                },
            )
        except Exception as exc:
            logging.exception("Live loop error")
            log_event(cfg.log_path, {"event": "error", "message": str(exc)})

        time.sleep(cfg.poll_interval_seconds)

# -----------------------------
# Example usage
# -----------------------------
def run_backtest(use_lib: bool = True, parallel_workers: int = 15) -> None:
    """
    Run backtest with optional Look-Inside-Bar (LIB) mode.
    
    Args:
        use_lib: If True, use 1s candles from Spot API for more realistic execution simulation.
                 If False, use original 1m-only backtest.
        parallel_workers: Number of parallel threads for fetching 1s data (default 10).
                         Use 1 for sequential fetching.
    """
    import time as time_module
    
    total_start = time_module.perf_counter()
    
    # Configuration
    cfg = BacktestConfig()
    symbol_futures = "BTCUSDC"
    symbol_spot = "BTCUSDC"
    start_utc = "2026-01-06 00:00:00"
    end_utc = "2026-01-10 00:00:00"
    
    # 1) Fetch signal candles from Futures (for signals)
    signal_interval = cfg.signal_interval
    print(f"Fetching {signal_interval} klines from Futures API...")
    fetch_signal_start = time_module.perf_counter()
    df_signal = fetch_klines_um_futures(
        symbol=symbol_futures,
        interval=signal_interval,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    fetch_signal_time = time_module.perf_counter() - fetch_signal_start
    print(f"Loaded {len(df_signal):,} {signal_interval} candles from Futures ({fetch_signal_time:.2f}s)")
    
    fetch_1s_time = 0.0
    if use_lib:
        # 2) Fetch 1s data from Spot (for Look-Inside-Bar execution)
        if parallel_workers > 1:
            print(f"\nFetching 1s klines from Spot API (parallel with {parallel_workers} workers)...")
            fetch_1s_start = time_module.perf_counter()
            df_1s = fetch_klines_spot_parallel(
                symbol=symbol_spot,
                interval="1s",
                start_utc=start_utc,
                end_utc=end_utc,
                max_workers=parallel_workers,
            )
        else:
            print(f"\nFetching 1s klines from Spot API (sequential)...")
            fetch_1s_start = time_module.perf_counter()
            df_1s = fetch_klines_spot(
                symbol=symbol_spot,
                interval="1s",
                start_utc=start_utc,
                end_utc=end_utc,
            )
        fetch_1s_time = time_module.perf_counter() - fetch_1s_start
        print(f"Loaded {len(df_1s):,} 1s candles from Spot ({fetch_1s_time:.2f}s)")
        
        # Optional: Save 1s data to parquet for faster re-runs
        # df_1s.to_parquet("btcusdc_1s_cache.parquet")
        
        # 3) Run LIB backtest
        print(f"\nRunning Look-Inside-Bar backtest...")
        backtest_start = time_module.perf_counter()
        trades, df_bt, stats, trailing_df = backtest_atr_grinder_lib(df_signal, df_1s, cfg)
        backtest_time = time_module.perf_counter() - backtest_start
    else:
        # Original backtest (1m only)
        print(f"\nRunning standard {signal_interval} backtest...")
        backtest_start = time_module.perf_counter()
        trades, df_bt, stats, trailing_df = backtest_atr_grinder(df_signal, cfg)
        backtest_time = time_module.perf_counter() - backtest_start

    total_time = time_module.perf_counter() - total_start

    # 4) Output
    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Save results
    trades.to_csv("trades.csv", index=False)
    df_bt.to_csv("backtest_series.csv")
    
    if not trailing_df.empty:
        trailing_df.to_csv("trailing_stops.csv", index=False)
        print(f"\nSaved: trades.csv, backtest_series.csv, trailing_stops.csv ({len(trailing_df)} trailing stop updates)")
    else:
        print("\nSaved: trades.csv, backtest_series.csv")
    
    if use_lib:
        # Show LIB-specific stats
        if not trades.empty and "lib_mode" in trades.columns:
            lib_trades = trades["lib_mode"].sum()
            total_trades = len(trades)
            print(f"\nLIB mode: {lib_trades}/{total_trades} trades executed with 1s resolution")
    
    # Timing summary
    print(f"\n=== TIMING ===")
    print(f"Fetch {signal_interval} data: {fetch_signal_time:>8.2f}s")
    if use_lib:
        print(f"Fetch 1s data:  {fetch_1s_time:>8.2f}s")
    print(f"Backtest:       {backtest_time:>8.2f}s")
    print(f"Total:          {total_time:>8.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATR 1m strategy")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument(
        "--no-lib",
        action="store_true",
        help="Disable Look-Inside-Bar mode (use 1m-only backtest instead of 1s resolution)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of parallel workers for fetching 1s data (default: 15, use 1 for sequential)",
    )
    args = parser.parse_args()

    if args.mode == "live":
        live_cfg = LiveConfig()
        run_live(live_cfg)
    else:
        run_backtest(use_lib=not args.no_lib, parallel_workers=args.workers)


if __name__ == "__main__":
    main()
