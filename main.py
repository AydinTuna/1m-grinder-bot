"""
ATR-based 1d strategy (swing context + large body trigger, trailing stop only)

Signal timeframe: 1d candles for entry signals and swing detection.
Trailing stop: 1m candles for Look-In-Bar trailing stop tracking.

Rules:
- Compute ATR (EMA) on 1d candles.
- Compute swing highs/lows on 1d candles (fractal/pivot left/right).
- If candle body >= thr2*ATR and price is near a swing high/low:
  - Swing high: red => SHORT; green close below => SHORT; green close above => LONG (limit at mid-body)
  - Swing low: green => LONG; red close above => LONG; red close below => SHORT (limit at mid-body)
- If candle body >= thr2*ATR and price is not near swing levels: enter in the same direction as the candle.
- Exit: Trailing stop only (no TP/SL on entry)
  - No initial stop until price reaches first ladder step (trail_gap_r, default 1.25R).
  - When price reaches 1.25R, first stop is placed at breakeven (0R).
  - Trailing stop is a ladder based on 1m candle close in R units (gap = trail_gap_r, buffer = trail_buffer_r).
  - Stop execution via Binance algo conditional orders.
  - Positions can go negative without being stopped - relies on strategy edge.

Live trading:
- Trades all USDT perpetual contracts on Binance (no USDC).
- Static position sizing: margin_usd x leverage (default: 5 USD x 20).

Implementation details:
- Body = abs(close - open)
- Entries use limit orders at mid-body price.
- Trailing stop updates occur on 1m candle close.
- ATR frozen at entry for R-based trailing.
- Fees and slippage are modeled in backtest.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import hmac
import http.client
import io
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from pathlib import Path
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Iterable, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from config import (
    BacktestConfig,
    LiveConfig,
    LIVE_TRADE_FIELDS,
    LIVE_SIGNAL_FIELDS,
    BACKTEST_SIGNAL_FIELDS,
    BACKTEST_TRADE_FIELDS,
    STRATEGY_VERSION,
    STRATEGY_VERSION_NOTE,
    OUTPUT_DIR,
    get_output_path,
    get_backtest_signals_dir,
    get_backtest_trades_dir,
    get_backtest_stats_dir,
    get_backtest_charts_dir,
    get_live_signals_dir,
    get_live_trades_dir,
    get_live_logs_dir,
    get_data_klines_dir,
    get_comparison_dir,
    ensure_output_dirs,
    save_config_snapshot,
    generate_trade_id,
)

FORCED_EXIT_REASON = "FORCED_EXIT_4H"
from swing_levels import (
    build_swing_atr_signals,
    build_market_structure_signals,
    compute_confirmed_swing_levels,
    compute_market_structure_levels,
    classify_market_structure,
    detect_swings,
    resample_ohlcv,
    MarketStructurePoint,
)

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
# Symbol discovery (public API)
# -----------------------------
def is_tradeable_symbol(symbol: Optional[str]) -> bool:
    if not symbol:
        return False
    return symbol.isascii() and symbol.isalnum()


def get_all_usdt_perpetuals_public() -> List[str]:
    """
    Fetch all USDT perpetual contract symbols from Binance Futures public API.
    No API keys required.
    
    Returns:
        List of symbol strings (e.g., ["BTCUSDT", "ETHUSDT", ...])
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        req = urllib.request.Request(url)
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to fetch exchange info: {e}")
        return []
    
    symbols = []
    for s in data.get("symbols", []):
        # Only include PERPETUAL contracts with USDT as quote asset
        if (s.get("contractType") == "PERPETUAL" 
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"):
            symbol = s.get("symbol")
            if is_tradeable_symbol(symbol):
                symbols.append(symbol)
    
    return sorted(symbols)


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

    # Compute market structure (HH/HL/LL/LH) levels
    structure_points, hh_series, hl_series, ll_series, lh_series = compute_market_structure_levels(
        df,
        df["atr"],
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["hh_level"] = hh_series
    df["hl_level"] = hl_series
    df["ll_level"] = ll_series
    df["lh_level"] = lh_series

    # Also compute raw swing levels for reference
    swing_high, swing_low = compute_confirmed_swing_levels(
        df,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["swing_high_level"] = swing_high
    df["swing_low_level"] = swing_low

    # Generate signals based on market structure
    df["signal"], df["signal_atr"], df["signal_entry_price"], df["signal_reason"], df["trend"] = build_market_structure_signals(
        df,
        df["atr"],
        structure_points,
        body_atr_mult=cfg.thr2,
        structure_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "signal_atr"] = np.nan
        df.loc[warmup_idx, "signal_entry_price"] = np.nan
        df.loc[warmup_idx, "signal_reason"] = ""
        df.loc[warmup_idx, "trend"] = 0

    # Equity curve (USD)
    equity = cfg.initial_capital
    equity_series = pd.Series(np.nan, index=df.index, dtype=float)
    equity_series.iloc[0] = equity

    positions: List[Dict[str, Any]] = []
    pending_entries: List[Dict[str, Any]] = []

    # Trade ID tracking for signal-trade linking
    trade_counter = 0

    trades: List[Dict] = []
    trailing_stop_updates: List[Dict] = []

    idx = df.index.to_list()
    daily_date = idx[0].date() if idx else None

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

    def create_position(
        side: int,
        entry_price: float,
        signal_atr_value: float,
        entry_time: pd.Timestamp,
        entry_index: int,
        signal_reason: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        nonlocal trade_counter
        trade_margin, trade_leverage, _ = resolve_trade_sizing(
            entry_price=entry_price,
            atr_value=signal_atr_value,
            sl_atr_mult=1.0,
            margin_cap=cfg.initial_capital,
            max_leverage=cfg.leverage,
            min_leverage=cfg.leverage,
            target_loss_usd=None,
        )
        if trade_margin is None or trade_leverage is None:
            return None

        if cfg.use_trailing_stop:
            target_price = None
            trail_r_value = signal_atr_value * 1.0  # R = 1x ATR
            trail_best_close_r = 0.0
            trail_stop_r = cfg.trail_initial_stop_r  # Initial stop R level from config
            stop_price = None  # No stop until price reaches breakeven
        else:
            target_price, stop_price = compute_tp_sl_prices(
                entry_price,
                side,
                signal_atr_value,
                2.0,  # tp_atr_mult (not used with trailing)
                1.0,  # sl_atr_mult
            )
            trail_r_value = None
            trail_best_close_r = None
            trail_stop_r = None

        trade_counter += 1
        side_str = "LONG" if side == 1 else "SHORT"
        trade_id = generate_trade_id(df.attrs.get("symbol", "UNK"), str(entry_time), side_str)

        return {
            "position": side,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "trail_r_value": trail_r_value,
            "trail_best_close_r": trail_best_close_r,
            "trail_stop_r": trail_stop_r,
            "entry_time": entry_time,
            "entry_index": entry_index,
            "used_signal_atr": signal_atr_value,
            "active_margin": trade_margin,
            "active_leverage": trade_leverage,
            "active_notional": trade_margin * trade_leverage,
            "trade_id": trade_id,
            "signal_reason": signal_reason,
        }

    for i in range(1, len(df)):
        t = idx[i]
        prev_t = idx[i - 1]
        if daily_date is not None:
            current_date = t.date()
            if current_date != daily_date:
                daily_date = current_date
                daily_pnl_usd = 0.0

        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])
        c = float(df.at[t, "close"])

        # --- EXIT logic ---
        # Exit only on candle CLOSE (not intra-candle high/low) for consistency
        # with higher timeframe close exit behavior
        for pos in list(positions):
            position = pos["position"]
            entry_price = pos["entry_price"]
            stop_price = pos["stop_price"]
            target_price = pos["target_price"]

            if entry_price is None or stop_price is None:
                continue

            sl_hit = (c <= stop_price) if position == 1 else (c >= stop_price)

            exited = False
            exit_reason = None
            exit_price = None

            if cfg.use_trailing_stop:
                if sl_hit:
                    exit_reason = "SL"
                    exit_price = apply_slippage(c, position, is_entry=False)
                    exited = True
            else:
                if target_price is not None:
                    tp_hit = (c >= target_price) if position == 1 else (c <= target_price)
                else:
                    tp_hit = False

                if sl_hit and tp_hit:
                    exit_reason = "SL"
                    exit_price = apply_slippage(c, position, is_entry=False)
                    exited = True
                elif tp_hit:
                    exit_reason = "TP"
                    exit_price = apply_slippage(c, position, is_entry=False)
                    exited = True
                elif sl_hit:
                    exit_reason = "SL"
                    exit_price = apply_slippage(c, position, is_entry=False)
                    exited = True

            if exited:
                roi_net = net_roi_from_prices(entry_price, exit_price, position, pos["active_leverage"])
                pnl_net = roi_net * pos["active_margin"]
                equity += pnl_net

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "signal_atr": pos["used_signal_atr"],
                    "stop_price": stop_price,
                    "exit_reason": exit_reason,
                    "roi_net": roi_net,
                    "pnl_net": pnl_net,
                    "margin_used": pos["active_margin"],
                    "notional": pos["active_notional"],
                    "equity_after": equity,
                    "bars_held": i - (pos["entry_index"] if pos["entry_index"] is not None else i),
                    "trade_id": pos["trade_id"],
                    "signal_reason": pos["signal_reason"],
                    "strategy_version": STRATEGY_VERSION,
                })

                positions.remove(pos)

        # --- Trailing stop update (candle close) ---
        # Note: stop_price may be None initially (no stop until breakeven)
        if cfg.use_trailing_stop:
            for pos in positions:
                position = pos["position"]
                entry_price = pos["entry_price"]
                if entry_price is None:
                    continue

                c = float(df.at[t, "close"])
                h = float(df.at[t, "high"])
                l = float(df.at[t, "low"])
                side_str = "LONG" if position == 1 else "SHORT"
                prev_stop_price = pos["stop_price"]

                if cfg.trailing_mode == "dynamic_atr":
                    signal_atr_for_trail = (
                        pos["trail_r_value"] if pos["trail_r_value"] is not None else float(df.at[t, "atr"])
                    )

                    if cfg.dynamic_trail_price_source == "high_low":
                        trail_price = h if position == 1 else l
                    else:
                        trail_price = c
                    atr_stop = compute_dynamic_atr_stop(
                        trail_price, signal_atr_for_trail, cfg.dynamic_trail_atr_mult, position
                    )

                    new_stop = atr_stop

                    activation_threshold = signal_atr_for_trail * cfg.dynamic_trail_activation_r
                    if position == 1:
                        price_moved_enough = c >= entry_price + activation_threshold
                        should_set_stop = price_moved_enough
                    else:
                        price_moved_enough = c <= entry_price - activation_threshold
                        should_set_stop = price_moved_enough

                    stop_moved = should_set_stop and (
                        pos["stop_price"] is None or new_stop != pos["stop_price"]
                    )
                    if stop_moved:
                        pos["stop_price"] = new_stop

                    trailing_stop_updates.append({
                        "timestamp": t,
                        "entry_time": pos["entry_time"],
                        "side": side_str,
                        "entry_price": entry_price,
                        "close_price": c,
                        "trail_price": trail_price,
                        "signal_atr": signal_atr_for_trail,
                        "atr_stop": atr_stop,
                        "prev_stop_price": prev_stop_price,
                        "new_stop_price": pos["stop_price"],
                        "trailing_mode": "dynamic_atr",
                        "dynamic_trail_atr_mult": cfg.dynamic_trail_atr_mult,
                        "dynamic_trail_activation_r": cfg.dynamic_trail_activation_r,
                        "dynamic_trail_price_source": cfg.dynamic_trail_price_source,
                        "price_moved_enough": price_moved_enough,
                        "stop_moved": stop_moved,
                    })

                elif (
                    cfg.trailing_mode == "r_ladder"
                    and pos["trail_r_value"] is not None
                    and pos["trail_best_close_r"] is not None
                    and pos["trail_stop_r"] is not None
                ):
                    close_r = compute_close_r(entry_price, c, position, pos["trail_r_value"])
                    if close_r is not None:
                        prev_best_r = pos["trail_best_close_r"]
                        prev_stop_r = pos["trail_stop_r"]

                        pos["trail_best_close_r"] = max(pos["trail_best_close_r"], close_r)
                        next_stop_r = compute_trailing_stop_r(
                            pos["trail_best_close_r"], pos["trail_stop_r"], cfg.trail_gap_r
                        )

                        should_place_stop = (
                            (pos["stop_price"] is None and next_stop_r >= 0) or
                            (next_stop_r > pos["trail_stop_r"] and next_stop_r >= 0)
                        )
                        if should_place_stop:
                            new_stop_price = compute_trailing_sl_price(
                                entry_price,
                                position,
                                pos["trail_r_value"],
                                next_stop_r,
                                cfg.trail_buffer_r,
                            )

                            trailing_stop_updates.append({
                                "timestamp": t,
                                "entry_time": pos["entry_time"],
                                "side": side_str,
                                "entry_price": entry_price,
                                "close_price": c,
                                "close_r": close_r,
                                "prev_best_r": prev_best_r,
                                "new_best_r": pos["trail_best_close_r"],
                                "prev_stop_r": prev_stop_r,
                                "new_stop_r": next_stop_r,
                                "prev_stop_price": prev_stop_price,
                                "new_stop_price": new_stop_price,
                                "r_value": pos["trail_r_value"],
                                "trail_gap_r": cfg.trail_gap_r,
                                "trail_buffer_r": cfg.trail_buffer_r,
                                "trailing_mode": "r_ladder",
                                "stop_moved": True,
                            })

                            pos["trail_stop_r"] = next_stop_r
                            pos["stop_price"] = new_stop_price
                        else:
                            trailing_stop_updates.append({
                                "timestamp": t,
                                "entry_time": pos["entry_time"],
                                "side": side_str,
                                "entry_price": entry_price,
                                "close_price": c,
                                "close_r": close_r,
                                "prev_best_r": pos["trail_best_close_r"],
                                "new_best_r": pos["trail_best_close_r"],
                                "prev_stop_r": pos["trail_stop_r"],
                                "new_stop_r": pos["trail_stop_r"],
                                "prev_stop_price": pos["stop_price"],
                                "new_stop_price": pos["stop_price"],
                                "r_value": pos["trail_r_value"],
                                "trail_gap_r": cfg.trail_gap_r,
                                "trail_buffer_r": cfg.trail_buffer_r,
                                "trailing_mode": "r_ladder",
                                "stop_moved": False,
                            })

        # --- ENTRY logic (signal on prev candle; optional mid-body limit) ---
        if pending_entries:
            pending_entries = [
                pending for pending in pending_entries
                if (i - pending["created_index"]) < limit_timeout_bars
            ]

        for pending in list(pending_entries):
            fill_raw = maybe_fill_limit(
                int(pending["side"]),
                float(pending["limit_price"]),
                o,
                h,
                l,
            )
            if fill_raw is None:
                continue

            entry_price = apply_slippage(float(fill_raw), int(pending["side"]), is_entry=True)
            pos = create_position(
                int(pending["side"]),
                entry_price,
                float(pending["signal_atr"]),
                t,
                i,
                pending.get("signal_reason"),
            )
            pending_entries.remove(pending)
            if pos is not None:
                positions.append(pos)

        prev_signal = int(df.at[prev_t, "signal"])
        prev_signal_atr = df.at[prev_t, "signal_atr"]
        prev_entry_price = df.at[prev_t, "signal_entry_price"]
        prev_signal_reason = df.at[prev_t, "signal_reason"] if "signal_reason" in df.columns else None

        if prev_signal != 0 and not (isinstance(prev_signal_atr, float) and math.isnan(prev_signal_atr)):
            side = prev_signal
            signal_atr_value = float(prev_signal_atr)

            if prev_entry_price is not None and not (
                isinstance(prev_entry_price, float) and math.isnan(prev_entry_price)
            ):
                fill_raw = maybe_fill_limit(side, float(prev_entry_price), o, h, l)
                if fill_raw is None:
                    pending_entries.append({
                        "side": side,
                        "limit_price": float(prev_entry_price),
                        "signal_atr": signal_atr_value,
                        "created_index": i,
                        "signal_reason": prev_signal_reason,
                    })
                else:
                    entry_price = apply_slippage(float(fill_raw), side, is_entry=True)
                    pos = create_position(side, entry_price, signal_atr_value, t, i, prev_signal_reason)
                    if pos is not None:
                        positions.append(pos)
            else:
                entry_price = apply_slippage(o, side, is_entry=True)
                pos = create_position(side, entry_price, signal_atr_value, t, i, prev_signal_reason)
                if pos is not None:
                    positions.append(pos)

        equity_series.at[t] = equity

    # Close open positions at end of backtest (mark-to-market)
    if positions:
        last_close = float(df.iloc[-1]["close"])
        for pos in list(positions):
            position = pos["position"]
            entry_price = pos["entry_price"]
            if entry_price is None:
                positions.remove(pos)
                continue
            exit_price = apply_slippage(last_close, position, is_entry=False)
            roi_net = net_roi_from_prices(entry_price, exit_price, position, pos["active_leverage"])
            pnl_net = roi_net * pos["active_margin"]
            equity += pnl_net

            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": df.index[-1],
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "signal_atr": pos["used_signal_atr"],
                "stop_price": pos["stop_price"],
                "exit_reason": "END_OF_BACKTEST",
                "forced_exit": False,
                "roi_net": roi_net,
                "pnl_net": pnl_net,
                "margin_used": pos["active_margin"],
                "notional": pos["active_notional"],
                "equity_after": equity,
                "bars_held": len(df) - 1 - (pos["entry_index"] if pos["entry_index"] is not None else 0),
                "trade_id": pos["trade_id"],
                "signal_reason": pos["signal_reason"],
                "strategy_version": STRATEGY_VERSION,
            })
            positions.remove(pos)

    df["equity"] = equity_series.ffill()

    trades_df = pd.DataFrame(trades)
    trailing_df = pd.DataFrame(trailing_stop_updates)
    stats = compute_stats(trades_df, df, cfg)

    return trades_df, df, stats, trailing_df


def compute_stats(trades_df: pd.DataFrame, df_bt: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "initial_capital": float(cfg.initial_capital),
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


def compute_signal_stats(
    signals_df: pd.DataFrame, 
    trades_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate performance metrics grouped by signal_reason and symbol.
    
    Args:
        signals_df: DataFrame of signals with columns: timestamp, symbol, side, signal, 
                   signal_atr, entry_price, signal_reason, trade_id (optional)
        trades_df: DataFrame of trades with columns: entry_time, exit_time, side, 
                  entry_price, exit_price, roi_net, pnl_net, trade_id, signal_reason
    
    Returns:
        Tuple of (stats_by_signal, stats_by_symbol) DataFrames
    """
    # Initialize empty results
    stats_by_signal = pd.DataFrame()
    stats_by_symbol = pd.DataFrame()
    
    if trades_df.empty:
        return stats_by_signal, stats_by_symbol
    
    # Stats by signal_reason
    if "signal_reason" in trades_df.columns:
        signal_groups = trades_df.groupby("signal_reason")
        stats_by_signal = signal_groups.agg(
            total_trades=("pnl_net", "count"),
            wins=("pnl_net", lambda x: (x > 0).sum()),
            losses=("pnl_net", lambda x: (x <= 0).sum()),
            total_pnl=("pnl_net", "sum"),
            avg_pnl=("pnl_net", "mean"),
            avg_roi=("roi_net", "mean"),
            best_trade=("pnl_net", "max"),
            worst_trade=("pnl_net", "min"),
        ).reset_index()
        
        # Calculate win rate
        stats_by_signal["win_rate"] = stats_by_signal["wins"] / stats_by_signal["total_trades"]
        
        # Calculate signals that didn't become trades (if signals_df available)
        if not signals_df.empty and "signal_reason" in signals_df.columns:
            signal_counts = signals_df.groupby("signal_reason").size().reset_index(name="total_signals")
            stats_by_signal = stats_by_signal.merge(signal_counts, on="signal_reason", how="left")
            stats_by_signal["total_signals"] = stats_by_signal["total_signals"].fillna(stats_by_signal["total_trades"])
            stats_by_signal["signals_skipped"] = stats_by_signal["total_signals"] - stats_by_signal["total_trades"]
        else:
            stats_by_signal["total_signals"] = stats_by_signal["total_trades"]
            stats_by_signal["signals_skipped"] = 0
        
        # Reorder columns
        cols_order = [
            "signal_reason", "total_signals", "total_trades", "signals_skipped",
            "wins", "losses", "win_rate", "total_pnl", "avg_pnl", "avg_roi",
            "best_trade", "worst_trade"
        ]
        stats_by_signal = stats_by_signal[[c for c in cols_order if c in stats_by_signal.columns]]
        
        # Sort by total PnL descending
        stats_by_signal = stats_by_signal.sort_values("total_pnl", ascending=False)
    
    # Stats by symbol
    if "symbol" in trades_df.columns:
        symbol_groups = trades_df.groupby("symbol")
        stats_by_symbol = symbol_groups.agg(
            total_trades=("pnl_net", "count"),
            wins=("pnl_net", lambda x: (x > 0).sum()),
            losses=("pnl_net", lambda x: (x <= 0).sum()),
            total_pnl=("pnl_net", "sum"),
            avg_pnl=("pnl_net", "mean"),
            avg_roi=("roi_net", "mean"),
            best_trade=("pnl_net", "max"),
            worst_trade=("pnl_net", "min"),
        ).reset_index()
        
        stats_by_symbol["win_rate"] = stats_by_symbol["wins"] / stats_by_symbol["total_trades"]
        
        # Sort by total PnL descending
        stats_by_symbol = stats_by_symbol.sort_values("total_pnl", ascending=False)
    
    return stats_by_signal, stats_by_symbol


def print_signal_stats(stats_by_signal: pd.DataFrame, stats_by_symbol: pd.DataFrame) -> None:
    """Print signal performance statistics in a formatted table."""
    
    if not stats_by_signal.empty:
        print("\n" + "-" * 80)
        print("PER-SIGNAL-TYPE PERFORMANCE:")
        print("-" * 80)
        print(f"  {'Signal Reason':<30} {'Signals':>8} {'Traded':>7} {'Wins':>5} {'Win%':>7} {'Total PnL':>12} {'Avg PnL':>10}")
        print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*12} {'-'*10}")
        
        for _, row in stats_by_signal.iterrows():
            reason = str(row.get("signal_reason", "unknown"))[:28]
            signals = int(row.get("total_signals", row.get("total_trades", 0)))
            traded = int(row.get("total_trades", 0))
            wins = int(row.get("wins", 0))
            win_rate = row.get("win_rate", 0) * 100
            total_pnl = row.get("total_pnl", 0)
            avg_pnl = row.get("avg_pnl", 0)
            
            print(f"  {reason:<30} {signals:>8} {traded:>7} {wins:>5} {win_rate:>6.1f}% {total_pnl:>+11.2f} {avg_pnl:>+9.3f}")
    
    if not stats_by_symbol.empty and len(stats_by_symbol) > 1:
        print("\n" + "-" * 80)
        print("TOP 10 SYMBOLS BY PnL:")
        print("-" * 80)
        print(f"  {'Symbol':<15} {'Trades':>7} {'Wins':>5} {'Win%':>7} {'Total PnL':>12} {'Avg PnL':>10}")
        print(f"  {'-'*13} {'-'*7} {'-'*5} {'-'*7} {'-'*12} {'-'*10}")
        
        for _, row in stats_by_symbol.head(10).iterrows():
            symbol = str(row.get("symbol", "unknown"))[:13]
            trades = int(row.get("total_trades", 0))
            wins = int(row.get("wins", 0))
            win_rate = row.get("win_rate", 0) * 100
            total_pnl = row.get("total_pnl", 0)
            avg_pnl = row.get("avg_pnl", 0)
            
            print(f"  {symbol:<15} {trades:>7} {wins:>5} {win_rate:>6.1f}% {total_pnl:>+11.2f} {avg_pnl:>+9.3f}")


# -----------------------------
# Live trading (Binance Futures)
# -----------------------------
@dataclass
class PositionState:
    """Per-position state for a single symbol tracked by the strategy."""
    symbol: str
    entry_price: float
    entry_side: int  # +1 long, -1 short
    entry_qty: float
    entry_atr: float  # ATR at entry (frozen for R-based trailing)
    trail_r_value: float  # R value = entry_atr (for trailing stop calculation)
    trail_best_close_r: float = 0.0  # best close in R units
    trail_stop_r: int = -1  # current trailing stop level in R (-1 = no stop until 1R profit)
    sl_algo_id: Optional[int] = None
    sl_algo_client_order_id: Optional[str] = None
    sl_price: Optional[float] = None
    entry_time_iso: str = ""
    entry_close_ms: int = 0
    entry_margin_usd: float = 0.0
    entry_leverage: int = 20
    trade_id: Optional[str] = None  # Unique trade ID for signal-trade linking
    signal_reason: Optional[str] = None  # Signal type that triggered this trade


@dataclass 
class PendingEntry:
    """Tracks a pending entry order for a symbol."""
    symbol: str
    order_id: int
    order_time: float
    pending_atr: float
    pending_side: int  # +1 long, -1 short
    entry_close_ms: int
    trade_id: Optional[str] = None  # Unique trade ID for signal-trade linking
    signal_reason: Optional[str] = None  # Signal type that triggered this entry


@dataclass
class LiveState:
    """Global state for multi-position live trading."""
    # Active positions tracked by strategy (symbol -> PositionState)
    positions: Dict[str, PositionState] = None  # type: ignore
    # Pending entry orders (symbol -> PendingEntry)
    pending_entries: Dict[str, PendingEntry] = None  # type: ignore
    # Last candle close times for signal generation
    last_signal_close_ms: Dict[str, int] = None  # type: ignore
    # Last candle close times for trailing stop updates
    last_trail_close_ms: Dict[str, int] = None  # type: ignore
    # Leverage set per symbol
    current_leverage_by_symbol: Dict[str, int] = None  # type: ignore
    
    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {}
        if self.pending_entries is None:
            self.pending_entries = {}
        if self.last_signal_close_ms is None:
            self.last_signal_close_ms = {}
        if self.last_trail_close_ms is None:
            self.last_trail_close_ms = {}
        if self.current_leverage_by_symbol is None:
            self.current_leverage_by_symbol = {}
    
    def position_count(self) -> int:
        """Returns the number of active strategy positions."""
        return len(self.positions)
    
    def can_open_position(self, max_positions: int) -> bool:
        """Check if we can open another position."""
        if max_positions <= 0:
            return True
        return self.position_count() < max_positions
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an active position for symbol."""
        return symbol in self.positions
    
    def has_pending_entry(self, symbol: str) -> bool:
        """Check if we have a pending entry for symbol."""
        return symbol in self.pending_entries


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_within_entry_window(window_minutes: int) -> bool:
    """
    Check if current UTC time is within the entry window after daily candle close.
    
    Daily candle closes at 00:00 UTC. This function returns True if we're within
    `window_minutes` after midnight UTC.
    
    Args:
        window_minutes: Number of minutes after 00:00 UTC to allow entries.
                       Set to 0 to disable this check (always returns True).
    
    Returns:
        True if within the entry window or check is disabled, False otherwise.
    """
    if window_minutes <= 0:
        return True  # Check disabled
    
    now = datetime.now(timezone.utc)
    minutes_since_midnight = now.hour * 60 + now.minute
    return minutes_since_midnight < window_minutes


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


def get_precision_from_tick_size(tick_size: float) -> int:
    """
    Get decimal precision from tick_size.
    E.g., tick_size=0.0001 -> precision=4
          tick_size=0.01 -> precision=2
          tick_size=1.0 -> precision=0
    """
    if tick_size <= 0:
        return 8  # default fallback
    tick_str = f"{tick_size:.10f}".rstrip('0')
    if '.' in tick_str:
        return len(tick_str.split('.')[1])
    return 0


def format_price_with_precision(value: Optional[float], tick_size: float) -> Optional[str]:
    """
    Format a price value with precision derived from tick_size.
    Returns string representation with proper decimal places.
    """
    if value is None:
        return None
    try:
        precision = get_precision_from_tick_size(tick_size)
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return None


def format_price_for_logging(value: Optional[float], tick_size: Optional[float]) -> str:
    """
    Format a price value for logging (CSV, JSON).
    Uses tick_size precision if available, otherwise falls back to 8 decimals.
    """
    if value is None:
        return ""
    try:
        if tick_size and tick_size > 0:
            precision = get_precision_from_tick_size(tick_size)
        else:
            precision = 8  # fallback for unknown symbols
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return ""


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
    Keeps a post-only LIMIT order outside the spread by at least 2 ticks.
    BUY: price <= bid - 2*tick
    SELL: price >= ask + 2*tick
    """
    if tick_size <= 0:
        return price
    side_u = str(side).upper()
    offset = 2 * tick_size
    if side_u == "BUY":
        return min(price, bid - offset)
    if side_u == "SELL":
        return max(price, ask + offset)
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


def get_all_usdt_perpetuals(client: BinanceUMFuturesREST) -> List[str]:
    """
    Fetch all USDT perpetual contract symbols from Binance Futures.
    Excludes USDC pairs and non-perpetual contracts.
    
    Returns:
        List of symbol strings (e.g., ["BTCUSDT", "ETHUSDT", ...])
    """
    info = client.exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        # Only include PERPETUAL contracts with USDT as quote asset
        if (s.get("contractType") == "PERPETUAL" 
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"):
            symbol = s.get("symbol")
            if is_tradeable_symbol(symbol):
                symbols.append(symbol)
    return sorted(symbols)


def get_all_usdt_perpetuals_with_filters(client: BinanceUMFuturesREST) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Fetch all USDT perpetual symbols AND their filters in a single API call.
    Much faster than calling get_symbol_filters() for each symbol individually.
    
    Returns:
        Tuple of (symbols_list, filters_by_symbol_dict)
    """
    info = client.exchange_info()
    symbols = []
    filters_by_symbol: Dict[str, Dict[str, float]] = {}
    
    for s in info.get("symbols", []):
        # Only include PERPETUAL contracts with USDT as quote asset
        if (s.get("contractType") == "PERPETUAL" 
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"):
            symbol = s.get("symbol")
            if not is_tradeable_symbol(symbol):
                continue
            symbols.append(symbol)
            
            # Extract filters
            tick_size = None
            step_size = None
            min_qty = None
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                elif f.get("filterType") == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    min_qty = float(f["minQty"])
            
            if tick_size is not None and step_size is not None and min_qty is not None:
                filters_by_symbol[symbol] = {
                    "tick_size": tick_size,
                    "step_size": step_size,
                    "min_qty": min_qty,
                }
    
    return sorted(symbols), filters_by_symbol


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


def market_order(
    symbol: str,
    side: str,
    quantity: str,
    client: BinanceUMFuturesREST,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
) -> Optional[Dict[str, object]]:
    """
    Places a market order on Binance Futures.
    Used for emergency position closes when SL cannot be placed.
    """
    params: Dict[str, object] = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
    }
    if client_order_id:
        params["newClientOrderId"] = client_order_id
    if reduce_only:
        params["reduceOnly"] = True

    try:
        return client.new_order(**params)
    except ClientError as error:
        error_code = getattr(error, "error_code", None)
        status_code = getattr(error, "status_code", None)
        error_message = getattr(error, "error_message", None)
        logging.error(
            "Market order error. status: %s, code: %s, message: %s",
            status_code,
            error_code,
            error_message,
        )
        return None


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
    price: Optional[str],
    client: BinanceUMFuturesREST,
    *,
    client_order_id: Optional[str] = None,
    algo_type: str = "CONDITIONAL",
    working_type: str = "CONTRACT_PRICE",
    price_protect: bool = False,
    reduce_only: bool = True,
    order_type: str = "STOP",
) -> Optional[Dict[str, object]]:
    order_type_clean = str(order_type or "STOP").upper()
    if order_type_clean not in {"STOP", "STOP_MARKET"}:
        order_type_clean = "STOP"
    base_params: Dict[str, object] = {
        "symbol": symbol,
        "side": side,
        "algoType": algo_type,
        "type": order_type_clean,
        "quantity": quantity,
        "triggerPrice": trigger_price,
        "workingType": working_type,
        "priceProtect": _bool_to_binance_flag(price_protect),
    }
    if order_type_clean != "STOP_MARKET":
        if price is None:
            logging.error("Algo STOP order missing price for STOP type. symbol=%s", symbol)
            return None
        base_params["price"] = price
        base_params["timeInForce"] = "GTC"
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


def algo_order_triggered_not_filled(order: Optional[Dict[str, object]]) -> bool:
    """
    Check if an algo order has been triggered (stop price hit) but not yet filled.
    This happens when the stop limit is triggered but the limit order hasn't executed.
    """
    if not order:
        return False
    status = algo_order_status(order)
    if status == "TRIGGERED":
        # Check if any qty was actually filled
        for key in ("executedQty", "executedQuantity", "executedVolume"):
            value = order.get(key)
            if value is None:
                continue
            try:
                if float(value) > 0:
                    return False  # Has fills, not just triggered
            except (TypeError, ValueError):
                continue
        return True
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


def append_live_signal(csv_path: str, signal_data: Dict[str, object]) -> None:
    """Append a signal to the live signals CSV file."""
    if not csv_path:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_SIGNAL_FIELDS)
        if not file_exists:
            writer.writeheader()
        row = {}
        for field in LIVE_SIGNAL_FIELDS:
            value = signal_data.get(field)
            row[field] = "" if value is None else value
        writer.writerow(row)


def save_live_swing_levels(
    file_path: str,
    symbol: str,
    df: pd.DataFrame,
    swing_timeframe: str,
    swing_left: int,
    swing_right: int,
    resample_rule: str,
) -> None:
    """
    Save detected swing levels and market structure (HH/HL/LL/LH) to a JSON file.
    
    Updates the file with swing levels for the given symbol, preserving data for other symbols.
    """
    if not file_path or df.empty:
        return
    
    # Detect swings on the specified timeframe
    if swing_timeframe == "1m":
        df_tf = df
        tf = "1m"
    else:
        df_tf = resample_ohlcv(df, rule=resample_rule)
        tf = swing_timeframe
    
    swings = detect_swings(df_tf, timeframe=tf, left=swing_left, right=swing_right)
    
    # Get current swing high/low levels (last confirmed)
    swing_highs = [s for s in swings if s.kind == "swing_high"]
    swing_lows = [s for s in swings if s.kind == "swing_low"]
    current_swing_high = swing_highs[-1].level if swing_highs else None
    current_swing_low = swing_lows[-1].level if swing_lows else None
    
    # Compute ATR for market structure classification
    atr = atr_ema(df_tf, length=14)
    
    # Classify swings into market structure (HH/HL/LL/LH)
    structure_points = classify_market_structure(swings, df_tf, atr)
    
    # Get current market structure levels
    hh_points = [sp for sp in structure_points if sp.structure_kind == "HH"]
    hl_points = [sp for sp in structure_points if sp.structure_kind == "HL"]
    ll_points = [sp for sp in structure_points if sp.structure_kind == "LL"]
    lh_points = [sp for sp in structure_points if sp.structure_kind == "LH"]
    
    current_hh = hh_points[-1].swing_point.high if hh_points else None
    current_hl = hl_points[-1].swing_point.low if hl_points else None
    current_ll = ll_points[-1].swing_point.low if ll_points else None
    current_lh = lh_points[-1].swing_point.high if lh_points else None
    
    # Determine current trend based on last structure point
    current_trend = "UNDEFINED"
    if structure_points:
        last_structure = structure_points[-1].structure_kind
        if last_structure in ("HH", "HL"):
            current_trend = "UPTREND"
        elif last_structure in ("LL", "LH"):
            current_trend = "DOWNTREND"
    
    # Load existing data or create new
    data: Dict[str, Any] = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    
    # Update data for this symbol
    data[symbol] = {
        "updated_at": _utc_now_iso(),
        "params": {
            "timeframe": swing_timeframe,
            "left": swing_left,
            "right": swing_right,
        },
        "current_levels": {
            "swing_high": current_swing_high,
            "swing_low": current_swing_low,
        },
        "market_structure": {
            "trend": current_trend,
            "hh": current_hh,
            "hl": current_hl,
            "ll": current_ll,
            "lh": current_lh,
        },
        "stats": {
            "bars": len(df_tf),
            "swing_highs": len(swing_highs),
            "swing_lows": len(swing_lows),
            "hh_count": len(hh_points),
            "hl_count": len(hl_points),
            "ll_count": len(ll_points),
            "lh_count": len(lh_points),
        },
        "swings": [s.to_dict() for s in swings],
        "structure_points": [sp.to_dict() for sp in structure_points],
    }
    
    # Save back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def load_klines_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load kline data from a CSV file (e.g., from kline_data_fetcher.py or Binance public data).

    Expected CSV columns (standard Binance kline format):
        open_time_ms, open, high, low, close, volume, close_time_ms, ...

    Returns a DataFrame indexed by UTC datetime with columns:
        open, high, low, close, volume, close_time_ms
    """
    df = pd.read_csv(csv_path)

    # Get timestamp column
    if "open_time_ms" in df.columns:
        ts_col = df["open_time_ms"]
    elif df.columns[0] in {"open_time", "timestamp"}:
        ts_col = df.iloc[:, 0]
    else:
        ts_col = df.iloc[:, 0]
    
    # Validate timestamps before conversion (filter out corrupted data)
    # Valid range: 2010-01-01 to 2100-01-01 in milliseconds
    MIN_VALID_MS = 1262304000000  # 2010-01-01 00:00:00 UTC
    MAX_VALID_MS = 4102444800000  # 2100-01-01 00:00:00 UTC
    
    ts_numeric = pd.to_numeric(ts_col, errors="coerce")
    valid_mask = (ts_numeric >= MIN_VALID_MS) & (ts_numeric <= MAX_VALID_MS) & ts_numeric.notna()
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logging.warning(f"Filtered {invalid_count:,} rows with invalid timestamps from {csv_path}")
        df = df[valid_mask].copy()
        ts_col = ts_numeric[valid_mask]
    
    if df.empty:
        raise ValueError(f"CSV file has no valid data after filtering: {csv_path}")
    
    # Convert to datetime
    df["dt"] = pd.to_datetime(ts_col, unit="ms", utc=True)

    df = df.set_index("dt")

    # Ensure required columns exist and convert to proper types
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
        df[col] = df[col].astype(float)

    # close_time_ms is optional but useful
    if "close_time_ms" in df.columns:
        df["close_time_ms"] = df["close_time_ms"].astype(int)
    else:
        # Estimate close_time_ms if not present (assume 1s interval as fallback)
        df["close_time_ms"] = (df.index.astype(np.int64) // 1_000_000) + 999

    # Keep only required columns
    df = df[["open", "high", "low", "close", "volume", "close_time_ms"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    logging.info(f"Loaded {len(df):,} candles from CSV: {csv_path}")
    return df


def resample_1s_to_interval(df_1s: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample 1s candle data to a larger interval (e.g., "1m", "5m", "15m").

    Args:
        df_1s: DataFrame with 1s candles (open, high, low, close, volume, close_time_ms)
        interval: Target interval string (e.g., "1m", "5m", "15m", "1h")

    Returns:
        DataFrame resampled to the target interval with proper OHLCV aggregation.
    """
    # Map interval string to pandas resample rule
    interval_map = {
        "1s": "1s",
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
    }

    resample_rule = interval_map.get(interval)
    if resample_rule is None:
        raise ValueError(f"Unsupported interval for resampling: {interval}")

    if interval == "1s":
        # No resampling needed
        return df_1s.copy()

    # Resample with proper OHLCV aggregation
    df_resampled = df_1s.resample(resample_rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "close_time_ms": "last",
    }).dropna()

    logging.info(f"Resampled {len(df_1s):,} 1s candles to {len(df_resampled):,} {interval} candles")
    return df_resampled


# -----------------------------
# Binance Public Data Fetching (data.binance.vision)
# -----------------------------
KLINE_DATA_DIR = Path(__file__).parent / "kline_data"
PUBLIC_DATA_BASE = "https://data.binance.vision"
SPOT_REST_BASE = "https://api.binance.com"
FUTURES_REST_BASE = "https://fapi.binance.com"

KLINE_CSV_HEADER = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]


def _parse_date(s: str) -> dt.date:
    """Parse YYYY-MM-DD string to date object."""
    return dt.datetime.strptime(s.split()[0], "%Y-%m-%d").date()


_CACHED_KLINE_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9]+)_(?P<interval>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_to_"
    r"(?P<end>\d{4}-\d{2}-\d{2})(?P<market_suffix>_futures)?\.csv$"
)


def _parse_cached_kline_filename(filename: str) -> Optional[Tuple[str, str, dt.date, dt.date, str]]:
    match = _CACHED_KLINE_RE.match(filename)
    if not match:
        return None
    try:
        start = dt.datetime.strptime(match.group("start"), "%Y-%m-%d").date()
        end = dt.datetime.strptime(match.group("end"), "%Y-%m-%d").date()
    except ValueError:
        return None
    market = "futures" if match.group("market_suffix") else "spot"
    return (
        match.group("symbol").upper(),
        match.group("interval").lower(),
        start,
        end,
        market,
    )


def _find_cached_klines_covering_range(
    symbol: str,
    interval: str,
    start: dt.date,
    end: dt.date,
    market: str,
) -> Optional[Tuple[Path, dt.date, dt.date]]:
    if not KLINE_DATA_DIR.exists():
        return None
    symbol = symbol.upper()
    interval = interval.lower()
    best: Optional[Tuple[int, Path, dt.date, dt.date]] = None
    for path in KLINE_DATA_DIR.glob("*.csv"):
        parsed = _parse_cached_kline_filename(path.name)
        if not parsed:
            continue
        p_symbol, p_interval, p_start, p_end, p_market = parsed
        if p_symbol != symbol or p_interval != interval or p_market != market:
            continue
        if p_start <= start and p_end >= end:
            span_days = (p_end - p_start).days
            if best is None or span_days < best[0]:
                best = (span_days, path, p_start, p_end)
    if not best:
        return None
    return best[1], best[2], best[3]


def _slice_klines_df_to_dates(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end + dt.timedelta(days=1), tz="UTC")
    return df.loc[(df.index >= start_dt) & (df.index < end_dt)]


def _daterange_inclusive(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    """Generate dates from start to end (inclusive)."""
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _date_to_ms_utc(d: dt.date) -> int:
    """Convert date to milliseconds UTC (start of day)."""
    dttm = dt.datetime.combine(d, dt.time.min, tzinfo=dt.timezone.utc)
    return int(dttm.timestamp() * 1000)


def _interval_to_ms_public(interval: str) -> int:
    """Convert interval string to milliseconds."""
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "s":
        return num * 1000
    if unit == "m":
        return num * 60_000
    if unit == "h":
        return num * 3_600_000
    if unit == "d":
        return num * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


def _public_daily_zip_url(symbol: str, interval: str, day: dt.date, market: str = "spot") -> str:
    """Build URL for Binance public daily kline zip file.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDC")
        interval: Kline interval (e.g., "1s", "1m")
        day: Date for the daily file
        market: "spot" or "futures" (UM perpetual)
    """
    if market == "futures":
        # UM Futures path: /data/futures/um/daily/klines/
        return (
            f"{PUBLIC_DATA_BASE}/data/futures/um/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{day.isoformat()}.zip"
        )
    else:
        # Spot path: /data/spot/daily/klines/
        return (
            f"{PUBLIC_DATA_BASE}/data/spot/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{day.isoformat()}.zip"
        )


def _build_ssl_context_public() -> ssl.SSLContext:
    """
    Build SSL context for public data fetching.
    
    Handles macOS certificate issues by trying certifi first,
    then falling back to system defaults or unverified context.
    """
    # Check environment variables for custom CA bundle
    ca_bundle = getenv("BINANCE_CA_BUNDLE") or getenv("SSL_CERT_FILE") or getenv("REQUESTS_CA_BUNDLE")
    ssl_verify_env = (getenv("BINANCE_SSL_VERIFY") or "").strip().lower()
    
    # If explicitly disabled
    if ssl_verify_env in {"0", "false", "no", "off"}:
        return ssl._create_unverified_context()
    
    # Try explicit CA bundle
    if ca_bundle:
        try:
            return ssl.create_default_context(cafile=ca_bundle)
        except Exception:
            pass
    
    # Try certifi (commonly installed, fixes macOS issues)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    
    # Fall back to system defaults
    try:
        return ssl.create_default_context()
    except Exception:
        # Last resort: unverified context
        return ssl._create_unverified_context()


# Cache the SSL context (built once per process)
_PUBLIC_SSL_CONTEXT: Optional[ssl.SSLContext] = None


def _get_ssl_context() -> ssl.SSLContext:
    """Get cached SSL context for public data fetching."""
    global _PUBLIC_SSL_CONTEXT
    if _PUBLIC_SSL_CONTEXT is None:
        _PUBLIC_SSL_CONTEXT = _build_ssl_context_public()
    return _PUBLIC_SSL_CONTEXT


def _fetch_public_day_zip(
    symbol: str,
    interval: str,
    day: dt.date,
    timeout: int = 60,
    market: str = "spot",
) -> Optional[List[List[str]]]:
    """
    Fetch daily kline data from Binance public data (data.binance.vision).
    Returns list of kline rows if successful, None if 404.
    
    Args:
        market: "spot" or "futures" (UM perpetual)
    """
    url = _public_daily_zip_url(symbol, interval, day, market=market)
    ssl_ctx = _get_ssl_context()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            zdata = io.BytesIO(resp.read())
            with zipfile.ZipFile(zdata) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise RuntimeError(f"Zip from {url} had no CSV files inside.")
                with zf.open(csv_names[0], "r") as f:
                    text = io.TextIOWrapper(f, encoding="utf-8")
                    reader = csv.reader(text)
                    # Filter out header rows (first element should be numeric timestamp)
                    rows = []
                    for row in reader:
                        if row and row[0].isdigit():
                            rows.append(row)
                    return rows
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _fetch_rest_klines_chunk(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    timeout: int = 30,
    market: str = "spot",
) -> List[List[str]]:
    """Fetch klines from Binance REST API for a single chunk.
    
    Args:
        market: "spot" or "futures" (UM perpetual)
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    query = urllib.parse.urlencode(params)
    if market == "futures":
        url = f"{FUTURES_REST_BASE}/fapi/v1/klines?{query}"
    else:
        url = f"{SPOT_REST_BASE}/api/v3/klines?{query}"
    ssl_ctx = _get_ssl_context()
    
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [[str(x) for x in k] for k in data]
    except urllib.error.HTTPError as e:
        logging.warning(f"REST API error {e.code} for {url[:100]}...")
        return []
    except Exception as e:
        logging.warning(f"REST API exception: {e}")
        return []


def _fetch_rest_day(
    symbol: str,
    interval: str,
    day: dt.date,
    polite_sleep_s: float = 0.2,
    market: str = "spot",
) -> List[List[str]]:
    """Fetch a full day of klines via REST API with pagination.
    
    Args:
        market: "spot" or "futures" (UM perpetual)
    """
    day_start_ms = _date_to_ms_utc(day)
    day_end_ms = _date_to_ms_utc(day + dt.timedelta(days=1))
    interval_ms = _interval_to_ms_public(interval)
    
    all_rows: List[List[str]] = []
    cur = day_start_ms
    
    while cur < day_end_ms:
        chunk = _fetch_rest_klines_chunk(
            symbol=symbol,
            interval=interval,
            start_ms=cur,
            end_ms=day_end_ms,
            limit=1000,
            market=market,
        )
        if not chunk:
            break
        
        all_rows.extend(chunk)
        last_open = int(chunk[-1][0])
        cur = last_open + interval_ms
        
        if len(chunk) < 1000:
            break
        
        time.sleep(polite_sleep_s)
    
    return all_rows


def fetch_klines_public_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    *,
    save_csv: bool = True,
    force_refetch: bool = False,
    market: str = "spot",
) -> pd.DataFrame:
    """
    Fetch klines from Binance Public Data (data.binance.vision) with REST API fallback.
    
    Uses daily zip files from public data when available, falls back to REST API
    for recent days or missing data.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDC")
        interval: Kline interval (e.g., "1s", "1m")
        start_date: Start date string "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        end_date: End date string "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        save_csv: If True, save to kline_data/ folder for caching
        force_refetch: If True, ignore cached CSV and refetch
        market: "spot" or "futures" (UM perpetual). Note: 1s interval only available on spot.
    
    Returns:
        DataFrame indexed by UTC datetime with OHLCV columns
    """
    symbol = symbol.upper()
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    
    if end < start:
        raise ValueError(f"end_date must be >= start_date: {start_date} to {end_date}")
    
    # Check for cached CSV (include market in filename to differentiate)
    market_suffix = "_futures" if market == "futures" else ""
    csv_filename = f"{symbol}_{interval}_{start.isoformat()}_to_{end.isoformat()}{market_suffix}.csv"
    csv_path = KLINE_DATA_DIR / csv_filename

    if not force_refetch:
        cached = _find_cached_klines_covering_range(symbol, interval, start, end, market)
        if cached:
            cached_path, cached_start, cached_end = cached
            print(f"Loading cached data from: {cached_path} (covers {cached_start} to {cached_end})")
            df_cached = load_klines_from_csv(str(cached_path))
            df_subset = _slice_klines_df_to_dates(df_cached, start, end)
            if not df_subset.empty:
                return df_subset
            logging.warning(
                "Cached kline file returned no rows after slicing; refetching. file=%s range=%s..%s",
                cached_path,
                start,
                end,
            )

    # Ensure kline_data directory exists
    KLINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    market_label = "futures" if market == "futures" else "spot"
    print(f"Fetching {interval} klines for {symbol} ({market_label}) from {start} to {end}...")
    
    combined_rows: List[List[str]] = []
    public_ok = 0
    public_missing = 0
    rest_rows = 0
    
    total_days = (end - start).days + 1
    
    for i, day in enumerate(_daterange_inclusive(start, end)):
        # Progress update
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing day {i + 1}/{total_days}: {day}")
        
        # Try public data first
        try:
            rows = _fetch_public_day_zip(symbol, interval, day, market=market)
        except Exception as e:
            logging.warning(f"Public data error for {day}: {e}. Falling back to REST.")
            rows = None
        
        if rows is None:
            public_missing += 1
            # REST API fallback
            try:
                rows = _fetch_rest_day(symbol, interval, day, market=market)
                rest_rows += len(rows)
            except Exception as e:
                logging.warning(f"REST API error for {day}: {e}. Skipping.")
                continue
        else:
            public_ok += 1
        
        combined_rows.extend(rows)
    
    # Sort by open_time_ms
    combined_rows.sort(key=lambda r: int(r[0]))
    
    print(f"Fetched {len(combined_rows):,} rows "
          f"(public: {public_ok} days, REST fallback: {public_missing} days/{rest_rows:,} rows)")
    
    # Save to CSV if requested
    if save_csv and combined_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(KLINE_CSV_HEADER)
            for row in combined_rows:
                if len(row) == 12:
                    w.writerow(row)
        print(f"Saved to: {csv_path}")
    
    # Convert to DataFrame
    if not combined_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_time_ms"])
    
    return load_klines_from_csv(str(csv_path)) if save_csv else _rows_to_df(combined_rows)


def _rows_to_df(rows: List[List[str]]) -> pd.DataFrame:
    """Convert raw kline rows to DataFrame."""
    df_rows = []
    for r in rows:
        if len(r) >= 7:
            df_rows.append({
                "dt": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "close_time_ms": int(r[6]),
            })
    df = pd.DataFrame(df_rows).set_index("dt")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def get_cached_kline_path(symbol: str, interval: str, start_date: str, end_date: str) -> Optional[Path]:
    """
    Check if cached kline CSV exists for the given parameters.
    Returns the path if exists, None otherwise.
    """
    symbol = symbol.upper()
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    csv_filename = f"{symbol}_{interval}_{start.isoformat()}_to_{end.isoformat()}.csv"
    csv_path = KLINE_DATA_DIR / csv_filename
    return csv_path if csv_path.exists() else None


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


def build_1s_minute_index(df_1s: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Pre-index 1m candles by DAY for O(1) lookups when using daily signal candles.
    
    This groups all 1m candles by their date, so when we have a daily candle
    timestamp, we can quickly get all the 1m candles for that day.
    
    Args:
        df_1s: DataFrame with 1m candles indexed by open time (UTC)
    
    Returns:
        Dict mapping day timestamp (00:00:00) -> DataFrame of 1m candles for that day
    """
    df_1s = df_1s.copy()
    # Floor timestamps to day (start of day)
    df_1s["_day"] = df_1s.index.floor("D")
    
    # Group by day and store each group
    day_index: Dict[pd.Timestamp, pd.DataFrame] = {}
    for day_ts, group in df_1s.groupby("_day"):
        day_index[day_ts] = group.drop(columns=["_day"])
    
    return day_index


def get_1s_candles_for_minute(
    minute_index: Dict[pd.Timestamp, pd.DataFrame],
    minute_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Get all 1m candles for a given day (for use with daily signal candles).
    
    O(1) lookup instead of O(n) scan.

    Args:
        minute_index: Pre-built dict from build_1s_minute_index() (actually day-indexed)
        minute_timestamp: The open time of the daily candle (will be floored to day)

    Returns:
        DataFrame with 1m candles for that day (empty DataFrame if not found)
    """
    # Floor to day to match the index
    day_ts = minute_timestamp.floor("D")
    return minute_index.get(day_ts, pd.DataFrame())


def backtest_atr_grinder_lib(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Look-Inside-Bar (LIB) Backtest using 1d candles for signals and 1m candles for trailing stop tracking.

    This provides more realistic backtesting by:
    - Checking SL hits in chronological order using 1m data
    - Updating trailing stops on each 1m candle close (Look-In-Bar mode)
    - Better determining limit order fills

    Args:
        df: Signal interval DataFrame (1d candles) with OHLCV
        df_1m: 1m candles DataFrame for trailing stop tracking
        cfg: Backtest configuration

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

    # Compute market structure (HH/HL/LL/LH) levels
    structure_points, hh_series, hl_series, ll_series, lh_series = compute_market_structure_levels(
        df,
        df["atr"],
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["hh_level"] = hh_series
    df["hl_level"] = hl_series
    df["ll_level"] = ll_series
    df["lh_level"] = lh_series

    # Also compute raw swing levels for reference
    swing_high, swing_low = compute_confirmed_swing_levels(
        df,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    df["swing_high_level"] = swing_high
    df["swing_low_level"] = swing_low

    # Generate signals based on market structure
    df["signal"], df["signal_atr"], df["signal_entry_price"], df["signal_reason"], df["trend"] = build_market_structure_signals(
        df,
        df["atr"],
        structure_points,
        body_atr_mult=cfg.thr2,
        structure_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    warmup_bars = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if warmup_bars > 0:
        warmup_idx = df.index[:warmup_bars]
        df.loc[warmup_idx, "signal"] = 0
        df.loc[warmup_idx, "signal_atr"] = np.nan
        df.loc[warmup_idx, "signal_entry_price"] = np.nan
        df.loc[warmup_idx, "signal_reason"] = ""
        df.loc[warmup_idx, "trend"] = 0

    # Equity curve (USD)
    equity = cfg.initial_capital
    equity_series = pd.Series(np.nan, index=df.index, dtype=float)
    equity_series.iloc[0] = equity

    positions: List[Dict[str, Any]] = []
    pending_entries: List[Dict[str, Any]] = []

    # Trade ID tracking for signal-trade linking
    trade_counter = 0

    trades: List[Dict] = []
    trailing_stop_updates: List[Dict] = []

    def should_trigger_forced_exit(state: Dict[str, Any], timestamp: pd.Timestamp) -> bool:
        """Return True when the forced-exit limit on configured intervals is reached."""
        if not cfg.use_trailing_stop or cfg.forced_exit_interval_count <= 0:
            return False
        if not is_exit_check_boundary(timestamp, cfg.forced_exit_interval):
            return False
        last_boundary = state.get("forced_exit_last_boundary_ts")
        intervals = state.get("forced_exit_intervals_since_entry", 0)
        is_new_boundary = last_boundary is None or timestamp > last_boundary
        if is_new_boundary:
            intervals += 1
            state["forced_exit_intervals_since_entry"] = intervals
            state["forced_exit_last_boundary_ts"] = timestamp
        return intervals >= cfg.forced_exit_interval_count

    def create_position(
        side: int,
        entry_price: float,
        signal_atr_value: float,
        entry_time: pd.Timestamp,
        entry_index: int,
        signal_reason: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        nonlocal trade_counter
        trade_margin, trade_leverage, _ = resolve_trade_sizing(
            entry_price=entry_price,
            atr_value=signal_atr_value,
            sl_atr_mult=1.0,
            margin_cap=cfg.initial_capital,
            max_leverage=cfg.leverage,
            min_leverage=cfg.leverage,
            target_loss_usd=None,
        )
        if trade_margin is None or trade_leverage is None:
            return None

        if cfg.use_trailing_stop:
            target_price = None
            trail_r_value = signal_atr_value * 1.0  # R = 1x ATR
            trail_best_close_r = 0.0
            trail_stop_r = cfg.trail_initial_stop_r  # Initial stop R level from config
            stop_price = None  # No stop until price reaches breakeven
        else:
            target_price, stop_price = compute_tp_sl_prices(
                entry_price, side, signal_atr_value, 2.0, 1.0
            )
            trail_r_value = None
            trail_best_close_r = None
            trail_stop_r = None

        trade_counter += 1
        side_str = "LONG" if side == 1 else "SHORT"
        trade_id = generate_trade_id(df.attrs.get("symbol", "UNK"), str(entry_time), side_str)

        return {
            "position": side,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "trail_r_value": trail_r_value,
            "trail_best_close_r": trail_best_close_r,
            "trail_stop_r": trail_stop_r,
            "entry_time": entry_time,
            "entry_index": entry_index,
            "used_signal_atr": signal_atr_value,
            "active_margin": trade_margin,
            "active_leverage": trade_leverage,
            "active_notional": trade_margin * trade_leverage,
            "trade_id": trade_id,
            "signal_reason": signal_reason,
            "forced_exit_intervals_since_entry": 0,
            "forced_exit_last_boundary_ts": None,
        }

    idx = df.index.to_list()
    daily_date = idx[0].date() if idx else None

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
        sl_price: Optional[float],
        tp_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        use_trailing: bool,
        forced_exit_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[pd.Timestamp]]:
        """Check for exit using candle closes at higher timeframe boundaries (e.g., 4h).

        Note: sl_price may be None if no stop has been set yet (breakeven-only stops).
        Exit only triggers when a higher timeframe candle CLOSES beyond the stop.
        """
        for t_1s, candle_1s in df_1s_chunk.iterrows():
            # Only check exit on higher timeframe boundaries (e.g., 4h close)
            if not is_exit_check_boundary(t_1s, cfg.trail_exit_check_interval):
                continue

            forced_exit_limit_hit = should_trigger_forced_exit(forced_exit_state, t_1s)
            c = float(candle_1s["close"])

            if sl_price is None:
                if not use_trailing and tp_price is not None:
                    if pos == 1 and c >= tp_price:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s
                    elif pos == -1 and c <= tp_price:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s
            else:
                if pos == 1:  # LONG
                    # Exit if candle CLOSES below stop
                    sl_hit = c <= sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c >= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s

                else:  # SHORT
                    # Exit if candle CLOSES above stop
                    sl_hit = c >= sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c <= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s

            if forced_exit_limit_hit:
                exit_price = apply_slippage(c, pos, is_entry=False)
                return True, FORCED_EXIT_REASON, exit_price, t_1s

        return False, None, None, None

    def update_trailing_stop_lib(
        pos: int,
        entry_p: float,
        r_value: float,
        best_close_r: float,
        stop_r: int,
        sl_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        entry_t: pd.Timestamp,
        side_str: str,
    ) -> Tuple[Optional[float], float, int]:
        """Update trailing stop using 1s closes (R-ladder mode).
        
        Only sets stops once activation threshold is reached; stop can move in either direction.
        """
        nonlocal trailing_stop_updates
        current_best_r = best_close_r
        current_stop_r = stop_r
        current_sl_price = sl_price

        for t_1s, candle_1s in df_1s_chunk.iterrows():
            c = float(candle_1s["close"])
            close_r = compute_close_r(entry_p, c, pos, r_value)

            if close_r is not None:
                # Check if we should set a stop
                should_update = False
                prev_best_r = current_best_r
                prev_stop_r = current_stop_r
                prev_sl_price = current_sl_price

                # Track best close R
                if close_r > current_best_r:
                    current_best_r = close_r

                next_stop_r = compute_trailing_stop_r(current_best_r, current_stop_r, cfg.trail_gap_r)

                # Place stop when:
                # 1. First time reaching ladder threshold (current_sl_price is None and next_stop_r >= 0)
                # 2. Stop level improves (next_stop_r > current_stop_r)
                # Only place stops at breakeven (0R) or better - no negative stops
                should_update = (
                    (current_sl_price is None and next_stop_r >= 0) or  # First stop at breakeven when price reaches 1.25R
                    (next_stop_r > current_stop_r and next_stop_r >= 0)  # Trailing improvement
                )
                if should_update:
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
                        "trailing_mode": "r_ladder",
                        "stop_moved": True,
                        "lib_mode": True,
                    })
                    current_stop_r = next_stop_r
                    current_sl_price = new_sl_price

        return current_sl_price, current_best_r, current_stop_r

    def process_trailing_and_exit_lib(
        pos: int,
        entry_p: float,
        r_value: float,
        best_close_r: float,
        stop_r: int,
        sl_price: Optional[float],
        tp_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        entry_t: pd.Timestamp,
        side_str: str,
        use_trailing: bool,
        forced_exit_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[pd.Timestamp], Optional[float], float, int]:
        """
        Process 1m candles sequentially: update trailing stop THEN check exit for each candle.
        
        Returns:
            (exited, exit_reason, exit_price, exit_time, final_sl_price, final_best_r, final_stop_r)
        """
        nonlocal trailing_stop_updates
        current_best_r = best_close_r
        current_stop_r = stop_r
        current_sl_price = sl_price

        for t_1s, candle_1s in df_1s_chunk.iterrows():
            o = float(candle_1s["open"])
            h = float(candle_1s["high"])
            l = float(candle_1s["low"])
            c = float(candle_1s["close"])
            forced_exit_limit_hit = should_trigger_forced_exit(forced_exit_state, t_1s)

            # Step 1: Update trailing stop based on close
            close_r = compute_close_r(entry_p, c, pos, r_value)
            if close_r is not None:
                prev_best_r = current_best_r
                prev_stop_r = current_stop_r
                prev_sl_price = current_sl_price

                if close_r > current_best_r:
                    current_best_r = close_r

                next_stop_r = compute_trailing_stop_r(current_best_r, current_stop_r, cfg.trail_gap_r)

                should_update = (
                    (current_sl_price is None and next_stop_r >= 0) or
                    (next_stop_r > current_stop_r and next_stop_r >= 0)
                )
                if should_update:
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
                        "trailing_mode": "r_ladder",
                        "stop_moved": True,
                        "lib_mode": True,
                    })
                    current_stop_r = next_stop_r
                    current_sl_price = new_sl_price

            # Step 2: Check exit only on higher timeframe close (e.g., 4h)
            # Only check exit if stop is set AND this is an exit check boundary
            if current_sl_price is not None and is_exit_check_boundary(t_1s, cfg.trail_exit_check_interval):
                if pos == 1:  # LONG
                    # Exit if candle CLOSES below stop
                    sl_hit = c <= current_sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c >= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r

                else:  # SHORT
                    # Exit if candle CLOSES above stop
                    sl_hit = c >= current_sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c <= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s, current_sl_price, current_best_r, current_stop_r

            if forced_exit_limit_hit:
                exit_price = apply_slippage(c, pos, is_entry=False)
                return True, FORCED_EXIT_REASON, exit_price, t_1s, current_sl_price, current_best_r, current_stop_r

        # No exit triggered
        return False, None, None, None, current_sl_price, current_best_r, current_stop_r

    def update_trailing_stop_dynamic_atr_lib(
        pos: int,
        entry_p: float,
        signal_atr: float,
        sl_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        df_signal: pd.DataFrame,
        signal_t: pd.Timestamp,
        entry_t: pd.Timestamp,
        side_str: str,
    ) -> Optional[float]:
        """Update trailing stop using dynamic ATR at trail_check_interval boundaries.
        
        Only sets stops once activation threshold is reached; stop can move in either direction.
        Uses signal_atr from entry for consistent risk management.
        Updates occur at trail_check_interval boundaries (e.g., 4h) using accumulated high/low.
        """
        nonlocal trailing_stop_updates
        current_sl_price = sl_price

        # Use signal_atr from entry for consistent risk, not current bar's ATR
        
        # Track accumulated high/low for the current trail_check_interval period
        period_high = None
        period_low = None
        period_close = None

        for t_1s, candle_1s in df_1s_chunk.iterrows():
            h = float(candle_1s["high"])
            l = float(candle_1s["low"])
            c = float(candle_1s["close"])
            
            # Accumulate high/low for the current period
            if period_high is None or h > period_high:
                period_high = h
            if period_low is None or l < period_low:
                period_low = l
            period_close = c

            # Only update trailing stop at trail_check_interval boundaries (e.g., 4h)
            if not is_exit_check_boundary(t_1s, cfg.trail_check_interval):
                continue
                
            prev_sl_price = current_sl_price

            # Select price source for trailing: high_low uses favorable extreme (period high for LONG, period low for SHORT)
            if cfg.dynamic_trail_price_source == "high_low":
                trail_price = period_high if pos == 1 else period_low
            else:
                trail_price = period_close
            atr_stop = compute_dynamic_atr_stop(trail_price, signal_atr, cfg.dynamic_trail_atr_mult, pos)

            # Allow ATR stop to move in either direction once activated.
            new_stop = atr_stop

            # Only set stop if price has moved activation_r in our favor
            activation_threshold = signal_atr * cfg.dynamic_trail_activation_r
            if pos == 1:
                price_moved_enough = period_close >= entry_p + activation_threshold
                should_set_stop = price_moved_enough
            else:
                price_moved_enough = period_close <= entry_p - activation_threshold
                should_set_stop = price_moved_enough

            stop_moved = should_set_stop and (current_sl_price is None or new_stop != current_sl_price)
            if stop_moved:
                trailing_stop_updates.append({
                    "timestamp": t_1s,
                    "entry_time": entry_t,
                    "side": side_str,
                    "entry_price": entry_p,
                    "close_price": period_close,
                    "trail_price": trail_price,
                    "signal_atr": signal_atr,
                    "atr_stop": atr_stop,
                    "prev_stop_price": prev_sl_price,
                    "new_stop_price": new_stop,
                    "trailing_mode": "dynamic_atr",
                    "dynamic_trail_atr_mult": cfg.dynamic_trail_atr_mult,
                    "dynamic_trail_activation_r": cfg.dynamic_trail_activation_r,
                    "dynamic_trail_price_source": cfg.dynamic_trail_price_source,
                    "price_moved_enough": price_moved_enough,
                    "stop_moved": True,
                    "lib_mode": True,
                })
                current_sl_price = new_stop
            
            # Reset period accumulators after processing the boundary
            period_high = None
            period_low = None
            period_close = None

        return current_sl_price

    def process_trailing_and_exit_dynamic_atr_lib(
        pos: int,
        entry_p: float,
        signal_atr: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        df_1s_chunk: pd.DataFrame,
        df_signal: pd.DataFrame,
        signal_t: pd.Timestamp,
        entry_t: pd.Timestamp,
        side_str: str,
        use_trailing: bool,
        forced_exit_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[pd.Timestamp], Optional[float]]:
        """
        Process 1m candles sequentially with dynamic ATR trailing: update trailing stop THEN check exit for each candle.
        
        Only sets stops at breakeven (0R) or better - no negative stops.
        Trailing stop updates occur at trail_check_interval boundaries (e.g., 4h) using accumulated high/low.
        
        Returns:
            (exited, exit_reason, exit_price, exit_time, final_sl_price)
        """
        nonlocal trailing_stop_updates
        current_sl_price = sl_price
        floor_stop = entry_p  # Floor at breakeven (0R), no negative stops

        # Use signal_atr from entry for consistent risk, not current bar's ATR
        
        # Track accumulated high/low for the current trail_check_interval period
        period_high = None
        period_low = None
        period_close = None

        for t_1s, candle_1s in df_1s_chunk.iterrows():
            o = float(candle_1s["open"])
            h = float(candle_1s["high"])
            l = float(candle_1s["low"])
            c = float(candle_1s["close"])
            forced_exit_limit_hit = should_trigger_forced_exit(forced_exit_state, t_1s)

            # Accumulate high/low for the current period
            if period_high is None or h > period_high:
                period_high = h
            if period_low is None or l < period_low:
                period_low = l
            period_close = c

            # Step 1: Update trailing stop only at trail_check_interval boundaries (e.g., 4h)
            if is_exit_check_boundary(t_1s, cfg.trail_check_interval):
                prev_sl_price = current_sl_price
                
                # Select price source for trailing: high_low uses favorable extreme (period high for LONG, period low for SHORT)
                if cfg.dynamic_trail_price_source == "high_low":
                    trail_price = period_high if pos == 1 else period_low
                else:
                    trail_price = period_close
                atr_stop = compute_dynamic_atr_stop(trail_price, signal_atr, cfg.dynamic_trail_atr_mult, pos)

                # Allow ATR stop to move in either direction once activated.
                new_stop = atr_stop

                # Only set stop if price has moved activation_r in our favor
                activation_threshold = signal_atr * cfg.dynamic_trail_activation_r
                if pos == 1:
                    price_moved_enough = period_close >= entry_p + activation_threshold
                    should_set_stop = price_moved_enough
                else:
                    price_moved_enough = period_close <= entry_p - activation_threshold
                    should_set_stop = price_moved_enough

                stop_moved = should_set_stop and (current_sl_price is None or new_stop != current_sl_price)

                if stop_moved:
                    trailing_stop_updates.append({
                        "timestamp": t_1s,
                        "entry_time": entry_t,
                        "side": side_str,
                        "entry_price": entry_p,
                        "close_price": period_close,
                        "trail_price": trail_price,
                        "signal_atr": signal_atr,
                        "atr_stop": atr_stop,
                        "prev_stop_price": prev_sl_price,
                        "new_stop_price": new_stop,
                        "trailing_mode": "dynamic_atr",
                        "dynamic_trail_atr_mult": cfg.dynamic_trail_atr_mult,
                        "dynamic_trail_activation_r": cfg.dynamic_trail_activation_r,
                        "dynamic_trail_price_source": cfg.dynamic_trail_price_source,
                        "price_moved_enough": price_moved_enough,
                        "stop_moved": True,
                        "lib_mode": True,
                    })
                    current_sl_price = new_stop
                
                # Reset period accumulators after processing the boundary
                period_high = None
                period_low = None
                period_close = None

            # Step 2: Check exit only on higher timeframe close (e.g., 4h)
            # Only check exit if stop is set AND this is an exit check boundary
            if current_sl_price is not None and is_exit_check_boundary(t_1s, cfg.trail_exit_check_interval):
                if pos == 1:  # LONG
                    # Exit if candle CLOSES below stop
                    sl_hit = c <= current_sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c >= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s, current_sl_price

                else:  # SHORT
                    # Exit if candle CLOSES above stop
                    sl_hit = c >= current_sl_price
                    tp_hit = (not use_trailing) and tp_price is not None and c <= tp_price

                    if sl_hit and tp_hit:
                        # Both hit - conservative: use SL
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price
                    elif sl_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "SL", exit_price, t_1s, current_sl_price
                    elif tp_hit:
                        exit_price = apply_slippage(c, pos, is_entry=False)
                        return True, "TP", exit_price, t_1s, current_sl_price

            if forced_exit_limit_hit:
                exit_price = apply_slippage(c, pos, is_entry=False)
                return True, FORCED_EXIT_REASON, exit_price, t_1s, current_sl_price

        # No exit triggered
        return False, None, None, None, current_sl_price

    limit_timeout_bars = max(1, int(cfg.entry_limit_timeout_bars))

    # Pre-index 1m data by minute for O(1) lookups (HUGE performance improvement)
    print("  Building 1m data index by minute...")
    minute_index = build_1s_minute_index(df_1m)
    print(f"  Indexed {len(minute_index):,} minutes")

    def process_entry_immediate_exit(
        pos: Dict[str, Any],
        df_1m_bar: pd.DataFrame,
        signal_t: pd.Timestamp,
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[pd.Timestamp]]:
        if df_1m_bar.empty or not cfg.use_trailing_stop or pos["trail_r_value"] is None:
            return False, None, None, None
        side_str = "LONG" if pos["position"] == 1 else "SHORT"
        if cfg.trailing_mode == "r_ladder" and pos["trail_best_close_r"] is not None and pos["trail_stop_r"] is not None:
            exited, exit_reason, exit_price_check, exit_time_check, stop_price, best_r, stop_r = process_trailing_and_exit_lib(
                pos["position"],
                pos["entry_price"],
                pos["trail_r_value"],
                pos["trail_best_close_r"],
                pos["trail_stop_r"],
                pos["stop_price"],
                pos["target_price"],
                df_1m_bar,
                pos["entry_time"],
                side_str,
                cfg.use_trailing_stop,
                pos,
            )
            pos["stop_price"] = stop_price
            pos["trail_best_close_r"] = best_r
            pos["trail_stop_r"] = stop_r
            return exited, exit_reason, exit_price_check, exit_time_check
        if cfg.trailing_mode == "dynamic_atr" and pos["used_signal_atr"] is not None:
            exited, exit_reason, exit_price_check, exit_time_check, stop_price = process_trailing_and_exit_dynamic_atr_lib(
                pos["position"],
                pos["entry_price"],
                pos["used_signal_atr"],
                pos["stop_price"],
                pos["target_price"],
                df_1m_bar,
                df,
                signal_t,
                pos["entry_time"],
                side_str,
                cfg.use_trailing_stop,
                pos,
            )
            pos["stop_price"] = stop_price
            return exited, exit_reason, exit_price_check, exit_time_check
        return False, None, None, None

    for i in range(1, len(df)):
        t = idx[i]
        prev_t = idx[i - 1]
        if daily_date is not None:
            current_date = t.date()
            if current_date != daily_date:
                daily_date = current_date
                daily_pnl_usd = 0.0

        df_1m_bar = get_1s_candles_for_minute(minute_index, t)
        has_1m_data = len(df_1m_bar) > 0

        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # --- EXIT logic using 1m data ---
        for pos in list(positions):
            position = pos["position"]
            entry_price = pos["entry_price"]
            target_price = pos["target_price"]

            exited = False
            exit_reason = None
            exit_price = None
            exit_time = t

            if entry_price is None:
                continue

            if has_1m_data:
                if cfg.use_trailing_stop:
                    side_str = "LONG" if position == 1 else "SHORT"
                    if cfg.trailing_mode == "dynamic_atr" and pos["used_signal_atr"] is not None:
                        pos["stop_price"] = update_trailing_stop_dynamic_atr_lib(
                            position,
                            entry_price,
                            pos["used_signal_atr"],
                            pos["stop_price"],
                            df_1m_bar,
                            df,
                            t,
                            pos["entry_time"],
                            side_str,
                        )
                    elif (
                        cfg.trailing_mode == "r_ladder"
                        and pos["trail_r_value"] is not None
                        and pos["trail_best_close_r"] is not None
                        and pos["trail_stop_r"] is not None
                    ):
                        pos["stop_price"], pos["trail_best_close_r"], pos["trail_stop_r"] = update_trailing_stop_lib(
                            position,
                            entry_price,
                            pos["trail_r_value"],
                            pos["trail_best_close_r"],
                            pos["trail_stop_r"],
                            pos["stop_price"],
                            df_1m_bar,
                            pos["entry_time"],
                            side_str,
                        )

                exited, exit_reason, exit_price, exit_time = check_exit_lib(
                    position,
                    pos["stop_price"],
                    target_price,
                    df_1m_bar,
                    cfg.use_trailing_stop,
                    pos,
                )
            else:
                c = float(df.at[t, "close"])
                sl_hit = False
                if pos["stop_price"] is not None:
                    sl_hit = (c <= pos["stop_price"]) if position == 1 else (c >= pos["stop_price"])

                if cfg.use_trailing_stop:
                    if sl_hit:
                        exit_reason = "SL"
                        exit_price = apply_slippage(c, position, is_entry=False)
                        exited = True
                else:
                    if target_price is not None:
                        tp_hit = (c >= target_price) if position == 1 else (c <= target_price)
                    else:
                        tp_hit = False

                    if sl_hit and tp_hit:
                        exit_reason = "SL"
                        exit_price = apply_slippage(c, position, is_entry=False)
                        exited = True
                    elif tp_hit:
                        exit_reason = "TP"
                        exit_price = apply_slippage(c, position, is_entry=False)
                        exited = True
                    elif sl_hit:
                        exit_reason = "SL"
                        exit_price = apply_slippage(c, position, is_entry=False)
                        exited = True

            if exited:
                roi_net = net_roi_from_prices(entry_price, exit_price, position, pos["active_leverage"])
                pnl_net = roi_net * pos["active_margin"]
                equity += pnl_net

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": exit_time,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "signal_atr": pos["used_signal_atr"],
                    "stop_price": pos["stop_price"],
                    "exit_reason": exit_reason,
                    "forced_exit": exit_reason == FORCED_EXIT_REASON,
                    "roi_net": roi_net,
                    "pnl_net": pnl_net,
                    "margin_used": pos["active_margin"],
                    "notional": pos["active_notional"],
                    "equity_after": equity,
                    "bars_held": i - (pos["entry_index"] if pos["entry_index"] is not None else i),
                    "lib_mode": has_1m_data,
                    "trade_id": pos["trade_id"],
                    "signal_reason": pos["signal_reason"],
                    "strategy_version": STRATEGY_VERSION,
                })

                positions.remove(pos)

        # --- Trailing stop update for 1m close (if no 1m data or using 1m fallback) ---
        if not has_1m_data and cfg.use_trailing_stop:
            for pos in positions:
                position = pos["position"]
                entry_price = pos["entry_price"]
                if entry_price is None:
                    continue

                c = float(df.at[t, "close"])
                h = float(df.at[t, "high"])
                l = float(df.at[t, "low"])
                side_str = "LONG" if position == 1 else "SHORT"
                prev_stop_price = pos["stop_price"]

                if cfg.trailing_mode == "dynamic_atr":
                    signal_atr_for_trail = (
                        pos["trail_r_value"] if pos["trail_r_value"] is not None else float(df.at[t, "atr"])
                    )

                    if cfg.dynamic_trail_price_source == "high_low":
                        trail_price = h if position == 1 else l
                    else:
                        trail_price = c
                    atr_stop = compute_dynamic_atr_stop(
                        trail_price, signal_atr_for_trail, cfg.dynamic_trail_atr_mult, position
                    )

                    new_stop = atr_stop

                    activation_threshold = signal_atr_for_trail * cfg.dynamic_trail_activation_r
                    if position == 1:
                        price_moved_enough = c >= entry_price + activation_threshold
                        should_set_stop = price_moved_enough
                    else:
                        price_moved_enough = c <= entry_price - activation_threshold
                        should_set_stop = price_moved_enough

                    stop_moved = should_set_stop and (
                        pos["stop_price"] is None or new_stop != pos["stop_price"]
                    )
                    if stop_moved:
                        pos["stop_price"] = new_stop

                    trailing_stop_updates.append({
                        "timestamp": t,
                        "entry_time": pos["entry_time"],
                        "side": side_str,
                        "entry_price": entry_price,
                        "close_price": c,
                        "trail_price": trail_price,
                        "signal_atr": signal_atr_for_trail,
                        "atr_stop": atr_stop,
                        "prev_stop_price": prev_stop_price,
                        "new_stop_price": pos["stop_price"],
                        "trailing_mode": "dynamic_atr",
                        "dynamic_trail_atr_mult": cfg.dynamic_trail_atr_mult,
                        "dynamic_trail_activation_r": cfg.dynamic_trail_activation_r,
                        "dynamic_trail_price_source": cfg.dynamic_trail_price_source,
                        "price_moved_enough": price_moved_enough,
                        "stop_moved": stop_moved,
                        "lib_mode": False,
                    })

                elif (
                    cfg.trailing_mode == "r_ladder"
                    and pos["trail_r_value"] is not None
                    and pos["trail_best_close_r"] is not None
                    and pos["trail_stop_r"] is not None
                ):
                    close_r = compute_close_r(entry_price, c, position, pos["trail_r_value"])
                    if close_r is not None:
                        prev_best_r = pos["trail_best_close_r"]
                        prev_stop_r = pos["trail_stop_r"]

                        pos["trail_best_close_r"] = max(pos["trail_best_close_r"], close_r)
                        next_stop_r = compute_trailing_stop_r(
                            pos["trail_best_close_r"], pos["trail_stop_r"], cfg.trail_gap_r
                        )

                        should_place_stop = (
                            (pos["stop_price"] is None and next_stop_r >= 0) or
                            (next_stop_r > pos["trail_stop_r"] and next_stop_r >= 0)
                        )
                        if should_place_stop:
                            new_stop_price = compute_trailing_sl_price(
                                entry_price, position, pos["trail_r_value"], next_stop_r, cfg.trail_buffer_r
                            )
                            trailing_stop_updates.append({
                                "timestamp": t,
                                "entry_time": pos["entry_time"],
                                "side": side_str,
                                "entry_price": entry_price,
                                "close_price": c,
                                "close_r": close_r,
                                "prev_best_r": prev_best_r,
                                "new_best_r": pos["trail_best_close_r"],
                                "prev_stop_r": prev_stop_r,
                                "new_stop_r": next_stop_r,
                                "prev_stop_price": prev_stop_price,
                                "new_stop_price": new_stop_price,
                                "r_value": pos["trail_r_value"],
                                "trail_gap_r": cfg.trail_gap_r,
                                "trail_buffer_r": cfg.trail_buffer_r,
                                "trailing_mode": "r_ladder",
                                "stop_moved": True,
                                "lib_mode": False,
                            })
                            pos["trail_stop_r"] = next_stop_r
                            pos["stop_price"] = new_stop_price

        # --- ENTRY logic ---
        if pending_entries:
            pending_entries = [
                pending for pending in pending_entries
                if (i - pending["created_index"]) < limit_timeout_bars
            ]

        for pending in list(pending_entries):
            fill_raw = None
            fill_time = t

            if has_1m_data:
                fill_raw, fill_time = maybe_fill_limit_1s(int(pending["side"]), float(pending["limit_price"]), df_1m_bar)
            else:
                side = int(pending["side"])
                limit_price = float(pending["limit_price"])
                if side == 1:
                    if o <= limit_price:
                        fill_raw = o
                    elif l <= limit_price <= h:
                        fill_raw = limit_price
                elif side == -1:
                    if o >= limit_price:
                        fill_raw = o
                    elif l <= limit_price <= h:
                        fill_raw = limit_price

            if fill_raw is None:
                continue

            entry_price = apply_slippage(float(fill_raw), int(pending["side"]), is_entry=True)
            pos = create_position(
                int(pending["side"]),
                entry_price,
                float(pending["signal_atr"]),
                fill_time if has_1m_data else t,
                i,
                pending.get("signal_reason"),
            )
            pending_entries.remove(pending)
            if pos is None:
                continue

            exited, exit_reason, exit_price_check, exit_time_check = process_entry_immediate_exit(pos, df_1m_bar, t)
            if exited:
                roi_net = net_roi_from_prices(entry_price, exit_price_check, pos["position"], pos["active_leverage"])
                pnl_net = roi_net * pos["active_margin"]
                equity += pnl_net
                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": exit_time_check,
                    "side": "LONG" if pos["position"] == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price_check,
                    "signal_atr": pos["used_signal_atr"],
                    "stop_price": pos["stop_price"],
                    "exit_reason": exit_reason,
                    "forced_exit": exit_reason == FORCED_EXIT_REASON,
                    "roi_net": roi_net,
                    "pnl_net": pnl_net,
                    "margin_used": pos["active_margin"],
                    "notional": pos["active_notional"],
                    "equity_after": equity,
                    "bars_held": 0,
                    "lib_mode": True,
                    "trade_id": pos["trade_id"],
                    "signal_reason": pos["signal_reason"],
                    "strategy_version": STRATEGY_VERSION,
                })
            else:
                positions.append(pos)

        prev_signal = int(df.at[prev_t, "signal"])
        prev_signal_atr = df.at[prev_t, "signal_atr"]
        prev_entry_price = df.at[prev_t, "signal_entry_price"]
        prev_signal_reason = df.at[prev_t, "signal_reason"] if "signal_reason" in df.columns else None

        if prev_signal != 0 and not (isinstance(prev_signal_atr, float) and math.isnan(prev_signal_atr)):
            side = prev_signal
            signal_atr_value = float(prev_signal_atr)

            if prev_entry_price is not None and not (isinstance(prev_entry_price, float) and math.isnan(prev_entry_price)):
                fill_raw = None
                fill_time = t
                if has_1m_data:
                    fill_raw, fill_time = maybe_fill_limit_1s(side, float(prev_entry_price), df_1m_bar)
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

                if fill_raw is None:
                    pending_entries.append({
                        "side": side,
                        "limit_price": float(prev_entry_price),
                        "signal_atr": signal_atr_value,
                        "created_index": i,
                        "signal_reason": prev_signal_reason,
                    })
                else:
                    entry_price = apply_slippage(float(fill_raw), side, is_entry=True)
                    pos = create_position(
                        side,
                        entry_price,
                        signal_atr_value,
                        fill_time if has_1m_data else t,
                        i,
                        prev_signal_reason,
                    )
                    if pos is not None:
                        exited, exit_reason, exit_price_check, exit_time_check = process_entry_immediate_exit(pos, df_1m_bar, t)
                        if exited:
                            roi_net = net_roi_from_prices(entry_price, exit_price_check, pos["position"], pos["active_leverage"])
                            pnl_net = roi_net * pos["active_margin"]
                            equity += pnl_net
                            trades.append({
                                "entry_time": pos["entry_time"],
                                "exit_time": exit_time_check,
                                "side": "LONG" if pos["position"] == 1 else "SHORT",
                                "entry_price": entry_price,
                                "exit_price": exit_price_check,
                                "signal_atr": pos["used_signal_atr"],
                                "stop_price": pos["stop_price"],
                                "exit_reason": exit_reason,
                                "forced_exit": exit_reason == FORCED_EXIT_REASON,
                                "roi_net": roi_net,
                                "pnl_net": pnl_net,
                                "margin_used": pos["active_margin"],
                                "notional": pos["active_notional"],
                                "equity_after": equity,
                                "bars_held": 0,
                                "lib_mode": True,
                                "trade_id": pos["trade_id"],
                                "signal_reason": pos["signal_reason"],
                                "strategy_version": STRATEGY_VERSION,
                            })
                        else:
                            positions.append(pos)
            else:
                entry_price = apply_slippage(o, side, is_entry=True)
                pos = create_position(side, entry_price, signal_atr_value, t, i, prev_signal_reason)
                if pos is not None:
                    exited, exit_reason, exit_price_check, exit_time_check = process_entry_immediate_exit(pos, df_1m_bar, t)
                    if exited:
                        roi_net = net_roi_from_prices(entry_price, exit_price_check, pos["position"], pos["active_leverage"])
                        pnl_net = roi_net * pos["active_margin"]
                        equity += pnl_net
                        trades.append({
                            "entry_time": pos["entry_time"],
                            "exit_time": exit_time_check,
                            "side": "LONG" if pos["position"] == 1 else "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price_check,
                            "signal_atr": pos["used_signal_atr"],
                            "stop_price": pos["stop_price"],
                            "exit_reason": exit_reason,
                            "forced_exit": exit_reason == FORCED_EXIT_REASON,
                            "roi_net": roi_net,
                            "pnl_net": pnl_net,
                            "margin_used": pos["active_margin"],
                            "notional": pos["active_notional"],
                            "equity_after": equity,
                            "bars_held": 0,
                            "lib_mode": True,
                            "trade_id": pos["trade_id"],
                            "signal_reason": pos["signal_reason"],
                            "strategy_version": STRATEGY_VERSION,
                        })
                    else:
                        positions.append(pos)

        equity_series.at[t] = equity

    df["equity"] = equity_series.ffill()

    # Close open positions at end of backtest (mark-to-market)
    if positions:
        last_close = float(df.iloc[-1]["close"])
        for pos in list(positions):
            entry_price = pos["entry_price"]
            if entry_price is None:
                positions.remove(pos)
                continue
            exit_price = apply_slippage(last_close, pos["position"], is_entry=False)
            roi_net = net_roi_from_prices(entry_price, exit_price, pos["position"], pos["active_leverage"])
            pnl_net = roi_net * pos["active_margin"]
            equity += pnl_net

            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": df.index[-1],
                "side": "LONG" if pos["position"] == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "signal_atr": pos["used_signal_atr"],
                "stop_price": pos["stop_price"],
                "exit_reason": "END_OF_BACKTEST",
                "roi_net": roi_net,
                "pnl_net": pnl_net,
                "margin_used": pos["active_margin"],
                "notional": pos["active_notional"],
                "equity_after": equity,
                "bars_held": len(df) - 1 - (pos["entry_index"] if pos["entry_index"] is not None else 0),
                "lib_mode": True,
                "trade_id": pos["trade_id"],
                "signal_reason": pos["signal_reason"],
                "strategy_version": STRATEGY_VERSION,
            })
            positions.remove(pos)

    trades_df = pd.DataFrame(trades)
    trailing_df = pd.DataFrame(trailing_stop_updates)
    stats = compute_stats(trades_df, df, cfg)

    return trades_df, df, stats, trailing_df


def compute_live_signal(df: pd.DataFrame, cfg: LiveConfig) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[str]]:
    warmup = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if len(df) <= warmup:
        return 0, None, None, None, None

    atr = atr_ema(df, cfg.atr_len)
    
    # Compute market structure (HH/HL/LL/LH) levels
    structure_points, hh_series, hl_series, ll_series, lh_series = compute_market_structure_levels(
        df,
        atr,
        swing_timeframe=cfg.swing_timeframe,
        left=cfg.swing_left,
        right=cfg.swing_right,
        resample_rule=cfg.swing_resample_rule,
    )
    
    # Generate signals based on market structure
    signal, signal_atr, signal_entry_price, signal_reason, trend = build_market_structure_signals(
        df,
        atr,
        structure_points,
        body_atr_mult=cfg.thr2,
        structure_proximity_atr_mult=cfg.swing_proximity_atr_mult,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
    )

    signal_val = int(signal.iloc[-1])
    atr_val = float(atr.iloc[-1])
    signal_atr_val = signal_atr.iloc[-1]
    entry_price_val = signal_entry_price.iloc[-1]
    reason_val = signal_reason.iloc[-1]
    if signal_val == 0 or pd.isna(signal_atr_val) or math.isnan(atr_val):
        return 0, None, atr_val if not math.isnan(atr_val) else None, None, None
    entry_price: Optional[float]
    if entry_price_val is not None and not pd.isna(entry_price_val):
        entry_price = float(entry_price_val)
    else:
        entry_price = None
    reason: Optional[str] = str(reason_val) if reason_val else None
    return signal_val, float(signal_atr_val), atr_val, entry_price, reason


# Backward compatibility functions for backtest code
# These are not used in live trading (trailing stop only, no TP/SL)

def compute_tp_sl_prices(
    entry_price: float,
    side: int,
    atr_value: float,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> Tuple[float, float]:
    """Calculate TP and SL prices (backward compatibility for backtest)."""
    move_tp = atr_value * tp_atr_mult
    move_sl = atr_value * sl_atr_mult
    if side == 1:
        target_price = entry_price + move_tp
        stop_price = entry_price - move_sl
    else:
        target_price = entry_price - move_tp
        stop_price = entry_price + move_sl
    return target_price, stop_price


def compute_sl_price(entry_price: float, side: int, atr_value: float, sl_atr_mult: float = 1.0) -> float:
    """Calculate SL price (backward compatibility for backtest)."""
    move_sl = atr_value * sl_atr_mult
    if side == 1:
        return entry_price - move_sl
    return entry_price + move_sl


def compute_margin_from_targets(
    entry_price: float,
    atr_value: float,
    sl_atr_mult: float,
    leverage: float,
    target_loss_usd: Optional[float],
) -> Optional[float]:
    """Backward compatibility for backtest."""
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
    """Backward compatibility for backtest."""
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
    """Backward compatibility for backtest."""
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


def compute_dynamic_atr_stop(
    close_price: float,
    current_atr: float,
    atr_mult: float,
    side: int,
) -> float:
    """Calculate stop price based on close - ATR*mult (long) or close + ATR*mult (short)."""
    if side == 1:  # LONG
        return close_price - (current_atr * atr_mult)
    else:  # SHORT
        return close_price + (current_atr * atr_mult)


def is_exit_check_boundary(timestamp: pd.Timestamp, interval: str) -> bool:
    """
    Check if timestamp is at an interval boundary for exit checking.
    
    For 1m candles indexed by open time, the candle at (boundary - 1min) represents
    the close of the higher timeframe candle.
    
    Examples for interval="4h":
        - 03:59 → True (closes at 04:00, which is a 4h boundary)
        - 07:59 → True (closes at 08:00)
        - 04:00 → False (this candle closes at 04:01, not a boundary)
    
    Args:
        timestamp: The 1m candle open timestamp
        interval: Exit check interval like "1h", "4h", "1d"
    
    Returns:
        True if this candle's close is at an interval boundary
    """
    # Parse interval
    if interval.endswith("m"):
        period_minutes = int(interval[:-1])
    elif interval.endswith("h"):
        period_minutes = int(interval[:-1]) * 60
    elif interval.endswith("d"):
        period_minutes = int(interval[:-1]) * 60 * 24
    else:
        # Default to 4h if unparseable
        period_minutes = 240
    
    # The candle close time = timestamp + 1 minute
    close_minute_of_day = timestamp.hour * 60 + timestamp.minute + 1
    
    # Handle midnight rollover (e.g., 23:59 close = 24:00 = 0 next day)
    if close_minute_of_day >= 24 * 60:
        close_minute_of_day = 0
    
    # Check if close time is at a boundary
    return close_minute_of_day % period_minutes == 0


# NOTE: compute_sl_limit_price removed - using market orders for trailing stop
# NOTE: compute_margin_from_targets, compute_required_leverage, resolve_trade_sizing removed
#       - using static position sizing (margin_usd x leverage)


def compute_position_qty(
    entry_price: float,
    margin_usd: float,
    leverage: int,
    step_size: float,
) -> str:
    """
    Compute position quantity for static sizing.
    
    Args:
        entry_price: Expected entry price
        margin_usd: Margin amount in USD (e.g., 5.0)
        leverage: Leverage multiplier (e.g., 20)
        step_size: Minimum quantity step from exchange filters
    
    Returns:
        Quantity string formatted to step size
    """
    if entry_price <= 0 or margin_usd <= 0 or leverage <= 0:
        return "0"
    notional = margin_usd * leverage
    qty = notional / entry_price
    return format_to_step(qty, step_size)



def run_live(cfg: LiveConfig) -> None:
    """
    Live trading for 1d ATR strategy with multi-position support.
    
    Features:
    - Trades all USDT perpetual contracts on Binance
    - 1d candles for signal generation
    - 1m candles for trailing stop updates (Look In Bar mode)
    - Trailing stop only (no TP/SL on entry)
    - Algo conditional stop orders for trailing exits
    - Static position sizing (margin_usd x leverage)
    """
    if load_dotenv is not None:
        load_dotenv()
    
    if getenv("LIVE_TRADING") != "1":
        raise RuntimeError("Set LIVE_TRADING=1 to enable live trading.")

    ensure_output_dirs()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # Initialize Binance client
    client = get_um_futures_client(cfg)
    
    # Fetch all USDT perpetual symbols and filters in a single API call
    logging.info("Fetching all USDT perpetual symbols and filters...")
    all_symbols, filters_by_symbol = get_all_usdt_perpetuals_with_filters(client)
    logging.info("Found %d USDT perpetual symbols with filters", len(all_symbols))
    
    # Initialize state
    state = LiveState()
    
    # Track ATR values per symbol for trailing stop R calculation
    last_atr_by_symbol: Dict[str, Optional[float]] = {}

    def safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def fetch_latest_atr(symbol: str) -> Optional[float]:
        cached = last_atr_by_symbol.get(symbol)
        if cached is not None and cached > 0:
            return cached
        history = max(cfg.atr_history_bars, cfg.atr_len + 2)
        if history > 1000:
            history = 1000
        try:
            klines = client.klines(symbol=symbol, interval=cfg.signal_interval, limit=history)
        except ClientError as exc:
            logging.warning("ATR klines fetch failed. symbol=%s error=%s", symbol, exc)
            return None
        if len(klines) < 2:
            return None
        df = klines_to_df(klines[:-1])
        if df.empty:
            return None
        atr = atr_ema(df, cfg.atr_len)
        if atr.empty:
            return None
        atr_val = float(atr.iloc[-1])
        if math.isnan(atr_val):
            return None
        last_atr_by_symbol[symbol] = atr_val
        return atr_val

    def restore_open_positions_from_exchange() -> int:
        try:
            raw_positions = client.get_position_risk()
        except ClientError as exc:
            logging.warning("Position restore failed. error=%s", exc)
            log_event(cfg.log_path, {"event": "position_restore_failed", "error": str(exc)})
            return 0

        positions = raw_positions if isinstance(raw_positions, list) else [raw_positions]
        restored = 0
        for pos in positions:
            symbol = str(pos.get("symbol") or "")
            if not symbol or symbol not in filters_by_symbol:
                continue
            amt = safe_float(pos.get("positionAmt"), 0.0)
            if amt == 0.0:
                continue
            if state.has_position(symbol):
                continue

            entry_price = safe_float(pos.get("entryPrice"), 0.0)
            if entry_price <= 0:
                entry_price = safe_float(pos.get("markPrice"), 0.0)
            if entry_price <= 0:
                continue

            side = 1 if amt > 0 else -1
            qty = abs(amt)
            entry_atr = fetch_latest_atr(symbol) or 0.0
            leverage = int(safe_float(pos.get("leverage"), cfg.leverage))
            if leverage <= 0:
                leverage = cfg.leverage
            margin_usd = cfg.margin_usd
            if entry_price > 0 and leverage > 0:
                margin_usd = (qty * entry_price) / leverage

            pos_state = PositionState(
                symbol=symbol,
                entry_price=entry_price,
                entry_side=side,
                entry_qty=qty,
                entry_atr=entry_atr,
                trail_r_value=entry_atr,
                trail_best_close_r=0.0,
                trail_stop_r=cfg.trail_initial_stop_r,
                entry_time_iso=_utc_now_iso(),
                entry_close_ms=int(time.time() * 1000),
                entry_margin_usd=margin_usd,
                entry_leverage=leverage,
            )
            state.positions[symbol] = pos_state
            restored += 1

            tick_size = filters_by_symbol.get(symbol, {}).get("tick_size", 0.00000001)
            log_event(cfg.log_path, {
                "event": "position_restored",
                "symbol": symbol,
                "side": "LONG" if side == 1 else "SHORT",
                "entry_price": format_price_for_logging(entry_price, tick_size),
                "quantity": format_float_2(qty),
                "entry_atr": format_price_for_logging(entry_atr, tick_size) if entry_atr > 0 else "",
            })

        return restored
    
    log_event(cfg.log_path, {
        "event": "startup",
        "total_symbols": len(all_symbols),
        "signal_interval": cfg.signal_interval,
        "trail_check_interval": cfg.trail_check_interval,
        "leverage": cfg.leverage,
        "margin_usd": format_float_2(cfg.margin_usd),
        "max_open_positions": cfg.max_open_positions,
        "entry_signal_workers": cfg.entry_signal_workers,
    })

    restored_positions = restore_open_positions_from_exchange()
    if restored_positions:
        log_event(cfg.log_path, {
            "event": "positions_restored",
            "count": restored_positions,
            "symbols": list(state.positions.keys()),
        })

    signal_workers = max(1, int(cfg.entry_signal_workers))
    signal_log_lock = threading.Lock()
    swing_levels_lock = threading.Lock()

    def chunk_symbols(symbols: List[str], worker_count: int) -> List[List[str]]:
        if not symbols:
            return []
        if worker_count <= 1:
            return [list(symbols)]
        worker_count = min(worker_count, len(symbols))
        chunk_size = int(math.ceil(len(symbols) / worker_count))
        return [list(symbols[i:i + chunk_size]) for i in range(0, len(symbols), chunk_size)]

    # -----------------------------------------------------------------
    # Helper: Update trailing stop for a position using candle data
    # -----------------------------------------------------------------
    def update_trailing_stop(symbol: str, pos_state: PositionState) -> None:
        """Update trailing stop for a position based on latest candle.
        
        Supports two modes:
        - dynamic_atr: Uses candle high (LONG) or low (SHORT) with ATR multiplier
        - r_ladder: Uses close price with R-unit ladder system
        """
        if symbol not in filters_by_symbol:
            return
        
        filters = filters_by_symbol[symbol]
        
        # Fetch latest candle for trailing stop check
        try:
            klines = client.klines(symbol=symbol, interval=cfg.trail_check_interval, limit=2)
        except ClientError as exc:
            logging.warning("Trailing klines fetch failed. symbol=%s error=%s", symbol, exc)
            return
        
        if len(klines) < 2:
            return
        
        # Use the last closed candle
        closed_klines = klines[:-1]
        close_time_ms = int(closed_klines[-1][6])
        
        # Check if this is a new candle
        if close_time_ms == state.last_trail_close_ms.get(symbol):
            return
        state.last_trail_close_ms[symbol] = close_time_ms
        
        df = klines_to_df(closed_klines)
        candle = df.iloc[-1]
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])
        
        entry_price = pos_state.entry_price
        side_sign = pos_state.entry_side
        signal_atr = pos_state.trail_r_value  # signal_atr is stored in trail_r_value
        
        if signal_atr <= 0:
            return
        
        next_sl_price = None
        candidate_stop_r = None
        
        if cfg.trailing_mode == "dynamic_atr":
            # Dynamic ATR trailing - uses high (LONG) or low (SHORT) with ATR multiplier
            
            # Select price source for trailing
            if cfg.dynamic_trail_price_source == "high_low":
                trail_price = high_price if side_sign == 1 else low_price
            else:
                trail_price = close_price
            
            # Calculate new stop using ATR
            atr_stop = compute_dynamic_atr_stop(trail_price, signal_atr, cfg.dynamic_trail_atr_mult, side_sign)
            
            # Only set stop if price has moved activation_r in our favor
            activation_threshold = signal_atr * cfg.dynamic_trail_activation_r
            if side_sign == 1:
                price_moved_enough = close_price >= entry_price + activation_threshold
            else:
                price_moved_enough = close_price <= entry_price - activation_threshold
            
            if not price_moved_enough:
                return
            
            if pos_state.sl_price is not None and atr_stop == pos_state.sl_price:
                return
            
            next_sl_price = atr_stop
            
        else:
            # R-ladder trailing (original logic)
            close_r = compute_close_r(entry_price, close_price, side_sign, signal_atr)
            if close_r is None:
                return
            
            # Update best close R
            pos_state.trail_best_close_r = max(pos_state.trail_best_close_r, close_r)
            
            # Calculate new trailing stop level
            candidate_stop_r = compute_trailing_stop_r(
                pos_state.trail_best_close_r,
                pos_state.trail_stop_r,
                cfg.trail_gap_r,
            )
            
            # Only place stops at breakeven (0R) or better - no negative stops
            if candidate_stop_r < 0:
                return
            
            # Place stop when:
            # 1. First time reaching ladder threshold (sl_price is None and candidate_stop_r >= 0)
            # 2. Stop level improves (candidate_stop_r > trail_stop_r)
            is_first_stop = pos_state.sl_price is None and candidate_stop_r >= 0
            is_improvement = candidate_stop_r > pos_state.trail_stop_r
            
            if not (is_first_stop or is_improvement):
                return
            
            # Calculate new SL price
            next_sl_price = compute_trailing_sl_price(
                entry_price,
                side_sign,
                signal_atr,
                candidate_stop_r,
                cfg.trail_buffer_r,
            )
        
        if next_sl_price is None:
            return
        
        next_sl_price_str = format_to_step(next_sl_price, filters["tick_size"])
        
        try:
            next_sl_price_rounded = float(next_sl_price_str)
        except (TypeError, ValueError):
            return
        
        # Check if new SL price improves over current
        improves = True
        if pos_state.sl_price is not None:
            current_sl_price = pos_state.sl_price
            improves = (next_sl_price_rounded > current_sl_price if side_sign == 1 
                       else next_sl_price_rounded < current_sl_price)
        
        if not improves or next_sl_price_rounded <= 0:
            return
        
        # Get current position quantity from exchange
        try:
            pos_info = get_position_info(client, symbol)
            position_amt = float(pos_info.get("positionAmt", 0.0) or 0.0)
        except Exception as exc:
            logging.warning("Failed to get position info for %s: %s", symbol, exc)
            return
        
        if position_amt == 0.0:
            # Position was closed manually
            log_event(cfg.log_path, {
                "event": "position_closed_manually",
                "symbol": symbol,
            })
            if symbol in state.positions:
                del state.positions[symbol]
            return
        
        qty = abs(position_amt)
        qty_str = format_to_step(qty, filters["step_size"])
        
        try:
            qty_num = float(qty_str)
        except (TypeError, ValueError):
            qty_num = 0.0
        
        if qty_num < filters["min_qty"]:
            return
        
        # Cancel existing SL algo order if any
        if pos_state.sl_algo_id:
            cancel_algo_order_safely(client, symbol, algo_id=pos_state.sl_algo_id)
            pos_state.sl_algo_id = None
            pos_state.sl_algo_client_order_id = None
        
        # Place trailing stop as an algo conditional STOP_MARKET order
        sl_side = "SELL" if side_sign == 1 else "BUY"
        sl_client_id = f"ATR_SL_T_{close_time_ms}"
        sl_order = algo_stop_limit_order(
            symbol=symbol,
            side=sl_side,
            quantity=qty_str,
            trigger_price=next_sl_price_str,
            price=None,
            client=client,
            client_order_id=sl_client_id,
            algo_type=cfg.algo_type,
            working_type=cfg.algo_working_type,
            price_protect=cfg.algo_price_protect,
            reduce_only=True,
            order_type="STOP_MARKET",
        )
        if not sl_order:
            log_event(cfg.log_path, {
                "event": "sl_trail_order_failed",
                "symbol": symbol,
                "error": "Algo STOP order failed (see logs)",
            })
            return

        pos_state.sl_algo_id = _extract_int(sl_order, ("algoId", "algoOrderId", "orderId", "id"))
        pos_state.sl_algo_client_order_id = sl_client_id
        pos_state.sl_price = next_sl_price_rounded
        if candidate_stop_r is not None:
            pos_state.trail_stop_r = candidate_stop_r
        
        # Get tick_size for proper price precision in log
        sl_tick_size = filters.get("tick_size", 0.00000001)
        log_event(cfg.log_path, {
            "event": "sl_trail_update",
            "symbol": symbol,
            "side": "LONG" if side_sign == 1 else "SHORT",
            "sl_price": format_price_for_logging(next_sl_price_rounded, sl_tick_size),
            "trailing_mode": cfg.trailing_mode,
            "trail_price": format_price_for_logging(high_price if side_sign == 1 else low_price, sl_tick_size) if cfg.trailing_mode == "dynamic_atr" else None,
            "stop_r": candidate_stop_r,
            "best_close_r": format_float_2(pos_state.trail_best_close_r) if cfg.trailing_mode == "r_ladder" else None,
            "order_id": pos_state.sl_algo_id,
        })

    # -----------------------------------------------------------------
    # Helper: Check for entry signal on a symbol using 1d candle
    # -----------------------------------------------------------------
    def check_entry_signal(symbol: str) -> Optional[Tuple[int, float, float, Optional[float], Optional[str]]]:
        """
        Check if there's an entry signal for the symbol based on 1d candle.
        
        Returns:
            Tuple of (signal, signal_atr, atr_value, signal_entry_price, signal_reason) if signal found,
            None otherwise.
        """
        if symbol not in filters_by_symbol:
            return None
        
        # Fetch 1d candles for signal generation
        history = max(cfg.atr_history_bars, cfg.atr_len + 2)
        if history > 1000:
            history = 1000
        
        try:
            klines = client.klines(symbol=symbol, interval=cfg.signal_interval, limit=history)
        except ClientError as exc:
            logging.warning("Signal klines fetch failed. symbol=%s error=%s", symbol, exc)
            return None
        
        if len(klines) < 2:
            return None
        
        closed_klines = klines[:-1]
        close_time_ms = int(closed_klines[-1][6])
        
        # Check if this is a new daily candle
        if close_time_ms == state.last_signal_close_ms.get(symbol):
            return None
        state.last_signal_close_ms[symbol] = close_time_ms
        
        df = klines_to_df(closed_klines)
        signal, signal_atr, atr_value, signal_entry_price, signal_reason = compute_live_signal(df, cfg)
        
        # Store ATR for trailing stop calculation
        last_atr_by_symbol[symbol] = atr_value
        
        # Save detected swing levels to JSON file
        with swing_levels_lock:
            save_live_swing_levels(
                cfg.live_swing_levels_file,
                symbol,
                df,
                cfg.swing_timeframe,
                cfg.swing_left,
                cfg.swing_right,
                cfg.swing_resample_rule,
            )
        
        candle = df.iloc[-1]
        body = abs(float(candle["close"]) - float(candle["open"]))
        
        # Get tick_size for proper price precision
        symbol_filters = filters_by_symbol.get(symbol, {})
        tick_size = symbol_filters.get("tick_size", 0.00000001)
        
        with signal_log_lock:
            log_event(cfg.log_path, {
                "event": "signal_candle_close",
                "symbol": symbol,
                "close_time_ms": close_time_ms,
                "open": format_price_for_logging(candle["open"], tick_size),
                "high": format_price_for_logging(candle["high"], tick_size),
                "low": format_price_for_logging(candle["low"], tick_size),
                "close": format_price_for_logging(candle["close"], tick_size),
                "body": format_price_for_logging(body, tick_size),
                "atr": format_price_for_logging(atr_value, tick_size),
                "signal": signal,
                "signal_reason": signal_reason,
            })
        
        if signal == 0 or signal_atr is None or atr_value is None:
            return None
        
        return (signal, signal_atr, atr_value, signal_entry_price, signal_reason)

    def scan_entry_signal_chunk(
        symbols_chunk: List[str],
    ) -> List[Tuple[str, Tuple[int, float, float, Optional[float], Optional[str]]]]:
        results: List[Tuple[str, Tuple[int, float, float, Optional[float], Optional[str]]]] = []
        for sym in symbols_chunk:
            if sym not in filters_by_symbol:
                continue
            try:
                signal_result = check_entry_signal(sym)
            except Exception as exc:
                logging.warning("Signal check failed. symbol=%s error=%s", sym, exc)
                continue
            if signal_result is None:
                continue
            results.append((sym, signal_result))
        return results

    async def scan_entry_signals(
        symbols: List[str],
    ) -> Dict[str, Tuple[int, float, float, Optional[float], Optional[str]]]:
        if not symbols:
            return {}
        worker_count = min(signal_workers, len(symbols))
        chunks = chunk_symbols(symbols, worker_count)
        loop = asyncio.get_running_loop()
        results: Dict[str, Tuple[int, float, float, Optional[float], Optional[str]]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            tasks = [
                loop.run_in_executor(executor, scan_entry_signal_chunk, chunk)
                for chunk in chunks
            ]
            for chunk_results in await asyncio.gather(*tasks):
                for sym, signal_result in chunk_results:
                    results[sym] = signal_result
        return results

    # -----------------------------------------------------------------
    # Helper: Place entry order for a symbol
    # -----------------------------------------------------------------
    def place_entry_order(
        symbol: str,
        signal: int,
        signal_atr: float,
        atr_value: float,
        signal_entry_price: Optional[float],
        trade_id: Optional[str] = None,
        signal_reason: Optional[str] = None,
    ) -> bool:
        """
        Place a limit entry order for the symbol.
        
        Returns:
            True if order was placed successfully, False otherwise.
        """
        filters = filters_by_symbol[symbol]
        
        # Small delay before entry
        delay = cfg.entry_delay_min_seconds
        if cfg.entry_delay_max_seconds > cfg.entry_delay_min_seconds:
            delay = random.uniform(cfg.entry_delay_min_seconds, cfg.entry_delay_max_seconds)
        time.sleep(delay)
        
        # Get current bid/ask
        try:
            bid, ask = get_book_ticker(client, symbol)
        except ClientError as exc:
            logging.warning("Book ticker fetch failed. symbol=%s error=%s", symbol, exc)
            return False
        
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid else 1.0
        if spread_pct > cfg.spread_max_pct:
            log_event(cfg.log_path, {
                "event": "skip_spread",
                "symbol": symbol,
                "spread_pct": format_float_2(spread_pct),
            })
            return False
        
        side = "BUY" if signal == 1 else "SELL"
        
        # Calculate limit price
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
        
        limit_price_str = format_price_to_tick(limit_price, tick, side=side, post_only=True)
        try:
            limit_price = float(limit_price_str)
        except (TypeError, ValueError):
            limit_price = 0.0
        
        if limit_price <= 0:
            return False
        
        # Static position sizing: margin_usd x leverage
        # Ensure leverage is set
        current_leverage = state.current_leverage_by_symbol.get(symbol)
        if current_leverage != cfg.leverage:
            try:
                client.change_leverage(symbol=symbol, leverage=cfg.leverage)
                state.current_leverage_by_symbol[symbol] = cfg.leverage
            except ClientError:
                log_event(cfg.log_path, {
                    "event": "leverage_change_error",
                    "symbol": symbol,
                    "leverage": cfg.leverage,
                })
                return False
        
        # Calculate quantity
        qty_str = compute_position_qty(limit_price, cfg.margin_usd, cfg.leverage, filters["step_size"])
        try:
            qty_num = float(qty_str)
        except (TypeError, ValueError):
            qty_num = 0.0
        
        if qty_num < filters["min_qty"]:
            log_event(cfg.log_path, {
                "event": "skip_qty",
                "symbol": symbol,
                "quantity": qty_str,
            })
            return False
        
        close_time_ms = state.last_signal_close_ms.get(symbol, int(time.time() * 1000))
        client_order_id = f"ATR_E_{close_time_ms}"
        
        # Place limit entry order
        entry_order = limit_order(
            symbol,
            side,
            qty_str,
            limit_price_str,
            client,
            client_order_id=client_order_id,
            tick_size=tick,
        )
        
        if entry_order:
            order_id = entry_order.get("orderId")
            
            # Track pending entry
            pending = PendingEntry(
                symbol=symbol,
                order_id=order_id,
                order_time=time.time(),
                pending_atr=signal_atr,
                pending_side=1 if side == "BUY" else -1,
                entry_close_ms=close_time_ms,
                trade_id=trade_id,
                signal_reason=signal_reason,
            )
            state.pending_entries[symbol] = pending
            
            log_event(cfg.log_path, {
                "event": "entry_order",
                "symbol": symbol,
                "order_id": order_id,
                "side": side,
                "limit_price": limit_price_str,
                "quantity": qty_str,
                "signal_atr": format_float_2(signal_atr),
                "margin_usd": format_float_2(cfg.margin_usd),
                "leverage": cfg.leverage,
            })
            return True
        return False

    # -----------------------------------------------------------------
    # Main trading loop
    # -----------------------------------------------------------------
    logging.info("Starting main trading loop...")
    
    while True:
        try:
            # --- Step 1: Sync positions with exchange ---
            # Get all positions and identify which are ours (have strategy tracking)
            tracked_symbols = set(state.positions.keys())
            pending_symbols = set(state.pending_entries.keys())
            
            # --- Step 2: Check pending entry orders ---
            for symbol in list(pending_symbols):
                pending = state.pending_entries[symbol]
                try:
                    order = client.query_order(symbol=symbol, orderId=pending.order_id)
                except ClientError as exc:
                    log_event(cfg.log_path, {
                        "event": "entry_order_query_error",
                        "symbol": symbol,
                        "order_id": pending.order_id,
                        "error": str(exc),
                    })
                    del state.pending_entries[symbol]
                    continue
                
                status = order.get("status")
                if status == "FILLED":
                    # Entry filled - create position state
                    filters = filters_by_symbol.get(symbol, {})
                    refreshed_position = get_position_info(client, symbol)
                    
                    try:
                        position_amt = float(refreshed_position.get("positionAmt", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        position_amt = 0.0
                    
                    entry_price = 0.0
                    if refreshed_position:
                        try:
                            entry_price = float(refreshed_position.get("entryPrice", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            entry_price = 0.0
                    if entry_price <= 0:
                        entry_price = float(order.get("avgPrice") or order.get("price") or 0.0)
                    
                    if position_amt != 0.0 and entry_price > 0:
                        side = 1 if position_amt > 0 else -1
                        qty = abs(position_amt)
                        
                        # R value for trailing stop = ATR at entry
                        r_value = pending.pending_atr
                        
                        pos_state = PositionState(
                            symbol=symbol,
                            entry_price=entry_price,
                            entry_side=side,
                            entry_qty=qty,
                            entry_atr=pending.pending_atr,
                            trail_r_value=r_value,
                            trail_best_close_r=0.0,
                            trail_stop_r=cfg.trail_initial_stop_r,  # Initial stop R level from config
                            entry_time_iso=_utc_now_iso(),
                            entry_close_ms=pending.entry_close_ms,
                            entry_margin_usd=cfg.margin_usd,
                            entry_leverage=cfg.leverage,
                            trade_id=pending.trade_id,
                            signal_reason=pending.signal_reason,
                        )
                        state.positions[symbol] = pos_state
                        
                        # Get tick_size for proper price precision in log
                        entry_tick_size = filters.get("tick_size", 0.00000001)
                        log_event(cfg.log_path, {
                            "event": "entry_filled",
                            "symbol": symbol,
                            "side": "LONG" if side == 1 else "SHORT",
                            "entry_price": format_price_for_logging(entry_price, entry_tick_size),
                            "quantity": format_float_2(qty),
                            "r_value": format_price_for_logging(r_value, entry_tick_size),
                        })
                    
                    del state.pending_entries[symbol]
                
                elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
                    log_event(cfg.log_path, {
                        "event": "entry_order_inactive",
                        "symbol": symbol,
                        "status": status,
                    })
                    del state.pending_entries[symbol]
                
                elif time.time() - pending.order_time > cfg.entry_order_timeout_seconds:
                    # Timeout - cancel order
                    try:
                        client.cancel_order(symbol=symbol, orderId=pending.order_id)
                    except ClientError:
                        pass
                    log_event(cfg.log_path, {
                        "event": "entry_order_timeout",
                        "symbol": symbol,
                    })
                    del state.pending_entries[symbol]
            
            # --- Step 3: Update trailing stops for tracked positions ---
            for symbol in list(state.positions.keys()):
                pos_state = state.positions[symbol]
                
                # Verify position still exists on exchange
                try:
                    pos_info = get_position_info(client, symbol)
                    position_amt = float(pos_info.get("positionAmt", 0.0) or 0.0)
                except Exception:
                    position_amt = 0.0
                
                if position_amt == 0.0:
                    # Position was closed (manually or by SL)
                    log_event(cfg.log_path, {
                        "event": "position_closed",
                        "symbol": symbol,
                        "exit_reason": "manual_or_sl",
                    })
                    
                    # Record trade
                    append_live_trade(cfg.live_trades_csv, {
                        "entry_time": pos_state.entry_time_iso,
                        "exit_time": _utc_now_iso(),
                        "symbol": symbol,
                        "side": "LONG" if pos_state.entry_side == 1 else "SHORT",
                        "entry_price": pos_state.entry_price,
                        "exit_price": pos_state.sl_price or 0,
                        "signal_atr": pos_state.entry_atr,
                        "stop_price": pos_state.sl_price or 0,
                        "exit_reason": "trailing_sl",
                        "margin_used": pos_state.entry_margin_usd,
                        "notional": pos_state.entry_margin_usd * pos_state.entry_leverage,
                        "trade_id": pos_state.trade_id,
                        "signal_reason": pos_state.signal_reason,
                        "strategy_version": STRATEGY_VERSION,
                    })
                    
                    # Cancel any pending SL orders
                    if pos_state.sl_algo_id:
                        cancel_algo_order_safely(client, symbol, algo_id=pos_state.sl_algo_id)
                    
                    del state.positions[symbol]
                    continue
                
                # Update trailing stop using 1m candles
                update_trailing_stop(symbol, pos_state)
            
            # --- Step 4: Check for new entry signals ---
            # Scan all symbols for entry signals concurrently (log all signals, even if skipped)
            eligible_symbols = [sym for sym in all_symbols if sym in filters_by_symbol]
            signal_results: Dict[str, Tuple[int, float, float, Optional[float], Optional[str]]] = {}
            if eligible_symbols:
                try:
                    signal_results = asyncio.run(scan_entry_signals(eligible_symbols))
                except Exception as exc:
                    logging.warning("Entry signal scan failed: %s", exc)
                    signal_results = {}

            for symbol in eligible_symbols:
                signal_result = signal_results.get(symbol)
                if signal_result is None:
                    continue
                
                signal, signal_atr, atr_value, signal_entry_price, signal_reason = signal_result
                side_str = "LONG" if signal == 1 else "SHORT"
                
                # Get candle data for logging
                try:
                    klines = client.klines(symbol=symbol, interval=cfg.signal_interval, limit=2)
                    if klines and len(klines) >= 2:
                        candle = klines[-2]  # Last closed candle
                        candle_open = float(candle[1])
                        candle_high = float(candle[2])
                        candle_low = float(candle[3])
                        candle_close = float(candle[4])
                    else:
                        candle_open = candle_high = candle_low = candle_close = 0.0
                except Exception:
                    candle_open = candle_high = candle_low = candle_close = 0.0
                
                # Determine signal status
                if state.has_position(symbol):
                    status = "SKIPPED_HAS_POS"
                elif state.has_pending_entry(symbol):
                    status = "SKIPPED_PENDING"
                elif not state.can_open_position(cfg.max_open_positions):
                    status = "SKIPPED_MAX_POS"
                elif not is_within_entry_window(cfg.entry_window_minutes):
                    status = "SKIPPED_OUTSIDE_WINDOW"
                else:
                    status = "ACTED"
                
                # Log signal to CSV and console
                # Get tick_size for proper price precision
                symbol_filters = filters_by_symbol.get(symbol, {})
                tick_size = symbol_filters.get("tick_size", 0.00000001)
                
                # Generate trade_id if signal is acted upon
                live_trade_id = None
                if status == "ACTED":
                    live_trade_id = generate_trade_id(symbol, _utc_now_iso(), side_str)
                
                signal_data = {
                    "timestamp": _utc_now_iso(),
                    "symbol": symbol,
                    "side": side_str,
                    "signal": signal,
                    "signal_atr": format_price_for_logging(signal_atr, tick_size),
                    "entry_price": format_price_for_logging(signal_entry_price, tick_size) if signal_entry_price else "MARKET",
                    "open": format_price_for_logging(candle_open, tick_size),
                    "high": format_price_for_logging(candle_high, tick_size),
                    "low": format_price_for_logging(candle_low, tick_size),
                    "close": format_price_for_logging(candle_close, tick_size),
                    "atr": format_price_for_logging(atr_value, tick_size),
                    "status": status,
                    "signal_reason": signal_reason or "",
                    "trade_id": live_trade_id,
                    "strategy_version": STRATEGY_VERSION,
                }
                append_live_signal(cfg.live_signals_csv, signal_data)
                
                # Print signal to terminal with proper precision
                precision = get_precision_from_tick_size(tick_size)
                entry_str = f"{signal_entry_price:.{precision}f}" if signal_entry_price else "MARKET"
                atr_str = f"{atr_value:.{precision}f}" if atr_value else "N/A"
                print(f"[SIGNAL] {symbol} {side_str} @ {entry_str} (ATR: {atr_str}) - {status} [{signal_reason or 'unknown'}]")
                
                # Log to event log
                log_event(cfg.log_path, {
                    "event": "signal_detected",
                    "symbol": symbol,
                    "side": side_str,
                    "signal_atr": format_price_for_logging(signal_atr, tick_size),
                    "entry_price": format_price_for_logging(signal_entry_price, tick_size) if signal_entry_price else None,
                    "atr": format_price_for_logging(atr_value, tick_size),
                    "status": status,
                    "signal_reason": signal_reason,
                })
                
                # Place entry order if conditions allow
                if status == "ACTED":
                    if place_entry_order(symbol, signal, signal_atr, atr_value, signal_entry_price, 
                                        trade_id=live_trade_id, signal_reason=signal_reason):
                        # Successfully placed entry order
                        logging.info("Entry order placed for %s", symbol)
                        
                        # Check if we've hit max positions
                        if not state.can_open_position(cfg.max_open_positions):
                            continue  # Continue scanning to log remaining signals
            
            # Log loop status
            position_count = state.position_count()
            pending_count = len(state.pending_entries)
            if position_count > 0 or pending_count > 0:
                log_event(cfg.log_path, {
                    "event": "loop_status",
                    "positions": position_count,
                    "pending_entries": pending_count,
                    "tracked_symbols": list(state.positions.keys()),
                })
        
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
def run_backtest_single(
    symbol: str,
    start_date: str,
    end_date: str,
    use_lib: bool,
    force_refetch: bool,
    market: str,
    cfg: BacktestConfig,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Run backtest for a single symbol. Returns (trades_df, stats_dict, signals_df).
    """
    import time as time_module
    
    signal_interval = cfg.signal_interval
    df_1m = pd.DataFrame()
    
    if use_lib:
        if verbose:
            print(f"  Loading 1m data...")
        df_1m = fetch_klines_public_data(
            symbol=symbol,
            interval="1m",
            start_date=start_date,
            end_date=end_date,
            save_csv=True,
            force_refetch=force_refetch,
            market=market,
        )
        if df_1m.empty:
            if verbose:
                print(f"  WARNING: No 1m data for {symbol}, skipping.")
            return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()
    
    if verbose:
        print(f"  Loading {signal_interval} data...")
    df_signal = fetch_klines_public_data(
        symbol=symbol,
        interval=signal_interval,
        start_date=start_date,
        end_date=end_date,
        save_csv=True,
        force_refetch=force_refetch,
        market=market,
    )
    
    if df_signal.empty:
        if verbose:
            print(f"  WARNING: No signal data for {symbol}, skipping.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()
    
    if verbose:
        print(f"  Running backtest ({len(df_signal)} candles)...")
    
    if use_lib and not df_1m.empty:
        trades, df_bt, stats, trailing_df = backtest_atr_grinder_lib(df_signal, df_1m, cfg)
    else:
        trades, df_bt, stats, trailing_df = backtest_atr_grinder(df_signal, cfg)
    
    # Add symbol column to trades
    if not trades.empty:
        trades["symbol"] = symbol
    
    # Extract all signals (non-zero)
    signals_df = pd.DataFrame()
    if "signal" in df_bt.columns:
        signals_mask = df_bt["signal"] != 0
        if signals_mask.any():
            # Include market structure levels (HH/HL/LL/LH) and trend
            cols_to_extract = ["open", "high", "low", "close", "signal", "signal_atr", "signal_entry_price", "signal_reason"]
            # Add market structure columns if they exist
            for col in ["hh_level", "hl_level", "ll_level", "lh_level", "trend"]:
                if col in df_bt.columns:
                    cols_to_extract.append(col)
            
            signals_df = df_bt[signals_mask][cols_to_extract].copy()
            signals_df = signals_df.reset_index()
            # Rename base columns
            rename_map = {
                df_bt.index.name or "index": "timestamp",
                "signal_entry_price": "entry_price",
            }
            signals_df = signals_df.rename(columns=rename_map)
            signals_df["symbol"] = symbol
            signals_df["side"] = signals_df["signal"].map({1: "LONG", -1: "SHORT"})
            # Map trend to string for readability
            if "trend" in signals_df.columns:
                signals_df["trend_dir"] = signals_df["trend"].map({1: "UPTREND", -1: "DOWNTREND", 0: "UNDEFINED"})
    
    # Add symbol column to trailing_df
    if not trailing_df.empty:
        trailing_df["symbol"] = symbol
    
    return trades, stats, signals_df, trailing_df


def run_backtest(
    symbol: str = "BTCUSDT",
    start_date: str = "2022-07-11",
    end_date: str = "2022-09-11",
    use_lib: bool = True,
    force_refetch: bool = False,
    market: str = "futures",
    all_symbols: bool = False,
    max_workers: int = 10,
) -> None:
    """
    Run backtest with automatic data fetching from Binance Public Data.
    
    Data is automatically fetched from data.binance.vision (with REST API fallback)
    and cached to kline_data/ folder for subsequent runs.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT") - ignored if all_symbols=True
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD" (inclusive)
        use_lib: If True, use 1m candles for Look-Inside-Bar trailing stop tracking.
                 If False, use signal-interval-only backtest.
        force_refetch: If True, re-fetch data even if cached CSV exists.
        market: "spot" or "futures" (UM perpetual). Default is "futures".
        all_symbols: If True, run backtest on all USDT perpetual contracts.
        max_workers: Number of parallel workers for concurrent processing (default: 10).
    """
    import time as time_module
    
    total_start = time_module.perf_counter()
    
    # Configuration
    cfg = BacktestConfig()
    signal_interval = cfg.signal_interval
    market_label = "Futures (Perpetual)" if market == "futures" else "Spot"
    
    # Determine symbols to backtest
    if all_symbols:
        print("=" * 60)
        print("Fetching all USDT perpetual symbols...")
        symbols = get_all_usdt_perpetuals_public()
        if not symbols:
            print("ERROR: Could not fetch symbol list.")
            return
        filtered_symbols = [sym for sym in symbols if is_tradeable_symbol(sym)]
        if len(filtered_symbols) != len(symbols):
            skipped = len(symbols) - len(filtered_symbols)
            print(f"Skipping {skipped} non-ASCII or non-alphanumeric symbols")
        symbols = filtered_symbols
        print(f"Found {len(symbols)} USDT perpetual contracts")
        print("=" * 60)
    else:
        if not is_tradeable_symbol(symbol):
            print(f"ERROR: Symbol not allowed (non-ASCII or non-alphanumeric): {symbol}")
            return
        symbols = [symbol]
    
    print("=" * 60)
    print(f"ATR Strategy Backtest")
    print(f"Symbols: {len(symbols)} {'(all USDT perps)' if all_symbols else ''}")
    print(f"Market: {market_label}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Signal Interval: {signal_interval}")
    print(f"LIB Mode: {'Enabled (1m resolution)' if use_lib else 'Disabled'}")
    max_pos_label = "UNLIMITED" if cfg.max_open_positions <= 0 else str(cfg.max_open_positions)
    print(f"Max Positions: {max_pos_label}")
    print(f"Workers: {max_workers if all_symbols and len(symbols) > 1 else 1}")
    print("=" * 60)
    
    # Run backtest for each symbol
    all_trades: List[pd.DataFrame] = []
    all_stats: List[Dict[str, Any]] = []
    all_signals: List[pd.DataFrame] = []
    all_trailing: List[pd.DataFrame] = []
    successful = 0
    failed = 0
    
    # Thread-safe counters and lock for progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list to allow mutation in nested function
    
    def process_symbol(sym: str) -> Tuple[str, pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame, Optional[str]]:
        """Process a single symbol and return results."""
        try:
            trades, stats, signals, trailing = run_backtest_single(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                use_lib=use_lib,
                force_refetch=force_refetch,
                market=market,
                cfg=cfg,
                verbose=False,  # Disable verbose for concurrent mode
            )
            return sym, trades, stats, signals, trailing, None
        except Exception as e:
            return sym, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), str(e)
    
    # Use concurrent processing for multiple symbols
    if all_symbols and len(symbols) > 1 and max_workers > 1:
        print(f"\nProcessing {len(symbols)} symbols concurrently...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(process_symbol, sym): sym for sym in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                sym, trades, stats, signals, trailing, error = future.result()
                
                with progress_lock:
                    completed_count[0] += 1
                    progress = completed_count[0]
                
                if error:
                    failed += 1
                    print(f"[{progress}/{len(symbols)}] {sym}: ERROR - {error}")
                else:
                    # Always collect signals even if no trades
                    if not signals.empty:
                        all_signals.append(signals)
                    
                    # Collect trailing stop updates
                    if not trailing.empty:
                        all_trailing.append(trailing)
                    
                    if not trades.empty:
                        all_trades.append(trades)
                        stats["symbol"] = sym
                        all_stats.append(stats)
                        successful += 1
                        pnl = stats.get('total_pnl_net', 0)
                        print(f"[{progress}/{len(symbols)}] {sym}: {len(trades)} trades, {len(signals)} signals, PnL: {pnl:.2f} USD")
                    else:
                        failed += 1
                        sig_count = len(signals) if not signals.empty else 0
                        print(f"[{progress}/{len(symbols)}] {sym}: No trades ({sig_count} signals)")
    else:
        # Sequential processing for single symbol or when workers=1
        for i, sym in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {sym}")
            try:
                trades, stats, signals, trailing = run_backtest_single(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    use_lib=use_lib,
                    force_refetch=force_refetch,
                    market=market,
                    cfg=cfg,
                    verbose=True,
                )
                # Always collect signals even if no trades
                if not signals.empty:
                    all_signals.append(signals)
                
                # Collect trailing stop updates
                if not trailing.empty:
                    all_trailing.append(trailing)
                
                if not trades.empty:
                    all_trades.append(trades)
                    stats["symbol"] = sym
                    all_stats.append(stats)
                    successful += 1
                    print(f"  -> {len(trades)} trades, {len(signals)} signals, PnL: {stats.get('total_pnl_net', 0):.2f} USD")
                else:
                    failed += 1
                    sig_count = len(signals) if not signals.empty else 0
                    print(f"  -> No trades ({sig_count} signals)")
            except Exception as e:
                failed += 1
                print(f"  -> ERROR: {e}")
    
    total_time = time_module.perf_counter() - total_start
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (AGGREGATED)")
    print("=" * 60)
    
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        
        # Sort by entry time to simulate actual execution order
        if "entry_time" in combined_trades.columns:
            combined_trades = combined_trades.sort_values("entry_time").reset_index(drop=True)
        
        # Calculate aggregate stats
        total_trades = len(combined_trades)
        total_pnl = combined_trades["pnl_net"].sum() if "pnl_net" in combined_trades.columns else 0
        wins = (combined_trades["pnl_net"] > 0).sum() if "pnl_net" in combined_trades.columns else 0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        print(f"  Symbols traded: {successful}")
        print(f"  Symbols skipped: {failed}")
        print(f"  Total trades: {total_trades}")
        print(f"  Total PnL: {total_pnl:.2f} USD")
        print(f"  Avg PnL per trade: {avg_pnl:.4f} USD")
        print(f"  Win rate: {win_rate:.2f}%")
        
        # Per-symbol summary
        print("\n" + "-" * 60)
        print("PER-SYMBOL SUMMARY (Top 10 by PnL):")
        symbol_pnl = combined_trades.groupby("symbol")["pnl_net"].agg(["sum", "count"])
        symbol_pnl.columns = ["total_pnl", "trades"]
        symbol_pnl = symbol_pnl.sort_values("total_pnl", ascending=False)
        
        for sym, row in symbol_pnl.head(10).iterrows():
            print(f"  {sym}: {row['total_pnl']:+.2f} USD ({int(row['trades'])} trades)")
        
        if len(symbol_pnl) > 10:
            print(f"  ... and {len(symbol_pnl) - 10} more symbols")
        
        print("\n" + "-" * 60)
        print("WORST 5 SYMBOLS:")
        for sym, row in symbol_pnl.tail(5).iterrows():
            print(f"  {sym}: {row['total_pnl']:+.2f} USD ({int(row['trades'])} trades)")
        
        # Ensure versioned output directories exist
        ensure_output_dirs()
        
        # Build filename suffix based on symbols and date range
        symbol_tag = "ALL" if all_symbols else symbol
        filename_suffix = f"{symbol_tag}_{start_date}_{end_date}"
        
        # Save trades to versioned directory
        trades_dir = get_backtest_trades_dir()
        trades_path = trades_dir / f"trades_{filename_suffix}.csv"
        combined_trades.to_csv(trades_path, index=False)
        
        print("\n" + "-" * 60)
        print(f"Output Files (v{STRATEGY_VERSION}):")
        print(f"  - {trades_path.relative_to(trades_path.parent.parent.parent.parent)} ({total_trades} trades)")
        
        # Save per-symbol stats
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_dir = get_backtest_stats_dir()
            stats_path = stats_dir / f"stats_by_symbol_{filename_suffix}.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"  - {stats_path.relative_to(stats_path.parent.parent.parent.parent)}")
        
        # Save trailing stop updates
        if all_trailing:
            combined_trailing = pd.concat(all_trailing, ignore_index=True)
            if "timestamp" in combined_trailing.columns:
                combined_trailing = combined_trailing.sort_values("timestamp").reset_index(drop=True)
            stats_dir = get_backtest_stats_dir()
            trailing_path = stats_dir / f"trailing_stops_{filename_suffix}.csv"
            combined_trailing.to_csv(trailing_path, index=False)
            print(f"  - {trailing_path.relative_to(trailing_path.parent.parent.parent.parent)} ({len(combined_trailing)} updates)")
        
        # Save config snapshot
        snapshot_path = save_config_snapshot(cfg, symbol_tag, start_date, end_date)
        print(f"  - {snapshot_path.relative_to(snapshot_path.parent.parent.parent)}")
    else:
        print("  No trades generated across all symbols.")
    
    # Signal summary and export
    if all_signals:
        combined_signals = pd.concat(all_signals, ignore_index=True)
        
        # Sort by timestamp
        if "timestamp" in combined_signals.columns:
            combined_signals = combined_signals.sort_values("timestamp").reset_index(drop=True)
        
        total_signals = len(combined_signals)
        long_signals = (combined_signals["signal"] == 1).sum()
        short_signals = (combined_signals["signal"] == -1).sum()
        
        print("\n" + "-" * 60)
        print("SIGNALS GENERATED:")
        print(f"  Total signals: {total_signals}")
        print(f"  LONG signals:  {long_signals}")
        print(f"  SHORT signals: {short_signals}")
        
        # Print signals to terminal (limit to 20 for readability)
        print("\n" + "-" * 60)
        display_limit = 20
        print(f"SIGNAL LIST {'(showing first ' + str(display_limit) + ')' if total_signals > display_limit else ''}:")
        print(f"  {'Timestamp':<22} {'Symbol':<12} {'Side':<6} {'Entry Price':<14} {'ATR':<12}")
        print(f"  {'-'*20} {'-'*10} {'-'*5} {'-'*13} {'-'*11}")
        
        for _, row in combined_signals.head(display_limit).iterrows():
            ts = str(row["timestamp"])[:19] if pd.notna(row["timestamp"]) else "N/A"
            sym = row.get("symbol", "N/A")
            side = row.get("side", "N/A")
            entry = row.get("entry_price", 0)
            atr = row.get("signal_atr", 0)
            entry_str = f"{entry:.6f}" if pd.notna(entry) else "MARKET"
            atr_str = f"{atr:.6f}" if pd.notna(atr) else "N/A"
            print(f"  {ts:<22} {sym:<12} {side:<6} {entry_str:<14} {atr_str:<12}")
        
        if total_signals > display_limit:
            print(f"  ... and {total_signals - display_limit} more signals")
        
        # Save signals to versioned directory
        signals_dir = get_backtest_signals_dir()
        signals_path = signals_dir / f"signals_{filename_suffix}.csv"
        combined_signals.to_csv(signals_path, index=False)
        print(f"\n  Saved to: {signals_path.relative_to(signals_path.parent.parent.parent.parent)} ({total_signals} signals)")
        
        # Compute and display signal performance stats
        if all_trades:
            combined_trades_for_stats = pd.concat(all_trades, ignore_index=True)
            stats_by_signal, stats_by_symbol_detailed = compute_signal_stats(combined_signals, combined_trades_for_stats)
            
            # Print signal stats
            print_signal_stats(stats_by_signal, stats_by_symbol_detailed)
            
            # Save signal stats to CSV
            if not stats_by_signal.empty:
                stats_dir = get_backtest_stats_dir()
                signal_stats_path = stats_dir / f"stats_by_signal_{filename_suffix}.csv"
                stats_by_signal.to_csv(signal_stats_path, index=False)
                print(f"\n  Signal stats saved to: {signal_stats_path.relative_to(signal_stats_path.parent.parent.parent.parent)}")
                if all_symbols:
                    agg_signal_stats_path = stats_dir / f"stats_by_signal_all_symbols_{start_date}_{end_date}.csv"
                    stats_by_signal.to_csv(agg_signal_stats_path, index=False)
                    print(f"  All-symbols signal stats saved to: {agg_signal_stats_path.relative_to(agg_signal_stats_path.parent.parent.parent.parent)}")
    else:
        print("\n" + "-" * 60)
        print("SIGNALS GENERATED:")
        print("  No signals generated.")
    
    # Timing summary
    print("\n" + "-" * 60)
    print("Timing:")
    print(f"  Total:               {total_time:>8.2f}s")
    avg_per_symbol = total_time / len(symbols) if symbols else 0
    print(f"  Avg per symbol:      {avg_per_symbol:>8.2f}s")
    if all_symbols and len(symbols) > 1 and max_workers > 1:
        print(f"  Effective speedup:   ~{max_workers}x (with {max_workers} workers)")
    print("=" * 60)


def run_stats() -> None:
    """
    Display live signal and trade statistics.
    Reads from the live signals and trades CSV files and computes performance metrics.
    """
    print("=" * 80)
    print(f"LIVE TRADING STATISTICS (Strategy v{STRATEGY_VERSION})")
    print("=" * 80)
    
    # Load live signals
    signals_path = get_live_signals_dir() / "signals.csv"
    trades_path = get_live_trades_dir() / "trades.csv"
    
    # Also check old paths for backwards compatibility
    old_signals_path = OUTPUT_DIR / "live_signals.csv"
    old_trades_path = OUTPUT_DIR / "live_trades.csv"
    
    signals_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    
    if signals_path.exists():
        signals_df = pd.read_csv(signals_path)
    elif old_signals_path.exists():
        signals_df = pd.read_csv(old_signals_path)
    
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
    elif old_trades_path.exists():
        trades_df = pd.read_csv(old_trades_path)
    
    print(f"\nSignals file: {signals_path if signals_path.exists() else old_signals_path}")
    print(f"Trades file: {trades_path if trades_path.exists() else old_trades_path}")
    
    if signals_df.empty and trades_df.empty:
        print("\nNo live signals or trades found.")
        return
    
    # Basic stats
    print(f"\nTotal signals logged: {len(signals_df)}")
    print(f"Total trades logged: {len(trades_df)}")
    
    if not signals_df.empty:
        if "status" in signals_df.columns:
            status_counts = signals_df["status"].value_counts()
            print("\nSignal Status Breakdown:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
    
    if not trades_df.empty:
        # Calculate trade stats
        total_pnl = trades_df["pnl_net"].sum() if "pnl_net" in trades_df.columns else 0
        wins = (trades_df["pnl_net"] > 0).sum() if "pnl_net" in trades_df.columns else 0
        losses = (trades_df["pnl_net"] <= 0).sum() if "pnl_net" in trades_df.columns else 0
        win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        print(f"\nTrade Summary:")
        print(f"  Total PnL: {total_pnl:+.2f} USD")
        print(f"  Wins: {wins}, Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        # Compute detailed stats
        stats_by_signal, stats_by_symbol = compute_signal_stats(signals_df, trades_df)
        print_signal_stats(stats_by_signal, stats_by_symbol)
    
    print("\n" + "=" * 80)


def run_compare(v1: str, v2: str, symbol: str = "ALL") -> None:
    """
    Compare backtest results between two strategy versions.
    
    Args:
        v1: First version string (e.g., "1.0.0")
        v2: Second version string (e.g., "1.1.0")
        symbol: Symbol to compare (default: ALL)
    """
    print("=" * 80)
    print(f"VERSION COMPARISON: v{v1} vs v{v2}")
    print("=" * 80)
    
    # Find stats files for each version
    v1_dir = OUTPUT_DIR / "backtest" / f"v{v1}"
    v2_dir = OUTPUT_DIR / "backtest" / f"v{v2}"
    
    if not v1_dir.exists():
        print(f"ERROR: Version {v1} directory not found: {v1_dir}")
        return
    if not v2_dir.exists():
        print(f"ERROR: Version {v2} directory not found: {v2_dir}")
        return
    
    # Find the most recent stats file for each version
    def find_latest_stats(version_dir: Path, stat_type: str) -> Optional[Path]:
        stats_dir = version_dir / "stats"
        if not stats_dir.exists():
            return None
        files = list(stats_dir.glob(f"{stat_type}_*.csv"))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    
    v1_signal_stats = find_latest_stats(v1_dir, "stats_by_signal")
    v2_signal_stats = find_latest_stats(v2_dir, "stats_by_signal")
    
    v1_trades = None
    v2_trades = None
    trades_dir_v1 = v1_dir / "trades"
    trades_dir_v2 = v2_dir / "trades"
    
    if trades_dir_v1.exists():
        trade_files_v1 = list(trades_dir_v1.glob("trades_*.csv"))
        if trade_files_v1:
            v1_trades = max(trade_files_v1, key=lambda f: f.stat().st_mtime)
    
    if trades_dir_v2.exists():
        trade_files_v2 = list(trades_dir_v2.glob("trades_*.csv"))
        if trade_files_v2:
            v2_trades = max(trade_files_v2, key=lambda f: f.stat().st_mtime)
    
    # Load and compare trades
    if v1_trades and v2_trades:
        df_v1 = pd.read_csv(v1_trades)
        df_v2 = pd.read_csv(v2_trades)
        
        total_v1 = len(df_v1)
        total_v2 = len(df_v2)
        pnl_v1 = df_v1["pnl_net"].sum() if "pnl_net" in df_v1.columns else 0
        pnl_v2 = df_v2["pnl_net"].sum() if "pnl_net" in df_v2.columns else 0
        wr_v1 = (df_v1["pnl_net"] > 0).sum() / total_v1 * 100 if total_v1 > 0 else 0
        wr_v2 = (df_v2["pnl_net"] > 0).sum() / total_v2 * 100 if total_v2 > 0 else 0
        
        print(f"\n{'Metric':<20} {'v' + v1:>15} {'v' + v2:>15} {'Change':>15}")
        print(f"{'-'*18} {'-'*15} {'-'*15} {'-'*15}")
        print(f"{'Total Trades':<20} {total_v1:>15} {total_v2:>15} {total_v2 - total_v1:>+15}")
        print(f"{'Total PnL':<20} {pnl_v1:>+14.2f} {pnl_v2:>+14.2f} {pnl_v2 - pnl_v1:>+14.2f}")
        print(f"{'Win Rate':<20} {wr_v1:>14.1f}% {wr_v2:>14.1f}% {wr_v2 - wr_v1:>+14.1f}pp")
        
        # Compare by signal type if available
        if v1_signal_stats and v2_signal_stats:
            stats_v1 = pd.read_csv(v1_signal_stats)
            stats_v2 = pd.read_csv(v2_signal_stats)
            
            if not stats_v1.empty and not stats_v2.empty:
                print(f"\n" + "-" * 80)
                print("PER-SIGNAL-TYPE COMPARISON:")
                print("-" * 80)
                
                # Merge on signal_reason
                merged = stats_v1.merge(
                    stats_v2, 
                    on="signal_reason", 
                    how="outer", 
                    suffixes=("_v1", "_v2")
                ).fillna(0)
                
                print(f"{'Signal Reason':<25} {'v' + v1 + ' Win%':>12} {'v' + v2 + ' Win%':>12} {'Change':>10}")
                print(f"{'-'*23} {'-'*12} {'-'*12} {'-'*10}")
                
                for _, row in merged.iterrows():
                    reason = str(row.get("signal_reason", "unknown"))[:23]
                    wr1 = row.get("win_rate_v1", 0) * 100
                    wr2 = row.get("win_rate_v2", 0) * 100
                    change = wr2 - wr1
                    marker = " *" if abs(change) > 5 else ""
                    print(f"{reason:<25} {wr1:>11.1f}% {wr2:>11.1f}% {change:>+9.1f}pp{marker}")
    else:
        print("\nCould not find trade files for comparison.")
        if not v1_trades:
            print(f"  Missing: v{v1} trades")
        if not v2_trades:
            print(f"  Missing: v{v2} trades")
    
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATR Strategy - Backtest & Live Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest with default dates (uses futures/perpetual data)
  python main.py --mode backtest
  
  # Run backtest with custom date range
  python main.py --mode backtest --start 2022-06-20 --end 2022-09-26
  
  # Run backtest with different symbol
  python main.py --mode backtest --symbol ETHUSDT --start 2023-01-01 --end 2023-03-31
  
  # Run backtest on ALL USDT perpetual contracts (like live trading)
  python main.py --mode backtest --all-symbols --start 2025-10-20 --end 2026-01-20
  
  # Run backtest on all symbols with 20 parallel workers (faster)
  python main.py --mode backtest --all-symbols --start 2025-10-20 --end 2026-01-20 --workers 20
  
  # Force re-fetch data (ignore cache)
  python main.py --mode backtest --start 2022-07-01 --end 2022-07-31 --force-refetch
  
  # Run without Look-Inside-Bar mode (faster, less accurate)
  python main.py --mode backtest --no-lib
  
  # Run backtest with spot data instead of futures
  python main.py --mode backtest --market spot
  
  # Run live trading
  python main.py --mode live
  
  # View live signal/trade statistics
  python main.py --mode stats
  
  # Compare backtest results between two strategy versions
  python main.py --mode compare --v1 1.0.0 --v2 1.1.0
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "stats", "compare"],
        default="backtest",
        help="Run mode: 'backtest', 'live', 'stats', or 'compare' (default: backtest)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT). Ignored if --all-symbols is set.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Run backtest on all USDT perpetual contracts (like live trading)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-07-11",
        help="Start date YYYY-MM-DD (default: 2022-07-11)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2022-09-11",
        help="End date YYYY-MM-DD (default: 2022-09-11)",
    )
    parser.add_argument(
        "--no-lib",
        action="store_true",
        help="Disable Look-Inside-Bar mode (use signal-interval only backtest)",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Force re-fetch data from Binance (ignore cached CSV)",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["spot", "futures"],
        default="futures",
        help="Market type: 'spot' or 'futures' (UM perpetual). Default: futures",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers for concurrent processing (default: 10)",
    )
    parser.add_argument(
        "--v1",
        type=str,
        default=None,
        help="First version for comparison (e.g., '1.0.0'). Used with --mode compare",
    )
    parser.add_argument(
        "--v2",
        type=str,
        default=None,
        help="Second version for comparison (e.g., '1.1.0'). Used with --mode compare",
    )
    args = parser.parse_args()

    if args.mode == "live":
        live_cfg = LiveConfig()
        run_live(live_cfg)
    elif args.mode == "stats":
        run_stats()
    elif args.mode == "compare":
        if not args.v1 or not args.v2:
            print("ERROR: --v1 and --v2 are required for compare mode")
            print("Example: python main.py --mode compare --v1 1.0.0 --v2 1.1.0")
            return
        run_compare(args.v1, args.v2, args.symbol)
    else:
        run_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            use_lib=not args.no_lib,
            force_refetch=args.force_refetch,
            market=args.market,
            all_symbols=args.all_symbols,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
