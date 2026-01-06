"""
ATR-based 1-minute grinder (body trigger, ATR-based TP/SL)

Rules:
- Compute ATR (EMA) on 1m candles.
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

    df["signal"], df["signal_atr"] = build_signals_body_opposite(
        df, df["atr"],
        thr1=cfg.thr1, thr2=cfg.thr2, tolerance_pct=cfg.signal_atr_tolerance_pct,
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
                    sl_atr_mult=cfg.sl_atr_mult,
                    margin_cap=cfg.initial_capital,
                    max_leverage=cfg.leverage,
                    min_leverage=cfg.min_leverage,
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
    ) -> None:
        self.key = key
        self.secret = secret
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.recv_window_ms = int(recv_window_ms)
        self.time_sync_interval_seconds = float(time_sync_interval_seconds)
        self.time_offset_ms = 0
        self._last_time_sync_monotonic: Optional[float] = None
        self._last_time_sync_attempt_monotonic: Optional[float] = None

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
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
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
    )
    try:
        client.sync_time()
    except Exception as exc:
        logging.warning("Binance time sync failed; signed requests may fail (-1021). error=%s", exc)
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

    while True:
        try:
            return client.new_order(**params)
        except ClientError as error:
            error_code = getattr(error, "error_code", None)
            status_code = getattr(error, "status_code", None)
            error_message = getattr(error, "error_message", None)

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
    working_type: str = "CONTRACT_PRICE",
    price_protect: bool = False,
    reduce_only: bool = True,
) -> Optional[Dict[str, object]]:
    base_params: Dict[str, object] = {
        "symbol": symbol,
        "side": side,
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
            try:
                return client.new_algo_order(**params)
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
        return client.new_algo_order(**base_params)
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


def compute_live_signal(df: pd.DataFrame, cfg: LiveConfig) -> Tuple[int, Optional[float], Optional[float]]:
    warmup = cfg.atr_len if cfg.atr_warmup_bars is None else cfg.atr_warmup_bars
    if len(df) <= warmup:
        return 0, None, None

    atr = atr_ema(df, cfg.atr_len)
    signal, signal_atr = build_signals_body_opposite(
        df,
        atr,
        thr1=cfg.thr1,
        thr2=cfg.thr2,
        tolerance_pct=cfg.signal_atr_tolerance_pct,
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
    interval_ms = interval_to_ms(cfg.interval)
    last_close_ms_by_symbol: Dict[str, Optional[int]] = {sym: None for sym in symbols}
    last_atr_by_symbol: Dict[str, Optional[float]] = {sym: None for sym in symbols}
    log_event(cfg.log_path, {
        "event": "startup",
        "symbols": symbols,
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
            klines = client.klines(symbol=symbol, interval=cfg.interval, limit=history)
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
        signal, signal_atr, atr_value = compute_live_signal(df, cfg)
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

        offset = atr_value * cfg.atr_offset_mult
        if signal == 1:
            limit_price = bid - offset
            side = "BUY"
        else:
            limit_price = ask + offset
            side = "SELL"

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
                    tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                    tp_price_str = format_to_step(tp_price, filters["tick_size"])
                    sl_price_str = format_to_step(sl_price, filters["tick_size"])
                    qty_str = format_to_step(qty, filters["step_size"])

                    tp_side = "SELL" if side == 1 else "BUY"
                    tp_client_id = f"ATR_TP_{entry_close_ms or ''}"
                    # Place TP as a resting limit to avoid taker fees when possible.
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
                    sl_algo_order = algo_stop_limit_order(
                        symbol=active_symbol,
                        side=sl_side,
                        quantity=qty_str,
                        trigger_price=sl_price_str,
                        price=sl_price_str,
                        client=client,
                        client_order_id=sl_client_id,
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
                        state.active_symbol = None

            if active_symbol and not in_position and state.had_position:
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

                        if state.sl_order_id is not None and state.sl_order_time is not None:
                            status = str((sl_order or {}).get("status") or "").upper()
                            if status != "FILLED" and (time.time() - state.sl_order_time) >= 3.0:
                                if cancel_order_safely(client, active_symbol, order_id=state.sl_order_id):
                                    state.sl_order_id = None
                                    state.sl_client_order_id = None
                                    state.sl_order_time = None
                                    sl_order = None

                        if state.sl_order_id is None:
                            close_side = "SELL" if position_amt > 0 else "BUY"
                            close_qty = abs(position_amt)
                            close_qty_str = format_to_step(close_qty, filters["step_size"])
                            try:
                                close_qty_num = float(close_qty_str)
                            except (TypeError, ValueError):
                                close_qty_num = 0.0
                            if close_qty_num >= filters["min_qty"] and bid > 0 and ask > 0:
                                close_price = ask if close_side == "SELL" else bid
                                close_price_str = format_price_to_tick(close_price, tick_size, side=close_side, post_only=True)
                                sl_client_id = f"ATR_SL_EXIT_{int(time.time())}"
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
                                state.sl_order_id = new_sl_order.get("orderId") if new_sl_order else None
                                state.sl_client_order_id = (
                                    (new_sl_order.get("clientOrderId") if new_sl_order else None)
                                    or (sl_client_id if new_sl_order else None)
                                )
                                state.sl_order_time = time.time() if new_sl_order else None
                                log_event(cfg.log_path, {
                                    "event": "sl_exit_order",
                                    "symbol": active_symbol,
                                    "side": close_side,
                                    "quantity": format_float_2(close_qty_num),
                                    "price": format_float_by_symbol(close_price, active_symbol),
                                    "order_id": state.sl_order_id,
                                    "client_order_id": state.sl_client_order_id,
                                })
                            else:
                                log_event(cfg.log_path, {
                                    "event": "sl_exit_skip_qty",
                                    "symbol": active_symbol,
                                    "position_amt": format_float_2(position_amt),
                                    "quantity": close_qty_str,
                                })

            # If in position and no exits placed, place them using current entryPrice
            if in_position and active_symbol and (state.tp_order_id is None or state.sl_algo_id is None or state.sl_price is None):
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

                tp_price, sl_price = compute_tp_sl_prices(entry_price, side, entry_atr, cfg.tp_atr_mult, cfg.sl_atr_mult)
                tp_price_str = format_to_step(tp_price, filters["tick_size"])
                sl_price_str = format_to_step(sl_price, filters["tick_size"])
                qty_str = format_to_step(qty, filters["step_size"])
                tp_side = "SELL" if side == 1 else "BUY"

                if state.sl_price is None:
                    state.sl_price = sl_price

                if state.sl_algo_id is None and not state.sl_triggered:
                    sl_side = "SELL" if side == 1 else "BUY"
                    sl_client_id = f"ATR_SL_RECOVER_{int(time.time())}"
                    sl_algo_order = algo_stop_limit_order(
                        symbol=active_symbol,
                        side=sl_side,
                        quantity=qty_str,
                        trigger_price=sl_price_str,
                        price=sl_price_str,
                        client=client,
                        client_order_id=sl_client_id,
                        working_type=cfg.algo_working_type,
                        price_protect=cfg.algo_price_protect,
                        reduce_only=True,
                    )
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
def run_backtest() -> None:
    # 1) Get data
    # Option A: Use Binance via ccxt (recommended)
    df = fetch_ohlcv_binance(
        symbol="BTC/USDT",
        timeframe="1m",
        start_utc="2025-06-30 00:00:00",
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
        leverage=50.0,
        initial_capital=100.0,
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
