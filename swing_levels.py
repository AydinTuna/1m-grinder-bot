"""
Swing high/low detection for 1m + 15m candles.

Definition (pivot / fractal-style):
  - Swing High at index i if high[i] is strictly greater than the highs of the
    previous `left` candles and the next `right` candles.
  - Swing Low at index i if low[i] is strictly lower than the lows of the
    previous `left` candles and the next `right` candles.

Because swing points require `right` future candles, each point also includes a
`confirm_ts` (when it becomes known in real-time).

Example:
  python3 swing_levels.py --csv backtest_series.csv --output swing_levels.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


SwingKind = Literal["swing_high", "swing_low"]
StructureKind = Literal["HH", "HL", "LL", "LH"]


@dataclass(frozen=True)
class SwingPoint:
    kind: SwingKind
    timeframe: str
    pivot_ts: str
    confirm_ts: str
    level: float
    bar_index: int
    open: float
    high: float
    low: float
    close: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "timeframe": self.timeframe,
            "pivot_ts": self.pivot_ts,
            "confirm_ts": self.confirm_ts,
            "level": self.level,
            "bar_index": self.bar_index,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }


@dataclass(frozen=True)
class MarketStructurePoint:
    """
    Represents a classified market structure point (HH, HL, LL, LH).
    
    Market Structure Rules:
    - HH (Higher High): current swing high's close > previous swing high's high
    - LH (Lower High): current swing high's close <= previous swing high's high
    - LL (Lower Low): current swing low's close < previous swing low's low
    - HL (Higher Low): current swing low's close >= previous swing low's low
    """
    structure_kind: StructureKind  # "HH", "HL", "LL", "LH"
    swing_point: SwingPoint  # The underlying swing point
    reference_level: float  # The previous swing's high (for HH/LH) or low (for LL/HL)
    effective_close: float  # The close used for comparison (may be from next candle if liquidity sweep)
    is_liquidity_sweep: bool  # True if the swing candle was a liquidity sweep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure_kind": self.structure_kind,
            "swing_point": self.swing_point.to_dict(),
            "reference_level": float(self.reference_level),
            "effective_close": float(self.effective_close),
            "is_liquidity_sweep": bool(self.is_liquidity_sweep),  # Convert numpy.bool_ to Python bool
        }


def is_liquidity_sweep(
    candle_open: float,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    atr_value: float,
) -> bool:
    """
    Detect if a candle is likely a liquidity sweep.
    
    Conditions:
    - Wick > ATR * 0.60 (large wick relative to ATR)
    
    Args:
        candle_open, candle_high, candle_low, candle_close: OHLC values
        atr_value: ATR value at this candle
    
    Returns:
        True if the candle appears to be a liquidity sweep
    """
    if atr_value is None or np.isnan(atr_value) or atr_value <= 0:
        return False
    
    body = abs(candle_close - candle_open)
    
    # Calculate wicks
    if candle_close >= candle_open:  # Green candle
        upper_wick = candle_high - candle_close
        lower_wick = candle_open - candle_low
    else:  # Red candle
        upper_wick = candle_high - candle_open
        lower_wick = candle_close - candle_low
    
    max_wick = max(upper_wick, lower_wick)
    
    # Liquidity sweep: wick larger than ATR * 0.60
    return max_wick > atr_value * 0.60


def classify_market_structure(
    swings: List[SwingPoint],
    df: pd.DataFrame,
    atr: pd.Series,
) -> List[MarketStructurePoint]:
    """
    Classify swing points into HH/HL/LL/LH market structure levels.
    
    Market Structure Rules:
    - HH (Higher High): current swing high's close > previous swing high's high
    - LH (Lower High): current swing high's close <= previous swing high's high
    - LL (Lower Low): current swing low's close < previous swing low's low
    - HL (Higher Low): current swing low's close >= previous swing low's low
    
    Liquidity Sweep Filter:
    - If wick > ATR * 0.60, it's likely a liquidity sweep
    - In this case, use the next candle's close instead of the swing candle's close
    
    Args:
        swings: List of detected swing points
        df: DataFrame with OHLC data
        atr: ATR series aligned with df
    
    Returns:
        List of MarketStructurePoint with HH/HL/LL/LH classification
    """
    if not swings:
        return []
    
    structure_points: List[MarketStructurePoint] = []
    
    # Track the last swing high and low for comparison
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    
    for swing in swings:
        bar_idx = swing.bar_index
        
        # Get ATR value at this bar
        atr_val = atr.iloc[bar_idx] if bar_idx < len(atr) else None
        
        # Check if this candle is a liquidity sweep
        sweep = is_liquidity_sweep(
            swing.open, swing.high, swing.low, swing.close, atr_val
        )
        
        # Determine effective close for comparison
        effective_close = swing.close
        
        if sweep:
            # Use next candle's close if available
            next_idx = bar_idx + 1
            if next_idx < len(df):
                next_candle = df.iloc[next_idx]
                effective_close = float(next_candle["close"])
        
        if swing.kind == "swing_high":
            if last_swing_high is not None:
                # Compare: current effective close vs previous swing high's HIGH (wick)
                ref_level = last_swing_high.high
                
                if effective_close > ref_level:
                    kind: StructureKind = "HH"
                else:
                    kind = "LH"
                
                structure_points.append(MarketStructurePoint(
                    structure_kind=kind,
                    swing_point=swing,
                    reference_level=ref_level,
                    effective_close=effective_close,
                    is_liquidity_sweep=sweep,
                ))
            
            # Update last swing high
            last_swing_high = swing
            
        elif swing.kind == "swing_low":
            if last_swing_low is not None:
                # Compare: current effective close vs previous swing low's LOW (wick)
                ref_level = last_swing_low.low
                
                if effective_close < ref_level:
                    kind = "LL"
                else:
                    kind = "HL"
                
                structure_points.append(MarketStructurePoint(
                    structure_kind=kind,
                    swing_point=swing,
                    reference_level=ref_level,
                    effective_close=effective_close,
                    is_liquidity_sweep=sweep,
                ))
            
            # Update last swing low
            last_swing_low = swing
    
    return structure_points


def compute_market_structure_levels(
    df: pd.DataFrame,
    atr: pd.Series,
    *,
    swing_timeframe: str = "1d",
    left: int = 2,
    right: int = 2,
    resample_rule: str = "1d",
) -> Tuple[List[MarketStructurePoint], pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detect swing points and classify them into market structure (HH/HL/LL/LH).
    
    Returns:
        structure_points: List of MarketStructurePoint
        hh_series: Series with HH levels (forward-filled)
        hl_series: Series with HL levels (forward-filled)
        ll_series: Series with LL levels (forward-filled)
        lh_series: Series with LH levels (forward-filled)
    """
    if df.empty:
        empty = pd.Series(np.nan, index=df.index, dtype=float)
        return [], empty.copy(), empty.copy(), empty.copy(), empty.copy()
    
    # Resample if needed
    if swing_timeframe == "1m" or swing_timeframe == df.index.freq:
        df_tf = df
    else:
        df_tf = resample_ohlcv(df, rule=resample_rule)
    
    # Resample ATR to match timeframe
    if len(df_tf) != len(df):
        # Need to align ATR with resampled df
        atr_tf = atr.resample(resample_rule).last().reindex(df_tf.index, method="ffill")
    else:
        atr_tf = atr
    
    # Detect raw swing points
    swings = detect_swings(df_tf, timeframe=swing_timeframe, left=left, right=right)
    
    # Classify into market structure
    structure_points = classify_market_structure(swings, df_tf, atr_tf)
    
    # Build forward-filled series for each structure type
    def levels_for_structure(kind: StructureKind) -> pd.Series:
        pts = [s for s in structure_points if s.structure_kind == kind]
        if not pts:
            return pd.Series(np.nan, index=df.index, dtype=float)
        
        # Use confirm_ts from the underlying swing point
        confirm_ts = pd.to_datetime([p.swing_point.confirm_ts for p in pts], utc=True)
        # Use high for swing_high (HH/LH), low for swing_low (HL/LL)
        if kind in ("HH", "LH"):
            levels = [float(p.swing_point.high) for p in pts]
        else:  # HL, LL
            levels = [float(p.swing_point.low) for p in pts]
        
        series = pd.Series(levels, index=confirm_ts, dtype=float).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.reindex(df.index, method="ffill")
    
    hh_series = levels_for_structure("HH")
    hl_series = levels_for_structure("HL")
    ll_series = levels_for_structure("LL")
    lh_series = levels_for_structure("LH")
    
    return structure_points, hh_series, hl_series, ll_series, lh_series


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_iso_z(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def load_ohlcv_csv(path: str, dt_col: str = "dt") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    dt_col = dt_col.strip().lower()

    if dt_col not in df.columns:
        raise ValueError(f"CSV missing datetime column '{dt_col}'. Columns: {list(df.columns)}")

    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Failed parsing {bad} datetime values from column '{dt_col}'.")

    df = df.drop(columns=[dt_col])
    df.index = pd.DatetimeIndex(dt, name="dt")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing OHLC columns: {missing}. Columns: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _pivot_flags(values: np.ndarray, left: int, right: int, kind: Literal["high", "low"]) -> np.ndarray:
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    n = int(values.shape[0])
    window = left + right + 1
    flags = np.zeros(n, dtype=bool)
    if n < window:
        return flags

    windows = sliding_window_view(values, window_shape=window)  # (n-window+1, window)
    center = windows[:, left]

    left_side = windows[:, :left]
    right_side = windows[:, left + 1 :]
    if kind == "high":
        core = (center > left_side.max(axis=1)) & (center > right_side.max(axis=1))
    else:
        core = (center < left_side.min(axis=1)) & (center < right_side.min(axis=1))
    flags[left : n - right] = core
    return flags


def detect_swings(
    df: pd.DataFrame,
    timeframe: str,
    left: int = 2,
    right: int = 2,
) -> List[SwingPoint]:
    if df.empty:
        return []

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    highs = df["high"].to_numpy(dtype=float, copy=False)
    lows = df["low"].to_numpy(dtype=float, copy=False)

    pivot_high = _pivot_flags(highs, left=left, right=right, kind="high")
    pivot_low = _pivot_flags(lows, left=left, right=right, kind="low")

    swings: List[SwingPoint] = []
    idx = df.index
    for i in range(left, len(df) - right):
        if not (pivot_high[i] or pivot_low[i]):
            continue

        pivot_ts = _to_iso_z(pd.Timestamp(idx[i]))
        confirm_ts = _to_iso_z(pd.Timestamp(idx[i + right]))
        row = df.iloc[i]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        if pivot_high[i]:
            swings.append(
                SwingPoint(
                    kind="swing_high",
                    timeframe=timeframe,
                    pivot_ts=pivot_ts,
                    confirm_ts=confirm_ts,
                    level=h,
                    bar_index=i,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                )
            )
        if pivot_low[i]:
            swings.append(
                SwingPoint(
                    kind="swing_low",
                    timeframe=timeframe,
                    pivot_ts=pivot_ts,
                    confirm_ts=confirm_ts,
                    level=l,
                    bar_index=i,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                )
            )

    swings.sort(key=lambda s: (s.pivot_ts, 0 if s.kind == "swing_low" else 1))
    return swings


def compute_confirmed_swing_levels(
    df_1m: pd.DataFrame,
    *,
    swing_timeframe: str = "15m",
    left: int = 2,
    right: int = 2,
    resample_rule: str = "15min",
) -> Tuple[pd.Series, pd.Series]:
    """
    Build the latest confirmed swing high/low levels *as-of each bar* in df_1m.

    Notes:
      - Swings are detected on the requested timeframe (default: 15m resample),
        but returned as 1m-aligned series (forward-filled from confirm time).
      - Values are NaN until a swing of that kind is confirmed.
    """
    if df_1m.empty:
        empty = pd.Series(np.nan, index=df_1m.index, dtype=float)
        return empty, empty

    if swing_timeframe == "1m":
        df_tf = df_1m
        tf = "1m"
    else:
        df_tf = resample_ohlcv(df_1m, rule=resample_rule)
        tf = swing_timeframe

    swings = detect_swings(df_tf, timeframe=tf, left=left, right=right)

    def levels_for(kind: SwingKind) -> pd.Series:
        pts = [s for s in swings if s.kind == kind]
        if not pts:
            return pd.Series(np.nan, index=df_1m.index, dtype=float)
        confirm_ts = pd.to_datetime([p.confirm_ts for p in pts], utc=True)
        levels = [float(p.level) for p in pts]
        series = pd.Series(levels, index=confirm_ts, dtype=float).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.reindex(df_1m.index, method="ffill")

    return levels_for("swing_high"), levels_for("swing_low")


def build_swing_atr_signals(
    df: pd.DataFrame,
    atr: pd.Series,
    swing_high_level: pd.Series,
    swing_low_level: pd.Series,
    *,
    body_atr_mult: float = 2.0,
    swing_proximity_atr_mult: float = 0.25,
    tolerance_pct: float = 0.0,
    volume: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Strategy rules (signal generated on candle close, entry after close):

    If price is at/near swing high:
      1) Red candle body >= body_atr_mult*ATR => SHORT
      2) Green candle body >= body_atr_mult*ATR and close < swing high => SHORT
      3) Green candle body >= body_atr_mult*ATR and close >= swing high => LONG (limit at mid-body)

    If price is at/near swing low:
      1) Green candle body >= body_atr_mult*ATR => LONG
      2) Red candle body >= body_atr_mult*ATR and close > swing low => LONG
      3) Red candle body >= body_atr_mult*ATR and close <= swing low => SHORT (limit at mid-body)

    Otherwise:
      - If candle body >= body_atr_mult*ATR => enter in same direction as candle

    Trend continuation override:
      If previous candle signaled a direction and current candle would flip:
        - If current body > prev body AND volume > prev volume AND same candle direction
        - Then continue the previous signal direction instead of flipping

    Returns:
      signal_dir: +1 long, -1 short, 0 none
      signal_atr: ATR on the signal candle (NaN if none)
      entry_price: NaN for market-on-next-bar, otherwise limit entry price
      signal_reason: string describing why signal was generated (empty if no signal)
    """
    if df.empty:
        empty_i = pd.Series(0, index=df.index, dtype=int)
        empty_f = pd.Series(np.nan, index=df.index, dtype=float)
        empty_s = pd.Series("", index=df.index, dtype=str)
        return empty_i, empty_f, empty_f, empty_s

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    body = (df["close"] - df["open"]).abs()
    delta = df["close"] - df["open"]
    candle_dir = np.sign(delta).astype(int)  # -1,0,+1
    is_green = candle_dir == 1
    is_red = candle_dir == -1

    tol = max(0.0, float(tolerance_pct or 0.0))
    target = float(body_atr_mult) * atr
    if tol > 0.0:
        target = target * (1.0 - tol)
    big_body = body >= target

    proximity = atr * float(swing_proximity_atr_mult)
    has_prox = proximity.notna()

    near_high = (
        big_body
        & has_prox
        & swing_high_level.notna()
        & (df["high"] >= (swing_high_level - proximity))
        & (df["low"] <= (swing_high_level + proximity))
    )
    near_low = (
        big_body
        & has_prox
        & swing_low_level.notna()
        & (df["low"] <= (swing_low_level + proximity))
        & (df["high"] >= (swing_low_level - proximity))
    )

    both = near_high & near_low
    if bool(both.any()):
        dist_high = (df["close"] - swing_high_level).abs()
        dist_low = (df["close"] - swing_low_level).abs()
        pick_high = dist_high <= dist_low
        near_high = near_high & (~both | pick_high)
        near_low = near_low & (~both | ~pick_high)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal_atr = pd.Series(np.nan, index=df.index, dtype=float)
    entry_price = pd.Series(np.nan, index=df.index, dtype=float)
    signal_reason = pd.Series("", index=df.index, dtype=str)

    mid_body = (df["open"] + df["close"]) / 2.0

    # Neither swing high nor swing low => same direction as candle (if big body)
    other = big_body & ~(near_high | near_low)
    signal[other] = candle_dir[other]
    # Set reason for momentum signals
    momentum_long = other & is_green
    momentum_short = other & is_red
    signal_reason[momentum_long] = "momentum_long"
    signal_reason[momentum_short] = "momentum_short"

    # Near swing high rules
    cond_high = near_high
    short_red = cond_high & is_red
    short_failed = cond_high & is_green & (df["close"] < swing_high_level)
    long_break = cond_high & is_green & (df["close"] >= swing_high_level)

    signal[short_red | short_failed] = -1
    signal[long_break] = 1
    entry_price[long_break] = mid_body[long_break]
    # Set reasons for swing high signals
    signal_reason[short_red] = "swing_high_rejection_short"
    signal_reason[short_failed] = "swing_high_failed_breakout_short"
    signal_reason[long_break] = "swing_high_breakout_long"

    # Near swing low rules
    cond_low = near_low
    long_green = cond_low & is_green
    long_failed = cond_low & is_red & (df["close"] > swing_low_level)
    short_break = cond_low & is_red & (df["close"] <= swing_low_level)

    signal[long_green | long_failed] = 1
    signal[short_break] = -1
    entry_price[short_break] = mid_body[short_break]
    # Set reasons for swing low signals
    signal_reason[long_green] = "swing_low_bounce_long"
    signal_reason[long_failed] = "swing_low_failed_breakdown_long"
    signal_reason[short_break] = "swing_low_breakdown_short"

    # Trend continuation override:
    # If previous candle signaled, and current candle shows momentum in same direction:
    # - body > prev_body
    # - volume > prev_volume (if available)
    # - candle direction matches previous signal (green for BUY, red for SELL)
    # Then continue the previous signal instead of flipping
    prev_signal = signal.shift(1).fillna(0).astype(int)
    prev_body = body.shift(1)

    # Conditions for continuation
    body_bigger = body > prev_body
    if volume is not None:
        prev_volume = volume.shift(1)
        vol_higher = volume > prev_volume
    else:
        vol_higher = True  # No volume data, skip volume check

    # Candle direction must match previous signal: green(+1) matches BUY(+1), red(-1) matches SELL(-1)
    dir_matches_prev_signal = (candle_dir == prev_signal)

    # Only apply continuation if we would otherwise flip (current signal differs from prev signal)
    would_flip = (signal != 0) & (prev_signal != 0) & (signal != prev_signal)

    # Continuation condition: would flip, but momentum continues (big body, body bigger, vol higher, same dir)
    continue_trend = would_flip & big_body & body_bigger & vol_higher & dir_matches_prev_signal

    # Override: continue the previous signal direction
    signal = signal.where(~continue_trend, prev_signal)

    # Also clear entry_price for continued signals (use market entry, not limit)
    entry_price = entry_price.where(~continue_trend, np.nan)
    
    # Set reason for trend continuation
    continue_long = continue_trend & (prev_signal == 1)
    continue_short = continue_trend & (prev_signal == -1)
    signal_reason[continue_long] = "trend_continuation_long"
    signal_reason[continue_short] = "trend_continuation_short"

    active = signal != 0
    signal_atr[active] = atr[active]

    return signal, signal_atr, entry_price, signal_reason


def build_market_structure_signals(
    df: pd.DataFrame,
    atr: pd.Series,
    structure_points: List[MarketStructurePoint],
    *,
    body_atr_mult: float = 2.0,
    structure_proximity_atr_mult: float = 0.5,
    tolerance_pct: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate trading signals based on market structure (HH/HL/LL/LH).
    
    Strategy Logic:
    
    TREND DETECTION:
    - Uptrend: Most recent structure is HH or HL
    - Downtrend: Most recent structure is LL or LH
    
    SIGNAL RULES:
    
    In UPTREND (bullish bias):
    - Near HL level + green candle with big body → LONG (trend continuation)
    - Near HH level + green candle closing above HH → LONG (breakout)
    - Near HH level + red candle with big body → Potential reversal, no signal
    
    In DOWNTREND (bearish bias):
    - Near LH level + red candle with big body → SHORT (trend continuation)
    - Near LL level + red candle closing below LL → SHORT (breakdown)
    - Near LL level + green candle with big body → Potential reversal, no signal
    
    BREAK OF STRUCTURE (BOS) SIGNALS:
    - In uptrend, if price closes below recent HL → SHORT (BOS)
    - In downtrend, if price closes above recent LH → LONG (BOS)
    
    Returns:
        signal_dir: +1 long, -1 short, 0 none
        signal_atr: ATR on the signal candle (NaN if none)
        entry_price: NaN for market-on-next-bar, otherwise limit entry price
        signal_reason: string describing why signal was generated
        trend: +1 uptrend, -1 downtrend, 0 undefined
    """
    if df.empty:
        empty_i = pd.Series(0, index=df.index, dtype=int)
        empty_f = pd.Series(np.nan, index=df.index, dtype=float)
        empty_s = pd.Series("", index=df.index, dtype=str)
        return empty_i, empty_f, empty_f, empty_s, empty_i.copy()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Initialize output series
    signal = pd.Series(0, index=df.index, dtype=int)
    signal_atr = pd.Series(np.nan, index=df.index, dtype=float)
    entry_price = pd.Series(np.nan, index=df.index, dtype=float)
    signal_reason = pd.Series("", index=df.index, dtype=str)
    trend = pd.Series(0, index=df.index, dtype=int)

    # Candle characteristics
    body = (df["close"] - df["open"]).abs()
    delta = df["close"] - df["open"]
    candle_dir = np.sign(delta).astype(int)
    is_green = candle_dir == 1
    is_red = candle_dir == -1

    # Big body threshold
    tol = max(0.0, float(tolerance_pct or 0.0))
    target = float(body_atr_mult) * atr
    if tol > 0.0:
        target = target * (1.0 - tol)
    big_body = body >= target

    mid_body = (df["open"] + df["close"]) / 2.0

    # Build structure level series (forward-filled from confirm time)
    # We need: last HH, last HL, last LL, last LH levels at each bar
    last_hh = pd.Series(np.nan, index=df.index, dtype=float)
    last_hl = pd.Series(np.nan, index=df.index, dtype=float)
    last_ll = pd.Series(np.nan, index=df.index, dtype=float)
    last_lh = pd.Series(np.nan, index=df.index, dtype=float)
    last_structure_kind = pd.Series("", index=df.index, dtype=str)

    for sp in structure_points:
        confirm_ts = pd.to_datetime(sp.swing_point.confirm_ts, utc=True)
        if confirm_ts not in df.index:
            # Find the nearest index >= confirm_ts
            mask = df.index >= confirm_ts
            if not mask.any():
                continue
            confirm_ts = df.index[mask][0]
        
        level = sp.effective_close
        kind = sp.structure_kind
        
        if kind == "HH":
            last_hh.loc[confirm_ts:] = level
            last_structure_kind.loc[confirm_ts:] = "HH"
        elif kind == "HL":
            last_hl.loc[confirm_ts:] = level
            last_structure_kind.loc[confirm_ts:] = "HL"
        elif kind == "LL":
            last_ll.loc[confirm_ts:] = level
            last_structure_kind.loc[confirm_ts:] = "LL"
        elif kind == "LH":
            last_lh.loc[confirm_ts:] = level
            last_structure_kind.loc[confirm_ts:] = "LH"

    # Determine trend at each bar based on last structure
    # Uptrend: last structure is HH or HL
    # Downtrend: last structure is LL or LH
    trend[(last_structure_kind == "HH") | (last_structure_kind == "HL")] = 1
    trend[(last_structure_kind == "LL") | (last_structure_kind == "LH")] = -1

    # Proximity detection
    proximity = atr * float(structure_proximity_atr_mult)
    has_prox = proximity.notna()

    # Near various structure levels
    near_hh = (
        has_prox
        & last_hh.notna()
        & (df["high"] >= (last_hh - proximity))
        & (df["low"] <= (last_hh + proximity))
    )
    near_hl = (
        has_prox
        & last_hl.notna()
        & (df["high"] >= (last_hl - proximity))
        & (df["low"] <= (last_hl + proximity))
    )
    near_ll = (
        has_prox
        & last_ll.notna()
        & (df["high"] >= (last_ll - proximity))
        & (df["low"] <= (last_ll + proximity))
    )
    near_lh = (
        has_prox
        & last_lh.notna()
        & (df["high"] >= (last_lh - proximity))
        & (df["low"] <= (last_lh + proximity))
    )

    # =========================================================================
    # UPTREND SIGNALS (trend == 1)
    # =========================================================================
    in_uptrend = trend == 1

    # 1. Near HL in uptrend + green big body → LONG (buying the dip)
    hl_bounce_long = in_uptrend & near_hl & big_body & is_green
    signal[hl_bounce_long] = 1
    signal_reason[hl_bounce_long] = "hl_bounce_long"

    # 2. Near HH in uptrend + green big body + close > HH → SHORT (fade breakout)
    hh_breakout_long = in_uptrend & near_hh & big_body & is_green & (df["close"] > last_hh)
    signal[hh_breakout_long] = 1
    entry_price[hh_breakout_long] = mid_body[hh_breakout_long]
    signal_reason[hh_breakout_long] = "hh_breakout_long"

    # 3. Near HH in uptrend + red big body (rejection) → potential reversal, SHORT
    hh_rejection_short = in_uptrend & near_hh & big_body & is_red
    signal[hh_rejection_short] = -1
    signal_reason[hh_rejection_short] = "hh_rejection_short"

    # =========================================================================
    # DOWNTREND SIGNALS (trend == -1)
    # =========================================================================
    in_downtrend = trend == -1

    # 1. Near LH in downtrend + red big body → SHORT (selling the rally)
    lh_rejection_short = in_downtrend & near_lh & big_body & is_red
    signal[lh_rejection_short] = -1
    signal_reason[lh_rejection_short] = "lh_rejection_short"

    # 2. Near LL in downtrend + red big body + close < LL → LONG (fade breakdown)
    ll_breakdown_short = in_downtrend & near_ll & big_body & is_red & (df["close"] < last_ll)
    signal[ll_breakdown_short] = 1
    entry_price[ll_breakdown_short] = mid_body[ll_breakdown_short]
    signal_reason[ll_breakdown_short] = "ll_breakdown_short"

    # 3. Near LL in downtrend + green big body (rejection) → potential reversal, LONG
    ll_rejection_long = in_downtrend & near_ll & big_body & is_green
    signal[ll_rejection_long] = 1
    signal_reason[ll_rejection_long] = "ll_rejection_long"

    # =========================================================================
    # BREAK OF STRUCTURE (BOS) SIGNALS
    # =========================================================================
    
    # BOS Short: In uptrend, price closes below recent HL with big red body
    bos_short = in_uptrend & big_body & is_red & last_hl.notna() & (df["close"] < last_hl)
    signal[bos_short] = 1
    signal_reason[bos_short] = "bos_short"

    # BOS Long: In downtrend, price closes above recent LH with big green body (inverted to SHORT)
    bos_long = in_downtrend & big_body & is_green & last_lh.notna() & (df["close"] > last_lh)
    signal[bos_long] = 1
    signal_reason[bos_long] = "bos_long"

    # =========================================================================
    # MOMENTUM SIGNALS (no clear structure nearby)
    # =========================================================================
    no_structure_nearby = ~(near_hh | near_hl | near_ll | near_lh)
    
    # In uptrend, momentum long on big green candle
    momentum_long = in_uptrend & no_structure_nearby & big_body & is_green
    signal[momentum_long] = signal[momentum_long].where(signal[momentum_long] != 0, 1)
    signal_reason[momentum_long] = signal_reason[momentum_long].where(
        signal_reason[momentum_long] != "", "momentum_uptrend_long"
    )

    # In downtrend, momentum short on big red candle
    momentum_short = in_downtrend & no_structure_nearby & big_body & is_red
    signal[momentum_short] = signal[momentum_short].where(signal[momentum_short] != 0, -1)
    signal_reason[momentum_short] = signal_reason[momentum_short].where(
        signal_reason[momentum_short] != "", "momentum_downtrend_short"
    )

    # Set signal ATR for active signals
    active = signal != 0
    signal_atr[active] = atr[active]

    return signal, signal_atr, entry_price, signal_reason, trend


def build_log_payload(
    input_csv: str,
    left: int,
    right: int,
    df_1m: pd.DataFrame,
    swings_1m: List[SwingPoint],
    df_15m: Optional[pd.DataFrame],
    swings_15m: Optional[List[SwingPoint]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "input": {"csv": str(input_csv)},
        "params": {"left": left, "right": right, "timeframes": ["1m", "15m"] if df_15m is not None else ["1m"]},
        "timeframes": {
            "1m": {
                "bars": int(len(df_1m)),
                "swing_highs": int(sum(1 for s in swings_1m if s.kind == "swing_high")),
                "swing_lows": int(sum(1 for s in swings_1m if s.kind == "swing_low")),
                "swings": [s.to_dict() for s in swings_1m],
            }
        },
    }

    if df_15m is not None and swings_15m is not None:
        payload["timeframes"]["15m"] = {
            "bars": int(len(df_15m)),
            "swing_highs": int(sum(1 for s in swings_15m if s.kind == "swing_high")),
            "swing_lows": int(sum(1 for s in swings_15m if s.kind == "swing_low")),
            "swings": [s.to_dict() for s in swings_15m],
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect 1m and 15m swing highs/lows from candles")
    parser.add_argument("--csv", required=True, help="Input CSV with dt/open/high/low/close columns")
    parser.add_argument("--dt-col", default="dt", help="Datetime column name (default: dt)")
    parser.add_argument("--left", type=int, default=2, help="Pivot left bars (default: 5)")
    parser.add_argument("--right", type=int, default=2 , help="Pivot right bars (default: 5)")
    parser.add_argument("--tail", type=int, default=0, help="Only use last N rows (0 = all)")
    parser.add_argument("--no-15m", action="store_true", help="Skip 15m resample/detection")
    parser.add_argument("--resample-rule", default="15min", help="Pandas resample rule for 15m (default: 15min)")
    parser.add_argument("--output", default="output/swing_levels.json", help="Output JSON path")
    args = parser.parse_args()

    df_1m = load_ohlcv_csv(args.csv, dt_col=args.dt_col)
    if args.tail and args.tail > 0:
        df_1m = df_1m.tail(int(args.tail))

    swings_1m = detect_swings(df_1m, timeframe="1m", left=args.left, right=args.right)

    df_15m: Optional[pd.DataFrame] = None
    swings_15m: Optional[List[SwingPoint]] = None
    if not args.no_15m:
        df_15m = resample_ohlcv(df_1m, rule=args.resample_rule)
        swings_15m = detect_swings(df_15m, timeframe="15m", left=args.left, right=args.right)

    payload = build_log_payload(
        input_csv=args.csv,
        left=int(args.left),
        right=int(args.right),
        df_1m=df_1m,
        swings_1m=swings_1m,
        df_15m=df_15m,
        swings_15m=swings_15m,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved swing levels log: {out_path}")


if __name__ == "__main__":
    main()
