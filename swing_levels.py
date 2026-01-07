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
) -> Tuple[pd.Series, pd.Series, pd.Series]:
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

    Returns:
      signal_dir: +1 long, -1 short, 0 none
      signal_atr: ATR on the signal candle (NaN if none)
      entry_price: NaN for market-on-next-bar, otherwise limit entry price
    """
    if df.empty:
        empty_i = pd.Series(0, index=df.index, dtype=int)
        empty_f = pd.Series(np.nan, index=df.index, dtype=float)
        return empty_i, empty_f, empty_f

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

    mid_body = (df["open"] + df["close"]) / 2.0

    # Neither swing high nor swing low => same direction as candle (if big body)
    other = big_body & ~(near_high | near_low)
    signal[other] = candle_dir[other]

    # Near swing high rules
    cond_high = near_high
    short_red = cond_high & is_red
    short_failed = cond_high & is_green & (df["close"] < swing_high_level)
    long_break = cond_high & is_green & (df["close"] >= swing_high_level)

    signal[short_red | short_failed] = -1
    signal[long_break] = 1
    entry_price[long_break] = mid_body[long_break]

    # Near swing low rules
    cond_low = near_low
    long_green = cond_low & is_green
    long_failed = cond_low & is_red & (df["close"] > swing_low_level)
    short_break = cond_low & is_red & (df["close"] <= swing_low_level)

    signal[long_green | long_failed] = 1
    signal[short_break] = -1
    entry_price[short_break] = mid_body[short_break]

    active = signal != 0
    signal_atr[active] = atr[active]

    return signal, signal_atr, entry_price


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
    parser.add_argument("--output", default="swing_levels.json", help="Output JSON path")
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
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved swing levels log: {out_path}")


if __name__ == "__main__":
    main()
