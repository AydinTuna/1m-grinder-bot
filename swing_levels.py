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
from typing import Any, Dict, List, Literal, Optional

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
