"""Visualize market structure swing levels (HH/HL/LH/LL) with candlesticks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from config import get_backtest_charts_dir, get_backtest_swing_levels_dir


KINDS = ("HH", "HL", "LH", "LL")
KIND_STYLE = {
    "HH": {"color": "green", "linewidth": 2.0, "alpha": 0.9},
    "HL": {"color": "teal", "linewidth": 2.0, "alpha": 0.9},
    "LH": {"color": "orange", "linewidth": 2.0, "alpha": 0.9},
    "LL": {"color": "red", "linewidth": 2.0, "alpha": 0.9},
}
DEFAULT_KLINE_DIR = Path("kline_data")
LINE_LENGTH_CANDLES = 3


def _latest_json(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("swing_levels_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No swing level JSON files found in {dir_path}")
    return candidates[0]


def _parse_swing_filename(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    stem = path.stem
    if not stem.startswith("swing_levels_"):
        return None, None, None
    rest = stem.replace("swing_levels_", "", 1)
    parts = rest.split("_")
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def _parse_kline_filename(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    parts = path.name.split("_")
    if len(parts) < 5:
        return None, None, None, None
    symbol = parts[0]
    timeframe = parts[1]
    start = None
    end = None
    if len(parts) >= 5 and parts[3] == "to":
        start = parts[2]
        end = parts[4]
    return symbol, timeframe, start, end


def _find_kline_csv(
    symbol: str,
    *,
    timeframe: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> Optional[Path]:
    if not DEFAULT_KLINE_DIR.exists():
        return None

    candidates = list(DEFAULT_KLINE_DIR.glob(f"{symbol}_*.csv"))
    if not candidates:
        return None

    def matches_range(path: Path) -> bool:
        _, _, f_start, f_end = _parse_kline_filename(path)
        if start and end and f_start and f_end:
            return f_start == start and f_end == end
        return False

    def matches_timeframe(path: Path) -> bool:
        _, f_tf, _, _ = _parse_kline_filename(path)
        return bool(timeframe and f_tf == timeframe)

    range_matches = [p for p in candidates if matches_range(p)]
    if range_matches:
        candidates = range_matches

    tf_matches = [p for p in candidates if matches_timeframe(p)]
    if tf_matches:
        candidates = tf_matches

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pick_symbol(data: Dict[str, object], symbol: str | None) -> str:
    if symbol:
        if symbol not in data:
            raise KeyError(f"Symbol '{symbol}' not found in JSON file.")
        return symbol
    if len(data) == 1:
        return next(iter(data.keys()))
    raise ValueError("Multiple symbols found. Provide --symbol to select one.")


def _extract_points(
    entry: Dict[str, object],
    ohlc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    points = entry.get("structure_points", [])
    ohlc_index = None
    if ohlc_df is not None and not ohlc_df.empty:
        ohlc_index = ohlc_df.index

    def next_candle_level(pivot_ts: Optional[str], swing_kind: Optional[str]) -> Optional[float]:
        if ohlc_index is None or not pivot_ts or not swing_kind:
            return None
        pivot = pd.to_datetime(pivot_ts, utc=True)
        pos = ohlc_index.searchsorted(pivot)
        if pos < len(ohlc_index) and ohlc_index[pos] != pivot and pos > 0:
            pos -= 1
        next_pos = pos + 1
        if next_pos >= len(ohlc_index):
            return None
        candle = ohlc_df.iloc[next_pos]
        if swing_kind == "swing_high":
            return float(candle["high"])
        if swing_kind == "swing_low":
            return float(candle["low"])
        return None

    for sp in points:
        kind = sp.get("structure_kind")
        if kind not in KINDS:
            continue
        swing_point = sp.get("swing_point", {})
        ts = swing_point.get("pivot_ts")
        level = swing_point.get("level")
        swing_kind = swing_point.get("kind")
        if not swing_kind:
            swing_kind = "swing_high" if kind in ("HH", "LH") else "swing_low"
        if sp.get("is_liquidity_sweep"):
            sweep_level = next_candle_level(swing_point.get("pivot_ts"), swing_kind)
            if sweep_level is not None:
                level = sweep_level
        if ts is None or level is None:
            continue
        rows.append(
            {
                "kind": kind,
                "timestamp": pd.to_datetime(ts, utc=True),
                "level": float(level),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["kind", "timestamp", "level"])
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df


def _parse_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        sample = float(numeric.dropna().iloc[0])
        unit = "ms" if sample > 1e12 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_ohlcv_csv(path: Path, dt_col: str = "") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    ts = None
    if dt_col:
        dt_col = dt_col.strip().lower()
        if dt_col in df.columns:
            ts = _parse_timestamp(df[dt_col])
            df = df.drop(columns=[dt_col])
    if ts is None:
        for candidate in ("open_time_ms", "open_time", "timestamp", "dt"):
            if candidate in df.columns:
                ts = _parse_timestamp(df[candidate])
                df = df.drop(columns=[candidate])
                break

    if ts is None:
        raise ValueError("Failed to find a datetime column for OHLC CSV.")

    df.index = pd.DatetimeIndex(ts, name="dt")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC CSV missing columns: {missing}")

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


def _timeframe_to_rule(timeframe: str) -> Optional[str]:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{tf[:-1]}min"
    if tf.endswith("h") and tf[:-1].isdigit():
        return f"{tf[:-1]}h"
    if tf.endswith("d") and tf[:-1].isdigit():
        return f"{tf[:-1]}d"
    return None


def _infer_candle_width_minutes(df: pd.DataFrame) -> float:
    if df.shape[0] < 2:
        return 1.0
    diffs = pd.Series(df.index).diff().dropna()
    if diffs.empty:
        return 1.0
    minutes = diffs.dt.total_seconds().median() / 60.0
    return max(0.2, float(minutes) * 0.8)


def _infer_candle_delta(df: pd.DataFrame) -> pd.Timedelta:
    if df.shape[0] < 2:
        return pd.Timedelta(minutes=1)
    diffs = pd.Series(df.index).diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=1)
    return diffs.median()


def draw_candlesticks(ax, ohlc_df: pd.DataFrame, width_minutes: float) -> None:
    if ohlc_df.empty:
        return

    width = width_minutes / (24 * 60)
    for idx, row in ohlc_df.iterrows():
        open_price = row["open"]
        close_price = row["close"]
        high_price = row["high"]
        low_price = row["low"]

        if close_price >= open_price:
            color = "green"
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = "red"
            body_bottom = close_price
            body_height = open_price - close_price

        x = mdates.date2num(idx.to_pydatetime())
        ax.plot([x, x], [low_price, high_price], color=color, linewidth=0.6, alpha=0.7)

        if body_height > 0:
            rect = Rectangle(
                (x - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.65,
                linewidth=0.4,
            )
            ax.add_patch(rect)
        else:
            ax.plot([x - width / 2, x + width / 2], [open_price, open_price], color=color, linewidth=0.8)


def _default_output_path(input_path: Path, symbol: str) -> Path:
    charts_dir = get_backtest_charts_dir()
    charts_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    if stem.startswith("swing_levels_"):
        stem = stem.replace("swing_levels_", "swing_structure_", 1)
    else:
        stem = f"swing_structure_{symbol}"
    return charts_dir / f"{stem}.png"


def plot_structure_levels(
    df: pd.DataFrame,
    symbol: str,
    entry: Dict[str, object],
    *,
    ohlc_df: Optional[pd.DataFrame],
    output_path: Path | None,
    show: bool,
) -> None:
    if df.empty:
        raise ValueError("No HH/HL/LH/LL structure points found to plot.")

    fig, ax = plt.subplots(figsize=(14, 8))
    if ohlc_df is not None and not ohlc_df.empty:
        width_minutes = _infer_candle_width_minutes(ohlc_df)
        draw_candlesticks(ax, ohlc_df, width_minutes=width_minutes)

    if ohlc_df is not None and not ohlc_df.empty:
        candle_delta = _infer_candle_delta(ohlc_df)
    else:
        candle_delta = _infer_candle_delta(df.set_index("timestamp"))
    line_len = candle_delta * LINE_LENGTH_CANDLES

    for kind in KINDS:
        group = df[df["kind"] == kind]
        if group.empty:
            continue
        start = group["timestamp"]
        end = group["timestamp"] + line_len
        ax.hlines(
            group["level"],
            xmin=start,
            xmax=end,
            label=kind,
            **KIND_STYLE.get(kind, {}),
        )

    params = entry.get("params", {}) if isinstance(entry.get("params"), dict) else {}
    timeframe = params.get("timeframe")
    left = params.get("left")
    right = params.get("right")

    title_bits: List[str] = []
    if timeframe:
        title_bits.append(f"timeframe {timeframe}")
    if left is not None and right is not None:
        title_bits.append(f"left/right {left}/{right}")
    subtitle = f" ({', '.join(title_bits)})" if title_bits else ""

    ax.set_title(f"{symbol} Market Structure Levels{subtitle}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize HH/HL/LH/LL swing structure levels from swing_levels JSON."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Path to swing_levels JSON file (defaults to latest in backtest swing_levels).",
    )
    parser.add_argument("--symbol", default="", help="Symbol to plot (required if file has multiple).")
    parser.add_argument("--csv", default="", help="OHLC CSV path for candlesticks (optional).")
    parser.add_argument("--dt-col", default="", help="Datetime column name for OHLC CSV (optional).")
    parser.add_argument(
        "--resample",
        default="",
        help="Optional pandas resample rule for OHLC (e.g. 1D, 15min).",
    )
    parser.add_argument("--output", default="", help="Output PNG path (default: backtest charts dir).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else _latest_json(get_backtest_swing_levels_dir())
    data = json.loads(input_path.read_text(encoding="utf-8"))
    symbol = _pick_symbol(data, args.symbol.strip() or None)
    entry = data[symbol]

    ohlc_df = None
    params = entry.get("params", {}) if isinstance(entry.get("params"), dict) else {}
    swing_tf = params.get("timeframe")
    resample_rule = args.resample.strip()
    if not resample_rule and isinstance(swing_tf, str):
        resample_rule = _timeframe_to_rule(swing_tf) or ""

    ohlc_path = Path(args.csv) if args.csv else None
    if ohlc_path is None:
        _, start, end = _parse_swing_filename(input_path)
        ohlc_path = _find_kline_csv(
            symbol,
            timeframe=swing_tf if isinstance(swing_tf, str) else None,
            start=start,
            end=end,
        )
    if ohlc_path is not None and ohlc_path.exists():
        ohlc_df = load_ohlcv_csv(ohlc_path, dt_col=args.dt_col)
        _, ohlc_tf, _, _ = _parse_kline_filename(ohlc_path)
        should_resample = bool(resample_rule)
        if ohlc_tf and isinstance(swing_tf, str) and ohlc_tf == swing_tf and not args.resample.strip():
            should_resample = False
        if should_resample:
            ohlc_df = resample_ohlcv(ohlc_df, resample_rule)
    else:
        raise FileNotFoundError("No OHLC data found. Provide --csv or ensure kline_data has the symbol.")

    df = _extract_points(entry, ohlc_df)

    output_path = Path(args.output) if args.output else _default_output_path(input_path, symbol)
    plot_structure_levels(
        df,
        symbol,
        entry,
        ohlc_df=ohlc_df,
        output_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
