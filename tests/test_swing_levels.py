import unittest

import numpy as np
import pandas as pd

from swing_levels import (
    MarketStructurePoint,
    SwingPoint,
    build_market_structure_signals,
    build_swing_atr_signals,
    is_liquidity_sweep,
)


def _swing_point(
    kind: str,
    confirm_ts: str,
    *,
    high: float,
    low: float,
    level: float,
) -> SwingPoint:
    return SwingPoint(
        kind=kind,
        timeframe="1d",
        pivot_ts=confirm_ts,
        confirm_ts=confirm_ts,
        level=level,
        bar_index=0,
        open=(high + low) / 2,
        high=high,
        low=low,
        close=(high + low) / 2,
    )


def _structure_point(
    structure_kind: str,
    confirm_ts: str,
    *,
    high: float,
    low: float,
) -> MarketStructurePoint:
    if structure_kind in ("HH", "LH"):
        kind = "swing_high"
        level = high
    else:
        kind = "swing_low"
        level = low
    sp = _swing_point(kind, confirm_ts, high=high, low=low, level=level)
    return MarketStructurePoint(
        structure_kind=structure_kind,
        swing_point=sp,
        reference_level=level,
        effective_close=sp.close,
        is_liquidity_sweep=False,
    )


def _uptrend_structure() -> list:
    """HH -> HL -> HH establishes uptrend (trend == 1)."""
    return [
        _structure_point("HH", "2025-01-01T00:00:00+00:00", high=120.0, low=115.0),
        _structure_point("HL", "2025-01-02T00:00:00+00:00", high=108.0, low=100.0),
        _structure_point("HH", "2025-01-03T00:00:00+00:00", high=125.0, low=118.0),
    ]


def _downtrend_structure() -> list:
    """LL -> LH -> LL establishes downtrend (trend == -1)."""
    return [
        _structure_point("LL", "2025-01-01T00:00:00+00:00", high=85.0, low=80.0),
        _structure_point("LH", "2025-01-02T00:00:00+00:00", high=90.0, low=85.0),
        _structure_point("LL", "2025-01-03T00:00:00+00:00", high=75.0, low=70.0),
    ]


class TestLiquiditySweep(unittest.TestCase):
    def test_liquidity_sweep_true_on_bullish_wick_dominance(self) -> None:
        sweep = is_liquidity_sweep(
            candle_open=100.0,
            candle_high=125.0,
            candle_low=95.0,
            candle_close=110.0,
        )
        self.assertTrue(sweep)

    def test_liquidity_sweep_true_on_bearish_wick_dominance(self) -> None:
        sweep = is_liquidity_sweep(
            candle_open=110.0,
            candle_high=115.0,
            candle_low=85.0,
            candle_close=100.0,
        )
        self.assertTrue(sweep)

    def test_liquidity_sweep_false_on_small_body(self) -> None:
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=100.0,
                candle_high=112.0,
                candle_low=90.0,
                candle_close=101.0,
            )
        )

    def test_liquidity_sweep_false_on_invalid_candle(self) -> None:
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=np.nan,
                candle_high=120.0,
                candle_low=95.0,
                candle_close=110.0,
            )
        )
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=100.0,
                candle_high=100.0,
                candle_low=100.0,
                candle_close=100.0,
            )
        )


class TestSwingAtrEntryPrice(unittest.TestCase):
    def test_long_breakout_entry_uses_close(self) -> None:
        df = pd.DataFrame(
            {
                "open": [0.0],
                "high": [3.2],
                "low": [0.0],
                "close": [3.0],
            },
            index=pd.to_datetime(["2025-01-01"], utc=True),
        )
        atr = pd.Series([1.0], index=df.index)
        swing_high = pd.Series([2.5], index=df.index)
        swing_low = pd.Series([np.nan], index=df.index)

        signal, signal_atr, entry_price, signal_reason = build_swing_atr_signals(
            df,
            atr,
            swing_high,
            swing_low,
            body_atr_mult=2.0,
            swing_proximity_atr_mult=0.25,
        )

        self.assertEqual(signal.iloc[0], 1)
        self.assertEqual(entry_price.iloc[0], df["close"].iloc[0])
        self.assertEqual(signal_reason.iloc[0], "swing_high_breakout_long")

    def test_short_breakdown_entry_uses_close(self) -> None:
        df = pd.DataFrame(
            {
                "open": [3.0],
                "high": [3.2],
                "low": [0.0],
                "close": [0.0],
            },
            index=pd.to_datetime(["2025-01-02"], utc=True),
        )
        atr = pd.Series([1.0], index=df.index)
        swing_high = pd.Series([np.nan], index=df.index)
        swing_low = pd.Series([0.5], index=df.index)

        signal, signal_atr, entry_price, signal_reason = build_swing_atr_signals(
            df,
            atr,
            swing_high,
            swing_low,
            body_atr_mult=2.0,
            swing_proximity_atr_mult=0.25,
        )

        self.assertEqual(signal.iloc[0], -1)
        self.assertEqual(entry_price.iloc[0], df["close"].iloc[0])
        self.assertEqual(signal_reason.iloc[0], "swing_low_breakdown_short")


class TestBosPenetrationDirection(unittest.TestCase):
    def _run_signals(
        self,
        ohlc_row: dict,
        structure_points: list,
        *,
        signal_ts: str = "2025-01-04T00:00:00+00:00",
    ):
        index = pd.to_datetime(
            [
                "2025-01-01T00:00:00+00:00",
                "2025-01-02T00:00:00+00:00",
                "2025-01-03T00:00:00+00:00",
                signal_ts,
            ],
            utc=True,
        )
        rows = [
            {"open": 110.0, "high": 112.0, "low": 108.0, "close": 111.0},
            {"open": 105.0, "high": 106.0, "low": 101.0, "close": 102.0},
            {"open": 120.0, "high": 122.0, "low": 118.0, "close": 121.0},
            ohlc_row,
        ]
        df = pd.DataFrame(rows, index=index)
        atr = pd.Series([2.0] * len(df), index=df.index)
        return build_market_structure_signals(
            df,
            atr,
            structure_points,
            body_atr_mult=1.0,
            structure_proximity_atr_mult=0.25,
            bos_penetration_body_ratio=0.5,
        )

    def test_bos_short_fade_deep_penetration_long(self) -> None:
        (
            signal,
            _signal_atr,
            _entry_price,
            signal_reason,
            trend,
            fade_dir,
            _fade_entry,
            _fade_tp,
            _fade_sl,
        ) = self._run_signals(
            {"open": 104.0, "high": 105.0, "low": 94.0, "close": 95.0},
            _uptrend_structure(),
        )
        ts = pd.Timestamp("2025-01-04T00:00:00+00:00", tz="UTC")
        self.assertEqual(trend.loc[ts], 1)
        self.assertEqual(signal.loc[ts], 1)
        self.assertEqual(signal_reason.loc[ts], "bos_short_fade")
        self.assertEqual(fade_dir.loc[ts], 0)

    def test_bos_short_break_shallow_penetration_short(self) -> None:
        (
            signal,
            _signal_atr,
            _entry_price,
            signal_reason,
            _trend,
            fade_dir,
            _fade_entry,
            _fade_tp,
            _fade_sl,
        ) = self._run_signals(
            {"open": 103.0, "high": 104.0, "low": 98.0, "close": 99.0},
            _uptrend_structure(),
        )
        ts = pd.Timestamp("2025-01-04T00:00:00+00:00", tz="UTC")
        self.assertEqual(signal.loc[ts], -1)
        self.assertEqual(signal_reason.loc[ts], "bos_short_break")
        self.assertEqual(fade_dir.loc[ts], 0)

    def test_bos_long_fade_deep_penetration_short_with_fade_fields(self) -> None:
        (
            signal,
            _signal_atr,
            _entry_price,
            signal_reason,
            trend,
            fade_dir,
            fade_entry,
            _fade_tp,
            _fade_sl,
        ) = self._run_signals(
            {"open": 87.0, "high": 96.0, "low": 86.0, "close": 95.0},
            _downtrend_structure(),
        )
        ts = pd.Timestamp("2025-01-04T00:00:00+00:00", tz="UTC")
        self.assertEqual(trend.loc[ts], -1)
        self.assertEqual(signal.loc[ts], -1)
        self.assertEqual(signal_reason.loc[ts], "bos_long_fade")
        self.assertEqual(fade_dir.loc[ts], -1)
        self.assertEqual(fade_entry.loc[ts], 95.0)

    def test_bos_long_break_shallow_penetration_long_no_fade(self) -> None:
        (
            signal,
            _signal_atr,
            _entry_price,
            signal_reason,
            _trend,
            fade_dir,
            fade_entry,
            _fade_tp,
            _fade_sl,
        ) = self._run_signals(
            {"open": 88.0, "high": 92.0, "low": 87.0, "close": 91.0},
            _downtrend_structure(),
        )
        ts = pd.Timestamp("2025-01-04T00:00:00+00:00", tz="UTC")
        self.assertEqual(signal.loc[ts], 1)
        self.assertEqual(signal_reason.loc[ts], "bos_long_break")
        self.assertEqual(fade_dir.loc[ts], 0)
        self.assertTrue(np.isnan(fade_entry.loc[ts]))


if __name__ == "__main__":
    unittest.main()
