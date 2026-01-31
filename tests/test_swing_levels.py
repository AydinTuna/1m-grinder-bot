import unittest

import numpy as np
import pandas as pd

from swing_levels import build_swing_atr_signals, is_liquidity_sweep


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


if __name__ == "__main__":
    unittest.main()
