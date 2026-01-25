import unittest

import numpy as np

from swing_levels import is_liquidity_sweep


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


if __name__ == "__main__":
    unittest.main()
