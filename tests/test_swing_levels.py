import unittest

import numpy as np

from swing_levels import is_liquidity_sweep


class TestLiquiditySweep(unittest.TestCase):
    def test_liquidity_sweep_true_on_large_wick(self) -> None:
        sweep = is_liquidity_sweep(
            candle_open=100.0,
            candle_high=121.0,
            candle_low=98.0,
            candle_close=110.0,
            atr_value=18.0,
        )
        self.assertTrue(sweep)

    def test_liquidity_sweep_false_on_threshold(self) -> None:
        sweep = is_liquidity_sweep(
            candle_open=100.0,
            candle_high=115.5,
            candle_low=98.0,
            candle_close=110.0,
            atr_value=10.0,
        )
        self.assertFalse(sweep)

    def test_liquidity_sweep_false_on_invalid_atr(self) -> None:
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=100.0,
                candle_high=120.0,
                candle_low=95.0,
                candle_close=110.0,
                atr_value=None,
            )
        )
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=100.0,
                candle_high=120.0,
                candle_low=95.0,
                candle_close=110.0,
                atr_value=np.nan,
            )
        )
        self.assertFalse(
            is_liquidity_sweep(
                candle_open=100.0,
                candle_high=120.0,
                candle_low=95.0,
                candle_close=110.0,
                atr_value=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
