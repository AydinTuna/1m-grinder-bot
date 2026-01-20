"""Shared configuration for the ATR strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


LIVE_TRADE_FIELDS: List[str] = [
    "entry_time",
    "exit_time",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "signal_atr",
    "stop_price",
    "exit_reason",
    "roi_net",
    "pnl_net",
    "margin_used",
    "notional",
    "equity_after",
    "bars_held",
]

LIVE_SIGNAL_FIELDS: List[str] = [
    "timestamp",
    "symbol",
    "side",
    "signal",
    "signal_atr",
    "entry_price",
    "open",
    "high",
    "low",
    "close",
    "atr",
    "status",  # "ACTED", "SKIPPED_MAX_POS", "SKIPPED_HAS_POS", "SKIPPED_PENDING", "SKIPPED_OUTSIDE_WINDOW"
]


@dataclass
class BacktestConfig:
    # Timeframe for signal generation (ATR calculation, entry signals)
    # Use "1d" for daily strategy. Execution uses 1m candles for trailing.
    signal_interval: str = "1d"
    
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    signal_atr_tolerance_pct: float = 0.1  # 0.1 = 10%
    swing_timeframe: str = "1d"
    swing_left: int = 2
    swing_right: int = 2
    swing_resample_rule: str = "1d"
    swing_proximity_atr_mult: float = 0.25
    entry_limit_timeout_bars: int = 1
    leverage: float = 20.0  # fixed leverage for static sizing
    initial_capital: float = 100.0  # starting equity (margin cap per trade)
    margin_usd: float = 5.0  # static margin per trade

    # Costs
    fee_rate: float = 0.0000    # per side (0.04% typical maker/taker varies)
    slippage: float = 0.0000    # price impact fraction applied on entry/exit

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls - trailing stop only
    use_trailing_stop: bool = True
    trailing_mode: str = "r_ladder"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 1.00  # ATR multiplier for dynamic trailing
    trail_check_interval: str = "1m"  # interval for trailing stop updates (Look In Bar)
    
    # Multi-position settings
    max_open_positions: int = 3  # max concurrent positions


@dataclass
class LiveConfig:
    # Timeframe for signal generation (ATR calculation, entry signals)
    # Use "1d" for daily strategy. Trailing stop uses trail_check_interval.
    signal_interval: str = "1d"
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    signal_atr_tolerance_pct: float = 0.05  # 0.05 = 5%
    swing_timeframe: str = "1d"
    swing_left: int = 1
    swing_right: int = 1
    swing_resample_rule: str = "1d"
    swing_proximity_atr_mult: float = 0.25
    atr_history_bars: int = 100  # bars to pull for stable ATR/EMA (1d candles)
    leverage: int = 5  # fixed leverage for static sizing
    margin_usd: float = 5.0  # static margin per trade

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls - trailing stop only (no TP/SL on entry)
    use_trailing_stop: bool = True
    trailing_mode: str = "r_ladder"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 1.25  # ATR multiplier for dynamic trailing
    trail_check_interval: str = "1m"  # interval for trailing stop updates (Look In Bar)

    # Multi-position settings
    max_open_positions: int = 3  # max concurrent positions

    # Entry time window (only enter positions within X minutes after daily candle close at 00:00 UTC)
    # Set to 0 to disable this check and allow entries anytime
    entry_window_minutes: int = 60

    # Algo order controls
    algo_type: str = "CONDITIONAL"
    algo_working_type: str = "CONTRACT_PRICE"
    algo_price_protect: bool = False

    # Execution controls
    entry_delay_min_seconds: float = 0.01
    entry_delay_max_seconds: float = 0.01
    spread_max_pct: float = 1.0  # disabled (set to 0.0001 for 0.01% filter)
    atr_offset_mult: float = 0.02
    poll_interval_seconds: float = 60.0  # check every minute for trailing
    entry_order_timeout_seconds: float = 55.0
    log_path: str = "trade_log.jsonl"
    live_trades_csv: str = "live_trades.csv"
    live_signals_csv: str = "live_signals.csv"
    post_only: bool = True
    use_testnet: bool = False
