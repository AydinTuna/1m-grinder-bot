"""Shared configuration for the ATR strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


LIVE_TRADE_FIELDS: List[str] = [
    "entry_time",
    "exit_time",
    "side",
    "entry_price",
    "exit_price",
    "signal_atr",
    "tp_atr_mult",
    "sl_atr_mult",
    "target_price",
    "stop_price",
    "exit_reason",
    "roi_net",
    "pnl_net",
    "margin_used",
    "notional",
    "equity_after",
    "bars_held",
]


@dataclass
class BacktestConfig:
    # Timeframe for signal generation (ATR calculation, entry signals)
    # Use "1m", "5m", "15m", "1h", etc. Execution still uses 1s candles.
    signal_interval: str = "1m"
    
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    signal_atr_tolerance_pct: float = 0.05  # 0.05 = 5%
    swing_timeframe: str = "15m"
    swing_left: int = 2
    swing_right: int = 2
    swing_resample_rule: str = "15min"
    swing_proximity_atr_mult: float = 0.25
    entry_limit_timeout_bars: int = 1
    leverage: float = 100.0  # max leverage for dynamic sizing
    min_leverage: float = 20.0
    initial_capital: float = 200.0  # starting equity (margin cap per trade)

    # Position sizing target (optional, profit implied by TP/SL ratio)
    target_loss_usd: Optional[float] = 0.10

    # Costs
    fee_rate: float = 0.0000    # per side (0.04% typical maker/taker varies)
    slippage: float = 0.0000    # price impact fraction applied on entry/exit

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 0.25
    use_trailing_stop: bool = True
    trailing_mode: str = "dynamic_atr"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 1.00  # ATR multiplier for dynamic trailing
    sl_maker_offset_atr_mult: float = 0.10  # offset for SL limit price to ensure fill (trigger-to-limit gap)


@dataclass
class LiveConfig:
    symbol: str = "BTCUSDC"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDC"])
    # Timeframe for signal generation (ATR calculation, entry signals)
    # Use "1m", "5m", "15m", "1h", etc. Price tracking uses poll_interval_seconds.
    signal_interval: str = "1m"
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    signal_atr_tolerance_pct: float = 0.05  # 0.05 = 5%
    swing_timeframe: str = "15m"
    swing_left: int = 2
    swing_right: int = 2
    swing_resample_rule: str = "15min"
    swing_proximity_atr_mult: float = 0.25
    atr_history_bars: int = 500  # bars to pull for stable ATR/EMA
    leverage: int = 100  # max leverage for dynamic sizing
    min_leverage: int = 20
    margin_usd: float = 200.0  # margin cap per trade if targets are set
    target_loss_usd: Optional[float] = 0.10

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 0.25
    use_trailing_stop: bool = True
    trailing_mode: str = "r_ladder"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 1.25  # ATR multiplier for dynamic trailing
    sl_maker_offset_atr_mult: float = 0.10  # offset for SL limit price to ensure fill (trigger-to-limit gap)
    sl_chase_timeout_seconds: float = 60.0  # seconds before chasing unfilled SL order

    # Algo order controls
    algo_type: str = "CONDITIONAL"
    algo_working_type: str = "CONTRACT_PRICE"
    algo_price_protect: bool = False

    # Execution controls
    tp_post_only: bool = True
    entry_delay_min_seconds: float = 5.0
    entry_delay_max_seconds: float = 5.0
    spread_max_pct: float = 0.0001  # 0.01%
    atr_offset_mult: float = 0.02
    poll_interval_seconds: float = 1.0
    entry_order_timeout_seconds: float = 55.0
    log_path: str = "trade_log.jsonl"
    live_trades_csv: str = "live_trades.csv"
    post_only: bool = True
    use_testnet: bool = False
