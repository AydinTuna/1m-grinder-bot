"""Shared configuration for the ATR strategy."""

from __future__ import annotations

from dataclasses import dataclass
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
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    leverage: float = 10.0  # max leverage for dynamic sizing
    min_leverage: float = 1.0
    initial_capital: float = 400.0  # starting equity (margin cap per trade)

    # Position sizing targets (optional)
    target_profit_usd: Optional[float] = 0.75
    target_loss_usd: Optional[float] = 0.50

    # Costs
    fee_rate: float = 0.0000    # per side (0.04% typical maker/taker varies)
    slippage: float = 0.0000    # price impact fraction applied on entry/exit

    # Strategy thresholds
    thr1: float = 1.5
    thr2: float = 2.0

    # Risk/exit controls
    tp_atr_mult: float = 1.5
    sl_atr_mult: float = 1.0


@dataclass
class LiveConfig:
    symbol: str = "BTCUSDC"
    interval: str = "1m"
    atr_len: int = 14
    atr_warmup_bars: Optional[int] = None  # defaults to atr_len when None
    leverage: int = 50  # max leverage for dynamic sizing
    min_leverage: int = 20
    margin_usd: float = 50.0  # margin cap per trade if targets are set
    target_profit_usd: Optional[float] = 0.75
    target_loss_usd: Optional[float] = 0.50

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls
    tp_atr_mult: float = 1.5
    sl_atr_mult: float = 1.0

    # Algo order controls
    algo_working_type: str = "CONTRACT_PRICE"
    algo_price_protect: bool = False

    # Execution controls
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
