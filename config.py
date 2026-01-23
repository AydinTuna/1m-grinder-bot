"""Shared configuration for the ATR strategy."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


# =============================================================================
# STRATEGY VERSIONING
# =============================================================================
# Increment when signal logic changes:
#   MAJOR: Breaking changes (signal logic completely rewritten)
#   MINOR: Signal modifications (direction changes, new signal types)
#   PATCH: Bug fixes, config tuning
STRATEGY_VERSION: str = "1.2.1"

# Brief description of current version
STRATEGY_VERSION_NOTE: str = "Update swing level detection logic"


# =============================================================================
# OUTPUT DIRECTORY STRUCTURE
# =============================================================================
OUTPUT_DIR: Path = Path(__file__).parent / "output"


def get_versioned_output_dir(category: str = "backtest") -> Path:
    """Get output directory for current strategy version.
    
    Args:
        category: One of "backtest", "live", or "data"
        
    Returns:
        Path to the appropriate output directory
    """
    if category == "backtest":
        return OUTPUT_DIR / "backtest" / f"v{STRATEGY_VERSION}"
    elif category == "live":
        return OUTPUT_DIR / "live"
    else:
        return OUTPUT_DIR / "data"


def get_backtest_signals_dir() -> Path:
    """Get directory for backtest signals CSV files."""
    return get_versioned_output_dir("backtest") / "signals"


def get_backtest_trades_dir() -> Path:
    """Get directory for backtest trades CSV files."""
    return get_versioned_output_dir("backtest") / "trades"


def get_backtest_stats_dir() -> Path:
    """Get directory for backtest statistics CSV files."""
    return get_versioned_output_dir("backtest") / "stats"


def get_backtest_charts_dir() -> Path:
    """Get directory for backtest chart images."""
    return get_versioned_output_dir("backtest") / "charts"


def get_live_signals_dir() -> Path:
    """Get directory for live signals CSV files."""
    return get_versioned_output_dir("live") / "signals"


def get_live_trades_dir() -> Path:
    """Get directory for live trades CSV files."""
    return get_versioned_output_dir("live") / "trades"


def get_live_logs_dir() -> Path:
    """Get directory for live trading logs."""
    return get_versioned_output_dir("live") / "logs"


def get_data_klines_dir() -> Path:
    """Get directory for cached kline data."""
    return get_versioned_output_dir("data") / "klines"


def get_data_swing_levels_dir() -> Path:
    """Get directory for swing level JSON files."""
    return get_versioned_output_dir("data") / "swing_levels"


def get_comparison_dir() -> Path:
    """Get directory for version comparison reports."""
    return OUTPUT_DIR / "backtest" / "comparison"


def ensure_output_dirs() -> None:
    """Create all output subdirectories if they don't exist."""
    dirs = [
        get_backtest_signals_dir(),
        get_backtest_trades_dir(),
        get_backtest_stats_dir(),
        get_backtest_charts_dir(),
        get_live_signals_dir(),
        get_live_trades_dir(),
        get_live_logs_dir(),
        get_data_klines_dir(),
        get_data_swing_levels_dir(),
        get_comparison_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_output_path(filename: str) -> str:
    """Get full path for an output file, ensuring the output directory exists.
    
    DEPRECATED: Use specific directory functions instead (get_backtest_signals_dir, etc.)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / filename)


def save_config_snapshot(cfg: Any, symbol: str = "ALL", start_date: str = "", end_date: str = "") -> Path:
    """Save config snapshot as JSON for reproducibility.
    
    Args:
        cfg: BacktestConfig or LiveConfig instance
        symbol: Symbol(s) being backtested
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        Path to the saved snapshot file
    """
    version_dir = get_versioned_output_dir("backtest")
    version_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot = {
        "strategy_version": STRATEGY_VERSION,
        "strategy_version_note": STRATEGY_VERSION_NOTE,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "config": asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else {}
    }
    
    snapshot_path = version_dir / "config_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    
    return snapshot_path


def generate_trade_id(symbol: str, timestamp: str, side: str) -> str:
    """Generate a unique trade ID for linking signals to trades.
    
    Args:
        symbol: Trading symbol
        timestamp: Entry timestamp (ISO format or any string)
        side: "LONG" or "SHORT"
        
    Returns:
        Unique trade ID string
    """
    # Clean timestamp to remove special characters
    ts_clean = str(timestamp).replace(":", "").replace("-", "").replace(" ", "_").replace("+", "")[:15]
    return f"{symbol}_{side}_{ts_clean}"


# =============================================================================
# FIELD DEFINITIONS FOR CSV FILES
# =============================================================================

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
    "trade_id",          # Unique ID linking to signal
    "signal_reason",     # Signal type that triggered this trade
    "strategy_version",  # Version of strategy that generated this trade
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
    "status",            # "ACTED", "SKIPPED_MAX_POS", "SKIPPED_HAS_POS", "SKIPPED_PENDING", "SKIPPED_OUTSIDE_WINDOW"
    "signal_reason",     # e.g. "swing_high_rejection_short", "momentum_long", etc.
    "trade_id",          # Unique ID if signal was acted upon (links to trade)
    "strategy_version",  # Version of strategy that generated this signal
]

# Fields for backtest signals with trade outcome tracking
BACKTEST_SIGNAL_FIELDS: List[str] = [
    "timestamp",
    "symbol",
    "side",
    "signal",
    "signal_atr",
    "entry_price",
    "signal_reason",
    "trade_id",          # Unique ID if signal resulted in a trade
    "trade_outcome",     # "WIN", "LOSS", or None if not traded
    "trade_pnl",         # PnL if traded, None otherwise
    "trade_roi",         # ROI if traded, None otherwise
    "strategy_version",  # Version of strategy that generated this signal
]

# Fields for backtest trades with signal linking
BACKTEST_TRADE_FIELDS: List[str] = [
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
    "trade_id",          # Unique ID linking to signal
    "signal_reason",     # Signal type that triggered this trade
    "strategy_version",  # Version of strategy that generated this trade
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
    swing_left: int = 1
    swing_right: int = 1
    swing_resample_rule: str = "1d"
    swing_proximity_atr_mult: float = 0.25
    entry_limit_timeout_bars: int = 1
    leverage: float = 5.0  # fixed leverage for static sizing
    initial_capital: float = 5.0  # starting equity (margin cap per trade)
    margin_usd: float = 5.0  # static margin per trade

    # Costs
    fee_rate: float = 0.0000    # per side (0.04% typical maker/taker varies)
    slippage: float = 0.0000    # price impact fraction applied on entry/exit

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls - trailing stop only
    use_trailing_stop: bool = True
    trailing_mode: str = "dynamic_atr"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_initial_stop_r: int = -2  # Initial stop R level (-1 = no stop until price reaches trail_gap_r profit)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 0.75  # ATR multiplier for dynamic trailing
    dynamic_trail_activation_r: float = 0.5  # R threshold before placing stop (0=immediate, 1=wait for 1R move)
    dynamic_trail_price_source: str = "high_low"  # "close" or "high_low" (high for LONG, low for SHORT)
    trail_check_interval: str = "4h"  # interval for trailing stop updates (use 4h high/low)
    trail_exit_check_interval: str = "4h"  # interval for exit check (only exit on 4h candle close)
    forced_exit_interval: str = "1d"  # boundary used to count forced-close intervals
    forced_exit_interval_count: int = 3  # close after surviving this many forced intervals

    # Multi-position settings
    max_open_positions: int = 0  # max concurrent positions (<= 0 means unlimited)


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
    atr_history_bars: int = 365  # bars to pull for stable ATR/EMA (1d candles)
    leverage: int = 20  # fixed leverage for static sizing
    margin_usd: float = 5.0  # static margin per trade

    # Strategy thresholds
    thr1: float = 2.0
    thr2: float = 2.0

    # Risk/exit controls - trailing stop only (no TP/SL on entry)
    use_trailing_stop: bool = True
    trailing_mode: str = "dynamic_atr"  # "r_ladder" (current) or "dynamic_atr" (new)
    trail_initial_stop_r: int = -1  # Initial stop R level (-1 = no stop until price reaches trail_gap_r profit)
    trail_gap_r: float = 1.25
    trail_buffer_r: float = 0.10
    dynamic_trail_atr_mult: float = 0.75  # ATR multiplier for dynamic trailing
    dynamic_trail_activation_r: float = 0.5  # R threshold before placing stop (0=immediate, 1=wait for 1R move)
    dynamic_trail_price_source: str = "high_low"  # "close" or "high_low" (high for LONG, low for SHORT)
    trail_check_interval: str = "4h"  # interval for trailing stop updates (use 4h high/low)
    trail_exit_check_interval: str = "4h"  # interval for exit check (exit immediately on 1m close when stop hit)

    # Multi-position settings
    max_open_positions: int = 0  # max concurrent positions (<= 0 means unlimited)

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
    log_path: str = "output/trade_log.jsonl"
    live_trades_csv: str = "output/live_trades.csv"
    live_signals_csv: str = "output/live_signals.csv"
    live_swing_levels_file: str = "output/swing_levels_live.json"  # detected swing levels for live trading
    post_only: bool = True
    use_testnet: bool = False
