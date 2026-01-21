"""
Batch Backtest Runner

Run multiple backtests with different parameter configurations from a JSON file
and generate a comparison report.

Usage:
    python batch_backtest.py --config backtest_params.json
    python batch_backtest.py --config backtest_params.json --start 2022-07-01 --end 2022-09-01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time as time_module
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import BacktestConfig, get_output_path
from main import (
    backtest_atr_grinder,
    backtest_atr_grinder_lib,
    fetch_klines_public_data,
)


def load_params(json_path: str) -> Dict[str, Any]:
    """
    Load and validate batch parameters from JSON file.
    
    Args:
        json_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing batch configuration
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON structure is invalid
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    
    # Validate required fields
    if "runs" not in params:
        raise ValueError("JSON config must contain 'runs' array")
    
    if not isinstance(params["runs"], list) or len(params["runs"]) == 0:
        raise ValueError("'runs' must be a non-empty array")
    
    # Set defaults for optional fields
    params.setdefault("start_date", "2022-07-11")
    params.setdefault("end_date", "2022-09-11")
    params.setdefault("symbol", "BTCUSDC")
    params.setdefault("market", "futures")
    params.setdefault("use_lib", True)
    
    # Validate each run has a name
    for i, run in enumerate(params["runs"]):
        if "name" not in run:
            run["name"] = f"Run {i + 1}"
    
    return params


def create_config_from_run(run_params: Dict[str, Any]) -> BacktestConfig:
    """
    Create a BacktestConfig instance with overridden parameters from run config.
    
    Args:
        run_params: Dictionary of parameters to override
        
    Returns:
        BacktestConfig with specified overrides
    """
    cfg = BacktestConfig()
    
    # Get valid field names from BacktestConfig
    valid_fields = {f.name for f in fields(BacktestConfig)}
    
    # Override config values from run parameters
    for key, value in run_params.items():
        if key == "name":
            continue  # Skip the name field
        if key in valid_fields:
            setattr(cfg, key, value)
        else:
            print(f"  Warning: Unknown parameter '{key}' ignored")
    
    return cfg


def run_single_backtest(
    cfg: BacktestConfig,
    df_signal: pd.DataFrame,
    df_1s: pd.DataFrame,
    use_lib: bool = True,
) -> Dict[str, Any]:
    """
    Execute a single backtest with the given configuration.
    
    Args:
        cfg: BacktestConfig instance
        df_signal: Signal interval DataFrame
        df_1s: 1-second DataFrame for LIB mode
        use_lib: Whether to use Look-Inside-Bar mode
        
    Returns:
        Dictionary of backtest statistics
    """
    if use_lib and not df_1s.empty:
        trades, df_bt, stats, trailing_df = backtest_atr_grinder_lib(df_signal, df_1s, cfg)
    else:
        trades, df_bt, stats, trailing_df = backtest_atr_grinder(df_signal, cfg)
    
    return stats


def print_results_table(results: List[Dict[str, Any]], params_to_show: List[str]) -> None:
    """
    Print results as a formatted terminal table.
    
    Args:
        results: List of result dictionaries
        params_to_show: List of parameter names to include in table
    """
    if not results:
        print("No results to display.")
        return
    
    # Define columns to show
    stat_columns = [
        ("trades", "Trades", "{:d}"),
        ("win_rate", "Win Rate", "{:.1%}"),
        ("total_pnl_net", "Total PnL", "${:.2f}"),
        ("final_equity", "Final Equity", "${:.2f}"),
        ("max_drawdown", "Max DD", "{:.1%}"),
        ("avg_roi_net", "Avg ROI", "{:.2%}"),
    ]
    
    # Build header
    header = ["Run Name"] + params_to_show + [col[1] for col in stat_columns]
    
    # Calculate column widths
    col_widths = [len(h) for h in header]
    
    rows = []
    for r in results:
        row = [r.get("name", "")]
        # Add parameter values
        for param in params_to_show:
            val = r.get(param, "-")
            row.append(str(val))
        # Add stat values
        for key, _, fmt in stat_columns:
            val = r.get(key, 0)
            if isinstance(val, (int, float)):
                try:
                    row.append(fmt.format(int(val) if fmt == "{:d}" else val))
                except (ValueError, TypeError):
                    row.append(str(val))
            else:
                row.append(str(val))
        rows.append(row)
        
        # Update column widths
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Print table
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    def format_row(row_data):
        return "|" + "|".join(f" {cell:>{col_widths[i]}} " for i, cell in enumerate(row_data)) + "|"
    
    print("\n" + separator)
    print(format_row(header))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def save_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save results to a CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to save.")
        return
    
    # Collect all unique keys
    all_keys = []
    for r in results:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    
    # Reorder: name first, then params, then stats
    priority_keys = ["name", "trailing_mode", "trail_initial_stop_r", "trail_gap_r", "trail_buffer_r", "margin_usd"]
    ordered_keys = []
    for k in priority_keys:
        if k in all_keys:
            ordered_keys.append(k)
            all_keys.remove(k)
    ordered_keys.extend(all_keys)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def run_batch(
    json_path: str,
    start_override: Optional[str] = None,
    end_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run batch backtests from JSON configuration.
    
    Args:
        json_path: Path to JSON configuration file
        start_override: Optional start date override (YYYY-MM-DD)
        end_override: Optional end date override (YYYY-MM-DD)
        
    Returns:
        List of result dictionaries for each run
    """
    total_start = time_module.perf_counter()
    
    # Load configuration
    print("=" * 70)
    print("BATCH BACKTEST RUNNER")
    print("=" * 70)
    print(f"\nLoading config: {json_path}")
    
    params = load_params(json_path)
    
    # Apply date overrides if provided
    start_date = start_override or params["start_date"]
    end_date = end_override or params["end_date"]
    symbol = params["symbol"]
    market = params["market"]
    use_lib = params.get("use_lib", True)
    runs = params["runs"]
    
    print(f"Symbol: {symbol}")
    print(f"Market: {market}")
    print(f"Period: {start_date} to {end_date}")
    print(f"LIB Mode: {'Enabled' if use_lib else 'Disabled'}")
    print(f"Number of runs: {len(runs)}")
    
    # Get signal interval from first run config (or default)
    first_cfg = create_config_from_run(runs[0])
    signal_interval = first_cfg.signal_interval
    
    # Fetch data once (shared across all runs)
    df_1s = pd.DataFrame()
    
    if use_lib:
        print(f"\n[1/3] Loading 1s candle data (spot - for LIB execution)...")
        fetch_1s_start = time_module.perf_counter()
        
        df_1s = fetch_klines_public_data(
            symbol=symbol,
            interval="1s",
            start_date=start_date,
            end_date=end_date,
            save_csv=True,
            force_refetch=False,
            market="spot",  # 1s only available on spot
        )
        
        fetch_1s_time = time_module.perf_counter() - fetch_1s_start
        print(f"Total 1s candles: {len(df_1s):,} ({fetch_1s_time:.2f}s)")
        
        if df_1s.empty:
            print("ERROR: No 1s data available. Cannot run backtest.")
            return []
    
    print(f"\n[2/3] Loading {signal_interval} candle data ({market})...")
    fetch_signal_start = time_module.perf_counter()
    
    df_signal = fetch_klines_public_data(
        symbol=symbol,
        interval=signal_interval,
        start_date=start_date,
        end_date=end_date,
        save_csv=True,
        force_refetch=False,
        market=market,
    )
    
    fetch_signal_time = time_module.perf_counter() - fetch_signal_start
    print(f"Total {signal_interval} candles: {len(df_signal):,} ({fetch_signal_time:.2f}s)")
    
    if df_signal.empty:
        print("ERROR: No signal data available. Cannot run backtest.")
        return []
    
    # Run backtests
    print(f"\n[3/3] Running {len(runs)} backtests...")
    print("-" * 70)
    
    results = []
    
    for i, run in enumerate(runs):
        run_name = run.get("name", f"Run {i + 1}")
        print(f"\n  [{i + 1}/{len(runs)}] {run_name}")
        
        # Create config with run-specific parameters
        cfg = create_config_from_run(run)
        
        # Show key parameters
        print(f"       trailing_mode: {cfg.trailing_mode}")
        print(f"       trail_initial_stop_r: {cfg.trail_initial_stop_r}")
        print(f"       trail_gap_r: {cfg.trail_gap_r}")
        
        # Run backtest
        run_start = time_module.perf_counter()
        stats = run_single_backtest(cfg, df_signal, df_1s, use_lib)
        run_time = time_module.perf_counter() - run_start
        
        # Build result dictionary
        result = {
            "name": run_name,
            "trailing_mode": cfg.trailing_mode,
            "trail_initial_stop_r": cfg.trail_initial_stop_r,
            "trail_gap_r": cfg.trail_gap_r,
            "trail_buffer_r": cfg.trail_buffer_r,
            "margin_usd": cfg.margin_usd,
        }
        result.update(stats)
        results.append(result)
        
        print(f"       -> Trades: {stats.get('trades', 0)}, "
              f"Win Rate: {stats.get('win_rate', 0):.1%}, "
              f"PnL: ${stats.get('total_pnl_net', 0):.2f} "
              f"({run_time:.2f}s)")
    
    total_time = time_module.perf_counter() - total_start
    
    # Print summary table
    print("\n" + "=" * 70)
    print("BATCH RESULTS SUMMARY")
    print("=" * 70)
    
    params_to_show = ["trail_initial_stop_r", "trail_gap_r", "trail_buffer_r"]
    print_results_table(results, params_to_show)
    
    # Save to CSV
    save_results_csv(results, get_output_path("batch_results.csv"))
    
    print(f"\nTotal time: {total_time:.2f}s")
    print("=" * 70)
    
    return results


def main() -> None:
    """CLI entry point for batch backtest runner."""
    parser = argparse.ArgumentParser(
        description="Run batch backtests with multiple parameter configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python batch_backtest.py --config backtest_params.json
  
  # Override date range
  python batch_backtest.py --config backtest_params.json --start 2022-06-01 --end 2022-12-31
  
JSON Config Format:
  {
    "start_date": "2025-10-01",
    "end_date": "2026-01-21",
    "symbol": "ARPAUSDT",
    "market": "futures",
    "use_lib": true,
    "runs": [
      {
        "name": "init_-1_gap_1.0",
        "trail_initial_stop_r": -1,
        "trail_gap_r": 1.0,
        "trail_buffer_r": 0.10
      },
      ...
    ]
  }
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override end date (YYYY-MM-DD)",
    )
    
    args = parser.parse_args()
    
    try:
        run_batch(
            json_path=args.config,
            start_override=args.start,
            end_override=args.end,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Config Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBatch run interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
