"""Visualize trailing stop levels from backtest data with OHLC candlesticks."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import requests
from datetime import datetime, timedelta
import time

from config import get_output_path


def load_data():
    """Load trailing stops and trades data from output folder."""
    trailing_stops = pd.read_csv(get_output_path("trailing_stops.csv"), parse_dates=["timestamp", "entry_time"])
    trades = pd.read_csv(get_output_path("trades.csv"), parse_dates=["entry_time", "exit_time"])
    return trailing_stops, trades


def fetch_ohlc_binance(symbol: str, start_time: datetime, end_time: datetime, 
                        interval: str = "1m", market: str = "futures") -> pd.DataFrame:
    """
    Fetch OHLC data from Binance API.
    
    Args:
        symbol: Trading pair (e.g., "ARPAUSDT")
        start_time: Start datetime
        end_time: End datetime
        interval: Kline interval (default "1m")
        market: "spot" or "futures"
    
    Returns:
        DataFrame with OHLC columns indexed by datetime
    """
    if market == "futures":
        base_url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        base_url = "https://api.binance.com/api/v3/klines"
    
    all_klines = []
    current_start = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1  # Next candle
            
            if len(klines) < 1000:
                break
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"Error fetching OHLC: {e}")
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def draw_candlesticks(ax, ohlc_df: pd.DataFrame, width_minutes: float = 0.6):
    """
    Draw candlestick chart on the given axis.
    
    Args:
        ax: Matplotlib axis
        ohlc_df: DataFrame with OHLC columns and datetime index
        width_minutes: Width of candle body in minutes
    """
    if ohlc_df.empty:
        return
    
    # Convert width from minutes to matplotlib date units
    width = width_minutes / (24 * 60)  # Convert minutes to days (matplotlib date unit)
    
    for idx, row in ohlc_df.iterrows():
        open_price = row["open"]
        close_price = row["close"]
        high_price = row["high"]
        low_price = row["low"]
        
        # Determine candle color
        if close_price >= open_price:
            color = "green"
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = "red"
            body_bottom = close_price
            body_height = open_price - close_price
        
        # Convert timestamp to matplotlib date number
        x = mdates.date2num(idx)
        
        # Draw wick (high-low line)
        ax.plot([x, x], [low_price, high_price], color=color, linewidth=0.8, alpha=0.8)
        
        # Draw body
        if body_height > 0:
            rect = Rectangle(
                (x - width/2, body_bottom), width, body_height,
                facecolor=color, edgecolor=color, alpha=0.8, linewidth=0.5
            )
            ax.add_patch(rect)
        else:
            # Doji - just a horizontal line
            ax.plot([x - width/2, x + width/2], [open_price, open_price], 
                    color=color, linewidth=1)


def plot_trailing_stops(trailing_stops: pd.DataFrame, trades: pd.DataFrame, symbol: str = None,
                         market: str = "futures"):
    """
    Plot trailing stop visualization for each trade with OHLC candlesticks.
    
    Shows:
    - 1m OHLC candlesticks
    - Entry price (green dashed horizontal)
    - ATR stop calculation (orange dotted - raw ATR-based stop)
    - Actual stop price (red line - after floor applied)
    - Exit point (red X marker)
    """
    if symbol:
        trailing_stops = trailing_stops[trailing_stops["symbol"] == symbol]
        trades = trades[trades["symbol"] == symbol]
    
    # Group by entry_time to separate different trades
    trade_groups = trailing_stops.groupby("entry_time")
    
    for entry_time, group in trade_groups:
        group = group.sort_values("timestamp")
        
        # Find corresponding trade
        trade = trades[trades["entry_time"] == entry_time]
        if trade.empty:
            continue
        trade = trade.iloc[0]
        
        # Fetch OHLC data for the trade period
        start_time = group["timestamp"].min() - timedelta(minutes=5)
        end_time = trade["exit_time"] + timedelta(minutes=5)
        
        print(f"Fetching OHLC data for {trade['symbol']} from {start_time} to {end_time}...")
        ohlc_df = fetch_ohlc_binance(trade["symbol"], start_time, end_time, 
                                      interval="1m", market=market)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Draw candlesticks
        if not ohlc_df.empty:
            draw_candlesticks(ax, ohlc_df, width_minutes=0.6)
            print(f"  Loaded {len(ohlc_df)} candles")
        else:
            # Fallback to close price line if OHLC fetch failed
            ax.plot(group["timestamp"], group["close_price"], 
                    label="Close Price", color="blue", linewidth=1.5, alpha=0.8)
            print("  OHLC fetch failed, using close price line")
        
        # Plot entry price (horizontal line)
        ax.axhline(y=trade["entry_price"], color="lime", linestyle="--", 
                   linewidth=2, label=f"Entry: {trade['entry_price']:.6f}", alpha=0.9)
        
        # Plot ATR stop (raw calculation before floor)
        ax.plot(group["timestamp"], group["atr_stop"], 
                label="ATR Stop (raw)", color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
        
        # Plot actual stop price (after floor applied)
        ax.plot(group["timestamp"], group["new_stop_price"], 
                label="Trailing Stop", color="magenta", linewidth=2.5, alpha=0.95)
        
        # Extend last trailing stop level to exit time for easier analysis
        last_ts = group["timestamp"].iloc[-1]
        last_stop = group["new_stop_price"].iloc[-1]
        if trade["exit_time"] > last_ts:
            ax.plot([last_ts, trade["exit_time"]], [last_stop, last_stop],
                    color="magenta", linewidth=2.5, alpha=0.5, linestyle="--")
        
        # Plot floor stop (breakeven) - only if column exists (not used in dynamic_atr mode)
        if "floor_stop" in group.columns and pd.notna(group["floor_stop"].iloc[0]):
            ax.axhline(y=group["floor_stop"].iloc[0], color="cyan", linestyle="-.", 
                       linewidth=1.5, label=f"Floor (BE): {group['floor_stop'].iloc[0]:.6f}", alpha=0.7)
        
        # Mark stop movement points
        stop_moved = group[group["stop_moved"] == True]
        if not stop_moved.empty:
            ax.scatter(stop_moved["timestamp"], stop_moved["new_stop_price"],
                       color="yellow", marker="o", s=50, zorder=6, alpha=0.9,
                       edgecolors="black", linewidths=0.5,
                       label=f"Stop Moved ({len(stop_moved)}x)")
        
        # Mark exit point
        ax.scatter([trade["exit_time"]], [trade["exit_price"]], 
                   color="red", marker="X", s=250, zorder=7, 
                   edgecolors="white", linewidths=1,
                   label=f"Exit: {trade['exit_price']:.6f} ({trade['exit_reason']})")
        
        # Mark entry point
        ax.scatter([group["timestamp"].iloc[0]], [trade["entry_price"]], 
                   color="lime", marker="^", s=200, zorder=7, 
                   edgecolors="black", linewidths=1,
                   label="Entry Point")
        
        # Formatting
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(
            f"{trade['symbol']} - {trade['side']} Trade\n"
            f"Entry: {trade['entry_price']:.6f} | Exit: {trade['exit_price']:.6f} | "
            f"PnL: ${trade['pnl_net']:.2f} ({trade['roi_net']*100:.1f}%)",
            fontsize=14
        )
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add info box
        stop_moved_count = len(group[group["stop_moved"] == True])
        info_text = (
            f"Signal ATR: {trade['signal_atr']:.6f}\n"
            f"Bars Held: {trade['bars_held']}\n"
            f"Stop Updates: {len(group)}\n"
            f"Stop Moved: {stop_moved_count}x"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure to output folder
        filename = f"trailing_stop_{trade['symbol']}_{trade['side']}_{entry_time.strftime('%Y%m%d')}.png"
        plt.savefig(get_output_path(filename), dpi=150, bbox_inches='tight')
        print(f"Saved: output/{filename}")
        
        plt.show()


def plot_stop_levels_detail(trailing_stops: pd.DataFrame, trades: pd.DataFrame, symbol: str = None,
                            market: str = "futures"):
    """
    Detailed plot showing stop level changes over time with OHLC candlesticks.
    
    Single overlapped view showing:
    - 1m OHLC candlesticks
    - Entry price, ATR stop, trailing stop
    - Stop movement markers
    """
    if symbol:
        trailing_stops = trailing_stops[trailing_stops["symbol"] == symbol]
        trades = trades[trades["symbol"] == symbol]
    
    trade_groups = trailing_stops.groupby("entry_time")
    
    for entry_time, group in trade_groups:
        group = group.sort_values("timestamp")
        
        trade = trades[trades["entry_time"] == entry_time]
        if trade.empty:
            continue
        trade = trade.iloc[0]
        
        # Fetch OHLC data for the trade period
        start_time = group["timestamp"].min() - timedelta(minutes=5)
        end_time = trade["exit_time"] + timedelta(minutes=5)
        
        print(f"Fetching OHLC data for {trade['symbol']} (detail) from {start_time} to {end_time}...")
        ohlc_df = fetch_ohlc_binance(trade["symbol"], start_time, end_time, 
                                      interval="1m", market=market)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Draw candlesticks
        if not ohlc_df.empty:
            draw_candlesticks(ax, ohlc_df, width_minutes=0.6)
            print(f"  Loaded {len(ohlc_df)} candles")
        else:
            # Fallback to close price line if OHLC fetch failed
            ax.plot(group["timestamp"], group["close_price"], 
                    label="Close Price", color="blue", linewidth=1.5, alpha=0.9)
            print("  OHLC fetch failed, using close price line")
        
        # Plot ATR stop (raw calculation before floor)
        ax.plot(group["timestamp"], group["atr_stop"], 
                label="ATR Stop (raw)", color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
        
        # Plot trailing stop
        ax.plot(group["timestamp"], group["new_stop_price"], 
                label="Trailing Stop", color="magenta", linewidth=2.5, alpha=0.95)
        
        # Extend last trailing stop level to exit time for easier analysis
        last_ts = group["timestamp"].iloc[-1]
        last_stop = group["new_stop_price"].iloc[-1]
        if trade["exit_time"] > last_ts:
            ax.plot([last_ts, trade["exit_time"]], [last_stop, last_stop],
                    color="magenta", linewidth=2.5, alpha=0.5, linestyle="--")
        
        # Plot entry price
        ax.axhline(y=trade["entry_price"], color="lime", linestyle="--", 
                   linewidth=2, label=f"Entry: {trade['entry_price']:.6f}", alpha=0.9)
        
        # Plot floor stop (breakeven) - only if column exists (not used in dynamic_atr mode)
        if "floor_stop" in group.columns and pd.notna(group["floor_stop"].iloc[0]):
            ax.axhline(y=group["floor_stop"].iloc[0], color="cyan", linestyle="-.", 
                       linewidth=1.5, label=f"Floor (BE): {group['floor_stop'].iloc[0]:.6f}", alpha=0.7)
        
        # Mark stop movement points
        stop_moved = group[group["stop_moved"] == True]
        if not stop_moved.empty:
            ax.scatter(stop_moved["timestamp"], stop_moved["new_stop_price"],
                       color="yellow", marker="o", s=50, zorder=6, alpha=0.9,
                       edgecolors="black", linewidths=0.5,
                       label=f"Stop Moved ({len(stop_moved)}x)")
        
        # Mark exit point
        ax.scatter([trade["exit_time"]], [trade["exit_price"]], 
                   color="red", marker="X", s=250, zorder=7,
                   edgecolors="white", linewidths=1,
                   label=f"Exit: {trade['exit_price']:.6f} ({trade['exit_reason']})")
        
        # Mark entry point
        ax.scatter([group["timestamp"].iloc[0]], [trade["entry_price"]], 
                   color="lime", marker="^", s=200, zorder=7,
                   edgecolors="black", linewidths=1,
                   label="Entry Point")
        
        # Formatting
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(
            f"{trade['symbol']} - {trade['side']} Trade\n"
            f"Entry: {trade['entry_price']:.6f} | Exit: {trade['exit_price']:.6f} | "
            f"PnL: ${trade['pnl_net']:.2f} ({trade['roi_net']*100:.1f}%)",
            fontsize=14
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add info box
        info_text = (
            f"Signal ATR: {trade['signal_atr']:.6f}\n"
            f"Bars Held: {trade['bars_held']}\n"
            f"Stop Updates: {len(group)}\n"
            f"Stop Moved: {len(stop_moved)}x"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure to output folder
        filename = f"trailing_stop_detail_{trade['symbol']}_{trade['side']}_{entry_time.strftime('%Y%m%d')}.png"
        plt.savefig(get_output_path(filename), dpi=150, bbox_inches='tight')
        print(f"Saved: output/{filename}")
        
        plt.show()


if __name__ == "__main__":
    print("Loading data...")
    trailing_stops, trades = load_data()
    
    print(f"Found {len(trades)} trades")
    print(f"Found {len(trailing_stops)} trailing stop updates")
    print()
    
    # Show trade summary
    print("Trade Summary:")
    print(trades[["symbol", "side", "entry_price", "exit_price", "pnl_net", "roi_net", "exit_reason"]].to_string())
    print()
    
    # Plot each trade
    print("\nGenerating visualizations...")
    plot_trailing_stops(trailing_stops, trades)
    
    print("\nGenerating detailed plots...")
    plot_stop_levels_detail(trailing_stops, trades)
