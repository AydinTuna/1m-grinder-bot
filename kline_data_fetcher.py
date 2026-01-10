#!/usr/bin/env python3
"""
binance_klines_fetch.py

Fetch Binance SPOT klines for a date range, using:
1) Binance Public Data (data.binance.vision) daily zipped klines, if available
2) Fallback to Spot REST API /api/v3/klines with pagination (limit=1000)

Outputs ONE combined CSV file.

Example (your case):
  python3 binance_klines_fetch.py --symbol BTCUSDC --interval 1s --start 2022-06-20 --end 2022-09-26 --out btcusdc_1s_20220620_20220926.csv

Dependencies:
  pip install requests
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests


PUBLIC_BASE = "https://data.binance.vision"
REST_BASE = "https://api.binance.com"


KLINE_HEADER = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def parse_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def daterange_inclusive(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def to_ms_utc(dttm: dt.datetime) -> int:
    if dttm.tzinfo is None:
        dttm = dttm.replace(tzinfo=dt.timezone.utc)
    else:
        dttm = dttm.astimezone(dt.timezone.utc)
    return int(dttm.timestamp() * 1000)


def interval_to_ms(interval: str) -> int:
    """
    Supports typical Binance intervals: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "s":
        return num * 1000
    if unit == "m":
        return num * 60_000
    if unit == "h":
        return num * 3_600_000
    if unit == "d":
        return num * 86_400_000
    if unit == "w":
        return num * 7 * 86_400_000
    if unit == "M":
        # month is variable; REST handles it, but for stepping we avoid it.
        raise ValueError("Monthly (1M) stepping is variable-length; use public data or implement month stepping.")
    raise ValueError(f"Unsupported interval: {interval}")


@dataclass
class FetchStats:
    public_days_ok: int = 0
    public_days_missing: int = 0
    rest_rows: int = 0


def public_daily_zip_url(symbol: str, interval: str, day: dt.date) -> str:
    # https://data.binance.vision/data/spot/daily/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-YYYY-MM-DD.zip
    return (
        f"{PUBLIC_BASE}/data/spot/daily/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{day.isoformat()}.zip"
    )


def try_fetch_public_day(symbol: str, interval: str, day: dt.date, session: requests.Session) -> Optional[List[List[str]]]:
    """
    Returns list of kline rows (as list of strings) if public zip exists, else None.
    """
    url = public_daily_zip_url(symbol, interval, day)
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    zdata = io.BytesIO(r.content)
    with zipfile.ZipFile(zdata) as zf:
        # Usually the zip contains a single CSV file
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"Zip from {url} had no CSV files inside.")
        # Use the first CSV
        with zf.open(csv_names[0], "r") as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            rows = [row for row in reader if row]  # keep non-empty
            return rows


def fetch_rest_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session,
    polite_sleep_s: float = 0.2,
) -> List[List[str]]:
    """
    Fetch klines from REST /api/v3/klines with pagination.
    Returns list of rows (each row is list[str]) using Binance response order.
    """
    out: List[List[str]] = []
    url = f"{REST_BASE}/api/v3/klines"

    limit = 1000  # REST max (Binance docs) :contentReference[oaicite:2]{index=2}
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        # Convert each kline array -> list[str]
        for k in data:
            out.append([str(x) for x in k])

        # Advance to next open time + interval
        last_open = int(data[-1][0])
        step = interval_to_ms(interval)
        next_cur = last_open + step
        if next_cur <= cur:
            # safety
            next_cur = cur + step
        cur = next_cur

        time.sleep(polite_sleep_s)

    return out


def normalize_and_write_csv(
    all_rows: Iterable[List[str]],
    out_path: str,
    include_header: bool = True,
) -> int:
    """
    Writes combined CSV with a standard header matching Binance kline array fields.
    Returns number of data rows written.
    """
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if include_header:
            w.writerow(KLINE_HEADER)

        for row in all_rows:
            # Expect 12 columns from Binance kline array
            if len(row) != 12:
                # Some public files might include extra/less columns; keep a strict guard.
                raise RuntimeError(f"Unexpected kline row length {len(row)} (expected 12). Row sample: {row[:5]}")
            w.writerow(row)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Fetch Binance SPOT klines (public data first, REST fallback) into one CSV.")
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDC")
    ap.add_argument("--interval", required=True, help="e.g. 1s, 1m, 5m")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC day boundary for public daily files)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True, help="Output CSV file path")
    ap.add_argument("--prefer-rest", action="store_true", help="Skip public data and use REST only.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between REST requests (default 0.2).")
    args = ap.parse_args()

    symbol = args.symbol.upper()
    interval = args.interval
    start_date = parse_ymd(args.start)
    end_date = parse_ymd(args.end)
    if end_date < start_date:
        raise SystemExit("end date must be >= start date")

    # We'll interpret start/end as full-day UTC range:
    # [start 00:00:00, end+1day 00:00:00)
    start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
    end_dt_excl = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc)
    start_ms = to_ms_utc(start_dt)
    end_ms = to_ms_utc(end_dt_excl)

    stats = FetchStats()
    session = requests.Session()

    combined_rows: List[List[str]] = []

    if not args.prefer_rest:
        # Try public daily zips day by day
        for day in daterange_inclusive(start_date, end_date):
            try:
                rows = try_fetch_public_day(symbol, interval, day, session)
            except requests.HTTPError as e:
                # If public data has an intermittent error, fallback to REST for that day
                print(f"[WARN] Public data error for {day}: {e}. Will fallback to REST for that day.", file=sys.stderr)
                rows = None

            if rows is None:
                stats.public_days_missing += 1
                # REST fallback for that day range
                day_start = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
                day_end_excl = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc)
                day_rows = fetch_rest_klines(
                    symbol=symbol,
                    interval=interval,
                    start_ms=to_ms_utc(day_start),
                    end_ms=to_ms_utc(day_end_excl),
                    session=session,
                    polite_sleep_s=args.sleep,
                )
                stats.rest_rows += len(day_rows)
                combined_rows.extend(day_rows)
            else:
                stats.public_days_ok += 1
                combined_rows.extend(rows)
    else:
        # REST-only for whole range
        rows = fetch_rest_klines(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            session=session,
            polite_sleep_s=args.sleep,
        )
        stats.rest_rows += len(rows)
        combined_rows.extend(rows)

    # Sort by open_time_ms just in case (public + rest mixed)
    combined_rows.sort(key=lambda r: int(r[0]))

    # Write single CSV
    written = normalize_and_write_csv(combined_rows, args.out, include_header=True)

    print(f"Done. Wrote {written:,} rows to {args.out}")
    print(f"Public days OK: {stats.public_days_ok}, public missing: {stats.public_days_missing}, REST rows: {stats.rest_rows:,}")


if __name__ == "__main__":
    main()
