from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten_json(value: Any, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Flatten nested JSON objects so each record fits a CSV row."""
    items: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}{separator}{key}" if prefix else str(key)
            items.update(flatten_json(nested_value, nested_prefix, separator))
        return items

    if isinstance(value, list):
        if not value:
            items[prefix] = ""
            return items

        for index, nested_value in enumerate(value):
            nested_prefix = f"{prefix}{separator}{index}" if prefix else str(index)
            items.update(flatten_json(nested_value, nested_prefix, separator))
        return items

    items[prefix] = value
    return items


def collect_headers(input_path: Path) -> list[str]:
    """Scan the JSONL file once to discover every CSV column."""
    headers: set[str] = set()

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            flattened = flatten_json(record)
            headers.update(flattened.keys())

    return sorted(headers)


def convert_jsonl_to_csv(input_path: Path, output_path: Path) -> None:
    """Convert a JSONL file into CSV with dynamically discovered columns."""
    headers = collect_headers(input_path)

    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8", newline="") as outfile,
    ):
        writer = csv.DictWriter(outfile, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            writer.writerow(flatten_json(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file into CSV."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="trade_logs.jsonl",
        help="Path to the input JSONL file. Defaults to trade_logs.jsonl.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Path to the output CSV file. Defaults to the input name with .csv extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".csv")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    convert_jsonl_to_csv(input_path, output_path)
    print(f"CSV created: {output_path}")


if __name__ == "__main__":
    main()
