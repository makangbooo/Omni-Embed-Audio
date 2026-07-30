#!/usr/bin/env python3
"""Write an immutable first-N-row Clotho caption CSV for a smoke run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    return parser.parse_args()


def select_rows(input_path: Path, output_path: Path, count: int) -> int:
    if count <= 0:
        raise ValueError("rows must be positive")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite smoke CSV: {output_path}")
    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Clotho CSV has no header")
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Clotho CSV contains only {len(rows)} rows")
    required = {"file_name", *(f"caption_{index}" for index in range(1, 6))}
    if not required <= set(reader.fieldnames):
        raise ValueError("Clotho CSV lacks the fixed filename/caption columns")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    count = select_rows(args.input.resolve(), args.output.resolve(), args.rows)
    print(f"SMOKE_CSV_ROWS={count}")
    print(f"OUTPUT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
