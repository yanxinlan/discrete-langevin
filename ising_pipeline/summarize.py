from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Ising experiment summaries into CSV files.")
    parser.add_argument("--results-root", type=Path, default=Path("results/ising"))
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    args = parse_args()
    sampling_rows = [json.loads(path.read_text()) for path in sorted((args.results_root / "sampling").glob("*.json"))]
    learning_rows: list[dict] = []
    for metrics_path in sorted((args.results_root / "learning").glob("*/metrics.jsonl")):
        lines = [line for line in metrics_path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        final = json.loads(lines[-1])
        parts = metrics_path.parent.name.split("_")
        final["run"] = metrics_path.parent.name
        final["sampler"] = parts[0]
        learning_rows.append(final)
    write_rows(args.results_root / "sampling_summary.csv", sampling_rows)
    write_rows(args.results_root / "learning_summary.csv", learning_rows)


if __name__ == "__main__":
    main()
