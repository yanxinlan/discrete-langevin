#!/usr/bin/env python3
"""Collect final AIS test likelihoods for the four requested Table 1 runs."""

import argparse
import csv
import re
from pathlib import Path


TEST_LL = re.compile(r"EMA Test log-likelihood:\s*([-+0-9.eE]+)")
PAPER_VALUES = {
    ("static_mnist", "dula"): -80.71,
    ("static_mnist", "dmala"): -79.46,
    ("omniglot", "dula"): -145.68,
    ("omniglot", "dmala"): -91.11,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/deep_ebm"))
    parser.add_argument("--output", type=Path, default=Path("results/deep_ebm/summary.csv"))
    args = parser.parse_args()

    rows = []
    for (dataset, sampler), paper_value in PAPER_VALUES.items():
        log_path = args.results_dir / dataset / sampler / "ais_log.txt"
        reproduced = None
        if log_path.exists():
            matches = TEST_LL.findall(log_path.read_text())
            if matches:
                reproduced = float(matches[-1])
        rows.append({
            "dataset": dataset,
            "sampler": sampler,
            "paper_test_log_likelihood": paper_value,
            "reproduced_test_log_likelihood": reproduced,
            "difference": None if reproduced is None else reproduced - paper_value,
            "ais_log": str(log_path),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        value = row["reproduced_test_log_likelihood"]
        result = "not finished" if value is None else f"{value:.2f} (delta {row['difference']:+.2f})"
        print(f"{row['dataset']:14} {row['sampler']:5} paper={row['paper_test_log_likelihood']:7.2f} reproduced={result}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
