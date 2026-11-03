from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DMALA acceptance and proposal size against step size."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/ising/macbook/step_size_acceptance"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ising/macbook/figure2_left"),
        help="Output path without an extension; both PNG and PDF are written.",
    )
    return parser.parse_args()


def load_results(input_dir: Path) -> list[dict]:
    rows = [json.loads(path.read_text()) for path in sorted(input_dir.glob("dmala_alpha*.json"))]
    if not rows:
        raise FileNotFoundError(f"no DMALA step-size summaries found in {input_dir}")
    rows.sort(key=lambda row: float(row["step_size"]))
    step_sizes = [float(row["step_size"]) for row in rows]
    if len(step_sizes) != len(set(step_sizes)):
        raise ValueError(f"duplicate step sizes found: {step_sizes}")
    return rows


def plot(rows: list[dict], output: Path) -> None:
    step_sizes = np.asarray([row["step_size"] for row in rows], dtype=float)
    acceptance = np.asarray([row["mean_acceptance"] for row in rows], dtype=float)
    proposed_hops = np.asarray([row["mean_proposed_hops"] for row in rows], dtype=float)

    figure, acceptance_axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    hop_axis = acceptance_axis.twinx()

    acceptance_color = "#4C78A8"
    hop_color = "#F58518"
    acceptance_axis.plot(
        step_sizes,
        acceptance,
        color=acceptance_color,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Acceptance probability",
    )
    hop_axis.plot(
        step_sizes,
        proposed_hops,
        color=hop_color,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Proposed coordinate changes",
    )

    acceptance_axis.set_xlabel(r"Step size ($\alpha$)")
    acceptance_axis.set_ylabel("Acceptance probability", color=acceptance_color)
    hop_axis.set_ylabel("Number of proposed coordinate changes", color=hop_color)
    acceptance_axis.tick_params(axis="y", colors=acceptance_color)
    hop_axis.tick_params(axis="y", colors=hop_color)
    acceptance_axis.set_xticks(step_sizes)
    acceptance_axis.set_xlim(step_sizes.min() - 0.02, step_sizes.max() + 0.02)
    acceptance_axis.set_ylim(0.5, 1.0)
    hop_axis.set_ylim(0.0, max(6.5, proposed_hops.max() * 1.08))
    acceptance_axis.grid(alpha=0.22)

    final_step = len(step_sizes) - 1
    acceptance_axis.annotate(
        f"{acceptance[final_step]:.1%}",
        (step_sizes[final_step], acceptance[final_step]),
        xytext=(-8, 12),
        textcoords="offset points",
        color=acceptance_color,
        ha="right",
    )
    hop_axis.annotate(
        f"{proposed_hops[final_step]:.2f}",
        (step_sizes[final_step], proposed_hops[final_step]),
        xytext=(-8, -18),
        textcoords="offset points",
        color=hop_color,
        ha="right",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_results(args.input_dir)
    plot(rows, args.output)
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
