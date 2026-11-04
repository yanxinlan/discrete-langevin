from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .plot_results import COLORS, LABELS, sampler_from_run_name


ORDER = ("gibbs-1", "gwg-1", "dula", "dmala")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregate Ising learning and coupling-recovery diagnostics."
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/ising/figure5_learning")
    )
    return parser.parse_args()


def load_metrics(root: Path) -> dict[str, list[list[dict]]]:
    runs: dict[str, list[list[dict]]] = {}
    for path in sorted((root / "learning").glob("*/metrics.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if records:
            sampler = sampler_from_run_name(path.parent.name)
            runs.setdefault(sampler, []).append(records)
    return runs


def plot_aggregate(root: Path, runs: dict[str, list[list[dict]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for sampler in ORDER:
        sampler_runs = runs.get(sampler, [])
        if not sampler_runs:
            continue
        iterations = np.asarray([row["iteration"] for row in sampler_runs[0]])
        log_rmse = np.asarray(
            [[row["log_rmse"] for row in records] for records in sampler_runs]
        )
        seconds = np.asarray(
            [[row["seconds"] for row in records] for records in sampler_runs]
        )
        mean = log_rmse.mean(axis=0)
        std = log_rmse.std(axis=0)
        mean_seconds = seconds.mean(axis=0)
        color = COLORS[sampler]
        label = LABELS[sampler]
        axes[0].plot(iterations, mean, color=color, label=label, linewidth=2)
        axes[0].fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)
        axes[1].plot(mean_seconds, mean, color=color, label=label, linewidth=2)
        axes[1].fill_between(mean_seconds, mean - std, mean + std, color=color, alpha=0.2)

    axes[0].set(
        xlabel="Training iterations",
        ylabel="log coupling RMSE (lower is better)",
        title="Parameter error vs training",
    )
    axes[1].set(
        xlabel="Runtime (seconds)",
        ylabel="log coupling RMSE (lower is better)",
        title="Parameter error vs runtime",
    )
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(root / "learning_aggregate.pdf")
    figure.savefig(root / "learning_aggregate.png", dpi=200)
    plt.close(figure)


def checkpoint_diagnostics(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "learning").glob("*/checkpoint.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state"]
        raw = state["raw_coupling"]
        learned = 0.5 * (raw + raw.T) * state["off_diagonal"]
        truth = checkpoint["ground_truth_coupling"]
        edge_mask = truth != 0
        nonedge_mask = (truth == 0) & state["off_diagonal"].bool()
        rows.append(
            {
                "run": path.parent.name,
                "sampler": sampler_from_run_name(path.parent.name),
                "all_rmse": float(torch.mean((learned - truth) ** 2).sqrt()),
                "edge_rmse": float(torch.mean((learned[edge_mask] - truth[edge_mask]) ** 2).sqrt()),
                "nonedge_rmse": float(torch.mean(learned[nonedge_mask] ** 2).sqrt()),
                "learned_edge_mean": float(learned[edge_mask].mean()),
                "learned_edge_std": float(learned[edge_mask].std()),
                "nonedge_mean_absolute": float(learned[nonedge_mask].abs().mean()),
                "maximum_false_edge": float(learned[nonedge_mask].abs().max()),
                "edge_fraction": float(edge_mask.float().mean()),
            }
        )
    return rows


def write_diagnostics(root: Path, rows: list[dict]) -> None:
    with (root / "coupling_recovery_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_recovery(root: Path, rows: list[dict]) -> None:
    samplers = [sampler for sampler in ORDER if any(row["sampler"] == sampler for row in rows)]
    labels = [LABELS[sampler] for sampler in samplers]
    x = np.arange(len(samplers))

    def values(key: str) -> tuple[np.ndarray, np.ndarray]:
        groups = [np.asarray([row[key] for row in rows if row["sampler"] == sampler]) for sampler in samplers]
        return np.asarray([group.mean() for group in groups]), np.asarray([group.std() for group in groups])

    edge_mean, edge_mean_sd = values("learned_edge_mean")
    edge_rmse, edge_rmse_sd = values("edge_rmse")
    nonedge_rmse, nonedge_rmse_sd = values("nonedge_rmse")
    colors = [COLORS[sampler] for sampler in samplers]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(x, edge_mean, yerr=edge_mean_sd, color=colors, alpha=0.85, capsize=4)
    axes[0].axhline(0.25, color="black", linestyle="--", linewidth=1.5, label="True coupling = 0.25")
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Mean learned coupling on true edges",
        title="Recovery of lattice edges",
        ylim=(0, 0.4),
    )
    axes[0].legend(frameon=False)

    width = 0.36
    axes[1].bar(x - width / 2, edge_rmse, width, yerr=edge_rmse_sd, label="True-edge RMSE", alpha=0.85)
    axes[1].bar(x + width / 2, nonedge_rmse, width, yerr=nonedge_rmse_sd, label="Non-edge RMSE", alpha=0.85)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        yscale="log",
        ylabel="RMSE (log scale; lower is better)",
        title="Sparse coupling diagnostics",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(root / "coupling_recovery.pdf")
    figure.savefig(root / "coupling_recovery.png", dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs = load_metrics(args.results_root)
    if not runs:
        raise RuntimeError(f"No metrics.jsonl files found under {args.results_root / 'learning'}")
    diagnostics = checkpoint_diagnostics(args.results_root)
    if not diagnostics:
        raise RuntimeError(f"No checkpoint.pt files found under {args.results_root / 'learning'}")
    plot_aggregate(args.results_root, runs)
    write_diagnostics(args.results_root, diagnostics)
    plot_recovery(args.results_root, diagnostics)
    print(f"Wrote aggregate learning and recovery figures to {args.results_root}")


if __name__ == "__main__":
    main()
