from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"gibbs": "#4C78A8", "gwg": "#F58518", "dula": "#54A24B", "dmala": "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated Ising sampling and learning curves.")
    parser.add_argument("--results-root", type=Path, default=Path("results/ising"))
    return parser.parse_args()


def sampling_runs(root: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    runs: dict[str, list[dict[str, np.ndarray]]] = {}
    for path in sorted((root / "sampling").glob("*.npz")):
        sampler = path.stem.split("_")[0]
        with np.load(path) as values:
            runs.setdefault(sampler, []).append({key: values[key] for key in values.files})
    return runs


def learning_runs(root: Path) -> dict[str, list[list[dict]]]:
    runs: dict[str, list[list[dict]]] = {}
    for path in sorted((root / "learning").glob("*/metrics.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if records:
            sampler = path.parent.name.split("_")[0]
            runs.setdefault(sampler, []).append(records)
    return runs


def plot_sampling(root: Path) -> None:
    runs = sampling_runs(root)
    if not runs:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for sampler, sampler_runs in sorted(runs.items()):
        steps = sampler_runs[0]["step"]
        curves = np.stack([run["log_rmse"] for run in sampler_runs])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = COLORS.get(sampler)
        axes[0].plot(steps, mean, label=sampler.upper(), color=color)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)
        for run in sampler_runs:
            axes[1].plot(run["seconds"], run["log_rmse"], color=color, alpha=0.35)
    axes[0].set(xlabel="Sampler iterations", ylabel="log RMSE", title="Ising sampling")
    axes[1].set(xlabel="Runtime (seconds)", ylabel="log RMSE", title="Runtime comparison")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(root / "sampling_curves.pdf")
    figure.savefig(root / "sampling_curves.png", dpi=180)
    plt.close(figure)


def plot_learning(root: Path) -> None:
    runs = learning_runs(root)
    if not runs:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for sampler, sampler_runs in sorted(runs.items()):
        color = COLORS.get(sampler)
        for index, records in enumerate(sampler_runs):
            label = sampler.upper() if index == 0 else None
            axes[0].plot(
                [row["iteration"] for row in records],
                [row["log_rmse"] for row in records],
                color=color,
                alpha=0.65,
                label=label,
            )
            axes[1].plot(
                [row["seconds"] for row in records],
                [row["log_rmse"] for row in records],
                color=color,
                alpha=0.65,
            )
    axes[0].set(xlabel="Training iterations", ylabel="log coupling RMSE", title="Ising learning")
    axes[1].set(xlabel="Runtime (seconds)", ylabel="log coupling RMSE", title="Runtime comparison")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(root / "learning_curves.pdf")
    figure.savefig(root / "learning_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    plot_sampling(args.results_root)
    plot_learning(args.results_root)


if __name__ == "__main__":
    main()
