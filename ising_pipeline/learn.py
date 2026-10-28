from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import choose_device, load_dataset, seed_everything
from .model import LearnableIsingModel
from .samplers import build_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn an Ising coupling matrix with persistent contrastive divergence.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampler", choices=("gibbs", "gwg", "dula", "dmala"), required=True)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-2)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def default_step_size(sampler: str) -> float:
    return 0.1 if sampler == "dula" else 0.2 if sampler == "dmala" else 1.0


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)
    data, ground_truth, data_metadata = load_dataset(args.data, device)
    model = LearnableIsingModel(ground_truth.dimension).to(device)
    step_size = args.step_size if args.step_size is not None else default_step_size(args.sampler)
    sampler = build_sampler(args.sampler, step_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    buffer = torch.bernoulli(torch.full((args.buffer_size, model.dimension), 0.5, device=device))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    with metrics_path.open("w") as metrics_file:
        for iteration in range(1, args.iterations + 1):
            indices = torch.randint(data.shape[0], (args.batch_size,), generator=generator)
            positive = data[indices.to(data.device)]
            for _ in range(args.sampling_steps):
                buffer = sampler.step(buffer, model)

            positive_energy = model(positive).mean()
            negative_energy = model(buffer.detach()).mean()
            objective = positive_energy - negative_energy
            loss = -objective + args.l1 * model.coupling.abs().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iteration == 1 or iteration % args.record_every == 0 or iteration == args.iterations:
                with torch.no_grad():
                    rmse = torch.sqrt(torch.mean((model.coupling - ground_truth.coupling) ** 2))
                    record = {
                        "iteration": iteration,
                        "seconds": time.perf_counter() - started,
                        "loss": float(loss.item()),
                        "objective": float(objective.item()),
                        "rmse": float(rmse.item()),
                        "log_rmse": float(torch.log(rmse.clamp_min(1e-30)).item()),
                        "acceptance": sampler.last_acceptance,
                        "proposed_hops": sampler.last_proposed_hops,
                        "realized_hops": sampler.last_hops,
                    }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(record)

    checkpoint = {
        "model_state": model.state_dict(),
        "buffer": buffer.cpu(),
        "ground_truth_coupling": ground_truth.coupling.cpu(),
        "ground_truth_bias": ground_truth.bias.cpu(),
        "data_metadata": data_metadata,
        "config": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir), "step_size": step_size},
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
