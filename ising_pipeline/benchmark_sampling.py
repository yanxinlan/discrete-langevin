from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .common import (
    choose_device,
    effective_sample_size,
    exact_mean,
    gibbs_sweeps,
    make_lattice_model,
    save_json,
    seed_everything,
)
from .samplers import build_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a sampler on a lattice Ising target.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--sampler", choices=("gibbs", "gwg", "dula", "dmala"), required=True)
    parser.add_argument("--side", type=int, default=5)
    parser.add_argument("--coupling", type=float, default=0.1)
    parser.add_argument("--bias", type=float, default=0.2)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--reference", choices=("exact", "gibbs"), default="exact")
    parser.add_argument("--reference-chains", type=int, default=1024)
    parser.add_argument("--reference-burn-in-sweeps", type=int, default=2000)
    parser.add_argument("--reference-sweeps", type=int, default=200)
    parser.add_argument("--exact-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def default_step_size(sampler: str) -> float:
    return 0.2 if sampler == "dula" else 0.4 if sampler == "dmala" else 1.0


@torch.no_grad()
def reference_mean(args: argparse.Namespace, model) -> tuple[torch.Tensor, float | None]:
    if args.reference == "exact":
        return exact_mean(model, args.exact_chunk_size)
    states = torch.bernoulli(
        torch.full((args.reference_chains, model.dimension), 0.5, device=model.coupling.device)
    )
    states = gibbs_sweeps(states, model, args.reference_burn_in_sweeps)
    spin_sum = torch.zeros(model.dimension, device=states.device)
    for _ in range(args.reference_sweeps):
        states = gibbs_sweeps(states, model, 1)
        spin_sum += (2.0 * states - 1.0).mean(dim=0)
    return spin_sum / args.reference_sweeps, None


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)
    model = make_lattice_model(args.side, args.coupling, args.bias, device)
    truth, log_z = reference_mean(args, model)
    step_size = args.step_size if args.step_size is not None else default_step_size(args.sampler)
    sampler = build_sampler(args.sampler, step_size)
    state = torch.bernoulli(torch.full((args.chains, model.dimension), 0.5, device=device))

    recorded_steps: list[int] = []
    elapsed: list[float] = []
    log_rmse: list[float] = []
    acceptance: list[float] = []
    proposed_hops: list[float] = []
    realized_hops: list[float] = []
    magnetization: list[float] = []
    spin_sum = torch.zeros(model.dimension, device=device)
    observations = 0
    interval_acceptance = 0.0
    interval_proposed = 0.0
    interval_realized = 0.0
    interval_count = 0
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        state = sampler.step(state, model)
        spin_sum += (2.0 * state - 1.0).sum(dim=0)
        observations += state.shape[0]
        interval_acceptance += sampler.last_acceptance
        interval_proposed += sampler.last_proposed_hops
        interval_realized += sampler.last_hops
        interval_count += 1
        if step == 1 or step % args.record_every == 0 or step == args.steps:
            estimate = spin_sum / observations
            rmse = torch.sqrt(torch.mean((estimate - truth) ** 2)).clamp_min(1e-30)
            recorded_steps.append(step)
            elapsed.append(time.perf_counter() - started)
            log_rmse.append(float(torch.log(rmse).item()))
            acceptance.append(interval_acceptance / interval_count)
            proposed_hops.append(interval_proposed / interval_count)
            realized_hops.append(interval_realized / interval_count)
            magnetization.append(float((2.0 * state - 1.0).mean().item()))
            interval_acceptance = interval_proposed = interval_realized = 0.0
            interval_count = 0

    magnetization_ess = effective_sample_size(np.asarray(magnetization)) * args.chains
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        step=np.asarray(recorded_steps),
        seconds=np.asarray(elapsed),
        log_rmse=np.asarray(log_rmse),
        acceptance=np.asarray(acceptance),
        proposed_hops=np.asarray(proposed_hops),
        realized_hops=np.asarray(realized_hops),
        magnetization=np.asarray(magnetization),
        reference_mean=truth.cpu().numpy(),
    )
    summary = {
        "sampler": args.sampler,
        "step_size": step_size,
        "seed": args.seed,
        "side": args.side,
        "dimension": model.dimension,
        "coupling": args.coupling,
        "bias": args.bias,
        "steps": args.steps,
        "chains": args.chains,
        "reference": args.reference,
        "log_partition": log_z,
        "final_log_rmse": log_rmse[-1],
        "mean_acceptance": float(np.mean(acceptance)),
        "mean_proposed_hops": float(np.mean(proposed_hops)),
        "mean_realized_hops": float(np.mean(realized_hops)),
        "magnetization_ess": magnetization_ess,
        "seconds": elapsed[-1],
        "device": str(device),
    }
    save_json(args.output.with_suffix(".json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
