from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .common import choose_device, gibbs_sweeps, make_lattice_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic lattice-Ising observations.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=25, help="Lattice side length; dimension is side squared.")
    parser.add_argument("--coupling", type=float, default=0.25)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=256, help="Parallel Gibbs chains used for generation.")
    parser.add_argument("--burn-in-sweeps", type=int, default=1000)
    parser.add_argument("--sweeps-between", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.chains < 1:
        raise ValueError("num-samples and chains must be positive")
    device = choose_device(args.device)
    seed_everything(args.seed)
    model = make_lattice_model(args.side, args.coupling, args.bias, device)
    states = torch.bernoulli(torch.full((args.chains, model.dimension), 0.5, device=device))

    started = time.perf_counter()
    states = gibbs_sweeps(states, model, args.burn_in_sweeps)
    collected: list[torch.Tensor] = []
    while sum(batch.shape[0] for batch in collected) < args.num_samples:
        states = gibbs_sweeps(states, model, args.sweeps_between)
        remaining = args.num_samples - sum(batch.shape[0] for batch in collected)
        collected.append(states[:remaining].cpu().clone())
    samples = torch.cat(collected, dim=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "side": args.side,
        "dimension": model.dimension,
        "coupling_strength": args.coupling,
        "bias_strength": args.bias,
        "num_samples": args.num_samples,
        "chains": args.chains,
        "burn_in_sweeps": args.burn_in_sweeps,
        "sweeps_between": args.sweeps_between,
        "seed": args.seed,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "energy_convention": "U(x)=s^T J s+b^T s, s=2x-1; symmetric J counts edges twice",
    }
    torch.save(
        {
            "samples": samples,
            "coupling": model.coupling.cpu(),
            "bias": model.bias.cpu(),
            "metadata": metadata,
        },
        args.output,
    )
    print(f"saved {samples.shape[0]} samples of dimension {samples.shape[1]} to {args.output}")
    print(metadata)


if __name__ == "__main__":
    main()
