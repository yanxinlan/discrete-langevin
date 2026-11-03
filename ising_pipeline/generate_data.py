from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .common import choose_device, cyclic_gibbs_steps, make_lattice_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic lattice-Ising observations.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=25, help="Lattice side length; dimension is side squared.")
    parser.add_argument("--coupling", type=float, default=0.25)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument(
        "--gibbs-steps",
        type=int,
        default=1_000_000,
        help="Single-coordinate Gibbs transitions applied to every parallel sample.",
    )
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.gibbs_steps < 1:
        raise ValueError("num-samples and gibbs-steps must be positive")
    device = choose_device(args.device)
    seed_everything(args.seed)
    model = make_lattice_model(args.side, args.coupling, args.bias, device)
    # This matches utils.generate_data in the released repository: all 2,000
    # instances are independent chains, initialized from the model's zero-bias
    # Bernoulli distribution, then advanced for 1,000,000 coordinate updates.
    initial_probability = torch.sigmoid(2.0 * model.bias)
    states = torch.bernoulli(initial_probability.expand(args.num_samples, -1))

    started = time.perf_counter()
    progress_every = max(1, args.gibbs_steps // 10)
    samples = cyclic_gibbs_steps(
        states, model, args.gibbs_steps, progress_every=progress_every
    ).cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "side": args.side,
        "dimension": model.dimension,
        "coupling_strength": args.coupling,
        "bias_strength": args.bias,
        "num_samples": args.num_samples,
        "chains": args.num_samples,
        "gibbs_steps_per_chain": args.gibbs_steps,
        "scan_order": "cyclic coordinates 0..d-1 (released code)",
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
