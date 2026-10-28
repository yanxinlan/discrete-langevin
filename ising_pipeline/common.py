from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import IsingModel, lattice_adjacency


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lattice_model(side: int, coupling: float, bias: float, device: torch.device) -> IsingModel:
    matrix = coupling * lattice_adjacency(side, device=device)
    return IsingModel(matrix, bias).to(device)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_dataset(path: str | Path, device: torch.device) -> tuple[torch.Tensor, IsingModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    samples = payload["samples"].to(device=device, dtype=torch.float32)
    model = IsingModel(payload["coupling"], payload["bias"]).to(device)
    return samples, model, payload.get("metadata", {})


@torch.no_grad()
def gibbs_sweeps(x: torch.Tensor, model, sweeps: int) -> torch.Tensor:
    """Run sequential heat-bath sweeps, vectorized over independent chains.

    Local fields are updated incrementally. This is especially important for
    lattice targets: it avoids one dense matrix multiplication per coordinate.
    """
    spins = 2.0 * x - 1.0
    coupling = model.coupling.detach()
    bias = model.bias.detach()
    local_field = spins @ coupling
    neighbors = [torch.nonzero(coupling[row], as_tuple=False).flatten() for row in range(model.dimension)]
    for _ in range(sweeps):
        for coordinate_index in torch.randperm(model.dimension).tolist():
            old_spin = spins[:, coordinate_index]
            delta = -4.0 * old_spin * local_field[:, coordinate_index] - 2.0 * bias[coordinate_index] * old_spin
            flip = torch.bernoulli(torch.sigmoid(delta)).bool()
            spin_change = torch.where(flip, -2.0 * old_spin, torch.zeros_like(old_spin))
            spins[:, coordinate_index] += spin_change
            affected = neighbors[coordinate_index]
            if affected.numel() > 0:
                local_field[:, affected] += spin_change[:, None] * coupling[coordinate_index, affected][None, :]
    return (spins + 1.0) / 2.0


@torch.no_grad()
def exact_mean(model, chunk_size: int = 1_000_000) -> tuple[torch.Tensor, float]:
    """Compute the exact spin mean by streaming over the finite state space."""
    dimension = model.dimension
    if dimension > 30:
        raise ValueError("exact enumeration is restricted to at most 30 variables")
    total_states = 1 << dimension
    device = model.coupling.device
    bit_positions = torch.arange(dimension, device=device, dtype=torch.int64)
    global_log_z = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
    global_mean = torch.zeros(dimension, device=device, dtype=torch.float64)

    for start in range(0, total_states, chunk_size):
        stop = min(start + chunk_size, total_states)
        integers = torch.arange(start, stop, device=device, dtype=torch.int64)
        states = ((integers[:, None] >> bit_positions[None, :]) & 1).to(torch.float32)
        energies = model(states).to(torch.float64)
        chunk_log_z = torch.logsumexp(energies, dim=0)
        chunk_weights = torch.softmax(energies, dim=0)
        chunk_mean = (chunk_weights[:, None] * (2.0 * states - 1.0).to(torch.float64)).sum(dim=0)
        combined_log_z = torch.logaddexp(global_log_z, chunk_log_z)
        global_weight = torch.exp(global_log_z - combined_log_z)
        chunk_weight = torch.exp(chunk_log_z - combined_log_z)
        global_mean = global_weight * global_mean + chunk_weight * chunk_mean
        global_log_z = combined_log_z
    return global_mean.to(torch.float32), float(global_log_z.item())


def effective_sample_size(series: np.ndarray) -> float:
    """Initial-positive-sequence ESS for a scalar time series."""
    values = np.asarray(series, dtype=np.float64)
    values = values - values.mean()
    n = len(values)
    if n < 3 or np.allclose(values, 0.0):
        return float(n)
    transform = np.fft.rfft(values, n=2 * n)
    autocov = np.fft.irfft(transform * np.conjugate(transform), n=2 * n)[:n]
    autocov /= np.arange(n, 0, -1)
    rho = autocov / autocov[0]
    positive_sum = 0.0
    for lag in range(1, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        positive_sum += pair
    return float(min(n, n / (1.0 + 2.0 * positive_sum)))
