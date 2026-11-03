from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F


def _bernoulli_log_prob(value: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")


class Sampler:
    name = "sampler"

    def __init__(self) -> None:
        self.last_acceptance = 1.0
        self.last_proposed_hops = 0.0
        self.last_hops = 0.0

    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        raise NotImplementedError


class GibbsSampler(Sampler):
    """One-coordinate heat-bath Gibbs kernel with shuffled sweep order."""

    name = "gibbs-1"

    def __init__(self) -> None:
        super().__init__()
        self._coordinates: list[int] = []

    def _next_coordinate(self, dimension: int, device: torch.device) -> int:
        del device
        if not self._coordinates:
            self._coordinates = list(range(dimension))
            np.random.shuffle(self._coordinates)
        return self._coordinates.pop(0)

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        coordinate_index = self._next_coordinate(model.dimension, x.device)
        spins = 2.0 * x - 1.0
        old_spin = spins[:, coordinate_index]
        local_field = spins @ model.coupling[:, coordinate_index]
        delta = -4.0 * old_spin * local_field - 2.0 * model.bias[coordinate_index] * old_spin
        flip = torch.bernoulli(torch.sigmoid(delta))
        result = x.clone()
        result[:, coordinate_index] = torch.where(
            flip.bool(), 1.0 - x[:, coordinate_index], x[:, coordinate_index]
        )
        self.last_acceptance = 1.0
        self.last_proposed_hops = float(flip.mean().item())
        self.last_hops = self.last_proposed_hops
        return result


class BlockGibbsSampler(Sampler):
    """Exact block Gibbs used for the released code's Gibbs-4 baseline."""

    def __init__(self, block_size: int = 4) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)
        self.name = f"gibbs-{self.block_size}"
        self._coordinates: list[int] = []

    def _next_block(self, dimension: int, device: torch.device) -> list[int]:
        del device
        if not self._coordinates:
            self._coordinates = list(range(dimension))
            np.random.shuffle(self._coordinates)
        block = self._coordinates[: self.block_size]
        self._coordinates = self._coordinates[self.block_size :]
        return block

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        coordinates = self._next_block(model.dimension, x.device)
        choices = torch.tensor(
            list(itertools.product((0.0, 1.0), repeat=len(coordinates))),
            dtype=x.dtype,
            device=x.device,
        )
        candidates = x[:, None, :].expand(-1, choices.shape[0], -1).clone()
        candidates[:, :, coordinates] = choices[None, :, :]
        logits = model(candidates.reshape(-1, model.dimension)).reshape(x.shape[0], -1)
        selected = torch.distributions.Categorical(logits=logits).sample()
        rows = torch.arange(x.shape[0], device=x.device)
        result = candidates[rows, selected]
        hops = (result != x).sum(dim=-1).float()
        self.last_acceptance = 1.0
        self.last_proposed_hops = float(hops.mean().item())
        self.last_hops = self.last_proposed_hops
        return result


class GWGSampler(Sampler):
    """Single-flip Gibbs-with-Gradients proposal with MH correction."""

    name = "gwg-1"

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        forward_logits = 0.5 * model.gradient_x(x) * (1.0 - 2.0 * x)
        coordinate = torch.distributions.Categorical(logits=forward_logits).sample()
        proposal = x.clone()
        rows = torch.arange(x.shape[0], device=x.device)
        proposal[rows, coordinate] = 1.0 - proposal[rows, coordinate]

        reverse_logits = 0.5 * model.gradient_x(proposal) * (1.0 - 2.0 * proposal)
        log_q_forward = torch.log_softmax(forward_logits, dim=-1)[rows, coordinate]
        log_q_reverse = torch.log_softmax(reverse_logits, dim=-1)[rows, coordinate]
        log_accept = model(proposal) - model(x) + log_q_reverse - log_q_forward
        accept = torch.log(torch.rand_like(log_accept)) < torch.minimum(log_accept, torch.zeros_like(log_accept))
        result = torch.where(accept[:, None], proposal, x)
        self.last_acceptance = float(accept.float().mean().item())
        self.last_proposed_hops = 1.0
        self.last_hops = self.last_acceptance
        return result


class MultiGWGSampler(Sampler):
    """Released-code GWG-k proposal using k categorical draws with replacement."""

    def __init__(self, number_of_draws: int = 4) -> None:
        super().__init__()
        if number_of_draws < 1:
            raise ValueError("number_of_draws must be positive")
        self.number_of_draws = int(number_of_draws)
        self.name = f"gwg-{self.number_of_draws}"

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        forward_logits = 0.5 * model.gradient_x(x) * (1.0 - 2.0 * x)
        forward = torch.distributions.Categorical(logits=forward_logits)
        # Shape: (number_of_draws, chains), matching MultiDiffSampler in the
        # released repository. Duplicate draws are collapsed before flipping.
        drawn_coordinates = forward.sample((self.number_of_draws,))
        changes = torch.zeros_like(x, dtype=torch.bool)
        chain_indices = torch.arange(x.shape[0], device=x.device)
        for draw in drawn_coordinates:
            changes[chain_indices, draw] = True
        proposal = torch.where(changes, 1.0 - x, x)

        log_q_forward = forward.log_prob(drawn_coordinates).sum(dim=0)
        reverse_logits = 0.5 * model.gradient_x(proposal) * (1.0 - 2.0 * proposal)
        reverse = torch.distributions.Categorical(logits=reverse_logits)
        log_q_reverse = reverse.log_prob(drawn_coordinates).sum(dim=0)
        log_accept = model(proposal) - model(x) + log_q_reverse - log_q_forward
        accept = torch.log(torch.rand_like(log_accept)) < torch.minimum(
            log_accept, torch.zeros_like(log_accept)
        )
        result = torch.where(accept[:, None], proposal, x)
        proposed_hops = changes.sum(dim=-1).float()
        self.last_acceptance = float(accept.float().mean().item())
        self.last_proposed_hops = float(proposed_hops.mean().item())
        self.last_hops = float((proposed_hops * accept).mean().item())
        return result


class DiscreteLangevinSampler(Sampler):
    """Binary DULA/DMALA using the proposal in Equation (2) of the paper."""

    def __init__(self, step_size: float, metropolis_adjusted: bool):
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = float(step_size)
        self.metropolis_adjusted = metropolis_adjusted
        self.name = "dmala" if metropolis_adjusted else "dula"

    def _proposal_logits(self, x: torch.Tensor, model) -> torch.Tensor:
        flip_direction = 1.0 - 2.0 * x
        return 0.5 * model.gradient_x(x) * flip_direction - 1.0 / (2.0 * self.step_size)

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        forward_logits = self._proposal_logits(x, model)
        changes = torch.bernoulli(torch.sigmoid(forward_logits))
        proposal = torch.where(changes.bool(), 1.0 - x, x)
        proposed_hops = changes.sum(dim=-1)

        if not self.metropolis_adjusted:
            self.last_acceptance = 1.0
            self.last_proposed_hops = float(proposed_hops.float().mean().item())
            self.last_hops = self.last_proposed_hops
            return proposal

        reverse_logits = self._proposal_logits(proposal, model)
        log_q_forward = _bernoulli_log_prob(changes, forward_logits).sum(dim=-1)
        log_q_reverse = _bernoulli_log_prob(changes, reverse_logits).sum(dim=-1)
        log_accept = model(proposal) - model(x) + log_q_reverse - log_q_forward
        accept = torch.log(torch.rand_like(log_accept)) < torch.minimum(log_accept, torch.zeros_like(log_accept))
        result = torch.where(accept[:, None], proposal, x)
        self.last_acceptance = float(accept.float().mean().item())
        self.last_proposed_hops = float(proposed_hops.float().mean().item())
        self.last_hops = float((proposed_hops * accept).float().mean().item())
        return result


def build_sampler(name: str, step_size: float = 0.2) -> Sampler:
    normalized = name.lower()
    if normalized in ("gibbs", "gibbs-1"):
        return GibbsSampler()
    if normalized == "gibbs-4":
        return BlockGibbsSampler(4)
    if normalized in ("gwg", "gwg-1"):
        return GWGSampler()
    if normalized == "gwg-4":
        return MultiGWGSampler(4)
    if normalized == "dula":
        return DiscreteLangevinSampler(step_size, metropolis_adjusted=False)
    if normalized == "dmala":
        return DiscreteLangevinSampler(step_size, metropolis_adjusted=True)
    raise ValueError(
        f"unknown sampler {name!r}; choose gibbs-1, gibbs-4, gwg-1, gwg-4, dula, or dmala"
    )
