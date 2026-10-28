from __future__ import annotations

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
    """Random-scan, one-coordinate heat-bath Gibbs kernel."""

    name = "gibbs"

    @torch.no_grad()
    def step(self, x: torch.Tensor, model) -> torch.Tensor:
        coordinate = torch.randint(model.dimension, (), device=x.device)
        coordinate_index = int(coordinate.item())
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


class GWGSampler(Sampler):
    """Single-flip Gibbs-with-Gradients proposal with MH correction."""

    name = "gwg"

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
    if normalized == "gibbs":
        return GibbsSampler()
    if normalized == "gwg":
        return GWGSampler()
    if normalized == "dula":
        return DiscreteLangevinSampler(step_size, metropolis_adjusted=False)
    if normalized == "dmala":
        return DiscreteLangevinSampler(step_size, metropolis_adjusted=True)
    raise ValueError(f"unknown sampler {name!r}; choose gibbs, gwg, dula, or dmala")
