from __future__ import annotations

import torch
from torch import nn


def lattice_adjacency(side: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the symmetric adjacency matrix of a periodic 2-D square lattice."""
    if side < 2:
        raise ValueError("side must be at least 2")
    n = side * side
    adjacency = torch.zeros((n, n), dtype=torch.float32, device=device)
    for row in range(side):
        for col in range(side):
            node = row * side + col
            for next_row, next_col in ((row, (col + 1) % side), ((row + 1) % side, col)):
                neighbor = next_row * side + next_col
                adjacency[node, neighbor] = 1.0
                adjacency[neighbor, node] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


class IsingModel(nn.Module):
    """Ising log-density on x in {0,1}^d with spins s=2x-1.

    The unnormalized log-density is U(x) = s^T J s + b^T s.  This is the
    matrix convention used by the original repository: a symmetric adjacency
    matrix counts every undirected edge twice.
    """

    def __init__(self, coupling: torch.Tensor, bias: torch.Tensor | float = 0.0):
        super().__init__()
        if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
            raise ValueError("coupling must be a square matrix")
        coupling = 0.5 * (coupling + coupling.T)
        coupling = coupling.clone()
        coupling.fill_diagonal_(0.0)
        if not torch.is_tensor(bias):
            bias = torch.full((coupling.shape[0],), float(bias), dtype=coupling.dtype)
        if bias.shape != (coupling.shape[0],):
            raise ValueError("bias must have shape (dimension,)")
        self.register_buffer("coupling", coupling)
        self.register_buffer("bias", bias.to(dtype=coupling.dtype, device=coupling.device))

    @property
    def dimension(self) -> int:
        return self.coupling.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spins = 2.0 * x - 1.0
        return (spins @ self.coupling * spins).sum(dim=-1) + (spins * self.bias).sum(dim=-1)

    def gradient_x(self, x: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        spins = 2.0 * x - 1.0
        return 4.0 * (spins @ self.coupling) + 2.0 * self.bias

    def flip_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Exact U(flip_i(x))-U(x) for every coordinate i."""
        return self.gradient_x(x) * (1.0 - 2.0 * x)


class LearnableIsingModel(nn.Module):
    """Dense symmetric Ising model used for parameter-recovery experiments."""

    def __init__(self, dimension: int, init_scale: float = 0.01):
        super().__init__()
        raw = torch.randn((dimension, dimension), dtype=torch.float32) * init_scale
        self.raw_coupling = nn.Parameter(raw)
        self.bias = nn.Parameter(torch.zeros(dimension, dtype=torch.float32))
        self.register_buffer("off_diagonal", 1.0 - torch.eye(dimension, dtype=torch.float32))

    @property
    def dimension(self) -> int:
        return self.raw_coupling.shape[0]

    @property
    def coupling(self) -> torch.Tensor:
        return 0.5 * (self.raw_coupling + self.raw_coupling.T) * self.off_diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spins = 2.0 * x - 1.0
        return (spins @ self.coupling * spins).sum(dim=-1) + (spins * self.bias).sum(dim=-1)

    def gradient_x(self, x: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        coupling = self.coupling if create_graph else self.coupling.detach()
        bias = self.bias if create_graph else self.bias.detach()
        spins = 2.0 * x - 1.0
        return 4.0 * (spins @ coupling) + 2.0 * bias

    def flip_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.gradient_x(x) * (1.0 - 2.0 * x)
