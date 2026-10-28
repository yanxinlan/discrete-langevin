import unittest
import itertools

import torch

from ising_pipeline.model import IsingModel, lattice_adjacency
from ising_pipeline.samplers import build_sampler


class IsingPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = IsingModel(0.1 * lattice_adjacency(3), 0.2)
        self.x = torch.bernoulli(torch.full((16, 9), 0.5))

    def test_flip_delta_matches_energy_difference(self):
        delta = self.model.flip_delta(self.x)
        for coordinate in range(self.model.dimension):
            flipped = self.x.clone()
            flipped[:, coordinate] = 1.0 - flipped[:, coordinate]
            torch.testing.assert_close(delta[:, coordinate], self.model(flipped) - self.model(self.x))

    def test_analytic_gradient_matches_autograd(self):
        x = self.x.clone().requires_grad_(True)
        automatic = torch.autograd.grad(self.model(x).sum(), x)[0]
        torch.testing.assert_close(self.model.gradient_x(x.detach()), automatic)

    def test_all_samplers_preserve_binary_shape(self):
        for name in ("gibbs", "gwg", "dula", "dmala"):
            result = build_sampler(name, 0.2).step(self.x, self.model)
            self.assertEqual(result.shape, self.x.shape)
            self.assertTrue(torch.all((result == 0) | (result == 1)))

    def test_dmala_kernel_has_exact_target_stationary_distribution(self):
        model = IsingModel(0.15 * lattice_adjacency(2), 0.1)
        states = torch.tensor(list(itertools.product((0.0, 1.0), repeat=4)))
        sampler = build_sampler("dmala", 0.3)
        logits = sampler._proposal_logits(states, model)
        transition = torch.zeros((len(states), len(states)))
        for source, x in enumerate(states):
            for destination, y in enumerate(states):
                changes = (x != y).float()
                log_q_forward = -torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[source], changes, reduction="sum"
                )
                reverse_changes = changes
                log_q_reverse = -torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[destination], reverse_changes, reduction="sum"
                )
                log_accept = model(y[None])[0] - model(x[None])[0] + log_q_reverse - log_q_forward
                accepted_probability = torch.exp(log_q_forward + torch.minimum(log_accept, torch.tensor(0.0)))
                if source == destination:
                    transition[source, source] += torch.exp(log_q_forward)
                else:
                    transition[source, destination] += accepted_probability
                    transition[source, source] += torch.exp(log_q_forward) - accepted_probability
        target = torch.softmax(model(states), dim=0)
        torch.testing.assert_close(transition.sum(dim=1), torch.ones(len(states)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(target @ transition, target, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
