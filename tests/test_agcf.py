"""Focused CPU tests for the paper-derived AGCF clean-room baseline."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole_gnn.model.general_recommender.agcf import (
    AGCF,
    _AdaptiveInverseMetric,
)
from recbole_gnn.utils import get_model


class _TinyGraphDataset:
    def __init__(self):
        self._interaction = sp.coo_matrix(
            (
                np.ones(5, dtype=np.float32),
                (
                    np.array([1, 1, 2, 2, 2]),
                    np.array([1, 2, 2, 3, 4]),
                ),
            ),
            shape=(3, 5),
        )

    @staticmethod
    def num(field):
        return {"user_id": 3, "item_id": 5}[field]

    def get_interactions(self):
        return (
            torch.as_tensor(self._interaction.row, dtype=torch.long),
            torch.as_tensor(self._interaction.col, dtype=torch.long),
        )

    def get_norm_adj_mat(self, enable_sparse=False):
        del enable_sparse
        users = torch.as_tensor(self._interaction.row, dtype=torch.long)
        items = torch.as_tensor(self._interaction.col, dtype=torch.long) + 3
        source = torch.cat((users, items))
        target = torch.cat((items, users))
        degree = torch.bincount(source, minlength=8).float()
        inverse_sqrt_degree = degree.clamp_min(1.0).rsqrt()
        weights = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
        return torch.stack((source, target)), weights


def _config(**updates):
    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "enable_sparse": False,
        "tail_analysis": False,
        "popularity_analysis": False,
        "embedding_size": 6,
        "metric_rank": 2,
        "metric_hidden_size": 7,
        "pnet_hidden_size": 5,
        "channel_rank": 3,
        "metric_epsilon": 1e-2,
        "structural_delta": 1e-3,
        "potential_strength": 0.05,
        "damping": 0.01,
        "evolution_time": 0.2,
        "output_steps": 2,
        "integration_steps": 2,
        "dynamics_chunk_size": 3,
        "checkpoint_dynamics": True,
        "eval_item_chunk_size": 2,
        "margin": 0.1,
    }
    config.update(updates)
    return config


class AdaptiveMetricForceTest(unittest.TestCase):
    def test_analytic_geometric_force_matches_autograd_and_is_trainable(self):
        torch.manual_seed(11)
        metric = _AdaptiveInverseMetric(dimension=4, hidden_size=5, rank=2)
        position = torch.randn(3, 4, requires_grad=True)
        momentum = torch.randn(3, 4, requires_grad=True)
        epsilon = 1e-2

        velocity, actual_force = metric.velocity_and_geometric_force(
            position, momentum, epsilon
        )
        factor = metric.factor(position)
        reference_velocity = (
            torch.einsum(
                "ndr,nr->nd",
                factor,
                torch.einsum("ndr,nd->nr", factor, momentum),
            )
            + epsilon * momentum
        )
        reference_force = torch.autograd.grad(
            (momentum * reference_velocity).sum(),
            position,
            create_graph=True,
        )[0]
        torch.testing.assert_close(velocity, reference_velocity)
        torch.testing.assert_close(actual_force, reference_force, atol=2e-6, rtol=2e-5)

        (velocity.square().mean() + actual_force.square().mean()).backward()
        self.assertTrue(torch.isfinite(position.grad).all())
        self.assertTrue(torch.isfinite(momentum.grad).all())
        for parameter in metric.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class AGCFModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.model = AGCF(_config(), _TinyGraphDataset())
        self.interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }

    def test_lazy_registry_and_sparse_formula_storage(self):
        self.assertIs(get_model("AGCF"), AGCF)
        self.assertEqual(AGCF.input_type, InputType.PAIRWISE)
        self.assertTrue(self.model.normalized_adjacency.is_sparse)
        self.assertEqual(tuple(self.model.normalized_adjacency.shape), (8, 8))
        # The implementation stores only thin metric factors, never Nd x Nd.
        largest_parameter = max(p.numel() for p in self.model.parameters())
        self.assertLess(largest_parameter, (8 * 6) ** 2)

    def test_local_and_channel_metrics_are_strictly_spd(self):
        diagnostics = self.model.geometry_diagnostics(sample_nodes=8)
        self.assertGreater(
            diagnostics["local_inverse_min_eigenvalue"],
            0.0,
        )
        self.assertGreater(diagnostics["channel_min_eigenvalue"], 0.0)
        self.assertGreaterEqual(
            diagnostics["local_inverse_min_eigenvalue"],
            self.model.metric_epsilon * 0.999,
        )
        self.assertGreaterEqual(
            diagnostics["channel_min_eigenvalue"],
            self.model.metric_epsilon * 0.999,
        )

    def test_checkpointed_hamiltonian_rollout_and_backward_are_finite(self):
        users, items = self.model.forward()
        self.assertEqual(users.shape, (3, 6))
        self.assertEqual(items.shape, (5, 6))
        self.assertTrue(torch.isfinite(users).all())
        self.assertTrue(torch.isfinite(items).all())

        loss = self.model.calculate_loss(self.interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            self.model.node_embedding.weight,
            self.model.pnet[0].weight,
            self.model.inverse_metric.input_layer.weight,
            self.model.inverse_metric.factor_layer.weight,
            self.model.channel_factor,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_no_grad_chunked_full_sort_matches_direct_distances(self):
        self.model.eval()
        requested_users = torch.tensor([1, 2])
        with torch.no_grad():
            scores = self.model.full_sort_predict(
                {"user_id": requested_users}
            ).reshape(2, 5)
            users, items = self.model.forward()
            direct = -self.model._squared_distance(
                users[requested_users].unsqueeze(1), items.unsqueeze(0)
            )
        torch.testing.assert_close(scores, direct, atol=2e-6, rtol=2e-5)

    def test_zero_output_intervals_returns_initial_positions(self):
        model = AGCF(
            _config(output_steps=0, checkpoint_dynamics=False),
            _TinyGraphDataset(),
        )
        users, items = model.forward()
        expected_users, expected_items = torch.split(
            model.node_embedding.weight, [3, 5], dim=0
        )
        torch.testing.assert_close(users, expected_users)
        torch.testing.assert_close(items, expected_items)

    def test_margin_config_is_not_silently_ignored(self):
        model = AGCF(_config(margin=0.37), _TinyGraphDataset())
        self.assertAlmostEqual(model.loss_margin, 0.37)


if __name__ == "__main__":
    unittest.main()
