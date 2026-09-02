"""CPU tests for the chart-based AGCF-SL(8) surrogate."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from recbole_gnn.model.general_recommender.agcfsl8coord import (
    AGCFSL8Coord,
    _orthonormal_sl_basis,
)
from recbole_gnn.utils import get_model
from slrec_experiments import geometry as sl_geometry


class _TinyGraphDataset:
    def __init__(self) -> None:
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
    def num(field: str) -> int:
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
        "embedding_size": 63,
        "metric_rank": 2,
        "metric_hidden_size": 12,
        "pnet_hidden_size": 11,
        "channel_rank": 3,
        "metric_epsilon": 1e-2,
        "structural_delta": 1e-3,
        "potential_strength": 0.02,
        "damping": 0.01,
        "evolution_time": 0.1,
        "output_steps": 1,
        "integration_steps": 1,
        "dynamics_chunk_size": 4,
        "checkpoint_dynamics": False,
        "matrix_dim": 8,
        "num_factors": 1,
        "sl_scale": 1.0,
        "coord_clip": 0.5,
        "schatten_p": 2,
        "symmetric_distance": False,
        "log_terms": 4,
        "log_jitter": 0.0,
        "pairwise_loss": "hinge",
        "loss_margin": 0.1,
        "eval_user_chunk_size": 1,
        "eval_item_chunk_size": 2,
    }
    config.update(updates)
    return config


class AGCFSL8CoordTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.dataset = _TinyGraphDataset()
        self.interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }

    def test_lazy_registry_and_exact_63d_chart(self):
        self.assertIs(get_model("AGCFSL8Coord"), AGCFSL8Coord)
        from recbole_gnn.model.general_recommender import (
            AGCFSL8Coord as public_model,
        )

        self.assertIs(public_model, AGCFSL8Coord)
        model = AGCFSL8Coord(_config(), self.dataset)
        self.assertEqual(model.embedding_size, 63)
        self.assertEqual(model.node_embedding.embedding_dim, 63)
        self.assertEqual(tuple(model.sl8_chart_basis.shape), (63, 8, 8))

        basis = _orthonormal_sl_basis(8)
        gram = basis.reshape(63, -1) @ basis.reshape(63, -1).T
        torch.testing.assert_close(gram, torch.eye(63), atol=2e-6, rtol=2e-6)
        traces = basis.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        torch.testing.assert_close(traces, torch.zeros(63), atol=1e-7, rtol=0)

    def test_chart_is_trace_free_and_exponential_lies_in_sl8(self):
        model = AGCFSL8Coord(_config(), self.dataset)
        users, items = model.forward()
        coordinates = torch.cat((users, items), dim=0)
        lie_algebra = model._coordinates_to_lie_algebra(coordinates)
        trace = lie_algebra.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        torch.testing.assert_close(trace, torch.zeros_like(trace), atol=2e-6, rtol=0)

        groups = model._to_group(coordinates)
        torch.testing.assert_close(
            torch.linalg.det(groups),
            torch.ones(groups.shape[0]),
            atol=3e-5,
            rtol=3e-5,
        )
        diagnostics = model.geometry_diagnostics(sample_nodes=8)
        self.assertEqual(diagnostics["chart_dimension"], 63)
        self.assertEqual(diagnostics["matrix_dimension"], 8)
        self.assertEqual(diagnostics["group_membership_violations"], 0)
        self.assertLessEqual(
            diagnostics["decoder_effective_frobenius_max"],
            model.coord_clip + 1e-6,
        )
        self.assertGreaterEqual(
            diagnostics["decoder_clip_saturation_fraction"], 0.0
        )
        self.assertLessEqual(
            diagnostics["decoder_clip_saturation_fraction"], 1.0
        )

    def test_distance_calls_exactly_one_matrix_log_and_frobenius_path(self):
        model = AGCFSL8Coord(_config(), self.dataset)
        left = model._to_group(torch.randn(2, 63) * 0.01)
        right = model._to_group(torch.randn(2, 63) * 0.01)
        original = sl_geometry.matrix_log_gregory
        with patch.object(
            sl_geometry, "matrix_log_gregory", wraps=original
        ) as matrix_log:
            actual = model._group_distance(left, right)
        self.assertEqual(matrix_log.call_count, 1)

        relative = sl_geometry.relative_matrix(left, right)
        explicit = torch.linalg.matrix_norm(
            original(relative, terms=model.log_terms, jitter=model.log_jitter),
            ord="fro",
            dim=(-2, -1),
        )
        torch.testing.assert_close(actual, explicit, atol=1e-6, rtol=1e-6)

    def test_k12_fast_distance_matches_reference_definition(self):
        model = AGCFSL8Coord(_config(log_terms=12), self.dataset)
        left = model._to_group(torch.randn(5, 63) * 0.01)
        right = model._to_group(torch.randn(5, 63) * 0.01)
        actual = model._group_distance(left, right)
        expected = sl_geometry.sl_semidistance(
            left,
            right,
            p=2,
            terms=12,
            jitter=model.log_jitter,
            symmetric=False,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_hinge_and_configurable_bpr_losses_backward(self):
        for pairwise_loss, negatives in (
            ("hinge", self.interaction["neg_item_id"]),
            ("bpr", torch.tensor([[4, 3], [1, 4]])),
        ):
            with self.subTest(pairwise_loss=pairwise_loss):
                model = AGCFSL8Coord(
                    _config(pairwise_loss=pairwise_loss), self.dataset
                )
                interaction = dict(self.interaction)
                interaction["neg_item_id"] = negatives
                loss = model.calculate_loss(interaction)
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in (
                    model.node_embedding.weight,
                    model.pnet[0].weight,
                    model.inverse_metric.input_layer.weight,
                    model.channel_factor,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_predict_and_chunked_cached_full_sort_match(self):
        model = AGCFSL8Coord(_config(), self.dataset)
        model.eval()
        requested_users = torch.tensor([1, 2])
        with torch.no_grad():
            chunked = model.full_sort_predict(
                {"user_id": requested_users}
            ).reshape(2, 5)
            cached_users, cached_items = model._full_sort_group_tables()
            direct = model._score_groups(
                cached_users[requested_users][:, None, ...],
                cached_items[None, ...],
            )
            second = model.full_sort_predict(
                {"user_id": requested_users}
            ).reshape(2, 5)
            pair = model.predict(
                {
                    "user_id": requested_users,
                    "item_id": torch.tensor([3, 4]),
                }
            )
        torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(second, direct, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            pair,
            direct[torch.arange(2), torch.tensor([3, 4])],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertIsNotNone(model.restore_user_group)
        self.assertIsNotNone(model.restore_item_group)

    def test_configuration_guards_keep_the_claim_fixed(self):
        with self.assertRaisesRegex(ValueError, "embedding_size=63"):
            AGCFSL8Coord(_config(embedding_size=64), self.dataset)
        with self.assertRaisesRegex(ValueError, "matrix_dim"):
            AGCFSL8Coord(_config(matrix_dim=4), self.dataset)
        with self.assertRaisesRegex(ValueError, "schatten_p=2"):
            AGCFSL8Coord(_config(schatten_p=8), self.dataset)
        with self.assertRaisesRegex(ValueError, "one-sided"):
            AGCFSL8Coord(_config(symmetric_distance=True), self.dataset)


if __name__ == "__main__":
    unittest.main()
