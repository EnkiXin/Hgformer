"""Data-free compatibility tests for the Hgformer SLRecGraph adapter."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole_gnn.model.general_recommender.slrecgraph import SLRecGraph
from recbole_gnn.utils import get_model


class _TinyDataset:
    def __init__(self):
        # RecBole reserves id 0; all actual interactions use positive ids.
        self._interaction = sp.coo_matrix(
            (
                np.ones(5, dtype=np.float32),
                (np.array([1, 1, 2, 2, 2]), np.array([1, 2, 2, 3, 4])),
            ),
            shape=(3, 5),
        )

    @staticmethod
    def num(field):
        return {"user_id": 3, "item_id": 5}[field]

    def inter_matrix(self, form="coo"):
        if form != "coo":
            raise ValueError("tiny fixture only implements COO")
        return self._interaction

    def get_interactions(self):
        return (
            torch.as_tensor(self._interaction.row, dtype=torch.long),
            torch.as_tensor(self._interaction.col, dtype=torch.long),
        )


def _config(**updates):
    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "tail_analysis": False,
        "popularity_analysis": False,
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "n_layers": 2,
        "init_std": 0.01,
        "reg_weight": 1e-5,
        "sl_scale": 1.0,
        "coord_clip": 1.0,
        "schatten_p": 2,
        "log_terms": 6,
        "log_jitter": 1e-7,
        "symmetric_distance": True,
        "score_scale": 1.0,
        "learnable_score_scale": True,
        "max_score_scale": 100.0,
        # Exercise both full-sort chunk axes on the tiny graph.
        "eval_user_chunk_size": 1,
        "eval_item_chunk_size": 2,
    }
    config.update(updates)
    return config


class SLRecGraphAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.model = SLRecGraph(_config(), _TinyDataset())

    def test_legacy_runner_discovers_pairwise_model(self):
        discovered = get_model("SLRecGraph")
        self.assertIs(discovered, SLRecGraph)
        self.assertEqual(discovered.input_type, InputType.PAIRWISE)

    def test_propagation_is_native_torch_sparse_and_group_is_sl(self):
        self.assertTrue(self.model.norm_adj_matrix.is_sparse)
        user_coordinates, item_coordinates = self.model.forward()
        groups = self.model._to_group(
            torch.cat((user_coordinates, item_coordinates), dim=0)
        )
        determinant = torch.linalg.det(groups)
        torch.testing.assert_close(
            determinant,
            torch.ones_like(determinant),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_pairwise_loss_is_finite_and_differentiable(self):
        interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }
        loss = self.model.calculate_loss(interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for embedding in (self.model.user_embedding, self.model.item_embedding):
            self.assertIsNotNone(embedding.weight.grad)
            self.assertTrue(torch.isfinite(embedding.weight.grad).all())

    def test_two_axis_chunked_full_sort_preserves_row_major_scores(self):
        self.model.eval()
        interaction = {"user_id": torch.tensor([1, 2])}
        chunked = self.model.full_sort_predict(interaction).view(2, 5)

        all_users, all_items = self.model._full_sort_group_tables()
        direct = self.model._score_groups(
            all_users[torch.tensor([1, 2])][:, None, :, :],
            all_items[None, :, :, :],
        )
        torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-6)

    def test_chart_decoder_uses_same_squared_distance_for_pair_and_full_sort(self):
        model = SLRecGraph(
            _config(sl_score_mode="tangent_euclidean"), _TinyDataset()
        )
        model.eval()
        requested_users = torch.tensor([1, 2])
        all_users, all_items = model._full_sort_effective_coordinate_tables()
        direct = -model._score_scale() * model._pairwise_squared_coordinate_distance(
            all_users[requested_users][:, None, ...], all_items[None, ...]
        )
        full = model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)
        torch.testing.assert_close(full, direct, atol=2e-6, rtol=2e-5)

        pairs = model.predict(
            {
                "user_id": requested_users,
                "item_id": torch.tensor([2, 4]),
            }
        )
        torch.testing.assert_close(
            pairs,
            direct[torch.arange(2), torch.tensor([2, 4])],
            atol=1e-7,
            rtol=1e-6,
        )

    def test_parameter_budget_is_explicit(self):
        entity_parameters = (self.model.n_users + self.model.n_items) * 64
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        # Entity tables plus one global learned score-scale scalar.
        self.assertEqual(trainable, entity_parameters + 1)

    def test_single_factor_scoring_regresses_to_original_sl_distance(self):
        all_users, all_items = self.model._full_sort_group_tables()
        users = all_users[torch.tensor([1, 2])]
        items = all_items[torch.tensor([2, 4])]

        actual = self.model._score_groups(users, items)
        # Before product factors were introduced, the tensors had shape
        # [B,n,n].  Removing the singleton factor recreates that exact path.
        from slrec_experiments.geometry import sl_semidistance

        expected_distance = sl_semidistance(
            users.squeeze(-3),
            items.squeeze(-3),
            p=self.model.schatten_p,
            terms=self.model.log_terms,
            jitter=self.model.log_jitter,
            symmetric=self.model.symmetric_distance,
        )
        expected = -self.model._score_scale() * expected_distance
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class SLRecGraphProductTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.model = SLRecGraph(
            _config(matrix_dim=4, num_factors=4, embedding_size=64),
            _TinyDataset(),
        )

    def test_product_shapes_determinants_and_equal_raw_budget(self):
        self.assertEqual(self.model.coordinate_dim, 64)
        self.assertEqual(self.model.intrinsic_dim, 60)
        users, items = self.model.forward()
        self.assertEqual(users.shape, (3, 4, 4, 4))
        self.assertEqual(items.shape, (5, 4, 4, 4))

        groups = self.model._to_group(torch.cat((users, items), dim=0))
        determinants = torch.linalg.det(groups)
        self.assertEqual(determinants.shape, (8, 4))
        torch.testing.assert_close(
            determinants,
            torch.ones_like(determinants),
            atol=2e-5,
            rtol=2e-5,
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, (self.model.n_users + self.model.n_items) * 64 + 1)

    def test_product_loss_supports_per_user_multiple_negatives(self):
        interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([[4, 3, 2], [1, 4, 2]]),
        }
        loss = self.model.calculate_loss(interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for embedding in (self.model.user_embedding, self.model.item_embedding):
            self.assertIsNotNone(embedding.weight.grad)
            self.assertTrue(torch.isfinite(embedding.weight.grad).all())

    def test_product_full_sort_matches_unchunked_broadcast(self):
        self.model.eval()
        requested_users = torch.tensor([1, 2])
        chunked = self.model.full_sort_predict(
            {"user_id": requested_users}
        ).view(2, 5)

        all_users, all_items = self.model._full_sort_group_tables()
        direct = self.model._score_groups(
            all_users[requested_users][:, None, ...],
            all_items[None, ...],
        )
        self.assertEqual(direct.shape, (2, 5))
        torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-6)

    def test_product_aggregation_definitions(self):
        distances = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        torch.testing.assert_close(
            self.model._aggregate_factor_distances(distances),
            torch.tensor([5.0]),
        )

        l1_model = SLRecGraph(
            _config(
                matrix_dim=4,
                num_factors=4,
                embedding_size=64,
                factor_aggregation="l1",
            ),
            _TinyDataset(),
        )
        mean_model = SLRecGraph(
            _config(
                matrix_dim=4,
                num_factors=4,
                embedding_size=64,
                factor_aggregation="mean",
            ),
            _TinyDataset(),
        )
        torch.testing.assert_close(
            l1_model._aggregate_factor_distances(distances), torch.tensor([7.0])
        )
        torch.testing.assert_close(
            mean_model._aggregate_factor_distances(distances), torch.tensor([1.75])
        )

    def test_invalid_product_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_factors"):
            SLRecGraph(_config(num_factors=0), _TinyDataset())
        with self.assertRaisesRegex(ValueError, "factor_aggregation"):
            SLRecGraph(
                _config(factor_aggregation="not-a-product-metric"),
                _TinyDataset(),
            )


if __name__ == "__main__":
    unittest.main()
