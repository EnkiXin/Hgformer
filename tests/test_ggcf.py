"""CPU contract tests for the paper-faithful GGCF clean-room model."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole_gnn.config import Config
from recbole_gnn.model.general_recommender.ggcf import GGCF
from recbole_gnn.utils import get_model


class _TinyTrainingDataset:
    """Training split with intentionally misleading base-class graph data."""

    def __init__(self) -> None:
        # RecBole reserves id zero.  The two isolated zero ids exercise GGCF's
        # explicit zero-neighbour centroid fallback.
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
        self.is_sparse = False

    @staticmethod
    def num(field: str) -> int:
        return {"user_id": 3, "item_id": 5}[field]

    def get_interactions(self):
        return (
            torch.as_tensor(self._interaction.row, dtype=torch.long),
            torch.as_tensor(self._interaction.col, dtype=torch.long),
        )

    def inter_matrix(self, form: str = "coo") -> sp.coo_matrix:
        if form != "coo":
            raise ValueError("tiny fixture only supports COO")
        return self._interaction

    def get_norm_adj_mat(self, enable_sparse: bool = False):
        del enable_sparse
        # GeneralGraphRecommender requires this method.  Include a fake
        # self-loop that is absent from get_interactions(); GGCF must discard
        # this representation and rebuild solely from the training pairs.
        users, items = self.get_interactions()
        item_nodes = items + 3
        source = torch.cat((users, item_nodes, torch.tensor([7])))
        target = torch.cat((item_nodes, users, torch.tensor([7])))
        return torch.stack((source, target)), torch.ones(source.numel())


def _config(**updates):
    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "enable_sparse": False,
        "tail_analysis": False,
        "popularity_analysis": False,
        "embedding_size": 8,
        "ggcf_branch_size": 4,
        "n_layers": 2,
        "reg_weight": 1e-4,
        "require_pow": True,
        "eval_item_chunk_size": 2,
        "lorentz_eps": 1e-7,
        "gamma_init": 0.1,
        "gamma_prime_init": 0.1,
        "lambda_init": 1.0,
        "ggcf_init_method": "normal",
        "init_std": 0.02,
        "hyperbolic_layer_fusion": "lorentz_centroid",
    }
    config.update(updates)
    return config


class GGCFGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.model = GGCF(_config(), _TinyTrainingDataset())

    def _assert_on_hyperboloid(
        self, points: torch.Tensor, atol: float = 2e-5
    ) -> None:
        quadratic = self.model._lorentz_inner(points, points)
        torch.testing.assert_close(
            quadratic,
            -torch.ones_like(quadratic),
            atol=atol,
            rtol=atol,
        )
        self.assertTrue(torch.all(points[..., 0] > 0))

    def test_exp_log_parallel_transport_and_gyro_ops_are_geometric(self):
        tangent = 0.1 * torch.randn(7, self.model.branch_size)
        points = self.model._lorentz_exp0(tangent)
        self._assert_on_hyperboloid(points)
        torch.testing.assert_close(
            self.model._lorentz_log0(points), tangent, atol=2e-5, rtol=2e-5
        )

        transported = self.model._parallel_transport_origin_to(
            points, tangent
        )
        torch.testing.assert_close(
            self.model._lorentz_inner(points, transported),
            torch.zeros(7),
            atol=2e-6,
            rtol=2e-6,
        )
        scaled = self.model._lorentz_scalar_mul(
            torch.full((7, 1), 0.3), points
        )
        added = self.model._lorentz_gyro_add(points, scaled)
        self._assert_on_hyperboloid(scaled)
        self._assert_on_hyperboloid(added, atol=4e-5)

    def test_sparse_and_dense_geometric_propagation_match(self):
        adjacency = self.model.normalized_adjacency
        dense = adjacency.to_dense()
        euclidean = torch.randn(8, self.model.branch_size)
        hyperbolic = self.model._lorentz_exp0(
            torch.randn(8, self.model.branch_size) * 0.03
        )

        sparse_e = self.model._graph_mm(adjacency, euclidean)
        dense_e = self.model._graph_mm(dense, euclidean)
        torch.testing.assert_close(sparse_e, dense_e)

        sparse_h = self.model._normalise_lorentz_ambient(
            self.model._graph_mm(adjacency, hyperbolic)
        )
        dense_h = self.model._normalise_lorentz_ambient(
            self.model._graph_mm(dense, hyperbolic)
        )
        torch.testing.assert_close(sparse_h, dense_h)
        self._assert_on_hyperboloid(sparse_h)

    def test_every_initial_propagated_and_fused_point_stays_on_manifold(self):
        users, items = self.model.forward()
        self.assertEqual(users.shape, (3, 9))
        self.assertEqual(items.shape, (5, 9))
        _, user_h = self.model._split_representation(users)
        _, item_h = self.model._split_representation(items)
        self._assert_on_hyperboloid(torch.cat((user_h, item_h), dim=0))

        diagnostics = self.model.geometry_diagnostics()
        self.assertEqual(len(diagnostics["layer_membership"]), 3)
        for layer in diagnostics["layer_membership"]:
            self.assertLess(layer["max_abs_quadratic_error"], 2e-5)
            self.assertGreater(layer["min_time_coordinate"], 0.0)
            self.assertEqual(layer["nonfinite_points"], 0)
        self.assertLess(
            diagnostics["final_membership"]["max_abs_quadratic_error"],
            2e-5,
        )


class GGCFModelContractTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.dataset = _TinyTrainingDataset()
        self.model = GGCF(_config(), self.dataset)
        self.interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }

    def test_lazy_discovery_dimension_budget_and_training_graph_only(self):
        self.assertIs(get_model("GGCF"), GGCF)
        from recbole_gnn.model.general_recommender import GGCF as public_model

        self.assertIs(public_model, GGCF)
        self.assertEqual(GGCF.input_type, InputType.PAIRWISE)
        self.assertEqual(self.model.embedding_size, 8)
        self.assertEqual(self.model.branch_size, 4)
        self.assertEqual(self.model.representation_size, 9)

        adjacency = self.model.normalized_adjacency.coalesce()
        self.assertTrue(adjacency.is_sparse)
        self.assertFalse(
            torch.any(adjacency.indices()[0] == adjacency.indices()[1])
        )
        # The fake node-7 self-loop returned by get_norm_adj_mat is not part
        # of the actual training interactions and therefore must be absent.
        self.assertEqual(float(adjacency.to_dense()[7, 7]), 0.0)
        torch.testing.assert_close(
            adjacency.to_dense(), adjacency.to_dense().transpose(0, 1)
        )
        expected = torch.zeros(8, 8)
        for user, item_node, weight in (
            (1, 4, 1.0 / np.sqrt(2.0)),
            (1, 5, 1.0 / np.sqrt(4.0)),
            (2, 5, 1.0 / np.sqrt(6.0)),
            (2, 6, 1.0 / np.sqrt(3.0)),
            (2, 7, 1.0 / np.sqrt(3.0)),
        ):
            expected[user, item_node] = weight
            expected[item_node, user] = weight
        torch.testing.assert_close(adjacency.to_dense(), expected)

    def test_invalid_or_ambiguous_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit budget"):
            GGCF(
                _config(embedding_size=64, ggcf_branch_size=64), self.dataset
            )

    def test_eq6_scalars_score_bpr_l2_and_backward(self):
        loss = self.model.calculate_loss(self.interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            self.model.euclidean_embedding.weight,
            self.model.hyperbolic_tangent_embedding.weight,
            self.model.gamma,
            self.model.gamma_prime,
            self.model.geometry_lambda,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_multi_negative_bpr_is_supported(self):
        interaction = dict(self.interaction)
        interaction["neg_item_id"] = torch.tensor([[4, 2], [1, 4]])
        loss = self.model.calculate_loss(interaction)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            torch.isfinite(self.model.euclidean_embedding.weight.grad).all()
        )

    def test_chunked_full_sort_matches_pair_scores_and_reuses_cache(self):
        self.model.eval()
        requested = torch.tensor([1, 2])
        with torch.no_grad():
            scores = self.model.full_sort_predict(
                {"user_id": requested}
            ).reshape(2, 5)
            cached_user_pointer = self.model.restore_user_e.data_ptr()
            cached_item_pointer = self.model.restore_item_e.data_ptr()
            users = self.model.restore_user_e[requested]
            items = self.model.restore_item_e
            direct = self.model._pair_score(
                users.unsqueeze(1), items.unsqueeze(0)
            )
            second = self.model.full_sort_predict(
                {"user_id": requested}
            ).reshape(2, 5)

        torch.testing.assert_close(scores, direct, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(second, direct, atol=2e-6, rtol=2e-5)
        self.assertEqual(self.model.restore_user_e.data_ptr(), cached_user_pointer)
        self.assertEqual(self.model.restore_item_e.data_ptr(), cached_item_pointer)

        # Any training loss must invalidate representations materialised before
        # an optimiser update; otherwise later validation could rank with stale
        # entity tables.
        training_loss = self.model.calculate_loss(self.interaction)
        self.assertTrue(torch.isfinite(training_loss))
        self.assertIsNone(self.model.restore_user_e)
        self.assertIsNone(self.model.restore_item_e)


class GGCFConfigContractTest(unittest.TestCase):
    def test_default_and_cd_overlay_compose_with_hgformer_protocol(self):
        root = Path(__file__).resolve().parents[1]
        base = root / "baseline_config_fixed" / "RecFormer_cd.yaml"
        overlay = root / "baseline_config_fixed" / "GGCF_cd.yaml"
        with patch("sys.argv", ["test_ggcf"]):
            config = Config(
                model="GGCF",
                dataset="Amazon_cd",
                config_file_list=[str(base), str(overlay)],
                config_dict={"use_gpu": False},
            )

        self.assertEqual(config["model"], "GGCF")
        self.assertEqual(config["embedding_size"], 64)
        self.assertEqual(config["ggcf_branch_size"], 32)
        self.assertEqual(config["n_layers"], 3)
        self.assertEqual(config["seed"], 2024)
        self.assertEqual(config["epochs"], 500)
        self.assertEqual(config["valid_metric"], "Recall@10")
        self.assertEqual(
            config["eval_args"],
            {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
        )

    def test_cd_overlay_does_not_override_data_or_evaluation_protocol(self):
        import yaml

        root = Path(__file__).resolve().parents[1]
        overlay = yaml.safe_load(
            (root / "baseline_config_fixed" / "GGCF_cd.yaml").read_text(
                encoding="utf-8"
            )
        )
        forbidden = {
            "dataset",
            "seed",
            "load_col",
            "user_inter_num_interval",
            "item_inter_num_interval",
            "val_interval",
            "metrics",
            "topk",
            "valid_metric",
            "eval_args",
        }
        self.assertFalse(forbidden.intersection(overlay))


if __name__ == "__main__":
    unittest.main()
