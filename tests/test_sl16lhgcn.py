"""CPU contracts for the optional dimension-16 SL-LHGCN variant."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import yaml

from recbole_gnn.model.general_recommender.sl16lhgcn import SL16LHGCN
from recbole_gnn.model.general_recommender.sl8lhgcn import SL8LHGCN
from recbole_gnn.model.general_recommender.slrecgraph import SLRecGraph
from recbole_gnn.config import Config
from recbole_gnn.utils import get_model


REPO_ROOT = Path(__file__).resolve().parents[1]


class _TinyDataset:
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

    def inter_matrix(self, form: str = "coo") -> sp.coo_matrix:
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
        "embedding_size": 256,
        "matrix_dim": 16,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "n_layers": 1,
        "gcn_layers": 1,
        "lhgcn_include_self": False,
        "lhgcn_self_loop_weight": 1.0,
        "lhgcn_layer_aggregation": "last",
        "sl_layer_norm": "none",
        "sl_gcn_mode": "ambient_retract",
        "sl_centroid_fallback_clip": 1.0,
        "sl_membership_check": True,
        "sl_membership_strict": True,
        "sl_membership_tolerance": 1e-4,
        "sl_distance_membership_check": True,
        "sl_distance_check_samples": 2,
        "sl_log_trace_tolerance": 1e-3,
        "init_std": 0.005,
        "embedding_init": "normal",
        "reg_weight": 0.0,
        "sl_scale": 1.0,
        "coord_clip": 0.5,
        "schatten_p": 2,
        "log_terms": 6,
        "log_jitter": 0.0,
        "symmetric_distance": False,
        "score_scale": 1.0,
        "learnable_score_scale": False,
        "max_score_scale": 100.0,
        "pairwise_loss": "lhgcn_hinge_squared_sum",
        "loss_margin": 0.1,
        "eval_user_chunk_size": 1,
        "eval_item_chunk_size": 2,
    }
    config.update(updates)
    return config


class SL16DiscoveryAndConstraintTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.dataset = _TinyDataset()

    def test_dynamic_and_public_model_discovery(self):
        self.assertIs(get_model("SL16LHGCN"), SL16LHGCN)
        from recbole_gnn.model.general_recommender import (
            SL16LHGCN as public_model,
        )

        self.assertIs(public_model, SL16LHGCN)

    def test_dimension_guards_are_variant_specific(self):
        sl16 = SL16LHGCN(_config(), self.dataset)
        self.assertEqual(sl16.matrix_dim, 16)
        self.assertEqual(sl16.coordinate_dim, 256)
        self.assertEqual(sl16.intrinsic_dim, 255)
        self.assertEqual(tuple(sl16.user_embedding.weight.shape), (3, 256))

        with self.assertRaisesRegex(ValueError, "SL16LHGCN.*matrix_dim: 16"):
            SL16LHGCN(
                _config(matrix_dim=8, embedding_size=64), self.dataset
            )
        with self.assertRaisesRegex(ValueError, "num_factors: 1"):
            SL16LHGCN(
                _config(num_factors=2, embedding_size=512), self.dataset
            )
        with self.assertRaisesRegex(ValueError, "SL8LHGCN.*matrix_dim: 8"):
            SL8LHGCN(_config(), self.dataset)

    def test_generic_slrecgraph_remains_dimension_agnostic(self):
        model = SLRecGraph(
            _config(n_layers=0, gcn_layers=0), self.dataset
        )
        self.assertEqual(model.matrix_dim, 16)
        self.assertEqual(model.coordinate_dim, 256)


class SL16EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.dataset = _TinyDataset()
        self.model = SL16LHGCN(_config(), self.dataset)

    def test_forward_loss_full_sort_and_manifold_membership(self):
        users, items = self.model.forward()
        groups = torch.cat((users, items), dim=0)
        self.assertEqual(tuple(groups.shape), (8, 1, 16, 16))
        sign, log_abs_det = torch.linalg.slogdet(groups)
        torch.testing.assert_close(sign, torch.ones_like(sign), atol=0, rtol=0)
        self.assertLess(float(log_abs_det.abs().max()), 1e-4)
        diagnostics = self.model.projection_diagnostics()
        self.assertEqual(diagnostics["output_membership_violations"], 0)
        self.assertEqual(
            diagnostics["initial_group_membership"]["membership_violations"],
            0,
        )

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

        self.model.eval()
        requested_users = torch.tensor([1, 2])
        full_scores = self.model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)
        self.assertTrue(torch.isfinite(full_scores).all())
        all_users, all_items = self.model._full_sort_group_tables()
        direct = self.model._score_groups(
            all_users[requested_users][:, None, ...], all_items[None, ...]
        )
        torch.testing.assert_close(full_scores, direct, atol=1e-6, rtol=1e-6)

    def test_chart_decoder_inherits_exact_norm_plus_gemm_full_sort(self):
        model = SL16LHGCN(
            _config(
                sl_gcn_mode="tangent_last",
                sl_score_mode="tangent_euclidean",
            ),
            self.dataset,
        )
        model.eval()
        requested_users = torch.tensor([1, 2])
        users, items = model._full_sort_effective_coordinate_tables()
        direct = -model._score_scale() * model._pairwise_squared_coordinate_distance(
            users[requested_users][:, None, ...], items[None, ...]
        )
        full_scores = model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)
        torch.testing.assert_close(
            full_scores, direct, atol=3e-6, rtol=3e-5
        )


class SL16OverlayTest(unittest.TestCase):
    def test_overlay_is_small_conservative_and_requires_the_sl8_base(self):
        base_path = (
            REPO_ROOT / "baseline_config_fixed" / "SL8LHGCN_reproduction.yaml"
        )
        overlay_path = (
            REPO_ROOT / "baseline_config_fixed" / "SL16LHGCN_reproduction.yaml"
        )
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
        merged = {**base, **overlay}

        self.assertEqual(base["sl_score_mode"], "group_log")
        self.assertTrue(base["fast_one_sided_frobenius"])
        self.assertEqual(base["full_sort_user_batch_size"], 64)
        self.assertEqual(merged["model"], "SL16LHGCN")
        self.assertEqual(merged["embedding_size"], 256)
        self.assertEqual(merged["matrix_dim"], 16)
        self.assertEqual(merged["num_factors"], 1)
        self.assertEqual(merged["pairwise_loss"], "lhgcn_hinge_squared_sum")
        self.assertEqual(merged["sl_gcn_mode"], "ambient_retract")
        self.assertLessEqual(overlay["train_batch_size"], 4096)
        self.assertLessEqual(overlay["eval_user_chunk_size"], 16)
        self.assertLessEqual(overlay["eval_item_chunk_size"], 256)
        self.assertEqual(overlay["full_sort_user_batch_size"], 16)
        self.assertTrue(overlay["fast_one_sided_frobenius"])
        self.assertEqual(merged["sl_score_mode"], "group_log")

        protocol_keys = {
            "dataset",
            "seed",
            "USER_ID_FIELD",
            "ITEM_ID_FIELD",
            "RATING_FIELD",
            "load_col",
            "user_inter_num_interval",
            "item_inter_num_interval",
            "val_interval",
            "metrics",
            "topk",
            "valid_metric",
            "eval_args",
        }
        self.assertFalse(protocol_keys.intersection(overlay))

    def test_composed_recbole_config_keeps_dataset_protocol(self):
        files = [
            REPO_ROOT / "baseline_config_fixed" / "RecFormer_toy.yaml",
            REPO_ROOT / "baseline_config_fixed" / "SL8LHGCN_reproduction.yaml",
            REPO_ROOT / "baseline_config_fixed" / "SL16LHGCN_reproduction.yaml",
        ]
        with patch.object(sys, "argv", ["test_sl16lhgcn"]):
            config = Config(
                model="SL16LHGCN",
                dataset="Amazon_toy",
                config_file_list=[str(path) for path in files],
                config_dict={"use_gpu": False},
            )

        self.assertEqual(config["model"], "SL16LHGCN")
        self.assertEqual(config["dataset"], "Amazon_toy")
        self.assertEqual(config["embedding_size"], 256)
        self.assertEqual(config["matrix_dim"], 16)
        self.assertEqual(config["num_factors"], 1)
        self.assertEqual(config["eval_args"]["mode"], "full")
        self.assertEqual(config["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertEqual(config["user_inter_num_interval"], "[5,inf)")
        self.assertEqual(config["item_inter_num_interval"], "[5,inf)")


if __name__ == "__main__":
    unittest.main()
