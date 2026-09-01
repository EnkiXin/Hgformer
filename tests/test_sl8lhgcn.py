"""CPU tests for the independent SL(8)-LHGCN ablation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender.sl8lhgcn import SL8LHGCN
from recbole_gnn.utils import get_model
from slrec_experiments.geometry import sl_semidistance, trace_free
from slrec_experiments.sl_lhgcn import project_ambient_to_sl


class _TinyDataset:
    def __init__(self) -> None:
        # RecBole reserves id zero.  Leaving both reserved nodes isolated also
        # exercises the explicit singular-centroid fallback.
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
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "n_layers": 2,
        "gcn_layers": 2,
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
        "sl_distance_check_samples": 4,
        "sl_log_trace_tolerance": 1e-3,
        "init_std": 0.01,
        "embedding_init": "normal",
        "reg_weight": 0.0,
        "sl_scale": 1.0,
        "coord_clip": 0.75,
        "schatten_p": 2,
        "log_terms": 6,
        "log_jitter": 1e-7,
        "symmetric_distance": True,
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


class AmbientSLProjectionTest(unittest.TestCase):
    def test_positive_negative_and_singular_components_are_explicit(self):
        positive = 2.0 * torch.eye(8)
        negative = torch.eye(8)
        negative[0, 0] = -2.0
        singular = torch.zeros(8, 8)
        nonfinite = torch.eye(8)
        nonfinite[0, 0] = float("nan")
        projected, diagnostics = project_ambient_to_sl(
            torch.stack((positive, negative, singular, nonfinite)),
            fallback_clip=0.5,
            active_mask=torch.tensor([True, True, False, False]),
            strict_membership=True,
        )

        determinants = torch.linalg.det(projected)
        torch.testing.assert_close(
            determinants,
            torch.ones_like(determinants),
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertEqual(diagnostics.total, 4)
        self.assertEqual(diagnostics.orientation_repairs, 1)
        self.assertEqual(diagnostics.singular_fallbacks, 2)
        self.assertEqual(diagnostics.active_singular_fallbacks, 0)
        self.assertEqual(diagnostics.inactive_singular_fallbacks, 2)
        self.assertEqual(diagnostics.output_membership_violations, 0)
        self.assertLess(diagnostics.max_abs_output_log_determinant, 2e-5)

    def test_projection_is_differentiable(self):
        ambient = (torch.eye(8) + 0.02 * torch.randn(3, 8, 8)).requires_grad_()
        projected, diagnostics = project_ambient_to_sl(ambient)
        projected.square().mean().backward()
        self.assertEqual(diagnostics.singular_fallbacks, 0)
        self.assertIsNotNone(ambient.grad)
        self.assertTrue(torch.isfinite(ambient.grad).all())


class SL8LHGCNTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.dataset = _TinyDataset()

    def test_legacy_runner_discovers_model(self):
        self.assertIs(get_model("SL8LHGCN"), SL8LHGCN)
        from recbole_gnn.model.general_recommender import (
            SL8LHGCN as public_model,
        )

        self.assertIs(public_model, SL8LHGCN)

    def test_combined_xavier_initialisation_matches_archived_table_bound(self):
        model = SL8LHGCN(
            _config(embedding_init="xavier_uniform_combined"), self.dataset
        )
        bound = (6.0 / (8 + 64)) ** 0.5
        all_raw = torch.cat(
            (model.user_embedding.weight, model.item_embedding.weight), dim=0
        )
        self.assertLessEqual(float(all_raw.abs().max()), bound)
        self.assertGreater(float(all_raw.abs().max()), 0.9 * bound)

    def test_ambient_forward_keeps_every_output_in_sl_and_reports_fallbacks(self):
        model = SL8LHGCN(_config(), self.dataset)
        users, items = model.forward()
        groups = torch.cat((users, items), dim=0)
        self.assertEqual(groups.shape, (8, 1, 8, 8))
        torch.testing.assert_close(
            torch.linalg.det(groups),
            torch.ones(8, 1),
            atol=3e-5,
            rtol=3e-5,
        )

        diagnostics = model.projection_diagnostics()
        self.assertEqual(diagnostics["mode"], "ambient_retract")
        self.assertEqual(diagnostics["projection_total"], 8 * 2)
        # The reserved user/item id-zero rows have no neighbours.
        self.assertGreaterEqual(diagnostics["singular_fallbacks"], 4)
        self.assertEqual(diagnostics["active_singular_fallbacks"], 0)
        self.assertEqual(diagnostics["inactive_singular_fallbacks"], 4)
        self.assertEqual(
            diagnostics["initial_group_membership"]["membership_violations"], 0
        )
        self.assertEqual(len(diagnostics["layer_membership"]), 2)
        for layer in diagnostics["layer_membership"]:
            self.assertEqual(layer["output_membership_violations"], 0)
            self.assertLess(layer["max_abs_output_log_determinant"], 1e-4)
        self.assertEqual(diagnostics["nonpositive_output_determinants"], 0)
        self.assertEqual(diagnostics["output_membership_violations"], 0)
        self.assertLess(diagnostics["max_abs_output_log_determinant"], 1e-4)
        self.assertTrue(diagnostics["materialized_full_entity_table"])
        self.assertEqual(diagnostics["materialized_group_entities"], 8)

    def test_tangent_last_matches_manual_final_layer_and_is_trace_free(self):
        model = SL8LHGCN(
            _config(sl_gcn_mode="tangent_last"), self.dataset
        )
        coordinates = trace_free(model._raw_coordinate_table())
        for _ in range(model.n_layers):
            coordinates = model._sparse_coordinate_step(
                model.norm_adj_matrix, coordinates
            )
        expected = model._to_group(coordinates)

        users, items = model.forward()
        actual = torch.cat((users, items), dim=0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            torch.linalg.det(actual),
            torch.ones(8, 1),
            atol=2e-5,
            rtol=2e-5,
        )
        diagnostics = model.projection_diagnostics()
        self.assertLess(diagnostics["max_abs_layer_trace"], 1e-6)

    def test_paper_style_self_loops_are_opt_in_and_avoid_isolated_fallback(self):
        model = SL8LHGCN(
            _config(lhgcn_include_self=True, gcn_layers=1, n_layers=1),
            self.dataset,
        )
        model.forward()
        diagonal = model.norm_adj_matrix.to_dense().diagonal()
        self.assertTrue(torch.all(diagonal > 0))
        self.assertEqual(
            model.projection_diagnostics()["singular_fallbacks"], 0
        )

    def test_faithful_hinge_loss_matches_explicit_squared_distance_sum(self):
        model = SL8LHGCN(_config(gcn_layers=1, n_layers=1), self.dataset)
        interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }
        users, items = model.forward()
        positive = model._group_distance(
            users[interaction["user_id"]], items[interaction["item_id"]]
        )
        negative = model._group_distance(
            users[interaction["user_id"]], items[interaction["neg_item_id"]]
        )
        expected = F.relu(
            positive.square() - negative.square() + model.loss_margin
        ).sum()
        actual = model.calculate_loss(interaction)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        actual.backward()
        for embedding in (model.user_embedding, model.item_embedding):
            self.assertIsNotNone(embedding.weight.grad)
            self.assertTrue(torch.isfinite(embedding.weight.grad).all())
        distance_diagnostics = model.projection_diagnostics()[
            "distance_membership"
        ]
        self.assertEqual(
            distance_diagnostics["relative_membership_violations"], 0
        )
        self.assertEqual(distance_diagnostics["nonfinite_approximate_logs"], 0)
        self.assertLess(
            distance_diagnostics["max_normalized_approximate_log_trace"],
            model.sl_log_trace_tolerance,
        )

    def test_bpr_control_supports_multiple_negatives(self):
        model = SL8LHGCN(
            _config(
                pairwise_loss="bpr_mean",
                learnable_score_scale=True,
                sl_gcn_mode="tangent_last",
            ),
            self.dataset,
        )
        materialised_sizes = []
        original_to_group = model._to_group

        def recording_to_group(coordinates):
            materialised_sizes.append(coordinates.numel() // (8 * 8))
            return original_to_group(coordinates)

        model._to_group = recording_to_group
        interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([[4, 3, 2], [1, 4, 2]]),
        }
        loss = model.calculate_loss(interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.log_score_scale.grad)
        self.assertTrue(torch.isfinite(model.log_score_scale.grad))
        # The full 8-entity table is propagated in cheap coordinates, but the
        # decoder exponentiates only two unique users and four unique items.
        self.assertEqual(materialised_sizes, [2, 4])
        diagnostics = model.projection_diagnostics()
        self.assertEqual(diagnostics["materialized_group_entities"], 6)
        self.assertFalse(diagnostics["materialized_full_entity_table"])

    def test_zero_layer_ambient_training_decodes_only_unique_batch_entities(self):
        model = SL8LHGCN(
            _config(gcn_layers=0, n_layers=0), self.dataset
        )
        materialised_sizes = []
        original_to_group = model._to_group

        def recording_to_group(coordinates):
            materialised_sizes.append(coordinates.numel() // (8 * 8))
            return original_to_group(coordinates)

        model._to_group = recording_to_group
        interaction = {
            "user_id": torch.tensor([1, 1, 2]),
            "item_id": torch.tensor([1, 2, 1]),
            "neg_item_id": torch.tensor([4, 4, 3]),
        }
        loss = model.calculate_loss(interaction)
        self.assertTrue(torch.isfinite(loss))
        # Two unique users and four unique items, rather than the complete
        # eight-entity table (or nine duplicated sampled occurrences).
        self.assertEqual(materialised_sizes, [2, 4])
        loss.backward()
        self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
        self.assertTrue(torch.isfinite(model.item_embedding.weight.grad).all())

    def test_predict_and_chunked_full_sort_use_the_same_sl_decoder(self):
        model = SL8LHGCN(_config(gcn_layers=1, n_layers=1), self.dataset)
        self.assertEqual(model.sl_score_mode, "group_log")
        model.eval()
        requested_users = torch.tensor([1, 2])
        chunked = model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)

        users, items = model._full_sort_group_tables()
        direct = model._score_groups(
            users[requested_users][:, None, ...], items[None, ...]
        )
        torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-6)

        pair_scores = model.predict(
            {
                "user_id": torch.tensor([1, 2]),
                "item_id": torch.tensor([3, 4]),
            }
        )
        torch.testing.assert_close(
            pair_scores,
            direct[torch.arange(2), torch.tensor([3, 4])],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_group_log_scores_do_not_depend_on_outer_user_batch_boundaries(self):
        model = SL8LHGCN(
            _config(
                gcn_layers=1,
                n_layers=1,
                schatten_p=2,
                log_terms=12,
                log_jitter=0.0,
                symmetric_distance=False,
                fast_one_sided_frobenius=True,
            ),
            self.dataset,
        )
        model.eval()
        users = torch.tensor([1, 2])
        combined = model.full_sort_predict({"user_id": users}).reshape(2, 5)
        split = torch.cat(
            [
                model.full_sort_predict({"user_id": user.reshape(1)}).reshape(
                    1, 5
                )
                for user in users
            ],
            dim=0,
        )
        torch.testing.assert_close(combined, split, atol=0, rtol=0)
        torch.testing.assert_close(
            torch.topk(combined, k=3, dim=1).indices,
            torch.topk(split, k=3, dim=1).indices,
            atol=0,
            rtol=0,
        )

    def test_chart_pair_distance_and_full_sort_gemm_are_numerically_equal(self):
        model = SL8LHGCN(
            _config(
                gcn_layers=1,
                n_layers=1,
                sl_gcn_mode="tangent_last",
                sl_score_mode="tangent_euclidean",
            ),
            self.dataset,
        )
        model.eval()
        requested_users = torch.tensor([1, 2])
        all_users, all_items = model._full_sort_effective_coordinate_tables()
        pair_squared = model._pairwise_squared_coordinate_distance(
            all_users[requested_users][:, None, ...],
            all_items[None, ...],
        )
        gemm_squared = model._gemm_squared_coordinate_distance(
            all_users[requested_users].reshape(2, -1),
            all_items.reshape(5, -1),
        )
        torch.testing.assert_close(
            gemm_squared, pair_squared, atol=2e-6, rtol=2e-5
        )

        full_scores = model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)
        direct_scores = -model._score_scale() * pair_squared
        torch.testing.assert_close(
            full_scores, direct_scores, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            torch.topk(full_scores, k=3, dim=1).indices,
            torch.topk(direct_scores, k=3, dim=1).indices,
            atol=0,
            rtol=0,
        )
        point_scores = model.predict(
            {
                "user_id": requested_users,
                "item_id": torch.tensor([3, 4]),
            }
        )
        torch.testing.assert_close(
            point_scores,
            direct_scores[torch.arange(2), torch.tensor([3, 4])],
            atol=1e-7,
            rtol=1e-6,
        )

    def test_chart_gemm_preserves_a_close_but_non_tied_topk_order(self):
        width = 64
        users = torch.zeros(2, width)
        users[:, 0] = torch.tensor([1.0, -1.0])
        items = users[0].repeat(5, 1)
        # Candidate distances for user 0 differ by only a few parts in 1e3,
        # while still leaving a resolvable FP32 ordering after cancellation.
        items[:, 1] = torch.tensor([0.01000, 0.01003, 0.01008, 0.2, 0.4])
        direct = (users[:, None, :] - items[None, :, :]).square().sum(dim=-1)
        gemm = SL8LHGCN._gemm_squared_coordinate_distance(users, items)
        torch.testing.assert_close(gemm, direct, atol=3e-7, rtol=3e-4)
        torch.testing.assert_close(
            torch.topk(-gemm, k=3, dim=1).indices,
            torch.topk(-direct, k=3, dim=1).indices,
            atol=0,
            rtol=0,
        )

    def test_chart_hinge_uses_squared_distance_once_and_skips_group_decoder(self):
        model = SL8LHGCN(
            _config(
                gcn_layers=1,
                n_layers=1,
                sl_gcn_mode="tangent_last",
                sl_score_mode="chart_euclidean_distance",
            ),
            self.dataset,
        )
        self.assertEqual(model.sl_score_mode, "tangent_euclidean")
        interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }
        users, items = model._effective_coordinate_tables()
        positive_squared = model._pairwise_squared_coordinate_distance(
            users[interaction["user_id"]], items[interaction["item_id"]]
        )
        negative_squared = model._pairwise_squared_coordinate_distance(
            users[interaction["user_id"]], items[interaction["neg_item_id"]]
        )
        expected = F.relu(
            positive_squared - negative_squared + model.loss_margin
        ).sum()

        # tangent_last must reuse the propagated trace-free coordinates; an
        # exp/log round trip would both waste time and change the control.
        with patch.object(
            model,
            "_to_group",
            side_effect=AssertionError("chart decoder materialised a group"),
        ):
            actual = model.calculate_loss(interaction)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
        self.assertTrue(torch.isfinite(model.item_embedding.weight.grad).all())

    def test_group_propagation_logs_each_entity_once_per_full_sort_cache(self):
        model = SL8LHGCN(
            _config(
                gcn_layers=1,
                n_layers=1,
                sl_score_mode="tangent_euclidean",
            ),
            self.dataset,
        )
        model.eval()
        from recbole_gnn.model.general_recommender import sl8lhgcn as module

        original_log = module.matrix_log_gregory
        logged_entity_counts = []

        def recording_log(groups, **kwargs):
            logged_entity_counts.append(groups.shape[0])
            return original_log(groups, **kwargs)

        with patch.object(module, "matrix_log_gregory", side_effect=recording_log):
            model.full_sort_predict({"user_id": torch.tensor([1])})
            model.full_sort_predict({"user_id": torch.tensor([2])})

        self.assertEqual(logged_entity_counts, [model.n_users + model.n_items])
        diagnostics = model.projection_diagnostics()
        self.assertEqual(diagnostics["sl_score_mode"], "tangent_euclidean")
        self.assertEqual(
            diagnostics["effective_coordinates"]["source"],
            "final_group_log",
        )

    def test_production_fast_scorer_preserves_reference_topk(self):
        model = SL8LHGCN(
            _config(
                gcn_layers=1,
                n_layers=1,
                schatten_p=2,
                log_terms=12,
                log_jitter=0.0,
                symmetric_distance=False,
                fast_one_sided_frobenius=True,
            ),
            self.dataset,
        )
        model.eval()
        requested_users = torch.tensor([1, 2])
        users, items = model._full_sort_group_tables()
        selected_users = users[requested_users][:, None, ...]
        all_items = items[None, ...]
        reference_distances = sl_semidistance(
            selected_users,
            all_items,
            p=2,
            terms=12,
            jitter=0.0,
            symmetric=False,
        ).squeeze(-1)
        fast_scores = model.full_sort_predict(
            {"user_id": requested_users}
        ).reshape(2, 5)
        torch.testing.assert_close(
            fast_scores, -reference_distances, atol=5e-6, rtol=5e-6
        )
        torch.testing.assert_close(
            torch.topk(fast_scores, k=3, dim=1).indices,
            torch.topk(-reference_distances, k=3, dim=1).indices,
            atol=0,
            rtol=0,
        )

    def test_configuration_guards_prevent_semantic_drift(self):
        with self.assertRaisesRegex(ValueError, "matrix_dim"):
            SL8LHGCN(_config(matrix_dim=4, embedding_size=16), self.dataset)
        with self.assertRaisesRegex(ValueError, "sl_layer_norm"):
            SL8LHGCN(_config(sl_layer_norm="invented"), self.dataset)
        with self.assertRaisesRegex(ValueError, "learnable_score_scale"):
            SL8LHGCN(
                _config(learnable_score_scale=True), self.dataset
            )
        with self.assertRaisesRegex(ValueError, "sl_score_mode"):
            SL8LHGCN(_config(sl_score_mode="invented"), self.dataset)
        with self.assertRaisesRegex(ValueError, "factor_aggregation: l2"):
            SL8LHGCN(
                _config(
                    sl_score_mode="tangent_euclidean",
                    factor_aggregation="l1",
                ),
                self.dataset,
            )


if __name__ == "__main__":
    unittest.main()
