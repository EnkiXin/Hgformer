"""Numerical and RecBole-interface tests for the controlled MixedGeoRec model."""

import math
import unittest

import torch

try:
    from slrec_experiments.mixedgeo import (
        MixedGeoRec,
        poincare_distance,
        poincare_distance_sq,
        poincare_expmap0,
        sphere_expmap0,
        spherical_distance,
        spherical_distance_sq,
    )
except ModuleNotFoundError:
    # Allows ``cd slrec_experiments && python -m unittest discover -s tests``;
    # this invocation also prevents the repository's legacy RecBole fork from
    # shadowing the clean RecBole 1.2.1 environment used by the experiments.
    from mixedgeo import (
        MixedGeoRec,
        poincare_distance,
        poincare_distance_sq,
        poincare_expmap0,
        sphere_expmap0,
        spherical_distance,
        spherical_distance_sq,
    )


class _TinyCOO:
    def __init__(self):
        self.row = [1, 1, 2, 3, 3, 3]
        self.col = [1, 2, 2, 1, 3, 4]

    def tocoo(self):
        return self


class _TinyDataset:
    def __init__(self):
        self._sizes = {"user_id": 4, "item_id": 6}

    def num(self, field):
        return self._sizes[field]

    def inter_matrix(self, form="coo"):
        if form != "coo":
            raise ValueError("test dataset only exposes COO")
        return _TinyCOO()

    def get_interactions(self):
        # Compatibility with the repository's legacy fork when tests are run
        # from its root; official RecBole 1.2.1 does not call this method here.
        return (
            torch.tensor(_TinyCOO().row, dtype=torch.long),
            torch.tensor(_TinyCOO().col, dtype=torch.long),
        )


def _config(**updates):
    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "embedding_size": 12,
        "hyperbolic_dim": 4,
        "euclidean_dim": 4,
        "spherical_dim": 4,
        "gate_mode": "global",
        "reg_weight": 1e-6,
        "eval_item_chunk_size": 2,
        "initializer_range": 0.04,
        # Harmless compatibility keys for the repository's older local fork.
        "tail_analysis": False,
        "popularity_analysis": False,
    }
    config.update(updates)
    return config


class GeometryTests(unittest.TestCase):
    def test_poincare_expmap_distance_and_gradient(self):
        tangent = torch.tensor(
            [[0.13, -0.08, 0.04], [-0.03, 0.11, 0.07]],
            dtype=torch.float64,
            requires_grad=True,
        )
        points = poincare_expmap0(tangent, curvature=0.7)
        boundary = 1.0 / math.sqrt(0.7)
        self.assertTrue(torch.all(torch.linalg.vector_norm(points, dim=-1) < boundary))

        forward = poincare_distance_sq(points[0], points[1], curvature=0.7)
        backward = poincare_distance_sq(points[1], points[0], curvature=0.7)
        self.assertTrue(torch.allclose(forward, backward, atol=1e-12, rtol=1e-10))
        self.assertEqual(
            poincare_distance_sq(points[0], points[0], curvature=0.7).item(),
            0.0,
        )
        forward.backward()
        self.assertIsNotNone(tangent.grad)
        self.assertTrue(torch.isfinite(tangent.grad).all())
        self.assertGreater(tangent.grad.abs().sum().item(), 0.0)

    def test_poincare_radial_distance(self):
        tangent = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
        origin = poincare_expmap0(torch.zeros_like(tangent))
        point = poincare_expmap0(tangent)
        # The Poincare metric at the origin has conformal factor two.
        expected = 2.0 * torch.linalg.vector_norm(tangent, dim=-1)
        actual = poincare_distance(origin, point)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-9))

    def test_sphere_expmap_unit_norm_distance_and_gradient(self):
        tangent = torch.tensor(
            [[0.2, -0.1, 0.05], [0.0, math.pi / 3.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        points = sphere_expmap0(tangent)
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(points, dim=-1),
                torch.ones(2, dtype=torch.float64),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        distance_sq = spherical_distance_sq(points[0], points[1])
        reverse_sq = spherical_distance_sq(points[1], points[0])
        self.assertTrue(torch.allclose(distance_sq, reverse_sq, atol=1e-12))
        self.assertLess(spherical_distance_sq(points[0], points[0]).item(), 1e-12)
        distance_sq.backward()
        self.assertTrue(torch.isfinite(tangent.grad).all())
        self.assertGreater(tangent.grad.abs().sum().item(), 0.0)

    def test_sphere_known_quarter_circle(self):
        north = sphere_expmap0(torch.zeros(1, 2, dtype=torch.float64))
        equator = sphere_expmap0(
            torch.tensor([[math.pi / 2.0, 0.0]], dtype=torch.float64)
        )
        actual = spherical_distance(north, equator)
        self.assertTrue(
            torch.allclose(
                actual,
                torch.tensor([math.pi / 2.0], dtype=torch.float64),
                atol=1e-10,
            )
        )


class MixedGeoRecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.dataset = _TinyDataset()
        self.interaction = {
            "user_id": torch.tensor([1, 2, 3], dtype=torch.long),
            "item_id": torch.tensor([1, 2, 3], dtype=torch.long),
            "neg_item_id": torch.tensor([4, 3, 1], dtype=torch.long),
        }

    def test_pairwise_loss_predict_full_sort_and_gradient(self):
        model = MixedGeoRec(_config(), self.dataset)
        self.assertEqual(sum(model.branch_dims.values()), 12)
        self.assertEqual(
            model.active_branches, ("hyperbolic", "euclidean", "spherical")
        )

        loss = model.calculate_loss(self.interaction)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
        self.assertTrue(torch.isfinite(model.item_embedding.weight.grad).all())

        model.eval()
        with torch.no_grad():
            prediction = model.predict(self.interaction)
            all_scores = model.full_sort_predict(
                {"user_id": torch.tensor([1, 3], dtype=torch.long)}
            )
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(all_scores.shape, (2 * self.dataset.num("item_id"),))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(all_scores).all())

    def test_he_configuration_infers_equal_split_when_sphere_is_disabled(self):
        config = _config()
        config.pop("hyperbolic_dim")
        config.pop("euclidean_dim")
        config["spherical_dim"] = 0
        model = MixedGeoRec(config, self.dataset)
        self.assertEqual(model.branch_dims, {"hyperbolic": 6, "euclidean": 6, "spherical": 0})
        self.assertEqual(model.active_branches, ("hyperbolic", "euclidean"))
        score = model.predict(self.interaction)
        self.assertTrue(torch.isfinite(score).all())

    def test_popularity_and_entity_gates_are_normalized_and_differentiable(self):
        users = self.interaction["user_id"]
        items = self.interaction["item_id"]
        for mode in ("popularity", "learned"):
            with self.subTest(mode=mode):
                model = MixedGeoRec(_config(gate_mode=mode), self.dataset)
                weights = model._gate_weights(users, items)
                self.assertEqual(weights.shape, (3, 3))
                self.assertTrue(
                    torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)
                )
                loss = model.calculate_loss(self.interaction)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                if mode == "popularity":
                    gate_gradients = [
                        parameter.grad
                        for parameter in model.popularity_gate.parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gate_gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gate_gradients))
                else:
                    self.assertEqual(model.gate_mode, "entity")
                    self.assertTrue(
                        torch.isfinite(model.user_gate_embedding.weight.grad).all()
                    )


if __name__ == "__main__":
    unittest.main()
