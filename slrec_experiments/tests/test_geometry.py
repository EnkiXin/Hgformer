"""Numerical and autograd checks for :mod:`slrec_experiments.geometry`."""

from __future__ import annotations

import unittest

import torch

from slrec_experiments.geometry import (
    matrix_log_gregory,
    one_sided_gregory_frobenius_distance_k12,
    schatten_norm,
    sl_semidistance,
    to_sl,
    trace_free,
)


class SpecialLinearGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_trace_free_projection(self) -> None:
        raw = torch.randn(7, 4, 4, dtype=torch.float64)
        projected = trace_free(raw)
        traces = projected.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        torch.testing.assert_close(traces, torch.zeros_like(traces), atol=1e-12, rtol=0)

    def test_exponential_has_unit_determinant(self) -> None:
        raw = 0.2 * torch.randn(6, 5, 5, dtype=torch.float64)
        group = to_sl(raw, max_frobenius=1.0)
        determinant = torch.linalg.det(group)
        torch.testing.assert_close(
            determinant, torch.ones_like(determinant), atol=2e-12, rtol=2e-12
        )

    def test_matrix_log_recovers_commuting_tangent(self) -> None:
        diagonal = torch.tensor([0.35, -0.10, -0.25], dtype=torch.float64)
        tangent = torch.diag(diagonal)
        group = torch.matrix_exp(tangent)
        recovered = matrix_log_gregory(group, terms=18, jitter=0.0)
        torch.testing.assert_close(recovered, tangent, atol=1e-12, rtol=1e-12)

    def test_matrix_log_recovers_small_noncommuting_tangent(self) -> None:
        raw = 0.08 * torch.randn(4, 3, 3, dtype=torch.float64)
        tangent = trace_free(raw)
        group = torch.matrix_exp(tangent)
        recovered = matrix_log_gregory(group, terms=18, jitter=0.0)
        torch.testing.assert_close(recovered, tangent, atol=2e-12, rtol=2e-12)

    def test_schatten_norm_orders(self) -> None:
        matrix = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
        self.assertAlmostEqual(schatten_norm(matrix, p=1).item(), 7.0, places=12)
        self.assertAlmostEqual(schatten_norm(matrix, p=2).item(), 5.0, places=12)
        self.assertAlmostEqual(
            schatten_norm(matrix, p="inf").item(), 4.0, places=12
        )

    def test_semidistance_is_symmetric_nonnegative_and_zero_on_diagonal(self) -> None:
        left = to_sl(0.1 * torch.randn(5, 3, 3, dtype=torch.float64))
        right = to_sl(0.1 * torch.randn(5, 3, 3, dtype=torch.float64))
        forward = sl_semidistance(left, right, terms=16, jitter=0.0)
        reverse = sl_semidistance(right, left, terms=16, jitter=0.0)
        diagonal = sl_semidistance(left, left, terms=16, jitter=0.0)

        self.assertTrue(torch.all(forward >= 0))
        torch.testing.assert_close(forward, reverse, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal), atol=1e-12, rtol=0)

    def test_semidistance_broadcasts_for_full_sort_chunks(self) -> None:
        users = to_sl(0.05 * torch.randn(2, 1, 3, 3, dtype=torch.float64))
        items = to_sl(0.05 * torch.randn(1, 4, 3, 3, dtype=torch.float64))
        distances = sl_semidistance(users, items, terms=12, jitter=0.0)
        self.assertEqual(distances.shape, (2, 4))
        self.assertTrue(torch.isfinite(distances).all())

    def test_fast_k12_one_sided_frobenius_matches_reference_and_gradients(self) -> None:
        raw_users = (
            0.08 * torch.randn(3, 1, 8, 8, dtype=torch.float64)
        ).requires_grad_()
        raw_items = (
            0.08 * torch.randn(1, 11, 8, 8, dtype=torch.float64)
        ).requires_grad_()
        users = to_sl(raw_users, max_frobenius=0.75)
        items = to_sl(raw_items, max_frobenius=0.75)
        reference = sl_semidistance(
            users,
            items,
            p=2,
            terms=12,
            jitter=1e-7,
            symmetric=False,
        )
        fast = one_sided_gregory_frobenius_distance_k12(
            users, items, jitter=1e-7
        )
        torch.testing.assert_close(fast, reference, atol=1e-12, rtol=1e-12)

        reference_gradients = torch.autograd.grad(
            reference.mean(), (raw_users, raw_items), retain_graph=True
        )
        fast_gradients = torch.autograd.grad(
            fast.mean(), (raw_users, raw_items)
        )
        for reference_gradient, gradient in zip(
            reference_gradients, fast_gradients
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0.0)
            torch.testing.assert_close(
                gradient, reference_gradient, atol=1e-11, rtol=1e-11
            )

    def test_fast_k12_fails_on_singular_cayley_denominator(self) -> None:
        # For even n, both I and -I have determinant +1, while B + A is
        # singular and the principal-log/Gregory path is outside its domain.
        identity = torch.eye(8).unsqueeze(0)
        with self.assertRaisesRegex(RuntimeError, "Cayley denominator"):
            one_sided_gregory_frobenius_distance_k12(
                identity, -identity, jitter=0.0
            )

    def test_gradients_are_finite(self) -> None:
        raw_left = (0.05 * torch.randn(3, 3, 3, dtype=torch.float64)).requires_grad_()
        raw_right = (0.05 * torch.randn(3, 3, 3, dtype=torch.float64)).requires_grad_()
        left = to_sl(raw_left, max_frobenius=0.8)
        right = to_sl(raw_right, max_frobenius=0.8)
        loss = sl_semidistance(left, right, terms=14, jitter=0.0).mean()
        loss.backward()

        for gradient in (raw_left.grad, raw_right.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0.0)

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            trace_free(torch.randn(2, 3))
        with self.assertRaises(ValueError):
            matrix_log_gregory(torch.eye(2), terms=0)
        with self.assertRaises(ValueError):
            schatten_norm(torch.eye(2), p=0.5)
        with self.assertRaises(ValueError):
            schatten_norm(torch.eye(2), p="not-an-order")


if __name__ == "__main__":
    unittest.main()
