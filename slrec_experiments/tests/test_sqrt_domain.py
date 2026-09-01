"""Checks for the inverse scaling-and-squaring (sqrt-extended) SL distance.

Regression context: the plain K=12 Gregory scorer is exact only to relative
distance ~3 and beyond it returns inflated values with exploding gradients
(fp32: ~2000x value inflation and 1e13 gradients at distance 5), which
destroyed training once LieBN pushed pair distances past ~4.
"""

from __future__ import annotations

import math
import unittest

import torch

from slrec_experiments.geometry import (
    matrix_sqrt_denman_beavers,
    one_sided_gregory_frobenius_distance_k12,
    one_sided_sqrt_extended_frobenius_distance,
    to_sl,
    trace_free,
)


N_DIM = 8


def _pair_at_distance(
    count: int, distance: float, *, dtype=torch.float64, seed: int = 1
) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    tangent = torch.randn(
        count, 1, N_DIM, N_DIM, generator=generator, dtype=dtype
    )
    # Symmetric trace-free tangents make exp(2T) symmetric positive definite,
    # so the principal log equals 2T at EVERY distance; generic tangents lose
    # that identity once eigenvalue imaginary parts cross pi.
    tangent = trace_free(0.5 * (tangent + tangent.transpose(-2, -1)))
    tangent = (
        tangent
        / torch.linalg.matrix_norm(tangent, ord="fro", dim=(-2, -1), keepdim=True)
        * (distance / 2.0)
    )
    return to_sl(tangent), to_sl(-tangent)  # geodesic distance == `distance`


def _rotation_pair(offset: float, *, dtype=torch.float64) -> tuple:
    """Return a pair whose relative 2-D rotation approaches eigenvalue -1."""

    angle = math.pi / 2.0 - offset
    tangent = torch.zeros(1, 1, N_DIM, N_DIM, dtype=dtype)
    tangent[0, 0, 0, 1] = -angle
    tangent[0, 0, 1, 0] = angle
    return to_sl(tangent), to_sl(-tangent)


class DenmanBeaversTest(unittest.TestCase):
    def test_square_root_squares_back(self) -> None:
        left, right = _pair_at_distance(16, 5.0)
        relative = torch.linalg.solve(left, right)
        root = matrix_sqrt_denman_beavers(relative)
        torch.testing.assert_close(
            root @ root, relative, atol=1e-8, rtol=1e-8
        )

    def test_branch_cut_rotation_fails_residual_guard(self) -> None:
        left, right = _rotation_pair(0.0)
        relative = torch.linalg.solve(left, right)
        with self.assertRaisesRegex(
            (RuntimeError, torch.linalg.LinAlgError), "Denman--Beavers"
        ):
            matrix_sqrt_denman_beavers(relative)


class SqrtExtendedDistanceTest(unittest.TestCase):
    def test_matches_plain_scorer_in_domain(self) -> None:
        left, right = _pair_at_distance(32, 2.0)
        plain = one_sided_gregory_frobenius_distance_k12(left, right)
        extended = one_sided_sqrt_extended_frobenius_distance(
            left, right, sqrt_steps=1
        )
        torch.testing.assert_close(extended, plain, atol=1e-8, rtol=1e-6)

    def test_mixed_float_dtypes_promote_before_solve(self) -> None:
        left, right = _pair_at_distance(4, 2.0, dtype=torch.float64)
        distance = one_sided_sqrt_extended_frobenius_distance(
            left.float(), right
        )
        self.assertEqual(distance.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(distance).all()))

    def test_correct_where_plain_scorer_fails(self) -> None:
        # exp(T) vs exp(-T) has exact one-sided distance ||2T||_F = distance.
        for distance in (4.0, 5.0, 6.0):
            left, right = _pair_at_distance(16, distance)
            extended = one_sided_sqrt_extended_frobenius_distance(
                left, right, sqrt_steps=1
            )
            relative_error = (extended - distance).abs().max() / distance
            self.assertLess(
                float(relative_error),
                1e-3,
                f"sqrt-extended distance wrong at d={distance}",
            )

    def test_fp32_gradients_stay_bounded_out_of_old_domain(self) -> None:
        # At d=5 the plain fp32 scorer reaches ~1e13 gradients; the extended
        # scorer must stay accurate with O(1) gradients.
        for distance in (4.0, 5.0, 6.0):
            generator = torch.Generator().manual_seed(3)
            tangent = trace_free(
                torch.randn(8, 1, N_DIM, N_DIM, generator=generator)
            )
            tangent = (
                tangent
                / torch.linalg.matrix_norm(
                    tangent, ord="fro", dim=(-2, -1), keepdim=True
                )
                * (distance / 2.0)
            )
            a = tangent.clone().requires_grad_(True)
            b = (-tangent).clone().requires_grad_(True)
            score = one_sided_sqrt_extended_frobenius_distance(
                to_sl(a), to_sl(b), sqrt_steps=1
            )
            self.assertLess(
                float((score - distance).abs().max()) / distance, 5e-3
            )
            score.square().sum().backward()
            for grad in (a.grad, b.grad):
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertLess(float(grad.abs().max()), 100.0)

    def test_two_steps_extend_further(self) -> None:
        left, right = _pair_at_distance(8, 10.0)
        extended = one_sided_sqrt_extended_frobenius_distance(
            left, right, sqrt_steps=2
        )
        relative_error = (extended - 10.0).abs().max() / 10.0
        self.assertLess(float(relative_error), 1e-3)

    def test_near_branch_rotation_fails_gregory_tail_guard(self) -> None:
        # At this offset Denman--Beavers still has a tiny square residual, but
        # the K=12 Gregory tail is too large for a trustworthy principal log.
        left, right = _rotation_pair(0.01)
        with self.assertRaisesRegex(RuntimeError, "Gregory matrix-log tail"):
            one_sided_sqrt_extended_frobenius_distance(left, right)

    def test_complete_tail_bound_rejects_cap_internal_bad_spectrum(self) -> None:
        # Each endpoint stays inside the LieBN radius 3, while the largest
        # post-sqrt Cayley value is 0.875.  The first omitted term alone looks
        # smaller than 1e-3 of the partial series, but the complete positive
        # tail is not; the geometric upper bound must reject it.
        largest = 4.0 * math.atanh(0.875)
        tangent = torch.full(
            (1, 1, N_DIM, N_DIM), 0.0, dtype=torch.float64
        )
        tangent[0, 0, 0, 0] = largest
        tangent[0, 0, 1:, 1:] = torch.eye(7, dtype=torch.float64) * (
            -largest / 7.0
        )
        endpoint_norm = torch.linalg.matrix_norm(
            tangent / 2.0, ord="fro", dim=(-2, -1)
        )
        self.assertLess(float(endpoint_norm), 3.0)
        left = to_sl(-tangent / 2.0)
        right = to_sl(tangent / 2.0)
        with self.assertRaisesRegex(RuntimeError, "Gregory matrix-log tail bound"):
            one_sided_sqrt_extended_frobenius_distance(left, right)

    def test_noncommuting_well_conditioned_pair_and_gradient(self) -> None:
        generator = torch.Generator().manual_seed(17)
        left_tangent = trace_free(
            torch.randn(6, 1, N_DIM, N_DIM, generator=generator)
        )
        right_tangent = trace_free(
            torch.randn(6, 1, N_DIM, N_DIM, generator=generator)
        )
        for tangent in (left_tangent, right_tangent):
            tangent.mul_(
                1.5
                / torch.linalg.matrix_norm(
                    tangent, ord="fro", dim=(-2, -1), keepdim=True
                )
            )
        commutator = left_tangent @ right_tangent - right_tangent @ left_tangent
        self.assertGreater(
            float(
                torch.linalg.matrix_norm(
                    commutator, ord="fro", dim=(-2, -1)
                ).min()
            ),
            0.01,
        )
        with torch.no_grad():
            reference = one_sided_sqrt_extended_frobenius_distance(
                to_sl(left_tangent), to_sl(right_tangent), sqrt_steps=2
            )
        left_tangent.requires_grad_(True)
        right_tangent.requires_grad_(True)
        distance = one_sided_sqrt_extended_frobenius_distance(
            to_sl(left_tangent), to_sl(right_tangent)
        )
        self.assertTrue(bool(torch.isfinite(distance).all()))
        torch.testing.assert_close(distance, reference, atol=2e-4, rtol=2e-4)
        distance.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(left_tangent.grad).all()))
        self.assertTrue(bool(torch.isfinite(right_tangent.grad).all()))


if __name__ == "__main__":
    unittest.main()
