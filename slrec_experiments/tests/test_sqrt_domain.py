"""Checks for the inverse scaling-and-squaring (sqrt-extended) SL distance.

Regression context: the plain K=12 Gregory scorer is exact only to relative
distance ~3 and beyond it returns inflated values with exploding gradients
(fp32: ~2000x value inflation and 1e13 gradients at distance 5), which
destroyed training once LieBN pushed pair distances past ~4.
"""

from __future__ import annotations

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


class DenmanBeaversTest(unittest.TestCase):
    def test_square_root_squares_back(self) -> None:
        left, right = _pair_at_distance(16, 5.0)
        relative = torch.linalg.solve(left, right)
        root = matrix_sqrt_denman_beavers(relative)
        torch.testing.assert_close(
            root @ root, relative, atol=1e-8, rtol=1e-8
        )


class SqrtExtendedDistanceTest(unittest.TestCase):
    def test_matches_plain_scorer_in_domain(self) -> None:
        left, right = _pair_at_distance(32, 2.0)
        plain = one_sided_gregory_frobenius_distance_k12(left, right)
        extended = one_sided_sqrt_extended_frobenius_distance(
            left, right, sqrt_steps=1
        )
        torch.testing.assert_close(extended, plain, atol=1e-8, rtol=1e-6)

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


if __name__ == "__main__":
    unittest.main()
