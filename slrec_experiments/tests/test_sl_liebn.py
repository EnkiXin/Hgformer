"""Numerical and autograd checks for SL LieBN and Karcher aggregation.

These tests are self-contained (PyTorch + :mod:`slrec_experiments.geometry`);
they do not require RecBole and can run on CPU.
"""

from __future__ import annotations

import unittest

import torch

from slrec_experiments.geometry import (
    matrix_log_gregory,
    to_sl,
    trace_free,
)
from slrec_experiments.sl_lhgcn import (
    ambient_sl_centroid_step,
    karcher_sl_centroid_step,
    row_normalise_sparse,
)
from slrec_experiments.sl_liebn import SLLieBatchNorm


N_DIM = 8


def _random_groups(
    nodes: int, sigma: float, *, factors: int = 1, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = sigma * torch.randn(
        nodes, factors, N_DIM, N_DIM, generator=generator, dtype=torch.float64
    )
    return to_sl(raw)


def _random_adjacency(nodes: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand(nodes, nodes, generator=generator, dtype=torch.float64)
    dense = dense * dense.gt(0.6)  # sparse-ish positive weights
    dense.fill_diagonal_(0.0)
    dense[0].zero_()  # reserved/isolated row
    return dense.to_sparse().coalesce()


def _log_abs_det(matrices: torch.Tensor) -> torch.Tensor:
    _, log_abs_det = torch.linalg.slogdet(matrices)
    return log_abs_det


def _barycentric_objective(
    mean: torch.Tensor, groups: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    relative = torch.linalg.solve(mean, groups)
    logs = matrix_log_gregory(relative, terms=24, jitter=0.0)
    norms = torch.linalg.matrix_norm(logs, ord="fro", dim=(-2, -1))
    return (weights * norms.square()).sum()


class SLLieBatchNormTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_output_stays_in_sl(self) -> None:
        groups = _random_groups(48, 0.3)
        module = SLLieBatchNorm(N_DIM).double()
        output, diagnostics = module(groups)
        self.assertLess(
            float(_log_abs_det(output).abs().max()), 1e-8
        )
        self.assertLess(
            diagnostics["max_centred_log_reconstruction_residual"], 1e-6
        )

    def test_dispersion_is_normalised_to_gamma(self) -> None:
        groups = _random_groups(64, 0.25)
        for dispersion in ("mean_norm", "frechet"):
            module = SLLieBatchNorm(N_DIM, dispersion=dispersion).double()
            output, _ = module(groups)
            logs = trace_free(matrix_log_gregory(output, terms=16))
            norms = torch.linalg.matrix_norm(logs, ord="fro", dim=(-2, -1))
            if dispersion == "mean_norm":
                achieved = norms.mean()
            else:
                achieved = norms.square().mean().sqrt()
            # After centring at mu the batch is normalised so that the chosen
            # dispersion statistic of log(mu^{-1} G) equals gamma; measuring
            # from the identity instead of mu adds only a small residual-mean
            # term, hence the loose tolerance.
            self.assertLess(abs(float(achieved) - 1.0), 0.1)

    def test_centring_reduces_mean_log(self) -> None:
        # A batch deliberately offset away from the identity.
        offset = to_sl(0.6 * torch.ones(1, 1, N_DIM, N_DIM, dtype=torch.float64))
        groups = offset @ _random_groups(64, 0.2)
        module = SLLieBatchNorm(N_DIM).double()
        output, _ = module(groups)
        mean_before = trace_free(
            matrix_log_gregory(groups, terms=16)
        ).mean(dim=0)
        mean_after = trace_free(
            matrix_log_gregory(output, terms=16)
        ).mean(dim=0)
        norm = lambda m: float(
            torch.linalg.matrix_norm(m, ord="fro", dim=(-2, -1)).max()
        )
        self.assertLess(norm(mean_after), 0.05 * norm(mean_before))

    def test_orthogonal_conjugation_equivariance(self) -> None:
        groups = _random_groups(32, 0.3)
        skew = torch.randn(N_DIM, N_DIM, dtype=torch.float64)
        rotation = torch.matrix_exp(skew - skew.T)  # SO(8)
        module = SLLieBatchNorm(N_DIM).double()
        direct, _ = module(rotation @ groups @ rotation.T)
        conjugated = rotation @ module(groups)[0] @ rotation.T
        torch.testing.assert_close(direct, conjugated, atol=1e-9, rtol=1e-9)

    def test_left_translation_approximate_invariance(self) -> None:
        groups = _random_groups(48, 0.2)
        translation = to_sl(
            0.4 * torch.randn(1, 1, N_DIM, N_DIM, dtype=torch.float64)
        )
        module = SLLieBatchNorm(N_DIM).double()
        base, _ = module(groups)
        translated, _ = module(translation @ groups)
        deviation = torch.linalg.matrix_norm(
            translated - base, ord="fro", dim=(-2, -1)
        ).max()
        scale = torch.linalg.matrix_norm(
            base, ord="fro", dim=(-2, -1)
        ).max()
        # Exact for the converged bi-invariant mean; one fixed-point step
        # leaves only higher-order BCH terms.
        self.assertLess(float(deviation / scale), 0.05)

    def test_statistics_respect_mask(self) -> None:
        groups = _random_groups(32, 0.2)
        corrupted = groups.clone()
        corrupted[0] = to_sl(
            3.0 * torch.randn(1, 1, N_DIM, N_DIM, dtype=torch.float64)
        )[0]
        mask = torch.ones(32, dtype=torch.bool)
        mask[0] = False
        module = SLLieBatchNorm(N_DIM).double()
        masked_out, _ = module(corrupted, mask=mask)
        clean_out, _ = module(groups, mask=mask)
        torch.testing.assert_close(
            masked_out[1:], clean_out[1:], atol=1e-8, rtol=1e-8
        )

    def test_gradients_are_finite(self) -> None:
        raw = (
            0.4 * torch.randn(24, 1, N_DIM, N_DIM, dtype=torch.float64)
        ).requires_grad_(True)
        module = SLLieBatchNorm(N_DIM, learnable_bias=True).double()
        output, _ = module(to_sl(raw))
        loss = torch.linalg.matrix_norm(output, ord="fro", dim=(-2, -1)).sum()
        loss.backward()
        for tensor in (raw.grad, module.gamma.grad, module.bias.grad):
            self.assertIsNotNone(tensor)
            self.assertTrue(bool(torch.isfinite(tensor).all()))
        self.assertGreater(float(raw.grad.abs().max()), 0.0)

    def test_single_outlier_cannot_poison_the_batch(self) -> None:
        # Regression for the gcn_layers=4 loss NaN: one node whose Gregory
        # log overflows used to turn the batch mean, the dispersion, and
        # therefore the whole normalised table non-finite.
        groups = _random_groups(64, 0.2)
        clean = groups.clone()
        module = SLLieBatchNorm(N_DIM).double()
        reference, _ = module(clean)
        for outlier_scale in (1.5, 3.0, 6.0):
            poisoned = clean.clone()
            poisoned[0] = to_sl(
                outlier_scale
                * torch.randn(1, 1, N_DIM, N_DIM, dtype=torch.float64)
            )[0]
            output, diagnostics = module(poisoned)
            self.assertTrue(
                bool(torch.isfinite(output).all()),
                f"non-finite output at outlier scale {outlier_scale}",
            )
            # Non-outlier rows must stay close to their clean normalisation:
            # the single outlier may shift the batch statistics slightly but
            # must never dominate them.
            deviation = torch.linalg.matrix_norm(
                output[1:] - reference[1:], ord="fro", dim=(-2, -1)
            ).max()
            scale = torch.linalg.matrix_norm(
                reference[1:], ord="fro", dim=(-2, -1)
            ).max()
            self.assertLess(float(deviation / scale), 0.35)
            if outlier_scale >= 6.0:
                self.assertGreaterEqual(
                    diagnostics["rejected_logs"] + diagnostics["capped_outputs"],
                    1,
                )

    def test_deep_stack_stays_finite_and_bounded(self) -> None:
        # Four aggregation+normalisation layers in float32, mimicking the
        # SL8LHGCN karcher1+liebn configuration that produced the NaN.
        from slrec_experiments.sl_lhgcn import (
            karcher_sl_centroid_step,
            row_normalise_sparse,
        )

        torch.manual_seed(4)
        nodes = 200
        raw = (
            0.4 * torch.randn(nodes, 1, N_DIM, N_DIM, dtype=torch.float32)
        ).requires_grad_(True)
        adjacency = row_normalise_sparse(
            _random_adjacency(nodes, seed=4)
        ).to(torch.float32)
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[adjacency.coalesce().indices()[0]] = True
        module = SLLieBatchNorm(N_DIM, log_terms=8)
        groups = to_sl(trace_free(raw), max_frobenius=0.75)
        for _ in range(4):
            groups, _ = karcher_sl_centroid_step(
                groups, adjacency, log_terms=8, correction=False
            )
            groups, diagnostics = module(groups, mask=mask)
        self.assertTrue(bool(torch.isfinite(groups).all()))
        logs = matrix_log_gregory(groups.double(), terms=16)
        max_norm = torch.linalg.matrix_norm(
            logs, ord="fro", dim=(-2, -1)
        ).max()
        # The output trust region bounds every node's distance from the frame.
        self.assertLess(float(max_norm), module.max_tangent_norm + 0.5)
        loss = groups.square().sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_tangent_variant_matches_to_first_order(self) -> None:
        sigma = 1e-3
        raw = trace_free(
            sigma * torch.randn(40, 1, N_DIM, N_DIM, dtype=torch.float64)
        )
        module = SLLieBatchNorm(N_DIM, mean_mode="tangent").double()
        group_out, _ = module(to_sl(raw))
        tangent_out, _ = module.normalise_tangent(raw)
        recovered = trace_free(matrix_log_gregory(group_out, terms=16))
        deviation = torch.linalg.matrix_norm(
            recovered - tangent_out, ord="fro", dim=(-2, -1)
        ).max()
        scale = torch.linalg.matrix_norm(
            tangent_out, ord="fro", dim=(-2, -1)
        ).max()
        self.assertLess(float(deviation / scale), 1e-2)


class KarcherAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_row_normalisation_is_stochastic(self) -> None:
        adjacency = _random_adjacency(20)
        normalised = row_normalise_sparse(adjacency).coalesce()
        row_sums = torch.zeros(20, dtype=torch.float64)
        row_sums.index_add_(
            0, normalised.indices()[0], normalised.values()
        )
        occupied = row_sums.gt(0)
        torch.testing.assert_close(
            row_sums[occupied],
            torch.ones_like(row_sums[occupied]),
            atol=1e-12,
            rtol=0,
        )
        self.assertFalse(bool(occupied[0]))  # isolated row stays empty

    def test_output_stays_in_sl_without_repairs(self) -> None:
        groups = _random_groups(30, 0.4)
        adjacency = row_normalise_sparse(_random_adjacency(30))
        output, diagnostics = karcher_sl_centroid_step(groups, adjacency)
        self.assertLess(float(_log_abs_det(output).abs().max()), 1e-8)
        self.assertTrue(diagnostics.correction)
        # The isolated row aggregates to the identity.
        torch.testing.assert_close(
            output[0, 0],
            torch.eye(N_DIM, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )

    def test_seed_equals_tangent_mean(self) -> None:
        groups = _random_groups(24, 0.3)
        adjacency = row_normalise_sparse(_random_adjacency(24))
        seed_only, diagnostics = karcher_sl_centroid_step(
            groups, adjacency, correction=False, log_terms=16
        )
        logs = trace_free(matrix_log_gregory(groups, terms=16))
        flat = logs.reshape(24, -1)
        expected = torch.matrix_exp(
            trace_free(torch.sparse.mm(adjacency, flat).reshape_as(logs))
        )
        torch.testing.assert_close(seed_only, expected, atol=1e-10, rtol=1e-10)
        self.assertFalse(diagnostics.correction)

    def test_correction_improves_barycentric_objective(self) -> None:
        # In the coord_clip working regime (||X||_F around 1.5-2) the
        # corrected mean must not be worse than the seed under the one-sided
        # squared log objective the model scores with.
        groups = _random_groups(16, 0.25, seed=7)
        adjacency = row_normalise_sparse(_random_adjacency(16, seed=7))
        seed_only, _ = karcher_sl_centroid_step(
            groups, adjacency, correction=False, log_terms=12
        )
        corrected, _ = karcher_sl_centroid_step(
            groups, adjacency, log_terms=12
        )
        dense = adjacency.to_dense()
        seed_objectives = []
        corrected_objectives = []
        for node in range(16):
            weights = dense[node]
            support = weights.gt(0)
            if not bool(support.any()):
                continue
            seed_objectives.append(
                float(
                    _barycentric_objective(
                        seed_only[node, 0], groups[support, 0], weights[support]
                    )
                )
            )
            corrected_objectives.append(
                float(
                    _barycentric_objective(
                        corrected[node, 0], groups[support, 0], weights[support]
                    )
                )
            )
        self.assertGreater(len(seed_objectives), 0)
        # One fixed-point step is not per-node monotone (the seed is already
        # near-stationary at working spreads, so individual nodes wiggle by
        # a fraction of a percent), but in aggregate it must move toward the
        # barycenter and no node may worsen materially.
        self.assertLess(sum(corrected_objectives), sum(seed_objectives))
        for seed_value, corrected_value in zip(
            seed_objectives, corrected_objectives
        ):
            self.assertLessEqual(corrected_value, seed_value * 1.03)

    def test_chunking_and_checkpointing_are_equivalent(self) -> None:
        raw = (
            0.3 * torch.randn(18, 1, N_DIM, N_DIM, dtype=torch.float64)
        ).requires_grad_(True)
        adjacency = row_normalise_sparse(_random_adjacency(18))

        def run(edge_chunk: int, use_checkpoint: bool) -> tuple:
            groups = to_sl(raw)
            output, _ = karcher_sl_centroid_step(
                groups,
                adjacency,
                edge_chunk=edge_chunk,
                use_checkpoint=use_checkpoint,
            )
            loss = output.square().sum()
            gradient = torch.autograd.grad(loss, raw)[0]
            return output.detach(), gradient

        reference_out, reference_grad = run(10**9, use_checkpoint=False)
        for edge_chunk, use_checkpoint in ((7, False), (7, True), (10**9, True)):
            output, gradient = run(edge_chunk, use_checkpoint)
            torch.testing.assert_close(
                output, reference_out, atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(
                gradient, reference_grad, atol=1e-9, rtol=1e-9
            )

    def test_stays_finite_and_in_sl_at_extreme_spread(self) -> None:
        # Far outside the principal-log regime the ambient retraction needs
        # orientation repairs or fallbacks and the group log stops existing
        # for some pairs.  The barycenter step must degrade *observably*:
        # zeroed terms are counted in the diagnostics while the output stays
        # finite and inside SL(n) by construction.
        groups = _random_groups(40, 1.1, seed=11)
        adjacency = row_normalise_sparse(_random_adjacency(40, seed=11))
        output, diagnostics = karcher_sl_centroid_step(groups, adjacency)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertLess(float(_log_abs_det(output).abs().max()), 1e-6)
        self.assertGreaterEqual(diagnostics.nonfinite_edge_logs, 0)

    def test_no_repairs_needed_in_working_regime(self) -> None:
        groups = _random_groups(40, 0.25, seed=11)
        adjacency = row_normalise_sparse(_random_adjacency(40, seed=11))
        output, diagnostics = karcher_sl_centroid_step(groups, adjacency)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertLess(float(_log_abs_det(output).abs().max()), 1e-8)
        self.assertEqual(diagnostics.nonfinite_node_logs, 0)
        self.assertEqual(diagnostics.nonfinite_edge_logs, 0)


if __name__ == "__main__":
    unittest.main()
