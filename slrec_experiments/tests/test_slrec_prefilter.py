"""Equivalence checks for the two-stage full-sort prefilter in ``SLRec``.

RecBole must be importable (any version providing
``recbole.model.abstract_recommender``); the tests bypass ``__init__`` and
exercise only the scoring methods, so no dataset or config object is needed.
"""

from __future__ import annotations

import unittest

import torch

try:
    from slrec_experiments.slrec import SLRec
    from recbole.data.interaction import Interaction
    from recbole.trainer.trainer import Trainer
    _SLREC_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - environment-dependent
    SLRec = None
    Interaction = None
    Trainer = None
    _SLREC_IMPORT_ERROR = error

from slrec_experiments.geometry import to_sl


def _build_model(
    *,
    n_users: int,
    n_items: int,
    matrix_dim: int,
    prefilter: str,
    candidates: int,
    symmetric: bool = False,
    sqrt_steps: int = 0,
) -> "SLRec":
    model = SLRec.__new__(SLRec)
    torch.nn.Module.__init__(model)
    model.sl_score_mode = "group_log"
    model.matrix_dim = matrix_dim
    model.num_factors = 1
    model.factor_aggregation = "l2"
    model.schatten_p = 2
    model.log_terms = 12
    model.log_jitter = 1e-7
    model.symmetric_distance = symmetric
    model.fast_one_sided_frobenius = not symmetric
    model.log_domain_sqrt_steps = sqrt_steps
    model.log_domain_sqrt_iterations = 12
    model.log_domain_sqrt_residual_tolerance = 1e-3
    model.log_domain_tail_tolerance = 1e-3
    model.sl_scale = 1.0
    model.coord_clip = 1.0
    model.max_score_scale = 100.0
    model.log_score_scale = torch.nn.Parameter(
        torch.tensor(0.0, dtype=torch.float64), requires_grad=False
    )
    model.eval_user_chunk_size = 7
    model.eval_item_chunk_size = 16
    model.eval_prefilter = prefilter
    model.eval_prefilter_candidates = candidates
    model.eval_tf32 = False
    model.n_items = n_items
    model.USER_ID = "user_id"
    generator = torch.Generator().manual_seed(2026)
    model.restore_user_group = to_sl(
        0.08
        * torch.randn(
            n_users, 1, matrix_dim, matrix_dim,
            generator=generator, dtype=torch.float64,
        ),
        max_frobenius=1.0,
    )
    model.restore_item_group = to_sl(
        0.08
        * torch.randn(
            n_items, 1, matrix_dim, matrix_dim,
            generator=generator, dtype=torch.float64,
        ),
        max_frobenius=1.0,
    )
    return model


@unittest.skipIf(SLRec is None, f"recbole unavailable: {_SLREC_IMPORT_ERROR}")
class PrefilterEquivalenceTest(unittest.TestCase):
    def test_exclusions_do_not_consume_shortlist(self) -> None:
        model = _build_model(
            n_users=self.N_USERS,
            n_items=self.N_ITEMS,
            matrix_dim=self.DIM,
            prefilter="frobenius",
            candidates=2,
        )
        interaction = {"user_id": torch.tensor([0])}
        history = (torch.tensor([0]), torch.tensor([1]))
        scores = model.full_sort_predict_with_exclusions(
            interaction, history
        ).reshape(1, self.N_ITEMS)
        fill = torch.finfo(scores.dtype).min
        self.assertEqual(scores[0, 0].item(), fill)
        self.assertEqual(scores[0, 1].item(), fill)
        self.assertEqual(int((scores > fill / 2).sum().item()), 2)

    N_USERS = 20
    N_ITEMS = 120
    DIM = 4

    def _scores(self, model: "SLRec") -> torch.Tensor:
        interaction = {"user_id": torch.arange(self.N_USERS)}
        with torch.no_grad():
            return model.full_sort_predict(interaction).reshape(
                self.N_USERS, self.N_ITEMS
            )

    def test_candidates_covering_catalog_match_exact_path(self) -> None:
        exact = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS, matrix_dim=self.DIM,
                prefilter="none", candidates=self.N_ITEMS,
            )
        )
        covered = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS, matrix_dim=self.DIM,
                prefilter="frobenius", candidates=self.N_ITEMS,
            )
        )
        torch.testing.assert_close(covered, exact, atol=0, rtol=0)

    def test_sqrt_scorer_matches_exact_on_real_prefilter_shortlist(self) -> None:
        exact = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS,
                matrix_dim=self.DIM, prefilter="none",
                candidates=self.N_ITEMS, sqrt_steps=1,
            )
        )
        covered = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS,
                matrix_dim=self.DIM, prefilter="frobenius",
                candidates=self.N_ITEMS - 1, sqrt_steps=1,
            )
        )
        fill = torch.finfo(torch.float64).min
        on_shortlist = covered > fill / 2
        self.assertTrue(bool((~on_shortlist).sum(dim=1).eq(1).all()))
        torch.testing.assert_close(
            covered[on_shortlist], exact[on_shortlist], atol=1e-12, rtol=1e-12
        )

    def test_sqrt_mode_disables_the_legacy_fused_scorer(self) -> None:
        model = _build_model(
            n_users=2, n_items=3, matrix_dim=2,
            prefilter="none", candidates=3, sqrt_steps=1,
        )
        self.assertFalse(model._uses_fast_one_sided_frobenius())

    def test_shortlisted_scores_equal_exact_scores(self) -> None:
        for symmetric in (False, True):
            exact = self._scores(
                _build_model(
                    n_users=self.N_USERS, n_items=self.N_ITEMS,
                    matrix_dim=self.DIM, prefilter="none",
                    candidates=self.N_ITEMS, symmetric=symmetric,
                )
            )
            filtered = self._scores(
                _build_model(
                    n_users=self.N_USERS, n_items=self.N_ITEMS,
                    matrix_dim=self.DIM, prefilter="frobenius",
                    candidates=self.N_ITEMS - 1, symmetric=symmetric,
                )
            )
            fill = torch.finfo(torch.float64).min
            on_shortlist = filtered > fill / 2
            # Exactly one non-candidate per user; candidates score identically.
            self.assertTrue(
                bool((~on_shortlist).sum(dim=1).eq(1).all()),
                "each user must have exactly one filled non-candidate",
            )
            torch.testing.assert_close(
                filtered[on_shortlist], exact[on_shortlist], atol=1e-12, rtol=1e-12
            )

    def test_topk_ranking_is_preserved_with_small_shortlist(self) -> None:
        exact = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS, matrix_dim=self.DIM,
                prefilter="none", candidates=self.N_ITEMS,
            )
        )
        filtered = self._scores(
            _build_model(
                n_users=self.N_USERS, n_items=self.N_ITEMS, matrix_dim=self.DIM,
                prefilter="frobenius", candidates=32,
            )
        )
        # Item 0 is RecBole's padding token and is excluded by the mask-aware
        # path before shortlist selection.
        exact[:, 0] = torch.finfo(exact.dtype).min
        exact_top = exact.topk(10, dim=1).indices
        filtered_top = filtered.topk(10, dim=1).indices
        # A 32-item shortlist over a 120-item catalog may miss an occasional
        # boundary item; the test asserts the semantics (top-k mostly
        # preserved), not bitwise equality, which is seed-sensitive.
        overlap = (
            (exact_top[:, :, None] == filtered_top[:, None, :])
            .any(dim=-1)
            .float()
            .mean()
        )
        self.assertGreaterEqual(float(overlap), 0.9)

    def test_padding_and_history_cannot_displace_valid_candidates(self) -> None:
        """The shortlist budget counts eligible items, not later masks."""

        exact_model = _build_model(
            n_users=1, n_items=7, matrix_dim=2,
            prefilter="none", candidates=7,
        )
        filtered_model = _build_model(
            n_users=1, n_items=7, matrix_dim=2,
            prefilter="frobenius", candidates=4,
        )

        # Item 0 and seen items 1/2 are deliberately closest to the user in
        # both the ambient surrogate and exact group-log score.
        parameters = torch.tensor(
            [0.0, 0.01, 0.02, 0.30, 0.40, 0.50, 0.60],
            dtype=torch.float64,
        )
        users = torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2)
        items = torch.stack(
            [torch.diag(torch.stack((value.exp(), (-value).exp())))
             for value in parameters]
        ).unsqueeze(1)
        for model in (exact_model, filtered_model):
            model.restore_user_group = users.clone()
            model.restore_item_group = items.clone()

        interaction = {"user_id": torch.tensor([0])}
        history_index = (torch.tensor([0, 0]), torch.tensor([1, 2]))
        with torch.no_grad():
            exact = exact_model.full_sort_predict(interaction).reshape(1, 7)
            filtered = filtered_model.full_sort_predict_with_exclusions(
                interaction, history_index
            ).reshape(1, 7)

        fill = torch.finfo(torch.float64).min
        returned = torch.nonzero(filtered[0] > fill / 2).flatten()
        torch.testing.assert_close(returned, torch.tensor([3, 4, 5, 6]))
        self.assertEqual(filtered[0, 0].item(), fill)
        self.assertEqual(filtered[0, 1].item(), fill)
        self.assertEqual(filtered[0, 2].item(), fill)

        exact[:, [0, 1, 2]] = fill
        torch.testing.assert_close(
            filtered.topk(3, dim=1).indices,
            exact.topk(3, dim=1).indices,
        )

    def test_trainer_forwards_history_only_to_opt_in_prefilter(self) -> None:
        class MaskAwareModel(torch.nn.Module):
            eval_prefilter = "frobenius"

            def __init__(self):
                super().__init__()
                self.received = None

            def full_sort_predict_with_exclusions(
                self, interaction, excluded_index
            ):
                self.received = excluded_index
                return torch.arange(10, dtype=torch.float32)

            def full_sort_predict(self, interaction):  # pragma: no cover
                raise AssertionError("unsafe history-unaware path was used")

        model = MaskAwareModel()
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.device = torch.device("cpu")
        trainer.tot_item_num = 5
        trainer.config = {"tail_analysis": False, "popularity_analysis": False}
        history = (torch.tensor([0]), torch.tensor([1]))
        batch = (
            Interaction({"user_id": torch.tensor([1, 2])}),
            history,
            torch.tensor([0, 1]),
            torch.tensor([3, 4]),
        )

        _, scores, _, _ = trainer._full_sort_batch_eval(batch)

        self.assertIs(model.received, history)
        self.assertTrue(torch.isneginf(scores[:, 0]).all())
        self.assertTrue(torch.isneginf(scores[0, 1]))


if __name__ == "__main__":
    unittest.main()
