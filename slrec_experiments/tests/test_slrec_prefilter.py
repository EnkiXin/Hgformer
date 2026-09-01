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
    _SLREC_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - environment-dependent
    SLRec = None
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
        exact_top = exact.topk(10, dim=1).indices
        filtered_top = filtered.topk(10, dim=1).indices
        torch.testing.assert_close(exact_top, filtered_top)


if __name__ == "__main__":
    unittest.main()
