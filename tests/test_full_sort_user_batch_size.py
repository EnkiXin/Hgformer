"""Contracts for exact full-ranking outer-user batching."""

import unittest

from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader


class _Dataset:
    item_num = 61_681


def _loader(config):
    loader = FullSortEvalDataLoader.__new__(FullSortEvalDataLoader)
    loader.config = config
    loader.dataset = _Dataset()
    loader.is_sequential = False
    loader.pr = 0
    loader._init_batch_size_and_step()
    return loader


class FullSortUserBatchSizeTest(unittest.TestCase):
    def test_legacy_pair_budget_is_unchanged_without_opt_in(self):
        loader = _loader({"eval_batch_size": 1_048_576})
        self.assertEqual(loader.step, 16)
        self.assertEqual(loader.batch_size, 16 * _Dataset.item_num)

    def test_explicit_user_batch_avoids_outer_chunk_fragmentation(self):
        loader = _loader(
            {
                "eval_batch_size": 1_048_576,
                "full_sort_user_batch_size": 64,
            }
        )
        self.assertEqual(loader.step, 64)
        self.assertEqual(loader.batch_size, 64 * _Dataset.item_num)

    def test_explicit_user_batch_must_be_positive_integer(self):
        for invalid in (0, -1, 1.5, True, "many"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _loader(
                        {
                            "eval_batch_size": 1_048_576,
                            "full_sort_user_batch_size": invalid,
                        }
                    )


if __name__ == "__main__":
    unittest.main()
