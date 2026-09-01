import argparse
import unittest

from slrec_experiments.sweep_sl8_scoring_chunks import (
    candidate_grid,
    parse_positive_int_csv,
    scorer_call_count,
)


class SL8ChunkSweepTest(unittest.TestCase):
    def test_positive_int_csv_deduplicates_in_order(self):
        self.assertEqual(parse_positive_int_csv("8,17,8, 32"), (8, 17, 32))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_positive_int_csv("8,0")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_positive_int_csv("")

    def test_global_scorer_call_count(self):
        self.assertEqual(
            scorer_call_count(10, 11, user_chunk=4, item_chunk=5), 9
        )

    def test_outer_batches_can_add_user_chunks(self):
        # Global chunks: ceil(10/4)=3.  Outer batches [6,4] instead produce
        # ceil(6/4)+ceil(4/4)=3 here, while [5,5] produce 2+2=4.
        self.assertEqual(
            scorer_call_count(
                10, 11, user_chunk=4, item_chunk=5, outer_users=5
            ),
            12,
        )

    def test_candidate_grid_skips_pair_cap_and_orders_by_size(self):
        selected, skipped = candidate_grid((8, 16), (32, 64), max_pairs=512)
        self.assertEqual(selected, [(8, 32), (8, 64), (16, 32)])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["pairs"], 1024)


if __name__ == "__main__":
    unittest.main()
