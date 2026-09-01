import argparse
import unittest

import torch

from slrec_experiments.benchmark_sl8_compile import (
    parse_positive_int_csv,
    production_sl8_score,
    requested_shapes,
)


class SL8CompileBenchmarkTest(unittest.TestCase):
    def test_csv_parser(self):
        self.assertEqual(parse_positive_int_csv("17, 32,17"), (17, 32))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_positive_int_csv("17,-1")

    def test_shape_cap(self):
        selected, skipped = requested_shapes((17,), (1024, 4096), 34_816)
        self.assertEqual(selected, [(17, 1024)])
        self.assertEqual(skipped[0]["pairs"], 69_632)

    def test_production_score_shape_and_finiteness(self):
        identity = torch.eye(8)
        users = identity.reshape(1, 1, 1, 8, 8).repeat(2, 1, 1, 1, 1)
        items = identity.reshape(1, 1, 1, 8, 8).repeat(1, 3, 1, 1, 1)
        scores = production_sl8_score(users, items)
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertTrue(bool(scores.eq(0).all()))


if __name__ == "__main__":
    unittest.main()
