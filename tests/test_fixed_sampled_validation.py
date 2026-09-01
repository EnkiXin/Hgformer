"""Tests for fixed-negative sampled validation in ``SLRecGraphTrainer``."""

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from recbole.trainer import Trainer

from recbole_gnn.trainer import SLRecGraphTrainer
from recbole_gnn.utils import get_trainer


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class FixedSampledValidationTest(unittest.TestCase):
    def setUp(self):
        self.trainer = SLRecGraphTrainer.__new__(SLRecGraphTrainer)
        self.trainer.config = {
            "fixed_sampled_validation": True,
            "fixed_sampled_validation_seed": 77,
            "seed": 2024,
            "eval_args": {"mode": "uni100"},
            "eval_neg_sample_args": {
                "strategy": "by",
                "by": 100,
                "distribution": "uniform",
            },
        }

    def test_model_specific_trainer_is_discovered(self):
        self.assertIs(get_trainer("general", "SLRecGraph"), SLRecGraphTrainer)

    def test_repeated_valid_epoch_replays_candidates_and_restores_rng(self):
        observed = []

        def fake_valid_epoch(_trainer, _valid_data, show_progress=False):
            candidates = np.random.randint(1, 1000, size=32)
            torch_values = torch.randint(1, 1000, size=(16,))
            python_value = random.random()
            observed.append((candidates, torch_values, python_value, show_progress))
            return float(candidates[0]), {"candidate_sum": int(candidates.sum())}

        np.random.seed(123)
        random.seed(456)
        torch.manual_seed(789)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()

        with patch.object(Trainer, "_valid_epoch", new=fake_valid_epoch):
            first = self.trainer._valid_epoch(object(), show_progress=True)
            second = self.trainer._valid_epoch(object(), show_progress=True)

        np.testing.assert_array_equal(observed[0][0], observed[1][0])
        torch.testing.assert_close(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][2:], observed[1][2:])
        self.assertEqual(first, second)
        self.assertTrue(_numpy_state_equal(numpy_state, np.random.get_state()))
        self.assertEqual(python_state, random.getstate())
        torch.testing.assert_close(torch_state, torch.random.get_rng_state())

    def test_full_ranking_bypasses_fixed_rng_wrapper(self):
        self.trainer.config["eval_args"]["mode"] = "full"
        observed = []

        def fake_valid_epoch(_trainer, _valid_data, show_progress=False):
            observed.append(np.random.randint(1, 2 ** 30, size=16))
            return 0.0, {}

        np.random.seed(123)
        initial_state = np.random.get_state()
        with patch.object(Trainer, "_valid_epoch", new=fake_valid_epoch):
            self.trainer._valid_epoch(object())
            self.trainer._valid_epoch(object())

        self.assertFalse(np.array_equal(observed[0], observed[1]))
        self.assertFalse(_numpy_state_equal(initial_state, np.random.get_state()))

    def test_disabled_flag_leaves_sampled_validation_unchanged(self):
        self.trainer.config["fixed_sampled_validation"] = False
        observed = []

        def fake_valid_epoch(_trainer, _valid_data, show_progress=False):
            observed.append(np.random.randint(1, 2 ** 30, size=16))
            return 0.0, {}

        np.random.seed(123)
        with patch.object(Trainer, "_valid_epoch", new=fake_valid_epoch):
            self.trainer._valid_epoch(object())
            self.trainer._valid_epoch(object())

        self.assertFalse(np.array_equal(observed[0], observed[1]))

    def test_rng_is_restored_when_validation_raises(self):
        np.random.seed(321)
        torch.manual_seed(654)
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()

        def failing_valid_epoch(_trainer, _valid_data, show_progress=False):
            np.random.random(5)
            torch.rand(5)
            raise RuntimeError("evaluation failed")

        with patch.object(Trainer, "_valid_epoch", new=failing_valid_epoch):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.trainer._valid_epoch(object())

        self.assertTrue(_numpy_state_equal(numpy_state, np.random.get_state()))
        torch.testing.assert_close(torch_state, torch.random.get_rng_state())


if __name__ == "__main__":
    unittest.main()
