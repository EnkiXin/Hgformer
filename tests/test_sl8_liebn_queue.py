from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from slrec_experiments import run_sl8_liebn_queue as queue


REPO = Path(__file__).resolve().parents[1]


class SL8LieBNQueueTest(unittest.TestCase):
    def test_absolute_python_path_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_python = root / "base-python"
            base_python.write_text("", encoding="utf-8")
            venv_python = root / "venv-python"
            os.symlink(base_python, venv_python)

            normalized = queue._absolute_without_resolving_symlinks(venv_python)

            self.assertEqual(normalized, venv_python.absolute())
            self.assertTrue(normalized.is_symlink())

    def test_rental_shards_are_disjoint_and_cover_requested_sweeps(self) -> None:
        lr_trials = queue._load_queue(
            REPO / "slrec_experiments/queues/rental_lr_l4.json"
        )
        clip_trials = queue._load_queue(
            REPO / "slrec_experiments/queues/rental_clip_l4.json"
        )
        self.assertEqual(
            {trial.learning_rate for trial in lr_trials},
            {0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.01},
        )
        self.assertEqual(
            {trial.coord_clip for trial in clip_trials},
            {0.0, 0.5, 1.0, 1.5, 2.0},
        )
        signatures = {
            (
                trial.layers,
                trial.batch_size,
                trial.learning_rate,
                trial.loss_margin,
                trial.coord_clip,
            )
            for trial in [*lr_trials, *clip_trials]
        }
        self.assertEqual(len(signatures), len(lr_trials) + len(clip_trials))
        self.assertTrue(all(trial.layers == 4 for trial in [*lr_trials, *clip_trials]))

    def test_child_contract_is_single_gpu_validation_only(self) -> None:
        trial = queue.Trial("example", 4, 16384, 0.003, 0.1, 0.75)
        args = argparse.Namespace(
            python=Path("/tmp/python"),
            repo=REPO,
            gpu_id=0,
            data_root=Path("/tmp/data"),
        )
        command = queue._command(
            args, trial, Path("/tmp/result.json"), Path("/tmp/checkpoints")
        )
        self.assertIn("--validation-only", command)
        self.assertIn("--eval_log_domain_sqrt_steps=0", command)
        self.assertIn("--log_domain_sqrt_steps=1", command)
        self.assertIn("--eval_prefilter=frobenius", command)
        self.assertIn("--eval_prefilter_candidates=4096", command)
        self.assertIn("--gpu_id=0", command)

    def test_complete_result_must_have_no_test_metrics(self) -> None:
        trial = queue.Trial("example", 4, 16384, 0.003, 0.1, 0.75)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            result = root / "result.json"
            payload = {
                "model": "SL8LHGCN",
                "dataset": "Amazon_cd",
                "epochs": 500,
                "eval_step": 10,
                "stopping_step": 2,
                "gcn_layers": 4,
                "n_layers": 4,
                "train_batch_size": 16384,
                "learning_rate": 0.003,
                "loss_margin": 0.1,
                "coord_clip": 0.75,
                "log_domain_sqrt_steps": 1,
                "eval_log_domain_sqrt_steps": 0,
                "log_domain_sqrt_iterations": 12,
                "log_domain_sqrt_residual_tolerance": 0.001,
                "log_domain_tail_tolerance": 0.001,
                "log_domain_guard_revision": "db_residual_spectral_tail_v1",
                "eval_prefilter": "frobenius",
                "eval_prefilter_candidates": 4096,
                "best_valid_result": {"recall@10": 0.1},
                "test_result": None,
                "checkpoint_file": str(checkpoint),
            }
            result.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(queue._result_is_complete(result, trial))
            payload["test_result"] = {"recall@10": 0.2}
            result.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(queue._result_is_complete(result, trial))


if __name__ == "__main__":
    unittest.main()
