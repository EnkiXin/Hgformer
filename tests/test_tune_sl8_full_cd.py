"""Contracts for the fixed-budget SL(8) full-ranking tuning driver."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from slrec_experiments.tune_sl8_full_cd import (
    CONFIG_NAMES,
    CORE_COORD_CLIPS,
    CORE_LEARNING_RATES,
    PAPER_PARAMETERS,
    Parameters,
    Trial,
    _completed_result,
    build_stage_trials,
    finite_search_manifest,
    load_complete_result,
    main,
    runtime_protocol,
    trial_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    defaults = {
        "repo": REPO_ROOT,
        "output_root": Path("/tmp/sl8-full-output"),
        "python": "python",
        "gpu_id": "0",
        "epochs": 500,
        "profile": "extended",
        "max_new_trials": None,
        "existing_paper_result": None,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class FiniteSearchTest(unittest.TestCase):
    def test_core_is_complete_lr_clip_cartesian_product(self):
        trials = build_stage_trials("core_lr_clip")

        self.assertEqual(
            len(trials), len(CORE_LEARNING_RATES) * len(CORE_COORD_CLIPS)
        )
        self.assertEqual(
            {
                (trial.parameters.learning_rate, trial.parameters.coord_clip)
                for trial in trials
            },
            set((lr, clip) for lr in CORE_LEARNING_RATES for clip in CORE_COORD_CLIPS),
        )
        self.assertTrue(all(trial.parameters.init_std == 0.012 for trial in trials))
        self.assertTrue(all(trial.parameters.pairwise_loss == "bpr" for trial in trials))

    def test_extended_stages_are_finite_and_do_not_retrain_anchor(self):
        expected_counts = {
            "initialisation": 3,
            "regularisation": 3,
            "metric_scale": 5,
            "loss_neg": 4,
        }
        for stage, expected in expected_counts.items():
            trials = build_stage_trials(stage, PAPER_PARAMETERS)
            self.assertEqual(len(trials), expected)
            self.assertNotIn(PAPER_PARAMETERS, [trial.parameters for trial in trials])

        losses = build_stage_trials("loss_neg", PAPER_PARAMETERS)
        self.assertEqual(
            {(t.parameters.pairwise_loss, t.parameters.negative_count) for t in losses},
            {("bpr", 4), ("hinge", 1), ("hinge", 4)},
        )
        self.assertEqual(
            sorted(
                t.parameters.loss_margin
                for t in losses
                if t.parameters.pairwise_loss == "hinge"
                and t.parameters.negative_count == 1
            ),
            [0.5, 1.0],
        )

    def test_profiles_disclose_trial_counts_and_fixed_epoch_protocol(self):
        core = finite_search_manifest("core", 500)
        extended = finite_search_manifest("extended", 500)

        self.assertEqual(core["finite_search"]["maximum_training_trials_for_profile"], 12)
        self.assertEqual(
            extended["finite_search"]["maximum_training_trials_for_profile"], 27
        )
        protocol = extended["protocol"]
        self.assertEqual(protocol["validation"]["mode"], "full")
        self.assertEqual(protocol["training_budget"]["eval_step"], 500)
        self.assertEqual(protocol["training_budget"]["validation_events_per_trial"], 1)
        self.assertFalse(protocol["test_evaluated"])


class CommandSafetyTest(unittest.TestCase):
    def test_bpr_command_is_geometry_only_full_validation_at_final_epoch(self):
        args = _args(gpu_id="7")
        trial = Trial("core_lr_clip", PAPER_PARAMETERS)
        command = trial_command(
            args,
            trial,
            Path("/tmp/result.json"),
            Path("/tmp/checkpoint"),
        )

        self.assertEqual(command[command.index("--model") + 1], "SLRecGraph")
        self.assertIn("--validation-only", command)
        self.assertNotIn("--no-save", command)
        self.assertIn("--epochs=500", command)
        self.assertIn("--eval_step=500", command)
        self.assertIn("--n_layers=0", command)
        self.assertIn("--matrix_dim=8", command)
        self.assertIn("--num_factors=1", command)
        self.assertIn("--symmetric_distance=false", command)
        self.assertIn("--gpu_id=7", command)
        self.assertNotIn("--gpu_id=0", command)
        self.assertIn("--neg_sampling={'uniform': 1}", command)
        self.assertFalse(any("uni100" in argument.lower() for argument in command))
        config_argument = command[command.index("--config-files") + 1]
        self.assertEqual(
            [Path(path).name for path in config_argument.split()], list(CONFIG_NAMES)
        )

    def test_hinge_uses_loss_only_adapter(self):
        parameters = replace(PAPER_PARAMETERS, pairwise_loss="hinge")
        command = trial_command(
            _args(),
            Trial("loss_neg", parameters),
            Path("/tmp/result.json"),
            Path("/tmp/checkpoint"),
        )

        self.assertEqual(command[command.index("--model") + 1], "SLRecGraphFullTune")
        self.assertIn("--pairwise_loss=hinge", command)


class ResumeSafetyTest(unittest.TestCase):
    def _write_result(
        self,
        root: Path,
        trial: Trial,
        *,
        protocol=None,
        test_result=None,
        result_path: Path | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / f"{trial.name}.pth"
        checkpoint.write_bytes(b"checkpoint")
        result = root / "result.json" if result_path is None else result_path
        result.parent.mkdir(parents=True, exist_ok=True)
        selected_protocol = runtime_protocol(500) if protocol is None else protocol
        payload = {
            "model": trial.parameters.model_name,
            "dataset": "Amazon_cd",
            "seed": 2024,
            "epochs": 500,
            "best_valid_score": 0.031,
            "best_valid_result": {"recall@10": 0.031, "ndcg@10": 0.016},
            "test_result": test_result,
            "checkpoint_file": str(checkpoint),
            "split_fingerprints": {
                "train": {"interactions": 1, "sha256": "train"},
                "valid": {"interactions": 1, "sha256": "valid"},
                "test": {"interactions": 1, "sha256": "test"},
            },
            "tuning": {
                "stage": trial.stage,
                "trial_name": trial.name,
                "parameters": asdict(trial.parameters),
                "protocol": selected_protocol,
                "test_evaluated": False,
            },
        }
        result.write_text(json.dumps(payload), encoding="utf-8")
        return result

    def test_completed_full_result_is_skipped(self):
        trial = Trial("core_lr_clip", PAPER_PARAMETERS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._write_result(Path(temporary_directory), trial)
            payload = load_complete_result(
                result,
                trial,
                expected_protocol=runtime_protocol(500),
            )
            resumed = _completed_result(
                result,
                trial,
                expected_protocol=runtime_protocol(500),
            )

        self.assertEqual(payload["best_valid_score"], 0.031)
        self.assertIsNotNone(resumed)

    def test_different_epoch_budget_is_not_skipped(self):
        trial = Trial("core_lr_clip", PAPER_PARAMETERS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._write_result(Path(temporary_directory), trial)
            resumed = _completed_result(
                result,
                trial,
                expected_protocol=runtime_protocol(100),
            )

        self.assertIsNone(resumed)

    def test_test_touched_result_is_rejected(self):
        trial = Trial("core_lr_clip", PAPER_PARAMETERS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._write_result(
                Path(temporary_directory),
                trial,
                test_result={"recall@10": 0.04},
            )
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                _completed_result(
                    result,
                    trial,
                    expected_protocol=runtime_protocol(500),
                )

    def test_sampled_validation_result_is_rejected_not_overwritten(self):
        trial = Trial("core_lr_clip", PAPER_PARAMETERS)
        sampled_protocol = runtime_protocol(500)
        sampled_protocol["validation"]["mode"] = "uni100"
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._write_result(
                Path(temporary_directory),
                trial,
                protocol=sampled_protocol,
            )
            with self.assertRaisesRegex(RuntimeError, "not selected by full-ranking"):
                _completed_result(
                    result,
                    trial,
                    expected_protocol=runtime_protocol(500),
                )

    def test_dry_run_writes_nothing_and_uses_real_artifact_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--profile",
                        "extended",
                        "--dry-run",
                    ]
                )
            plan = json.loads(output.getvalue())

            self.assertEqual(return_code, 0)
            self.assertEqual(len(plan["instantiated_core_trials"]), 12)
            self.assertTrue(
                all(
                    "/sl8-full-tuning/stages/core_lr_clip/" in item["source"]
                    for item in plan["instantiated_core_trials"]
                )
            )
            self.assertFalse(output_root.exists())

    def test_main_skips_all_completed_core_trials_and_builds_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            results_dir = (
                output_root
                / "sl8-full-tuning"
                / "stages"
                / "core_lr_clip"
                / "results"
            )
            for trial in build_stage_trials("core_lr_clip"):
                self._write_result(
                    results_dir,
                    trial,
                    result_path=results_dir / f"{trial.name}.json",
                )

            output = io.StringIO()
            with patch(
                "slrec_experiments.tune_sl8_full_cd.run_and_tee",
                side_effect=AssertionError("completed trials must not train"),
            ), redirect_stdout(output):
                return_code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--profile",
                        "core",
                    ]
                )

            summary = json.loads(
                (output_root / "sl8-full-tuning" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(return_code, 0)
            self.assertEqual(summary["state"], "complete")
            self.assertEqual(summary["completed_training_trials"], 12)
            self.assertIsNotNone(summary["best"])
            self.assertFalse(summary["test_evaluated"])


try:
    import torch

    from recbole_gnn.model.general_recommender.slrecgraphfulltune import (
        pairwise_ranking_loss,
    )
except ModuleNotFoundError:  # Minimal local test environments need not ship torch.
    torch = None
    pairwise_ranking_loss = None


@unittest.skipUnless(torch is not None, "PyTorch is required for loss arithmetic")
class LossAblationTest(unittest.TestCase):
    def test_bpr_variant_is_exact_existing_softplus_objective(self):
        positive = torch.tensor([2.0, 0.5])
        negative = torch.tensor([0.25, 1.0])

        actual = pairwise_ranking_loss(
            positive, negative, variant="bpr", margin=99.0
        )
        expected = torch.nn.functional.softplus(negative - positive).mean()

        torch.testing.assert_close(actual, expected)

    def test_hinge_margin(self):
        positive = torch.tensor([2.0, 0.5])
        negative = torch.tensor([0.25, 1.0])
        actual = pairwise_ranking_loss(
            positive, negative, variant="hinge", margin=1.0
        )
        expected = torch.relu(1.0 + negative - positive).mean()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
