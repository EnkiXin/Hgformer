"""Safety and continuation contracts for geometry-only SL(4) tuning."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from slrec_experiments.tune_slrec_geometry_cd import (
    CONFIG_NAMES,
    GEOMETRY_DEFAULTS,
    Parameters,
    Trial,
    build_trials,
    load_complete_result,
    main,
    resolve_base_parameters,
    trial_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    defaults = {
        "stage": "lr",
        "repo": REPO_ROOT,
        "output_root": Path("/tmp/slrec-output"),
        "python": "python",
        "gpu_id": "0",
        "epochs": 500,
        "eval_step": 5,
        "stopping_step": 12,
        "values": None,
        "learning_rate": None,
        "coord_clip": None,
        "init_std": None,
        "reg_weight": None,
        "resume_from": None,
        "existing_base_result": None,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class StageConstructionTest(unittest.TestCase):
    def test_stage_one_has_requested_learning_rate_grid(self):
        base = Parameters(**GEOMETRY_DEFAULTS)
        trials = build_trials("lr", base)

        self.assertEqual(
            [trial.parameters.learning_rate for trial in trials],
            [5e-4, 1e-3, 3e-3],
        )
        self.assertTrue(all(trial.parameters.coord_clip == 0.75 for trial in trials))
        self.assertTrue(all(trial.parameters.init_std == 0.012 for trial in trials))
        self.assertTrue(all(trial.parameters.reg_weight == 0.0 for trial in trials))

    def test_later_stage_uses_explicit_preceding_selection(self):
        args = _args(
            stage="init_std",
            learning_rate=1e-3,
            coord_clip=0.5,
        )
        base = resolve_base_parameters(args)
        trials = build_trials("init_std", base)

        self.assertEqual([t.parameters.init_std for t in trials], [0.008, 0.012, 0.02])
        self.assertTrue(all(t.parameters.learning_rate == 1e-3 for t in trials))
        self.assertTrue(all(t.parameters.coord_clip == 0.5 for t in trials))

    def test_later_stage_refuses_implicit_stale_defaults(self):
        with self.assertRaisesRegex(ValueError, "continue the preceding selection"):
            resolve_base_parameters(_args(stage="reg_weight", learning_rate=1e-3))

    def test_resume_summary_supplies_previous_best(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = Path(temporary_directory) / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "best": {
                            "parameters": {
                                "learning_rate": 5e-4,
                                "coord_clip": 1.0,
                                "init_std": 0.02,
                                "reg_weight": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            base = resolve_base_parameters(
                _args(stage="reg_weight", resume_from=summary)
            )

        self.assertEqual(base, Parameters(5e-4, 1.0, 0.02, 0.0))


class CommandSafetyTest(unittest.TestCase):
    def test_command_is_validation_only_and_saves_checkpoint(self):
        args = _args()
        trial = Trial("lr", Parameters(**GEOMETRY_DEFAULTS))
        command = trial_command(
            args,
            trial,
            Path("/tmp/result.json"),
            Path("/tmp/checkpoint"),
        )

        self.assertIn("--validation-only", command)
        self.assertNotIn("--no-save", command)
        self.assertFalse(any("test" in argument.lower() for argument in command))
        self.assertIn("--n_layers=0", command)
        self.assertIn("--seed=2024", command)
        self.assertIn("--eval_step=5", command)
        config_argument = command[command.index("--config-files") + 1]
        self.assertEqual(
            [Path(path).name for path in config_argument.split()],
            list(CONFIG_NAMES),
        )


class ResultSafetyTest(unittest.TestCase):
    def _write_result(self, root: Path, *, test_result=None, metadata=True) -> Path:
        checkpoint = root / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        result = root / "result.json"
        payload = {
            "model": "SLRecGraph",
            "dataset": "Amazon_cd",
            "seed": 2024,
            "best_valid_score": 0.1,
            "best_valid_result": {"recall@10": 0.1},
            "test_result": test_result,
            "checkpoint_file": str(checkpoint),
            "config_files": [
                str(REPO_ROOT / "baseline_config_fixed" / name)
                for name in CONFIG_NAMES
            ],
        }
        if metadata:
            payload["tuning"] = {
                "parameters": GEOMETRY_DEFAULTS,
            }
        result.write_text(json.dumps(payload), encoding="utf-8")
        return result

    def test_completed_result_requires_checkpoint_and_never_test_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = self._write_result(root)
            payload = load_complete_result(
                result,
                expected_parameters=Parameters(**GEOMETRY_DEFAULTS),
            )
            self.assertEqual(payload["best_valid_score"], 0.1)

            touched_test = self._write_result(root, test_result={"recall@10": 0.2})
            with self.assertRaisesRegex(RuntimeError, "touched the test split"):
                load_complete_result(
                    touched_test,
                    expected_parameters=Parameters(**GEOMETRY_DEFAULTS),
                )

    def test_dry_run_recognises_legacy_3e3_base_without_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = self._write_result(root, metadata=False)
            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main(
                    [
                        "--stage",
                        "lr",
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(root / "output"),
                        "--existing-base-result",
                        str(legacy),
                        "--dry-run",
                    ]
                )

            plan = json.loads(output.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(
                [trial["status"] for trial in plan["trials"]],
                ["run", "run", "skip"],
            )
            self.assertTrue(all("--validation-only" in t["command"] for t in plan["trials"]))
            self.assertFalse((root / "output" / "sl4-geometry-tuning").exists())


if __name__ == "__main__":
    unittest.main()
