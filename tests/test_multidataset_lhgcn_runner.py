"""Contracts for the validation-only, single-GPU LHGCN runner."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from slrec_experiments.run_multidataset_lhgcn import (
    BASELINE_PARAMETERS,
    CORE_STAGES,
    DATASETS,
    SEARCH_VALUES,
    Trial,
    _gpu_token,
    _result_metadata,
    annotate_result,
    build_stage_trials,
    candidate_from_result,
    completed_result,
    execute_stage_for_dataset,
    exclusive_gpu_lock,
    main,
    runtime_protocol,
    search_manifest,
    trial_command,
    validate_lhgcn_protocol,
    write_stage_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**updates):
    defaults = {
        "repo": REPO_ROOT,
        "data_root": REPO_ROOT / "dataset",
        "output_root": Path("/tmp/lhgcn-tests"),
        "python": "python",
        "gpu_id": "7",
        "model_entrypoint": "HGCF",
        "epochs": 500,
        "eval_step": 50,
        "profile": "core",
    }
    defaults.update(updates)
    return argparse.Namespace(**defaults)


def _protocol(spec=DATASETS[0]):
    return runtime_protocol(
        validate_lhgcn_protocol(REPO_ROOT, spec),
        epochs=500,
        eval_step=50,
        model_entrypoint="HGCF",
    )


class RegistryAndProtocolTest(unittest.TestCase):
    def test_reuses_the_seven_dataset_registry(self):
        self.assertEqual(
            [spec.dataset for spec in DATASETS],
            [
                "Amazon_cd",
                "Amazon_movies",
                "Amazon_toy",
                "Amazon_book",
                "DoubanBook",
                "DoubanMovie",
                "DoubanMusic",
            ],
        )

    def test_every_dataset_keeps_hgformer_full_ranking_protocol(self):
        for spec in DATASETS:
            protocol = validate_lhgcn_protocol(REPO_ROOT, spec)
            self.assertEqual(protocol["dataset"], spec.dataset)
            self.assertEqual(protocol["seed"], 2024)
            self.assertEqual(protocol["filters"]["rating"], "[3,inf)")
            self.assertEqual(protocol["validation"]["eval_args"]["mode"], "full")
            self.assertEqual(
                protocol["validation"]["eval_args"]["split"],
                {"RS": [0.8, 0.1, 0.1]},
            )
            self.assertEqual(protocol["validation"]["metrics"], ["Recall", "NDCG"])


class SearchDesignTest(unittest.TestCase):
    def test_baseline_is_archived_hgcf_lgcn_setting(self):
        trial = build_stage_trials("baseline")[0]
        self.assertEqual(trial.parameters, BASELINE_PARAMETERS)
        self.assertEqual(trial.parameters.gcn_layers, 4)
        self.assertEqual(trial.parameters.curve, 0.5)
        self.assertEqual(trial.parameters.weight_decay, 0.0)

    def test_layers_one_through_eight_are_exhaustive_without_retraining_anchor(self):
        trials = build_stage_trials("gcn_layers", BASELINE_PARAMETERS)
        self.assertEqual({trial.parameters.gcn_layers for trial in trials}, set(range(1, 9)) - {4})
        self.assertTrue(all(trial.parameters.curve == 0.5 for trial in trials))

    def test_each_later_core_stage_changes_only_one_parameter(self):
        for stage in CORE_STAGES[2:]:
            trials = build_stage_trials(stage, BASELINE_PARAMETERS)
            self.assertEqual(
                {getattr(trial.parameters, stage) for trial in trials},
                set(SEARCH_VALUES[stage]) - {getattr(BASELINE_PARAMETERS, stage)},
            )
            for trial in trials:
                before = BASELINE_PARAMETERS.__dict__
                after = trial.parameters.__dict__
                changed = {key for key in before if before[key] != after[key]}
                self.assertEqual(changed, {stage})

    def test_profiles_have_finite_non_cartesian_job_counts(self):
        self.assertEqual(search_manifest(_args(profile="baseline"))["new_trials_per_dataset"], 1)
        self.assertEqual(search_manifest(_args(profile="core"))["new_trials_per_dataset"], 20)
        self.assertEqual(search_manifest(_args(profile="extended"))["new_trials_per_dataset"], 35)


class CommandAndGpuSafetyTest(unittest.TestCase):
    def test_physical_gpu_index_is_single_and_canonical(self):
        self.assertEqual(_gpu_token("00"), "0")
        for invalid in ("-1", "0,1", "GPU-deadbeef", "cuda:0", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _gpu_token(invalid)

    def test_command_is_validation_only_and_child_sees_logical_gpu_zero(self):
        spec = DATASETS[0]
        trial = Trial("baseline", BASELINE_PARAMETERS)
        command = trial_command(
            _args(), spec, trial, Path("/tmp/result.json"), Path("/tmp/checkpoint")
        )
        self.assertEqual(command[command.index("--model") + 1], "HGCF")
        self.assertIn("--validation-only", command)
        self.assertIn("--gpu_id=0", command)
        self.assertNotIn("--gpu_id=7", command)
        self.assertIn("--conv=lGCN", command)
        self.assertIn("--epochs=500", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn("--stopping_step=11", command)
        self.assertFalse(any("uni100" in token.lower() for token in command))
        names = [
            Path(path).name
            for path in command[command.index("--config-files") + 1].split()
        ]
        self.assertEqual(names, ["RecFormer_cd.yaml", "LHGCN_reproduction.yaml"])

    def test_lock_rejects_a_second_runner_for_the_same_card(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gpu.lock"
            with exclusive_gpu_lock(path, "3"):
                with self.assertRaisesRegex(RuntimeError, "already reserved"):
                    with exclusive_gpu_lock(path, "3"):
                        pass

    def test_multiple_visible_devices_are_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "never"
            with self.assertRaisesRegex(ValueError, "one non-negative"):
                main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output),
                        "--gpu-id",
                        "0,1",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            self.assertFalse(output.exists())


class ValidationResultTest(unittest.TestCase):
    def _raw_result(self, checkpoint: Path, spec=DATASETS[0]):
        return {
            "model": "HGCF",
            "dataset": spec.dataset,
            "seed": 2024,
            "best_valid_score": 0.123,
            "best_valid_result": {"recall@10": 0.123, "ndcg@10": 0.08},
            "test_result": None,
            "checkpoint_file": str(checkpoint),
            "split_fingerprints": {
                "train": {"interactions": 8, "sha256": "train"},
                "valid": {"interactions": 1, "sha256": "valid"},
                "test": {"interactions": 1, "sha256": "test"},
            },
        }

    def test_annotated_result_resumes_only_for_the_same_protocol(self):
        spec = DATASETS[0]
        trial = Trial("baseline", BASELINE_PARAMETERS)
        protocol = _protocol(spec)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pth"
            checkpoint.touch()
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(self._raw_result(checkpoint, spec)), encoding="utf-8"
            )
            annotate_result(
                result_path,
                spec=spec,
                trial=trial,
                protocol=protocol,
                model_entrypoint="HGCF",
            )
            resumed = completed_result(
                result_path,
                spec=spec,
                trial=trial,
                protocol=protocol,
                model_entrypoint="HGCF",
            )
            changed = completed_result(
                result_path,
                spec=spec,
                trial=trial,
                protocol=runtime_protocol(
                    validate_lhgcn_protocol(REPO_ROOT, spec),
                    epochs=100,
                    eval_step=50,
                    model_entrypoint="HGCF",
                ),
                model_entrypoint="HGCF",
            )
        self.assertIsNotNone(resumed)
        self.assertIsNone(changed)

    def test_test_touched_result_is_never_silently_retrained(self):
        spec = DATASETS[0]
        trial = Trial("baseline", BASELINE_PARAMETERS)
        protocol = _protocol(spec)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pth"
            checkpoint.touch()
            result_path = root / "result.json"
            payload = self._raw_result(checkpoint, spec)
            payload["test_result"] = {"recall@10": 0.2}
            payload["lhgcn_tuning"] = _result_metadata(spec, trial, protocol)
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                completed_result(
                    result_path,
                    spec=spec,
                    trial=trial,
                    protocol=protocol,
                    model_entrypoint="HGCF",
                )

    def test_stage_summary_carries_anchor_and_selects_validation_winner(self):
        spec = DATASETS[0]
        protocol = _protocol(spec)
        anchor = {
            "trial_name": "baseline",
            "stage": "baseline",
            "parameters": BASELINE_PARAMETERS.__dict__,
            "best_valid_score": 0.1,
            "split_fingerprints": {"train": "a", "valid": "b", "test": "c"},
        }
        candidate = {
            **anchor,
            "trial_name": "layer-5",
            "stage": "gcn_layers",
            "best_valid_score": 0.2,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = write_stage_summary(
                Path(temporary_directory) / "summary.json",
                spec=spec,
                stage="gcn_layers",
                protocol=protocol,
                anchor=anchor,
                new_candidates=[candidate],
                expected_trial_names=["layer-5"],
            )
        self.assertEqual(summary["state"], "complete")
        self.assertEqual(summary["winner"]["trial_name"], "layer-5")
        self.assertEqual(summary["anchor_carried"]["trial_name"], "baseline")


class BudgetAndResumeTest(unittest.TestCase):
    def test_one_new_job_is_counted_once_and_then_resumed_without_training(self):
        spec = DATASETS[0]
        protocol = _protocol(spec)
        calls = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = _args(output_root=root)

            def fake_run(command, log_path, cwd, env, *, lock_fd):
                del log_path, cwd, env, lock_fd
                calls.append(command)
                result_path = Path(command[command.index("--result-file") + 1])
                checkpoint_dir = Path(
                    next(
                        token.split("=", 1)[1]
                        for token in command
                        if token.startswith("--checkpoint_dir=")
                    )
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = checkpoint_dir / "fake.pth"
                checkpoint.touch()
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "model": "HGCF",
                            "dataset": spec.dataset,
                            "seed": 2024,
                            "best_valid_score": 0.1,
                            "best_valid_result": {"recall@10": 0.1},
                            "test_result": None,
                            "checkpoint_file": str(checkpoint),
                            "split_fingerprints": {
                                "train": {"sha256": "train"},
                                "valid": {"sha256": "valid"},
                                "test": {"sha256": "test"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            lock_path = root / "test-lock"
            with lock_path.open("w", encoding="utf-8") as lock:
                with patch(
                    "slrec_experiments.run_multidataset_lhgcn._run_and_tee",
                    fake_run,
                ), redirect_stdout(io.StringIO()):
                    first_budget = [1]
                    first_complete = execute_stage_for_dataset(
                        args,
                        spec,
                        "baseline",
                        None,
                        protocol,
                        first_budget,
                        {"CUDA_VISIBLE_DEVICES": "7"},
                        lock.fileno(),
                    )
                    resumed_budget = [1]
                    resumed_complete = execute_stage_for_dataset(
                        args,
                        spec,
                        "baseline",
                        None,
                        protocol,
                        resumed_budget,
                        {"CUDA_VISIBLE_DEVICES": "7"},
                        lock.fileno(),
                    )

        self.assertTrue(first_complete)
        self.assertEqual(first_budget, [0])
        self.assertTrue(resumed_complete)
        self.assertEqual(resumed_budget, [1])
        self.assertEqual(len(calls), 1)


class DryRunTest(unittest.TestCase):
    def test_dry_run_writes_nothing_and_plans_baseline_before_dependent_stages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--datasets",
                        "amazon-cd",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(stream.getvalue())

        self.assertEqual(code, 0)
        self.assertFalse(output_root.exists())
        self.assertFalse(plan["test_evaluated"])
        self.assertEqual(plan["search"]["stage_order"], list(CORE_STAGES))
        stages = plan["datasets"][0]["stages"]
        self.assertEqual(stages[0]["status"], "ready")
        self.assertEqual(len(stages[0]["jobs"]), 1)
        self.assertTrue(stages[1]["status"].startswith("blocked-awaiting"))

    def test_data_audit_cannot_be_skipped_during_real_execution(self):
        with self.assertRaisesRegex(ValueError, "only with --dry-run"):
            main(
                [
                    "--repo",
                    str(REPO_ROOT),
                    "--output-root",
                    "/tmp/never-created-lhgcn",
                    "--skip-data-audit",
                ]
            )


if __name__ == "__main__":
    unittest.main()
