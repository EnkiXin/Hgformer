"""Contracts for the strict Amazon-Toy SL8-LHGCN layer/batch grid."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from slrec_experiments.run_toy_sl8lhgcn_grid import (
    ANCHOR,
    BATCH_SIZES,
    DATASET,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SPLIT_INTERACTIONS,
    LAYERS,
    Trial,
    annotate_result,
    completed_result,
    expected_checkpoint_values,
    exclusive_gpu_lock,
    grid_trials,
    load_lhgcn_reference,
    main,
    manifold_summary,
    result_paths,
    trial_command,
    validate_protocol,
    write_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo=REPO_ROOT,
        data_root=REPO_ROOT / "dataset",
        output_root=root,
        python="python",
        gpu_id="7",
    )


def _fingerprints(suffix: str = "0") -> dict[str, dict[str, object]]:
    return {
        split: {
            "interactions": count,
            "sha256": (suffix * 64)[:64],
        }
        for split, count in EXPECTED_SPLIT_INTERACTIONS.items()
    }


def _diagnostics(trial: Trial) -> dict[str, object]:
    layer_records = [
        {
            "layer": index,
            "total": 25_226,
            "orientation_repairs": 0,
            "singular_fallbacks": 2,
            "active_singular_fallbacks": 0,
            "inactive_singular_fallbacks": 2,
            "input_membership_violations": 25_226,
            "output_nonpositive_determinants": 0,
            "output_nonfinite_log_determinants": 0,
            "output_membership_violations": 0,
            "max_abs_output_log_determinant": 2e-6,
        }
        for index in range(1, trial.gcn_layers + 1)
    ]
    return {
        "mode": "ambient_retract",
        "layers": trial.gcn_layers,
        "projection_total": 25_226 * trial.gcn_layers,
        "orientation_repairs": 0,
        "singular_fallbacks": 2 * trial.gcn_layers,
        "active_singular_fallbacks": 0,
        "inactive_singular_fallbacks": 2 * trial.gcn_layers,
        "initial_group_membership": {
            "total": 25_226,
            "membership_violations": 0,
        },
        "layer_membership": layer_records,
        "nonpositive_output_determinants": 0,
        "nonfinite_output_log_determinants": 0,
        "output_membership_violations": 0,
        "membership_tolerance": 1e-4,
        "max_abs_output_log_determinant": 2e-6,
        "distance_membership": {
            "samples": 16,
            "relative_membership_violations": 0,
            "nonfinite_approximate_logs": 0,
            "max_normalized_approximate_log_trace": 2e-5,
            "log_trace_tolerance": 1e-3,
            "max_approximate_log_reconstruction_residual": 1e-3,
        },
    }


def _write_raw_result(root: Path, trial: Trial, *, recall: float = 0.0653) -> Path:
    checkpoint = root / f"{trial.name}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": expected_checkpoint_values(trial),
            "epoch": 499,
            "state_dict": {},
        },
        checkpoint,
    )
    result = root / f"{trial.name}.raw.json"
    payload = {
        "model": "SL8LHGCN",
        "dataset": DATASET,
        "config_files": [
            str(REPO_ROOT / "baseline_config_fixed" / "RecFormer_toy.yaml"),
            str(REPO_ROOT / "baseline_config_fixed" / "SL8LHGCN_reproduction.yaml"),
        ],
        "seed": 2024,
        "epochs": 500,
        "stopping_step": 1000,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "best_valid_score": recall,
        "best_valid_result": {"recall@10": recall, "ndcg@10": 0.0355},
        "test_result": None,
        "checkpoint_file": str(checkpoint),
        "split_fingerprints": _fingerprints(),
        "model_diagnostics": _diagnostics(trial),
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


class GridAndCommandTest(unittest.TestCase):
    def test_grid_is_exact_cartesian_product(self):
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 1_614_465)
        trials = grid_trials()
        self.assertEqual(len(trials), 15)
        self.assertEqual(
            {(trial.gcn_layers, trial.train_batch_size) for trial in trials},
            {(layer, batch) for layer in LAYERS for batch in BATCH_SIZES},
        )
        self.assertEqual(ANCHOR, (4, 65_536))

    def test_protocol_is_current_toy_full_ranking_validation_only(self):
        protocol = validate_protocol(REPO_ROOT)
        self.assertEqual(protocol["dataset"], "Amazon_toy")
        self.assertEqual(protocol["seed"], 2024)
        self.assertEqual(protocol["split"]["args"]["mode"], "full")
        self.assertTrue(protocol["evaluation"]["validation_only"])
        self.assertFalse(protocol["evaluation"]["held_out_test_evaluated"])
        self.assertEqual(
            protocol["fixed_model_training"]["pairwise_loss"],
            "lhgcn_hinge_squared_sum",
        )
        self.assertEqual(
            protocol["fixed_model_training"]["sl_gcn_mode"], "ambient_retract"
        )

    def test_command_pins_single_child_gpu_and_all_non_grid_parameters(self):
        trial = Trial(8, 131_072)
        command = trial_command(
            _args(Path("/tmp/grid")),
            trial,
            Path("/tmp/raw.json"),
            Path("/tmp/checkpoint"),
        )
        joined = " ".join(command)
        self.assertIn("--validation-only", command)
        self.assertNotIn("test", " ".join(token.lower() for token in command))
        self.assertIn("--gpu_id=7", command)
        self.assertIn("--gcn_layers=8", command)
        self.assertIn("--n_layers=8", command)
        self.assertIn("--train_batch_size=131072", command)
        self.assertIn("--epochs=500", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn("--sl_gcn_mode=ambient_retract", command)
        self.assertIn("--pairwise_loss=lhgcn_hinge_squared_sum", command)
        self.assertIn("--loss_margin=0.1", command)
        self.assertIn("--learnable_score_scale=false", command)
        self.assertIn("RecFormer_toy.yaml", joined)
        self.assertIn("SL8LHGCN_reproduction.yaml", joined)


class ResumeAndDiagnosticsTest(unittest.TestCase):
    def test_legacy_result_is_adopted_only_after_checkpoint_contract_check(self):
        protocol = validate_protocol(REPO_ROOT)
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = _write_raw_result(root, trial)
            final = root / "final.json"
            annotate_result(
                raw,
                final,
                repo=REPO_ROOT,
                trial=trial,
                protocol=protocol,
                runtime={
                    "started_at": None,
                    "finished_at": "2026-08-29T00:00:00Z",
                    "duration_seconds": None,
                    "source": "verified-existing-result",
                },
                reused_from=raw,
            )
            complete, reason = completed_result(
                final, repo=REPO_ROOT, trial=trial, protocol=protocol
            )
            self.assertIsNotNone(complete)
            self.assertIsNone(reason)

            checkpoint = Path(complete["checkpoint_file"])
            bad_config = expected_checkpoint_values(trial)
            bad_config["train_batch_size"] = 32_768
            torch.save(
                {"config": bad_config, "epoch": 499, "state_dict": {}}, checkpoint
            )
            no_longer_complete, reason = completed_result(
                final, repo=REPO_ROOT, trial=trial, protocol=protocol
            )
            self.assertIsNone(no_longer_complete)
            self.assertIn("checkpoint trial contract mismatch", reason)

    def test_each_layer_and_distance_path_must_pass_membership(self):
        trial = Trial(4, 65_536)
        compact = manifold_summary(_diagnostics(trial), trial)
        self.assertTrue(compact["passed"])
        self.assertEqual(len(compact["per_layer"]), 4)
        self.assertEqual(compact["active_singular_fallbacks"], 0)

        invalid = _diagnostics(trial)
        invalid["layer_membership"][2]["output_membership_violations"] = 1
        with self.assertRaisesRegex(ValueError, "layer 3 left SL"):
            manifold_summary(invalid, trial)

    def test_test_touched_artifact_is_never_silently_resumed(self):
        protocol = validate_protocol(REPO_ROOT)
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = _write_raw_result(root, trial)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            payload["test_result"] = {"recall@10": 0.1}
            raw.write_text(json.dumps(payload), encoding="utf-8")
            # Give it otherwise matching metadata to exercise the hard guard.
            from slrec_experiments.run_toy_sl8lhgcn_grid import trial_metadata

            payload["toy_sl8_grid"] = trial_metadata(trial, protocol)
            raw.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                completed_result(
                    raw, repo=REPO_ROOT, trial=trial, protocol=protocol
                )


class SummaryLockAndDryRunTest(unittest.TestCase):
    def test_summary_contains_deltas_runtime_and_manifold_diagnostics(self):
        protocol = validate_protocol(REPO_ROOT)
        reference = load_lhgcn_reference(None)
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = result_paths(root, trial)
            raw = _write_raw_result(root, trial, recall=0.0653)
            annotate_result(
                raw,
                paths["result"],
                repo=REPO_ROOT,
                trial=trial,
                protocol=protocol,
                runtime={
                    "started_at": "2026-08-29T00:00:00Z",
                    "finished_at": "2026-08-29T00:02:00Z",
                    "duration_seconds": 120.0,
                    "source": "runner-measured",
                },
            )
            summary = write_summary(
                root / "summary.json",
                repo=REPO_ROOT,
                output_root=root,
                protocol=protocol,
                reference=reference,
            )
        self.assertEqual(summary["state"], "incomplete")
        self.assertEqual(summary["grid"]["completed_trials"], 1)
        candidate = summary["ranking"][0]
        self.assertAlmostEqual(candidate["delta_vs_lhgcn"]["recall@10"], -0.0006)
        self.assertEqual(candidate["runtime"]["duration_seconds"], 120.0)
        self.assertTrue(candidate["manifold"]["passed"])
        self.assertEqual(len(candidate["manifold"]["per_layer"]), 4)

    def test_gpu_lock_rejects_a_second_owner(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = Path(temporary_directory) / "gpu7.lock"
            with exclusive_gpu_lock(lock, "7"):
                with self.assertRaisesRegex(RuntimeError, "already reserved"):
                    with exclusive_gpu_lock(lock, "7"):
                        pass

    def test_dry_run_for_gpu7_plans_15_jobs_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "not-created"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--gpu-id",
                        "7",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(output_root.exists())
        self.assertEqual(plan["single_physical_gpu"], "7")
        self.assertEqual(plan["child_cuda_visible_devices"], "7")
        self.assertEqual(plan["child_config_gpu_id"], "7")
        self.assertEqual(plan["child_torch_device_after_mask"], "cuda:0")
        self.assertTrue(plan["strict_serial"])
        self.assertEqual(len(plan["jobs"]), 15)
        self.assertTrue(all(job["status"] == "run" for job in plan["jobs"]))
        self.assertFalse(plan["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
