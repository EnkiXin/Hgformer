"""Contracts for adaptive Amazon-Toy SL8-LHGCN tuning."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import torch

from slrec_experiments import run_toy_sl8lhgcn_grid as layer_grid
from slrec_experiments.run_toy_sl8lhgcn_staged import (
    EPOCH_EXTENSION_BUDGETS,
    EXPECTED_NEW_TRIALS,
    PARAMETER_STATUS,
    PHYSICAL_GPU,
    STAGES,
    Artifact,
    CandidateState,
    EpochExtensionState,
    Parameters,
    Prerequisite,
    PrerequisiteNotReady,
    StagedTrial,
    TuningState,
    apply_candidate,
    epoch_extension_trials,
    expected_checkpoint_values,
    load_complete_prerequisite,
    main,
    select_winner,
    stage_trials,
    summary_payload,
    trial_command,
    validate_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fingerprints() -> dict[str, dict[str, object]]:
    return {
        split: {"interactions": count, "sha256": token * 64}
        for (split, count), token in zip(
            layer_grid.EXPECTED_SPLIT_INTERACTIONS.items(), ("a", "b", "c")
        )
    }


def _diagnostics(trial: layer_grid.Trial) -> dict[str, object]:
    per_layer = [
        {
            "layer": index,
            "total": 25_226,
            "orientation_repairs": 0,
            "singular_fallbacks": 10,
            "active_singular_fallbacks": 0,
            "inactive_singular_fallbacks": 10,
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
        "singular_fallbacks": 10 * trial.gcn_layers,
        "active_singular_fallbacks": 0,
        "inactive_singular_fallbacks": 10 * trial.gcn_layers,
        "initial_group_membership": {
            "total": 25_226,
            "membership_violations": 0,
        },
        "layer_membership": per_layer,
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


def _write_complete_prerequisite(root: Path) -> Path:
    protocol = layer_grid.validate_protocol(REPO_ROOT)
    trial = layer_grid.Trial(4, 65_536)
    parameters = Parameters(4, 65_536)
    checkpoint = root / "winner.pth"
    torch.save(
        {
            "config": expected_checkpoint_values(parameters),
            "epoch": 499,
            "state_dict": {},
        },
        checkpoint,
    )
    raw = root / "winner.raw.json"
    raw.write_text(
        json.dumps(
            {
                "model": "SL8LHGCN",
                "dataset": layer_grid.DATASET,
                "config_files": [
                    str(
                        REPO_ROOT
                        / "baseline_config_fixed"
                        / layer_grid.BASE_CONFIG_NAME
                    ),
                    str(
                        REPO_ROOT
                        / "baseline_config_fixed"
                        / layer_grid.MODEL_OVERLAY_NAME
                    ),
                ],
                "seed": layer_grid.SEED,
                "epochs": layer_grid.EPOCHS,
                "stopping_step": layer_grid.STOPPING_STEP,
                "parameter_count": layer_grid.EXPECTED_PARAMETER_COUNT,
                "best_valid_score": 0.07,
                "best_valid_result": {"recall@10": 0.07, "ndcg@10": 0.04},
                "test_result": None,
                "checkpoint_file": str(checkpoint),
                "split_fingerprints": _fingerprints(),
                "model_diagnostics": _diagnostics(trial),
            }
        ),
        encoding="utf-8",
    )
    result = root / "winner.json"
    layer_grid.annotate_result(
        raw,
        result,
        repo=REPO_ROOT,
        trial=trial,
        protocol=protocol,
        runtime={
            "started_at": "2026-08-29T00:00:00Z",
            "finished_at": "2026-08-29T00:01:00Z",
            "duration_seconds": 60.0,
            "source": "test-fixture",
        },
    )
    payload = layer_grid._load_mapping(result)
    winner = layer_grid._trial_candidate(result, trial, payload)
    ranking = [winner] + [
        {
            "trial": f"dummy-{index}",
            "gcn_layers": 2,
            "train_batch_size": 32_768,
            "recall@10": 0.06 - index / 10_000,
            "ndcg@10": 0.03,
            "test_evaluated": False,
        }
        for index in range(1, 15)
    ]
    summary = root / "layer-batch-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "amazon-toy-sl8lhgcn-layer-batch-grid-summary",
                "state": "complete",
                "dataset": layer_grid.DATASET,
                "protocol": protocol,
                "grid": {
                    "gcn_layers": list(layer_grid.LAYERS),
                    "train_batch_size": list(layer_grid.BATCH_SIZES),
                    "expected_trials": 15,
                    "completed_trials": 15,
                    "failed_trials": [],
                    "pending_trials": [],
                    "invalid_results": {},
                },
                "winner": winner,
                "ranking": ranking,
                "split_fingerprints": _fingerprints(),
                "manifold_acceptance": {"every_completed_trial_passed": True},
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo=REPO_ROOT,
        data_root=REPO_ROOT / "dataset",
        output_root=root,
        python="python",
        gpu_id=PHYSICAL_GPU,
    )


class StageDefinitionTest(unittest.TestCase):
    def test_domains_and_carry_forward_reduce_grid_to_21_new_trials(self):
        self.assertEqual(
            [stage.key for stage in STAGES],
            [
                "learning_rate",
                "loss_margin",
                "coord_clip",
                "schatten_p",
                "weight_decay",
                "self_loop",
            ],
        )
        self.assertEqual([len(stage.candidates) for stage in STAGES], [4, 4, 5, 5, 5, 4])
        self.assertEqual(sum(len(stage.candidates) - 1 for stage in STAGES), 21)
        self.assertEqual(EXPECTED_NEW_TRIALS, 21)
        self.assertEqual(
            [candidate.updates["schatten_p"] for candidate in STAGES[3].candidates],
            [1, 2, 4, 8, "inf"],
        )

    def test_each_stage_has_one_parent_and_next_stage_keeps_prior_winner(self):
        parent_parameters = Parameters(4, 65_536)
        parent = Artifact(
            result_path=Path("/tmp/parent.json"),
            result_sha256="a" * 64,
            parameters=parent_parameters,
            recall_at_10=0.06,
            ndcg_at_10=0.03,
            checkpoint_file="/tmp/parent.pth",
            split_fingerprints=_fingerprints(),
            artifact_signature="parent",
            source="fixture",
        )
        lr_trials = stage_trials(STAGES[0], parent)
        self.assertEqual(sum(t.parameters == parent_parameters for t in lr_trials), 1)
        selected_parameters = apply_candidate(
            parent_parameters, STAGES[0].candidates[1]
        )
        selected_parent = replace(parent, parameters=selected_parameters)
        margin_trials = stage_trials(STAGES[1], selected_parent)
        self.assertTrue(
            all(t.parameters.learning_rate == 3e-4 for t in margin_trials)
        )

    def test_exact_metric_tie_prefers_unchanged_parent(self):
        parameters = Parameters(4, 65_536)
        parent = Artifact(
            Path("/tmp/p.json"),
            "a" * 64,
            parameters,
            0.07,
            0.04,
            "/tmp/p.pth",
            _fingerprints(),
            "parent",
            "fixture",
        )
        trials = stage_trials(STAGES[0], parent)
        states = []
        for trial in trials:
            is_parent = trial.parameters == parent.parameters
            artifact = parent if is_parent else replace(
                parent,
                parameters=trial.parameters,
                artifact_signature=trial.candidate_label,
            )
            states.append(CandidateState(trial, "complete", artifact))
        winner = select_winner(states, parent)
        self.assertEqual(winner.artifact.artifact_signature, "parent")

    def test_epoch_extensions_are_separate_and_never_select_a_winner(self):
        parameters = Parameters(4, 65_536)
        parent = Artifact(
            Path("/tmp/p.json"),
            "a" * 64,
            parameters,
            0.07,
            0.04,
            "/tmp/p.pth",
            _fingerprints(),
            "parent",
            "fixture",
        )
        extensions = epoch_extension_trials(parent)
        self.assertEqual(
            [trial.parameters.epochs for trial in extensions],
            list(EPOCH_EXTENSION_BUDGETS),
        )
        self.assertTrue(all(trial.stage_key == "epoch_extension" for trial in extensions))
        self.assertTrue(all(trial.parent_artifact_signature == "parent" for trial in extensions))

    def test_extension_metrics_cannot_replace_frozen_500_winner_in_summary(self):
        parameters = Parameters(4, 65_536)
        parent = Artifact(
            Path("/tmp/p.json"),
            "a" * 64,
            parameters,
            0.07,
            0.04,
            "/tmp/p.pth",
            _fingerprints(),
            "parent",
            "fixture",
        )
        extension_trial = epoch_extension_trials(parent)[0]
        better_long_budget = replace(
            parent,
            parameters=extension_trial.parameters,
            recall_at_10=0.09,
            ndcg_at_10=0.06,
            artifact_signature="longer",
        )
        prerequisite = Prerequisite(
            Path("/tmp/summary.json"),
            "s" * 64,
            parent.result_path,
            parent.result_sha256,
            parent.split_fingerprints,
            parent.parameters,
            parent.recall_at_10,
            parent.ndcg_at_10,
            parent.checkpoint_file,
            parent.artifact_signature,
        )
        state = TuningState(
            prerequisite=prerequisite,
            stages=(),
            current_stage=None,
            final_artifact=parent,
            selection_complete=True,
            epoch_extension=EpochExtensionState(
                parent_500=parent,
                candidates=(
                    CandidateState(
                        extension_trial, "complete", better_long_budget
                    ),
                ),
                complete=True,
            ),
            complete=True,
        )
        payload = summary_payload(
            state,
            protocol=layer_grid.validate_protocol(REPO_ROOT),
            output_root=Path("/tmp/nonexistent-staged-output"),
            data_audit=None,
        )
        self.assertEqual(payload["winner"]["recall@10"], 0.07)
        self.assertIsNone(payload["epoch_extension"]["winner"])
        self.assertTrue(
            payload["epoch_extension"]["excluded_from_500_epoch_model_selection"]
        )


class PrerequisiteAndCommandTest(unittest.TestCase):
    def test_complete_prerequisite_is_deeply_loaded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = _write_complete_prerequisite(root)
            prerequisite = load_complete_prerequisite(summary, REPO_ROOT)
        self.assertEqual(prerequisite.parameters, Parameters(4, 65_536))
        self.assertEqual(prerequisite.recall_at_10, 0.07)
        self.assertEqual(prerequisite.split_fingerprints, _fingerprints())

    def test_incomplete_prerequisite_is_not_used_provisionally(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = _write_complete_prerequisite(root)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["state"] = "incomplete"
            payload["grid"]["completed_trials"] = 3
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PrerequisiteNotReady, "3/15"):
                load_complete_prerequisite(summary, REPO_ROOT)

    def test_command_is_gpu7_validation_only_and_encodes_geometry(self):
        parent = Artifact(
            Path("/tmp/p.json"),
            "a" * 64,
            Parameters(8, 131_072),
            0.07,
            0.04,
            "/tmp/p.pth",
            _fingerprints(),
            "parent",
            "fixture",
        )
        parameters = replace(
            parent.parameters,
            learning_rate=1e-3,
            loss_margin=0.3,
            coord_clip=1.5,
            schatten_p="inf",
            weight_decay=5e-3,
            lhgcn_include_self=True,
            lhgcn_self_loop_weight=0.5,
            epochs=1000,
        )
        trial = StagedTrial(7, "epoch_extension", 1, "epochs_1000", parameters, "parent")
        command = trial_command(
            _args(Path("/tmp/staged")),
            trial,
            Path("/tmp/raw.json"),
            Path("/tmp/checkpoint"),
        )
        self.assertIn("--validation-only", command)
        self.assertIn("--gpu_id=7", command)
        self.assertIn("--epochs=1000", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn("--schatten_p=inf", command)
        self.assertIn("--coord_clip=1.5", command)
        self.assertIn("--lhgcn_include_self=true", command)
        self.assertIn("--lhgcn_self_loop_weight=0.5", command)
        self.assertIn("--learning_rate=0.001", command)
        self.assertIn("--weight_decay=0.005", command)

    def test_staged_validator_hard_rejects_any_test_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = _write_complete_prerequisite(root)
            prerequisite = load_complete_prerequisite(summary, REPO_ROOT)
            payload = layer_grid._load_mapping(prerequisite.winner_result_path)
            payload["test_result"] = {"recall@10": 0.9}
            parent = Artifact(
                prerequisite.winner_result_path,
                prerequisite.winner_result_sha256,
                prerequisite.parameters,
                prerequisite.recall_at_10,
                prerequisite.ndcg_at_10,
                prerequisite.checkpoint_file,
                prerequisite.split_fingerprints,
                prerequisite.artifact_signature,
                "fixture",
            )
            trial = stage_trials(STAGES[0], parent)[0]
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                validate_result(
                    payload,
                    repo=REPO_ROOT,
                    trial=trial,
                    prerequisite=prerequisite,
                    protocol=layer_grid.validate_protocol(REPO_ROOT),
                    require_metadata=False,
                )


class PlanningAndParameterStatusTest(unittest.TestCase):
    def test_dry_run_plans_only_first_adaptive_stage_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = _write_complete_prerequisite(root)
            output_root = root / "not-created"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--layer-batch-summary",
                        str(summary),
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
        self.assertEqual(plan["selection_state"], "incomplete")
        self.assertEqual(plan["current_stage"]["key"], "learning_rate")
        self.assertEqual(len(plan["jobs_for_current_stage"]), 3)
        self.assertTrue(
            all("--validation-only" in job["command"] for job in plan["jobs_for_current_stage"])
        )
        self.assertTrue(plan["future_stages_are_adaptive"])
        self.assertIsNone(plan["epoch_extension"])
        self.assertEqual(plan["strategy"]["new_trials_expected"], 21)

    def test_non_gpu7_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = _write_complete_prerequisite(root)
            with self.assertRaisesRegex(ValueError, "only on physical GPU 7"):
                main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(root / "out"),
                        "--layer-batch-summary",
                        str(summary),
                        "--gpu-id",
                        "6",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )

    def test_dead_and_deferred_parameters_are_explicit(self):
        dead = PARAMETER_STATUS["inactive_or_dead_in_current_model"]
        self.assertIn("init_std", dead)
        self.assertIn("reg_weight", dead)
        self.assertIn("factor_aggregation", dead)
        self.assertIn("lhgcn_self_loop_weight_when_off", dead)
        deferred = PARAMETER_STATUS["active_but_deferred_or_not_for_model_selection"]
        self.assertIn("sl_scale", deferred)
        self.assertIn("log_terms", deferred)
        self.assertIn("sl_gcn_mode", deferred)
        self.assertIn("epochs", PARAMETER_STATUS["separate_budget_sensitivity"])


if __name__ == "__main__":
    unittest.main()
