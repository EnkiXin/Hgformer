"""Contracts for the strict AGCF MovieLens blocked-search runner."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from slrec_experiments.run_agcf_movielens_sweep import (
    ANCHOR,
    CONFIG_NAMES,
    DATASET,
    EPOCHS,
    EVAL_STEP,
    MODEL,
    MOVIELENS_FILTERED,
    SEED,
    STAGE_DEFINITIONS,
    STAGE_ORDER,
    STOPPING_STEP,
    Parameters,
    Trial,
    _validate_args,
    annotate_or_load_result,
    audit_movielens_source,
    build_stage_trials,
    campaign_contract,
    candidate_from_plain_result,
    candidate_from_result,
    dry_run_plan,
    load_continuation_candidate,
    resolve_stage_anchor,
    stage_summary_path,
    trial_command,
    validate_model_registration,
    validate_protocol,
    write_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(root: Path, **updates) -> argparse.Namespace:
    values = {
        "repo": REPO_ROOT,
        "output_root": root,
        "data_path": REPO_ROOT / "dataset",
        "python": "python",
        "gpu_id": "0",
        "stage": "anchor",
        "resume_from": None,
        "continuation_result": None,
        "max_new_trials": None,
        "plan_only": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _contract(data_root: Path | None = None) -> dict:
    root = data_root or (REPO_ROOT / "dataset")
    return campaign_contract(validate_protocol(REPO_ROOT), root)


class AGCFMovieLensSweepProtocolTest(unittest.TestCase):
    def test_source_protocol_model_and_anchor_are_hard_pinned(self):
        source = audit_movielens_source(REPO_ROOT / "dataset")
        self.assertTrue(source["verified"])
        self.assertEqual(source["lines_including_header"], 1_000_210)
        self.assertEqual(
            source["sha256"],
            "e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389",
        )

        protocol = validate_protocol(REPO_ROOT, source_audit=source)
        self.assertEqual(protocol["dataset"], DATASET)
        self.assertEqual(protocol["seed"], SEED)
        self.assertEqual(protocol["expected_filtered_dataset"], MOVIELENS_FILTERED)
        self.assertEqual(protocol["split"]["mode"], "full")
        self.assertEqual(protocol["split"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertTrue(protocol["evaluation"]["validation_only"])
        self.assertFalse(protocol["evaluation"]["held_out_test_evaluated"])
        self.assertEqual(
            [Path(item["path"]).name for item in protocol["config_files"]],
            list(CONFIG_NAMES),
        )
        self.assertEqual(
            ANCHOR,
            Parameters(
                metric_rank=16,
                channel_rank=64,
                train_batch_size=4096,
                learning_rate=0.001,
                margin=0.1,
                output_steps=1,
                integration_steps=1,
                potential_strength=0.1,
                damping=0.01,
                weight_decay=0.0,
            ),
        )
        validate_model_registration(REPO_ROOT)

    def test_every_requested_parameter_is_covered_by_blocked_stages(self):
        expected = {
            "metric-rank": ("metric_rank", (4, 8, 16, 32, 64)),
            "channel-rank": ("channel_rank", (16, 32, 64)),
            "batch-size": (
                "train_batch_size",
                (512, 1024, 2048, 4096, 8192),
            ),
            "learning-rate": (
                "learning_rate",
                (1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3),
            ),
            "margin": ("margin", (0.02, 0.05, 0.1, 0.2, 0.3)),
            "output-steps": ("output_steps", (1, 2, 3, 4, 6)),
            "integration-steps": ("integration_steps", (1, 2, 4, 8)),
            "potential-strength": (
                "potential_strength",
                (0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
            ),
            "damping": (
                "damping",
                (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
            ),
            "weight-decay": (
                "weight_decay",
                (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
            ),
        }
        self.assertEqual(STAGE_ORDER[0], "anchor")
        for stage, (field, values) in expected.items():
            self.assertEqual(STAGE_DEFINITIONS[stage]["field"], field)
            self.assertEqual(STAGE_DEFINITIONS[stage]["values"], values)
            trials = build_stage_trials(stage, ANCHOR)
            # The unchanged inherited anchor is deliberately not retrained.
            self.assertEqual(len(trials), len(values) - 1)
            for trial in trials:
                changed = {
                    key
                    for key, value in trial.parameters.recbole_values().items()
                    if value != ANCHOR.recbole_values()[key]
                }
                self.assertEqual(changed, {field})

        total_new = len(build_stage_trials("anchor", ANCHOR)) + sum(
            len(build_stage_trials(stage, ANCHOR)) for stage in STAGE_ORDER[1:]
        )
        self.assertEqual(total_new, 42)


class AGCFMovieLensSweepCommandTest(unittest.TestCase):
    def test_command_is_one_gpu_validation_only_and_exact_paper_training_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _args(root, gpu_id="7")
            trial = build_stage_trials("anchor", ANCHOR)[0]
            command = trial_command(
                args, trial, root / "result.json", root / "checkpoint"
            )

        self.assertEqual(command[command.index("--model") + 1], MODEL)
        self.assertEqual(command[command.index("--dataset") + 1], DATASET)
        self.assertIn("--validation-only", command)
        self.assertNotIn("--no-save", command)
        self.assertNotIn("--test", command)
        self.assertIn(f"--epochs={EPOCHS}", command)
        self.assertIn(f"--eval_step={EVAL_STEP}", command)
        self.assertIn(f"--stopping_step={STOPPING_STEP}", command)
        self.assertIn("--dynamics_chunk_size=4096", command)
        self.assertIn("--checkpoint_dynamics=False", command)
        self.assertIn("--eval_item_chunk_size=4096", command)
        self.assertIn("--seed=2020", command)
        self.assertIn("--gpu_id=7", command)
        self.assertNotIn("--gpu_id=0", command)
        self.assertIn("--metric_rank=16", command)
        self.assertIn("--channel_rank=64", command)
        self.assertIn("--train_batch_size=4096", command)
        self.assertIn("--learning_rate=0.001", command)
        self.assertIn("--margin=0.1", command)
        configs = command[command.index("--config-files") + 1].split()
        self.assertEqual([Path(item).name for item in configs], list(CONFIG_NAMES))

    def test_plan_is_serial_and_contains_no_test_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary), stage="all", gpu_id="3")
            plan = dry_run_plan(args, _contract())

        self.assertEqual(plan["physical_gpu"], "3")
        self.assertEqual(plan["serial_workers"], 1)
        self.assertTrue(plan["validation_only"])
        self.assertFalse(plan["test_evaluated"])
        self.assertEqual([item["stage"] for item in plan["stages"]], list(STAGE_ORDER))
        commands = [job["command"] for stage in plan["stages"] for job in stage["jobs"]]
        self.assertTrue(commands)
        self.assertTrue(all("--validation-only" in command for command in commands))
        self.assertTrue(all("--gpu_id=3" in command for command in commands))
        self.assertTrue(all("--eval_step=1" in command for command in commands))

    def test_multi_gpu_and_negative_budget_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_args(_args(Path("/tmp/agcf-ml"), gpu_id="0,1"))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_args(_args(Path("/tmp/agcf-ml"), max_new_trials=-1))
        with self.assertRaisesRegex(ValueError, "requires"):
            _validate_args(
                _args(
                    Path("/tmp/agcf-ml"),
                    stage="metric-rank",
                    continuation_result=Path("anchor.json"),
                )
            )


class AGCFMovieLensSweepResumeTest(unittest.TestCase):
    def _write_raw_result(
        self,
        root: Path,
        trial: Trial,
        contract: dict,
        *,
        test_result=None,
    ) -> Path:
        checkpoint = root / f"{trial.name}.pth"
        config = {
            "model": MODEL,
            "dataset": DATASET,
            "seed": SEED,
            "embedding_size": 64,
            "epochs": EPOCHS,
            "eval_step": EVAL_STEP,
            "stopping_step": STOPPING_STEP,
            "valid_metric": "Recall@10",
            "data_path": str(Path(contract["training"]["data_root"]) / DATASET),
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
            "metric_hidden_size": 64,
            "pnet_hidden_size": 64,
            "evolution_time": 1.0,
            "dynamics_chunk_size": 4096,
            "checkpoint_dynamics": False,
            "eval_item_chunk_size": 4096,
            **trial.parameters.recbole_values(),
        }
        torch.save({"config": config, "state_dict": {}}, checkpoint)
        result = root / f"{trial.name}.json"
        result.write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "dataset": DATASET,
                    "seed": SEED,
                    "best_valid_score": 0.12,
                    "best_valid_result": {
                        "recall@10": 0.12,
                        "ndcg@10": 0.15,
                    },
                    "test_result": test_result,
                    "checkpoint_file": str(checkpoint),
                    "split_fingerprints": {
                        "train": {"interactions": 100, "sha256": "train"},
                        "valid": {"interactions": 10, "sha256": "valid"},
                        "test": {"interactions": 10, "sha256": "test"},
                    },
                }
            ),
            encoding="utf-8",
        )
        return result

    def test_raw_child_result_is_safely_recovered_and_exactly_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _contract(root / "data")
            trial = build_stage_trials("anchor", ANCHOR)[0]
            result = self._write_raw_result(root, trial, contract)
            payload = annotate_or_load_result(result, trial, contract)
            self.assertIn("agcf_movielens_runner", payload)
            candidate = candidate_from_result(result, trial, contract)
            summary = stage_summary_path(root, "anchor")
            write_summary(
                summary,
                stage="anchor",
                contract=contract,
                candidates=[candidate],
                planned_new_trial_count=1,
                completed_new_trial_count=1,
                inherited_anchor=None,
                complete=True,
            )
            loaded = load_continuation_candidate(
                summary, contract, required_stage="anchor"
            )
            self.assertEqual(loaded.name, trial.name)

            args = _args(root, stage="batch-size", plan_only=False)
            anchor, selected_path = resolve_stage_anchor(
                args, contract, plan_only=False
            )
            self.assertEqual(anchor.name, trial.name)
            self.assertEqual(Path(selected_path), summary)

    def test_resume_rejects_test_metrics_and_parameter_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _contract(root / "data")
            trial = build_stage_trials("anchor", ANCHOR)[0]
            result = self._write_raw_result(
                root, trial, contract, test_result={"recall@10": 1.0}
            )
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                annotate_or_load_result(result, trial, contract)

            result = self._write_raw_result(root, trial, contract)
            annotate_or_load_result(result, trial, contract)
            payload = json.loads(result.read_text(encoding="utf-8"))
            payload["agcf_movielens_runner"]["parameters"]["margin"] = 0.3
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume contract mismatch"):
                annotate_or_load_result(result, trial, contract)

    def test_plain_anchor_import_is_read_only_and_skips_anchor_in_all_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _contract(root / "data")
            trial = build_stage_trials("anchor", ANCHOR)[0]
            result = self._write_raw_result(root, trial, contract)
            before = result.read_bytes()
            imported = candidate_from_plain_result(result, ANCHOR, contract)
            self.assertEqual(result.read_bytes(), before)
            self.assertEqual(imported.parameters, ANCHOR)

            args = _args(
                root,
                stage="all",
                continuation_result=result,
                gpu_id="2",
            )
            _validate_args(args)
            plan = dry_run_plan(
                args,
                contract,
                initial_anchor=imported,
                continuation_note="external anchor",
            )
            self.assertNotIn("anchor", [stage["stage"] for stage in plan["stages"]])
            self.assertEqual(
                plan["imported_initial_anchor"]["source"], str(result.resolve())
            )
            self.assertEqual(result.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
