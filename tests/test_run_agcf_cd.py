"""Contracts for the single-GPU, validation-only AGCF Amazon-CD runner."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from slrec_experiments.run_agcf_cd import (
    DATASET,
    FAMILIES,
    Parameters,
    Trial,
    _validate_args,
    annotate_result,
    audit_amazon_cd_source,
    build_stage_trials,
    campaign_contract,
    config_paths,
    dry_run_plan,
    load_complete_result,
    stage_order,
    trial_command,
    validate_model_registration,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(root: Path, **updates):
    values = {
        "repo": REPO_ROOT,
        "output_root": root,
        "data_path": REPO_ROOT / "dataset",
        "python": "python",
        "gpu_id": "0",
        "epochs": 500,
        "eval_step": 10,
        "stopping_step": 30,
        "stage": "pilot",
        "resume_from": None,
        "max_new_trials": None,
        "dry_run": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class AGCFRunnerProtocolTest(unittest.TestCase):
    def test_both_families_inherit_the_exact_recformer_cd_protocol(self):
        expected_configs = {
            "agcf": ["RecFormer_cd.yaml", "AGCF_cd.yaml"],
            "agcf-sl8coord": [
                "RecFormer_cd.yaml",
                "AGCF_cd.yaml",
                "AGCFSL8Coord_cd.yaml",
            ],
        }
        for slug, family in FAMILIES.items():
            audit = validate_protocol(REPO_ROOT, family)
            self.assertEqual(audit["dataset"], DATASET)
            self.assertEqual(audit["seed"], 2024)
            self.assertEqual(audit["split"]["mode"], "full")
            self.assertEqual(audit["split"]["split"], {"RS": [0.8, 0.1, 0.1]})
            self.assertEqual(audit["evaluation"]["selection_metric"], "Recall@10")
            self.assertTrue(audit["evaluation"]["validation_only"])
            self.assertFalse(audit["evaluation"]["held_out_test_evaluated"])
            self.assertEqual(
                [Path(item["path"]).name for item in audit["config_files"]],
                expected_configs[slug],
            )
            self.assertEqual(
                audit["expected_filtered_dataset"],
                {
                    "framework_users": 66_317,
                    "framework_items": 58_869,
                    "interactions": 952_547,
                    "token_users": 66_316,
                    "token_items": 58_868,
                },
            )
            validate_model_registration(REPO_ROOT, family)

    def test_raw_amazon_cd_release_is_hard_pinned(self):
        audit = audit_amazon_cd_source(REPO_ROOT / "dataset")
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["bytes"], 152_336_079)
        self.assertEqual(audit["lines_including_header"], 3_749_005)
        self.assertEqual(
            audit["sha256"],
            "7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4",
        )

    def test_sl_overlay_is_model_only_and_pins_the_stated_chart_semantics(self):
        path = REPO_ROOT / "baseline_config_fixed" / "AGCFSL8Coord_cd.yaml"
        overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
        forbidden = {
            "dataset",
            "seed",
            "val_interval",
            "user_inter_num_interval",
            "item_inter_num_interval",
            "metrics",
            "topk",
            "valid_metric",
            "eval_args",
        }
        self.assertFalse(forbidden.intersection(overlay))
        self.assertEqual(overlay["model"], "AGCFSL8Coord")
        self.assertEqual(overlay["embedding_size"], 63)
        self.assertEqual(overlay["matrix_dim"], 8)
        self.assertEqual(overlay["num_factors"], 1)
        self.assertEqual(overlay["schatten_p"], 2)
        self.assertFalse(overlay["symmetric_distance"])
        self.assertEqual(overlay["log_terms"], 12)
        self.assertEqual(overlay["log_jitter"], 0.0)
        self.assertEqual(overlay["coord_clip"], 1.0)

    def test_search_is_small_staged_and_not_a_cartesian_product(self):
        for slug, family in FAMILIES.items():
            protocol = validate_protocol(REPO_ROOT, family)
            anchor = Parameters.from_mapping(protocol["base_parameters"])
            counts = {
                stage: len(build_stage_trials(stage, anchor, family))
                for stage in stage_order(family)
            }
            self.assertEqual(counts["pilot"], 1)
            self.assertEqual(counts["dynamics"], 3)
            self.assertEqual(counts["metric"], 3)
            self.assertEqual(counts["forces"], 5)
            self.assertEqual(counts["optimizer"], 3)
            if slug == "agcf-sl8coord":
                self.assertEqual(counts["sl8-chart"], 2)
            # Every non-pilot variant changes only a compact blocked subset;
            # there is no product of all omitted hyperparameters.
            self.assertLessEqual(sum(counts.values()), 17)


class AGCFRunnerCommandTest(unittest.TestCase):
    def test_default_pilots_are_serial_validation_only_and_save_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in FAMILIES.values():
                args = _args(root, gpu_id="7")
                protocol = validate_protocol(REPO_ROOT, family)
                parameters = Parameters.from_mapping(protocol["base_parameters"])
                trial = build_stage_trials("pilot", parameters, family)[0]
                result = root / family.slug / "result.json"
                checkpoint = root / family.slug / "checkpoint"
                command = trial_command(args, family, trial, result, checkpoint)

                self.assertEqual(command[command.index("--model") + 1], family.model)
                self.assertIn("--validation-only", command)
                self.assertNotIn("--no-save", command)
                self.assertIn("--epochs=500", command)
                self.assertIn("--eval_step=10", command)
                self.assertIn("--stopping_step=30", command)
                self.assertIn(f"--data_path={REPO_ROOT / 'dataset'}", command)
                self.assertIn("--gpu_id=7", command)
                self.assertNotIn("--gpu_id=0", command)
                self.assertIn("--use_gpu=True", command)
                self.assertIn("--result-file", command)
                self.assertTrue(any(arg.startswith("--checkpoint_dir=") for arg in command))
                self.assertFalse(any("final-test" in arg.lower() for arg in command))
                config_arg = command[command.index("--config-files") + 1]
                self.assertEqual(
                    [Path(item).name for item in config_arg.split()],
                    list(family.config_names),
                )

    def test_dry_run_all_contains_only_one_process_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            family = FAMILIES["agcf-sl8coord"]
            args = _args(Path(temporary), stage="all")
            protocol = validate_protocol(REPO_ROOT, family)
            contract = campaign_contract(protocol, args)
            plan = dry_run_plan(args, family, contract)

        self.assertEqual(plan["serial_workers"], 1)
        self.assertEqual(plan["physical_gpu"], "0")
        self.assertTrue(plan["validation_only"])
        self.assertFalse(plan["test_evaluated"])
        commands = [job["command"] for stage in plan["stages"] for job in stage["jobs"]]
        self.assertTrue(commands)
        self.assertTrue(all("--validation-only" in command for command in commands))
        self.assertTrue(all("--gpu_id=0" in command for command in commands))

    def test_multi_gpu_tokens_are_rejected(self):
        args = _args(Path("/tmp/agcf"), gpu_id="0,1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_args(args, FAMILIES["agcf"])


class AGCFRunnerResumeTest(unittest.TestCase):
    def _write_checkpoint_and_raw_result(
        self,
        root: Path,
        family,
        trial: Trial,
        contract,
    ) -> Path:
        training = contract["training"]
        config = {
            "model": family.model,
            "dataset": DATASET,
            "seed": 2024,
            "embedding_size": family.embedding_size,
            "epochs": training["epochs"],
            "eval_step": training["eval_step"],
            "stopping_step": training["stopping_step"],
            "valid_metric": "Recall@10",
            "data_path": str(
                Path(contract["protocol"]["raw_source"]["data_root"]) / DATASET
            ),
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
            **trial.parameters.recbole_values(family),
        }
        if family.is_sl8_chart:
            config.update(
                {
                    "matrix_dim": 8,
                    "num_factors": 1,
                    "schatten_p": 2,
                    "symmetric_distance": False,
                    "pairwise_loss": "hinge",
                }
            )
        checkpoint = root / "checkpoint.pth"
        torch.save({"config": config, "state_dict": {}}, checkpoint)
        result = root / "result.json"
        result.write_text(
            json.dumps(
                {
                    "model": family.model,
                    "dataset": DATASET,
                    "seed": 2024,
                    "best_valid_score": 0.05,
                    "best_valid_result": {"recall@10": 0.05, "ndcg@10": 0.03},
                    "test_result": None,
                    "checkpoint_file": str(checkpoint),
                    "split_fingerprints": {
                        "train": {"interactions": 10, "sha256": "train"},
                        "valid": {"interactions": 2, "sha256": "valid"},
                        "test": {"interactions": 2, "sha256": "test"},
                    },
                }
            ),
            encoding="utf-8",
        )
        return result

    def test_resume_requires_exact_metadata_checkpoint_and_no_test_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            family = FAMILIES["agcf"]
            args = _args(root)
            protocol = validate_protocol(REPO_ROOT, family)
            contract = campaign_contract(protocol, args)
            parameters = Parameters.from_mapping(protocol["base_parameters"])
            trial = Trial("pilot", "paper-guided-anchor", parameters)
            result = self._write_checkpoint_and_raw_result(
                root, family, trial, contract
            )
            annotate_result(result, family, trial, contract)
            complete = load_complete_result(result, family, trial, contract)
            self.assertEqual(complete["best_valid_score"], 0.05)

            payload = json.loads(result.read_text(encoding="utf-8"))
            payload["test_result"] = {"recall@10": 1.0}
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                load_complete_result(result, family, trial, contract)


if __name__ == "__main__":
    unittest.main()
