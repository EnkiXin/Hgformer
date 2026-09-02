"""Focused contracts for the one-trial AGCF-SL8Coord MovieLens pilot."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from slrec_experiments.run_agcf_sl8coord_movielens_pilot import (
    CONFIG_NAMES,
    DATASET,
    EPOCHS,
    EVAL_STEP,
    MODEL,
    PILOT_PARAMETERS,
    SEED,
    STOPPING_STEP,
    _validate_args,
    artifact_paths,
    campaign_contract,
    trial_command,
    validate_protocol,
    validate_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(root: Path, **updates) -> argparse.Namespace:
    values = {
        "repo": REPO_ROOT,
        "data_path": REPO_ROOT / "dataset",
        "output_root": root,
        "python": "python",
        "gpu_id": "0",
        "plan_only": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class AGCFSL8CoordMovieLensPilotTest(unittest.TestCase):
    def test_last_overlay_fixes_sl8_anchor_without_protocol_fields(self):
        path = REPO_ROOT / "baseline_config_fixed" / CONFIG_NAMES[-1]
        overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(overlay["model"], MODEL)
        for key, value in PILOT_PARAMETERS.items():
            self.assertEqual(overlay[key], value)
        forbidden = {
            "dataset", "seed", "eval_args", "metrics", "topk", "valid_metric",
            "epochs", "eval_step", "stopping_step", "val_interval",
        }
        self.assertFalse(forbidden.intersection(overlay))

        protocol = validate_protocol(REPO_ROOT)
        self.assertEqual([Path(item["path"]).name for item in protocol["config_files"]], list(CONFIG_NAMES))
        self.assertEqual(protocol["split"]["mode"], "full")
        self.assertTrue(protocol["evaluation"]["validation_only"])
        self.assertFalse(protocol["evaluation"]["held_out_test_evaluated"])

    def test_command_is_one_gpu_exact_validation_only_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _args(root, gpu_id="7")
            command = trial_command(args, artifact_paths(root))
        self.assertEqual(command[command.index("--model") + 1], MODEL)
        self.assertEqual(command[command.index("--dataset") + 1], DATASET)
        self.assertIn("--validation-only", command)
        self.assertNotIn("--test", command)
        self.assertNotIn("--no-save", command)
        self.assertIn(f"--gpu_id=7", command)
        self.assertIn(f"--epochs={EPOCHS}", command)
        self.assertIn(f"--eval_step={EVAL_STEP}", command)
        self.assertIn(f"--stopping_step={STOPPING_STEP}", command)
        self.assertIn(f"--seed={SEED}", command)
        configs = command[command.index("--config-files") + 1].split()
        self.assertEqual([Path(item).name for item in configs], list(CONFIG_NAMES))

    def test_multi_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_args(_args(Path("/tmp/sl8-pilot"), gpu_id="0,1"))

    def test_result_requires_untouched_test_and_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = validate_protocol(REPO_ROOT)
            contract = campaign_contract(REPO_ROOT, root / "data", protocol)
            checkpoint = root / "model.pth"
            config = {
                "model": MODEL,
                "dataset": DATASET,
                "seed": SEED,
                "epochs": EPOCHS,
                "eval_step": EVAL_STEP,
                "stopping_step": STOPPING_STEP,
                "valid_metric": "Recall@10",
                "data_path": str((root / "data").resolve() / DATASET),
                "eval_args": {
                    "split": {"RS": [0.8, 0.1, 0.1]},
                    "group_by": "user",
                    "order": "RO",
                    "mode": "full",
                },
                **PILOT_PARAMETERS,
            }
            torch.save({"config": config, "state_dict": {}}, checkpoint)
            result = root / "result.json"
            payload = {
                "model": MODEL,
                "dataset": DATASET,
                "seed": SEED,
                "best_valid_score": 0.1,
                "best_valid_result": {"recall@10": 0.1},
                "test_result": None,
                "checkpoint_file": str(checkpoint),
                "split_fingerprints": {
                    "train": {"interactions": 10, "sha256": "a"},
                    "valid": {"interactions": 2, "sha256": "b"},
                    "test": {"interactions": 2, "sha256": "c"},
                },
            }
            result.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(validate_result(result, contract)["test_result"], None)

            payload["test_result"] = {"recall@10": 1.0}
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                validate_result(result, contract)


if __name__ == "__main__":
    unittest.main()
