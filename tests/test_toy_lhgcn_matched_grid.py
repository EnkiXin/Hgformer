"""Contracts for the matched Amazon-Toy LHGCN layer/batch grid."""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from recbole_gnn.model.general_recommender.lhgcn import LHGCN
from slrec_experiments.run_toy_lhgcn_matched_grid import (
    ANCHOR,
    BATCH_SIZES,
    DATASET,
    EMBEDDING_PARAMETER_COUNT,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SPLIT_FINGERPRINTS,
    LAYERS,
    LORENTZ_BN_PARAMETER_COUNT,
    Trial,
    annotate_result,
    completed_result,
    exclusive_gpu_lock,
    expected_checkpoint_values,
    grid_trials,
    main,
    result_paths,
    trial_command,
    validate_checkpoint_contract,
    validate_protocol,
    write_summary,
)
from tests.test_lhgcn_adapter import _TinyGraphDataset, _config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo=REPO_ROOT,
        data_root=REPO_ROOT / "dataset",
        output_root=root,
        python="python",
        gpu_id="7",
    )


def _state_dict() -> dict[str, torch.Tensor]:
    beta = torch.zeros(64)
    beta[0] = math.sqrt(2.0)
    return {
        "embedding.weight": torch.zeros(25_226, 64),
        "gcn_conv.layer_norm.gamma": torch.ones(1),
        "gcn_conv.layer_norm.curve": torch.tensor(0.5),
        "gcn_conv.layer_norm.beta": beta,
    }


def _write_checkpoint(
    root: Path,
    trial: Trial,
    *,
    config_updates: dict[str, object] | None = None,
    state_updates: dict[str, torch.Tensor] | None = None,
) -> Path:
    config = expected_checkpoint_values(trial)
    config.update(config_updates or {})
    state = _state_dict()
    state.update(state_updates or {})
    checkpoint = root / f"{trial.name}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config, "epoch": 499, "state_dict": state},
        checkpoint,
    )
    return checkpoint


def _write_raw_result(
    root: Path,
    trial: Trial,
    *,
    recall: float = 0.0659,
    ndcg: float = 0.0375,
) -> Path:
    checkpoint = _write_checkpoint(root, trial)
    result = root / f"{trial.name}.raw.json"
    payload = {
        "model": "LHGCN",
        "dataset": DATASET,
        "config_files": [
            str(REPO_ROOT / "baseline_config_fixed" / "RecFormer_toy.yaml"),
            str(REPO_ROOT / "baseline_config_fixed" / "LHGCN_reproduction.yaml"),
        ],
        "seed": 2024,
        "epochs": 500,
        "stopping_step": 1000,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "best_valid_score": recall,
        "best_valid_result": {"recall@10": recall, "ndcg@10": ndcg},
        "test_result": None,
        "checkpoint_file": str(checkpoint),
        "split_fingerprints": EXPECTED_SPLIT_FINGERPRINTS,
        "model_diagnostics": None,
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


class GridProtocolAndCommandTest(unittest.TestCase):
    def test_grid_is_exact_and_parameter_budget_accounts_for_shared_gamma(self):
        self.assertEqual(ANCHOR, (4, 65_536))
        self.assertEqual(EMBEDDING_PARAMETER_COUNT, 1_614_464)
        self.assertEqual(LORENTZ_BN_PARAMETER_COUNT, 1)
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 1_614_465)
        trials = grid_trials()
        self.assertEqual(len(trials), 15)
        self.assertEqual(
            {(trial.gcn_layers, trial.train_batch_size) for trial in trials},
            {(layer, batch) for layer in LAYERS for batch in BATCH_SIZES},
        )

    def test_protocol_is_exact_toy_full_ranking_validation_only(self):
        protocol = validate_protocol(REPO_ROOT)
        self.assertEqual(protocol["dataset"], DATASET)
        self.assertEqual(protocol["seed"], 2024)
        self.assertEqual(protocol["split"]["args"]["mode"], "full")
        self.assertEqual(
            protocol["split"]["exact_fingerprints"],
            EXPECTED_SPLIT_FINGERPRINTS,
        )
        self.assertTrue(protocol["evaluation"]["validation_only"])
        self.assertFalse(protocol["evaluation"]["held_out_test_evaluated"])
        self.assertEqual(
            protocol["fixed_model_training"]["released_equivalent"],
            "HGCF + conv=lGCN",
        )

    def test_command_uses_explicit_adapter_and_only_varies_layer_and_batch(self):
        command = trial_command(
            _args(Path("/tmp/lhgcn-grid")),
            Trial(10, 131_072),
            Path("/tmp/raw.json"),
            Path("/tmp/checkpoints"),
        )
        joined = " ".join(command)
        self.assertIn("--validation-only", command)
        self.assertNotIn("test", " ".join(token.lower() for token in command))
        self.assertEqual(command[command.index("--model") + 1], "LHGCN")
        self.assertIn("--conv=lGCN", command)
        self.assertIn("--gpu_id=7", command)
        self.assertIn("--epochs=500", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn("--stopping_step=1000", command)
        self.assertIn("--gcn_layers=10", command)
        self.assertIn("--train_batch_size=131072", command)
        self.assertFalse(any(token.startswith("--n_layers=") for token in command))
        self.assertIn("RecFormer_toy.yaml", joined)
        self.assertIn("LHGCN_reproduction.yaml", joined)

    def test_released_lgcn_reuses_one_batchnorm_at_all_depths(self):
        parameter_counts = []
        for layers in (2, 10):
            model = LHGCN(_config(gcn_layers=layers), _TinyGraphDataset())
            parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
            self.assertEqual(
                [name for name, _ in model.named_parameters()],
                ["embedding.weight", "gcn_conv.layer_norm.gamma"],
            )
            self.assertEqual(model.gcn_conv.layer_norm.gamma.numel(), 1)
        self.assertEqual(parameter_counts, [65, 65])


class DeepCheckpointAndResumeTest(unittest.TestCase):
    def test_checkpoint_proves_exact_embedding_plus_shared_gamma_structure(self):
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit = validate_checkpoint_contract(
                _write_checkpoint(root, trial), REPO_ROOT, trial
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["combined_embedding_shape"], [25_226, 64])
        self.assertEqual(audit["combined_embedding_parameters"], 1_614_464)
        self.assertEqual(audit["shared_lorentz_batch_norm_gamma_parameters"], 1)
        self.assertEqual(audit["parameter_count"], 1_614_465)
        self.assertEqual(audit["depth_dependent_parameters"], 0)

    def test_wrong_config_or_state_cannot_be_resumed(self):
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wrong_conv = _write_checkpoint(
                root / "conv", trial, config_updates={"conv": "resSumGCN"}
            )
            with self.assertRaisesRegex(ValueError, "checkpoint trial contract mismatch"):
                validate_checkpoint_contract(wrong_conv, REPO_ROOT, trial)

            wrong_gamma = _write_checkpoint(
                root / "gamma",
                trial,
                state_updates={"gcn_conv.layer_norm.gamma": torch.ones(2)},
            )
            with self.assertRaisesRegex(ValueError, "one shared LorentzBatchNorm gamma"):
                validate_checkpoint_contract(wrong_gamma, REPO_ROOT, trial)

            extra_parameter = _write_checkpoint(root / "extra", trial)
            checkpoint = torch.load(extra_parameter, map_location="cpu", weights_only=False)
            checkpoint["state_dict"]["gcn_conv.fake.weight"] = torch.ones(1)
            torch.save(checkpoint, extra_parameter)
            with self.assertRaisesRegex(ValueError, "not the parameter-free released lGCN"):
                validate_checkpoint_contract(extra_parameter, REPO_ROOT, trial)

    def test_resume_requires_exact_split_hashes_and_never_accepts_test_metrics(self):
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
                    "source": "unit-test",
                },
            )
            complete, reason = completed_result(
                final, repo=REPO_ROOT, trial=trial, protocol=protocol
            )
            self.assertIsNotNone(complete)
            self.assertIsNone(reason)

            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["split_fingerprints"]["valid"]["sha256"] = "0" * 64
            final.write_text(json.dumps(payload), encoding="utf-8")
            invalid, reason = completed_result(
                final, repo=REPO_ROOT, trial=trial, protocol=protocol
            )
            self.assertIsNone(invalid)
            self.assertIn("exact seed-2024 Toy split", reason)

            payload["split_fingerprints"] = EXPECTED_SPLIT_FINGERPRINTS
            payload["test_result"] = {"recall@10": 0.1}
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                completed_result(
                    final, repo=REPO_ROOT, trial=trial, protocol=protocol
                )


class SummaryLockAndDryRunTest(unittest.TestCase):
    def test_summary_pairs_only_the_same_layer_and_batch_sl8_cell(self):
        protocol = validate_protocol(REPO_ROOT)
        trial = Trial(*ANCHOR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = result_paths(root, trial)
            raw = _write_raw_result(root, trial, recall=0.0659, ndcg=0.0375)
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
            sl8_summary = root / "sl8-summary.json"
            sl8_summary.write_text(
                json.dumps(
                    {
                        "kind": "amazon-toy-sl8lhgcn-layer-batch-grid-summary",
                        "dataset": DATASET,
                        "state": "incomplete",
                        "test_evaluated": False,
                        "split_fingerprints": EXPECTED_SPLIT_FINGERPRINTS,
                        "ranking": [
                            {
                                "gcn_layers": 4,
                                "train_batch_size": 65_536,
                                "result_file": "/tmp/sl8.json",
                                "recall@10": 0.0717,
                                "ndcg@10": 0.0389,
                                "test_evaluated": False,
                            },
                            {
                                "gcn_layers": 2,
                                "train_batch_size": 32_768,
                                "result_file": "/tmp/sl8-l2.json",
                                "recall@10": 0.085,
                                "ndcg@10": 0.048,
                                "test_evaluated": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = write_summary(
                root / "summary.json",
                repo=REPO_ROOT,
                output_root=root,
                protocol=protocol,
                sl8_summary_path=sl8_summary,
            )
        self.assertEqual(summary["state"], "incomplete")
        self.assertEqual(summary["paired_sl8_comparison"]["completed_same_cell_pairs"], 1)
        candidate = summary["ranking"][0]
        self.assertEqual(candidate["gcn_layers"], 4)
        self.assertEqual(candidate["train_batch_size"], 65_536)
        self.assertAlmostEqual(
            candidate["paired_sl8"]["delta_sl8_minus_lhgcn"]["recall@10"],
            0.0058,
        )

    def test_shared_gpu_lock_rejects_a_second_owner(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = Path(temporary_directory) / "shared-gpu7.lock"
            with exclusive_gpu_lock(lock, "7"):
                with self.assertRaisesRegex(RuntimeError, "already reserved"):
                    with exclusive_gpu_lock(lock, "7"):
                        pass

    def test_dry_run_defaults_to_gpu7_and_writes_nothing(self):
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
