"""Contracts for the Amazon-CD SL8 multi-fidelity runner."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slrec_experiments.run_sl8_cd_multifidelity import (
    BATCH_SIZES,
    EVAL_BATCH_SIZE,
    EVAL_ITEM_CHUNK_SIZE,
    EVAL_USER_CHUNK_SIZE,
    FINAL_500,
    LAYERS,
    LEARNING_RATES,
    LOSS_MARGINS,
    RERANK_100,
    SCREEN_50,
    Parameters,
    _gpu_token,
    adaptive_survivor_count,
    completed_result,
    exclusive_gpu_lock,
    full_grid,
    manifest,
    rank_candidates,
    result_paths,
    select_diverse_top16,
    spearman_rank,
    trial_command,
    trial_metadata,
    expected_checkpoint_values,
    validate_checkpoint_contract,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**updates):
    defaults = {
        "repo": REPO_ROOT,
        "data_root": REPO_ROOT / "dataset",
        "output_root": Path("/tmp/sl8-cd-mf-tests"),
        "python": "python",
        "gpu_id": "7",
    }
    defaults.update(updates)
    return argparse.Namespace(**defaults)


def _candidate(parameters: Parameters, recall: float, ndcg: float):
    return {
        "trial": parameters.name,
        "parameters": parameters.__dict__,
        "gcn_layers": parameters.gcn_layers,
        "recall@10": recall,
        "ndcg@10": ndcg,
    }


class GridAndScheduleTest(unittest.TestCase):
    def test_complete_requested_grid_has_exactly_96_unique_trials(self):
        grid = full_grid()
        self.assertEqual(len(grid), 96)
        self.assertEqual(len(set(grid)), 96)
        self.assertEqual({item.gcn_layers for item in grid}, set(LAYERS))
        self.assertEqual({item.train_batch_size for item in grid}, set(BATCH_SIZES))
        self.assertEqual({item.learning_rate for item in grid}, set(LEARNING_RATES))
        self.assertEqual({item.loss_margin for item in grid}, set(LOSS_MARGINS))
        self.assertEqual({item.schatten_p for item in grid}, {2})

    def test_manifest_discloses_adaptive_counts_and_validation_cost(self):
        payload = manifest(validate_protocol(REPO_ROOT))
        self.assertEqual(payload["grid"]["cartesian_trials"], 96)
        self.assertEqual(
            payload["totals"]["fresh_training_runs"], {"minimum": 118, "maximum": 122}
        )
        self.assertEqual(
            payload["totals"]["exact_full_ranking_validations"],
            {"minimum": 216, "maximum": 220},
        )
        self.assertFalse(payload["held_out_test_evaluated"])

    def test_exact_chunks_are_the_measured_end_to_end_winner(self):
        self.assertEqual(EVAL_BATCH_SIZE, 1_048_576)
        self.assertEqual(EVAL_USER_CHUNK_SIZE, 64)
        self.assertEqual(EVAL_ITEM_CHUNK_SIZE, 1024)

    def test_layer_diverse_selection_takes_four_per_layer(self):
        candidates = []
        for index, parameters in enumerate(full_grid()):
            candidates.append(_candidate(parameters, 1.0 - index / 1000.0, 0.5))
        selected = select_diverse_top16(candidates)
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            {layer: sum(item.gcn_layers == layer for item in selected) for layer in LAYERS},
            {layer: 4 for layer in LAYERS},
        )

    def test_ranking_uses_ndcg_then_name_as_tie_breaks(self):
        first, second = full_grid()[:2]
        ranking = rank_candidates(
            [_candidate(first, 0.1, 0.02), _candidate(second, 0.1, 0.03)]
        )
        self.assertEqual(ranking[0]["trial"], second.name)

    def test_spearman_is_exact_for_same_and_reversed_ranks(self):
        parameters = full_grid()[:4]
        ascending = [
            _candidate(item, 1.0 - index / 10.0, 0.1)
            for index, item in enumerate(parameters)
        ]
        reversed_ranks = [
            _candidate(item, 0.1 + index / 10.0, 0.1)
            for index, item in enumerate(parameters)
        ]
        self.assertEqual(spearman_rank(ascending, ascending), 1.0)
        self.assertEqual(spearman_rank(ascending, reversed_ranks), -1.0)
        self.assertEqual(adaptive_survivor_count(0.499999), 8)
        self.assertEqual(adaptive_survivor_count(0.5), 4)


class CommandAndGpuSafetyTest(unittest.TestCase):
    def test_physical_gpu_must_be_one_nonnegative_index(self):
        self.assertEqual(_gpu_token("07"), "7")
        for invalid in ("", "-1", "0,1", "cuda:0", "GPU-uuid"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _gpu_token(invalid)

    def test_screen_command_is_fresh_validation_only_and_exact_full_sort(self):
        parameters = full_grid()[0]
        command = trial_command(
            _args(),
            SCREEN_50,
            parameters,
            Path("/tmp/raw.json"),
            Path("/tmp/checkpoint"),
        )
        self.assertIn("--validation-only", command)
        self.assertNotIn("--no-save", command)
        self.assertFalse(any("test" in token.lower() for token in command))
        self.assertFalse(any("uni100" in token.lower() for token in command))
        self.assertIn("--gpu_id=7", command)
        self.assertNotIn("--gpu_id=0", command)
        self.assertIn("--epochs=50", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn(f"--eval_batch_size={EVAL_BATCH_SIZE}", command)
        self.assertIn(f"--eval_user_chunk_size={EVAL_USER_CHUNK_SIZE}", command)
        self.assertIn(f"--eval_item_chunk_size={EVAL_ITEM_CHUNK_SIZE}", command)
        self.assertIn("--fast_one_sided_frobenius=true", command)
        self.assertIn("--schatten_p=2", command)
        self.assertFalse(any("resume" in token.lower() for token in command))

    def test_finalists_are_fresh_500_epoch_runs_with_eval_step_10(self):
        parameters = full_grid()[0]
        command = trial_command(
            _args(), FINAL_500, parameters, Path("/tmp/raw"), Path("/tmp/ckpt")
        )
        self.assertIn("--epochs=500", command)
        self.assertIn("--eval_step=10", command)

    def test_lock_rejects_a_second_runner_on_same_physical_card(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gpu7.lock"
            with exclusive_gpu_lock(path, "7"):
                with self.assertRaisesRegex(RuntimeError, "already reserved"):
                    with exclusive_gpu_lock(path, "7"):
                        pass


class ResumeEvidenceTest(unittest.TestCase):
    def test_checkpoint_epoch_must_be_the_only_scheduled_validation_epoch(self):
        parameters = full_grid()[0]
        config = {**expected_checkpoint_values(SCREEN_50, parameters), "gpu_id": 7}
        with patch(
            "slrec_experiments.run_sl8_cd_multifidelity._load_checkpoint",
            return_value=(config, {"epoch": 49, "state_dict": {"x": 1}}),
        ), patch(
            "slrec_experiments.run_sl8_cd_multifidelity._sha256",
            return_value="a" * 64,
        ):
            epoch, digest = validate_checkpoint_contract(
                Path("/tmp/checkpoint.pth"),
                repo=REPO_ROOT,
                stage=SCREEN_50,
                parameters=parameters,
                gpu_id="7",
            )
        self.assertEqual(epoch, 49)
        self.assertEqual(digest, "a" * 64)

        with patch(
            "slrec_experiments.run_sl8_cd_multifidelity._load_checkpoint",
            return_value=(config, {"epoch": 48, "state_dict": {"x": 1}}),
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint epoch"):
                validate_checkpoint_contract(
                    Path("/tmp/checkpoint.pth"),
                    repo=REPO_ROOT,
                    stage=SCREEN_50,
                    parameters=parameters,
                    gpu_id="7",
                )

    def _payload(
        self,
        root: Path,
        parameters: Parameters,
        protocol,
        *,
        touched_test=False,
    ):
        paths = result_paths(root, SCREEN_50, parameters)
        paths["checkpoint_dir"].mkdir(parents=True)
        checkpoint = paths["checkpoint_dir"] / "model.pth"
        checkpoint.write_bytes(b"trusted checkpoint fixture")
        fingerprints = {
            split: {"interactions": count, "sha256": split[0] * 64}
            for split, count in {
                "train": 746_199,
                "valid": 103_174,
                "test": 103_174,
            }.items()
        }
        diagnostics = {
            "mode": "ambient_retract",
            "layers": parameters.gcn_layers,
            "active_singular_fallbacks": 0,
            "output_membership_violations": 0,
            "nonpositive_output_determinants": 0,
            "nonfinite_output_log_determinants": 0,
            "layer_membership": [{} for _ in range(parameters.gcn_layers)],
            "distance_membership": {
                "relative_membership_violations": 0,
                "nonfinite_approximate_logs": 0,
            },
        }
        metadata = trial_metadata(SCREEN_50, parameters, protocol, "7", full_grid())
        return paths, {
            "model": "SL8LHGCN",
            "dataset": "Amazon_cd",
            "seed": 2024,
            "epochs": 50,
            "stopping_step": 1000,
            "config_files": [
                str(REPO_ROOT / "baseline_config_fixed/RecFormer_cd.yaml"),
                str(REPO_ROOT / "baseline_config_fixed/SL8LHGCN_reproduction.yaml"),
            ],
            "best_valid_score": 0.012,
            "best_valid_result": {"recall@10": 0.012, "ndcg@10": 0.006},
            "test_result": {"recall@10": 1.0} if touched_test else None,
            "checkpoint_file": str(checkpoint),
            "split_fingerprints": fingerprints,
            "model_diagnostics": diagnostics,
            "sl8_multifidelity": metadata,
            "checkpoint_artifact": {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": "a" * 64,
            },
        }

    def test_completed_result_requires_metadata_and_checkpoint_hash(self):
        protocol = validate_protocol(REPO_ROOT)
        parameters = full_grid()[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, payload = self._payload(root, parameters, protocol)
            paths["result"].parent.mkdir(parents=True)
            import json

            paths["result"].write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "slrec_experiments.run_sl8_cd_multifidelity.validate_checkpoint_contract",
                return_value=(49, "a" * 64),
            ):
                complete, reason = completed_result(
                    paths["result"],
                    repo=REPO_ROOT,
                    stage=SCREEN_50,
                    parameters=parameters,
                    protocol=protocol,
                    gpu_id="7",
                    parent_parameters=full_grid(),
                    checkpoint_dir=paths["checkpoint_dir"],
                )
            self.assertIsNotNone(complete)
            self.assertIsNone(reason)

            payload["checkpoint_artifact"]["sha256"] = "b" * 64
            paths["result"].write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "slrec_experiments.run_sl8_cd_multifidelity.validate_checkpoint_contract",
                return_value=(49, "a" * 64),
            ):
                complete, reason = completed_result(
                    paths["result"],
                    repo=REPO_ROOT,
                    stage=SCREEN_50,
                    parameters=parameters,
                    protocol=protocol,
                    gpu_id="7",
                    parent_parameters=full_grid(),
                    checkpoint_dir=paths["checkpoint_dir"],
                )
            self.assertIsNone(complete)
            self.assertIn("hash changed", reason)

    def test_test_touched_result_is_fatal_not_silently_retrained(self):
        protocol = validate_protocol(REPO_ROOT)
        parameters = full_grid()[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, payload = self._payload(root, parameters, protocol, touched_test=True)
            paths["result"].parent.mkdir(parents=True)
            import json

            paths["result"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                completed_result(
                    paths["result"],
                    repo=REPO_ROOT,
                    stage=SCREEN_50,
                    parameters=parameters,
                    protocol=protocol,
                    gpu_id="7",
                    parent_parameters=full_grid(),
                    checkpoint_dir=paths["checkpoint_dir"],
                )


if __name__ == "__main__":
    unittest.main()
