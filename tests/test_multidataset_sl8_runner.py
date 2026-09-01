"""Safety and orchestration contracts for the seven-dataset SL8 runner."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from slrec_experiments.run_multidataset_sl8 import (
    DATASETS,
    DatasetSpec,
    FilteredReference,
    CountRange,
    _selection_metadata,
    audit_source_file,
    completed_selection,
    final_test_command,
    main,
    profile_trials,
    recformer_command,
    select_datasets,
    sl8_command,
    tuning_protocol,
    validate_fixed_protocol,
    write_tuning_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    defaults = {
        "repo": REPO_ROOT,
        "data_root": REPO_ROOT / "dataset",
        "output_root": Path("/tmp/multidataset-sl8"),
        "python": "python",
        "gpu_id": "0",
        "epochs": 500,
        "eval_step": 50,
        "tuning_profile": "core",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class RegistryAndProtocolTest(unittest.TestCase):
    def test_registry_contains_the_seven_requested_datasets(self):
        self.assertEqual(
            [item.dataset for item in DATASETS],
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
        douban = [item for item in DATASETS if item.douban_official]
        self.assertEqual(len(douban), 3)
        self.assertTrue(all(item.source_bytes for item in douban))
        self.assertTrue(all(item.source_sha256 for item in douban))

    def test_each_fixed_config_is_the_protocol_authority(self):
        for spec in DATASETS:
            protocol = validate_fixed_protocol(REPO_ROOT, spec)
            self.assertEqual(protocol["dataset"], spec.dataset)
            self.assertEqual(protocol["seed"], 2024)
            self.assertEqual(protocol["filters"]["rating"], "[3,inf)")
            self.assertEqual(protocol["validation"]["eval_args"]["mode"], "full")
            self.assertEqual(
                protocol["validation"]["eval_args"]["split"],
                {"RS": [0.8, 0.1, 0.1]},
            )
            self.assertEqual(protocol["validation"]["metrics"], ["Recall", "NDCG"])
            self.assertEqual(protocol["validation"]["topk"], [5, 10, 20, 50])

    def test_movies_keeps_ten_core_and_other_datasets_keep_five_core(self):
        intervals = {
            spec.dataset: validate_fixed_protocol(REPO_ROOT, spec)["filters"]["users"]
            for spec in DATASETS
        }
        self.assertEqual(intervals["Amazon_movies"], "[10,inf)")
        self.assertTrue(
            all(value == "[5,inf)" for key, value in intervals.items() if key != "Amazon_movies")
        )


class DataAcceptanceTest(unittest.TestCase):
    def test_small_douban_file_is_rejected_as_possible_copd(self):
        spec = next(item for item in DATASETS if item.dataset == "DoubanBook")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / spec.dataset
            directory.mkdir()
            (directory / f"{spec.dataset}.inter").write_text(
                "user_id:token\titem_id:token\trating:float\n1\t1\t5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CoPD/small Douban"):
                audit_source_file(root, spec)

    def test_source_rows_header_size_and_hash_are_accepted(self):
        content = (
            "user_id:token\titem_id:token\trating:float\n"
            "u1\ti1\t5\n"
            "u2\ti2\t3\n"
        ).encode("utf-8")
        import hashlib

        spec = DatasetSpec(
            slug="synthetic",
            dataset="Synthetic",
            recformer_config="unused.yaml",
            source_rows=2,
            source_bytes=len(content),
            source_sha256=hashlib.sha256(content).hexdigest(),
            filtered=FilteredReference(
                CountRange.exact(2), CountRange.exact(2), CountRange.exact(2)
            ),
            source_release="synthetic fixture",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / spec.dataset
            directory.mkdir()
            path = directory / f"{spec.dataset}.inter"
            path.write_bytes(content)
            audit = audit_source_file(root, spec)

        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["source_rows"], 2)
        self.assertEqual(audit["bytes"], len(content))

    def test_bad_header_is_rejected(self):
        base = DATASETS[2]
        content = b"uid:token\tiid:token\trating:float\n1\t1\t5\n"
        spec = replace(
            base,
            source_rows=1,
            source_bytes=len(content),
            source_sha256=None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / spec.dataset
            directory.mkdir()
            (directory / f"{spec.dataset}.inter").write_bytes(content)
            with self.assertRaisesRegex(ValueError, "header lacks"):
                audit_source_file(root, spec)


class ShardingTest(unittest.TestCase):
    def test_dataset_shards_are_disjoint_and_cover_registry(self):
        shards = [select_datasets(["all"], index, 3) for index in range(3)]
        flattened = [item.slug for shard in shards for item in shard]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {item.slug for item in DATASETS})

    def test_names_and_comma_separated_slugs_are_supported(self):
        selected = select_datasets(["amazon-cd,DoubanMusic"], 0, 1)
        self.assertEqual([item.dataset for item in selected], ["Amazon_cd", "DoubanMusic"])

    def test_invalid_shard_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 <= index < count"):
            select_datasets(["all"], 2, 2)


class CommandContractTest(unittest.TestCase):
    def test_recformer_selection_is_validation_only_and_uses_fixed_config(self):
        spec = DATASETS[1]
        command = recformer_command(
            _args(gpu_id="7"), spec, Path("/tmp/selection.json"), Path("/tmp/checkpoints")
        )
        self.assertEqual(command[command.index("--model") + 1], "RecFormer")
        self.assertIn("--validation-only", command)
        self.assertIn(spec.recformer_config, command[command.index("--config-files") + 1])
        self.assertIn("--gpu_id=7", command)
        self.assertNotIn("--gpu_id=0", command)
        self.assertFalse(any("uni100" in token.lower() for token in command))

    def test_sl8_is_geometry_only_and_validates_full_ranking_every_fifty_epochs(self):
        spec = DATASETS[0]
        trial = profile_trials("paper")[0]
        command = sl8_command(
            _args(gpu_id="7"), trial=trial, spec=spec, result=Path("/tmp/result.json"), checkpoints=Path("/tmp/checkpoints")
        )
        self.assertEqual(command[command.index("--model") + 1], "SLRecGraph")
        self.assertIn("--validation-only", command)
        self.assertIn("--matrix_dim=8", command)
        self.assertIn("--embedding_size=64", command)
        self.assertIn("--n_layers=0", command)
        self.assertIn("--eval_step=50", command)
        self.assertIn("--epochs=500", command)
        self.assertIn("--stopping_step=11", command)
        self.assertIn("--symmetric_distance=false", command)
        self.assertIn("--gpu_id=7", command)
        self.assertNotIn("--gpu_id=0", command)
        self.assertFalse(any("uni100" in token.lower() for token in command))
        config_names = [Path(path).name for path in command[command.index("--config-files") + 1].split()]
        self.assertEqual(config_names, ["RecFormer_cd.yaml", "SLRecGraph_ablation_sl8.yaml"])

    def test_core_profile_reuses_the_twelve_cd_lr_clip_trials(self):
        trials = profile_trials("core")
        self.assertEqual(len(trials), 12)
        self.assertEqual(
            {(trial.parameters.learning_rate, trial.parameters.coord_clip) for trial in trials},
            {
                (learning_rate, coord_clip)
                for learning_rate in (0.001, 0.003, 0.006)
                for coord_clip in (0.5, 0.75, 1.0, 1.5)
            },
        )

    def test_final_test_uses_checkpoint_without_retraining_or_valid_evaluation(self):
        command = final_test_command(
            _args(), Path("/tmp/selection.json"), Path("/tmp/model.pth"), Path("/tmp/test.json")
        )
        self.assertIn("evaluate_recbole_gnn_checkpoint.py", " ".join(command))
        self.assertIn("--skip-valid", command)
        self.assertEqual(
            command[command.index("--full-sort-user-batch-size") + 1], "64"
        )
        self.assertNotIn("run_recbole_gnn.py", " ".join(command))


class DryRunTest(unittest.TestCase):
    def test_dry_run_writes_nothing_and_plans_recformer_before_twelve_sl8_trials(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            output = io.StringIO()
            with redirect_stdout(output):
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
            plan = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertFalse(output_root.exists())
        jobs = plan["datasets"][0]["jobs"]
        self.assertEqual(len(jobs), 13)
        self.assertEqual(jobs[0]["kind"], "recformer-validation-selection")
        self.assertTrue(all(job["status"] == "run" for job in jobs))
        self.assertFalse(plan["test_evaluated"])

    def test_skip_data_audit_is_forbidden_for_real_execution(self):
        with self.assertRaisesRegex(ValueError, "permitted only with --dry-run"):
            main(
                [
                    "--repo",
                    str(REPO_ROOT),
                    "--output-root",
                    "/tmp/never-created",
                    "--datasets",
                    "amazon-cd",
                    "--skip-data-audit",
                ]
            )


class ResumeAndSummaryTest(unittest.TestCase):
    def _write_selection(
        self,
        root: Path,
        spec,
        protocol,
        trial,
        *,
        kind="sl8",
        test_result=None,
    ):
        checkpoint = root / "model.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        path = root / "selection.json"
        model = "RecFormer" if kind == "recformer" else trial.parameters.model_name
        payload = {
            "model": model,
            "dataset": spec.dataset,
            "seed": 2024,
            "best_valid_score": 0.04,
            "best_valid_result": {"recall@10": 0.04, "ndcg@10": 0.02},
            "test_result": test_result,
            "checkpoint_file": str(checkpoint),
            "split_fingerprints": {
                "train": {"interactions": 8, "sha256": "a"},
                "valid": {"interactions": 1, "sha256": "b"},
                "test": {"interactions": 1, "sha256": "c"},
            },
            "multidataset": _selection_metadata(kind, spec, protocol, trial),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_matching_selection_resumes_but_changed_eval_step_does_not(self):
        spec = DATASETS[0]
        base = validate_fixed_protocol(REPO_ROOT, spec)
        protocol = tuning_protocol(base, 500, 50)
        trial = profile_trials("paper")[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_selection(Path(temporary_directory), spec, protocol, trial)
            resumed = completed_selection(
                path, kind="sl8", spec=spec, protocol=protocol, trial=trial
            )
            changed = completed_selection(
                path,
                kind="sl8",
                spec=spec,
                protocol=tuning_protocol(base, 500, 100),
                trial=trial,
            )
        self.assertIsNotNone(resumed)
        self.assertIsNone(changed)

    def test_test_touched_selection_is_never_silently_overwritten(self):
        spec = DATASETS[0]
        protocol = tuning_protocol(validate_fixed_protocol(REPO_ROOT, spec), 500, 50)
        trial = profile_trials("paper")[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_selection(
                Path(temporary_directory),
                spec,
                protocol,
                trial,
                test_result={"recall@10": 0.05},
            )
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                completed_selection(path, kind="sl8", spec=spec, protocol=protocol, trial=trial)

    def test_summary_refuses_different_split_fingerprints(self):
        spec = DATASETS[0]
        first = {
            "trial_name": "a",
            "best_valid_score": 0.1,
            "split_fingerprints": {"train": "one"},
        }
        second = {
            "trial_name": "b",
            "best_valid_score": 0.2,
            "split_fingerprints": {"train": "two"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "different splits"):
                write_tuning_summary(
                    Path(temporary_directory) / "summary.json",
                    spec,
                    "core",
                    {},
                    [first, second],
                    2,
                )


if __name__ == "__main__":
    unittest.main()
