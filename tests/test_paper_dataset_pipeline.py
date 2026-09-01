"""Contracts for the six-paper-dataset, physical-GPU-7 pipeline."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from slrec_experiments.run_multidataset_lhgcn import main as legacy_lhgcn_main
from slrec_experiments.run_multidataset_sl8 import main as legacy_sl8_main
from slrec_experiments.run_paper_dataset_pipeline import (
    AMAZON_BOOK_PAPER_COUNTS,
    BATCH_SIZES,
    CURRENT_TUNING_STAGES,
    EXPANDED_TUNING_STAGES,
    LHGCNParameters,
    LAYERS,
    PAPER_DATASETS,
    PHYSICAL_GPU,
    SL_BATCH_SIZES,
    SL8Parameters,
    _candidate,
    _normalise_gpu,
    annotate_selection,
    completed_selection,
    control_jobs,
    default_lock_path,
    exclusive_gpu_lock,
    final_test_command,
    full_cartesian_sl8_jobs,
    full_cartesian_sl8_parameters,
    grid_jobs,
    grid_parameters,
    lhgcn_capacity_job,
    load_grid_winner,
    lhgcn_full_cartesian_jobs,
    lhgcn_full_cartesian_parameters,
    main,
    protocol_config_files,
    practical_joint_jobs,
    practical_joint_parameters,
    selection_command,
    select_paper_datasets,
    sl_model_config_files,
    staged_new_trial_count,
    tuning_jobs,
    validate_parameter_activity,
    validate_pipeline_protocol,
    write_grid_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(output_root: Path = Path("/tmp/paper-pipeline-tests"), **updates):
    defaults = {
        "repo": REPO_ROOT,
        "data_root": REPO_ROOT / "dataset",
        "output_root": output_root,
        "python": "python3",
        "gpu_id": PHYSICAL_GPU,
        "tuning_profile": "expanded",
        "sl8_search": "practical",
        "lhgcn_search": "matched",
        "lock_file": default_lock_path(),
    }
    defaults.update(updates)
    return argparse.Namespace(**defaults)


def _raw_selection(job, spec, checkpoint: Path, score: float = 0.1, test_result=None):
    return {
        "model": job.model,
        "dataset": spec.dataset,
        "seed": 2024,
        "best_valid_score": score,
        "best_valid_result": {"recall@10": score, "ndcg@10": score / 2},
        "test_result": test_result,
        "checkpoint_file": str(checkpoint),
        "split_fingerprints": {
            "train": {"interactions": 80, "sha256": "train"},
            "valid": {"interactions": 10, "sha256": "valid"},
            "test": {"interactions": 10, "sha256": "test"},
        },
    }


class RegistryAndPaperProtocolTest(unittest.TestCase):
    def test_default_order_is_only_the_six_paper_datasets(self):
        self.assertEqual(
            [spec.slug for spec in PAPER_DATASETS],
            [
                "amazon-cd",
                "amazon-movies",
                "amazon-book",
                "douban-book",
                "douban-movie",
                "douban-music",
            ],
        )
        self.assertEqual(select_paper_datasets(["all"]), PAPER_DATASETS)
        with self.assertRaisesRegex(ValueError, "non-paper"):
            select_paper_datasets(["amazon-toy"])

    def test_every_dataset_keeps_seed_split_metrics_and_full_ranking(self):
        for spec in PAPER_DATASETS:
            protocol = validate_pipeline_protocol(REPO_ROOT, spec)
            authority = protocol["protocol_authority"]
            self.assertEqual(authority["seed"], 2024)
            self.assertEqual(authority["validation"]["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
            self.assertEqual(authority["validation"]["eval_args"]["mode"], "full")
            self.assertEqual(authority["validation"]["metrics"], ["Recall", "NDCG"])
            self.assertEqual(authority["validation"]["topk"], [5, 10, 20, 50])

    def test_amazon_book_always_applies_exact_paper_eight_core(self):
        spec = next(spec for spec in PAPER_DATASETS if spec.slug == "amazon-book")
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        authority = protocol["protocol_authority"]
        self.assertEqual(authority["filters"]["users"], "[8,inf)")
        self.assertEqual(authority["filters"]["items"], "[8,inf)")
        self.assertEqual(
            authority["paper_protocol_overlay"]["expected_filtered_counts"],
            AMAZON_BOOK_PAPER_COUNTS,
        )
        self.assertEqual(
            [path.name for path in protocol_config_files(REPO_ROOT, spec)],
            ["RecFormer_book.yaml", "PaperProtocol_amazon_book_8core.yaml"],
        )

    def test_douban_movie_accepts_actual_count_and_records_paper_typo(self):
        spec = next(spec for spec in PAPER_DATASETS if spec.slug == "douban-movie")
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        note = protocol["published_count_note"]
        self.assertEqual(note["accepted_interactions"], 2_552_305)
        self.assertEqual(note["paper_table_interactions"], 2_553_305)

    def test_lightgcn_uses_historical_optimisation_but_not_historical_split_file(self):
        protocols = {
            spec.slug: validate_pipeline_protocol(REPO_ROOT, spec)
            for spec in PAPER_DATASETS
        }
        cd = protocols["amazon-cd"]["lightgcn"]["parameters"]
        movies = protocols["amazon-movies"]["lightgcn"]["parameters"]
        book = protocols["amazon-book"]["lightgcn"]["parameters"]
        self.assertEqual((cd["n_layers"], cd["train_batch_size"]), (2, 4096))
        self.assertEqual((movies["n_layers"], movies["reg_weight"]), (2, 0.01))
        self.assertEqual((book["n_layers"], book["train_batch_size"]), (1, 131072))

    def test_lhgcn_is_matched_to_each_recformer_model_training_subset(self):
        cd = next(spec for spec in PAPER_DATASETS if spec.slug == "amazon-cd")
        matched = validate_pipeline_protocol(REPO_ROOT, cd)["lhgcn"][
            "parameters_matched_to_recformer"
        ]
        self.assertEqual(matched["gcn_layers"], 7)
        self.assertEqual(matched["curve"], 0.1)
        self.assertEqual(matched["margin"], 0.3)
        self.assertEqual(matched["train_batch_size"], 131072)


class SearchDesignTest(unittest.TestCase):
    def test_curve_is_proven_dead_for_sl8_but_active_for_lhgcn(self):
        activity = validate_parameter_activity(REPO_ROOT)
        self.assertEqual(activity["sl8lhgcn"]["curve"], "dead-not-read")
        self.assertIn("active", activity["lhgcn"]["curve"])

    def test_layer_batch_grid_is_exactly_five_by_three(self):
        parameters = grid_parameters()
        self.assertEqual(len(parameters), 15)
        self.assertEqual(
            {(p.gcn_layers, p.train_batch_size) for p in parameters},
            {(layer, batch) for layer in LAYERS for batch in BATCH_SIZES},
        )
        self.assertTrue(all(p.learning_rate == 5e-4 for p in parameters))

    def test_current_and_expanded_profiles_are_unique_and_expanded_is_larger(self):
        self.assertEqual(staged_new_trial_count("current"), 21)
        self.assertEqual(staged_new_trial_count("expanded"), 39)
        for stages in (CURRENT_TUNING_STAGES, EXPANDED_TUNING_STAGES):
            self.assertEqual(len({stage.key for stage in stages}), len(stages))
            for stage in stages:
                normalised = [json.dumps(value, sort_keys=True) for value in stage.values]
                self.assertEqual(len(normalised), len(set(normalised)), stage.key)

    def test_staged_jobs_change_one_axis_and_do_not_retrain_anchor(self):
        spec = PAPER_DATASETS[0]
        args = _args()
        anchor = SL8Parameters(gcn_layers=4, train_batch_size=65536)
        stage = CURRENT_TUNING_STAGES[0]
        jobs = tuning_jobs(args, spec, 1, stage, anchor)
        self.assertEqual(len(jobs), 3)
        self.assertNotIn(anchor.learning_rate, {job.parameters["learning_rate"] for job in jobs})
        for job in jobs:
            changed = {
                key
                for key, value in job.parameters.items()
                if value != getattr(anchor, key)
            }
            self.assertEqual(changed, {"learning_rate"})

    def test_practical_block_is_joint_geometry_by_margin_after_layer_batch(self):
        anchor = SL8Parameters(gcn_layers=6, train_batch_size=32768)
        candidates = practical_joint_parameters(anchor)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(
            {(candidate.schatten_p, candidate.loss_margin) for candidate in candidates},
            {(p, margin) for p in (1, 2, 4, 8, "inf") for margin in (0.05, 0.1, 0.2, 0.3)},
        )
        self.assertTrue(all(candidate.gcn_layers == 6 for candidate in candidates))
        self.assertTrue(all(candidate.train_batch_size == 32768 for candidate in candidates))
        self.assertEqual(len(practical_joint_jobs(_args(), PAPER_DATASETS[0], anchor)), 19)

    def test_full_primary_sl8_cartesian_has_300_effective_trials_and_no_curve(self):
        parameters = full_cartesian_sl8_parameters()
        self.assertEqual(len(parameters), 300)
        self.assertTrue(all("curve" not in parameter.__dict__ for parameter in parameters))
        jobs = full_cartesian_sl8_jobs(_args(), PAPER_DATASETS[0])
        self.assertEqual(len(jobs), 300)
        self.assertTrue(
            all(not any(token.startswith("--curve=") for token in job.extra_args) for job in jobs)
        )

    def test_sl16_has_an_independent_300_trial_budget_and_conservative_batches(self):
        parameters = full_cartesian_sl8_parameters(16)
        self.assertEqual(len(parameters), 300)
        self.assertEqual({parameter.matrix_dim for parameter in parameters}, {16})
        self.assertEqual(
            {parameter.train_batch_size for parameter in parameters},
            set(SL_BATCH_SIZES[16]),
        )
        self.assertEqual(len(grid_parameters(16)), 15)

    def test_lhgcn_full_cartesian_has_375_trials_and_passes_active_curve(self):
        spec = PAPER_DATASETS[0]
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        anchor = protocol["lhgcn"]["parameters_matched_to_recformer"]
        parameters = lhgcn_full_cartesian_parameters(LHGCNParameters(**anchor))
        self.assertEqual(len(parameters), 375)
        jobs = lhgcn_full_cartesian_jobs(_args(), spec, protocol)
        self.assertEqual(len(jobs), 375)
        self.assertTrue(all(any(token.startswith("--curve=") for token in job.extra_args) for job in jobs))


class CommandAndGpuSafetyTest(unittest.TestCase):
    def test_only_physical_gpu_seven_is_accepted(self):
        self.assertEqual(_normalise_gpu("07"), "7")
        for invalid in ("0", "6", "0,7", "cuda:7", "GPU-id", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _normalise_gpu(invalid)

    def test_every_training_command_is_validation_only_and_passes_physical_seven(self):
        args = _args()
        for spec in PAPER_DATASETS:
            protocol = validate_pipeline_protocol(REPO_ROOT, spec)
            jobs = [*control_jobs(args, spec, protocol), *grid_jobs(args, spec)]
            for job in jobs:
                command = selection_command(args, spec, job)
                self.assertIn("--validation-only", command)
                self.assertIn("--gpu_id=7", command)
                self.assertNotIn("--gpu_id=0", command)
                self.assertIn("--use_gpu=true", command)
                self.assertFalse(any("uni100" in token.lower() for token in command))

    def test_book_commands_apply_correction_before_every_model_overlay(self):
        args = _args()
        spec = next(spec for spec in PAPER_DATASETS if spec.slug == "amazon-book")
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        for job in [*control_jobs(args, spec, protocol), *grid_jobs(args, spec)]:
            command = selection_command(args, spec, job)
            names = [
                Path(path).name
                for path in command[command.index("--config-files") + 1].split()
            ]
            self.assertEqual(names[:2], ["RecFormer_book.yaml", "PaperProtocol_amazon_book_8core.yaml"])

    def test_sl16_uses_required_overlay_order_and_capacity_matched_control(self):
        args = _args()
        spec = next(spec for spec in PAPER_DATASETS if spec.slug == "amazon-book")
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        names = [path.name for path in sl_model_config_files(REPO_ROOT, spec, 16)]
        self.assertEqual(
            names,
            [
                "RecFormer_book.yaml",
                "PaperProtocol_amazon_book_8core.yaml",
                "SL8LHGCN_reproduction.yaml",
                "SL16LHGCN_reproduction.yaml",
            ],
        )
        sl16_job = grid_jobs(args, spec, 16)[0]
        command = selection_command(args, spec, sl16_job)
        self.assertEqual(sl16_job.model, "SL16LHGCN")
        for expected in (
            "--embedding_size=256",
            "--matrix_dim=16",
            "--sl_distance_check_samples=4",
            "--gpu_id=7",
            "--validation-only",
        ):
            self.assertIn(expected, command)

        capacity = lhgcn_capacity_job(args, spec, protocol)
        self.assertEqual(capacity.model, "LHGCN")
        self.assertIn("--embedding_size=256", capacity.extra_args)
        self.assertIn("--train_batch_size=4096", capacity.extra_args)

    def test_recformer_is_fixed_except_for_execution_and_protocol_correction(self):
        args = _args()
        spec = PAPER_DATASETS[0]
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        job = control_jobs(args, spec, protocol)[2]
        command = selection_command(args, spec, job)
        self.assertEqual(job.model, "RecFormer")
        self.assertFalse(any(token.startswith("--learning_rate=") for token in command))
        self.assertFalse(any(token.startswith("--gcn_layers=") for token in command))
        self.assertFalse(any(token.startswith("--alpha=") for token in command))

    def test_final_test_loads_checkpoint_without_training_or_validation(self):
        command = final_test_command(
            _args(), Path("/tmp/selection.json"), Path("/tmp/model.pth"), Path("/tmp/test.json")
        )
        self.assertIn("evaluate_recbole_gnn_checkpoint.py", " ".join(command))
        self.assertIn("--skip-valid", command)
        self.assertEqual(
            command[command.index("--full-sort-user-batch-size") + 1], "64"
        )
        self.assertNotIn("run_recbole_gnn.py", " ".join(command))

    def test_gpu_lock_is_shared_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = Path(temporary_directory) / "gpu7.lock"
            with exclusive_gpu_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "already reserved"):
                    with exclusive_gpu_lock(lock):
                        pass


class ResumeAndSelectionIsolationTest(unittest.TestCase):
    def test_annotated_selection_resumes_only_exact_job_contract(self):
        spec = PAPER_DATASETS[0]
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = _args(root)
            job = control_jobs(args, spec, protocol)[0]
            checkpoint = root / "checkpoint.pth"
            checkpoint.touch()
            job.result_path.parent.mkdir(parents=True)
            job.result_path.write_text(
                json.dumps(_raw_selection(job, spec, checkpoint)), encoding="utf-8"
            )
            annotate_selection(
                job.result_path, args=args, spec=spec, protocol=protocol, job=job
            )
            resumed = completed_selection(
                job.result_path, args=args, spec=spec, protocol=protocol, job=job
            )
            payload = json.loads(job.result_path.read_text(encoding="utf-8"))
            payload["paper_dataset_pipeline"]["parameters"]["n_layers"] = 99
            job.result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume contract"):
                completed_selection(
                    job.result_path, args=args, spec=spec, protocol=protocol, job=job
                )
        self.assertIsNotNone(resumed)

    def test_test_touched_selection_is_fatal_not_silently_retrained(self):
        spec = PAPER_DATASETS[0]
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = _args(root)
            job = control_jobs(args, spec, protocol)[0]
            checkpoint = root / "checkpoint.pth"
            checkpoint.touch()
            job.result_path.parent.mkdir(parents=True)
            job.result_path.write_text(
                json.dumps(
                    _raw_selection(job, spec, checkpoint, test_result={"recall@10": 1.0})
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "held-out test"):
                annotate_selection(
                    job.result_path, args=args, spec=spec, protocol=protocol, job=job
                )

    def test_complete_grid_winner_is_selected_by_validation_only(self):
        spec = PAPER_DATASETS[0]
        protocol = validate_pipeline_protocol(REPO_ROOT, spec)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = _args(root)
            candidates = []
            jobs = grid_jobs(args, spec)
            for index, job in enumerate(jobs):
                checkpoint = root / "checkpoints" / f"{index}.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                job.result_path.parent.mkdir(parents=True, exist_ok=True)
                score = 0.01 + index / 1000
                job.result_path.write_text(
                    json.dumps(_raw_selection(job, spec, checkpoint, score=score)),
                    encoding="utf-8",
                )
                result = annotate_selection(
                    job.result_path, args=args, spec=spec, protocol=protocol, job=job
                )
                candidates.append(_candidate(job, result))
            summary = write_grid_summary(args, spec, protocol, candidates)
            winner = load_grid_winner(args, spec, protocol)
        self.assertEqual(summary["state"], "complete")
        self.assertEqual(winner["label"], jobs[-1].label)
        self.assertFalse(winner["test_evaluated"])


class PlanningAndLegacyGuardTest(unittest.TestCase):
    def test_dry_run_writes_nothing_and_plans_cd_first_with_expanded_search(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "never-created"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(output_root),
                        "--datasets",
                        "all",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(output_root.exists())
        self.assertEqual(plan["dataset_order"][0], "amazon-cd")
        self.assertEqual(plan["dataset_order"], [spec.slug for spec in PAPER_DATASETS])
        self.assertEqual(len(plan["datasets"][0]["selection_jobs"]), 18)
        tuning = plan["datasets"][0]["tuning"]
        self.assertEqual(tuning["total_maximum_sl8_jobs"], 34)
        self.assertEqual(tuning["block_2"]["maximum_new_jobs"], 19)
        self.assertEqual(plan["parameter_activity"]["sl8lhgcn"]["curve"], "dead-not-read")
        self.assertFalse(plan["test_evaluated"])

    def test_optional_factorial_dry_runs_report_counts_without_huge_job_lists(self):
        cases = (
            (["--phase", "tune", "--sl8-search", "full-cartesian"], "tuning", 300),
            (["--phase", "lhgcn-grid", "--lhgcn-search", "full-cartesian"], "lhgcn_factorial", 375),
        )
        for flags, key, count in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary_directory:
                output = io.StringIO()
                with redirect_stdout(output):
                    main(
                        [
                            "--repo",
                            str(REPO_ROOT),
                            "--output-root",
                            str(Path(temporary_directory) / "out"),
                            "--datasets",
                            "amazon-cd",
                            *flags,
                            "--dry-run",
                            "--skip-data-audit",
                        ]
                    )
                plan = json.loads(output.getvalue())
                block = plan["datasets"][0][key]
                self.assertEqual(block["job_count"], count)
                self.assertTrue(block["all_jobs_omitted_from_plan"])

    def test_dual_dimension_plan_is_separate_and_accounts_for_capacity_control(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(Path(temporary_directory) / "out"),
                        "--datasets",
                        "amazon-cd",
                        "--sl-dims",
                        "8",
                        "16",
                        "--sl-search",
                        "practical",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(output.getvalue())
        dataset = plan["datasets"][0]
        self.assertEqual(plan["sl_dims"], [8, 16])
        self.assertEqual(len(dataset["selection_jobs"]), 34)
        self.assertEqual(set(dataset["sl_searches"]), {"8", "16"})
        self.assertEqual(dataset["sl_searches"]["8"]["total_maximum_jobs"], 34)
        self.assertEqual(dataset["sl_searches"]["16"]["total_maximum_jobs"], 34)
        self.assertEqual(
            dataset["sl_searches"]["16"]["raw_entity_parameter_ratio_vs_sl8"],
            4.0,
        )
        self.assertEqual(
            dataset["sl_searches"]["16"]["dense_cubic_compute_proxy_vs_sl8"],
            8.0,
        )

    def test_sl_all_phase_contains_only_the_two_requested_new_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(Path(temporary_directory) / "out"),
                        "--datasets",
                        "amazon-cd",
                        "--phase",
                        "sl-all",
                        "--sl-dims",
                        "8",
                        "16",
                        "--sl-search",
                        "practical",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(output.getvalue())
        jobs = plan["datasets"][0]["selection_jobs"]
        self.assertEqual(len(jobs), 30)
        self.assertEqual({job["model"] for job in jobs}, {"SL8LHGCN", "SL16LHGCN"})
        self.assertFalse(
            {"LightGCN", "LHGCN", "RecFormer"}
            & {job["model"] for job in jobs}
        )

    def test_sl_final_test_scope_does_not_require_control_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--repo",
                        str(REPO_ROOT),
                        "--output-root",
                        str(Path(temporary_directory) / "out"),
                        "--datasets",
                        "amazon-cd",
                        "--phase",
                        "sl-final-test",
                        "--sl-dims",
                        "8",
                        "16",
                        "--dry-run",
                        "--skip-data-audit",
                    ]
                )
            plan = json.loads(output.getvalue())
        dataset = plan["datasets"][0]
        self.assertEqual(dataset["selection_jobs"], [])
        self.assertEqual(dataset["final_test"]["scope"], "special-linear-only")

    def test_skip_data_audit_is_forbidden_for_real_execution(self):
        with self.assertRaisesRegex(ValueError, "permitted only"):
            main(
                [
                    "--repo",
                    str(REPO_ROOT),
                    "--output-root",
                    "/tmp/never-paper-pipeline",
                    "--datasets",
                    "amazon-cd",
                    "--skip-data-audit",
                ]
            )

    def test_legacy_multidataset_runners_refuse_book_five_core(self):
        common = [
            "--repo",
            str(REPO_ROOT),
            "--output-root",
            "/tmp/never-legacy-book",
            "--datasets",
            "amazon-book",
            "--dry-run",
            "--skip-data-audit",
        ]
        with self.assertRaisesRegex(RuntimeError, "8-core"):
            legacy_sl8_main(common)
        with self.assertRaisesRegex(RuntimeError, "8-core"):
            legacy_lhgcn_main(common)


if __name__ == "__main__":
    unittest.main()
