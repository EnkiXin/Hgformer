"""Contracts for the one-GPU SL8 row-mean + LieBN first screen."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from slrec_experiments.dataset_registry import DATASETS


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = (
    REPO_ROOT
    / "baseline_config_fixed"
    / "SL8LHGCN_liebn_rowmean_4070ti.yaml"
)
SPACE = (
    REPO_ROOT
    / "baseline_config_flexible"
    / "SL8LHGCN"
    / "SL8LHGCN_liebn_layers_50.test"
)
RUNNER = REPO_ROOT / "slrec_experiments" / "run_sl8_liebn_layers_cd.ps1"
GRID_RUNNER = (
    REPO_ROOT / "slrec_experiments" / "run_sl8_liebn_layers_batches_cd.ps1"
)
MANIFEST_BUILDER = REPO_ROOT / "slrec_experiments" / "build_sl8_stage_a_manifest.py"


class SL8LieBNLayerScreenTest(unittest.TestCase):
    def test_overlay_is_exact_formal_rowmean_liebn_screen(self):
        base = yaml.safe_load(
            (REPO_ROOT / "baseline_config_fixed" / "SL8LHGCN_cd.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
        expected = {
            "sl_gcn_mode": "karcher1",
            "sl_karcher_correction": False,
            "sl_layer_norm": "liebn",
            "liebn_mean": "karcher1",
            "liebn_dispersion": "mean_norm",
            "liebn_learnable_bias": False,
            "sl_score_mode": "group_log",
            "eval_prefilter": "none",
            "schatten_p": 2,
            "epochs": 50,
            "eval_step": 50,
            "stopping_step": 1000,
            "train_batch_size": 16384,
            "full_sort_user_batch_size": 64,
            "eval_user_chunk_size": 64,
            "eval_item_chunk_size": 1024,
        }
        for key, value in expected.items():
            self.assertEqual(config[key], value, key)
        self.assertEqual(base["eval_args"]["mode"], "full")
        self.assertEqual(base["pairwise_loss"], "lhgcn_hinge_squared_sum")
        self.assertEqual(base["loss_margin"], 0.1)

    def test_space_is_one_dimension_with_exactly_four_layers(self):
        active = [
            line.strip()
            for line in SPACE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(active, ["gcn_layers choice [2,4,6,8]"])
        self.assertFalse(any(line.startswith("n_layers ") for line in active))

    def test_powershell_runner_is_serial_single_gpu_and_validation_only(self):
        script = RUNNER.read_text(encoding="utf-8")
        self.assertIn("[int[]]$LayerGrid = @(2, 4, 6, 8)", script)
        self.assertIn("$Layers = @($LayerGrid)", script)
        self.assertIn("$Layers = @(2)", script)
        self.assertIn("$RunEpochs = 1", script)
        self.assertIn('$env:CUDA_VISIBLE_DEVICES = [string]$Gpu', script)
        self.assertIn("Split-Path $RepoRoot -Parent", script)
        self.assertIn('"--validation-only"', script)
        self.assertIn('"--epochs=$RunEpochs"', script)
        self.assertIn("$RunEpochs = 500", script)
        self.assertIn("Formal screen requires epochs=500", script)
        self.assertIn('"--n_layers=$Layer"', script)
        self.assertIn('"--gpu_id=$Gpu"', script)
        self.assertNotIn('"--gpu_id=0"', script)
        self.assertIn('"--full_sort_user_batch_size=$EvalUsers"', script)
        self.assertIn('"--eval_user_chunk_size=$EvalUsers"', script)
        self.assertIn('"--eval_item_chunk_size=$EvalItems"', script)
        self.assertIn('"--sl_score_mode=group_log"', script)
        self.assertIn('"--stopping_step=$StoppingStep"', script)
        self.assertIn("ExpectedEvalStep", script)
        self.assertIn("ExpectedStoppingStep", script)
        self.assertIn('$PrefilterMode = if ($AcceleratedPrefilter)', script)
        self.assertIn('"--eval_prefilter=$PrefilterMode"', script)
        self.assertIn("Test-CompletedValidationResult", script)
        self.assertNotIn("Start-Job", script)

    def test_accelerated_prefilter_is_explicit_and_mask_aware(self):
        script = RUNNER.read_text(encoding="utf-8")
        self.assertIn("[switch]$AcceleratedPrefilter", script)
        self.assertIn("[int]$PrefilterCandidates = 4096", script)
        self.assertIn("_PFfrobeniusC$PrefilterCandidates", script)
        root = RUNNER.parents[1]
        trainer = (root / "recbole/trainer/trainer.py").read_text(encoding="utf-8")
        app = (root / "recbole/trainer/app_trainer.py").read_text(encoding="utf-8")
        for source in (trainer, app):
            self.assertIn("full_sort_predict_with_exclusions", source)
            self.assertIn("eval_prefilter", source)
        self.assertNotIn("ForEach-Object -Parallel", script)

        # One foreach loop launches one external Python process and waits for
        # its exit code before advancing to the next layer.
        self.assertEqual(len(re.findall(r"foreach \(\$Layer in \$Layers\)", script)), 1)
        self.assertRegex(
            script,
            r"& \$Python @Arguments\s+if \(\$LASTEXITCODE -ne 0\)",
        )

    def test_batch_grid_is_recoverable_serial_and_ordered(self):
        script = GRID_RUNNER.read_text(encoding="utf-8")
        self.assertIn("build_sl8_stage_a_manifest.py", script)
        self.assertIn("cell_count -ne 61", script)
        self.assertIn("foreach ($Cell in $Manifest.cells)", script)
        self.assertIn("run_sl8_liebn_layers_cd.ps1", script)
        self.assertIn("-Epochs 500 -EvalStep 10 -StoppingStep 2", script)
        self.assertIn("-LayerGrid @([int]$Cell.layer)", script)
        self.assertIn("-AcceleratedPrefilter", script)
        self.assertIn("-PrefilterCandidates 4096", script)
        self.assertIn("catch {", script)
        self.assertIn("CELL_FAILED", script)
        self.assertIn("Sort-Object", script)
        self.assertNotIn("Start-Job", script)
        self.assertNotIn("ForEach-Object -Parallel", script)

    def test_stage_a_manifest_design_is_deterministic_and_balanced(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage_a_manifest", MANIFEST_BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cells, removed = module.build_cells()
        self.assertEqual(len(cells), 61)
        self.assertEqual(len(removed), 1)
        keys = [(c["layer"], c["batch"], c["learning_rate"], c["loss_margin"], c["coord_clip"]) for c in cells]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(cells[0]["layer"], 0)
        self.assertEqual(cells[0]["batch"], 65536)
        self.assertEqual((cells[0]["learning_rate"], cells[0]["loss_margin"], cells[0]["coord_clip"]), (.005, .1, .75))
        hyper = [c for c in cells if c["source"] == "hparam"]
        self.assertEqual(len(hyper), 41)
        for lr in module.LRS:
            self.assertTrue(set(module.MARGINS).issubset({c["loss_margin"] for c in hyper if c["learning_rate"] == lr}))
        for label, _ in module.CLIPS:
            self.assertTrue(set(module.MARGINS).issubset({c["loss_margin"] for c in hyper if c["coord_clip_label"] == label}))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            module.main(str(manifest))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol"]["epochs"], 500)
        self.assertEqual(payload["protocol"]["eval_step"], 10)
        self.assertEqual(payload["protocol"]["stopping_step"], 2)
        self.assertEqual(payload["cell_count"], 61)

    def test_readme_names_every_registered_direct_source(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for record in DATASETS:
            self.assertIn(record.download_url, readme, record.slug)
        for label in (
            "Amazon CD",
            "Amazon Movies",
            "Amazon Book",
            "Douban Book",
            "Douban Movie",
            "Douban Music",
            "Amazon Toy",
            "MovieLens-100K",
        ):
            self.assertIn(label, readme)
        self.assertIn("not Amazon 2018", readme)
        self.assertIn("do not substitute the much smaller CoPD files", readme)
        self.assertIn("DATASETS.md", readme)


if __name__ == "__main__":
    unittest.main()
