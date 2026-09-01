"""Contracts for the one-GPU SL8 row-mean + LieBN first screen."""

from __future__ import annotations

import re
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
        self.assertIn("$Layers = @(2, 4, 6, 8)", script)
        self.assertIn("$Layers = @(2)", script)
        self.assertIn("$RunEpochs = 1", script)
        self.assertIn('$env:CUDA_VISIBLE_DEVICES = [string]$Gpu', script)
        self.assertIn("Split-Path $RepoRoot -Parent", script)
        self.assertIn('"--validation-only"', script)
        self.assertIn('"--n_layers=$Layer"', script)
        self.assertIn('"--gpu_id=$Gpu"', script)
        self.assertNotIn('"--gpu_id=0"', script)
        self.assertIn('"--full_sort_user_batch_size=$EvalUsers"', script)
        self.assertIn('"--eval_user_chunk_size=$EvalUsers"', script)
        self.assertIn('"--eval_item_chunk_size=$EvalItems"', script)
        self.assertIn('"--sl_score_mode=group_log"', script)
        self.assertIn('"--eval_prefilter=none"', script)
        self.assertIn("Test-CompletedValidationResult", script)
        self.assertNotIn("Start-Job", script)
        self.assertNotIn("ForEach-Object -Parallel", script)

        # One foreach loop launches one external Python process and waits for
        # its exit code before advancing to the next layer.
        self.assertEqual(len(re.findall(r"foreach \(\$Layer in \$Layers\)", script)), 1)
        self.assertRegex(
            script,
            r"& \$Python @Arguments\s+if \(\$LASTEXITCODE -ne 0\)",
        )

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
