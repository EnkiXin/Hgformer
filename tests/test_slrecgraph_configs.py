"""Contracts for the composable Amazon-CD geometry-only configurations."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from recbole_gnn.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "baseline_config_fixed"
BASE = CONFIG_ROOT / "SLRecGraph_cd.yaml"
GEOMETRY = CONFIG_ROOT / "SLRecGraph_geometry_sl4.yaml"
SAMPLED = CONFIG_ROOT / "SLRecGraph_tune_sampled.yaml"
FULL = CONFIG_ROOT / "SLRecGraph_eval_full.yaml"


def _resolved_config(*overlays: Path) -> Config:
    # RecBole also parses process-wide CLI arguments.  Isolate the contract
    # test from pytest/unittest flags so only these files influence it.
    with patch.object(sys, "argv", ["test_slrecgraph_configs"]):
        return Config(
            model="SLRecGraph",
            dataset="Amazon_cd",
            config_file_list=[str(BASE), *(str(path) for path in overlays)],
            config_dict={"use_gpu": False},
        )


class GeometryOnlyConfigTest(unittest.TestCase):
    def test_geometry_overlay_disables_graph_and_selects_one_sided_sl4(self):
        config = _resolved_config(GEOMETRY)

        self.assertEqual(config["matrix_dim"], 4)
        self.assertEqual(config["embedding_size"], 16)
        self.assertEqual(config["num_factors"], 1)
        self.assertEqual(config["n_layers"], 0)
        self.assertFalse(config["symmetric_distance"])
        self.assertEqual(config["log_jitter"], 0.0)
        self.assertAlmostEqual(config["init_std"], 0.012)
        self.assertAlmostEqual(config["coord_clip"], 0.75)
        self.assertEqual(config["reg_weight"], 0.0)

    def test_sampled_tuning_preserves_split_and_draws_100_negatives(self):
        config = _resolved_config(GEOMETRY, SAMPLED)

        self.assertEqual(config["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertEqual(config["eval_args"]["group_by"], "user")
        self.assertEqual(config["eval_args"]["order"], "RO")
        self.assertEqual(config["eval_args"]["mode"], "uni100")
        self.assertEqual(
            config["eval_neg_sample_args"],
            {"strategy": "by", "by": 100, "distribution": "uniform"},
        )
        self.assertEqual(config["epochs"], 500)
        self.assertEqual(config["eval_step"], 5)
        self.assertTrue(config["fixed_sampled_validation"])
        self.assertEqual(config["fixed_sampled_validation_seed"], 2024)
        self.assertEqual(config["metric_decimal_place"], 6)

    def test_final_overlay_restores_exact_full_ranking_protocol(self):
        config = _resolved_config(GEOMETRY, FULL)

        self.assertEqual(config["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertEqual(config["eval_args"]["group_by"], "user")
        self.assertEqual(config["eval_args"]["order"], "RO")
        self.assertEqual(config["eval_args"]["mode"], "full")
        self.assertIsNone(config["fixed_sampled_validation"])
        self.assertEqual(
            config["eval_neg_sample_args"],
            {"strategy": "full", "distribution": "uniform"},
        )
        self.assertEqual(config["eval_batch_size"], 40960000)
        self.assertEqual(config["eval_user_chunk_size"], 64)
        self.assertEqual(config["eval_item_chunk_size"], 4096)


if __name__ == "__main__":
    unittest.main()
