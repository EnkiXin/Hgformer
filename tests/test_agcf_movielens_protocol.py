"""Static contract for the exact AGCF MovieLens-1M protocol layer."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT / "baseline_config_fixed" / "AGCF_movielens_protocol.yaml"
)
AGCF_MODEL = REPO_ROOT / "baseline_config_fixed" / "AGCF_cd.yaml"
SL8_MODEL = REPO_ROOT / "baseline_config_fixed" / "AGCFSL8Coord_cd.yaml"


def _read(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


class AGCFMovieLensProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _read(PROTOCOL)
        cls.agcf = _read(AGCF_MODEL)
        cls.sl8 = _read(SL8_MODEL)

    def test_table_two_filter_and_counts_are_pinned(self):
        self.assertEqual(self.protocol["dataset"], "AGCF_MovieLens")
        self.assertEqual(self.protocol["val_interval"], {"rating": "[3,inf)"})
        self.assertEqual(self.protocol["user_inter_num_interval"], "[5,inf)")
        self.assertEqual(self.protocol["item_inter_num_interval"], "[5,inf)")
        self.assertEqual(
            self.protocol["agcf_table2_expected_runtime_counts"],
            {"users": 6039, "items": 3308, "interactions": 835789},
        )

    def test_paper_evaluation_contract(self):
        self.assertEqual(
            self.protocol["eval_args"],
            {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
        )
        self.assertEqual(self.protocol["metrics"], ["Recall", "NDCG"])
        self.assertEqual(self.protocol["topk"], [10, 20])
        self.assertEqual(self.protocol["valid_metric"], "Recall@10")
        self.assertEqual(self.protocol["epochs"], 500)
        self.assertEqual(self.protocol["stopping_step"], 30)
        self.assertEqual(self.protocol["embedding_size"], 64)

    def test_unreported_choices_are_not_presented_as_paper_values(self):
        self.assertEqual(
            self.protocol["agcf_dataset_protocol_profile"],
            "paper_table_exact_filter_seed_unknown",
        )
        unknown = set(self.protocol["agcf_paper_unknown_protocol_fields"])
        self.assertTrue(
            {"seed", "train_batch_size", "neg_sampling", "eval_step"}.issubset(
                unknown
            )
        )

    def test_model_overlays_do_not_override_the_protocol(self):
        protected = {
            "dataset",
            "seed",
            "val_interval",
            "user_inter_num_interval",
            "item_inter_num_interval",
            "eval_args",
            "metrics",
            "topk",
            "valid_metric",
        }
        self.assertFalse(protected.intersection(self.agcf))
        self.assertFalse(protected.intersection(self.sl8))


if __name__ == "__main__":
    unittest.main()
