"""Config contract for the paper-guided AGCF Amazon-CD pilot.

These tests intentionally do not import or instantiate AGCF.  They pin the
overlay and comparison protocol independently of model implementation details.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "baseline_config_fixed" / "RecFormer_cd.yaml"
OVERLAY_PATH = REPO_ROOT / "baseline_config_fixed" / "AGCF_cd.yaml"
PROTOCOL_NOTE = REPO_ROOT / "AGCF_PROTOCOL.md"


def _yaml_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


class AGCFConfigProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _yaml_mapping(BASE_PATH)
        cls.overlay = _yaml_mapping(OVERLAY_PATH)
        # RecBole applies external config files in order at the top level.
        cls.merged = {**cls.base, **cls.overlay}

    def test_recformer_cd_is_the_protocol_authority(self):
        expected = {
            "dataset": "Amazon_cd",
            "reproducibility": True,
            "seed": 2024,
            "learner": "adam",
            "epochs": 500,
            "stopping_step": 30,
            "user_inter_num_interval": "[5,inf)",
            "item_inter_num_interval": "[5,inf)",
            "val_interval": {"rating": "[3,inf)"},
            "metrics": ["Recall", "NDCG"],
            "topk": [5, 10, 20, 50],
            "valid_metric": "Recall@10",
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
        }
        self.assertEqual(
            {key: self.base.get(key) for key in expected},
            expected,
        )

    def test_overlay_cannot_redefine_data_split_or_full_ranking(self):
        protocol_keys = {
            "dataset",
            "reproducibility",
            "seed",
            "field_separator",
            "USER_ID_FIELD",
            "ITEM_ID_FIELD",
            "RATING_FIELD",
            "load_col",
            "user_inter_num_interval",
            "item_inter_num_interval",
            "val_interval",
            "metrics",
            "topk",
            "valid_metric",
            "eval_args",
        }
        self.assertFalse(protocol_keys.intersection(self.overlay))
        for key in protocol_keys:
            if key in self.base:
                self.assertEqual(self.merged[key], self.base[key], key)

    def test_conservative_pilot_uses_the_canonical_model_keys(self):
        expected = {
            "model": "AGCF",
            "embedding_size": 64,
            "metric_rank": 4,
            "metric_hidden_size": 64,
            "pnet_hidden_size": 64,
            "channel_rank": 4,
            "metric_epsilon": 0.001,
            "structural_delta": 0.001,
            "output_steps": 1,
            "integration_steps": 1,
            "evolution_time": 1.0,
            "potential_strength": 0.1,
            "damping": 0.01,
            "margin": 0.1,
            "dynamics_chunk_size": 4096,
            "checkpoint_dynamics": True,
            "eval_item_chunk_size": 4096,
        }
        self.assertEqual(
            {key: self.merged.get(key) for key in expected},
            expected,
        )
        # Reject the earlier draft naming scheme: duplicate aliases can drift
        # while the model silently reads only one copy.
        legacy_aliases = {
            "agcf_metric_rank",
            "agcf_metric_hidden_size",
            "agcf_metric_epsilon",
            "agcf_delta",
            "agcf_output_steps",
            "agcf_integration_steps",
            "agcf_evolution_time",
            "agcf_potential_strength",
            "agcf_damping",
            "loss_margin",
        }
        self.assertFalse(legacy_aliases.intersection(self.overlay))

    def test_paper_unknowns_are_machine_readable_and_not_defaults_claims(self):
        self.assertEqual(
            self.overlay["agcf_protocol_profile"],
            "conservative_pilot_not_paper_reproduction",
        )
        reported = set(self.overlay["agcf_paper_reported_fields"])
        unknown = set(self.overlay["agcf_paper_unknown_pilot_fields"])
        self.assertTrue({"embedding_size", "evolution_time"}.issubset(reported))
        required_unknown = {
            "metric_rank",
            "metric_epsilon",
            "structural_delta",
            "output_steps",
            "integration_steps",
            "potential_strength",
            "damping",
            "margin",
            "learning_rate",
            "weight_decay",
        }
        self.assertTrue(required_unknown.issubset(unknown))
        self.assertFalse(reported.intersection(unknown))

    def test_pilot_values_satisfy_basic_numerical_contracts(self):
        self.assertLess(self.merged["metric_rank"], self.merged["embedding_size"])
        self.assertLessEqual(self.merged["channel_rank"], self.merged["embedding_size"])
        for key in (
            "metric_hidden_size",
            "pnet_hidden_size",
            "output_steps",
            "integration_steps",
            "dynamics_chunk_size",
            "eval_item_chunk_size",
        ):
            self.assertIsInstance(self.merged[key], int, key)
            self.assertGreater(self.merged[key], 0, key)
        for key in (
            "metric_epsilon",
            "structural_delta",
            "evolution_time",
            "potential_strength",
            "damping",
            "margin",
        ):
            self.assertGreater(self.merged[key], 0.0, key)

    def test_short_protocol_note_documents_composition_and_non_equivalence(self):
        note = PROTOCOL_NOTE.read_text(encoding="utf-8")
        for required in (
            "RecFormer_cd.yaml",
            "AGCF_cd.yaml",
            "--validation-only",
            "113,303",
            "not a claim",
        ):
            self.assertIn(required, note)


if __name__ == "__main__":
    unittest.main()
