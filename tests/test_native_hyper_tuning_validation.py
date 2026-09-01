import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_test import CachedObjective
from recbole.trainer.hyper_tuning import HyperTuning
from recbole_gnn import quick_start


class NativeHyperTuningValidationTest(unittest.TestCase):
    def test_objective_can_skip_held_out_test(self):
        config = {
            "enable_sparse": None,
            "seed": 2024,
            "reproducibility": True,
            "model": "SL8LHGCN",
            "MODEL_TYPE": "GENERAL",
            "device": "cpu",
            "valid_metric_bigger": True,
        }
        train_data = SimpleNamespace(dataset=object())
        valid_data = object()
        test_data = object()
        model = Mock()
        model.to.return_value = model
        trainer = Mock()
        trainer.fit.return_value = (0.123, {"recall@10": 0.123})

        with (
            patch.object(quick_start, "Config", return_value=config),
            patch.object(quick_start, "init_seed"),
            patch.object(quick_start, "create_dataset", return_value=object()),
            patch.object(
                quick_start,
                "data_preparation",
                return_value=(train_data, valid_data, test_data),
            ),
            patch.object(
                quick_start,
                "get_model",
                return_value=lambda _config, _dataset: model,
            ),
            patch.object(
                quick_start,
                "get_trainer",
                return_value=lambda _config, _model: trainer,
            ),
        ):
            result = quick_start.objective_function(evaluate_test=False)

        self.assertEqual(result["best_valid_score"], 0.123)
        self.assertIsNone(result["test_result"])
        trainer.evaluate.assert_not_called()

    def test_export_accepts_validation_only_result(self):
        tuner = HyperTuning.__new__(HyperTuning)
        tuner.params2result = {
            "gcn_layers:2": {
                "best_valid_result": {"recall@10": 0.1},
                "test_result": None,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            tuner.export_result(str(path))
            text = path.read_text(encoding="utf-8")
        self.assertIn("Test result: not evaluated", text)

    def test_trial_cache_persists_and_reuses_completed_result(self):
        objective = Mock(
            return_value={
                "best_valid_score": 0.2,
                "valid_score_bigger": True,
                "best_valid_result": {"recall@10": 0.2},
                "test_result": None,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.json"
            first = CachedObjective(objective, path)
            expected = first({"gcn_layers": 2}, ["fixed.yaml"])
            second = CachedObjective(objective, path)
            actual = second({"gcn_layers": 2}, ["fixed.yaml"])
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertEqual(objective.call_count, 1)
        self.assertIn('{"gcn_layers":2}', persisted)


if __name__ == "__main__":
    unittest.main()
