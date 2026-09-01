"""Regression tests for sampled validation followed by exact full ranking."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from recbole.data.dataloader.general_dataloader import NegSampleEvalDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole_gnn.quick_start import evaluate_recbole_gnn_checkpoint


class _TinySplit:
    uid_field = "user_id"
    iid_field = "item_id"
    user_num = 4
    item_num = 32

    def __init__(self, users, items):
        self.inter_feat = Interaction(
            {
                self.uid_field: torch.tensor(users, dtype=torch.long),
                self.iid_field: torch.tensor(items, dtype=torch.long),
            }
        )

    def __getitem__(self, index):
        return self.inter_feat[index]


class SampledEvalBatchTest(unittest.TestCase):
    def test_cat_interactions_flattens_uneven_batches_and_all_columns(self):
        first = Interaction(
            {
                "user_id": torch.tensor([1, 1]),
                "item_id": torch.tensor([2, 3]),
                "label": torch.tensor([1.0, 1.0]),
                "context": torch.tensor([7, 8]),
            }
        )
        second = Interaction(
            {
                "user_id": torch.tensor([2]),
                "item_id": torch.tensor([4]),
                "label": torch.tensor([0.0]),
                "context": torch.tensor([9]),
            }
        )

        merged = cat_interactions([first, second])

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.columns, first.columns)
        torch.testing.assert_close(merged["user_id"], torch.tensor([1, 1, 2]))
        torch.testing.assert_close(merged["context"], torch.tensor([7, 8, 9]))

    def test_uni_sampled_batch_contains_every_user_in_the_batch(self):
        dataset = _TinySplit(users=[1, 1, 2], items=[2, 3, 4])
        loader = NegSampleEvalDataLoader.__new__(NegSampleEvalDataLoader)
        loader.neg_sample_args = {"strategy": "by"}
        loader.uid_list = np.array([1, 2])
        loader.uid2index = np.array(
            [None, slice(0, 2), slice(2, 3)], dtype=object
        )
        loader.uid2items_num = np.array([0, 2, 1])
        loader.pr = 0
        loader.step = 2
        loader.times = 2
        loader.dataset = dataset
        loader.iid_field = "item_id"

        def sampled(interaction):
            return Interaction(
                {
                    "user_id": torch.cat(
                        (interaction["user_id"], interaction["user_id"])
                    ),
                    "item_id": torch.cat(
                        (interaction["item_id"], interaction["item_id"] + 10)
                    ),
                    "label": torch.cat(
                        (torch.ones(len(interaction)), torch.zeros(len(interaction)))
                    ),
                }
            )

        loader._neg_sampling = sampled
        interaction, row_idx, positive_u, positive_i = loader._next_batch_data()

        self.assertEqual(len(interaction), 6)
        torch.testing.assert_close(
            interaction["user_id"], torch.tensor([1, 1, 1, 1, 2, 2])
        )
        torch.testing.assert_close(row_idx, torch.tensor([0, 0, 0, 0, 1, 1]))
        torch.testing.assert_close(positive_u, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(positive_i, torch.tensor([2, 3, 4]))


class _FakeModel(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.loaded_other = None
        self.cache_cleared = False

    def load_other_parameter(self, value):
        self.loaded_other = value

    def _clear_full_sort_cache(self):
        self.cache_cleared = True


class _Collector:
    def __init__(self):
        self.collected = None

    def data_collect(self, train_data):
        self.collected = train_data


class _FakeTrainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.eval_collector = _Collector()

    def evaluate(self, data, load_best_model, show_progress):
        if self.config["eval_args"]["mode"] != "full":
            raise AssertionError("checkpoint evaluator did not switch to full ranking")
        if load_best_model:
            raise AssertionError("state should already be loaded with map_location=cpu")
        return {"Recall@10": float(len(data.dataset.inter_feat)) / 10.0}


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset


class FullCheckpointEvaluationTest(unittest.TestCase):
    def test_checkpoint_config_rebuilds_same_split_in_full_mode(self):
        config = {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "uni100",
            },
            "eval_neg_sample_args": {
                "strategy": "by",
                "by": 100,
                "distribution": "uniform",
            },
            "use_gpu": False,
            "device": torch.device("cpu"),
            "seed": 2024,
            "reproducibility": True,
            "show_progress": False,
            "model": "FakeModel",
            "dataset": "tiny",
            "MODEL_TYPE": "general",
            "eval_batch_size": 100,
            "eval_user_chunk_size": 1,
            "eval_item_chunk_size": 2,
        }
        train = _Loader(_TinySplit([1, 1, 2], [2, 3, 4]))
        valid = _Loader(_TinySplit([1, 2], [5, 6]))
        test = _Loader(_TinySplit([1, 2], [7, 8]))
        saved_model = _FakeModel(config, train.dataset)
        saved_model.weight.data.fill_(7.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "model.pth"
            torch.save(
                {
                    "config": config,
                    "epoch": 11,
                    "best_valid_score": 0.25,
                    "state_dict": saved_model.state_dict(),
                    "other_parameter": {"cache": "stale"},
                },
                checkpoint,
            )
            with (
                patch("recbole_gnn.quick_start.init_seed"),
                patch("recbole_gnn.quick_start.init_logger"),
                patch("recbole_gnn.quick_start.create_dataset", return_value=object()),
                patch(
                    "recbole_gnn.quick_start.data_preparation",
                    return_value=(train, valid, test),
                ),
                patch(
                    "recbole_gnn.quick_start.get_model",
                    return_value=_FakeModel,
                ),
                patch(
                    "recbole_gnn.quick_start.get_trainer",
                    return_value=_FakeTrainer,
                ),
            ):
                result = evaluate_recbole_gnn_checkpoint(
                    checkpoint,
                    eval_batch_size=4096,
                    eval_user_chunk_size=8,
                    eval_item_chunk_size=16,
                    full_sort_user_batch_size=8,
                    device="cpu",
                )

        self.assertEqual(config["eval_args"]["mode"], "uni100")
        self.assertEqual(result["selection_eval_mode"], "uni100")
        self.assertEqual(result["evaluation_eval_mode"], "full")
        self.assertEqual(result["checkpoint_epoch"], 11)
        self.assertEqual(result["eval_batch_size"], 4096)
        self.assertEqual(result["eval_user_chunk_size"], 8)
        self.assertEqual(result["eval_item_chunk_size"], 16)
        self.assertEqual(result["full_sort_user_batch_size"], 8)
        self.assertEqual(result["split_fingerprints"]["train"]["interactions"], 3)
        self.assertEqual(result["valid_result"], {"Recall@10": 0.2})
        self.assertEqual(result["test_result"], {"Recall@10": 0.2})


if __name__ == "__main__":
    unittest.main()
