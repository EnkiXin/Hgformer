"""Regression tests for the standalone archived LHGCN adapter."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from recbole.utils import InputType
from recbole_gnn.model.general_recommender.hgcf import HGCF
from recbole_gnn.model.general_recommender.lhgcn import LHGCN
from recbole_gnn.utils import get_model


class _TinyGraphDataset:
    def __init__(self):
        # RecBole reserves id zero.  Keeping it isolated exercises the same
        # zero-degree handling as a real filtered recommendation graph.
        self._interaction = sp.coo_matrix(
            (
                np.ones(5, dtype=np.float32),
                (
                    np.array([1, 1, 2, 2, 2]),
                    np.array([1, 2, 2, 3, 4]),
                ),
            ),
            shape=(3, 5),
        )
        self.inter_num = self._interaction.nnz

    @staticmethod
    def num(field):
        return {"user_id": 3, "item_id": 5}[field]

    def inter_matrix(self, form="coo"):
        if form != "coo":
            raise ValueError("tiny fixture only implements COO")
        return self._interaction

    def get_interactions(self):
        return (
            torch.as_tensor(self._interaction.row, dtype=torch.long),
            torch.as_tensor(self._interaction.col, dtype=torch.long),
        )

    def get_norm_adj_mat(self, enable_sparse=False):
        del enable_sparse
        users = torch.as_tensor(self._interaction.row, dtype=torch.long)
        items = torch.as_tensor(self._interaction.col, dtype=torch.long) + 3
        source = torch.cat((users, items))
        target = torch.cat((items, users))
        edge_index = torch.stack((source, target))
        # GeneralGraphRecommender requires this legacy PyG representation, but
        # HGCF/LHGCN build and use their own scipy-normalised sparse adjacency.
        edge_weight = torch.ones(source.numel(), dtype=torch.float32)
        return edge_index, edge_weight


def _config(**updates):
    config = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"),
        "enable_sparse": False,
        "embedding_size": 8,
        "gcn_layers": 2,
        "curve": 0.5,
        "margin": 0.1,
        "scale": 0.05,
        "conv": "lGCN",
        "learner": "adam",
        "tail_analysis": False,
        "popularity_analysis": False,
    }
    config.update(updates)
    return config


class LHGCNAdapterTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _TinyGraphDataset()
        torch.manual_seed(2024)
        self.reference = HGCF(_config(), self.dataset)
        torch.manual_seed(7)
        self.model = LHGCN(_config(), self.dataset)
        self.model.load_state_dict(self.reference.state_dict())
        self.reference.eval()
        self.model.eval()
        self.interaction = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 3]),
            "neg_item_id": torch.tensor([4, 1]),
        }

    def test_legacy_runner_discovers_pairwise_model(self):
        self.assertIs(get_model("LHGCN"), LHGCN)
        self.assertEqual(LHGCN.input_type, InputType.PAIRWISE)
        self.assertFalse(hasattr(self.model, "encoder"))
        self.assertFalse(hasattr(self.model, "transformer"))

    def test_forward_matches_hgcf_with_lgcn(self):
        expected_users, expected_items = self.reference.forward()
        actual_users, actual_items = self.model.forward()
        torch.testing.assert_close(actual_users, expected_users, atol=0, rtol=0)
        torch.testing.assert_close(actual_items, expected_items, atol=0, rtol=0)

    def test_squared_distance_hinge_sum_matches_hgcf(self):
        expected = self.reference.calculate_loss(self.interaction)
        actual = self.model.calculate_loss(self.interaction)
        self.assertEqual(actual.ndim, 0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        users, items = self.model.forward()
        positive_distance = self.model.manifold.sqdist(
            users[self.interaction["user_id"]],
            items[self.interaction["item_id"]],
            self.model.curve,
        )
        negative_distance = self.model.manifold.sqdist(
            users[self.interaction["user_id"]],
            items[self.interaction["neg_item_id"]],
            self.model.curve,
        )
        manual = (
            positive_distance - negative_distance + self.model.margin
        ).clamp_min(0).sum()
        torch.testing.assert_close(actual, manual, atol=0, rtol=0)

    def test_full_sort_negative_distance_matches_hgcf(self):
        query = {"user_id": torch.tensor([1, 2])}
        expected = self.reference.full_sort_predict(query)
        actual = self.model.full_sort_predict(query)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_lgcn_is_defaulted_and_other_convolutions_are_rejected(self):
        config = _config(conv=None)
        model = LHGCN(config, self.dataset)
        self.assertEqual(config["conv"], "lGCN")
        self.assertEqual(model.gcn_conv.conv, "lGCN")

        with self.assertRaisesRegex(ValueError, "conv='lGCN'"):
            LHGCN(_config(conv="resSumGCN"), self.dataset)


if __name__ == "__main__":
    unittest.main()
