import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HypLinear, LorentzBatchNorm
from hyperbolic_gnn.model.hgcn.manifolds.hyperboloid import Hyperboloid
from recbole.evaluator.metrics import MRR, NDCG
from recbole_gnn.model.general_recommender.recformer import RecFormer


class HgformerCompatibilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.manifold = Hyperboloid()
        self.curve = 0.5

    def test_hyperbolic_layers_run_on_cpu_and_eval_is_deterministic(self):
        raw = torch.randn(5, 8) * 0.05
        points = self.manifold.expmap0(raw, self.curve)
        layer = HypLinear(self.manifold, 8, 8, self.curve, self.curve).eval()
        first = layer(points)
        second = layer(points)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

        norm = LorentzBatchNorm(self.manifold, 8, self.curve)
        self.assertEqual(norm.curve.device.type, 'cpu')
        self.assertIn('curve', norm.state_dict())
        self.assertIn('beta', norm.state_dict())

    def test_recformer_sampled_predict_matches_two_value_forward(self):
        class PredictHarness:
            USER_ID = 'user_id'
            ITEM_ID = 'item_id'

            @staticmethod
            def forward():
                return (
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.tensor([[0.0, 0.0], [5.0, 6.0]]),
                )

        interaction = {
            'user_id': torch.tensor([1]),
            'item_id': torch.tensor([1]),
        }
        score = RecFormer.predict(PredictHarness(), interaction)
        self.assertEqual(score.item(), 39.0)

    def test_ranking_metric_helpers_use_real_numpy_dtypes(self):
        pos_index = np.array([[False, True, False], [True, False, False]])
        pos_len = np.array([1, 1])
        mrr = MRR.metric_info(MRR.__new__(MRR), pos_index)
        ndcg = NDCG.metric_info(NDCG.__new__(NDCG), pos_index, pos_len)
        self.assertTrue(np.issubdtype(mrr.dtype, np.floating))
        self.assertTrue(np.issubdtype(ndcg.dtype, np.floating))
        self.assertTrue(np.isfinite(mrr).all())
        self.assertTrue(np.isfinite(ndcg).all())


if __name__ == '__main__':
    unittest.main()
