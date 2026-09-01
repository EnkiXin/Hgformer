import unittest
from unittest.mock import patch

import torch

from slrec_experiments.geometry import one_sided_gregory_frobenius_distance_k12
from slrec_experiments.profile_sl8_cd import ProfilerCompiledFullSortScorer


class _FakeSL8Model(torch.nn.Module):
    matrix_dim = 8
    num_factors = 1
    log_jitter = 0.0

    def __init__(self):
        super().__init__()
        self.audit_pending = True
        self.audit_count = 0

    def _uses_fast_one_sided_frobenius(self):
        return True

    @staticmethod
    def _align_pair_shapes(left, right):
        return left, right

    def _record_distance_membership_diagnostics(self, _left, _right):
        if self.audit_pending:
            self.audit_count += 1
            self.audit_pending = False

    @staticmethod
    def _score_scale():
        return torch.tensor(1.0)

    def _score_groups(self, left, right):
        distance = one_sided_gregory_frobenius_distance_k12(
            left, right, jitter=self.log_jitter
        )
        return -distance.squeeze(-1)


def _groups(users, items):
    identity = torch.eye(8)
    left = identity.reshape(1, 1, 1, 8, 8).repeat(users, 1, 1, 1, 1)
    right = identity.reshape(1, 1, 1, 8, 8).repeat(1, items, 1, 1, 1)
    return left, right


class ProfilerCompiledFullSortScorerTest(unittest.TestCase):
    def test_caches_four_tail_shapes_and_preserves_one_time_audit(self):
        model = _FakeSL8Model()
        original_method = model._score_groups
        wrapper = ProfilerCompiledFullSortScorer(model)

        with patch(
            "slrec_experiments.profile_sl8_cd.torch.compile",
            side_effect=lambda function, **_kwargs: function,
        ) as compile_mock:
            with wrapper.installed():
                for users, items in ((2, 3), (2, 1), (1, 3), (1, 1)):
                    left, right = _groups(users, items)
                    expected = original_method(left, right)
                    actual = model._score_groups(left, right)
                    self.assertTrue(torch.equal(actual, expected))
                    # A repeated main shape must reuse the compiled callable.
                    if (users, items) == (2, 3):
                        self.assertTrue(
                            torch.equal(model._score_groups(left, right), expected)
                        )

        report = wrapper.report()
        self.assertEqual(compile_mock.call_count, 4)
        self.assertEqual(report["shape_cache_entries"], 4)
        self.assertEqual(report["total_calls"], 5)
        self.assertEqual(report["total_compiled_calls"], 5)
        self.assertEqual(report["total_eager_fallback_calls"], 0)
        self.assertEqual(model.audit_count, 1)
        # Removing the instance override restores the class descriptor.
        left, right = _groups(1, 1)
        self.assertTrue(torch.equal(model._score_groups(left, right), original_method(left, right)))

    def test_compile_failure_is_cached_as_eager_fallback(self):
        model = _FakeSL8Model()
        wrapper = ProfilerCompiledFullSortScorer(model)
        left, right = _groups(2, 3)
        expected = model._score_groups(left, right)

        def failing_compile(_function, **_kwargs):
            def fail(*_args):
                raise RuntimeError("synthetic compile failure")

            return fail

        with patch(
            "slrec_experiments.profile_sl8_cd.torch.compile",
            side_effect=failing_compile,
        ) as compile_mock:
            with wrapper.installed():
                first = model._score_groups(left, right)
                second = model._score_groups(left, right)

        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))
        self.assertEqual(compile_mock.call_count, 1)
        report = wrapper.report()
        self.assertEqual(report["shape_cache_entries"], 1)
        self.assertEqual(report["total_eager_fallback_calls"], 2)
        self.assertIn(
            "synthetic compile failure",
            report["shape_records"][0]["fallback_error"],
        )


if __name__ == "__main__":
    unittest.main()
