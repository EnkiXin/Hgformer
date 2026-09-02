from __future__ import annotations

import copy
import unittest
from collections import Counter

from slrec_experiments import validate_sl8_liebn_plan as plan


class SL8LieBNPlanManifestTest(unittest.TestCase):
    def test_canonical_plan_has_20_priority_and_10_secondary_configs(self) -> None:
        payload = plan.load_and_validate()
        trials = payload["trials"]
        signatures = {plan._signature(trial) for trial in trials}
        self.assertEqual(len(trials), 30)
        self.assertEqual(len(signatures), 30)
        self.assertEqual(
            Counter(trial["group"] for trial in trials),
            Counter({"priority_l2_l4": 20, "planned_secondary": 10}),
        )
        self.assertEqual(payload["protocol"]["schatten_p"], 2)
        self.assertEqual(payload["protocol"]["eval_prefilter_candidates"], 4096)

    def test_priority_is_one_factor_not_batch_by_lr_cartesian(self) -> None:
        payload = plan.load_and_validate()
        priority = [
            trial for trial in payload["trials"] if trial["group"] == "priority_l2_l4"
        ]
        self.assertEqual(Counter(trial["layers"] for trial in priority), {2: 10, 4: 10})
        self.assertTrue(
            all(
                trial["batch_size"] == 16384 or trial["learning_rate"] == 0.005
                for trial in priority
            )
        )
        expansion = payload["optional_expansions"][0]
        self.assertFalse(expansion["enabled"])
        self.assertEqual(expansion["additional_trial_count"], 36)
        self.assertEqual(expansion["resulting_priority_trial_count"], 56)

    def test_duplicate_combination_is_rejected(self) -> None:
        payload = plan.load_and_validate()
        duplicate = copy.deepcopy(payload)
        duplicate["trials"][1].update(
            {key: duplicate["trials"][0][key] for key in plan.PARAMETER_KEYS}
        )
        with self.assertRaisesRegex(ValueError, "parameter combinations must be unique"):
            plan.validate_plan(duplicate)

    def test_out_of_plan_axis_value_is_rejected(self) -> None:
        payload = plan.load_and_validate()
        invalid = copy.deepcopy(payload)
        invalid["trials"][1]["learning_rate"] = 0.123
        invalid["trials"][1]["axis_value"] = 0.123
        with self.assertRaisesRegex(ValueError, "unexpected allowed values"):
            plan.validate_plan(invalid)

    def test_l8_is_conditional_and_outside_planned_30(self) -> None:
        payload = plan.load_and_validate()
        extension = payload["conditional_extensions"][0]
        self.assertEqual(extension["layers"], 8)
        self.assertFalse(extension["included_in_planned_30"])
        self.assertNotIn(extension, payload["trials"])


if __name__ == "__main__":
    unittest.main()
