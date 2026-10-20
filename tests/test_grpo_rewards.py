"""CPU checks of GRPO reward components, group advantages and answer lengths."""
import unittest

import numpy as np

from nucleus_rl.grpo_rewards import (COMPONENTS, completion_lengths, group_advantages, merge_extras,
                                     parse_reward_spec, reward_components, total_reward)


def two_nuclei():
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[2:6, 2:6] = 1
    gt[9:14, 9:14] = 2
    return gt


def raw_from(instances):
    return np.stack([instances == index for index in np.unique(instances) if index > 0])


class RewardSpecTests(unittest.TestCase):
    def test_parses_weights_and_rejects_bad_items(self):
        self.assertEqual(parse_reward_spec("format=1, pq=0.5"), {"format": 1.0, "pq": 0.5})
        for spec in ("", "pq", "dice=1", "pq=1,pq=2", "pq=-1", "pq=nan", "format=0,pq=0"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_reward_spec(spec)


class RewardComponentTests(unittest.TestCase):
    def test_perfect_answer_earns_every_component(self):
        gt = two_nuclei()
        components = reward_components(True, 2, gt, raw_from(gt), gt)
        self.assertEqual(set(components), set(COMPONENTS))
        self.assertTrue(all(value == 1.0 for value in components.values()), components)

    def test_invalid_or_empty_answers_earn_nothing_but_format(self):
        gt = two_nuclei()
        empty = np.zeros_like(gt)
        none = np.zeros((0, *gt.shape), dtype=bool)
        self.assertEqual(set(reward_components(False, 0, empty, none, gt).values()), {0.0})
        valid_empty = reward_components(True, 0, empty, none, gt)
        self.assertEqual(valid_empty["format"], 1.0)
        self.assertEqual(sum(valid_empty.values()), 1.0)

    def test_one_mask_over_two_nuclei_loses_count_and_separation(self):
        gt = two_nuclei()
        merged = np.zeros_like(gt)
        merged[1:15, 1:15] = 1
        self.assertEqual(merge_extras(raw_from(merged), gt), 1)
        components = reward_components(True, 1, merged, raw_from(merged), gt)
        self.assertEqual(components["count"], 0.5)
        self.assertEqual(components["separation"], 0.0)
        self.assertEqual(components["pq"], 0.0)
        self.assertGreater(components["fg_iou"], 0.0)

    def test_duplicate_masks_lose_separation(self):
        gt = two_nuclei()
        raw = np.stack([gt == 1, gt == 1, gt == 2])
        components = reward_components(True, 3, gt, raw, gt)
        self.assertAlmostEqual(components["separation"], 1 - 1 / 3)

    def test_weighted_total(self):
        components = {"format": 1.0, "fg_iou": 0.5, "pq": 0.25, "count": 1.0, "separation": 1.0}
        self.assertEqual(total_reward(components, {"format": 1, "pq": 2}), 1.5)


class AdvantageTests(unittest.TestCase):
    def test_group_relative_scaling_and_zero_variance(self):
        advantages = group_advantages([1.0, 2.0, 3.0, 2.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(advantages[2], 0)
        self.assertLess(advantages[0], 0)
        self.assertEqual(group_advantages([0.7, 0.7, 0.7]), [0.0, 0.0, 0.0])
        for rewards in ([1.0], [1.0, float("nan")]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                group_advantages(rewards)

    def test_lengths_stop_at_first_end_token(self):
        rows = [[5, 6, 99, 0, 0], [5, 7, 8, 9, 10], [98, 1]]
        self.assertEqual(completion_lengths(rows, [98, 99]), [(3, True), (5, False), (1, True)])


if __name__ == "__main__":
    unittest.main()
