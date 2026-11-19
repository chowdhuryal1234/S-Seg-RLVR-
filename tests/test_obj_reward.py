"""CPU checks of the per-object credit reward component (obj) and its runner flag."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nucleus_rl.grpo import build_parser, validate_args
from nucleus_rl.grpo_rewards import (COMPONENTS, check_obj_settings, group_advantages, object_credit,
                                     parse_reward_spec, reward_components)
from nucleus_rl.rewards import instance_metrics

REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def square(canvas, label, top, left, size=4):
    canvas[top:top + size, left:left + size] = label


def three_nuclei():
    gt = np.zeros((24, 24), dtype=np.int32)
    square(gt, 1, 1, 1)
    square(gt, 2, 1, 12)
    square(gt, 3, 14, 6, size=5)
    return gt


def raw_from(instances):
    ids = [index for index in np.unique(instances) if index > 0]
    if not ids:
        return np.zeros((0, *np.shape(instances)), dtype=bool)
    return np.stack([instances == index for index in ids])


def obj(prediction, reference, fp_weight=0.5, cap=8, valid=True, count=None):
    count = len(raw_from(prediction)) if count is None else count
    return reward_components(valid, count, prediction, raw_from(prediction), reference,
                             obj_fp_weight=fp_weight, obj_cap=cap)["obj"]


class ObjectCreditTests(unittest.TestCase):
    def test_formula_on_hand_counted_cases(self):
        gt = three_nuclei()
        offset = 0.5 * 8
        self.assertAlmostEqual(obj(gt, gt), (3 + offset) / (3 + offset))
        one = np.where(gt == 1, 1, 0)
        self.assertAlmostEqual(obj(one, gt), (1 + offset) / (3 + offset))
        with_fp = gt.copy()
        square(with_fp, 7, 8, 18)
        self.assertAlmostEqual(obj(with_fp, gt), (3 - 0.5 + offset) / (3 + offset))
        shifted = np.zeros_like(gt)
        square(shifted, 1, 1, 2)  # IoU 12/20 with nucleus 1
        self.assertAlmostEqual(obj(shifted, gt), (0.6 + offset) / (3 + offset))

    def test_matches_sum_of_matched_iou_minus_lambda_fp(self):
        gt = three_nuclei()
        pred = np.zeros_like(gt)
        square(pred, 1, 1, 2)      # TP, IoU 0.6
        square(pred, 2, 1, 12)     # TP, IoU 1
        square(pred, 4, 8, 18)     # FP
        square(pred, 5, 19, 19)    # FP
        for lam in (0.1, 0.25, 0.5, 1.0):
            with self.subTest(lam=lam):
                raw = (0.6 + 1.0) - lam * 2
                self.assertAlmostEqual(obj(pred, gt, lam), (raw + lam * 8) / (3 + lam * 8))
                self.assertAlmostEqual(object_credit(instance_metrics(pred, gt), lam, 8), obj(pred, gt, lam))

    def test_extra_box_breakeven_follows_lambda(self):
        """An added box raises expected credit iff P(hit) * IoU > (1 - P(hit)) * lambda."""
        gt = three_nuclei()
        base = np.where(gt == 1, 1, 0)
        hit = base.copy()
        hit[gt == 2] = 2
        miss = base.copy()
        square(miss, 2, 8, 18)
        for lam, p in ((0.25, 0.3), (0.5, 0.45), (0.5, 0.2), (0.25, 0.15)):
            with self.subTest(lam=lam, p=p):
                expected_gain = p * obj(hit, gt, lam) + (1 - p) * obj(miss, gt, lam) - obj(base, gt, lam)
                self.assertEqual(expected_gain > 0, p * 1.0 > (1 - p) * lam)

    def test_range_order_and_empty_answers(self):
        gt = three_nuclei()
        empty = np.zeros_like(gt)
        all_wrong = np.zeros_like(gt)
        for label, (top, left) in enumerate([(8, 18), (19, 19), (8, 0), (19, 0), (10, 10), (0, 20),
                                             (6, 6), (20, 14), (12, 0), (16, 20)], start=1):
            square(all_wrong, label, top, left, size=2)
        self.assertEqual(obj(empty, gt, valid=False, count=0), 0.0)
        empty_valid = obj(empty, gt, count=0)
        self.assertAlmostEqual(empty_valid, 4 / 7)
        wrong = obj(all_wrong, gt)  # 10 false positives, capped at 8
        self.assertEqual(wrong, 0.0)
        self.assertLess(wrong, empty_valid)
        self.assertLessEqual(obj(gt, gt), 1.0)
        both_empty = obj(empty, empty, count=0)
        self.assertEqual(both_empty, 1.0)
        self.assertIn("obj", COMPONENTS)

    def test_affine_within_an_image_so_advantages_match_raw_credit(self):
        gt = three_nuclei()
        preds = [gt, np.where(gt == 1, 1, 0), np.zeros_like(gt)]
        extra = gt.copy()
        square(extra, 9, 8, 18)
        preds.append(extra)
        lam = 0.25
        scaled = [obj(p, gt, lam) for p in preds]
        raw = []
        for p in preds:
            m = instance_metrics(p, gt)
            matched = m["pq"] * (m["tp"] + 0.5 * m["fp"] + 0.5 * m["fn"]) if m["tp"] else 0.0
            raw.append(matched - lam * m["fp"])
        np.testing.assert_allclose(group_advantages(scaled), group_advantages(raw), atol=1e-3)

    def test_settings_checks(self):
        for bad in ((0, 8), (-0.5, 8), (float("nan"), 8), (True, 8), (0.5, 0), (0.5, 2.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_obj_settings(*bad)


class ObjectCreditRunnerTests(unittest.TestCase):
    def args(self, *flags):
        return build_parser().parse_args([
            "--manifest", "train.jsonl", "--eval-manifest", "val.jsonl", "--output", "run",
            "--model", "Qwen/Qwen3.5-9B", "--revision", REVISION, "--init-adapter", "sft",
            "--sam-checkpoint", "sam.pt", "--coordinate-frame", "normalized1000",
            "--min-pixels", "50176", "--max-pixels", "200704", *flags])

    def test_flag_spec_and_recording(self):
        from nucleus_rl.sft import json_write
        self.assertEqual(parse_reward_spec("format=1,obj=1"), {"format": 1.0, "obj": 1.0})
        args = self.args("--reward", "format=1,obj=1", "--obj-fp-weight", "0.25")
        validate_args(args)
        defaults = self.args()
        validate_args(defaults)
        self.assertEqual(defaults.obj_fp_weight, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arguments.json"
            json_write(path, vars(args))
            recorded = json.loads(path.read_text())
        self.assertEqual(recorded["obj_fp_weight"], 0.25)
        self.assertEqual(recorded["reward_weights"], {"format": 1.0, "obj": 1.0})
        for flags in (("--obj-fp-weight", "0"), ("--obj-fp-weight", "-1"), ("--obj-fp-weight", "inf")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_args(self.args(*flags))


if __name__ == "__main__":
    unittest.main()
