"""CPU checks of the recall-weighted PQ reward component (wpq) and its runner flags."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nucleus_rl.grpo import build_parser, validate_args
from nucleus_rl.grpo_rewards import (COMPONENTS, check_wpq_weights, group_advantages, parse_reward_spec,
                                     reward_components, total_reward, weighted_pq)
from nucleus_rl.rewards import instance_metrics, parse_completion

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


def synthetic_cases():
    """(name, valid, object_count, prediction, reference) covering every PQ branch."""
    gt = three_nuclei()
    empty = np.zeros_like(gt)
    shifted = np.zeros_like(gt)
    square(shifted, 1, 1, 2)                 # IoU 12/20: a true positive with SQ < 1
    square(shifted, 2, 1, 12)                # exact
    square(shifted, 5, 8, 18)                # false positive
    missing = gt.copy()
    missing[gt == 3] = 0                     # one false negative
    merged = np.zeros_like(gt)
    merged[0:6, 0:17] = 1                    # one mask over two nuclei: FP + 2 FN
    return [
        ("perfect", True, 3, gt, gt),
        ("shifted_plus_fp", True, 3, shifted, gt),
        ("missing_one", True, 2, missing, gt),
        ("merged", True, 1, merged, gt),
        ("empty_prediction_valid", True, 0, empty, gt),
        ("empty_masks_from_objects", True, 2, empty, gt),
        ("empty_gt_with_prediction", True, 3, gt, empty),
        ("both_empty_with_objects", True, 1, empty, empty),
        ("invalid_answer", False, 0, empty, gt),
        ("invalid_answer_empty_gt", False, 0, empty, empty),
    ]


class WeightedPQEqualsPQTests(unittest.TestCase):
    def test_default_weights_reproduce_pq_exactly(self):
        for name, valid, count, pred, gt in synthetic_cases():
            with self.subTest(case=name):
                components = reward_components(valid, count, pred, raw_from(pred), gt)
                self.assertEqual(components["wpq"], components["pq"])
                explicit = reward_components(valid, count, pred, raw_from(pred), gt,
                                             wpq_fp_weight=0.5, wpq_fn_weight=0.5)
                self.assertEqual(explicit, components)

    def test_metric_level_identity(self):
        for name, _, _, pred, gt in synthetic_cases():
            with self.subTest(case=name):
                metrics = instance_metrics(pred, gt)
                self.assertEqual(weighted_pq(metrics), metrics["pq"])
                self.assertEqual(weighted_pq(metrics, 0.5, 0.5), metrics["pq"])

    def test_invalid_answer_is_handled_like_pq(self):
        gt = three_nuclei()
        empty = np.zeros_like(gt)
        none = np.zeros((0, *gt.shape), dtype=bool)
        for valid in (False, True):
            components = reward_components(valid, 0, empty, none, gt, wpq_fp_weight=0.1, wpq_fn_weight=2.0)
            self.assertEqual(components["wpq"], components["pq"])
            self.assertEqual(components["wpq"], 0.0)
        self.assertIn("wpq", COMPONENTS)


class WeightedPQValueTests(unittest.TestCase):
    def test_formula_matches_hand_computation(self):
        gt = three_nuclei()
        _, valid, count, pred, _ = synthetic_cases()[1]
        metrics = instance_metrics(pred, gt)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 1, 1))
        matched = 12 / 20 + 1.0
        for fp_weight, fn_weight in ((0.5, 0.5), (0.25, 1.0), (1.0, 0.25), (2.0, 3.0)):
            with self.subTest(fp=fp_weight, fn=fn_weight):
                expected = matched / (2 + fp_weight * 1 + fn_weight * 1)
                value = reward_components(valid, count, pred, raw_from(pred), gt,
                                          wpq_fp_weight=fp_weight, wpq_fn_weight=fn_weight)["wpq"]
                self.assertAlmostEqual(value, expected, places=12)

    def test_recall_weighting_penalises_misses_more_than_extras(self):
        gt = three_nuclei()
        missing = gt.copy()
        missing[gt == 3] = 0
        extra = gt.copy()
        square(extra, 9, 8, 18)
        recall = {"wpq_fp_weight": 0.25, "wpq_fn_weight": 1.0}
        miss = reward_components(True, 2, missing, raw_from(missing), gt, **recall)["wpq"]
        over = reward_components(True, 4, extra, raw_from(extra), gt, **recall)["wpq"]
        self.assertAlmostEqual(miss, 2 / 3)
        self.assertAlmostEqual(over, 3 / 3.25)
        self.assertLess(miss, over)
        # Standard PQ (2/2.5 vs 3/3.5) separates a miss from an extra far less.
        pq_miss = reward_components(True, 2, missing, raw_from(missing), gt)["pq"]
        pq_over = reward_components(True, 4, extra, raw_from(extra), gt)["pq"]
        self.assertAlmostEqual(pq_miss, 0.8)
        self.assertAlmostEqual(pq_over, 3 / 3.5)
        self.assertGreater(over - miss, pq_over - pq_miss)

    def test_edge_cases_stay_in_unit_interval(self):
        for name, valid, count, pred, gt in synthetic_cases():
            for weights in ((0.01, 0.01), (0.25, 1.0), (5.0, 5.0)):
                with self.subTest(case=name, weights=weights):
                    value = reward_components(valid, count, pred, raw_from(pred), gt,
                                              wpq_fp_weight=weights[0], wpq_fn_weight=weights[1])["wpq"]
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
        empty = np.zeros((8, 8), dtype=np.int32)
        both_empty = reward_components(True, 1, empty, raw_from(empty), empty, wpq_fp_weight=0.2, wpq_fn_weight=3)
        self.assertEqual(both_empty["wpq"], both_empty["pq"])
        self.assertEqual(both_empty["wpq"], 1.0)

    def test_weights_must_be_finite_and_positive(self):
        gt = three_nuclei()
        for weights in ((0.0, 0.5), (0.5, 0.0), (-1.0, 0.5), (0.5, float("nan")), (float("inf"), 0.5), (True, 0.5)):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    check_wpq_weights(*weights)
                with self.assertRaises(ValueError):
                    reward_components(True, 3, gt, raw_from(gt), gt,
                                      wpq_fp_weight=weights[0], wpq_fn_weight=weights[1])


class WeightedPQRunnerTests(unittest.TestCase):
    def args(self, *flags):
        return build_parser().parse_args([
            "--manifest", "train.jsonl", "--eval-manifest", "val.jsonl", "--output", "run",
            "--model", "Qwen/Qwen3.5-9B", "--revision", REVISION, "--init-adapter", "sft",
            "--sam-checkpoint", "sam.pt", "--coordinate-frame", "normalized1000",
            "--min-pixels", "50176", "--max-pixels", "200704", *flags])

    def test_reward_spec_and_flags(self):
        self.assertEqual(parse_reward_spec("format=1,wpq=1"), {"format": 1.0, "wpq": 1.0})
        args = self.args("--reward", "format=1,wpq=1", "--wpq-fp-weight", "0.25", "--wpq-fn-weight", "1")
        validate_args(args)
        self.assertEqual(args.reward_weights, {"format": 1.0, "wpq": 1.0})
        self.assertEqual((args.wpq_fp_weight, args.wpq_fn_weight), (0.25, 1.0))
        defaults = self.args()
        validate_args(defaults)
        self.assertEqual((defaults.wpq_fp_weight, defaults.wpq_fn_weight), (0.5, 0.5))
        for flags in (("--wpq-fp-weight", "0"), ("--wpq-fn-weight", "-0.5"), ("--wpq-fn-weight", "nan")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_args(self.args(*flags))

    def test_weights_are_recorded_like_arguments_json(self):
        from nucleus_rl.sft import json_write
        args = self.args("--reward", "format=1,wpq=1", "--wpq-fp-weight", "0.25", "--wpq-fn-weight", "1.5")
        validate_args(args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arguments.json"
            json_write(path, vars(args))  # exactly what grpo.main writes
            recorded = json.loads(path.read_text())
        self.assertEqual((recorded["wpq_fp_weight"], recorded["wpq_fn_weight"]), (0.25, 1.5))
        self.assertEqual(recorded["reward_weights"], {"format": 1.0, "wpq": 1.0})

    def test_group_scoring_path_with_wpq(self):
        """The grpo_micro scoring loop: parse -> components -> weighted total -> advantages."""
        args = self.args("--reward", "format=1,wpq=1", "--wpq-fp-weight", "0.25", "--wpq-fn-weight", "1")
        validate_args(args)
        gt = three_nuclei()
        answers = {  # each completion's text and the SAM instance map it would produce
            '<answer>{"objects":[{"box":[1,1,5,5],"point":[3,3]},{"box":[12,1,16,5],"point":[14,3]},'
            '{"box":[6,14,11,19],"point":[8,16]}]}</answer>': gt,
            '<answer>{"objects":[{"box":[1,1,5,5],"point":[3,3]}]}</answer>': np.where(gt == 1, 1, 0),
            '<answer>{"objects":[]}</answer>': np.zeros_like(gt),
            "not an answer": np.zeros_like(gt),
        }
        rewards = []
        for text, instances in answers.items():
            parsed = parse_completion(text, width=gt.shape[1], height=gt.shape[0], max_objects=args.max_objects)
            objects = parsed.objects if parsed.valid else ()
            components = reward_components(parsed.valid, len(objects), instances, raw_from(instances), gt,
                                           wpq_fp_weight=args.wpq_fp_weight, wpq_fn_weight=args.wpq_fn_weight)
            rewards.append(total_reward(components, args.reward_weights))
        self.assertEqual(rewards[0], 2.0)
        self.assertAlmostEqual(rewards[1], 1.0 + 1 / (1 + 1.0 * 2))
        self.assertEqual(rewards[2:], [1.0, 0.0])
        advantages = group_advantages(rewards)
        self.assertEqual(len(advantages), 4)
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(advantages[0], advantages[1])


if __name__ == "__main__":
    unittest.main()
