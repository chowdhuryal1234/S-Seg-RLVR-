"""Meaningful CPU tests for parsing, overlap handling and reward blind spots."""

import json
import unittest

import numpy as np

from nucleus_rl.rewards import (
    evaluate_prediction, foreground_iou, instance_metrics,
    masks_to_instances, parse_completion,
)


def completion(objects):
    return "<answer>" + json.dumps({"objects": objects}) + "</answer>"


ONE = {"box": [0, 0, 4, 3], "point": [1, 1]}


class ParseTests(unittest.TestCase):
    def parse(self, text, **kwargs):
        return parse_completion(text, width=4, height=3, **kwargs)

    def test_valid_with_reasoning_and_boundary_edges(self):
        result = self.parse("<think>One nucleus is visible.</think>\n" + completion([ONE]))
        self.assertTrue(result.valid)
        self.assertEqual(result.objects[0].box, (0, 0, 4, 3))

    def test_fractional_coordinates_allowed(self):
        obj = {"box": [0.2, 0.5, 3.8, 2.9], "point": [1.5, 1.5]}
        self.assertTrue(self.parse(completion([obj])).valid)

    def test_empty_and_overlapping_objects_are_valid(self):
        self.assertTrue(self.parse(completion([])).valid)
        self.assertTrue(self.parse(completion([ONE, ONE])).valid)

    def test_object_limit(self):
        self.assertTrue(self.parse(completion([ONE] * 8)).valid)
        self.assertFalse(self.parse(completion([ONE] * 9)).valid)

    def test_reject_truncated_and_extra_text(self):
        valid = completion([ONE])
        for text in [valid[:-10], valid + " done", "Here: " + valid,
                     valid + valid, "```json\n" + valid + "\n```"]:
            with self.subTest(text=text):
                self.assertFalse(self.parse(text).valid)

    def test_reject_duplicate_keys_at_any_level(self):
        for text in [
            '<answer>{"objects":[],"objects":[]}</answer>',
            '<answer>{"objects":[{"box":[0,0,4,3],"box":[0,0,1,1],"point":[0,0]}]}</answer>',
        ]:
            self.assertFalse(self.parse(text).valid)

    def test_only_one_well_formed_optional_reasoning_block(self):
        answer = completion([ONE])
        for prefix in ["<think>a</think>outside<think>b</think>",
                       "<think>a</think><think>b</think>",
                       "<think><think>nested</think></think>"]:
            self.assertFalse(self.parse(prefix + answer).valid)

    def test_reject_unknown_keys(self):
        self.assertFalse(self.parse('<answer>{"objects":[],"score":1}</answer>').valid)
        self.assertFalse(self.parse(completion([{**ONE, "label": "nucleus"}])).valid)

    def test_reject_boolean_and_nonfinite_coordinates(self):
        for bad in [True, False, float("nan"), float("inf"), "1", None]:
            with self.subTest(bad=bad):
                self.assertFalse(self.parse(completion([{**ONE, "point": [bad, 1]}])).valid)
        self.assertFalse(self.parse('<answer>{"objects":[{"box":[0,0,1e400,3],"point":[0,0]}]}</answer>').valid)

    def test_reject_reflected_degenerate_outside_boxes(self):
        for box in [[3, 0, 1, 3], [0, 0, 0, 3], [-1, 0, 4, 3], [0, 0, 5, 3], [0, 3, 4, 1]]:
            self.assertFalse(self.parse(completion([{**ONE, "box": box}])).valid)

    def test_point_must_be_inside_half_open_box(self):
        for point in [[4, 1], [1, 3], [-1, 1], [0, 0]]:
            obj = {"box": [1, 1, 4, 3], "point": point}
            self.assertFalse(self.parse(completion([obj])).valid)


class MaskAndMetricTests(unittest.TestCase):
    def test_probability_wins_then_score_then_input_order(self):
        masks = np.array([[[0.7, 0.9, 0.5]], [[0.8, 0.9, 0.5]]])
        np.testing.assert_array_equal(masks_to_instances(masks, [0.9, 0.1]), [[2, 1, 0]])
        np.testing.assert_array_equal(masks_to_instances(masks, [0.1, 0.9]), [[2, 2, 0]])
        np.testing.assert_array_equal(masks_to_instances(masks, [0.9, 0.9]), [[2, 1, 0]])

    def test_binary_masks_and_empty_candidates(self):
        masks = np.array([[[True, True]], [[False, True]]])
        np.testing.assert_array_equal(masks_to_instances(masks, [0.2, 0.8]), [[1, 2]])
        np.testing.assert_array_equal(masks_to_instances(np.zeros((0, 3, 4))), np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            masks_to_instances([])

    def test_reject_logits_nan_and_mismatched_shapes(self):
        for masks in [np.array([[[1.1]]]), np.array([[[np.nan]]])]:
            with self.assertRaises(ValueError):
                masks_to_instances(masks)
        with self.assertRaises(ValueError):
            foreground_iou(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_same_foreground_different_instances_exposes_iou_blind_spot(self):
        gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        merged = np.ones_like(gt)
        self.assertEqual(foreground_iou(merged, gt), 1.0)
        self.assertEqual(instance_metrics(gt, gt)["pq"], 1.0)
        metrics = instance_metrics(merged, gt)
        self.assertEqual(metrics["pq"], 0.0)  # Each IoU is exactly .5: strict > .5.
        self.assertEqual((metrics["pred_count"], metrics["gt_count"]), (1, 2))

    def test_disconnected_single_id_remains_one_instance(self):
        ids = np.array([[9, 0, 9], [0, 0, 0]])
        result = instance_metrics(ids, ids)
        self.assertEqual(result["gt_count"], 1)
        self.assertEqual(result["pq"], 1.0)

    def test_arbitrary_ids_and_one_false_positive(self):
        gt = np.array([[99, 99, 0, 0]])
        pred = np.array([[10001, 10001, 0, 3]])
        result = instance_metrics(pred, gt)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["pq"], 2 / 3)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_empty_metric_conventions(self):
        empty = np.zeros((2, 2), dtype=np.uint16)
        full = np.ones((2, 2), dtype=np.uint16)
        self.assertEqual(foreground_iou(empty, empty), 1.0)
        self.assertEqual(instance_metrics(empty, empty)["pq"], 1.0)
        self.assertEqual(instance_metrics(full, empty)["fp"], 1)
        self.assertEqual(instance_metrics(full, empty)["pq"], 0.0)
        self.assertEqual(instance_metrics(full, empty)["recall"], 1.0)
        self.assertEqual(instance_metrics(empty, full)["recall"], 0.0)
        self.assertEqual(instance_metrics(empty, full)["precision"], 0.0)

    def test_integer_ids_required(self):
        for bad in [np.array([[0.2]]), np.array([[-1]]), np.array([[np.inf]])]:
            with self.assertRaises(ValueError):
                instance_metrics(bad, np.zeros((1, 1)))


class RewardTests(unittest.TestCase):
    def test_explicit_prompt_frame_preserves_original_mask_metrics(self):
        gt = np.zeros((128, 128), dtype=np.uint16)
        gt[80:115, 80:115] = 300
        text = completion([{"box": [140, 140, 200, 200], "point": [168, 168]}])
        result = evaluate_prediction(text, gt.copy(), gt, prompt_size=(224, 224))
        self.assertTrue(result["format_valid"])
        self.assertEqual(result["seg_reward"], 1.0)
        self.assertEqual(result["pq"], 1.0)
        self.assertEqual(result["total_reward"], 2.0)
        self.assertEqual(result["gt_count"], 1)
        default = evaluate_prediction(text, gt.copy(), gt)
        self.assertFalse(default["format_valid"])
        self.assertEqual(default["total_reward"], 0.0)

    def test_outside_explicit_frame_is_invalid_even_with_perfect_mask(self):
        gt = np.ones((128, 128), dtype=np.uint16)
        for obj in [
            {"box": [0, 0, 225, 224], "point": [168, 168]},
            {"box": [0, 0, 224, 224], "point": [225, 168]},
            {"box": [0, 0, 224, 224], "point": [168, 224]},
        ]:
            result = evaluate_prediction(completion([obj]), gt, gt, prompt_size=(224, 224))
            self.assertFalse(result["format_valid"])
            self.assertEqual(result["total_reward"], 0.0)

    def test_prompt_dimensions_are_width_then_height(self):
        gt = np.ones((64, 128), dtype=np.uint16)
        text = completion([{"box": [140, 56, 224, 112], "point": [200, 100]}])
        self.assertEqual(evaluate_prediction(text, gt, gt, prompt_size=(224, 112))["total_reward"], 2.0)
        self.assertEqual(evaluate_prediction(text, gt, gt, prompt_size=(112, 224))["total_reward"], 0.0)

    def test_invalid_prompt_size_is_a_caller_error(self):
        gt = np.ones((3, 4), dtype=np.uint16)
        for size in [(0, 224), (224, -1), (224.0, 224), (True, 224),
                     (224,), (224, 224, 224), [224, 224], "224x224", (224, float("nan"))]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                evaluate_prediction(completion([ONE]), gt, gt, prompt_size=size)

    def test_valid_perfect_prediction(self):
        gt = np.ones((3, 4), dtype=np.uint16)
        result = evaluate_prediction(completion([ONE]), gt, gt)
        self.assertEqual(result["format_reward"], 1.0)
        self.assertEqual(result["seg_reward"], 1.0)
        self.assertEqual(result["total_reward"], 2.0)

    def test_invalid_never_earns_segmentation_reward(self):
        gt = np.ones((3, 4), dtype=np.uint16)
        result = evaluate_prediction("broken output", gt, gt)
        self.assertEqual(result["total_reward"], 0.0)
        self.assertEqual(result["pred_count"], 0)
        empty = np.zeros_like(gt)
        self.assertEqual(evaluate_prediction("broken", empty, empty)["total_reward"], 0.0)

    def test_valid_empty_is_format_only_for_nonempty_gt(self):
        gt = np.ones((3, 4), dtype=np.uint16)
        result = evaluate_prediction(completion([]), gt, gt)
        self.assertEqual(result["format_reward"], 1.0)
        self.assertEqual(result["seg_reward"], 0.0)
        self.assertEqual(result["total_reward"], 1.0)

    def test_weights_do_not_add_instance_metrics_to_reward(self):
        gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0]])
        result = evaluate_prediction(completion([ONE]), (gt > 0).astype(int), gt,
                                     seg_weight=0.7, format_weight=0.3)
        self.assertEqual(result["pq"], 0.0)
        self.assertEqual(result["total_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
