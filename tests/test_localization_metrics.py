"""Geometry, assignment and association controls without models or SciPy."""
import json
import math
import unittest

import numpy as np

from nucleus_rl.localization_metrics import _match_box_iou, analyze, geometry
from nucleus_rl.rewards import ObjectPrompt


def paired(boxes, points=None):
    """Original 128-pixel prompts and their unchanged 224-pixel coordinates."""
    if points is None:
        points = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    original = [ObjectPrompt(tuple(box), tuple(point)) for box, point in zip(boxes, points)]
    raw = [ObjectPrompt(tuple(v * 1.75 for v in obj.box), tuple(v * 1.75 for v in obj.point)) for obj in original]
    return raw, original


class LocalizationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((128, 128), dtype=np.uint16)
        self.mask[10:20, 10:20] = 7
        self.mask[50:60, 50:60] = 300

    def test_geometry_sorted_instance_ids_exclusive_boxes_and_centroids(self):
        mask = self.mask.copy()
        mask[22, 22] = 7  # Same ID is one instance despite disconnected pixels.
        output = geometry(mask)
        self.assertEqual((output["gt_count"], output["original_size"], output["processed_size"]), (2, [128, 128], [224, 224]))
        first, second = output["instances"]
        self.assertEqual((first["display_id"], first["instance_id"], first["area"]), ("G1", 7, 101))
        self.assertEqual(first["original_box"], [10, 10, 23, 23])
        self.assertEqual(first["processed_box"], [17.5, 17.5, 40.25, 40.25])
        self.assertAlmostEqual(first["centroid_original"][0], (100 * 14.5 + 22) / 101)
        self.assertEqual(second["instance_id"], 300)
        rectangular = geometry(self.mask, processed_size=(224, 112))["instances"][1]
        self.assertEqual(rectangular["processed_box"], [87.5, 43.75, 105., 52.5])

    def test_exact_multi_match_preserves_prediction_order_and_identities(self):
        raw, original = paired([[50, 50, 60, 60], [10, 10, 20, 20]])
        output = analyze(raw, original, self.mask)
        self.assertEqual(output["box_iou_matrix"], [[0., 1.], [1., 0.]])
        self.assertEqual(output["box_matches"], [{"prediction_id": "P1", "gt_id": "G2", "box_iou": 1.},
                                                 {"prediction_id": "P2", "gt_id": "G1", "box_iou": 1.}])
        self.assertEqual((output["box_recall"], output["absolute_count_error"]), (1., 0))
        self.assertEqual(output["matched_gt_ids"], ["G1", "G2"])
        self.assertEqual(output["predictions"][0]["point_gt_instance_id"], 300)
        self.assertEqual(output["predictions"][0]["edge_errors_processed_xyxy"], [0., 0., 0., 0.])
        self.assertEqual(output["translation_diagnostic"]["mean_offset_processed_xy"], [0., 0.])
        json.dumps(output, allow_nan=False)

    def test_duplicates_and_miss_can_have_zero_count_error(self):
        raw, original = paired([[10, 10, 20, 20], [10, 10, 20, 20]])
        output = analyze(raw, original, self.mask)
        self.assertEqual(output["absolute_count_error"], 0)
        self.assertEqual(output["box_recall"], .5)
        self.assertEqual(output["box_true_positive_count"], 1)
        self.assertEqual(output["missed_gt_ids"], ["G2"])
        self.assertEqual(output["unmatched_prediction_ids"], ["P2"])
        self.assertEqual(output["best_box_duplicate_count"], 1)
        self.assertEqual(output["point_duplicate_count"], 1)
        self.assertEqual(output["predicted_box_overlap_pair_count"], 1)
        self.assertEqual(output["translation_diagnostic"]["association_count"], 0)
        self.assertIsNone(output["translation_diagnostic"]["mean_offset_processed_xy"])

    def test_point_inside_box_can_be_outside_nucleus_and_uses_floor(self):
        mask = np.zeros((128, 128), dtype=np.uint16)
        mask[10:20, 10:12] = 10
        mask[10:12, 10:20] = 10
        raw, original = paired([[10, 10, 20, 20]], [(18.9, 18.1)])
        output = analyze(raw, original, mask)
        self.assertEqual(output["box_recall"], 1.)
        pred = output["predictions"][0]
        self.assertEqual(pred["point_floor_xy"], [18, 18])
        self.assertEqual(pred["point_status"], "background")
        self.assertIsNone(pred["point_gt_id"])
        # Background membership does not assert the point is far from a nucleus.
        raw, original = paired([[10, 10, 20, 20]], [(11.99, 18.9)])
        pred = analyze(raw, original, mask)["predictions"][0]
        self.assertEqual((pred["point_floor_xy"], pred["point_gt_id"]), ([11, 18], "G1"))

    def test_matching_maximizes_cardinality_before_total_iou(self):
        # Highest-IoU-first would take P1/G1 and lose the second feasible match.
        self.assertEqual(_match_box_iou([[.9, .51], [.85, 0]]), [(0, 1), (1, 0)])
        # Equal cardinality: off-diagonal sum 1.50 exceeds diagonal sum 1.42.
        self.assertEqual(_match_box_iou([[.91, .8], [.7, .51]]), [(0, 1), (1, 0)])
        self.assertEqual(_match_box_iou([[.75, .75], [.75, .75]]), [(0, 0), (1, 1)])
        self.assertEqual(_match_box_iou([[.5, 0], [0, .50001]]), [(1, 1)])

    def test_zero_overlap_never_forces_a_gt_association_or_offset(self):
        raw, original = paired([[90, 90, 100, 100]])
        output = analyze(raw, original, self.mask)
        pred = output["predictions"][0]
        self.assertEqual(pred["best_box_iou"], 0.)
        self.assertIsNone(pred["best_box_gt_id"])
        self.assertIsNone(pred["center_offset_processed_xy"])
        self.assertIsNone(pred["edge_errors_processed_xyxy"])
        self.assertEqual(output["box_recall"], 0.)
        self.assertEqual(output["translation_diagnostic"]["association_count"], 0)

    def test_shared_offset_is_diagnostic_not_a_true_positive_match(self):
        raw, original = paired([[18, 10, 28, 20], [58, 50, 68, 60]])
        output = analyze(raw, original, self.mask)
        self.assertEqual(output["box_true_positive_count"], 0)
        diagnostic = output["translation_diagnostic"]
        self.assertEqual(diagnostic["association_count"], 2)
        self.assertEqual(diagnostic["mean_offset_processed_xy"], [14., 0.])
        self.assertEqual(diagnostic["rms_residual_processed"], 0.)
        self.assertTrue(diagnostic["diagnostic_associations_not_true_positive_matches"])
        self.assertTrue(diagnostic["does_not_establish_global_shift"])
        self.assertLess(diagnostic["associations"][0]["box_iou"], .5)

    def test_rms_residual_and_single_association_are_not_overinterpreted(self):
        raw, original = paired([[14, 12, 24, 22], [52, 54, 62, 64]])
        diagnostic = analyze(raw, original, self.mask)["translation_diagnostic"]
        self.assertEqual(diagnostic["mean_offset_processed_xy"], [5.25, 5.25])
        self.assertEqual(diagnostic["rms_residual_processed_xy"], [1.75, 1.75])
        self.assertAlmostEqual(diagnostic["rms_residual_processed"], math.sqrt(2) * 1.75)
        single = analyze(raw[:1], original[:1], self.mask)["translation_diagnostic"]
        self.assertEqual(single["association_count"], 1)
        self.assertIsNone(single["mean_offset_processed_xy"])
        self.assertIsNone(single["rms_residual_processed"])

    def test_tied_best_gt_boxes_are_excluded_from_translation_diagnostic(self):
        mask = self.mask.copy()
        # Disjoint instance masks may have identical enclosing boxes.
        ys, xs = np.indices((10, 10))
        mask[10:20, 10:20] = np.where((ys + xs) % 2, 7, 8)
        raw, original = paired([[10, 10, 20, 20], [50, 50, 60, 60]])
        output = analyze(raw, original, mask)
        self.assertTrue(output["predictions"][0]["best_box_iou_tied"])
        diagnostic = output["translation_diagnostic"]
        self.assertEqual(diagnostic["association_count"], 1)
        self.assertEqual(diagnostic["associations"][0]["gt_id"], "G3")
        self.assertIsNone(diagnostic["mean_offset_processed_xy"])

    def test_empty_outputs_and_empty_gt_conventions(self):
        output = analyze([], [], self.mask)
        self.assertEqual(output["predictions"], [])
        self.assertEqual(output["box_iou_matrix"], [])
        self.assertEqual(output["box_iou_matrix_gt_ids"], ["G1", "G2"])
        self.assertEqual((output["box_recall"], output["absolute_count_error"]), (0., 2))
        self.assertEqual(output["missed_gt_ids"], ["G1", "G2"])
        empty_gt = np.zeros((128, 128), dtype=np.uint16)
        self.assertEqual(analyze([], [], empty_gt)["box_recall"], 1.)
        raw, original = paired([[10, 10, 20, 20]])
        prediction_without_gt = analyze(raw, original, empty_gt)
        self.assertEqual(prediction_without_gt["box_iou_matrix"], [[]])
        self.assertEqual(prediction_without_gt["unmatched_prediction_ids"], ["P1"])
        self.assertEqual(prediction_without_gt["absolute_count_error"], 1)

    def test_optional_points_are_distinguished_from_background(self):
        output = analyze([{"box": [17.5, 17.5, 35., 35.]}], [{"box": [10, 10, 20, 20]}], self.mask)
        pred = output["predictions"][0]
        self.assertEqual(pred["point_status"], "not_provided")
        self.assertIsNone(pred["point_floor_xy"])
        self.assertEqual(output["box_true_positive_count"], 1)

    def test_frame_changes_invalid_coordinates_and_bad_masks_are_rejected(self):
        raw, original = paired([[10, 10, 20, 20]])
        with self.assertRaisesRegex(ValueError, "fixed-scale conversion"):
            analyze(raw, [ObjectPrompt((11, 10, 20, 20), (15, 15))], self.mask)
        with self.assertRaises(ValueError):
            analyze([{"box": [True, 10, 20, 20]}], original, self.mask)
        with self.assertRaises(ValueError):
            analyze([{"box": [10, 10, 20, 20], "point": [20, 15]}], original, self.mask)
        for mask in (np.array([[np.nan]]), np.array([[-1]]), np.array([[1.5]])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                geometry(mask)


if __name__ == "__main__":
    unittest.main()
