"""Raw-mask duplicate diagnostics must not be confused with final-mask PQ."""
import unittest

import numpy as np

from nucleus_rl.ablation_metrics import duplicate_metrics, summarize_records
from nucleus_rl.rewards import instance_metrics, masks_to_instances


class DuplicateMetricTests(unittest.TestCase):
    def test_overlap_resolution_can_hide_two_extra_predictions(self):
        reference = np.zeros((5, 6), dtype=np.uint16)
        reference[1:4, 2:5] = 300
        masks = np.stack([reference > 0] * 3)
        diagnostic = duplicate_metrics(masks, reference)
        self.assertEqual(diagnostic["same_gt_duplicate_count"], 2)
        self.assertEqual(diagnostic["raw_overlap_pair_count"], 3)
        self.assertEqual(diagnostic["raw_best_gt_ids"], [300, 300, 300])
        self.assertEqual(diagnostic["raw_best_gt_ious"], [1.0] * 3)
        final = instance_metrics(masks_to_instances(masks), reference)
        self.assertEqual(final["pq"], 1.0)
        self.assertEqual(final["pred_count"], 1)
        self.assertEqual(final["count_error"], 0)

    def test_same_gt_duplicates_can_have_low_mutual_iou(self):
        reference = np.array([[9, 9, 9]])
        masks = np.array([[[1, 1, 0]], [[0, 1, 1]]], dtype=bool)
        result = duplicate_metrics(masks, reference)
        self.assertEqual(result["same_gt_duplicate_count"], 1)
        self.assertEqual(result["raw_overlap_pair_count"], 0)
        np.testing.assert_allclose(result["raw_best_gt_ious"], [2 / 3, 2 / 3])

    def test_overlap_proxy_does_not_prove_a_same_gt_duplicate(self):
        reference = np.zeros((4, 4), dtype=int)
        reference[0, 0] = 8
        mask = np.zeros_like(reference, dtype=bool)
        mask[2:, 2:] = True
        result = duplicate_metrics(np.stack([mask, mask]), reference)
        self.assertEqual(result["raw_overlap_pair_count"], 1)
        self.assertEqual(result["same_gt_duplicate_count"], 0)
        self.assertEqual(result["raw_best_gt_ids"], [None, None])

    def test_exact_half_iou_is_not_a_match_or_overlap_pair(self):
        reference = np.array([[25, 25]])
        masks = np.array([[[1, 1]], [[1, 0]]])
        result = duplicate_metrics(masks, reference)
        self.assertEqual(result["raw_best_gt_ious"], [1.0, 0.5])
        self.assertEqual(result["raw_best_gt_ids"], [25, None])
        self.assertEqual(result["same_gt_duplicate_count"], 0)
        self.assertEqual(result["raw_overlap_pair_count"], 0)

    def test_empty_masks_and_empty_reference(self):
        reference = np.zeros((3, 4), dtype=int)
        for masks in ([], np.zeros((0, 3, 4), dtype=bool), np.zeros((2, 3, 4), dtype=bool)):
            with self.subTest(shape=np.asarray(masks).shape):
                result = duplicate_metrics(masks, reference)
                self.assertEqual(result["same_gt_duplicate_count"], 0)
                self.assertEqual(result["raw_overlap_pair_count"], 0)
                self.assertEqual(result["nonempty_raw_mask_count"], 0)
        result = duplicate_metrics(np.ones((2, 3, 4), dtype=bool), reference)
        self.assertEqual(result["raw_mask_count"], 2)
        self.assertEqual(result["nonempty_raw_mask_count"], 2)
        self.assertEqual(result["raw_overlap_pair_count"], 1)
        self.assertEqual(result["same_gt_duplicate_count"], 0)

    def test_instance_ids_are_preserved_not_connected_components(self):
        # Two disconnected regions have the same reference ID.
        reference = np.array([[7, 0, 7, 0, 40, 40]])
        masks = np.stack([reference == 7, reference == 40, reference == 7])
        original = duplicate_metrics(masks, reference)
        reordered = duplicate_metrics(masks[::-1], reference)
        self.assertEqual(original["raw_best_gt_ids"], [7, 40, 7])
        for key in ("same_gt_duplicate_count", "raw_overlap_pair_count"):
            self.assertEqual(original[key], 1)
            self.assertEqual(original[key], reordered[key])
        np.testing.assert_array_equal(reference, [[7, 0, 7, 0, 40, 40]])

    def test_invalid_inputs_rejected_instead_of_silently_thresholded(self):
        reference = np.zeros((2, 2), dtype=int)
        for masks in (np.zeros((2, 2)), np.zeros((1, 3, 2)),
                      np.full((1, 2, 2), .6), np.full((1, 2, 2), np.nan),
                      np.full((1, 2, 2), -1), np.full((1, 2, 2), "x")):
            with self.subTest(shape=masks.shape, dtype=masks.dtype):
                with self.assertRaises(ValueError):
                    duplicate_metrics(masks, reference)
        for invalid_reference in (np.full((2, 2), .5), np.full((2, 2), -1),
                                  np.full((2, 2), np.inf), np.zeros((0, 2))):
            with self.assertRaises(ValueError):
                duplicate_metrics([], invalid_reference)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(*, gt=1, pred=1, tp=1, pq=1.0, valid=True, duplicates=0, pairs=0):
        return {"gt_count": gt, "pred_count": pred, "tp": tp,
                "fp": pred - tp, "fn": gt - tp, "pq": pq,
                "recall": tp / gt if gt else 1.0, "count_error": abs(pred - gt),
                "format_valid": valid, "same_gt_duplicate_count": duplicates,
                "raw_overlap_pair_count": pairs}

    def test_macro_micro_and_invalid_completion_denominators(self):
        rows = [self.row(duplicates=1, pairs=1),
                self.row(gt=9, pred=2, tp=0, pq=0.0),
                self.row(gt=2, pred=0, tp=0, pq=0.0, valid=False)]
        result = summarize_records(iter(rows))
        self.assertEqual(result["n_records"], 3)
        self.assertEqual(result["macro_pq"], 1 / 3)
        self.assertEqual(result["macro_recall"], 1 / 3)
        self.assertEqual(result["micro_recall"], 1 / 12)
        self.assertEqual(result["count_mae"], 3.0)
        self.assertEqual(result["format_valid_rate"], 2 / 3)
        self.assertEqual(result["total_pred_count"], 3)
        self.assertEqual(result["total_gt_count"], 12)
        self.assertEqual(result["total_tp"], 1)
        self.assertEqual(result["total_fp"], 2)
        self.assertEqual(result["total_fn"], 11)
        self.assertEqual(result["total_same_gt_duplicate_count"], 1)
        self.assertEqual(result["total_raw_overlap_pair_count"], 1)

    def test_empty_records_are_distinct_from_observed_empty_gt(self):
        empty = summarize_records([])
        self.assertEqual(empty["n_records"], 0)
        self.assertEqual(empty["total_gt_count"], 0)
        for key in ("macro_pq", "macro_recall", "micro_recall", "count_mae", "format_valid_rate"):
            self.assertIsNone(empty[key])
        observed = summarize_records([self.row(gt=0, pred=2, tp=0, pq=0.0)])
        self.assertEqual(observed["macro_recall"], 1.0)
        self.assertEqual(observed["micro_recall"], 1.0)
        self.assertEqual(observed["total_fp"], 2)
        self.assertEqual(observed["count_mae"], 2.0)

    def test_missing_or_inconsistent_metrics_are_not_silently_aggregated(self):
        missing = self.row()
        del missing["raw_overlap_pair_count"]
        with self.assertRaises(KeyError):
            summarize_records([missing])
        for change in ({"tp": 2}, {"count_error": 1}, {"recall": .5},
                       {"pq": float("nan")}, {"format_valid": "false"},
                       {"same_gt_duplicate_count": True}, {"gt_count": -1}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    summarize_records([{**self.row(), **change}])


if __name__ == "__main__":
    unittest.main()
