"""CPU checks of the sampled-support evaluation: GT boxes, one-to-one matching, never-found accounting."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from support_eval import aggregate, box_iou, build_parser, match, tile_support  # noqa: E402


class SupportEvalTests(unittest.TestCase):
    def test_box_iou_and_one_to_one_matching(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        # two predictions on the same nucleus count once; IoU exactly 0.5 does not count
        self.assertEqual(match([[0, 0, 10, 10], [1, 0, 10, 10]], gt), {0})
        self.assertEqual(match([[0, 0, 10, 10], [20, 20, 30, 30]], gt), {0, 1})
        self.assertEqual(match([[0, 0, 10, 10]], [[0, 0, 10, 20]]), set())  # IoU 0.5 is not > 0.5
        self.assertEqual(match([], gt), set())

    def test_tile_support_counts_never_found_and_slivers(self):
        gt = [{"sliver": False}, {"sliver": True}, {"sliver": False}]
        record = tile_support(gt, [{0}, {0, 2}, set()])
        self.assertEqual(record["never"], 1)            # GT 1 (a sliver) is never hit
        self.assertEqual(record["never_no_sliver"], 0)
        self.assertEqual(record["all_found"], 0)        # GT 0 missed by the invalid third sample
        self.assertEqual(record["hit_counts"], [2, 0, 1])
        self.assertEqual(record["matched_per_sample"], [1, 2, 0])
        with self.assertRaises(ValueError):
            tile_support(gt, [])

    def test_aggregate_pools_and_splits_by_dataset(self):
        a = {"dataset": "puma", **tile_support([{"sliver": False}] * 2, [{0}, {0, 1}])}
        b = {"dataset": "consep", **tile_support([{"sliver": False}, {"sliver": True}], [set(), set()])}
        out = aggregate([a, b])
        self.assertEqual(out["ALL"]["gt"], 4)
        self.assertAlmostEqual(out["ALL"]["per_sample_recall"], (1.5 + 0) / 4)
        self.assertAlmostEqual(out["ALL"]["never_found"], 2 / 4)
        self.assertAlmostEqual(out["ALL"]["never_found_no_sliver"], 1 / 3)
        self.assertAlmostEqual(out["puma"]["union_recall"], 1.0)
        self.assertAlmostEqual(out["puma"]["all_found"], 1 / 2)
        self.assertEqual(set(out), {"ALL", "puma", "consep"})

    def test_parser_defaults_match_grpo_sampling(self):
        args = build_parser().parse_args(["--eval-manifest", "v", "--output", "o", "--model", "m", "--revision", "r",
                                          "--init-adapter", "a", "--min-pixels", "1", "--max-pixels", "1",
                                          "--max-objects", "176"])
        self.assertEqual((args.samples, args.temperature, args.iou_threshold), (8, 1.0, 0.5))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required")
    def test_gt_boxes_are_half_open_and_sliver_flag_follows_manifest(self):
        import numpy as np
        from support_eval import gt_instances
        mask = np.zeros((6, 6), dtype=np.uint16)
        mask[1:3, 2:5] = 7
        mask[5, 0] = 9
        items = {item["gid"]: item for item in gt_instances(mask, sliver_ids=[9])}
        self.assertEqual(items[7]["box"], [2, 1, 5, 3])
        self.assertEqual(items[7]["area"], 6)
        self.assertTrue(items[9]["sliver"])
        self.assertFalse(items[7]["sliver"])
        # without the manifest field: sliver = truncated and < 100 px
        items = {item["gid"]: item for item in gt_instances(mask, None, truncated_ids=[7])}
        self.assertTrue(items[7]["sliver"])
        self.assertFalse(items[9]["sliver"])


if __name__ == "__main__":
    unittest.main()
