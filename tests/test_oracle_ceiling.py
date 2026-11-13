"""CPU-only tests for the GT-prompt oracle ceiling, using a fake segmenter (never real SAM)."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.rewards import masks_to_instances
from nucleus_rl.sft_data import instance_prompt_objects
from scripts.oracle_ceiling import (
    aggregate, build_parser, cap_label, check_rows, markdown_table, oracle_objects,
    load_mask, parse_cap, parse_manifest_spec, run_manifest, write_summary,
)


def _labels(count, size=32):
    """``count`` 2x2 squares on a grid; IDs deliberately not in spatial order."""
    labels = np.zeros((size, size), dtype=np.uint16)
    for index in range(count):
        y, x = 3 * (index // 8), 3 * (index % 8)
        labels[y:y + 2, x:x + 2] = 100 - index
    return labels


class BoxSegmenter:
    """Fake SAM: fills each prompt box, so tight GT boxes on squares give perfect masks."""

    def __init__(self):
        self.calls = []

    def predict(self, image, objects, image_key=None):
        width, height = image.size
        self.calls.append((image_key, len(objects)))
        masks = np.zeros((len(objects), height, width), dtype=bool)
        for index, obj in enumerate(objects):
            x0, y0, x1, y1 = (int(v) for v in obj.box)
            masks[index, y0:y1, x0:x1] = True
        if not objects:
            return np.zeros((height, width), dtype=np.int32)
        return masks_to_instances(masks)


class ParsingTests(unittest.TestCase):
    def test_parse_cap(self):
        self.assertIsNone(parse_cap("all"))
        self.assertIsNone(parse_cap("None"))
        self.assertEqual(parse_cap("8"), 8)
        for bad in ("0", "-1", "x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_cap(bad)
        self.assertEqual(cap_label(None), "all")
        self.assertEqual(cap_label(8), "cap8")

    def test_manifest_spec_and_parser(self):
        self.assertEqual(parse_manifest_spec("a=x/y.jsonl"), ("a", Path("x/y.jsonl")))
        self.assertEqual(parse_manifest_spec("d/consep_v1/validation.jsonl")[0], "consep_v1/validation")
        args = build_parser().parse_args(["--manifest", "a=m.jsonl", "--manifest", "n.jsonl",
                                          "--out", "o", "--max-objects", "all", "8"])
        self.assertEqual(args.max_objects, [None, 8])
        self.assertEqual(len(args.manifest), 2)
        self.assertEqual(build_parser().parse_args(["--manifest", "m", "--out", "o"]).max_objects, [None])

    def test_check_rows_rejects_test_split(self):
        check_rows([{"split": "train"}, {"split": "validation"}])
        with self.assertRaises(ValueError):
            check_rows([{"split": "validation"}, {"split": "test"}])


class ObjectTests(unittest.TestCase):
    def test_all_objects_match_sft_targets_beyond_eight(self):
        labels = _labels(12)
        objects = oracle_objects(labels)
        self.assertEqual(len(objects), 12)
        self.assertEqual(objects, instance_prompt_objects(labels, max_objects=12))
        for obj in objects:
            x, y = obj["point"]
            self.assertGreater(labels[y, x], 0)

    def test_cap_takes_first_objects_in_target_order(self):
        labels = _labels(12)
        self.assertEqual(oracle_objects(labels, 8), oracle_objects(labels)[:8])
        self.assertEqual(oracle_objects(labels, 50), oracle_objects(labels))

    def test_empty_mask_gives_no_objects(self):
        self.assertEqual(oracle_objects(np.zeros((4, 4), dtype=np.uint8)), [])


class RunTests(unittest.TestCase):
    def _write_rows(self, root, counts):
        rows = []
        for index, count in enumerate(counts):
            labels = _labels(count)
            image_path, mask_path = root / f"i{index}.png", root / f"m{index}.png"
            Image.fromarray(np.full(labels.shape + (3,), 128, dtype=np.uint8)).save(image_path)
            Image.fromarray(labels).save(mask_path)
            rows.append({"id": f"t{index}", "image_path": str(image_path), "mask_path": str(mask_path),
                         "split": "validation"})
        return rows

    def test_perfect_masks_and_cap_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self._write_rows(root, [3, 12])
            segmenter = BoxSegmenter()
            results = run_manifest("toy/dense", rows, segmenter, [None, 8], root)
            full, capped = results["toy/dense|all"], results["toy/dense|cap8"]
            self.assertEqual((full["tp"], full["fp"], full["fn"], full["gt"]), (15, 0, 0, 15))
            self.assertAlmostEqual(full["pq"], 1.0)
            self.assertEqual((capped["tp"], capped["fp"], capped["fn"], capped["prompted"]), (11, 0, 4, 11))
            self.assertAlmostEqual(capped["pq"], (1.0 + 8 / 10) / 2)
            self.assertEqual(segmenter.calls, [(rows[0]["image_path"], 3), (rows[0]["image_path"], 3),
                                               (rows[1]["image_path"], 12), (rows[1]["image_path"], 8)])
            lines = (root / "toy__dense__cap8.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["id"] for line in lines], ["t0", "t1"])

    def test_load_mask_npy_and_count_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _labels(5).astype(np.int32)
            path = Path(tmp) / "m.npy"
            np.save(path, labels)
            row = {"id": "t", "mask_path": str(path), "instance_count": 5}
            np.testing.assert_array_equal(load_mask(row), labels)
            with self.assertRaises(ValueError):
                load_mask({**row, "instance_count": 4})

    def test_aggregate_macro_and_pooled(self):
        records = [
            {"pq": 1.0, "sq": 1.0, "dq": 1.0, "tp": 1, "fp": 0, "fn": 0, "gt_count": 1,
             "prompted": 1, "count_error": 0, "matched_iou_sum": 1.0},
            {"pq": 0.0, "sq": 0.0, "dq": 0.0, "tp": 0, "fp": 1, "fn": 3, "gt_count": 3,
             "prompted": 1, "count_error": 2, "matched_iou_sum": 0.0},
        ]
        result = aggregate(records)
        self.assertAlmostEqual(result["pq"], 0.5)
        self.assertAlmostEqual(result["pooled_pq"], 1.0 / 3.0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 3))
        self.assertAlmostEqual(result["count_mae"], 1.0)

    def test_summary_merges_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = {"set": "a", "cap": "all", "tiles": 1, "gt": 1, "pq": 0.5, "sq": 0.5, "dq": 1.0,
                   "tp": 1, "fp": 0, "fn": 0, "recall": 1.0, "count_mae": 0.0, "pooled_pq": 0.5}
            write_summary(out, {"a|all": row}, {"run": 1})
            summary = write_summary(out, {"b|all": {**row, "set": "b"}}, {"run": 2})
            self.assertEqual(sorted(summary["results"]), ["a|all", "b|all"])
            self.assertEqual(len(summary["runs"]), 2)
            table = (out / "summary.md").read_text()
            self.assertIn("| a | all |", table)
            self.assertIn("| b | all |", table)
            self.assertEqual(table, markdown_table(summary["results"]))


if __name__ == "__main__":
    unittest.main()
