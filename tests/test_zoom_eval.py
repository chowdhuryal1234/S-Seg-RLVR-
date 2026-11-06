"""CPU tests for the 2x2 zoom probe geometry, merge rule and analysis (no model, no real SAM)."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.evaluate import ArtifactScorer
from nucleus_rl.rewards import ObjectPrompt, masks_to_instances, parse_completion
from unittest import mock

import scripts.zoom_eval as zoom_eval
from scripts.zoom_eval import (
    analysis, answer_text, box_iou, duplicates, gt_instances, interior_border_distance, merge_objects,
    per_gt_status, subcrop_offsets, subcrop_windows, to_tile,
)


def _obj(box, point, sub, index=0, window=None):
    window = window or subcrop_windows(128, 128, 80)[sub]
    return {"box": box, "point": point, "sub": sub, "index": index,
            "border_distance": interior_border_distance(box, window, (128, 128))}


class Geometry(unittest.TestCase):
    def test_offsets_and_windows(self):
        self.assertEqual(subcrop_offsets(128, 80), [0, 48])
        windows = subcrop_windows(128, 128, 80)
        self.assertEqual(windows, [(0, 0, 80, 80), (48, 0, 128, 80), (0, 48, 80, 128), (48, 48, 128, 128)])
        # 32 px overlap between neighbouring windows; together they cover the tile.
        self.assertEqual(windows[0][2] - windows[1][0], 32)
        covered = np.zeros((128, 128), bool)
        for x0, y0, x1, y1 in windows:
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all())
        self.assertEqual(subcrop_offsets(128, 128), [0])
        with self.assertRaises(ValueError):
            subcrop_offsets(128, 129)

    def test_normalized_to_tile(self):
        window = (48, 48, 128, 128)
        obj = ObjectPrompt(box=(0., 500., 1000., 1000.), point=(250., 750.))
        mapped = to_tile(obj, window, (1000, 1000))
        self.assertEqual(mapped["box"], [48., 88., 128., 128.])
        self.assertEqual(mapped["point"], [68., 108.])
        # Processed-frame prompts (448 px) map the same way.
        mapped = to_tile(ObjectPrompt(box=(0., 0., 448., 224.), point=(224., 112.)), (0, 48, 80, 128), (448, 448))
        self.assertEqual(mapped["box"], [0., 48., 80., 88.])
        self.assertEqual(mapped["point"], [40., 68.])

    def test_interior_border_distance(self):
        # Window 0 = (0,0,80,80): only its right (x=80) and bottom (y=80) borders cut the tile.
        self.assertEqual(interior_border_distance([0, 0, 10, 10], (0, 0, 80, 80), (128, 128)), 75)
        self.assertEqual(interior_border_distance([60, 30, 76, 40], (0, 0, 80, 80), (128, 128)), 12)
        # Window 3 = (48,48,128,128): left/top cut the tile.
        self.assertEqual(interior_border_distance([50, 100, 54, 110], (48, 48, 128, 128), (128, 128)), 4)
        self.assertTrue(math.isinf(interior_border_distance([0, 0, 10, 10], (0, 0, 128, 128), (128, 128))))


class Merge(unittest.TestCase):
    def test_iou_duplicate_keeps_more_central_copy(self):
        # Nucleus at x 70..84 lies in the overlap of windows 0 and 1. Window 0's copy is cut near x=80
        # (centre 4 px from its border); window 1's copy is 29 px from its left border x=48.
        near_border = _obj([70, 10, 80, 20], [75, 15], sub=0)
        central = _obj([70, 10, 84, 20], [77, 15], sub=1)
        self.assertGreater(box_iou(near_border["box"], central["box"]), 0.5)
        kept, dropped = merge_objects([near_border, central])
        self.assertEqual([(k["sub"], k["box"]) for k in kept], [(1, [70, 10, 84, 20])])
        self.assertEqual(dropped[0]["duplicate_of"], [1, 0])

    def test_point_in_box_duplicate(self):
        # Sliver from window 0 (IoU < 0.5 with the full box) but its point falls inside window 1's full box.
        sliver = _obj([76, 40, 80, 50], [78, 45], sub=0)
        full = _obj([70, 38, 90, 54], [80, 46], sub=1)
        self.assertLess(box_iou(sliver["box"], full["box"]), 0.5)
        self.assertTrue(duplicates(sliver, full))
        kept, dropped = merge_objects([sliver, full])
        self.assertEqual([k["sub"] for k in kept], [1])
        self.assertEqual(len(dropped), 1)

    def test_same_subcrop_objects_never_merged(self):
        a = _obj([10, 10, 30, 30], [20, 20], sub=0, index=0)
        b = _obj([12, 12, 30, 30], [21, 21], sub=0, index=1)
        kept, dropped = merge_objects([a, b])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_distinct_nuclei_kept_and_cap_32(self):
        objects = []
        windows = subcrop_windows(128, 128, 80)
        for sub in range(len(windows)):
            for index in range(8):
                # Separate rows per sub-crop (inside its window) so no two objects overlap.
                by = [2, 12, 100, 110][sub]
                bx = [2, 50, 2, 50][sub] + 9 * index
                objects.append(_obj([bx, by, bx + 6, by + 6], [bx + 3, by + 3], sub, index, windows[sub]))
        kept, dropped = merge_objects(objects)
        self.assertEqual(len(kept) + len(dropped), 32)
        text = answer_text(kept)
        parsed = parse_completion(text, width=128, height=128, max_objects=32)
        self.assertTrue(parsed.valid, parsed.error)
        self.assertEqual(len(parsed.objects), len(kept))

    def test_answer_text_roundtrips_floats(self):
        obj = _obj([48.08, 3.84, 60.0, 17.6], [51.2, 10.24], sub=1)
        parsed = parse_completion(answer_text([obj]), width=128, height=128, max_objects=32)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.objects[0].box, (48.08, 3.84, 60.0, 17.6))


def _squares(size=128):
    labels = np.zeros((size, size), dtype=np.uint16)
    labels[10:20, 10:20] = 5      # found by both
    labels[60:70, 60:70] = 9      # missed by plain (no box near), found by zoom
    labels[0:6, 120:128] = 11     # border sliver, missed by both
    return labels


class BoxSegmenter:
    """Fake SAM: fills each prompt box."""
    def predict(self, image, objects, image_key=None):
        width, height = image.size
        masks = np.zeros((len(objects), height, width), dtype=bool)
        for index, obj in enumerate(objects):
            x0, y0, x1, y1 = (int(round(v)) for v in obj.box)
            masks[index, y0:y1, x0:x1] = True
        self.last_raw_masks = masks
        return masks_to_instances(masks) if len(objects) else np.zeros((height, width), np.int32)


class Analysis(unittest.TestCase):
    def test_per_gt_status(self):
        labels = _squares()
        instances = np.zeros_like(labels, dtype=np.int32)
        instances[10:20, 10:20] = 1
        rows = per_gt_status(labels, [[10, 10, 20, 20]], instances)
        by_id = {r["gt_id"]: r for r in rows}
        self.assertTrue(by_id[5]["matched"])
        self.assertAlmostEqual(by_id[5]["best_box_iou"], 1.0)
        self.assertFalse(by_id[9]["matched"])
        self.assertEqual(by_id[9]["best_box_iou"], 0.0)
        self.assertTrue(by_id[11]["touches_border"])
        self.assertEqual({g["gt_id"] for g in gt_instances(labels)}, {5, 9, 11})

    def test_end_to_end_analysis_with_fake_segmenter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(tmp / "img.png")
            Image.fromarray(_squares()).save(tmp / "mask.png")
            row = {"id": "t0", "image_path": str(tmp / "img.png"), "mask_path": str(tmp / "mask.png"),
                   "patient_id": "p", "split": "validation", "dataset": "monuseg"}
            sam = BoxSegmenter()
            run = tmp / "run"
            plain_text = '<answer>{"objects":[{"box":[78,78,157,157],"point":[117,117]}]}</answer>'
            plain_row = {**row, "coordinate_frame": "normalized1000", "original_width": 128, "original_height": 128,
                         "processed_width": 448, "processed_height": 448, "prompt_width": 1000, "prompt_height": 1000}
            plain = ArtifactScorer(sam, run / "plain", 8)
            plain.score(plain_text, plain_row)
            zoom_objects = [_obj([10, 10, 20, 20], [15, 15], 0), _obj([60, 60, 70, 70], [65, 65], 3),
                            _obj([30, 100, 40, 110], [35, 105], 2)]  # last one is a false positive
            kept, _ = merge_objects(zoom_objects)
            zoom = ArtifactScorer(sam, run / "zoom", 32)
            zoom.score(answer_text(kept), {**row, "coordinate_frame": "original"})
            result = analysis(run, [row])
            pooled = result["pooled"]
            self.assertEqual((pooled["plain"]["tp"], pooled["plain"]["fp"], pooled["plain"]["fn"]), (1, 0, 2))
            self.assertEqual((pooled["zoom"]["tp"], pooled["zoom"]["fp"], pooled["zoom"]["fn"]), (2, 1, 1))
            self.assertEqual(pooled["no_box"]["count"], 2)       # ids 9 and 11
            self.assertEqual(pooled["no_box"]["plain_fn"], 2)
            self.assertEqual(pooled["no_box"]["zoom_fn"], 1)
            self.assertAlmostEqual(pooled["no_box"]["relative_fn_drop"], 0.5)
            self.assertEqual(pooled["flips"], {"gained": 1, "lost": 0})
            self.assertFalse(result["verdict"]["tp_gain_ge_15"])
            self.assertFalse(result["verdict"]["pass"])
            self.assertIn("monuseg", result["per_dataset"])
            self.assertTrue((run / "RESULTS.md").read_text().startswith("# Zoom probe results"))
            json.loads((run / "results.json").read_text())


class FakeImageProcessor:
    patch_size = 16

    def __call__(self, images, return_tensors=None):
        return {"image_grid_thw": [[1, 28, 28]]}   # 448 x 448, as min=max=200704 gives every crop


class FakeModel:
    def modules(self):
        return []

    def eval(self):
        return self


class ZoomLoop(unittest.TestCase):
    def test_zoom_evaluate_maps_merges_and_scores(self):
        """Canned sub-crop answers (normalized1000) -> tile prompts -> one SAM call -> scorer."""
        answers = {
            # Window 0 (0,0,80,80): nucleus 5 at 10..20 -> normalized 125..250.
            (0, 0): '<answer>{"objects":[{"box":[125,125,250,250],"point":[187,187]}]}</answer>',
            # Window 1 (48,0,128,80): nothing.
            (48, 0): '<answer>{"objects":[]}</answer>',
            # Window 2 (0,48,80,128): nucleus 9 at 60..70 -> x 750..875, y (12..22)/80 -> 150..275.
            (0, 48): '<answer>{"objects":[{"box":[750,150,875,275],"point":[812,212]}]}</answer>',
            # Window 3 (48,48,128,128): the same nucleus 9 again (x 12..22 -> 150..275).
            (48, 48): '<answer>{"objects":[{"box":[150,150,275,275],"point":[212,212]}]}</answer>',
        }
        calls = []

        def fake_generate(model, processor, image, args):
            calls.append(image.size)
            key = next(k for k in answers if k not in [c for c in seen])
            seen.append(key)
            return answers[key], (1000, 1000), (448, 448), 10
        seen = []
        processor = type("P", (), {"image_processor": FakeImageProcessor()})()
        args = type("A", (), {"merged_max_objects": 32, "sub_size": 80, "max_objects": 8})()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(zoom_eval, "generate_answer", fake_generate):
            tmp = Path(tmp)
            Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(tmp / "img.png")
            Image.fromarray(_squares()).save(tmp / "mask.png")
            row = {"id": "t0", "image_path": str(tmp / "img.png"), "mask_path": str(tmp / "mask.png"),
                   "patient_id": "p", "split": "validation", "dataset": "monuseg"}
            summary = zoom_eval.zoom_evaluate(FakeModel(), processor, [row], BoxSegmenter(), tmp / "zoom", args)
            self.assertEqual(calls, [(80, 80)] * 4)
            record = json.loads((tmp / "zoom" / "completions.jsonl").read_text())
            detail = json.loads((tmp / "zoom" / "zoom_details.jsonl").read_text())
            self.assertEqual(len(detail["all_objects"]), 3)
            self.assertEqual(len(detail["kept"]), 2)            # nucleus 9 seen twice -> merged
            self.assertEqual((record["tp"], record["fp"], record["fn"]), (2, 0, 1))
            self.assertEqual(summary["completions"], 1)
            boxes = sorted(o["box"] for o in record["converted_objects"])
            self.assertEqual(boxes, [[10.0, 10.0, 20.0, 20.0], [60.0, 60.0, 70.0, 70.0]])


if __name__ == "__main__":
    unittest.main()
