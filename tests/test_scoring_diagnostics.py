import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.evaluate import ArtifactScorer


class RawMaskDiagnosticsTests(unittest.TestCase):
    def test_duplicates_survive_overlap_resolution_without_changing_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = np.zeros((8, 8), dtype=np.uint8)
            gt[2:6, 2:6] = 1
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "image.png")
            Image.fromarray(gt).save(root / "mask.png")
            row = {"id": "toy", "patient_id": "toy", "image_path": str(root / "image.png"),
                   "mask_path": str(root / "mask.png")}

            class Segmenter:
                def predict(self, image, objects, image_key=None):
                    self.last_raw_masks = np.stack([gt > 0, gt > 0])
                    return gt.copy()

            scorer = ArtifactScorer(Segmenter(), root / "output")
            obj = {"box": [2, 2, 6, 6], "point": [3, 3]}
            answer = "<answer>" + json.dumps({"objects": [obj, obj]}) + "</answer>"
            valid = scorer.score(answer, row)
            self.assertEqual(valid["pred_count"], 1)
            self.assertEqual(valid["same_gt_duplicate_count"], 1)
            self.assertEqual(valid["raw_overlap_pair_count"], 1)
            self.assertEqual(valid["total_reward"], 2.0)
            with np.load(root / "output/000001_raw_masks.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["masks"].shape, (2, 8, 8))
            invalid = scorer.score("invalid", row)
            self.assertEqual(invalid["raw_mask_count"], 0)
            self.assertEqual(invalid["same_gt_duplicate_count"], 0)
            self.assertEqual(invalid["total_reward"], 0.0)
            summary = scorer.summary()["instance_diagnostics"]
            self.assertEqual(summary["macro_pq"], 0.5)
            self.assertEqual(summary["micro_recall"], 0.5)



class AnswerCapAccountingTests(unittest.TestCase):
    def test_gt_beyond_the_answer_cap_is_scored_as_fn_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = np.zeros((16, 16), dtype=np.uint8)
            for index in range(4):  # four 2x2 nuclei
                gt[4 * index:4 * index + 2, 0:2] = index + 1
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / "image.png")
            Image.fromarray(gt).save(root / "mask.png")
            row = {"id": "toy", "patient_id": "toy", "image_path": str(root / "image.png"),
                   "mask_path": str(root / "mask.png")}

            class Segmenter:  # returns exactly the first two GT nuclei
                def predict(self, image, objects, image_key=None):
                    out = np.where(gt <= 2, gt, 0).astype(np.int32)
                    self.last_raw_masks = np.stack([out == 1, out == 2])
                    return out

            scorer = ArtifactScorer(Segmenter(), root / "output", max_objects=2)
            objects = [{"box": [0, 0, 2, 2], "point": [0, 0]}, {"box": [0, 4, 2, 6], "point": [0, 4]}]
            record = scorer.score("<answer>" + json.dumps({"objects": objects}) + "</answer>", row)
            self.assertEqual((record["gt_count"], record["tp"], record["fn"]), (4, 2, 2))
            self.assertEqual((record["answer_cap"], record["gt_beyond_answer_cap"]), (2, 2))
            cap = scorer.summary()["answer_cap"]
            self.assertEqual((cap["gt_total"], cap["gt_beyond_answer_cap"], cap["crops_with_gt_over_cap"]), (4, 2, 1))
            self.assertEqual((cap["pooled_tp"], cap["pooled_fp"], cap["pooled_fn"]), (2, 0, 2))
            self.assertAlmostEqual(cap["pooled_pq"], 2 / 3)
            self.assertAlmostEqual(cap["pooled_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
