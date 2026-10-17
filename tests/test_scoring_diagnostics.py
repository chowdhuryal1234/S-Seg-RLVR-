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


if __name__ == "__main__":
    unittest.main()
