"""CPU-only oracle diagnostic tests using a fake predictor, never real SAM."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.prompt_diagnostic import (
    RawRecordingSegmenter, build_parser, oracle_prompt_objects, run_diagnostic,
    select_training_rows, validate_args,
)
from nucleus_rl.rewards import masks_to_instances, parse_completion


class PromptTests(unittest.TestCase):
    def test_point_in_concave_object_even_when_centroid_is_background(self):
        labels = np.zeros((5, 5), dtype=np.uint16)
        labels[0, :] = 300
        labels[4, :] = 300
        labels[:, 0] = 300
        labels[:, 4] = 300
        self.assertEqual(labels[2, 2], 0)
        obj = oracle_prompt_objects(labels)[0]
        x, y = obj["point"]
        self.assertEqual(labels[y, x], 300)
        self.assertEqual(obj["box"], [0, 0, 5, 5])

    def test_padding_preserves_points_and_clips_bounds(self):
        labels = np.zeros((8, 9), dtype=np.uint16)
        labels[0:2, 0:2] = 42
        labels[6:8, 8] = 900
        tight = oracle_prompt_objects(labels)
        padded = oracle_prompt_objects(labels, padding=5)
        full = oracle_prompt_objects(labels, full_image_boxes=True)
        self.assertEqual([obj["point"] for obj in tight], [obj["point"] for obj in padded])
        self.assertEqual([obj["point"] for obj in tight], [obj["point"] for obj in full])
        self.assertEqual(padded[0]["box"], [0, 0, 7, 7])
        self.assertTrue(all(obj["box"] == [0, 0, 9, 8] for obj in full))
        for objects in (tight, padded, full):
            text = '<answer>' + json.dumps({"objects": objects}) + '</answer>'
            self.assertTrue(parse_completion(text, width=9, height=8).valid)

    def test_unique_instance_ids_not_foreground_components(self):
        labels = np.array([[10, 10, 50, 50]])
        self.assertEqual(len(oracle_prompt_objects(labels)), 2)

    def test_empty_and_more_than_eight_instances_rejected(self):
        for labels in [np.zeros((4, 4)), np.arange(1, 10).reshape(3, 3)]:
            with self.assertRaises(ValueError):
                oracle_prompt_objects(labels)

    def test_train_only_and_fixed_first_rows(self):
        rows = [{"id": str(i), "split": "train"} for i in range(8)]
        self.assertEqual([row["id"] for row in select_training_rows(rows, 4)], ["0", "1", "2", "3"])
        rows[-1]["split"] = "val"
        with self.assertRaises(ValueError):
            select_training_rows(rows, 4)

    def test_resource_bounds(self):
        args = build_parser().parse_args(["--manifest", "train.jsonl", "--sam-checkpoint", "sam.pt", "--output", "run"])
        validate_args(args)
        args.max_images = 5
        with self.assertRaises(ValueError):
            validate_args(args)
        args.max_images, args.threads = 4, 9
        with self.assertRaises(ValueError):
            validate_args(args)


class FakePredictor:
    def predict(self, *, box, point):
        mask = np.zeros((1, 8, 8), dtype=bool)
        x0, y0, x1, y1 = [int(value) for value in box]
        mask[:, y0:y1, x0:x1] = True
        return mask, np.array([0.9]), np.ones((1, 4, 4), dtype=np.float32)


class FakeFrozenSAM:
    def __init__(self):
        self.predictor = FakePredictor()
        self.metadata = {"checkpoint_sha256": "fake-test-checkpoint", "parameters_before": "unchanged-test", "trainable_parameters": 0}

    def predict(self, image, objects, image_key=None):
        predictions = [self.predictor.predict(box=obj.box, point=obj.point) for obj in objects]
        masks = np.concatenate([result[0] for result in predictions], axis=0)
        scores = np.concatenate([result[1] for result in predictions])
        return masks_to_instances(masks, scores)

    def verify_unchanged(self):
        return {**self.metadata, "parameters_after": "unchanged-test", "unchanged": True}


class ArtifactTests(unittest.TestCase):
    def test_raw_masks_and_oracle_scope_persist_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[2:4, 2:4] = 300
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "image.png")
            Image.fromarray(labels).save(root / "mask.png")
            row = {"id": "tiny", "split": "train", "patient_id": "fake-patient",
                   "image_path": str(root / "image.png"), "mask_path": str(root / "mask.png"),
                   "truncated_instance_ids": []}
            result = run_diagnostic([row], FakeFrozenSAM(), root / "output", include_full_image_boxes=True)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["vlm_loaded"])
            self.assertTrue(result["oracle_prompts"])
            self.assertTrue(result["sam_freeze_audit"]["unchanged"])
            self.assertTrue(Path(result["contact_sheet"]).is_file())
            records = [json.loads(line) for line in (root / "output/oracle_comparisons.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["seg_reward"], 1.0)
            self.assertLess(records[1]["seg_reward"], 1.0)
            for record in records:
                self.assertEqual(record["gt_instance_ids_in_prompt_order"], [300])
                with np.load(record["raw_sam_outputs_path"], allow_pickle=False) as raw:
                    self.assertEqual(raw["masks"].shape, (1, 8, 8))
                    self.assertEqual(raw["scores"].shape, (1,))
                    self.assertEqual(raw["low_resolution_logits"].shape, (1, 4, 4))
            standalone = json.loads((root / "output/tight/completions.jsonl").read_text().splitlines()[0])
            self.assertTrue(standalone["ground_truth_used_for_prompting"])
            standalone_summary = json.loads((root / "output/tight/summary.json").read_text())
            self.assertTrue(standalone_summary["oracle_prompts"])
            with self.assertRaises(ValueError):
                run_diagnostic([row], FakeFrozenSAM(), root / "output")

    def test_temporary_predictor_hook_restored_on_error(self):
        base = FakeFrozenSAM()
        original = base.predictor.predict
        wrapper = RawRecordingSegmenter(base)
        def fail(*args, **kwargs):
            raise RuntimeError("fake inference failure")
        base.predict = fail
        with self.assertRaises(RuntimeError):
            wrapper.predict(Image.new("RGB", (8, 8)), [])
        self.assertEqual(base.predictor.predict, original)


if __name__ == "__main__":
    unittest.main()
