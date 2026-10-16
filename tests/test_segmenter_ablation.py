"""Fixed-prompt D controls with real scoring and a deterministic CPU segmenter."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import masks_to_instances, parse_completion
from nucleus_rl.segmenter import file_sha256
from nucleus_rl.segmenter_ablation import build_parser, load_replay_inputs, run, run_one_segmenter


def save_json(path, value):
    path.write_text(json.dumps(value))


def save_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FakePredictor:
    def predict(self, *, box, **kwargs):
        mask = np.zeros((1, 8, 8), dtype=bool)
        x0, y0, x1, y1 = map(int, box)
        mask[0, y0:y1, x0:x1] = True
        return mask, np.array([0.9]), np.zeros((1, 4, 4))


class FakeSAM:
    def __init__(self, checkpoint=None, config=None, device="cpu"):
        self.predictor = FakePredictor()
        self.calls = []
        self.metadata = {"checkpoint_sha256": file_sha256(checkpoint) if checkpoint else "fake",
                         "parameters_before": "same", "trainable_parameters": 0, "config": config}

    def predict(self, image, objects, image_key=None):
        self.calls.append(tuple(objects))
        if not objects:
            return np.zeros((image.height, image.width), dtype=np.int32)
        results = [self.predictor.predict(box=obj.box, point_coords=[obj.point]) for obj in objects]
        return masks_to_instances(np.concatenate([item[0] for item in results]),
                                  scores=np.concatenate([item[1] for item in results]))

    def verify_unchanged(self):
        return {**self.metadata, "parameters_after": "same", "unchanged": True}


class SegmenterAblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.tiny_checkpoint = self.root / "tiny.pt"
        self.tiny_checkpoint.write_bytes(b"tiny fake checkpoint")
        self.manifest = self.root / "train.jsonl"
        self.rows, self.selection, self.saved = [], [], []
        first = {"box": [2, 2, 6, 6], "point": [4, 4]}
        second = {"box": [8, 8, 12, 12], "point": [10, 10]}
        for index, image_id in enumerate(load_official30_cohort()["image_ids"][:8]):
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[1:3, 1:3], labels[4:6, 4:6] = 300, 301
            image_path, mask_path = self.root / f"{index}.png", self.root / f"{index}_mask.png"
            Image.new("RGB", (8, 8), (index, 50, 100)).save(image_path)
            Image.fromarray(labels).save(mask_path)
            row = {"id": image_id + "_x0_y0", "source_image_id": image_id,
                   "patient_id": "-".join(image_id.split("-")[:3]), "split": "train",
                   "image_path": image_path.name, "mask_path": mask_path.name, "instance_count": 2}
            self.rows.append(row)
            self.selection.append({"id": row["id"], "patient_id": row["patient_id"],
                                   "source_image_id": image_id, "gt_count": 2,
                                   "image_sha256": file_sha256(image_path), "mask_sha256": file_sha256(mask_path)})
            for arm in ("one_example", "three_examples"):
                objects = [first] if arm == "one_example" else [first, first if index == 2 else second]
                text = "<answer>" + json.dumps({"objects": objects}) + "</answer>"
                if index == 1 and arm == "one_example":
                    text = '<answer>{"objects":['  # Invalid after a valid record; must not inherit its masks.
                parsed = parse_completion(text, width=16, height=16)
                converted, scales = objects_to_original(parsed.objects, (16, 16), (8, 8))
                self.saved.append({"id": row["id"], "patient_id": row["patient_id"],
                                   "variant": arm, "text": text, "format_valid": parsed.valid,
                                   "coordinate_frame": "processed", "prompt_size": [16, 16],
                                   "original_size": [8, 8], "scale_to_original": list(scales),
                                   "raw_objects": [asdict(obj) for obj in parsed.objects],
                                   "converted_objects": [asdict(obj) for obj in converted]})
        save_lines(self.manifest, self.rows)
        save_json(self.source / "plan.json", {"selected_crops": self.selection,
                                             "manifest_sha256": file_sha256(self.manifest)})
        save_json(self.source / "result.json", {"selected_crops": self.selection,
                                               "status": "completed_paired_inference",
                                               "optimizer_steps": 0, "all_paired_pixels_identical": True,
                                               "policy_trainable_parameters": 0, "model": {"adapter": None},
                                               "sam": {"checkpoint_sha256": file_sha256(self.tiny_checkpoint),
                                                       "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                                                       "unchanged": True, "trainable_parameters": 0}})
        save_lines(self.source / "paired_records.jsonl", self.saved)

    def load(self):
        return load_replay_inputs(self.source, self.manifest)

    def test_replay_preserves_text_and_processed_frame_and_separates_oracle(self):
        replay = self.load()
        self.assertEqual(len(replay["cases"]), 24)
        source_lookup = {(r["id"], r["variant"]): r for r in self.saved}
        for case in replay["cases"]:
            row = case["row"]
            if case["prompt_arm"] == "oracle_tight":
                self.assertTrue(case["ground_truth_used_for_prompting"])
                self.assertEqual(row["coordinate_frame"], "original")
                objects = parse_completion(case["text"], width=8, height=8).objects
                self.assertEqual(len(objects), 2)  # GT IDs 300/301, not a binary connected-component count.
                self.assertEqual(objects[0].box, (1, 1, 3, 3))
            else:
                self.assertFalse(case["ground_truth_used_for_prompting"])
                saved = source_lookup[(row["id"], case["prompt_arm"])]
                self.assertEqual(case["text"], saved["text"])
                self.assertEqual((row["prompt_width"], row["original_width"]), (16, 8))
        self.assertTrue(replay["source_crop_hashes_verified"])

    def test_changed_crop_bytes_are_rejected_before_model_loading(self):
        Image.new("RGB", (8, 8), "red").save(self.root / self.rows[0]["image_path"])
        with self.assertRaisesRegex(ValueError, "image_sha256 mismatch"):
            self.load()

    def test_training_only_and_exact_source_selection_are_required(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["split"] = "validation"
        save_lines(self.manifest, changed)
        with self.assertRaises(ValueError):
            self.load()
        save_lines(self.manifest, self.rows)
        plan = json.loads((self.source / "plan.json").read_text())
        plan["selected_crops"] = list(reversed(plan["selected_crops"]))
        save_json(self.source / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "agree on exactly eight"):
            self.load()

    def test_missing_invalid_completion_is_not_silently_dropped(self):
        save_lines(self.source / "paired_records.jsonl", [r for r in self.saved if r["format_valid"]])
        with self.assertRaisesRegex(ValueError, "sixteen A completions"):
            self.load()

    def test_coordinate_or_parser_drift_is_rejected(self):
        changes = (("scale_to_original", [1, 1]), ("converted_objects", []),
                   ("format_valid", False), ("prompt_size", [True, 16]))
        for field, value in changes:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.saved)
                changed[0][field] = value
                save_lines(self.source / "paired_records.jsonl", changed)
                with self.assertRaises(ValueError):
                    self.load()

    def test_source_policy_must_be_frozen_base_and_segmenter_must_match_tiny(self):
        result_path = self.source / "result.json"
        original = json.loads(result_path.read_text())
        changes = (("policy_trainable_parameters", 10), ("model", {"adapter": "some-lora"}),
                   ("model", {}), ("sam", {**original["sam"], "config": "large.yaml"}))
        for field, value in changes:
            with self.subTest(field=field, value=value):
                save_json(result_path, {**original, field: value})
                with self.assertRaises(ValueError):
                    self.load()

    def test_real_scorer_retains_invalid_zero_and_raw_duplicates(self):
        replay = self.load()
        segmenter = FakeSAM()
        with contextlib.redirect_stdout(io.StringIO()):
            observed = run_one_segmenter("tiny", segmenter, replay["cases"], self.root / "scored")
        records = observed["records"]
        self.assertEqual(len(records), 24)
        self.assertEqual(segmenter.calls[0][0].box, (1, 1, 3, 3))
        invalid = next(r for r in records if not r["format_valid"])
        self.assertEqual((invalid["total_reward"], invalid["pred_count"], invalid["raw_mask_count"]), (0, 0, 0))
        self.assertEqual(np.load(invalid["raw_masks_path"])["masks"].shape, (0, 8, 8))
        self.assertFalse(np.load(invalid["prediction_path"]).any())
        duplicate = next(r for r in records if r["id"] == self.rows[2]["id"] and r["prompt_arm"] == "three_examples")
        self.assertEqual((duplicate["raw_mask_count"], duplicate["pred_count"],
                          duplicate["same_gt_duplicate_count"], duplicate["raw_overlap_pair_count"]), (2, 1, 1, 1))
        one = observed["variants"]["tiny_one_example"]
        self.assertEqual((one["n_records"], one["format_valid_count"]), (8, 7))
        self.assertAlmostEqual(one["macro_pq"], 7 * (2 / 3) / 8)
        self.assertAlmostEqual(one["micro_recall"], 7 / 16)
        oracle = observed["variants"]["tiny_oracle_tight"]
        self.assertEqual((oracle["macro_pq"], oracle["mean_total_reward"]), (1, 2))
        self.assertTrue(oracle["oracle_prompts"])
        standalone = [json.loads(line) for line in (self.root / "scored/tiny_oracle_tight/completions.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["oracle_prompts"] and r["ground_truth_used_for_prompting"] for r in standalone))

    def test_cli_run_keeps_models_sequential_and_prompts_identical(self):
        try:
            import torch  # Only the existing CPU runtime; no actual SAM or VLM is loaded.
        except ImportError:
            self.skipTest("Optional CPU torch runtime is unavailable")
        tiny, large = self.root / "tiny.pt", self.root / "large.pt"
        tiny.write_bytes(b"tiny fake checkpoint")
        large.write_bytes(b"large fake checkpoint")
        args = build_parser().parse_args([
            "--source-run", str(self.source), "--manifest", str(self.manifest),
            "--tiny-checkpoint", str(tiny), "--large-checkpoint", str(large),
            "--output", str(self.root / "run"), "--device", "cpu", "--threads", "1"])
        references = []

        def construct(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in references), "Previous SAM remains live")
            model = FakeSAM(*args, **kwargs)
            references.append(weakref.ref(model))
            return model

        with patch("nucleus_rl.segmenter.FrozenSAM2", side_effect=construct), contextlib.redirect_stdout(io.StringIO()):
            result = run(args)
        self.assertEqual(len(references), 2)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(len(result["variants"]), 6)
        self.assertEqual((len(result["policy_variants"]), len(result["oracle_variants"])), (4, 2))
        self.assertEqual((result["optimizer_steps"], result["qwen_generation_calls"], result["vlm_loaded"]), (0, 0, False))
        self.assertTrue(result["fixed_prompts_identical"])
        self.assertTrue(all(model["unchanged"] for model in result["models"].values()))
        records = [json.loads(line) for line in Path(result["records_path"]).read_text().splitlines()]
        self.assertEqual(len(records), 48)
        by_key = {(r["segmenter_size"], r["id"], r["prompt_arm"]): r for r in records}
        for case in self.load()["cases"]:
            left = by_key[("tiny", case["row"]["id"], case["prompt_arm"])]
            right = by_key[("large", case["row"]["id"], case["prompt_arm"])]
            for key in ("text", "prompt_size", "converted_objects", "fixed_prompt_sha256"):
                self.assertEqual(left[key], right[key])
        args.output = str(self.root / "wrong_checkpoint_run")
        tiny.write_bytes(b"different tiny weights")
        with patch("nucleus_rl.segmenter.FrozenSAM2") as constructor:
            with self.assertRaisesRegex(ValueError, "Tiny checkpoint SHA256"):
                run(args)
            constructor.assert_not_called()
        self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
