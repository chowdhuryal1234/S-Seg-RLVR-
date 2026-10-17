"""Saved-result verification prevents persuasive plots of mismatched artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.ablation_metrics import duplicate_metrics, summarize_records
from nucleus_rl.rewards import instance_metrics
from scripts.render_followup_comparison import (
    D_VARIANTS, VARIANTS, load_policy_comparison, load_segmenter_comparison, render_report,
)


def fixture(root):
    dataset, base, large, segmenter = [root / name for name in ("data", "base", "large", "segmenter")]
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    selection, annotations = [], {}
    for index in range(8):
        crop_id = f"fixture_{index:02d}"
        mask = np.zeros((128, 128), dtype=np.uint16)
        for nucleus in range(index % 3 + 1):
            x = 10 + nucleus * 30
            mask[30:48, x:x + 14] = nucleus + 1
        pixels = np.full((128, 128, 3), [240, 190, 210], dtype=np.uint8)
        pixels[mask > 0] = [90, 50, 125]
        image_path, mask_path = dataset / "images" / f"{crop_id}.png", dataset / "masks" / f"{crop_id}.png"
        Image.fromarray(pixels).save(image_path)
        Image.fromarray(mask).save(mask_path)
        annotations[crop_id] = mask
        selection.append({"id": crop_id, "patient_id": f"fixture_patient_{index}", "source_image_id": f"fixture_source_{index}",
                          "gt_count": index % 3 + 1, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                          "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()})
    sam = {"unchanged": True, "trainable_parameters": 0, "parameters_before": "frozen", "parameters_after": "frozen",
           "checkpoint_sha256": "tiny_checkpoint", "config": "tiny_config", "source_commit": "same_sam_source"}
    for run, model in ((base, "Qwen/Qwen2.5-VL-7B-Instruct"), (large, "Qwen/Qwen2.5-VL-32B-Instruct"), (segmenter, None)):
        run.mkdir()
        variants = D_VARIANTS if model is None else VARIANTS
        records = []
        for variant in variants:
            (run / variant).mkdir()
            for index, item in enumerate(selection):
                mask = annotations[item["id"]]
                prediction = mask.copy()
                raw = np.array([mask == n for n in range(1, item["gt_count"] + 1)])
                path, raw_path = run / variant / f"{index + 1:06d}.npy", run / variant / f"{index + 1:06d}.npz"
                np.save(path, prediction)
                np.savez_compressed(raw_path, masks=raw)
                objects = [{"box": [10, 30, 24, 48], "point": [16, 39]}]
                record = {"id": item["id"], "variant": variant, "text": "<answer>fixture</answer>",
                          "prompt": [{"role": "user", "content": "fixture prompt " + variant}],
                          "prompt_size": [224, 224], "original_size": [128, 128], "coordinate_frame": "processed",
                          "scale_to_original": [128 / 224, 128 / 224], "pixel_values_sha256": f"pixels{index}",
                          "prediction_path": str(path), "raw_masks_path": str(raw_path), "format_valid": True,
                          "parse_error": None, "hit_token_cap": False, "raw_objects": objects, "converted_objects": objects,
                          **instance_metrics(prediction, mask), **duplicate_metrics(raw, mask)}
                if model is None:
                    record.update(oracle_prompts=variant.endswith("oracle_tight"),
                                  ground_truth_used_for_prompting=variant.endswith("oracle_tight"))
                records.append(record)
        summaries = {variant: summarize_records(r for r in records if r["variant"] == variant) for variant in variants}
        if model is not None:
            for summary in summaries.values():
                summary["token_cap_hits"] = 0
            result = {"status": "completed_paired_inference", "optimizer_steps": 0, "policy_trainable_parameters": 0,
                      "all_paired_pixels_identical": True, "sam": sam, "model": {"model": model, "adapter": None, "resolved_revision": "fixture"}}
            filename = "paired_records.jsonl"
            plan = {"args": {"seed": 42, "max_completion_length": 512, "min_pixels": 50176, "max_pixels": 200704,
                             "coordinate_frame": "processed"}, "decoding": "Greedy"}
            (run / "plan.json").write_text(json.dumps(plan))
        else:
            result = {"status": "completed_fixed_prompt_segmenter_ablation", "optimizer_steps": 0, "vlm_loaded": False,
                      "qwen_generation_calls": 0, "fixed_prompts_identical": True, "source_crop_hashes_verified": True,
                      "models": {"tiny": sam, "large": {**sam, "checkpoint_sha256": "large_checkpoint", "config": "large_config"}}}
            filename = "records.jsonl"
        result.update(selected_crops=selection, variants=summaries)
        (run / "result.json").write_text(json.dumps(result))
        (run / filename).write_text("".join(json.dumps(r) + "\n" for r in records))
    return base, large, segmenter, dataset


class FollowupReportTests(unittest.TestCase):
    def test_renders_every_case_and_keeps_oracle_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            output = root / "report"
            result = render_report(*args, output)
            self.assertEqual(result["verified_cases"], 8)
            self.assertEqual(len(result["sheets"]), 10)
            self.assertTrue(all(Path(path).is_file() for path in result["sheets"]))
            self.assertIn("Privileged diagnostic", (output / "index.html").read_text())
            self.assertEqual(len((output / "per_crop.csv").read_text().splitlines()), 81)
            with self.assertRaisesRegex(ValueError, "new empty"):
                render_report(*args, output)

    def test_raw_mask_tampering_is_rejected_even_when_final_maps_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            base, large, _, dataset = fixture(Path(directory))
            path = large / "one_example" / "000001.npz"
            np.savez_compressed(path, masks=np.zeros((0, 128, 128), dtype=bool))
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                load_policy_comparison(base, large, dataset)

    def test_oracle_mislabeling_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            base, large, d, dataset = fixture(Path(directory))
            original, _ = load_policy_comparison(base, large, dataset)
            path = d / "records.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            oracle = next(row for row in rows if row["variant"] == "tiny_oracle_tight")
            oracle["ground_truth_used_for_prompting"] = False
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "oracle provenance"):
                load_segmenter_comparison(d, original[0], original[1])


if __name__ == "__main__":
    unittest.main()
