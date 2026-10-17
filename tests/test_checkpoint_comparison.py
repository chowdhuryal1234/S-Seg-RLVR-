"""CPU-only checks that cross-checkpoint runs preserve the paired reference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from nucleus_rl.prompt_ablation import MODEL, REVISION, VARIANTS, verify_reference_run


class CheckpointReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "frozen_sam.pt"
        # No model is deserialized: only file content hashes are relevant here.
        self.checkpoint.write_bytes(b"test-only SAM checkpoint bytes")
        self.args = SimpleNamespace(
            max_completion_length=512, seed=42, min_pixels=224 ** 2,
            max_pixels=448 ** 2, coordinate_frame="processed", prompt_style="example",
            sam_checkpoint=str(self.checkpoint), sam_config="configs/sam2.1/sam2.1_hiera_t.yaml",
            model=MODEL, revision=REVISION,
        )
        self.selection = [
            {"id": f"crop-{index}", "patient_id": f"patient-{index}",
             "source_image_id": f"source-{index}", "gt_count": index + 2,
             "image_sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
             "mask_sha256": hashlib.sha256(f"mask-{index}".encode()).hexdigest()}
            for index in range(2)
        ]
        self.result = {
            "status": "completed_paired_inference", "optimizer_steps": 0,
            "policy_trainable_parameters": 0,
            "model": {"adapter": None, "model": MODEL, "resolved_revision": REVISION},
            "selected_crops": copy.deepcopy(self.selection),
            "sam": {"checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                    "config": self.args.sam_config},
        }
        self.plan = {"args": vars(self.args).copy()}
        self.records = [
            {"id": item["id"], "variant": variant, "text": f"completion for {item['id']}/{variant}"}
            for item in self.selection for variant in VARIANTS
        ]
        self.write_reference()

    def write_reference(self):
        (self.root / "result.json").write_text(json.dumps(self.result))
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        (self.root / "paired_records.jsonl").write_text(
            "\n".join(json.dumps(row) for row in self.records) + "\n"
        )

    def verify(self, selection=None, args=None):
        return verify_reference_run(
            self.root, self.selection if selection is None else selection,
            self.args if args is None else args,
        )

    def test_complete_reference_maps_pairs_independent_of_record_order(self):
        self.records.reverse()
        self.write_reference()
        checked = self.verify()
        self.assertEqual(set(checked), {(item["id"], variant)
                                      for item in self.selection for variant in VARIANTS})
        for original in self.records:
            self.assertEqual(checked[original["id"], original["variant"]], original)

    def test_rejects_changed_seed_and_each_other_inference_control(self):
        changes = {
            "seed": 43, "max_completion_length": 256, "min_pixels": 128 ** 2,
            "max_pixels": 560 ** 2, "coordinate_frame": "original", "prompt_style": "schema",
        }
        for key, value in changes.items():
            with self.subTest(control=key):
                args = copy.copy(self.args)
                setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, f"Reference control differs: {key}"):
                    self.verify(args=args)

    def test_rejects_changed_sam_bytes_even_at_same_path(self):
        self.checkpoint.write_bytes(b"different model at the same path")
        with self.assertRaisesRegex(ValueError, "Reference SAM weights/config differ"):
            self.verify()

    def test_rejects_changed_sam_config(self):
        args = copy.copy(self.args)
        args.sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        with self.assertRaisesRegex(ValueError, "Reference SAM weights/config differ"):
            self.verify(args=args)

    def test_rejects_changed_source_image_or_mask_hash(self):
        for key in ("image_sha256", "mask_sha256"):
            with self.subTest(hash=key):
                selection = copy.deepcopy(self.selection)
                selection[0][key] = "0" * 64
                with self.assertRaisesRegex(ValueError, "Reference selection or image/mask hashes differ"):
                    self.verify(selection=selection)

    def test_rejects_changed_selection_membership_or_order(self):
        changed = copy.deepcopy(self.selection)
        changed[0]["id"] = "different-crop"
        for selection in (changed, self.selection[:1], list(reversed(self.selection))):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, "Reference selection or image/mask hashes differ"):
                    self.verify(selection=selection)

    def test_rejects_duplicate_pair_even_if_record_count_looks_plausible(self):
        self.records[-1] = copy.deepcopy(self.records[0])
        self.write_reference()
        with self.assertRaisesRegex(ValueError, "Duplicate reference pair"):
            self.verify()

    def test_rejects_missing_pair(self):
        self.records.pop()
        self.write_reference()
        with self.assertRaisesRegex(ValueError, "Missing or unexpected reference pair"):
            self.verify()

    def test_rejects_unexpected_pair(self):
        self.records[-1]["variant"] = "unplanned_prompt"
        self.write_reference()
        with self.assertRaisesRegex(ValueError, "Missing or unexpected reference pair"):
            self.verify()

    def test_changed_model_or_revision_requires_token_hashes_for_every_pair(self):
        for change in ({"model": "Qwen/Qwen2.5-VL-32B-Instruct"}, {"revision": "a" * 40}):
            with self.subTest(change=change):
                args = SimpleNamespace(**{**vars(self.args), **change})
                with self.assertRaisesRegex(ValueError, "requires recorded input token hashes"):
                    self.verify(args=args)
                # All-but-one is still insufficient evidence for paired inputs.
                records = copy.deepcopy(self.records)
                for row in self.records[:-1]:
                    row["input_ids_sha256"] = "b" * 64
                self.write_reference()
                with self.assertRaisesRegex(ValueError, "requires recorded input token hashes"):
                    self.verify(args=args)
                self.records = records
                self.write_reference()

    def test_changed_checkpoint_accepts_complete_recorded_token_hashes(self):
        for row in self.records:
            row["input_ids_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        self.write_reference()
        args = SimpleNamespace(**{**vars(self.args), "model": "Qwen/Qwen2.5-VL-32B-Instruct", "revision": "a" * 40})
        self.assertEqual(len(self.verify(args=args)), 4)

    def test_rejects_incomplete_trained_trainable_or_adapted_reference(self):
        original = copy.deepcopy(self.result)
        changes = ({"status": "running"}, {"optimizer_steps": 1},
                   {"policy_trainable_parameters": 1}, {"model": {"adapter": "/saved/lora"}})
        for change in changes:
            with self.subTest(change=change):
                self.result = {**original, **change}
                self.write_reference()
                with self.assertRaisesRegex(ValueError, "completed, unadapted, frozen inference run"):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
