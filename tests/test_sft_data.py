"""SFT supervision geometry, augmentation isolation, and token-loss boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.rewards import parse_completion
from nucleus_rl.sft_data import (
    D4_TRANSFORMS, apply_d4, assistant_token_labels, build_sft_example,
    coordinate_size, instance_prompt_objects, prepare_example,
    sft_prompt_messages, target_completion, validate_sft_splits,
)


class GeometryTests(unittest.TestCase):
    def test_all_eight_d4_operators_are_distinct_and_move_both_arrays_together(self):
        labels = np.arange(15, dtype=np.uint8).reshape(3, 5)
        rgb = np.repeat(labels[..., None], 3, axis=2)
        fingerprints = set()
        for transform in range(8):
            image, mask = apply_d4(Image.fromarray(rgb), labels, transform)
            self.assertTrue(np.array_equal(np.asarray(image)[..., 0], mask))
            self.assertTrue(mask.flags.c_contiguous)
            self.assertFalse(np.shares_memory(mask, labels))
            fingerprints.add((mask.shape, mask.tobytes()))
        self.assertEqual(len(fingerprints), 8)
        self.assertEqual(len(D4_TRANSFORMS), 8)
        self.assertTrue(np.array_equal(labels, np.arange(15).reshape(3, 5)))

    def test_tight_boxes_use_exclusive_edges_including_partial_border_objects(self):
        mask = np.zeros((7, 11), dtype=np.uint16)
        mask[0:2, 0:3] = 31
        mask[4:7, 9:11] = 7
        objects = instance_prompt_objects(mask)
        self.assertEqual(objects[0]["box"], [0, 0, 3, 2])
        self.assertEqual(objects[1]["box"], [9, 4, 11, 7])
        for obj in objects:
            x, y = obj["point"]
            self.assertGreater(mask[y, x], 0)

    def test_concave_centroid_is_replaced_with_a_real_foreground_pixel(self):
        mask = np.zeros((7, 7), dtype=np.uint16)
        mask[1:6, 1] = mask[1:6, 5] = 91
        mask[5, 1:6] = 91
        yy, xx = np.argwhere(mask == 91).mean(axis=0)
        self.assertEqual(mask[round(yy), round(xx)], 0)
        obj = instance_prompt_objects(mask)[0]
        x, y = obj["point"]
        self.assertEqual(mask[y, x], 91)

    def test_spatial_order_is_independent_of_instance_id_and_recomputed_after_rotation(self):
        mask = np.zeros((8, 12), dtype=np.uint16)
        mask[1:3, 7:10] = 8
        mask[5:7, 1:4] = 2
        objects = instance_prompt_objects(mask)
        self.assertEqual(objects[0]["box"], [7, 1, 10, 3])
        swapped = mask.copy()
        swapped[mask == 8], swapped[mask == 2] = 2, 8
        self.assertEqual(objects, instance_prompt_objects(swapped))
        _, rotated = apply_d4(Image.new("RGB", (12, 8)), mask, 2)
        rotated_objects = instance_prompt_objects(rotated)
        self.assertEqual(rotated_objects[0]["box"], [8, 1, 11, 3])

    def test_processed_and_normalized_targets_preserve_subpixel_geometry(self):
        mask = np.zeros((7, 11), dtype=np.uint16)
        mask[4:7, 8:11] = 1
        for mode, size in (("processed", (28, 14)), ("normalized1000", (1000, 1000))):
            objects = instance_prompt_objects(mask, coordinate_mode=mode,
                                              prompt_size=size if mode == "processed" else None)
            text = target_completion(mask, coordinate_mode=mode,
                                     prompt_size=size if mode == "processed" else None)
            parsed = parse_completion(text, width=size[0], height=size[1])
            self.assertTrue(parsed.valid, parsed.error)
            self.assertEqual(objects[0]["box"][2:], [size[0], size[1]])
            box = objects[0]["box"]
            self.assertAlmostEqual(box[0] * 11 / size[0], 8)
            self.assertAlmostEqual(box[1] * 7 / size[1], 4)

    def test_generated_target_matches_exact_existing_completion_contract(self):
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[1:3, 1:3] = 19
        self.assertEqual(target_completion(mask),
                         '<answer>{"objects":[{"box":[1,1,3,3],"point":[1,1]}]}</answer>')
        self.assertTrue(parse_completion(target_completion(mask), width=8, height=8).valid)

    def test_invalid_masks_and_oversized_targets_fail_instead_of_silent_truncation(self):
        bad = (np.zeros((3, 3)), np.array([[float("nan")]]), np.array([[-1]]),
               np.array([[1.5]]), np.ones((1, 1, 1)))
        for mask in bad:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                instance_prompt_objects(mask)
        with self.assertRaisesRegex(ValueError, "one to 8"):
            instance_prompt_objects(np.arange(1, 10).reshape(3, 3))

    def test_coordinate_frames_are_explicit_and_reject_inconsistent_dimensions(self):
        self.assertEqual(coordinate_size((11, 7), coordinate_mode="normalized1000"), (1000, 1000))
        self.assertEqual(coordinate_size((11, 7), coordinate_mode="processed", prompt_size=(28, 14)), (28, 14))
        for kwargs in ({"coordinate_mode": "processed"}, {"coordinate_mode": "guess"},
                       {"coordinate_mode": "normalized1000", "prompt_size": (100, 100)},
                       {"coordinate_mode": "original", "prompt_size": (7, 11)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                coordinate_size((11, 7), **kwargs)

    def test_invalid_d4_and_misaligned_image_mask_fail(self):
        image, mask = Image.new("RGB", (5, 3)), np.ones((3, 5), dtype=np.uint16)
        for transform in (-1, 8, True, 0.5):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                apply_d4(image, mask, transform)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            apply_d4(image, mask.T)


class DataIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        sources = load_official30_cohort()["image_ids"]
        self.train = self.make_row("train-crop", sources[0], "train", color=(10, 20, 30))
        self.validation = self.make_row("validation-crop", sources[1], "validation", color=(30, 20, 10))

    def tearDown(self):
        self.temporary.cleanup()

    def make_row(self, ident, source, split, color):
        image_path, mask_path = self.directory / f"{ident}.png", self.directory / f"{ident}-mask.png"
        Image.new("RGB", (11, 7), color).save(image_path)
        mask = np.zeros((7, 11), dtype=np.uint16)
        mask[1:4, 6:10] = 30
        Image.fromarray(mask).save(mask_path)
        return {"id": ident, "patient_id": "-".join(source.split("-")[:3]),
                "source_image_id": source, "split": split,
                "image_path": str(image_path), "mask_path": str(mask_path), "instance_count": 1}

    def test_fixed_canonical_split_is_audited_with_both_file_hashes(self):
        audit = validate_sft_splits([self.train], [self.validation])
        self.assertEqual(audit["status"], "official30_sft_split_verified")
        self.assertEqual(audit["train_crops"], 1)
        self.assertEqual(audit["validation_crops"], 1)
        self.assertEqual(len(audit["all_files_sha256"]), 2)
        for row in audit["all_files_sha256"]:
            self.assertEqual(len(row["image_sha256"]), 64)
            self.assertEqual(len(row["mask_sha256"]), 64)
        # Identical mask geometry across distinct patients is permitted.
        self.assertEqual(audit["all_files_sha256"][0]["mask_sha256"],
                         audit["all_files_sha256"][1]["mask_sha256"])

    def test_noncanonical_sources_and_patient_mismatch_are_rejected(self):
        extra = load_official30_cohort()["known_noncanonical_image_ids"][0]
        bad = {**self.train, "source_image_id": extra, "patient_id": "-".join(extra.split("-")[:3])}
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            validate_sft_splits([bad], [self.validation])
        with self.assertRaisesRegex(ValueError, "Patient ID"):
            validate_sft_splits([{**self.train, "patient_id": "TCGA-FAKE-1234"}], [self.validation])

    def test_declared_file_hashes_are_verified_and_detect_tampering(self):
        train = {**self.train}
        for key in ("image_path", "mask_path"):
            train[key.replace("_path", "_sha256")] = hashlib.sha256(Path(train[key]).read_bytes()).hexdigest()
        validate_sft_splits([train], [self.validation])
        for hash_key in ("image_sha256", "mask_sha256"):
            with self.subTest(hash_key=hash_key), self.assertRaisesRegex(ValueError, hash_key):
                validate_sft_splits([{**train, hash_key: "0" * 64}], [self.validation])
        # A still-readable altered image must fail its frozen identity check.
        Image.new("RGB", (11, 7), (80, 90, 100)).save(train["image_path"])
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            validate_sft_splits([train], [self.validation])

    def test_all_validation_masks_are_decoded_and_shape_count_checked_before_gpu(self):
        mask_path = Path(self.validation["mask_path"])
        Image.fromarray(np.ones((3, 4), dtype=np.uint16)).save(mask_path)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            validate_sft_splits([self.train], [self.validation])
        Image.fromarray(np.zeros((7, 11), dtype=np.uint16)).save(mask_path)
        with self.assertRaisesRegex(ValueError, "instance count"):
            validate_sft_splits([self.train], [self.validation])

    def test_same_source_patient_or_row_cannot_cross_splits(self):
        bad = {**self.train, "split": "validation"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sft_splits([self.train], [bad])
        bad = {**self.validation, "source_image_id": self.train["source_image_id"],
               "patient_id": self.train["patient_id"]}
        with self.assertRaisesRegex(ValueError, "patient_id"):
            validate_sft_splits([self.train], [bad])

    def test_aliased_paths_and_exact_image_duplicates_are_rejected(self):
        bad = {**self.validation, "image_path": self.train["image_path"]}
        with self.assertRaisesRegex(ValueError, "image_path"):
            validate_sft_splits([self.train], [bad])
        Path(self.validation["image_path"]).write_bytes(Path(self.train["image_path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "byte duplicates"):
            validate_sft_splits([self.train], [self.validation])

    def test_wrong_split_missing_files_and_relative_paths_fail(self):
        with self.assertRaisesRegex(ValueError, "validation-only"):
            validate_sft_splits([self.train], [{**self.validation, "split": "test"}])
        with self.assertRaises(FileNotFoundError):
            validate_sft_splits([self.train], [{**self.validation, "mask_path": str(self.directory / "missing.png")}])
        with self.assertRaisesRegex(ValueError, "Resolve manifest"):
            validate_sft_splits([{**self.train, "image_path": "relative.png"}], [self.validation])

    def test_validation_masks_never_become_sft_targets_even_without_augmentation(self):
        for transform in (0, 1, 7):
            with self.subTest(transform=transform), self.assertRaisesRegex(ValueError, "training rows only"):
                build_sft_example(self.validation, transform=transform)

    def test_transform_changes_target_and_preserves_original_files(self):
        before = Path(self.train["mask_path"]).read_bytes()
        sample = build_sft_example(self.train, transform=1, coordinate_mode="normalized1000")
        self.assertEqual(sample["image"].size, (7, 11))
        self.assertEqual(sample["mask"].shape, (11, 7))
        self.assertEqual(sample["augmentation"], "rot90")
        self.assertTrue(parse_completion(sample["completion"], width=1000, height=1000).valid)
        self.assertEqual(Path(self.train["mask_path"]).read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "instance count"):
            build_sft_example({**self.train, "instance_count": 2})

    def test_prepared_example_measures_processed_dimensions_after_rotation(self):
        class FakeImageProcessor:
            patch_size = 14

            def __call__(self, *, images, return_tensors):
                self.last_image_size = images.size
                return {"image_grid_thw": [[1, 2, 4]]}

        class FakeProcessor:
            image_processor = FakeImageProcessor()

        processor = FakeProcessor()
        sample = prepare_example(self.train, processor, augmentation_id=1)
        self.assertEqual(processor.image_processor.last_image_size, (7, 11))
        self.assertEqual(sample["prompt_size"], (56, 28))
        self.assertEqual(sample["geometry"]["original_width"], 7)
        self.assertEqual(sample["geometry"]["processed_width"], 56)
        self.assertIn("56 pixels wide and 28 pixels high", sample["messages"][-1]["content"])
        self.assertTrue(parse_completion(sample["completion"], width=56, height=28).valid)

    def test_normalized_prompt_has_explicit_frame_and_schema(self):
        messages = sft_prompt_messages(1000, 1000, coordinate_mode="normalized1000")
        self.assertIn("normalized coordinates from 0 to 1000", messages[-1]["content"])
        self.assertNotIn("Do not use normalized", messages[-1]["content"])
        with self.assertRaises(ValueError):
            sft_prompt_messages(128, 128, coordinate_mode="normalized1000")


class TeacherForcingTests(unittest.TestCase):
    def test_only_assistant_suffix_has_loss_and_eos_is_kept_when_it_equals_pad(self):
        # Token 9 is both EOS and PAD. Attention, rather than token ID, decides
        # which occurrence should contribute to the assistant loss.
        labels = assistant_token_labels([1, 2, 3, 4, 9, 9], [1, 2, 3],
                                        attention_mask=[1, 1, 1, 1, 1, 0], eos_token_id=9)
        self.assertEqual(labels, [-100, -100, -100, 4, 9, -100])

    def test_prompt_token_boundary_mismatch_and_truncated_answers_fail(self):
        cases = (([1, 3, 4], [1, 2]), ([1, 2], [1, 2]))
        for full, prompt in cases:
            with self.subTest(full=full), self.assertRaisesRegex(ValueError, "exact prompt"):
                assistant_token_labels(full, prompt)
        with self.assertRaisesRegex(ValueError, "EOS"):
            assistant_token_labels([1, 2, 3], [1, 2], eos_token_id=9)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            assistant_token_labels([1, 2, 3], [1, 2], attention_mask=[1, 1, 0])
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            assistant_token_labels([1, 2, 3], [1], attention_mask=[1, 1])


if __name__ == "__main__":
    unittest.main()
