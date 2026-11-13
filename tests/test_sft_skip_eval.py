"""--skip-before-eval and mixed MoNuSeg/CoNSeP/PUMA manifests in the SFT runner (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from nucleus_rl.cohort import DEFAULT_COHORT_PATH, load_official30_cohort
from nucleus_rl.sft import before_evaluation, build_parser, dataset_counts, json_write, main

REQUIRED = ["--manifest", "train", "--eval-manifest", "val", "--output", "out", "--model", "m",
            "--sam-checkpoint", "sam"]


class SkipBeforeEvalTests(unittest.TestCase):
    def test_flag_defaults_off_and_parses_like_grpo(self):
        self.assertFalse(build_parser().parse_args(REQUIRED).skip_before_eval)
        self.assertTrue(build_parser().parse_args(REQUIRED + ["--skip-before-eval"]).skip_before_eval)

    def test_skip_never_calls_evaluation(self):
        args = SimpleNamespace(skip_before_eval=True)
        with patch("nucleus_rl.sft.evaluate_policy") as evaluate:
            self.assertEqual(before_evaluation("model", "processor", ["row"], "sam", Path("out"), args), (None, 0.0))
        evaluate.assert_not_called()

    def test_default_runs_one_before_pass_into_before_directory(self):
        args = SimpleNamespace(skip_before_eval=False)
        with patch("nucleus_rl.sft.evaluate_policy", return_value={"mean_pq": 0.5}) as evaluate:
            summary, seconds = before_evaluation("model", "processor", ["row"], "sam", Path("out"), args)
        self.assertEqual(summary, {"mean_pq": 0.5})
        self.assertGreaterEqual(seconds, 0.0)
        evaluate.assert_called_once_with("model", "processor", ["row"], "sam", Path("out") / "before", args)

    def test_flag_is_recorded_in_arguments_json(self):
        args = build_parser().parse_args(REQUIRED + ["--skip-before-eval"])
        with tempfile.TemporaryDirectory() as directory:
            json_write(Path(directory) / "arguments.json", vars(args))
            self.assertIs(json.loads((Path(directory) / "arguments.json").read_text())["skip_before_eval"], True)


class MixedManifestTests(unittest.TestCase):
    """Rows shaped like data/cleanmix_v1/train.jsonl next to official-30 MoNuSeg rows, no --cohort-protocol."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = load_official30_cohort()["image_ids"]
        (self.root / "sam.pt").write_bytes(b"sam")

    def tearDown(self):
        self.temporary.cleanup()

    def crop(self, ident, count, color):
        mask = np.zeros((32, 32), dtype=np.uint16)
        for index in range(count):  # disjoint 2x2 nuclei on a 6-pixel grid
            y, x = divmod(index, 5)
            mask[6 * y + 1:6 * y + 3, 6 * x + 1:6 * x + 3] = index + 1
        Image.new("RGB", (32, 32), color).save(self.root / f"{ident}.png")
        Image.fromarray(mask).save(self.root / f"{ident}-mask.png")
        return {"id": ident, "image_path": f"{ident}.png", "mask_path": f"{ident}-mask.png", "instance_count": count}

    def monuseg(self, ident, source, split, count=2, color=(10, 20, 30)):
        return {**self.crop(ident, count, color), "split": split, "source_image_id": source,
                "patient_id": "-".join(source.split("-")[:3])}

    def external(self, ident, dataset, source, member, count=3, color=(40, 50, 60)):
        return {**self.crop(ident, count, color), "split": "train", "dataset": dataset, "source_image_id": source,
                "patient_id": source, "source_image_member": member}

    def write(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(self.root / name)

    def mixed_train(self, extra=()):
        return self.write("train.jsonl", [
            self.monuseg("mono-train", self.sources[0], "train"),
            self.external("consep_train_3_x0_y0", "consep", "consep_train_3", "CoNSeP/Train/Images/train_3.png",
                          color=(70, 80, 90)),
            self.external("puma_primary_roi_001_x0_y0", "puma", "puma_primary_roi_001",
                          "01_training_dataset_tif_ROIs/training_set_primary_roi_001.tif", color=(90, 80, 70)),
            *extra])

    def run_main(self, train, validation, *flags):
        output = self.root / f"out-{len(list(self.root.glob('out-*')))}"
        captured = {}

        def fake_run(args, rows, val, audit, out):
            captured.update(args=args, rows=rows, validation=val, audit=audit)
            return "ran"

        with patch("nucleus_rl.sft.run", side_effect=fake_run):
            result = main(["--manifest", train, "--eval-manifest", validation, "--output", str(output),
                           "--model", "m", "--sam-checkpoint", str(self.root / "sam.pt"), *flags])
        self.assertEqual(result, "ran")
        return captured, json.loads((output / "preflight.json").read_text())

    def test_mixed_train_manifest_passes_preflight_and_reports_datasets(self):
        validation = self.write("val.jsonl", [self.monuseg("mono-val", self.sources[1], "validation", color=(1, 2, 3))])
        captured, preflight = self.run_main(self.mixed_train(), validation, "--skip-before-eval")
        self.assertEqual(preflight["train_datasets"], {"consep": 1, "monuseg": 1, "puma": 1})
        self.assertEqual(preflight["validation_datasets"], {"monuseg": 1})
        self.assertIs(preflight["skip_before_eval"], True)
        self.assertEqual(set(captured["audit"]["train_source_image_ids"]),
                         {self.sources[0], "consep_train_3", "puma_primary_roi_001"})
        self.assertTrue(captured["args"].skip_before_eval)

    def test_external_validation_rows_are_accepted(self):
        consep_val = {**self.external("consep_train_7_x0_y0", "consep", "consep_train_7",
                                      "CoNSeP/Train/Images/train_7.png", color=(5, 6, 7)), "split": "validation"}
        validation = self.write("val.jsonl", [consep_val])
        _, preflight = self.run_main(self.mixed_train(), validation)
        self.assertEqual(preflight["validation_datasets"], {"consep": 1})
        self.assertIs(preflight["skip_before_eval"], False)

    def test_cohort_protocol_still_rejects_external_rows(self):
        validation = self.write("val.jsonl", [self.monuseg("mono-val", self.sources[1], "validation", color=(1, 2, 3))])
        with self.assertRaisesRegex(ValueError, "official30"):
            self.run_main(self.mixed_train(), validation, "--cohort-protocol", str(DEFAULT_COHORT_PATH))

    def test_max_objects_8_and_32_with_dense_validation(self):
        dense_val = self.write("val.jsonl", [self.monuseg("dense-val", self.sources[1], "validation", count=20,
                                                          color=(1, 2, 3))])
        sparse_train = self.mixed_train()
        for cap in ("8", "32"):
            captured, _ = self.run_main(sparse_train, dense_val, "--max-objects", cap)
            self.assertEqual(captured["args"].max_objects, int(cap))
        # A training crop above the cap has no valid target and fails before GPU loading.
        dense_train = self.mixed_train([self.monuseg("dense-train", self.sources[2], "train", count=12, color=(9, 9, 9))])
        with self.assertRaisesRegex(ValueError, "allowed 1..8"):
            self.run_main(dense_train, dense_val, "--max-objects", "8")
        captured, _ = self.run_main(dense_train, dense_val, "--max-objects", "32")
        self.assertEqual(len(captured["rows"]), 4)
        with self.assertRaises(ValueError):
            self.run_main(sparse_train, dense_val, "--max-objects", "33")


class DatasetCountTests(unittest.TestCase):
    def test_rows_without_dataset_count_as_monuseg(self):
        self.assertEqual(dataset_counts([{}, {"dataset": "puma"}, {"dataset": "monuseg"}]), {"monuseg": 2, "puma": 1})


if __name__ == "__main__":
    unittest.main()
