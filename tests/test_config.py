"""Check the pilot's resource and parameter-scope safeguards without torch."""
import unittest
import json
from pathlib import Path
import tempfile

from nucleus_rl.train import build_parser, validate_args, text_lora_targets, assert_text_adapters_only


class FakeParameter:
    def __init__(self, trainable):
        self.requires_grad = trainable


class FakeModel:
    def named_modules(self):
        return [(name, object()) for name in (
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.visual.layers.0.self_attn.q_proj",
        )]

    def named_parameters(self):
        return [
            ("model.language_model.layers.0.self_attn.q_proj.lora_A.weight", FakeParameter(True)),
            ("model.visual.weight", FakeParameter(False)),
        ]


class ConfigTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--mode", "train", "--manifest", "train.jsonl", "--eval-manifest", "val.jsonl",
            "--output", "run", "--sam-checkpoint", "sam.pt", *extra,
        ])

    def test_default_small_group_validates(self):
        args = self.args()
        validate_args(args)
        self.assertEqual((args.group_size, args.max_steps, args.beta), (2, 10, 0.04))
        self.assertEqual(args.coordinate_frame, "processed")

    def test_wrong_group_rejected(self):
        with self.assertRaises(ValueError):
            validate_args(self.args("--group-size", "1"))

    def test_training_requires_separate_validation(self):
        args = self.args()
        args.eval_manifest = None
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_finite_learning_rate_required(self):
        with self.assertRaises(ValueError):
            validate_args(self.args("--learning-rate", "nan"))

    def test_lora_targeting_excludes_visual_and_other_projections(self):
        targets = text_lora_targets(FakeModel())
        self.assertEqual(len(targets), 2)
        self.assertTrue(all("language_model" in name for name in targets))

    def test_trainable_visual_parameter_rejected(self):
        model = FakeModel()
        assert_text_adapters_only(model)
        model.named_parameters = lambda: [("model.visual.lora_A.weight", FakeParameter(True))]
        with self.assertRaises(RuntimeError):
            assert_text_adapters_only(model)

    def test_artifacts_preserve_valid_and_invalid_predictions(self):
        import numpy as np
        from PIL import Image
        from nucleus_rl.evaluate import ArtifactScorer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.zeros((8, 8), dtype=np.uint16)
            reference[2:5, 2:5] = 300
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "image.png")
            Image.fromarray(reference).save(root / "mask.png")
            row = {"id": "a", "patient_id": "p1", "image_path": str(root / "image.png"), "mask_path": str(root / "mask.png")}

            class FakeSegmenter:
                def predict(self, image, objects, image_key=None):
                    return reference.astype(np.int32)

            scorer = ArtifactScorer(FakeSegmenter(), root / "artifacts")
            valid = scorer.score('<answer>{"objects":[{"box":[2,2,5,5],"point":[3,3]}]}</answer>', row)
            invalid = scorer.score("broken JSON", row)
            scorer.record_group([valid, invalid])
            self.assertEqual(valid["total_reward"], 2.0)
            self.assertEqual(invalid["total_reward"], 0.0)
            self.assertFalse(np.load(invalid["prediction_path"]).any())
            self.assertTrue(Path(valid["overlay_path"]).is_file())
            summary = scorer.summary()
            self.assertEqual(summary["invalid_completions"], 1)
            self.assertEqual(summary["groups_with_reward_variation"], 1)
            saved = [json.loads(line) for line in (root / "artifacts" / "completions.jsonl").read_text().splitlines()]
            self.assertEqual(saved[0]["gt_count"], 1)

    def test_processed_coordinates_map_once_and_preserve_segmentation(self):
        import numpy as np
        from PIL import Image
        from nucleus_rl.evaluate import ArtifactScorer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.zeros((128, 128), dtype=np.uint16)
            reference[64:112, 80:120] = 1
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(root / "image.png")
            Image.fromarray(reference).save(root / "mask.png")
            original = {"id": "a", "patient_id": "p1", "image_path": str(root / "image.png"), "mask_path": str(root / "mask.png")}
            processed = {**original, "coordinate_frame": "processed", "original_width": 128, "original_height": 128,
                         "processed_width": 224, "processed_height": 224, "prompt_width": 224, "prompt_height": 224}

            class RecordingSegmenter:
                def __init__(self):
                    self.calls = []

                def predict(self, image, objects, image_key=None):
                    self.calls.append(objects)
                    output = np.zeros((image.height, image.width), dtype=np.int32)
                    for i, obj in enumerate(objects, 1):
                        x0, y0, x1, y1 = (round(value) for value in obj.box)
                        output[y0:y1, x0:x1] = i
                    return output

            segmenter = RecordingSegmenter()
            scorer = ArtifactScorer(segmenter, root / "artifacts")
            processed_result = scorer.score('<answer>{"objects":[{"box":[140,112,210,196],"point":[175,154]}]}</answer>', processed)
            original_result = scorer.score('<answer>{"objects":[{"box":[80,64,120,112],"point":[100,88]}]}</answer>', original)
            self.assertEqual(segmenter.calls[0], segmenter.calls[1])
            self.assertEqual(segmenter.calls[0][0].box, (80, 64, 120, 112))
            self.assertEqual(segmenter.calls[0][0].point, (100, 88))
            self.assertEqual(processed_result["total_reward"], 2.0)
            self.assertEqual(processed_result["pq"], original_result["pq"])
            np.testing.assert_array_equal(np.load(processed_result["prediction_path"]), np.load(original_result["prediction_path"]))
            self.assertEqual(processed_result["raw_objects"][0]["box"], (140, 112, 210, 196))
            self.assertEqual(processed_result["converted_objects"][0]["box"], (80, 64, 120, 112))
            invalid = scorer.score('<answer>{"objects":[{"box":[140,112,225,196],"point":[175,154]}]}</answer>', processed)
            self.assertEqual(invalid["total_reward"], 0.0)
            self.assertEqual(len(segmenter.calls), 2, "Invalid processed coordinates must never be clipped into valid SAM prompts")

    def test_nonsquare_coordinate_mapping_preserves_xy_axes(self):
        from nucleus_rl.evaluate import objects_to_original
        from nucleus_rl.rewards import ObjectPrompt

        objects = (ObjectPrompt((4, 3, 12, 9), (8, 6)),)
        converted, scales = objects_to_original(objects, (16, 12), (8, 4))
        self.assertEqual(scales, (.5, 1 / 3))
        self.assertEqual(converted[0].box, (2, 1, 6, 3))
        self.assertEqual(converted[0].point, (4, 2))

    def test_frame_preparation_uses_actual_processor_grid_and_detects_drift(self):
        import numpy as np
        from PIL import Image
        from nucleus_rl.evaluate import prepare_coordinate_rows, assert_processed_dimensions, prompt_size_for_row

        class ImageProcessor:
            patch_size = 14

            def __call__(self, images, return_tensors):
                return {"image_grid_thw": np.asarray([[1, 16, 20]])}

        class Processor:
            image_processor = ImageProcessor()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (160, 128)).save(path)
            # A mask path is deliberately absent: choosing a frame must not inspect GT.
            rows = prepare_coordinate_rows([{"id": "x", "image_path": str(path)}], Processor(), "processed")
            self.assertEqual(prompt_size_for_row(rows[0], (160, 128)), (280, 224))
            self.assertEqual((rows[0]["original_width"], rows[0]["original_height"]), (160, 128))
            assert_processed_dimensions(Processor(), {"image_grid_thw": [[1, 16, 20]]}, rows[0])
            with self.assertRaises(ValueError):
                assert_processed_dimensions(Processor(), {"image_grid_thw": [[1, 20, 16]]}, rows[0])
            original = prepare_coordinate_rows([{"image_path": str(path)}], Processor(), "original")[0]
            self.assertEqual(prompt_size_for_row(original, (160, 128)), (160, 128))


if __name__ == "__main__":
    unittest.main()
