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


if __name__ == "__main__":
    unittest.main()
