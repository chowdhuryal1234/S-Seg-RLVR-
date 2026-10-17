"""Normalized coordinates remain distinct from measured processor geometry."""
import unittest

from nucleus_rl.evaluate import objects_to_original, prompt_size_for_row, prompt_for_image
from nucleus_rl.rewards import ObjectPrompt


class NormalizedCoordinatesTests(unittest.TestCase):
    def test_scale_back_to_original_from_normalized_frame(self):
        row = dict(coordinate_frame="normalized1000", original_width=128, original_height=64,
                   prompt_width=1000, prompt_height=1000, processed_width=256, processed_height=128)
        frame = prompt_size_for_row(row, (128, 64))
        objects, _ = objects_to_original([ObjectPrompt((250, 250, 500, 750), (375, 500))], frame, (128, 64))
        self.assertEqual(objects[0].box, (32, 16, 64, 48))
        self.assertEqual(objects[0].point, (48, 32))

    def test_reject_accidental_processed_dimensions_as_normalized(self):
        row = dict(coordinate_frame="normalized1000", prompt_width=256, prompt_height=256,
                   processed_width=256, processed_height=256)
        with self.assertRaises(ValueError):
            prompt_size_for_row(row, (128, 128))

    def test_normalized_prompt_declares_fixed_frame(self):
        text = str(prompt_for_image(1000, 1000, coordinate_frame="normalized1000"))
        self.assertIn("NORMALIZED", text)
        with self.assertRaises(ValueError):
            prompt_for_image(224, 224, coordinate_frame="normalized1000")


if __name__ == "__main__":
    unittest.main()
