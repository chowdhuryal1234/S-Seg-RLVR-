"""Controls specific to the paired formatting experiment, without GPU imports."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import prompt_for_image
from nucleus_rl.prompt_ablation import paired_prompts, select_fixed_rows
from nucleus_rl.rewards import parse_completion


class PromptAblationTests(unittest.TestCase):
    def test_only_numeric_example_changes_and_coordinates_are_valid(self):
        for width, height in ((224, 224), (280, 224), (128, 128)):
            prompts = paired_prompts(width, height)
            self.assertEqual(prompts["one_example"], prompt_for_image(width, height, 8, "processed", "example"))
            self.assertEqual(prompts["one_example"][0], prompts["three_examples"][0])
            parsed = {}
            normalized = {}
            for variant, messages in prompts.items():
                text = messages[-1]["content"]
                start = text.index("Formatting example with invented coordinates, unrelated to the supplied image:\n")
                concrete = re.findall(r"<answer>.*?</answer>", text[start:])[0]
                parsed[variant] = parse_completion(concrete, width=width, height=height)
                self.assertTrue(parsed[variant].valid)
                normalized[variant] = text.replace(concrete, "NUMERIC_EXAMPLE")
            self.assertEqual(normalized["one_example"], normalized["three_examples"])
            self.assertEqual(len(parsed["one_example"].objects), 1)
            self.assertEqual(len(parsed["three_examples"].objects), 3)
            self.assertEqual(parsed["one_example"].objects[0], parsed["three_examples"].objects[0])
            self.assertEqual(len(set(obj.box for obj in parsed["three_examples"].objects)), 3)

    def test_selection_is_input_order_independent_and_uses_count_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = load_official30_cohort()["image_ids"]
            rows = []
            for index, count in enumerate((8, 2, 4, 5, 6, 7, 4, 5, 8, 6)):
                labels = np.zeros((16, 16), dtype=np.uint16)
                for n in range(count):
                    labels[n, n] = n + 1
                path = root / f"{index}.png"
                Image.fromarray(labels).save(path)
                source = ids[index]
                rows.append({"id": source + "_x0_y0", "source_image_id": source,
                             "patient_id": "-".join(source.split("-")[:3]), "split": "train",
                             "mask_path": str(path), "instance_count": count})
            a = select_fixed_rows(rows)
            b = select_fixed_rows(list(reversed(rows)))
            self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
            self.assertEqual([r["diagnostic_gt_count"] for r in a[:6]], [2, 4, 5, 6, 7, 8])
            self.assertEqual(len(a), 8)
            bad = copy.deepcopy(rows)
            bad[0]["split"] = "validation"
            with self.assertRaises(ValueError):
                select_fixed_rows(bad)
            bad = copy.deepcopy(rows)
            bad[0]["instance_count"] = 1
            with self.assertRaisesRegex(ValueError, "Unexpected visible instance count"):
                select_fixed_rows(bad)


if __name__ == "__main__":
    unittest.main()
