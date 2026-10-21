"""CPU checks of the paired PQ bootstrap used for the GRPO-from-SFT comparisons."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_pq import compare, completions_path


def write_completions(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"crop{i}", "pq": value}) + "\n" for i, value in enumerate(values)))
    return path


class BootstrapTests(unittest.TestCase):
    def test_identical_evaluations_are_within_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_completions(Path(tmp) / "a.jsonl", [0.2, 0.5, 0.7, 0.4])
            row = compare("same", a, a, draws=200)
        self.assertEqual((row["mean"], row["low"], row["high"]), (0.0, 0.0, 0.0))
        self.assertEqual(row["verdict"], "within noise")

    def test_consistent_gain_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = [0.1 * (i % 7) for i in range(40)]
            a = write_completions(Path(tmp) / "a.jsonl", [v + 0.05 for v in base])
            b = write_completions(Path(tmp) / "b.jsonl", base)
            row = compare("gain", a, b, draws=500)
        self.assertAlmostEqual(row["mean"], 0.05)
        self.assertGreater(row["low"], 0)
        self.assertEqual((row["better"], row["worse"], row["verdict"]), (40, 0, "clear"))

    def test_rejects_different_crop_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_completions(Path(tmp) / "a.jsonl", [0.1, 0.2])
            b = write_completions(Path(tmp) / "b.jsonl", [0.1, 0.2, 0.3])
            with self.assertRaises(ValueError):
                compare("mismatch", a, b, draws=10)

    def test_finds_runs_in_either_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            grpo, sft = Path(tmp) / "grpo", Path(tmp) / "sft"
            write_completions(sft / "old" / "after" / "completions.jsonl", [0.1])
            self.assertEqual(completions_path([grpo, sft], "old", "after"), sft / "old" / "after" / "completions.jsonl")
            self.assertEqual(completions_path([grpo, sft], "new", "after"), grpo / "new" / "after" / "completions.jsonl")


if __name__ == "__main__":
    unittest.main()
