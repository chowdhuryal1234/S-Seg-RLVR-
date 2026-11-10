"""CPU tests for the NuNext reproduction: tokens, matching, augmentation, data prep, GRPO shaping."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nucleus_rl.nunext import coords
from nucleus_rl.nunext.data import d4_transform
from nucleus_rl.nunext.grpo import fgas_token_advantages, group_advantages, keep_group

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class CoordinateTokens(unittest.TestCase):
    def test_quantize_round_trip_within_half_bin(self):
        bins, size = 256, 256.0
        for value in (0.0, 0.5, 17.3, 128.0, 255.5, 256.0):
            index = coords.quantize(value, size, bins)
            self.assertLessEqual(abs(coords.dequantize(index, size, bins) - value), size / (bins - 1) / 2 + 1e-9)

    def test_quantize_clamps(self):
        self.assertEqual(coords.quantize(-5, 100, 10), 0)
        self.assertEqual(coords.quantize(500, 100, 10), 9)

    def test_raster_order_sorts_by_row_then_column(self):
        points = [(200, 10), (10, 200), (5, 10)]
        self.assertEqual(coords.raster_order(points, 255, 255, 256), [(5, 10), (200, 10), (10, 200)])

    def test_encode_then_parse_text(self):
        text = coords.encode_points([(10, 20), (30, 5)], 256, 256, 256)
        parsed = coords.parse_text(text, 256)
        self.assertTrue(parsed.valid)
        self.assertEqual(len(parsed.bins), 2)
        self.assertEqual(parsed.bins[0][1], coords.quantize(5, 256, 256))  # y=5 comes first

    def test_parse_token_ids(self):
        first, bins, end = 1000, 16, 7
        ok = coords.parse_token_ids([1000, 1003, 1015, 1000, end, 99], first, bins, [end])
        self.assertTrue(ok.valid)
        self.assertEqual(ok.bins, ((0, 3), (15, 0)))
        self.assertTrue(coords.parse_token_ids([end], first, bins, [end]).valid)  # empty image
        self.assertFalse(coords.parse_token_ids([1000, end], first, bins, [end]).valid)  # odd
        self.assertFalse(coords.parse_token_ids([1000, 5, end], first, bins, [end]).valid)  # text token
        self.assertFalse(coords.parse_token_ids([1000, 1001], first, bins, [end]).valid)  # truncated
        self.assertFalse(coords.parse_token_ids([1016, 1000, end], first, bins, [end]).valid)  # out of range

    def test_parse_text_rejects_extra_text(self):
        self.assertFalse(coords.parse_text("<|coord_1|><|coord_2|> nucleus", 256).valid)


@unittest.skipUnless(HAS_SCIPY, "scipy required")
class Matching(unittest.TestCase):
    def test_hungarian_then_threshold(self):
        from nucleus_rl.nunext.matching import match_points
        gt = [(10, 10), (50, 50)]
        pred = [(11, 10), (80, 80), (49, 52)]
        match = match_points(pred, gt, radius=6)
        self.assertEqual((match.tp, match.fp, match.fn), (2, 1, 0))
        self.assertEqual(match.pred_is_tp, (True, False, True))
        self.assertAlmostEqual(match.f1, 0.8)

    def test_one_to_one(self):
        from nucleus_rl.nunext.matching import match_points
        match = match_points([(10, 10), (10, 11)], [(10, 10)], radius=6)
        self.assertEqual((match.tp, match.fp, match.fn), (1, 1, 0))

    def test_empty_cases(self):
        from nucleus_rl.nunext.matching import match_points
        self.assertEqual(match_points([], [], 6).f1, 1.0)
        self.assertEqual(match_points([], [(1, 1)], 6).f1, 0.0)
        self.assertEqual(match_points([(1, 1)], [], 6).f1, 0.0)

    def test_f1_equals_detection_quality(self):
        from nucleus_rl.nunext.matching import Match
        m = Match(7, 3, 5)
        self.assertAlmostEqual(m.f1, m.tp / (m.tp + 0.5 * m.fp + 0.5 * m.fn))

    def test_aggregate_micro(self):
        from nucleus_rl.nunext.matching import Match, aggregate
        summary = aggregate([Match(1, 0, 1), Match(3, 1, 0)])
        self.assertAlmostEqual(summary["micro_f1"], 8 / (8 + 1 + 1))


class D4(unittest.TestCase):
    def test_marked_pixel_follows_its_point(self):
        size = 8
        for view in range(8):
            pixels = np.zeros((size, size, 3), dtype=np.uint8)
            row, col = 1, 5
            pixels[row, col] = 255
            out, points = d4_transform(pixels, [(col + 0.5, row + 0.5)], view)
            x, y = points[0]
            ys, xs = np.nonzero(out[:, :, 0])
            self.assertEqual((xs[0] + 0.5, ys[0] + 0.5), (x, y), f"view {view}")

    def test_views_are_distinct(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        outputs = {d4_transform(pixels, [], view)[0].tobytes() for view in range(8)}
        self.assertEqual(len(outputs), 8)


class GRPOShaping(unittest.TestCase):
    def test_group_advantages_zero_mean(self):
        advantages = group_advantages([0.1, 0.5, 0.9])
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=9)
        self.assertGreater(advantages[2], 0)

    def test_low_variance_filter(self):
        self.assertFalse(keep_group([0.2, 0.2, 0.2], 0.0))
        self.assertTrue(keep_group([0.2, 0.2, 0.21], 0.0))
        self.assertFalse(keep_group([0.2, 0.2, 0.21], 0.01))
        self.assertTrue(keep_group([0.0, 0.5, 1.0], 0.01))

    def test_paper_example_amplification(self):
        # Fig. 3 (left): without LVGF the 0.21 rollout outranks the 1.0 rollout of a wide group.
        narrow = group_advantages([0.20, 0.20, 0.21])
        wide = group_advantages([0.0, 0.5, 1.0])
        self.assertGreater(narrow.max(), wide.max())

    def test_fgas_positive_advantage_decays_false_positives(self):
        shaped = fgas_token_advantages(2.0, 5, True, [True, False], 0.5)
        np.testing.assert_allclose(shaped, [2.0, 2.0, 1.0, 1.0, 2.0])

    def test_fgas_negative_advantage_decays_true_positives(self):
        shaped = fgas_token_advantages(-2.0, 5, True, [True, False], 0.5)
        np.testing.assert_allclose(shaped, [-1.0, -1.0, -2.0, -2.0, -2.0])

    def test_fgas_off_or_invalid(self):
        np.testing.assert_allclose(fgas_token_advantages(1.0, 3, True, [False], 1.0), [1, 1, 1])
        np.testing.assert_allclose(fgas_token_advantages(1.0, 3, False, [], 0.5), [1, 1, 1])


@unittest.skipUnless(HAS_TORCH, "torch required")
class SoftLabels(unittest.TestCase):
    def test_sigma_zero_is_cross_entropy(self):
        import torch
        from nucleus_rl.nunext.modeling import gaussian_targets, soft_label_loss
        torch.manual_seed(0)
        vocab, first, bins = 40, 20, 16
        logits = torch.randn(5, vocab)
        labels = torch.tensor([20, 25, 35, 21, 3])
        expected = torch.nn.functional.cross_entropy(logits, labels)
        actual = soft_label_loss(logits, labels, first, bins, gaussian_targets(bins, 0.0))
        self.assertAlmostEqual(float(actual), float(expected), places=5)

    def test_soft_labels_credit_neighbours(self):
        import torch
        from nucleus_rl.nunext.modeling import gaussian_targets, soft_label_loss
        vocab, first, bins = 40, 20, 16
        labels = torch.tensor([28])
        near, far = torch.full((1, vocab), -10.0), torch.full((1, vocab), -10.0)
        near[0, 29] = 10.0   # one bin off
        far[0, 35] = 10.0    # seven bins off
        kernel = gaussian_targets(bins, 1.5)
        self.assertLess(float(soft_label_loss(near, labels, first, bins, kernel)),
                        float(soft_label_loss(far, labels, first, bins, kernel)))
        self.assertTrue(torch.allclose(kernel.sum(1), torch.ones(bins)))


class PanNukePreparation(unittest.TestCase):
    def test_convert_fold_writes_centroids(self):
        from scripts.prepare_pannuke import convert_fold, instance_centroids
        mask = np.zeros((16, 16, 6), dtype=np.float64)
        mask[2:4, 2:4, 0] = 1          # neoplastic, centre (3.5, 3.5) in continuous coords
        mask[10:12, 6:8, 4] = 7        # epithelial
        mask[10:11, 12:13, 1] = 3      # one-pixel inflammatory
        centroids, classes = instance_centroids(mask)
        self.assertEqual(centroids[0], [3.0, 3.0])
        self.assertEqual(set(classes), {"neoplastic", "epithelial", "inflammatory"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Fold 1" / "images" / "fold1").mkdir(parents=True)
            (root / "Fold 1" / "masks" / "fold1").mkdir(parents=True)
            np.save(root / "Fold 1" / "images" / "fold1" / "images.npy", np.zeros((2, 16, 16, 3)))
            np.save(root / "Fold 1" / "masks" / "fold1" / "masks.npy", np.stack([mask, np.zeros_like(mask)]))
            report = convert_fold(root / "Fold 1", 1, root / "out")
            rows = [json.loads(line) for line in (root / "out" / "fold1.jsonl").read_text().splitlines()]
        self.assertEqual(report["nuclei"], 3)
        self.assertEqual([len(r["centroids"]) for r in rows], [3, 0])


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow required")
class PanNukeParquet(unittest.TestCase):
    def test_parquet_matches_npy_conversion(self):
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        from scripts.prepare_pannuke import convert_fold, convert_parquet

        mask = np.zeros((16, 16, 6))
        mask[2:4, 2:4, 0] = 1
        mask[10:12, 6:9, 4] = 7
        mask[10:11, 12:13, 1] = 3

        def png(array, mode):
            buffer = io.BytesIO()
            Image.fromarray(array).convert(mode).save(buffer, format="PNG")
            return {"bytes": buffer.getvalue(), "path": None}

        instances, categories = [], []
        for channel in range(5):  # same split as RationAI's gen_script.py
            for value in np.unique(mask[..., channel])[1:]:
                instances.append(png((mask[..., channel] == value).astype(np.uint8) * 255, "1"))
                categories.append(channel)
        table = pa.table({"image": [png(np.zeros((16, 16, 3), np.uint8), "RGB")],
                          "instances": [instances], "categories": [categories], "tissue": [0]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(table, root / "fold1.parquet")
            (root / "npy" / "f").mkdir(parents=True)
            np.save(root / "npy" / "f" / "images.npy", np.zeros((1, 16, 16, 3)))
            np.save(root / "npy" / "f" / "masks.npy", mask[None])
            convert_parquet(root / "fold1.parquet", 1, root / "a")
            convert_fold(root / "npy", 1, root / "b")
            a = json.loads((root / "a" / "fold1.jsonl").read_text())
            b = json.loads((root / "b" / "fold1.jsonl").read_text())
        self.assertEqual(a["centroids"], b["centroids"])
        self.assertEqual(a["classes"], b["classes"])


class CommandLines(unittest.TestCase):
    def test_parsers_build(self):
        from nucleus_rl.nunext import grpo, sft
        sft.build_parser().parse_args(["--train-manifest", "a", "--eval-manifest", "b", "--output", "o"])
        grpo.build_parser().parse_args(["--init-model", "m", "--train-manifest", "a", "--eval-manifest", "b",
                                        "--output", "o"])

    def test_ranks_interleave_without_overlap(self):
        from nucleus_rl.nunext.sft import exposure_index
        seen = sorted(exposure_index(m, r, 4) for m in range(5) for r in range(4))
        self.assertEqual(seen, list(range(20)))  # 4 ranks x 5 micro-steps = single-process order 0..19

    def test_dist_helpers_are_noops_without_torchrun(self):
        from nucleus_rl.nunext import dist
        self.assertFalse(dist.enabled())
        self.assertEqual(dist.gather_objects([1, 2]), [1, 2])
        self.assertEqual(dist.mean(3.5), 3.5)
        self.assertTrue(dist.any_true(True))

    def test_launcher_sets_ranks_and_stops_on_failure(self):
        import subprocess, sys
        ok = subprocess.run([sys.executable, "-m", "nucleus_rl.nunext.launch", "--nproc", "2", "--", "-c",
                             "import os; assert os.environ['WORLD_SIZE'] == '2'; "
                             "assert os.environ['NUNEXT_DIST_INIT'].startswith('file://')"],
                            capture_output=True, text=True, timeout=120)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        failing = subprocess.run([sys.executable, "-m", "nucleus_rl.nunext.launch", "--nproc", "2", "--", "-c",
                                  "import os, sys, time; r = int(os.environ['RANK']); "
                                  "sys.exit(5) if r == 1 else time.sleep(60)"],
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(failing.returncode, 5)
        self.assertIn("rank 1 exited with code 5", failing.stdout)

    def test_lr_schedule(self):
        from nucleus_rl.nunext.sft import lr_at
        self.assertAlmostEqual(lr_at(0, 100, 10, 1.0), 0.1)
        self.assertAlmostEqual(lr_at(10, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(lr_at(100, 100, 10, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
