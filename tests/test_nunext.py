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

    def test_salvage_cuts_loops_and_truncation(self):
        first, bins, end = 1000, 16, 7
        c = lambda x, y: [first + x, first + y]
        looping = c(1, 1) + c(5, 1) + c(2, 3) + c(2, 3) + c(9, 4) + c(1, 1) + c(5, 1) + c(2, 3) + [first + 3]
        salvaged = coords.salvage_token_ids(looping, first, bins, [end])
        self.assertEqual(salvaged.bins, ((1, 1), (5, 1), (2, 3), (9, 4)))   # repeat dropped, loop cut
        self.assertFalse(coords.parse_token_ids(looping, first, bins, [end]).valid)
        valid = c(1, 1) + c(2, 3) + [end]
        self.assertEqual(coords.salvage_token_ids(valid, first, bins, [end]).bins,
                         coords.parse_token_ids(valid, first, bins, [end]).bins)

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


class CoordinateInit(unittest.TestCase):
    def test_digit_codes_are_distinct_and_ordered(self):
        from nucleus_rl.nunext.modeling import digit_weights
        codes = [tuple(d for d, _ in digit_weights(k, 256)) for k in range(256)]
        self.assertEqual(len(set(codes)), 256)
        self.assertEqual(codes, sorted(codes))
        self.assertEqual(codes[0], ("0", "0", "0"))
        self.assertEqual(codes[-1], ("9", "9", "9"))


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


class ParallelRuns(unittest.TestCase):
    def test_on_gpu_sets_visible_devices(self):
        import subprocess, sys
        out = subprocess.run([sys.executable, "-m", "nucleus_rl.nunext.on_gpu", "--gpu", "5", "--", "-c",
                              "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "5")

    def test_run_parallel_assigns_gpus_in_waves_and_reports_failures(self):
        import sys
        from unittest import mock
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dbx"))
        import nunext_common as nx
        calls = []

        def fake_run_cli(argv, local, remote):
            calls.append((argv[argv.index("--gpu") + 1], remote))
            if remote == "r3":
                raise RuntimeError("boom")

        jobs = [(["-m", "x"], Path(f"l{i}"), f"r{i}") for i in range(5)]
        with mock.patch.object(nx.dbx, "run_cli", fake_run_cli), mock.patch.object(nx.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError):
                nx.run_parallel(jobs, gpus=2)
        self.assertEqual(sorted(calls, key=lambda c: c[1]),
                         [("0", "r0"), ("1", "r1"), ("0", "r2"), ("1", "r3"), ("0", "r4")])


class PanopticQuality(unittest.TestCase):
    def test_best_masks_restores_a_squeezed_batch_and_picks_the_top_score(self):
        from nucleus_rl.nunext.pq_eval import best_masks
        masks = np.zeros((3, 4, 4)); masks[1, 0, 0] = 1
        picked, scores = best_masks(masks, np.array([0.1, 0.9, 0.5]))
        self.assertEqual(picked.shape, (1, 4, 4))
        self.assertTrue(picked[0, 0, 0])
        self.assertAlmostEqual(float(scores[0]), 0.9)

    def test_smallest_mask_choice(self):
        from nucleus_rl.nunext.pq_eval import best_masks
        masks = np.zeros((1, 3, 6, 6))
        masks[0, 0, :3, :3] = 1      # 9 px, low score
        masks[0, 1, :, :] = 1        # whole image, highest score
        picked, _ = best_masks(masks, np.array([[0.2, 0.9, 0.1]]), "smallest")
        self.assertEqual(int(picked[0].sum()), 9)
        empty, _ = best_masks(np.zeros((1, 3, 6, 6)), np.array([[0.2, 0.9, 0.1]]), "smallest")
        self.assertEqual(int(empty[0].sum()), 0)

    def test_negative_prompts(self):
        from nucleus_rl.nunext.pq_eval import negative_points, prompt_arrays
        points = [(0, 0), (3, 4), (10, 0), (100, 100)]
        negatives = negative_points(points, 2, 12)
        self.assertEqual(negatives[0], [(3.0, 4.0), (10.0, 0.0)])
        self.assertEqual(negatives[3], [])               # nobody within 12 px
        coords, labels = prompt_arrays(points, negatives)
        self.assertEqual(coords.shape, (4, 3, 2))
        self.assertEqual(labels.tolist()[0], [1, 0, 0])
        self.assertEqual(labels.tolist()[3], [1, -1, -1])  # padded, not negatives
        self.assertIsNone(negative_points(points, 0, 12))
        self.assertEqual(prompt_arrays(points, None)[1].shape, (4, 1))

    def test_pseudo_boxes(self):
        from nucleus_rl.nunext.pq_eval import pseudo_boxes
        boxes = pseudo_boxes([(20, 20), (40, 20), (250, 250)], 256, 256, 0.7, 8, 12)
        self.assertEqual(boxes[0], (8.0, 8.0, 32.0, 32.0))          # 0.7 x 20 = 14 -> capped at 12
        self.assertEqual(boxes[2], (238.0, 238.0, 256.0, 256.0))    # clipped to the image
        tight = pseudo_boxes([(20, 20), (30, 20)], 256, 256, 0.7, 8, 12)
        self.assertEqual(tight[0], (12.0, 12.0, 28.0, 28.0))        # 0.7 x 10 = 7 -> raised to 8
        self.assertEqual(pseudo_boxes([(5, 5)], 64, 64, 0.7, 8, 12), [(0.0, 0.0, 17.0, 17.0)])  # alone: max

    def test_point_prompts_to_pq(self):
        from PIL import Image
        from nucleus_rl.nunext.pq_eval import score_image, summarize
        reference = np.zeros((16, 16), dtype=np.uint16)
        reference[2:6, 2:6] = 1
        reference[9:14, 8:13] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(root / "i.png")
            Image.fromarray(reference).save(root / "m.png")
            row = {"id": "a", "image_path": str(root / "i.png"), "mask_path": str(root / "m.png"),
                   "centroids": [[4.0, 4.0], [10.5, 11.5]]}
            ref = reference.astype(np.int64)

            def segment(image, points, boxes, negatives=None):
                return np.stack([ref == ref[int(y), int(x)] for x, y in points]), np.ones(len(points))

            exact = score_image(row, "oracle-point", None, segment)
            missed = score_image(row, "pred-point", {"a": [[4.0, 4.0]]}, segment)
        self.assertEqual((exact["tp"], exact["fp"], exact["fn"]), (2, 0, 0))
        self.assertAlmostEqual(summarize([exact])["pq"], 1.0)
        self.assertEqual((missed["tp"], missed["fn"]), (1, 1))
        self.assertAlmostEqual(summarize([missed])["pq"], 1 / 1.5)


class BoxTokens(unittest.TestCase):
    def test_points_encode_identically(self):
        points = [(200, 10), (10, 200), (5, 10), (5, 10)]
        self.assertEqual(coords.encode_objects(points, 256, 256, 256), coords.encode_points(points, 256, 256, 256))

    def test_boxes_sort_by_centre_and_round_trip(self):
        objects = [(100.0, 50.0, 20.0, 10.0), (30.0, 20.0, 8.0, 16.0)]
        text = coords.encode_objects(objects, 256, 256, 256)
        values = [int(v) for v in coords._TOKEN_PATTERN.findall(text)]
        self.assertEqual(len(values), 8)
        first = 1000
        end = 2000
        parsed = coords.parse_token_ids([first + v for v in values] + [end], first, 256, [end], per_object=4)
        self.assertTrue(parsed.valid)
        self.assertEqual(len(parsed.bins), 2)
        self.assertEqual(parsed.bins[0][:2], (coords.quantize(30, 256, 256), coords.quantize(20, 256, 256)))
        (x0, y0, x1, y1), _ = coords.bins_to_boxes(parsed.bins, 256, 256, 256)
        self.assertAlmostEqual(x1 - x0, 8.0, delta=1.1)
        self.assertAlmostEqual(y1 - y0, 16.0, delta=1.1)
        self.assertAlmostEqual((x0 + x1) / 2, 30.0, delta=0.6)
        points = coords.bins_to_points(parsed.bins, 256, 256, 256)
        self.assertAlmostEqual(points[1][0], 100.0, delta=0.6)

    def test_parse_requires_whole_boxes(self):
        first, end = 1000, 2000
        self.assertFalse(coords.parse_token_ids([1001, 1002, 1003, 1004, 1005, 1006, end], first, 256, [end],
                                                per_object=4).valid)
        salvaged = coords.salvage_token_ids([1001, 1002, 1003, 1004, 1005, 1006], first, 256, [end], per_object=4)
        self.assertEqual(salvaged.bins, ((1, 2, 3, 4),))

    def test_boxes_are_clipped_to_the_image(self):
        self.assertEqual(coords.bins_to_boxes([(0, 0, 10, 10)], 255, 255, 256), [(0.0, 0.0, 5.0, 5.0)])
        with self.assertRaises(ValueError):
            coords.bins_to_boxes([(0, 0)], 255, 255, 256)

    def test_fgas_groups_four_tokens_per_nucleus(self):
        shaped = fgas_token_advantages(2.0, 9, True, [True, False], 0.5, per_object=4)
        np.testing.assert_allclose(shaped, [2, 2, 2, 2, 1, 1, 1, 1, 2])

    @unittest.skipUnless(HAS_SCIPY, "scipy required")
    def test_box_sizes(self):
        from nucleus_rl.nunext.data import box_sizes
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[2:5, 3:10] = 1           # 7 wide, 3 high
        self.assertEqual(box_sizes(labels, 2), [(7.0, 3.0), (2.0, 2.0)])

    @unittest.skipUnless(HAS_SCIPY, "scipy required")
    def test_box_targets_follow_every_d4_view(self):
        from PIL import Image
        from nucleus_rl.nunext.data import load_example
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[2:6, 4:16] = 1            # 12 wide, 4 high, centre (10, 4)
        labels[20:30, 22:25] = 2         # 3 wide, 10 high, centre (23.5, 25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(root / "i.png")
            Image.fromarray(labels).save(root / "m.png")
            row = {"id": "a", "image_path": str(root / "i.png"), "mask_path": str(root / "m.png"),
                   "width": 32, "height": 32, "centroids": [[10.0, 4.0], [23.5, 25.0]]}
            for view in range(8):
                example = load_example(row, 256, view, boxes=True)
                for x, y, w, h in example["objects"]:
                    # Each box, centred on its transformed centroid, must cover the matching transformed nucleus.
                    self.assertIn((round(w), round(h)), {(12, 4), (4, 12), (3, 10), (10, 3)})
                swapped = view % 2 == 1
                sizes = sorted((round(w), round(h)) for _, _, w, h in example["objects"])
                self.assertEqual(sizes, sorted([(4, 12), (10, 3)] if swapped else [(12, 4), (3, 10)]))
                self.assertEqual(len(coords._TOKEN_PATTERN.findall(example["target"])), 8)
            with self.assertRaises(ValueError):
                load_example({k: v for k, v in row.items() if k != "mask_path"}, 256, 0, boxes=True)

    def test_pred_box_prompts(self):
        from PIL import Image
        from nucleus_rl.nunext.pq_eval import score_image
        reference = np.zeros((16, 16), dtype=np.uint16)
        reference[2:6, 2:6] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(root / "i.png")
            Image.fromarray(reference).save(root / "m.png")
            row = {"id": "a", "image_path": str(root / "i.png"), "mask_path": str(root / "m.png"),
                   "centroids": [[4.0, 4.0]]}
            seen = []

            def segment(image, points, boxes, negatives=None):
                seen.append(boxes)
                mask = np.zeros((1, 16, 16), bool)
                x0, y0, x1, y1 = (int(v) for v in boxes[0])
                mask[0, y0:y1, x0:x1] = True
                return mask, np.ones(1)

            result = score_image(row, "pred-box", {"a": {"points": [[4, 4]], "boxes": [[2, 2, 6, 6]]}}, segment)
            self.assertEqual(seen, [[(2, 2, 6, 6)]])
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
            with self.assertRaises(ValueError):
                score_image(row, "pred-box", {"a": [[4, 4]]}, segment)   # a point-only run


@unittest.skipUnless(HAS_SCIPY, "scipy required")
class CappedMatching(unittest.TestCase):
    PRED = [[16.0, 21.0], [6.0, 11.0]]
    GT = [[17.0, 19.0], [32.0, 26.0]]

    def test_paper_matching_loses_a_close_pair(self):
        from nucleus_rl.nunext.matching import match_points
        # Raw-distance optimum pairs (16,21)-(32,26) and (6,11)-(17,19), both > 12 px,
        # although (16,21) is 2.2 px from (17,19).
        self.assertEqual(match_points(self.PRED, self.GT, 12).tp, 0)
        capped = match_points(self.PRED, self.GT, 12, capped=True)
        self.assertEqual((capped.tp, capped.fp, capped.fn), (1, 1, 1))
        self.assertEqual(capped.pred_is_tp, (True, False))

    def test_capped_never_finds_fewer(self):
        from nucleus_rl.nunext.matching import match_points
        rng = np.random.default_rng(0)
        for _ in range(200):
            pred, gt = rng.uniform(0, 60, (rng.integers(0, 8), 2)), rng.uniform(0, 60, (rng.integers(0, 8), 2))
            self.assertGreaterEqual(match_points(pred, gt, 12, True).tp, match_points(pred, gt, 12).tp)

    def test_rescore_reports_both(self):
        from nucleus_rl.nunext.rescore import rescore
        result = rescore([{"id": "a", "valid": True, "pred": self.PRED},
                          {"id": "b", "valid": False, "pred": [[1, 1]]}],
                         [{"id": "a", "centroids": self.GT}, {"id": "b", "centroids": [[1, 1]]}])
        self.assertEqual(result["paper"]["tp"], 0)
        self.assertEqual(result["capped"]["tp"], 1)
        self.assertEqual(result["capped"]["fn"], 2)      # invalid record counts as no predictions
        self.assertNotIn("tta_vote_sweep", result)


@unittest.skipUnless(HAS_SCIPY, "scipy required")
class TestTimeAugmentation(unittest.TestCase):
    def test_inverse_undoes_every_view(self):
        from nucleus_rl.nunext import tta
        points = [(3.0, 7.5), (250.0, 12.0), (128.0, 200.25)]
        for view in range(8):
            _, moved = d4_transform(np.zeros((256, 256, 3), np.uint8), points, view)
            np.testing.assert_allclose(tta.inverse_points(moved, view, 256.0), points, atol=1e-9)

    def test_sizes_swap_on_quarter_turns(self):
        from nucleus_rl.nunext import tta
        self.assertEqual(tta.inverse_objects([(10, 20, 4, 9)], 1, 256)[0][2:], (9.0, 4.0))
        self.assertEqual(tta.inverse_objects([(10, 20, 4, 9)], 4, 256)[0][2:], (4.0, 9.0))
        self.assertEqual(len(tta.inverse_objects([(10, 20)], 3, 256)[0]), 2)

    def test_merge_votes(self):
        from nucleus_rl.nunext import tta
        views = [[(50 + 0.5 * v, 50.0, 10.0, 12.0), (100.0, 100.0, 8.0, 8.0)] for v in range(8)]
        views[2].append((200.0, 30.0, 6.0, 6.0))                 # found in one view only
        views[5] = [(50.0, 50.0, 10.0, 12.0), (52.0, 50.0, 10.0, 12.0)]   # two points near one nucleus
        merged = tta.merge_views(views, radius=8.0)
        votes = {tuple(round(v) for v in m["point"]): m["votes"] for m in merged}
        self.assertEqual(votes[(30, 200)[::-1]], 1)
        self.assertEqual(votes[(100, 100)], 7)
        big = [m for m in merged if abs(m["point"][1] - 50) < 1]
        self.assertEqual(len(big), 1)                            # one member per view per cluster
        self.assertEqual((big[0]["votes"], big[0]["size"]), (8, (10.0, 12.0)))
        self.assertEqual(sum(m["votes"] for m in merged), sum(len(v) for v in views) - 1)   # duplicate dropped
        points, boxes = tta.select(merged, 4, 256, 256)
        self.assertEqual(len(points), 2)
        self.assertEqual(boxes[1], (96.0, 96.0, 104.0, 104.0))
        self.assertEqual(tta.select(tta.merge_views([[(5.0, 5.0)]], 8.0), 1, 256, 256), ([(5.0, 5.0)], None))

    def test_tta_record_recovers_every_view(self):
        from PIL import Image
        from nucleus_rl.nunext.data import load_example
        from nucleus_rl.nunext.evaluate import tta_record
        from nucleus_rl.nunext.pq_eval import prediction_entry
        labels = np.zeros((64, 64), dtype=np.uint16)
        labels[4:10, 6:20] = 1
        labels[40:60, 44:50] = 2
        bins, first, end = 256, 1000, 5000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(root / "i.png")
            Image.fromarray(labels).save(root / "m.png")
            row = {"id": "a", "image_path": str(root / "i.png"), "mask_path": str(root / "m.png"),
                   "width": 64, "height": 64, "centroids": [[13.0, 7.0], [47.0, 50.0]]}
            view_ids = []
            for view in range(8):
                target = load_example(row, bins, view, boxes=True)["target"]
                view_ids.append([first + int(v) for v in coords._TOKEN_PATTERN.findall(target)] + [end])
            view_ids[6] = [first]                                # one truncated view
            record = tta_record(row, view_ids, first, bins, [end], 12.0, 4, 8.0, 4)
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (2, 0, 0))
        self.assertEqual(record["valid_views"], 7)
        self.assertEqual(record["tta"]["votes"], [7, 7])
        np.testing.assert_allclose(record["pred"], row["centroids"], atol=0.6)
        np.testing.assert_allclose(record["tta"]["sizes"], [[14, 6], [6, 20]], atol=0.6)
        entry = prediction_entry(record, min_votes=7)
        self.assertEqual(len(entry["boxes"]), 2)
        self.assertEqual(prediction_entry(record, min_votes=8)["points"], [])
        with self.assertRaises(ValueError):
            prediction_entry({"pred": [], "valid": True}, min_votes=2)

    def test_vote_sweep(self):
        from nucleus_rl.nunext.rescore import vote_sweep
        records = [{"id": "a", "tta": {"points": [[10, 10], [40, 40]], "votes": [8, 2]}}]
        sweep = vote_sweep(records, [{"id": "a", "centroids": [[10, 10]]}])
        self.assertEqual(sweep["1"]["paper"]["fp"], 1)
        self.assertEqual(sweep["3"]["capped"]["micro_f1"], 1.0)
        self.assertEqual(sweep["8"]["paper"]["tp"], 1)


@unittest.skipUnless(HAS_SCIPY, "scipy required")
class MissDiagnostics(unittest.TestCase):
    def test_gt_is_tp(self):
        from nucleus_rl.nunext.matching import match_points
        self.assertEqual(match_points([[10, 10]], [[11, 10], [90, 90]], 12).gt_is_tp, (True, False))
        self.assertEqual(match_points([], [[1, 1]], 12).gt_is_tp, (False,))

    def test_early_stop_shows_up_as_tail_misses(self):
        from nucleus_rl.nunext.rescore import miss_positions
        gt = [[20.0 + 30 * (i % 8), 10.0 + 30 * (i // 8)] for i in range(64)]   # 8 x 8 grid, raster order
        stopped = {"id": "a", "valid": True, "pred": gt[:32]}                 # stops halfway down
        scattered = {"id": "a", "valid": True, "pred": gt[1::2]}              # misses every other nucleus
        rows = [{"id": "a", "height": 256, "centroids": gt}]
        early = miss_positions([stopped], rows, bands=4)
        self.assertEqual(early["by_rank"], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(early["by_y"][:2], [1.0, 1.0])
        self.assertEqual(early["tail"]["after_last_prediction"], 32)
        self.assertEqual(early["tail"]["before_it"], 0)
        self.assertEqual(early["tail"]["after_last_if_random"], 16.0)          # 32 misses x 32/64 after the last
        spread = miss_positions([scattered], rows, bands=4)
        self.assertEqual(spread["by_rank"], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(spread["tail"]["after_last_prediction"], 0)
        self.assertEqual(early["count_ratio"]["50-100"], {"pred_over_gt": 0.5, "images": 1})

    def test_end_bias_shifts_only_end_tokens(self):
        from nucleus_rl.nunext.modeling import EndBias
        scores = np.zeros((2, 6))
        out = EndBias({4, 1}, -2.5)(None, scores)
        np.testing.assert_allclose(out[0], [0, -2.5, 0, 0, -2.5, 0])

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
