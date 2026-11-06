"""Crop preparation for any Images/Labels dataset, the MoNuSeg layout and the SAM oracle scoring."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import read_manifest
from scripts import monuseg_layout, prepare_consep, sam_oracle


def disks(labels, centers, start=1, radius=4):
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]
    for offset, (cy, cx) in enumerate(centers):
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = start + offset
    return labels


def make_root(tmp, label_suffix=".npy"):
    """One 256 px Test image: a sparse, a dense, an empty and a 40-nucleus quadrant."""
    root = Path(tmp, "CoNSeP")
    for name in ("Images", "Labels"):
        Path(root, "Test", name).mkdir(parents=True)
    labels = np.zeros((256, 256), dtype=np.int32)
    disks(labels, [(20, 20), (60, 60), (100, 100)], start=1)                                # sparse (3)
    disks(labels, [(12 + 14 * (i // 4), 140 + 14 * (i % 4)) for i in range(12)], start=10)  # dense (12)
    disks(labels, [(134 + 12 * (i // 8), 132 + 12 * (i % 8)) for i in range(40)], start=100, radius=3)  # > 32
    rgb = (np.random.default_rng(0).random((256, 256, 4)) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGBA").save(root / "Test/Images/test_1.png")
    if label_suffix == ".npy":
        np.save(root / "Test/Labels/test_1.npy", labels)
    else:
        from scipy.io import savemat
        savemat(root / "Test/Labels/test_1.mat", {"inst_map": labels.astype(float)})
    return root


class FakeSam:
    """Returns each prompted nucleus's own annotated mask, optionally grown by `grow` pixels."""

    def __init__(self, reference_for, grow=0):
        self.reference_for, self.grow = reference_for, grow
        self.metadata = {"checkpoint_sha256": "fake"}

    def predict(self, image, objects, image_key=None):
        from nucleus_rl.rewards import masks_to_instances
        reference = self.reference_for[image_key]
        masks = []
        for obj in objects:
            x, y = int(obj.point[0]), int(obj.point[1])
            mask = reference == reference[y, x]
            for _ in range(self.grow):
                mask = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
            masks.append(mask)
        self.last_raw_masks = np.stack(masks)
        return masks_to_instances(self.last_raw_masks, scores=np.ones(len(masks)))


class CropPreparationTests(unittest.TestCase):
    def test_grid_crops_split_into_sparse_and_dense(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "out")
            report = prepare_consep.prepare(make_root(tmp), out)
            self.assertEqual(report["subsets"]["sparse"]["selected"], 1)
            self.assertEqual(report["subsets"]["dense"]["selected"], 1)
            self.assertEqual(report["skipped_crops"], {"empty": 1, "over_32": 1})
            for subset, count in (("sparse", 3), ("dense", 12)):
                rows = read_manifest(out / f"{subset}.jsonl")
                self.assertEqual([r["instance_count"] for r in rows], [count])
                self.assertEqual(rows[0]["dataset"], "consep")
                with Image.open(rows[0]["image_path"]) as crop:
                    self.assertEqual(crop.mode, "RGB")

    def test_matlab_labels(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        with tempfile.TemporaryDirectory() as tmp:
            report = prepare_consep.prepare(make_root(tmp, ".mat"), Path(tmp, "out"))
            self.assertEqual(report["subsets"]["dense"]["selected"], 1)

    def test_refuses_nonempty_output_and_caps_are_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "out")
            out.mkdir()
            (out / "keep.txt").write_text("x")
            with self.assertRaises(FileExistsError):
                prepare_consep.prepare(make_root(tmp), out)
        rows = [{"id": f"{s}_{i}", "source_image_id": s} for s in ("a", "b") for i in range(5)]
        picked = prepare_consep.balanced_sample(rows, 4, seed=1)
        self.assertEqual(sorted(r["source_image_id"] for r in picked), ["a", "a", "b", "b"])


class MonusegLayoutTests(unittest.TestCase):
    def test_official30_rasterised_and_flagged_images_excluded(self):
        ids = load_official30_cohort()["image_ids"]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "monuseg")
            (source / "images").mkdir(parents=True)
            (source / "annotations").mkdir()
            square = '<Region Id="{i}" NegativeROA="{n}"><Vertices>' + "".join(
                f'<Vertex X="{x}" Y="{y}"/>' for x, y in ((4, 4), (20, 4), (20, 20), (4, 20))) + "</Vertices></Region>"
            for index, image_id in enumerate(ids):
                Image.new("RGB", (32, 32)).save(source / "images" / f"{image_id}.tif")
                regions = square.format(i=1, n=1 if index == 0 else 0)
                (source / "annotations" / f"{image_id}.xml").write_text(
                    f"<Annotations><Annotation><Regions>{regions}</Regions></Annotation></Annotations>")
            out = Path(tmp, "layout")
            report = monuseg_layout.build(source, out)
            self.assertEqual(sorted(report["excluded"]), [ids[0]])
            self.assertEqual(len(report["kept"]), 29)
            labels = np.load(out / "All" / "Labels" / f"{ids[1]}.npy")
            self.assertEqual(int(labels.max()), 1)
            crops = prepare_consep.prepare(out, Path(tmp, "crops"), ("All",), crop_size=32, dataset="monuseg")
            self.assertEqual(crops["subsets"]["sparse"]["selected"], 29)


class OracleScoringTests(unittest.TestCase):
    def crops(self, tmp):
        out = Path(tmp, "out")
        prepare_consep.prepare(make_root(tmp), out)
        rows = read_manifest(out / "dense.jsonl")
        return out / "dense.jsonl", {r["image_path"]: np.asarray(Image.open(r["mask_path"])).astype(np.int64)
                                     for r in rows}

    def test_perfect_segmenter_scores_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, references = self.crops(tmp)
            summary = sam_oracle.summarize(sam_oracle.run(manifest, FakeSam(references), max_objects=32))
            self.assertEqual((summary["mean_pq"], summary["tp"], summary["fp"]), (1.0, 12, 0))
            self.assertEqual(summary["nucleus_iou"]["median_iou"], 1.0)

    def test_boundary_disagreement_lowers_nucleus_iou_not_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, references = self.crops(tmp)
            summary = sam_oracle.summarize(sam_oracle.run(manifest, FakeSam(references, grow=1), max_objects=32))
            self.assertLess(summary["nucleus_iou"]["median_iou"], 1.0)
            self.assertGreater(summary["nucleus_iou"]["median_iou"], 0.5)
            self.assertEqual(summary["sam_mask_larger_than_annotation"], 1.0)


if __name__ == "__main__":
    unittest.main()
