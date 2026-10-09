"""Data contracts: source grouping, exclusion flags, geometry, and preserved IDs."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.audit_monuseg import audit_archive, audit_datasets, parse_regions, patient_id
from scripts.prepare_monuseg import candidate_crops, prepare, rasterize_regions, select_balanced, split_patients

HAS_IMAGES = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("PIL") is not None


def xml(regions=None):
    if regions is None:
        regions = [("3", "0", [(2, 2), (4, 2), (4, 4), (2, 4)])]
    body = "".join('<Region Id="%s" NegativeROA="%s"><Vertices>%s</Vertices></Region>' %
                   (ident, negative, "".join(f'<Vertex X="{x}" Y="{y}"/>' for x, y in points))
                   for ident, negative, points in regions)
    return f"<Annotations><Annotation><Regions>{body}</Regions></Annotation></Annotations>".encode()


def add_pair(archive, image_id, annotation=None, image=b"audit does not decode pixels"):
    archive.writestr(f"Training/Tissue Images/{image_id}.tif", image)
    archive.writestr(f"Training/Annotations/{image_id}.xml", annotation if annotation is not None else xml())


class ArchiveAuditTests(unittest.TestCase):
    def test_resource_forks_are_ignored_and_count_mismatch_is_not_a_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.zip"
            with ZipFile(path, "w") as archive:
                add_pair(archive, "TCGA-AA-0001-SLIDE1")
                add_pair(archive, "TCGA-AA-0001-SLIDE2")
                archive.writestr("__MACOSX/Training/._TCGA-AA-0001-SLIDE1.xml", b"invalid")
                archive.writestr("Training/._TCGA-AA-0001-SLIDE1.tif", b"invalid")
            report = audit_archive(path, expected_images=30, verify_crc=True)
            self.assertEqual(report["pair_count"], 2)
            self.assertEqual(report["patient_count"], 1)
            self.assertTrue(report["expected_count_mismatch"])
            self.assertTrue(report["pair_validation_passed"])
            self.assertEqual(report["crc_verification"], "passed")
            self.assertEqual(len(report["patients_with_multiple_images"]["TCGA-AA-0001"]), 2)

    def test_pair_validation_reports_unmatched_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.zip"
            with ZipFile(path, "w") as archive:
                add_pair(archive, "TCGA-AA-0001-SLIDE")
                archive.writestr("extra/TCGA-AA-0001-SLIDE.tif", b"duplicate identity")
                archive.writestr("Training/TCGA-AA-0002-SLIDE.tif", b"no xml")
                archive.writestr("Training/TCGA-AA-0003-SLIDE.xml", xml())
            report = audit_archive(path)
            self.assertFalse(report["pair_validation_passed"])
            self.assertEqual(report["issues"]["images_without_xml"], ["TCGA-AA-0002-SLIDE"])
            self.assertEqual(report["issues"]["xml_without_images"], ["TCGA-AA-0003-SLIDE"])
            self.assertIn("TCGA-AA-0001-SLIDE", report["issues"]["duplicate_image_ids"])

    def test_distinct_slides_from_one_patient_are_detected_across_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            train, test = Path(directory) / "train.zip", Path(directory) / "test.zip"
            with ZipFile(train, "w") as archive:
                add_pair(archive, "TCGA-AA-0001-SLIDE1")
            with ZipFile(test, "w") as archive:
                add_pair(archive, "TCGA-AA-0001-SLIDE2")
            report = audit_datasets(train, test)
            self.assertEqual(report["train_test_overlap"]["image_ids"], [])
            self.assertEqual(report["train_test_overlap"]["patient_ids"], ["TCGA-AA-0001"])

    def test_negative_regions_and_fractional_coordinates_survive_parsing(self):
        regions = parse_regions(xml([("19", "1", [(0.5, -1), (4, 2.25), (3, 4)])]))
        self.assertTrue(regions[0]["negative"])
        self.assertEqual(regions[0]["points"][0], (0.5, -1.0))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parse_regions(xml([("1", "0", [(float("nan"), 1), (2, 3), (4, 5)])]))
        with self.assertRaises(ValueError):
            patient_id("not-a-TCGA-image")

    def test_degenerate_region_is_counted_but_explicitly_flagged(self):
        regions = parse_regions(xml([("broken", "0", [(1, 1), (1, 1), (2, 2)])]))
        self.assertEqual(len(regions), 1)
        self.assertTrue(regions[0]["degenerate"])

    def test_group_split_is_deterministic_disjoint_and_deduplicated(self):
        patients = [f"TCGA-AA-{i:04}" for i in range(15)]
        groups = split_patients(patients + patients, 8, 4, 42)
        self.assertEqual(groups, split_patients(list(reversed(patients)), 8, 4, 42))
        self.assertEqual(len(groups["train"]), 8)
        self.assertEqual(len(groups["validation"]), 4)
        self.assertFalse(set(groups["train"]) & set(groups["validation"]))
        with self.assertRaisesRegex(ValueError, "eligible patients"):
            split_patients(patients[:3], 8, 4)


@unittest.skipUnless(HAS_IMAGES, "Pillow and numpy are needed for rasterization and crop tests")
class ImagePreparationTests(unittest.TestCase):
    def test_earliest_polygon_wins_and_fragmentation_is_reported(self):
        regions = parse_regions(xml([
            ("first", "0", [(4, 0), (5, 0), (5, 8), (4, 8)]),
            ("second", "0", [(1, 1), (7, 1), (7, 7), (1, 7)]),
        ]))
        labels, log = rasterize_regions(regions, 10, 10)
        self.assertEqual(int(labels[3, 4]), 1)
        self.assertEqual(int(labels[3, 2]), 2)
        self.assertEqual(log[1]["overlap_pixels_removed"], 14)
        self.assertEqual(log[1]["fragments_4_connected"], 2)
        self.assertEqual(log[1]["assigned_pixels"], 35)

    def test_negative_roa_is_rejected_instead_of_guessed_as_a_hole(self):
        regions = parse_regions(xml([("1", "1", [(1, 1), (2, 1), (2, 2)])]))
        with self.assertRaisesRegex(ValueError, "NegativeROA"):
            rasterize_regions(regions, 10, 10)

    def test_clipping_and_dropped_polygon_are_logged(self):
        regions = parse_regions(xml([
            ("edge", "0", [(-3, 2), (3, 2), (3, 5), (-3, 5)]),
            ("outside", "0", [(20, 20), (25, 20), (25, 25)]),
        ]))
        _, log = rasterize_regions(regions, 10, 10)
        self.assertTrue(log[0]["clipped_to_image"])
        self.assertEqual(log[0]["assigned_pixels"], 16)
        self.assertTrue(log[1]["dropped"])

    def test_instance_ids_above_255_roundtrip_as_png(self):
        import numpy as np
        from PIL import Image
        outside = [(str(i), "0", [(20, 20), (25, 20), (25, 25)]) for i in range(299)]
        regions = parse_regions(xml(outside + [("last", "0", [(2, 2), (4, 2), (4, 4)])]))
        labels, _ = rasterize_regions(regions, 10, 10)
        encoded = io.BytesIO()
        Image.fromarray(labels).save(encoded, format="PNG")
        encoded.seek(0)
        decoded = np.asarray(Image.open(encoded))
        self.assertEqual(int(decoded.max()), 300)
        np.testing.assert_array_equal(labels, decoded)

    def test_complete_instances_required_even_when_distant_fragment_is_interior(self):
        import numpy as np
        labels = np.zeros((16, 16), dtype=np.uint16)
        labels[2:4, 2:4] = 1
        labels[10:12, 10:12] = 1
        self.assertEqual(candidate_crops(labels, 8), [])
        labels[10:12, 10:12] = 2
        self.assertEqual(len(candidate_crops(labels, 8)), 2)
        labels[2, 7:9] = 3
        self.assertEqual(len(candidate_crops(labels, 8)), 1)

    def test_visible_policy_keeps_partial_objects_and_records_them_without_erasing_pixels(self):
        import numpy as np
        labels = np.zeros((8, 16), dtype=np.uint16)
        labels[2:4, 6:10] = 7
        original = labels.copy()
        self.assertEqual(candidate_crops(labels, 8), [])
        crops = candidate_crops(labels, 8, border_policy="visible")
        self.assertEqual(len(crops), 2)
        self.assertTrue(all(c["truncated_instance_ids"] == [7] for c in crops))
        self.assertTrue(all(c["instance_count"] == 1 for c in crops))
        np.testing.assert_array_equal(labels, original)

    def test_insufficient_candidates_fail_without_relaxing_conditions(self):
        with self.assertRaisesRegex(ValueError, "no silent relaxation"):
            select_balanced([], 2, 42)

    def test_end_to_end_excludes_negative_image_and_writes_disjoint_training_groups(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            archive_path, out = Path(directory) / "training.zip", Path(directory) / "prepared"
            buffer = io.BytesIO()
            Image.new("RGB", (16, 16), (190, 120, 170)).save(buffer, format="TIFF")
            with ZipFile(archive_path, "w") as archive:
                for index in range(4):
                    annotation = xml([("1", "1" if index == 3 else "0", [(2, 2), (4, 2), (4, 4), (2, 4)])])
                    add_pair(archive, f"TCGA-AA-{index:04}-SLIDE", annotation, buffer.getvalue())
            args = argparse.Namespace(train_archive=archive_path, out=out, crop_size=8, max_nuclei=8,
                                      train_patients=2, val_patients=1, train_crops=2, val_crops=1,
                                      seed=42, train_patient_ids=None, val_patient_ids=None)
            report = prepare(args)
            self.assertFalse(report["original_test_accessed"])
            self.assertEqual(len(report["excluded_images"]), 1)
            rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            train = {row["patient_id"] for row in rows if row["split"] == "train"}
            validation = {row["patient_id"] for row in rows if row["split"] == "validation"}
            self.assertFalse(train & validation)
            self.assertEqual(len(rows), 3)
            for row in rows:
                mask = np.asarray(Image.open(out / row["mask_path"]))
                self.assertEqual(set(np.unique(mask)), {0, 1})
                self.assertEqual(row["instance_count"], 1)
                self.assertIsNone(row["physical_spacing"]["um_per_pixel"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare(args)

    def test_portable_manifest_resolves_after_entire_dataset_directory_moves(self):
        from PIL import Image
        from nucleus_rl.evaluate import read_manifest
        with tempfile.TemporaryDirectory() as directory:
            archive_path, out = Path(directory) / "training.zip", Path(directory) / "original"
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buffer, format="TIFF")
            with ZipFile(archive_path, "w") as archive:
                add_pair(archive, "TCGA-AA-0001-SLIDE", image=buffer.getvalue())
                add_pair(archive, "TCGA-AA-0002-SLIDE", image=buffer.getvalue())
            args = argparse.Namespace(train_archive=archive_path, out=out, crop_size=8, max_nuclei=8,
                                      train_patients=1, val_patients=1, train_crops=1, val_crops=1,
                                      seed=42, train_patient_ids=None, val_patient_ids=None)
            prepare(args)
            relocated = Path(directory) / "different-machine-layout"
            out.rename(relocated)
            for name in ("manifest", "train", "validation"):
                raw = [json.loads(line) for line in (relocated / f"{name}.jsonl").read_text().splitlines()]
                for row in raw:
                    for key in ("image_path", "mask_path", "gt_mask"):
                        self.assertFalse(Path(row[key]).is_absolute())
                        self.assertTrue((relocated / row[key]).is_file())
                loaded = read_manifest(relocated / f"{name}.jsonl")
                for row in loaded:
                    self.assertTrue(Path(row["image_path"]).is_relative_to(relocated.resolve()))
                    self.assertTrue(Path(row["mask_path"]).is_relative_to(relocated.resolve()))
                    with Image.open(row["image_path"]) as loaded_image:
                        self.assertEqual(loaded_image.size, (8, 8))

    def test_test_archive_rejected_before_any_training_data_is_opened(self):
        args = argparse.Namespace(train_archive=Path("MoNuSeg_Testing_data.zip"), out=Path("unused"))
        with self.assertRaisesRegex(ValueError, "test archive"):
            prepare(args)


if __name__ == "__main__":
    unittest.main()
