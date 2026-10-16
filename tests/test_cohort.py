"""Official cohort identity, split boundaries, and opt-in preparation behavior."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from nucleus_rl.cohort import load_official30_cohort, select_cohort_pairs, validate_training_rows
from scripts.prepare_monuseg import prepare


def pair(image_id):
    return {"id": image_id, "patient_id": "-".join(image_id.split("-")[:3])}


def row(image_id, suffix="_x0_y0"):
    return {"id": image_id + suffix, "source_image_id": image_id,
            "patient_id": pair(image_id)["patient_id"], "split": "train"}


class CohortMembershipTests(unittest.TestCase):
    def setUp(self):
        self.cohort = load_official30_cohort()
        self.ids = self.cohort["image_ids"]
        self.extra = self.cohort["known_noncanonical_image_ids"]

    def test_protocol_contains_exact_pdf_identity_and_seven_archive_extras(self):
        self.assertEqual(len(self.ids), 30)
        self.assertEqual(len(self.extra), 7)
        self.assertIn("TCGA-G2-A2EK-01A-02-TSB", self.ids)
        self.assertIn("TCGA-DK-A2I6-01A-01-TS1", self.ids)
        self.assertEqual(self.cohort["source"]["sha256"],
                         "212d8f53c88acfdcb35edf56ae7033af53c0b1a3e86baaf4b2c4d804b165521c")
        self.assertEqual(len(self.cohort["protocol_sha256"]), 64)

    def test_37_pairs_filter_by_identity_preserve_order_and_report_every_extra(self):
        pairs = [pair(i) for i in reversed(self.ids + self.extra)]
        before = copy.deepcopy(pairs)
        selected, report = select_cohort_pairs(pairs, self.cohort)
        self.assertEqual([p["id"] for p in selected], list(reversed(self.ids)))
        self.assertEqual(report["canonical_pair_count_before_annotation_exclusions"], 30)
        self.assertEqual(report["excluded_noncanonical_image_ids"], sorted(self.extra))
        self.assertEqual(report["excluded_known_extra_image_ids"], sorted(self.extra))
        self.assertEqual(report["excluded_other_image_ids"], [])
        self.assertEqual(pairs, before)

    def test_exact_count_30_does_not_approve_a_substituted_noncanonical_image(self):
        pairs = [pair(i) for i in self.ids[1:] + self.extra[:1]]
        self.assertEqual(len(pairs), 30)
        with self.assertRaisesRegex(ValueError, "Missing official30"):
            select_cohort_pairs(pairs)

    def test_duplicate_archive_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate archive"):
            select_cohort_pairs([pair(i) for i in self.ids + self.ids[:1]])

    def test_changed_30_id_protocol_cannot_redefine_canonical_membership(self):
        changed = copy.deepcopy(self.cohort)
        changed["image_ids"][0] = self.extra[0]
        with self.assertRaisesRegex(ValueError, "differ from the verified source PDF"):
            validate_training_rows([row(self.extra[0])], changed)

    def test_training_subset_from_old_manifest_is_accepted_with_audit(self):
        rows = [row(self.ids[0]), row(self.ids[0], "_x128_y0"), row(self.ids[1])]
        report = validate_training_rows(rows)
        self.assertEqual(report["selected_rows"], 3)
        self.assertEqual(report["selected_source_image_ids"], sorted(self.ids[:2]))
        self.assertEqual(report["selected_row_ids"], [r["id"] for r in rows])
        self.assertTrue(report["all_rows_train"])
        self.assertEqual(report["source"], self.cohort["source"])

    def test_validation_or_test_rows_never_enter_training_diagnostic(self):
        for split in ("validation", "test", None):
            bad = row(self.ids[0])
            bad["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "only rows with split=train"):
                validate_training_rows([bad])

    def test_noncanonical_source_rejected_even_when_marked_train(self):
        for image_id in self.extra:
            with self.subTest(image_id=image_id), self.assertRaisesRegex(ValueError, "Noncanonical"):
                validate_training_rows([row(image_id)])

    def test_patient_mismatch_empty_and_duplicate_rows_rejected(self):
        bad = row(self.ids[0])
        bad["patient_id"] = "TCGA-AA-0000"
        with self.assertRaisesRegex(ValueError, "Patient ID"):
            validate_training_rows([bad])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_training_rows([])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_training_rows([row(self.ids[0]), row(self.ids[0])])


class CohortPreparationTests(unittest.TestCase):
    def test_opt_in_filters_before_decoding_and_reports_quality_exclusions_separately(self):
        from PIL import Image

        cohort = load_official30_cohort()
        ids = cohort["image_ids"]
        extra = cohort["known_noncanonical_image_ids"]
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (190, 120, 170)).save(buffer, format="TIFF")
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "training.zip"
            out = Path(directory) / "official30_crops"
            with ZipFile(archive_path, "w") as archive:
                for image_id in ids + extra:
                    # Noncanonical TIFF payloads are intentionally undecodable:
                    # they must never reach image preparation.
                    image = buffer.getvalue() if image_id in ids else b"not an image"
                    negative = "1" if image_id == ids[0] else "0"
                    annotation = (
                        f'<Annotations><Annotation><Regions><Region Id="1" NegativeROA="{negative}">'
                        '<Vertices><Vertex X="2" Y="2"/><Vertex X="4" Y="2"/>'
                        '<Vertex X="4" Y="4"/><Vertex X="2" Y="4"/></Vertices>'
                        '</Region></Regions></Annotation></Annotations>'
                    )
                    archive.writestr(f"Training/Tissue Images/{image_id}.tif", image)
                    archive.writestr(f"Training/Annotations/{image_id}.xml", annotation)
            args = argparse.Namespace(
                train_archive=archive_path, out=out, cohort="official30",
                crop_size=8, max_nuclei=8, border_policy="complete",
                train_patients=2, val_patients=1, train_crops=2, val_crops=1,
                seed=42, train_patient_ids=None, val_patient_ids=None,
            )
            report = prepare(args)
            self.assertEqual(report["cohort"]["canonical_pair_count_before_annotation_exclusions"], 30)
            self.assertEqual(report["cohort"]["canonical_pairs_after_annotation_exclusions"], 29)
            self.assertEqual(report["cohort"]["excluded_noncanonical_image_ids"], sorted(extra))
            self.assertEqual([item["id"] for item in report["excluded_images"]], ids[:1])
            rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["source_image_id"] in ids[1:] for r in rows))
            self.assertTrue(all(r["cohort_id"] == cohort["cohort_id"] for r in rows))
            self.assertTrue(all(r["cohort_protocol_sha256"] == cohort["protocol_sha256"] for r in rows))
            training = [r for r in rows if r["split"] == "train"]
            self.assertEqual(validate_training_rows(training)["selected_rows"], 2)
            self.assertTrue(report["not_an_official_or_published_benchmark"])


if __name__ == "__main__":
    unittest.main()
