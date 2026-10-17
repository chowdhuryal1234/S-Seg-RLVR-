"""Explicit official MoNuSeg source membership, independent of crop selection.

The tracked protocol is derived from the official patient PDF. No image labels,
connected components, model outputs, or presumed organ names determine membership.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


COHORT_ID = "monuseg2018_official_training30"
SOURCE_PDF_SHA256 = "212d8f53c88acfdcb35edf56ae7033af53c0b1a3e86baaf4b2c4d804b165521c"
IMAGE_IDS_SHA256 = "0ee985fff547dd78dbd5dd3826c03d513a3f24477219149bc071f4c4684042a8"
DEFAULT_COHORT_PATH = Path(__file__).resolve().parents[1] / "protocols" / "monuseg_official30.json"
_IMAGE_ID = re.compile(r"TCGA-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+")


def _patient_id(image_id: str) -> str:
    return "-".join(image_id.split("-")[:3])


def _validate_cohort(cohort: dict[str, Any]) -> None:
    ids = cohort.get("image_ids")
    if (cohort.get("schema_version") != 1 or cohort.get("cohort_id") != COHORT_ID
            or not isinstance(ids, list) or len(ids) != 30
            or any(not isinstance(i, str) or not _IMAGE_ID.fullmatch(i) for i in ids)
            or len(set(ids)) != 30 or len({_patient_id(i) for i in ids}) != 30):
        raise ValueError("Official30 protocol requires the 30 unique canonical image/patient IDs")
    digest = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    if digest != IMAGE_IDS_SHA256 or cohort.get("image_ids_sha256") != IMAGE_IDS_SHA256:
        raise ValueError("Official30 image IDs differ from the verified source PDF")
    source = cohort.get("source", {})
    if source.get("sha256") != SOURCE_PDF_SHA256 or not source.get("url"):
        raise ValueError("Official30 source PDF provenance is missing or differs")
    extra = cohort.get("known_noncanonical_image_ids", [])
    if (not isinstance(extra, list)
            or any(not isinstance(i, str) or not _IMAGE_ID.fullmatch(i) for i in extra)
            or len(extra) != len(set(extra)) or set(extra) & set(ids)):
        raise ValueError("Invalid known noncanonical image IDs")


def load_official30_cohort(path: str | Path | None = None) -> dict[str, Any]:
    """Read the tracked protocol and verify exact membership, not only its size."""
    raw = Path(path or DEFAULT_COHORT_PATH).read_bytes()
    cohort = json.loads(raw)
    _validate_cohort(cohort)
    cohort["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    return cohort


def _provenance(cohort: dict[str, Any]) -> dict[str, Any]:
    return {"cohort_id": cohort["cohort_id"], "source": dict(cohort["source"]),
            "image_ids_sha256": cohort["image_ids_sha256"],
            "protocol_sha256": cohort.get("protocol_sha256")}


def select_cohort_pairs(pairs: list[dict[str, Any]], cohort: dict[str, Any] | None = None
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all 30 canonical archive pairs; fail if any canonical image is absent.

    This runs before annotation-quality exclusions. Extra archive images are
    reported by full ID and excluded. Input order and dictionaries are preserved.
    """
    cohort = load_official30_cohort() if cohort is None else cohort
    _validate_cohort(cohort)
    canonical = set(cohort["image_ids"])
    ids = [pair.get("id") for pair in pairs]
    if any(not isinstance(i, str) for i in ids):
        raise ValueError("Archive pairs require string image IDs")
    duplicates = sorted(i for i, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate archive image IDs: {duplicates}")
    missing = sorted(canonical - set(ids))
    if missing:
        raise ValueError(f"Missing official30 image IDs: {missing}")
    selected = [pair for pair in pairs if pair["id"] in canonical]
    for pair in selected:
        if pair.get("patient_id") != _patient_id(pair["id"]):
            raise ValueError(f"Patient ID does not match source image: {pair['id']}")
    excluded = sorted(set(ids) - canonical)
    report = {**_provenance(cohort), "archive_pair_count": len(pairs),
              "canonical_pair_count_before_annotation_exclusions": len(selected),
              "canonical_image_ids": sorted(canonical),
              "excluded_noncanonical_image_ids": excluded,
              "excluded_known_extra_image_ids": sorted(set(excluded) & set(cohort["known_noncanonical_image_ids"])),
              "excluded_other_image_ids": sorted(set(excluded) - set(cohort["known_noncanonical_image_ids"])),
              "membership_rule": "Exact full image ID in the verified official PDF; annotation quality and crop eligibility are separate filters."}
    return selected, report


def validate_training_rows(rows: list[dict[str, Any]], cohort: dict[str, Any] | None = None
                           ) -> dict[str, Any]:
    """Require nonempty, distinct training rows from canonical source images.

    A subset of the 30 is allowed; image membership and patient consistency are
    checked even for old manifests without cohort metadata. This does not check
    pixel contents or certify the rasterizer, labels, or a benchmark protocol.
    """
    cohort = load_official30_cohort() if cohort is None else cohort
    _validate_cohort(cohort)
    canonical = set(cohort["image_ids"])
    if not rows:
        raise ValueError("Official30 diagnostic requires nonempty training rows")
    row_ids = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Official30 diagnostic accepts only rows with split=train")
        source_id = row.get("source_image_id")
        if not isinstance(source_id, str) or source_id not in canonical:
            raise ValueError(f"Noncanonical or missing source_image_id: {source_id!r}")
        if row.get("patient_id") != _patient_id(source_id):
            raise ValueError(f"Patient ID does not match source image: {source_id}")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise ValueError("Training row IDs must be nonempty and unique")
        row_ids.add(row_id)
    return {**_provenance(cohort), "status": "official30_training_membership_verified",
            "selected_rows": len(rows), "selected_row_ids": [r["id"] for r in rows],
            "selected_source_image_ids": sorted({r["source_image_id"] for r in rows}),
            "selected_patient_ids": sorted({r["patient_id"] for r in rows}),
            "all_rows_train": True,
            "scope": "Source identity and split metadata only; no validation/test rows, pixel-content verification, or performance claim."}
