#!/usr/bin/env python3
"""Read-only MoNuSeg ZIP audit. Standard library only; never chooses a cohort."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile


def real_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not name.endswith("/") and "__MACOSX" not in path.parts and not path.name.startswith("._")


def patient_id(image_id: str) -> str:
    parts = image_id.split("-")
    if len(parts) < 3 or parts[0] != "TCGA" or not all(parts[:3]):
        raise ValueError(f"Cannot derive TCGA patient ID from {image_id!r}")
    return "-".join(parts[:3])


def parse_regions(xml_bytes: bytes) -> list[dict]:
    """Keep XML IDs and exclusion flags; no claim that every Region is a nucleus."""
    root = ET.fromstring(xml_bytes)
    regions = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "Region":
            continue
        flag = node.get("NegativeROA", "0")
        if flag not in {"0", "1"}:
            raise ValueError(f"Unsupported NegativeROA={flag!r}")
        points = []
        for vertex in node.iter():
            if vertex.tag.rsplit("}", 1)[-1] != "Vertex":
                continue
            point = (float(vertex.attrib["X"]), float(vertex.attrib["Y"]))
            if not all(math.isfinite(v) for v in point):
                raise ValueError("Non-finite polygon coordinate")
            points.append(point)
        regions.append({"xml_id": node.get("Id"), "negative": flag == "1", "points": points,
                        "degenerate": len(set(points)) < 3})
    if not regions:
        raise ValueError("XML contains no Region entries")
    return regions


def discover_pairs(archive: ZipFile) -> tuple[list[dict], dict]:
    images, annotations = defaultdict(list), defaultdict(list)
    for name in archive.namelist():
        if not real_member(name):
            continue
        path = PurePosixPath(name)
        if path.suffix.lower() in {".tif", ".tiff"}:
            images[path.stem].append(name)
        elif path.suffix.lower() == ".xml":
            annotations[path.stem].append(name)
    image_ids, xml_ids = set(images), set(annotations)
    duplicate_images = {key: value for key, value in images.items() if len(value) != 1}
    duplicate_xml = {key: value for key, value in annotations.items() if len(value) != 1}
    invalid_ids = []
    pairs = []
    for image_id in sorted(image_ids & xml_ids):
        if image_id in duplicate_images or image_id in duplicate_xml:
            continue
        try:
            patient = patient_id(image_id)
        except ValueError:
            invalid_ids.append(image_id)
            continue
        pairs.append({"id": image_id, "patient_id": patient,
                      "image_member": images[image_id][0], "xml_member": annotations[image_id][0]})
    issues = {"images_without_xml": sorted(image_ids - xml_ids),
              "xml_without_images": sorted(xml_ids - image_ids),
              "duplicate_image_ids": duplicate_images, "duplicate_xml_ids": duplicate_xml,
              "unrecognized_patient_ids": invalid_ids}
    return pairs, issues


def audit_archive(path: str | Path, expected_images: int | None = None, verify_crc: bool = False) -> dict:
    path = Path(path)
    with ZipFile(path) as archive:
        pairs, issues = discover_pairs(archive)
        xml_errors = []
        for pair in pairs:
            try:
                regions = parse_regions(archive.read(pair["xml_member"]))
                ids = [region["xml_id"] for region in regions]
                pair.update(region_entries=len(regions),
                            positive_region_entries=sum(not r["negative"] for r in regions),
                            negative_region_entries=sum(r["negative"] for r in regions),
                            degenerate_region_entries=sum(r["degenerate"] for r in regions),
                            degenerate_region_ids=[r["xml_id"] for r in regions if r["degenerate"]],
                            duplicate_region_ids=sorted(str(key) for key, count in Counter(ids).items() if count > 1))
            except (ET.ParseError, ValueError, KeyError) as error:
                xml_errors.append({"id": pair["id"], "error": str(error)})
        patients = defaultdict(list)
        for pair in pairs:
            patients[pair["patient_id"]].append(pair["id"])
        bad_crc = archive.testzip() if verify_crc else None
    issues["xml_errors"] = xml_errors
    if verify_crc and bad_crc:
        issues["crc_failure_member"] = bad_crc
    return {"archive": str(path.resolve()), "archive_bytes": path.stat().st_size,
            "pair_count": len(pairs), "patient_count": len(patients), "pairs": pairs,
            "patients_with_multiple_images": {p: ids for p, ids in patients.items() if len(ids) > 1},
            "region_entries": sum(p.get("region_entries", 0) for p in pairs),
            "positive_region_entries": sum(p.get("positive_region_entries", 0) for p in pairs),
            "negative_region_entries": sum(p.get("negative_region_entries", 0) for p in pairs),
            "degenerate_region_entries": sum(p.get("degenerate_region_entries", 0) for p in pairs),
            "expected_images": expected_images,
            "expected_count_mismatch": expected_images is not None and len(pairs) != expected_images,
            "crc_verification": "failed" if bad_crc else ("passed" if verify_crc else "not_requested"),
            "pair_validation_passed": not any(issues.values()), "issues": issues,
            "interpretation": "Region totals include exclusions; they are not verified nucleus counts. Count mismatch does not select or drop a cohort."}


def audit_datasets(train_path: str | Path, test_path: str | Path | None = None,
                   verify_crc: bool = False, expected_train: int = 30) -> dict:
    train = audit_archive(train_path, expected_train, verify_crc)
    result = {"schema_version": 1, "train": train}
    if test_path is not None:
        test = audit_archive(test_path, 14, verify_crc)
        result["test"] = test
        result["train_test_overlap"] = {
            "image_ids": sorted({p["id"] for p in train["pairs"]} & {p["id"] for p in test["pairs"]}),
            "patient_ids": sorted({p["patient_id"] for p in train["pairs"]} & {p["patient_id"] for p in test["pairs"]})}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", required=True, type=Path)
    parser.add_argument("--test-archive", type=Path, help="Optional audit only; never used by crop preparation")
    parser.add_argument("--expected-train-count", type=int, default=30)
    parser.add_argument("--verify-crc", action="store_true", help="Read every ZIP member to verify CRC (can take time)")
    parser.add_argument("--output", type=Path, help="Default: print JSON to stdout")
    args = parser.parse_args(argv)
    result = audit_datasets(args.train_archive, args.test_archive, args.verify_crc, args.expected_train_count)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    overlap = result.get("train_test_overlap", {})
    valid = all(result[key]["pair_validation_passed"] for key in ("train", "test") if key in result)
    return 0 if valid and not any(overlap.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
