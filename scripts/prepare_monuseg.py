#!/usr/bin/env python3
"""Prepare a bounded, provisional engineering split from the original TRAIN ZIP only.

Requires Pillow and numpy. This is neither the official MoNuSeg split nor a
reproduction of any published result. Whole-image/whole-patient grouping precedes
crop selection. No test-archive argument is accepted.
"""
from __future__ import annotations

import argparse
from collections import deque
import io
import json
import math
from pathlib import Path
import random
import sys
from zipfile import ZipFile

try:
    from audit_monuseg import audit_archive, parse_regions
except ImportError:
    from scripts.audit_monuseg import audit_archive, parse_regions


def split_patients(patient_ids: list[str], train_count: int = 8, val_count: int = 4,
                   seed: int = 42) -> dict[str, list[str]]:
    if train_count < 1 or val_count < 1:
        raise ValueError("Both engineering splits require at least one patient")
    patients = sorted(set(patient_ids))
    if len(patients) < train_count + val_count:
        raise ValueError(f"Need {train_count + val_count} eligible patients; found {len(patients)}")
    random.Random(seed).shuffle(patients)
    return {"train": patients[:train_count], "validation": patients[train_count:train_count + val_count]}


def connected_components(binary) -> int:
    """Count 4-connected fragments in a small polygon ROI without scipy."""
    remaining = set(zip(*binary.nonzero()))
    count = 0
    while remaining:
        count += 1
        queue = [remaining.pop()]
        while queue:
            y, x = queue.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
    return count


def rasterize_regions(regions: list[dict], width: int, height: int):
    """Positive polygons, earliest wins, round vertices to nearest integer.

    Any NegativeROA rejects the whole image. This intentionally avoids guessing
    whether an exclusion region is a hole or a distinct annotation convention.
    Pixel filling uses Pillow's inclusive polygon rasterizer; MATLAB parity is
    not claimed. Stable ordinal IDs are retained even when a region is dropped.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if any(region["negative"] for region in regions):
        raise ValueError("NegativeROA present: exclude this image conservatively")
    if any(region.get("degenerate") for region in regions):
        raise ValueError("Degenerate polygon present: exclude this image conservatively")
    if len(regions) > 65535:
        raise ValueError("More than 65,535 Regions cannot be represented as uint16 IDs")
    if width < 1 or height < 1:
        raise ValueError("Invalid image dimensions")
    labels = np.zeros((height, width), dtype=np.uint16)
    log = []
    for ordinal, region in enumerate(regions, 1):
        points = region["points"]
        rounded = [(math.floor(x + 0.5), math.floor(y + 0.5)) for x, y in points]
        clipped = any(x < 0 or y < 0 or x > width - 1 or y > height - 1 for x, y in points)
        left = max(0, min(x for x, _ in rounded))
        top = max(0, min(y for _, y in rounded))
        right = min(width, max(x for x, _ in rounded) + 1)
        bottom = min(height, max(y for _, y in rounded) + 1)
        entry = {"instance_id": ordinal, "xml_id": region["xml_id"], "clipped_to_image": clipped,
                 "input_polygon_pixels": 0, "overlap_pixels_removed": 0,
                 "assigned_pixels": 0, "fragments_4_connected": 0, "dropped": True}
        if right > left and bottom > top:
            polygon = Image.new("1", (right - left, bottom - top))
            ImageDraw.Draw(polygon).polygon([(x - left, y - top) for x, y in rounded], fill=1)
            filled = np.asarray(polygon, dtype=bool)
            view = labels[top:bottom, left:right]
            assigned = filled & (view == 0)
            pixels = int(assigned.sum())
            entry.update(input_polygon_pixels=int(filled.sum()),
                         overlap_pixels_removed=int((filled & (view != 0)).sum()),
                         assigned_pixels=pixels, fragments_4_connected=connected_components(assigned),
                         dropped=pixels == 0)
            view[assigned] = ordinal
        log.append(entry)
    return labels, log


def candidate_crops(labels, crop_size: int = 128, max_nuclei: int = 8,
                    forbidden_ids=(), border_policy: str = "complete") -> list[dict]:
    """Non-overlapping grid crops with an explicit boundary supervision policy.

    The default complete policy compares crop/full-image pixel counts, including
    disconnected fragments, and requires a zero foreground border. The explicit
    visible policy retains partial objects and identifies them in metadata.
    Neither policy erases partial-object pixels or ignores foreground pixels.
    """
    import numpy as np

    if crop_size < 1 or max_nuclei < 1:
        raise ValueError("Crop size and max nuclei must be positive")
    if border_policy not in {"complete", "visible"}:
        raise ValueError("border_policy must be complete or visible")
    height, width = labels.shape
    full_counts = np.bincount(labels.ravel())
    forbidden = set(forbidden_ids)
    candidates = []
    for y in range(0, height - crop_size + 1, crop_size):
        for x in range(0, width - crop_size + 1, crop_size):
            crop = labels[y:y + crop_size, x:x + crop_size]
            ids, counts = np.unique(crop[crop > 0], return_counts=True)
            if not 1 <= len(ids) <= max_nuclei:
                continue
            source_clipped = sorted(set(int(i) for i in ids) & forbidden)
            truncated = [int(i) for i, count in zip(ids, counts) if count != full_counts[i]]
            border_ids = sorted(set(int(i) for i in np.concatenate((crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1])) if i))
            if border_policy == "complete" and (source_clipped or truncated or border_ids):
                continue
            candidates.append({"bbox_xyxy": [x, y, x + crop_size, y + crop_size],
                               "instance_ids": [int(i) for i in ids], "instance_count": len(ids),
                               "truncated_instance_ids": truncated, "border_touching_instance_ids": border_ids,
                               "source_clipped_instance_ids": source_clipped})
    return candidates


def select_balanced(candidates: list[dict], required: int, seed: int) -> list[dict]:
    """Round-robin across patients to avoid selecting only the first easy image."""
    if required < 1:
        raise ValueError("Each split requires at least one crop")
    if len(candidates) < required:
        raise ValueError(f"Only {len(candidates)} eligible crops under the requested border policy; requested {required}. "
                         "Reduce the requested crop count explicitly, increase patient count, or revise the protocol; no silent relaxation.")
    buckets = {}
    for candidate in candidates:
        buckets.setdefault(candidate["patient_id"], []).append(candidate)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    queues = {key: deque(buckets[key]) for key in keys}
    selected = []
    while len(selected) < required:
        for key in keys:
            if queues[key]:
                selected.append(queues[key].popleft())
            if len(selected) == required:
                return selected
    return selected


def prepare(args) -> dict:
    import numpy as np
    from PIL import Image

    archive_path = Path(args.train_archive).resolve()
    border_policy = getattr(args, "border_policy", "complete")
    if "test" in archive_path.name.lower():
        raise ValueError("Refusing a test archive: preparation accepts original training data only")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; use a new directory")
    audit = audit_archive(archive_path, expected_images=30)
    if not audit["pair_validation_passed"]:
        raise ValueError(f"Archive pair validation failed: {audit['issues']}")
    if any("test" in pair[key].lower() for pair in audit["pairs"] for key in ("image_member", "xml_member")):
        raise ValueError("Refusing archive members identified as test data")
    excluded = [{"id": pair["id"], "reason": "NegativeROA or degenerate polygon present; semantics/geometry not guessed",
                 "negative_region_entries": pair["negative_region_entries"],
                 "degenerate_region_entries": pair["degenerate_region_entries"]}
                for pair in audit["pairs"] if pair["negative_region_entries"] or pair["degenerate_region_entries"]]
    eligible_pairs = [pair for pair in audit["pairs"] if not pair["negative_region_entries"]
                      and not pair["degenerate_region_entries"]]
    eligible_patients = {pair["patient_id"] for pair in eligible_pairs}
    if args.train_patient_ids or args.val_patient_ids:
        if not args.train_patient_ids or not args.val_patient_ids:
            raise ValueError("Supply both explicit train and validation patient ID lists")
        groups = {"train": args.train_patient_ids.split(","), "validation": args.val_patient_ids.split(",")}
        if any(len(ids) != len(set(ids)) or not set(ids) <= eligible_patients for ids in groups.values()):
            raise ValueError("Explicit patient lists contain duplicates or ineligible/unknown IDs")
        if set(groups["train"]) & set(groups["validation"]):
            raise ValueError("Train and validation patient groups overlap")
    else:
        groups = split_patients(list(eligible_patients), args.train_patients, args.val_patients, args.seed)
    split_by_patient = {patient: split for split, ids in groups.items() for patient in ids}
    all_candidates = {"train": [], "validation": []}
    cache, image_audits = {}, []
    with ZipFile(archive_path) as archive:
        for pair in eligible_pairs:
            split = split_by_patient.get(pair["patient_id"])
            if split is None:
                continue
            with Image.open(io.BytesIO(archive.read(pair["image_member"]))) as source:
                width, height = source.size
                image = source.convert("RGB")
                tiff_resolution = {str(tag): str(source.tag_v2.get(tag)) for tag in (282, 283, 296)
                                   if hasattr(source, "tag_v2") and source.tag_v2.get(tag) is not None}
            regions = parse_regions(archive.read(pair["xml_member"]))
            labels, raster_log = rasterize_regions(regions, width, height)
            forbidden = [r["instance_id"] for r in raster_log if r["clipped_to_image"]]
            candidates = candidate_crops(labels, args.crop_size, args.max_nuclei, forbidden, border_policy)
            spacing = {"um_per_pixel": None, "source": "not_verified", "tiff_resolution_tags": tiff_resolution,
                       "note": "TIFF display resolution is not assumed to be physical tissue spacing; no resizing performed."}
            for candidate in candidates:
                candidate.update(source_image_id=pair["id"], patient_id=pair["patient_id"], split=split,
                                 source_image_member=pair["image_member"], source_xml_member=pair["xml_member"],
                                 physical_spacing=spacing)
            all_candidates[split].extend(candidates)
            cache[pair["id"]] = (image, labels)
            image_audits.append({"id": pair["id"], "patient_id": pair["patient_id"], "split": split,
                                 "width": width, "height": height, "eligible_crops": len(candidates),
                                 "rasterization": raster_log})
    selected = []
    for split, required in (("train", args.train_crops), ("validation", args.val_crops)):
        try:
            selected.extend(select_balanced(all_candidates[split], required, args.seed))
        except ValueError as error:
            detail = {item["id"]: item["eligible_crops"] for item in image_audits if item["split"] == split}
            raise ValueError(f"{split}: {error} Per-image eligible counts: {detail}") from error
    condition = {"crop_size": args.crop_size, "grid_stride": args.crop_size, "min_nuclei": 1,
                 "max_nuclei": args.max_nuclei, "border_policy": border_policy,
                 "zero_foreground_border": border_policy == "complete",
                 "all_visible_instances_complete_in_source_mask": border_policy == "complete",
                 "clipped_to_source_image_instances_excluded": border_policy == "complete", "ignored_pixels": 0,
                 "negative_roa_images_excluded": True,
                 "degenerate_polygon_images_excluded": True,
                 "partial_instance_policy": "All visible pixels of every included instance remain supervised; truncated IDs are listed per crop.",
                 "selection_bias": f"At most {args.max_nuclei} visible nuclei and selected engineering patients; not representative whole-slide performance."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir()
    (out / "masks").mkdir()
    records = []
    for candidate in selected:
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        sample_id = f"{candidate['source_image_id']}_x{x0}_y{y0}"
        image_path, mask_path = out / "images" / f"{sample_id}.png", out / "masks" / f"{sample_id}.png"
        image, labels = cache[candidate["source_image_id"]]
        mask_crop = labels[y0:y1, x0:x1].astype(np.uint16)
        centroids = []
        for instance_id in candidate["instance_ids"]:
            ys, xs = np.where(mask_crop == instance_id)
            centroids.append({"instance_id": instance_id, "xy": [float(xs.mean()), float(ys.mean())]})
        image.crop((x0, y0, x1, y1)).save(image_path)
        Image.fromarray(mask_crop).save(mask_path)
        records.append({**candidate, "id": sample_id, "image_path": image_path.relative_to(out).as_posix(),
                        "mask_path": mask_path.relative_to(out).as_posix(),
                        "gt_mask": mask_path.relative_to(out).as_posix(), "path_base": "manifest_directory",
                        "source_archive": str(archive_path),
                        "instance_count_definition": "Distinct nonzero IDs with at least one visible pixel; includes partial objects, not centroid-in-crop filtering.",
                        "visible_centroids_xy": centroids,
                        "visible_centroid_definition": "Arithmetic mean of visible instance pixels in crop-local (x,y); can fall outside a nonconvex instance. GT-derived diagnostic metadata, never an evaluation-time prompt.",
                        "sampling_condition": condition})
    for name, rows in (("manifest", records), ("train", [r for r in records if r["split"] == "train"]),
                       ("validation", [r for r in records if r["split"] == "validation"])):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"schema_version": 1, "status": "provisional_engineering_split",
              "manifest_path_contract": "image_path, mask_path, and gt_mask are relative to the directory containing the manifest; source archive paths are provenance only.",
              "not_an_official_or_published_benchmark": True, "original_test_accessed": False,
              "source_archive": str(archive_path), "seed": args.seed, "patient_groups": groups,
              "selected_crop_counts": {split: sum(r["split"] == split for r in records) for split in groups},
              "candidate_crop_counts": {split: len(rows) for split, rows in all_candidates.items()},
              "sampling_condition": condition, "excluded_images": excluded, "image_audits": image_audits,
              "training_archive_audit": audit,
              "rasterizer": "Pillow inclusive polygons, vertices rounded floor(x+0.5), earliest XML region wins overlap. No MATLAB-parity claim."}
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-nuclei", type=int, default=8)
    parser.add_argument("--border-policy", choices=("complete", "visible"), default="complete",
                        help="complete requires every nucleus fully inside the crop; visible supervises every visible instance including recorded partial objects")
    parser.add_argument("--train-patients", type=int, default=8)
    parser.add_argument("--val-patients", type=int, default=4)
    parser.add_argument("--train-patient-ids", help="Optional comma-separated TCGA patient IDs; requires validation list too")
    parser.add_argument("--val-patient-ids", help="Optional comma-separated TCGA patient IDs")
    parser.add_argument("--train-crops", type=int, default=32)
    parser.add_argument("--val-crops", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = prepare(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("status", "selected_crop_counts", "candidate_crop_counts", "patient_groups")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
