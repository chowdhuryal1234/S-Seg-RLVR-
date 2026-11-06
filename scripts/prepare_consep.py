#!/usr/bin/env python3
"""Cut CoNSeP (or any Images/Labels dataset) into 128 px evaluation crops, sparse and dense.

Evaluation only: nothing written here trains a model. The layout is the public
CoNSeP release, ``<root>/{Train,Test}/Images/*.png`` with instance labels in
``<root>/{Train,Test}/Labels/*.mat`` (key ``inst_map``); ``.npy`` or 16-bit PNG
instance maps with the same stem are also accepted. Crops follow the MoNuSeg
protocol: a 128 px grid with stride 128, every visible labelled pixel kept,
partial nuclei included. Crops with 1-8 visible nuclei form ``sparse.jsonl``,
9-32 form ``dense.jsonl``; empty and denser crops are counted and skipped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

SUBSETS = (("sparse", 1, 8), ("dense", 9, 32))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_dir(parent, name):
    """Case-insensitive child directory lookup (Train/train, Images/images)."""
    matches = [p for p in Path(parent).iterdir() if p.is_dir() and p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one '{name}' folder in {parent}, found {[p.name for p in matches]}")
    return matches[0]


def load_instance_map(path):
    path = Path(path)
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        data = loadmat(path)
        if "inst_map" not in data:
            raise KeyError(f"{path} has no 'inst_map' (keys: {sorted(k for k in data if not k.startswith('__'))})")
        labels = data["inst_map"]
    elif path.suffix.lower() == ".npy":
        labels = np.load(path, allow_pickle=False)
        if labels.ndim == 3:  # HoVer-Net style [..., 0] instance channel
            labels = labels[..., 0]
    else:
        labels = np.array(Image.open(path))
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.all(np.equal(np.mod(labels, 1), 0)) or labels.min() < 0:
        raise ValueError(f"{path}: expected a 2-D nonnegative integer instance map")
    labels = labels.astype(np.int64)
    if labels.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"{path}: instance IDs exceed uint16")
    return labels


def label_path(labels_dir, stem):
    for suffix in (".mat", ".npy", ".png"):
        candidate = Path(labels_dir) / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No label for {stem} in {labels_dir}")


def balanced_sample(rows, limit, seed):
    """Round-robin over source images so a cap never favours the first images."""
    if limit is None or len(rows) <= limit:
        return rows
    buckets = {}
    for row in rows:
        buckets.setdefault(row["source_image_id"], []).append(row)
    rng = random.Random(seed)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    picked = []
    while len(picked) < limit:
        for key in keys:
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].pop())
    return sorted(picked, key=lambda row: row["id"])


def prepare(root, out, splits=("Test",), crop_size=128, max_per_subset=None, seed=20261106, dataset="consep"):
    """Crop every image under <root>/<split>/Images. Any dataset in this layout works (``dataset`` names it)."""
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new destination")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    candidates = {name: [] for name, _, _ in SUBSETS}
    skipped, histogram, sources = Counter(), Counter(), []
    for split in splits:
        split_dir = find_dir(root, split)
        images_dir, labels_dir = find_dir(split_dir, "Images"), find_dir(split_dir, "Labels")
        for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff")):
            labels_file = label_path(labels_dir, image_path.stem)
            image = Image.open(image_path).convert("RGB")
            labels = load_instance_map(labels_file)
            if labels.shape != (image.height, image.width):
                raise ValueError(f"{image_path.name}: image {image.size} and label {labels.shape[::-1]} differ")
            source_id = f"{dataset}-{split.lower()}-{image_path.stem}"
            full_counts = np.bincount(labels.ravel())
            sources.append({"source_image_id": source_id, "image": image_path.name, "label": labels_file.name,
                            "image_sha256": digest(image_path), "label_sha256": digest(labels_file),
                            "size": list(image.size), "instances": int(np.count_nonzero(np.unique(labels)))})
            for y in range(0, image.height - crop_size + 1, crop_size):
                for x in range(0, image.width - crop_size + 1, crop_size):
                    crop = labels[y:y + crop_size, x:x + crop_size]
                    ids, counts = np.unique(crop[crop > 0], return_counts=True)
                    histogram[len(ids)] += 1
                    subset = next((name for name, low, high in SUBSETS if low <= len(ids) <= high), None)
                    if subset is None:
                        skipped["empty" if len(ids) == 0 else "over_32"] += 1
                        continue
                    crop_id = f"{source_id}_x{x}_y{y}"
                    image_out, mask_out = out / "images" / f"{crop_id}.png", out / "masks" / f"{crop_id}.png"
                    image.crop((x, y, x + crop_size, y + crop_size)).save(image_out)
                    Image.fromarray(crop.astype(np.uint16)).save(mask_out)
                    candidates[subset].append({
                        "id": crop_id, "split": "validation", "dataset": dataset, "subset": subset,
                        "source_split": split, "patient_id": source_id, "source_image_id": source_id,
                        "image_path": image_out.relative_to(out).as_posix(),
                        "mask_path": mask_out.relative_to(out).as_posix(),
                        "bbox_xyxy": [x, y, x + crop_size, y + crop_size],
                        "instance_count": int(len(ids)), "instance_ids": ids.tolist(),
                        "truncated_instance_ids": [int(i) for i, c in zip(ids, counts) if c != full_counts[i]],
                        "image_sha256": digest(image_out), "mask_sha256": digest(mask_out)})
    selected = {name: balanced_sample(rows, max_per_subset, seed) for name, rows in candidates.items()}
    kept = {row["id"] for rows in selected.values() for row in rows}
    for rows in candidates.values():
        for row in rows:
            if row["id"] not in kept:
                (out / row["image_path"]).unlink()
                (out / row["mask_path"]).unlink()
    for name, rows in selected.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"dataset": dataset, "splits": list(splits), "crop_size": crop_size, "stride": crop_size,
              "subsets": {name: {"min_nuclei": low, "max_nuclei": high, "candidates": len(candidates[name]),
                                 "selected": len(selected[name]),
                                 "nuclei": sum(r["instance_count"] for r in selected[name])}
                          for name, low, high in SUBSETS},
              "skipped_crops": dict(skipped), "visible_count_histogram": dict(sorted(histogram.items())),
              "max_per_subset": max_per_subset, "seed": seed, "sources": sources,
              "scope": "Evaluation-only crops; the grid leaves right/bottom remainders uncovered (1000 px -> 7 x 128 px)."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="CoNSeP folder containing Train/ and Test/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="Test", help="Comma-separated CoNSeP splits to crop (default Test)")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-per-subset", type=int, help="Image-balanced cap on crops per subset")
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args(argv)
    report = prepare(args.root, args.out, tuple(s.strip() for s in args.splits.split(",") if s.strip()),
                     args.crop_size, args.max_per_subset, args.seed)
    print(json.dumps({k: report[k] for k in ("subsets", "skipped_crops")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
