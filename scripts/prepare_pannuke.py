#!/usr/bin/env python3
"""Convert PanNuke folds into PNG images plus centroid manifests for NuNext detection.

Input: one directory per fold containing ``images.npy`` (N,256,256,3) and ``masks.npy``
(N,256,256,6), anywhere below it (the official zips nest them as
``Fold 1/images/fold1/images.npy``). Channels 0-4 of ``masks.npy`` are per-class instance
maps (neoplastic, inflammatory, connective, dead, epithelial); channel 5 is background.

Output: ``<out>/images/fold{k}_{index:04d}.png`` and ``<out>/fold{k}.jsonl`` with one row
per image: id, fold, image path, width, height, centroids [[x, y], ...] in pixel units
(the mean pixel position of each instance), and their class names. Arrays are memory-mapped,
so an 8 GB ``masks.npy`` does not need to fit in RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ("neoplastic", "inflammatory", "connective", "dead", "epithelial")


def find_array(root: Path, name: str) -> Path:
    found = sorted(root.rglob(name))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under {root}, found {len(found)}")
    return found[0]


def instance_centroids(mask: np.ndarray) -> tuple[list[list[float]], list[str]]:
    """Centroids of every instance in a (H, W, 6) PanNuke mask, sorted top-to-bottom."""
    if mask.ndim != 3 or mask.shape[2] < len(CLASSES):
        raise ValueError("mask must have shape (H, W, >=5)")
    rows = []
    for channel, name in enumerate(CLASSES):
        layer = np.asarray(mask[:, :, channel])
        ids = np.unique(layer)
        for instance in ids[ids > 0]:
            ys, xs = np.nonzero(layer == instance)
            # +0.5: pixel centres, so a one-pixel nucleus at column 0 sits at x = 0.5.
            rows.append((float(xs.mean() + 0.5), float(ys.mean() + 0.5), name))
    rows.sort(key=lambda row: (row[1], row[0]))
    return [[round(x, 2), round(y, 2)] for x, y, _ in rows], [name for _, _, name in rows]


def convert_fold(fold_dir: Path, fold: int, out: Path, limit: int | None = None) -> dict:
    from PIL import Image

    images = np.load(find_array(fold_dir, "images.npy"), mmap_mode="r")
    masks = np.load(find_array(fold_dir, "masks.npy"), mmap_mode="r")
    if images.shape[0] != masks.shape[0] or images.shape[1:3] != masks.shape[1:3]:
        raise ValueError(f"images {images.shape} and masks {masks.shape} disagree")
    count = images.shape[0] if limit is None else min(limit, images.shape[0])
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / f"fold{fold}.jsonl"
    nuclei = 0
    with manifest.open("w") as stream:
        for index in range(count):
            identifier = f"fold{fold}_{index:04d}"
            pixels = np.clip(np.asarray(images[index]), 0, 255).astype(np.uint8)
            path = out / "images" / f"{identifier}.png"
            Image.fromarray(pixels).save(path)
            centroids, classes = instance_centroids(masks[index])
            nuclei += len(centroids)
            stream.write(json.dumps({
                "id": identifier, "fold": fold, "image_path": str(path.resolve()),
                "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
                "centroids": centroids, "classes": classes,
            }) + "\n")
    return {"fold": fold, "images": count, "nuclei": nuclei, "manifest": str(manifest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", action="append", required=True, metavar="K=DIR",
                        help="Fold number and its extracted directory, e.g. 1=/data/pannuke/Fold1 (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="Only the first N images per fold (smoke tests)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    report = []
    for spec in args.fold:
        number, _, directory = spec.partition("=")
        if not number.isdigit() or not directory:
            raise SystemExit(f"--fold must be K=DIR, got {spec!r}")
        report.append(convert_fold(Path(directory), int(number), out, args.limit))
        print(json.dumps(report[-1]), flush=True)
    (out / "report.json").write_text(json.dumps({"folds": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
