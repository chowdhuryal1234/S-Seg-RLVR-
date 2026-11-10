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


def mask_centroid(png: bytes) -> tuple[float, float] | None:
    """Centroid (pixel centres) of one binary instance mask stored as an image."""
    import io
    from PIL import Image

    ys, xs = np.nonzero(np.asarray(Image.open(io.BytesIO(png)).convert("L")) > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


def convert_parquet(path: Path, fold: int, out: Path, limit: int | None = None) -> dict:
    """Same output as convert_fold, from the Hugging Face RationAI/PanNuke parquet layout.

    That dataset stores each image with one binary mask per instance (``instances``) and its class
    index (``categories``), made from the official masks.npy by splitting every class channel into
    its non-zero instance IDs, i.e. the same instances convert_fold uses.
    """
    import io
    import pyarrow.parquet as pq
    from PIL import Image

    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / f"fold{fold}.jsonl"
    count = nuclei = empty_masks = 0
    with manifest.open("w") as stream:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "instances", "categories"]):
            for record in batch.to_pylist():
                if limit is not None and count >= limit:
                    break
                identifier = f"fold{fold}_{count:04d}"
                image = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
                image_path = out / "images" / f"{identifier}.png"
                image.save(image_path)
                rows = []
                for mask, category in zip(record["instances"] or [], record["categories"] or []):
                    centroid = mask_centroid(mask["bytes"])
                    if centroid is None:
                        empty_masks += 1
                        continue
                    rows.append((*centroid, CLASSES[int(category)]))
                rows.sort(key=lambda row: (row[1], row[0]))
                nuclei += len(rows)
                stream.write(json.dumps({
                    "id": identifier, "fold": fold, "image_path": str(image_path.resolve()),
                    "width": image.width, "height": image.height,
                    "centroids": [[round(x, 2), round(y, 2)] for x, y, _ in rows],
                    "classes": [name for _, _, name in rows],
                }) + "\n")
                count += 1
            if limit is not None and count >= limit:
                break
    return {"fold": fold, "images": count, "nuclei": nuclei, "empty_masks": empty_masks,
            "manifest": str(manifest), "source": str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fold", action="append", metavar="K=DIR",
                        help="Fold number and its extracted official directory, e.g. 1=/data/pannuke/Fold1 (repeatable)")
    source.add_argument("--parquet", action="append", metavar="K=FILE",
                        help="Fold number and its RationAI/PanNuke parquet file (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="Only the first N images per fold (smoke tests)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    report = []
    for spec in args.fold or args.parquet:
        number, _, location = spec.partition("=")
        if not number.isdigit() or not location:
            raise SystemExit(f"Expected K=PATH, got {spec!r}")
        convert = convert_fold if args.fold else convert_parquet
        report.append(convert(Path(location), int(number), out, args.limit))
        print(json.dumps(report[-1]), flush=True)
    (out / "report.json").write_text(json.dumps({"folds": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
