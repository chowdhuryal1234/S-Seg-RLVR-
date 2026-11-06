#!/usr/bin/env python3
"""Rasterise MoNuSeg into the <out>/All/{Images,Labels} layout used by prepare_consep.py.

Uses the project's own rasteriser (prepare_monuseg.rasterize_regions: positive
polygons, earliest wins), so oracle crops carry exactly the labels the models
were trained and scored on. Only the 30 official training images are kept.
Images with NegativeROA or degenerate polygons are excluded, as in every
MoNuSeg dataset here, and listed with the overlap and fragment counts that
describe each image's annotation geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import load_official30_cohort
from scripts.audit_monuseg import parse_regions
from scripts.prepare_monuseg import rasterize_regions


def build(source, out):
    source, out = Path(source), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    images = {p.stem: p for p in source.rglob("*") if p.suffix.lower() in (".tif", ".tiff")}
    xmls = {p.stem: p for p in source.rglob("*.xml")}
    canonical = load_official30_cohort()["image_ids"]
    missing = [i for i in canonical if i not in images or i not in xmls]
    if missing:
        raise FileNotFoundError(f"Official30 images without an image/XML pair: {missing}")
    (out / "All" / "Images").mkdir(parents=True)
    (out / "All" / "Labels").mkdir(parents=True)
    from PIL import Image
    report = {"kept": [], "excluded": {}}
    for image_id in canonical:
        regions = parse_regions(xmls[image_id].read_bytes())
        with Image.open(images[image_id]) as image:
            width, height = image.size
        try:
            labels, log = rasterize_regions(regions, width, height)
        except ValueError as error:
            report["excluded"][image_id] = str(error)
            continue
        shutil.copyfile(images[image_id], out / "All" / "Images" / images[image_id].name)
        np.save(out / "All" / "Labels" / f"{image_id}.npy", labels)
        report["kept"].append({
            "id": image_id, "regions": len(log), "instances": int(len(np.unique(labels[labels > 0]))),
            "dropped_regions": sum(e["dropped"] for e in log),
            "regions_with_overlap": sum(e["overlap_pixels_removed"] > 0 for e in log),
            "overlap_pixels_removed": sum(e["overlap_pixels_removed"] for e in log),
            "fragmented_regions": sum(e["fragments_4_connected"] > 1 for e in log)})
    (out / "layout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Folder holding the MoNuSeg training .tif and .xml files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build(args.source, args.out)
    print(f"kept {len(report['kept'])}, excluded {sorted(report['excluded'])}")


if __name__ == "__main__":
    main()
