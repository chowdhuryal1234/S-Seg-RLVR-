#!/usr/bin/env python3
"""SAM oracle: how well can frozen SAM2 reproduce a dataset's own annotations?

Every annotated nucleus gets the prompt a perfect policy would give: its tight
box plus the foreground pixel nearest its centroid (scripts/prompt_diagnostic.py).
Frozen SAM2 segments from those prompts and the result is scored against the
same annotation with the project's PQ (IoU > 0.5). No VLM and no training.

Because the prompts are perfect, a low oracle PQ means SAM's idea of a nucleus
disagrees with the annotation (boundary convention, merged or split nuclei,
rasterisation), not that the policy failed. Comparing datasets cut with the same
crop protocol separates "the labels limit PQ" from "the policy limits PQ".
Per-nucleus IoU (prompt i's raw SAM mask vs annotated nucleus i) shows where.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE_BINS = ((0, 50), (50, 150), (150, 400), (400, 10 ** 9))  # visible pixels per nucleus in the crop


def _mean(values):
    return statistics.fmean(values) if values else None


def nucleus_records(reference, raw_masks, instance_ids, truncated):
    """IoU of each prompt's own SAM mask with the nucleus it was derived from."""
    records = []
    for index, instance_id in enumerate(instance_ids):
        target = reference == instance_id
        predicted = raw_masks[index]
        union = np.logical_or(target, predicted).sum()
        records.append({"instance_id": int(instance_id), "area": int(target.sum()),
                        "iou": float(np.logical_and(target, predicted).sum() / union) if union else 0.0,
                        "predicted_area": int(predicted.sum()), "truncated": int(instance_id) in truncated})
    return records


def summarize(crops):
    nuclei = [n for crop in crops for n in crop["nuclei"]]
    ious = [n["iou"] for n in nuclei]

    def iou_stats(items):
        values = [n["iou"] for n in items]
        return {"nuclei": len(values), "mean_iou": _mean(values),
                "median_iou": statistics.median(values) if values else None,
                "iou_over_0.5": sum(v > 0.5 for v in values) / len(values) if values else None}

    by_source = defaultdict(list)
    for crop in crops:
        by_source[crop["source_image_id"]].append(crop)
    return {
        "crops": len(crops), "mean_pq": _mean([c["pq"] for c in crops]),
        "mean_precision": _mean([c["precision"] for c in crops]), "mean_recall": _mean([c["recall"] for c in crops]),
        "tp": sum(c["tp"] for c in crops), "fp": sum(c["fp"] for c in crops), "fn": sum(c["fn"] for c in crops),
        "gt": sum(c["gt_count"] for c in crops), "mean_count_error": _mean([c["count_error"] for c in crops]),
        "nucleus_iou": iou_stats(nuclei),
        "nucleus_iou_by_size": {f"{lo}-{hi if hi < 10 ** 9 else 'inf'}px": iou_stats([n for n in nuclei if lo <= n["area"] < hi])
                                for lo, hi in SIZE_BINS},
        "nucleus_iou_truncated_vs_whole": {"truncated": iou_stats([n for n in nuclei if n["truncated"]]),
                                           "whole": iou_stats([n for n in nuclei if not n["truncated"]])},
        "sam_mask_larger_than_annotation": (sum(n["predicted_area"] > n["area"] for n in nuclei) / len(nuclei)
                                            if nuclei else None),
        "by_source_image": {source: {"crops": len(items), "mean_pq": _mean([c["pq"] for c in items]),
                                     "median_nucleus_iou": statistics.median([n["iou"] for c in items for n in c["nuclei"]])}
                            for source, items in sorted(by_source.items())},
        "pooled_nucleus_iou_mean": _mean(ious),
    }


def run(manifest, sam, max_objects):
    from PIL import Image
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.rewards import ObjectPrompt, instance_metrics
    from scripts.prompt_diagnostic import oracle_prompt_objects
    crops = []
    for row in read_manifest(manifest):
        image = Image.open(row["image_path"]).convert("RGB")
        reference = np.asarray(Image.open(row["mask_path"])).astype(np.int64)
        instance_ids = np.unique(reference[reference > 0])
        objects = [ObjectPrompt(box=tuple(o["box"]), point=tuple(o["point"]))
                   for o in oracle_prompt_objects(reference, padding=0, max_objects=max_objects)]
        instances = sam.predict(image, objects, image_key=row["image_path"])
        metrics = instance_metrics(instances, reference)
        crops.append({"id": row["id"], "source_image_id": row.get("source_image_id", row["id"]),
                      "subset": row.get("subset"), **{k: metrics[k] for k in (
                          "pq", "precision", "recall", "tp", "fp", "fn", "gt_count", "pred_count", "count_error")},
                      "nuclei": nucleus_records(reference, sam.last_raw_masks, instance_ids,
                                                set(row.get("truncated_instance_ids", [])))})
    return crops


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True,
                        help="name=path; repeat for each crop manifest (e.g. consep_sparse=.../sparse.jsonl)")
    parser.add_argument("--sam", action="append", required=True,
                        help="name=checkpoint=config; e.g. tiny=checkpoints/sam2.1_hiera_tiny.pt=configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    from nucleus_rl.segmenter import FrozenSAM2
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    manifests = dict(item.split("=", 1) for item in args.manifest)
    summary = {"scope": "Oracle prompts derived from the annotation itself; no VLM, no training.",
               "max_objects": args.max_objects, "results": {}}
    for spec in args.sam:
        name, checkpoint, config = spec.split("=", 2)
        sam = FrozenSAM2(checkpoint, config, device=args.device)
        for label, path in manifests.items():
            crops = run(path, sam, args.max_objects)
            with (output / f"{name}_{label}_crops.jsonl").open("w") as stream:
                stream.writelines(json.dumps(crop) + "\n" for crop in crops)
            summary["results"][f"{name}|{label}"] = {"manifest": path, "sam": sam.metadata["checkpoint_sha256"],
                                                     **summarize(crops)}
            result = summary["results"][f"{name}|{label}"]
            print(f"{name:6} {label:22} crops={result['crops']:4d} PQ={result['mean_pq']:.3f} "
                  f"nucleus IoU median={result['nucleus_iou']['median_iou']:.3f} "
                  f">0.5={result['nucleus_iou']['iou_over_0.5']:.3f}", flush=True)
        sam.verify_unchanged()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
