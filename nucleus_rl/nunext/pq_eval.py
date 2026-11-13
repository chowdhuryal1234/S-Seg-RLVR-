"""Panoptic quality for NuNext points: point prompts -> frozen SAM2 masks -> PQ against PanNuke instances.

The detector outputs nucleus centres only; PQ needs one mask per nucleus. Each point is sent to a
frozen SAM2 as a positive point prompt (multimask output, highest-scoring mask kept), overlapping
masks are resolved per pixel (nucleus_rl.rewards.masks_to_instances), and the label map is scored
with the project's PQ (nucleus_rl.rewards.instance_metrics: a match is IoU > 0.5).

Modes:
  oracle-point  ground-truth centroids as point prompts: the ceiling for this detector + this SAM
  oracle-box    ground-truth tight box + centroid: the prompt style of the project's earlier branches
  pred-point    the detector's predicted points, read from an evaluate run's predictions.jsonl

Inference-only variants for point prompts: --negatives K adds the K nearest other prompt points
(within --negative-radius px) as negative points, so SAM is told where the neighbouring nuclei are;
--mask-choice smallest keeps the smallest non-empty of SAM's three candidate masks instead of the
highest-scoring one (a single point is ambiguous between a nucleus, a cluster and the tissue).

Reported: dataset-level PQ = sum of matched IoU / (TP + FP/2 + FN/2) over all images (with DQ and
SQ), and the mean per-image PQ over images that have at least one annotated nucleus.
Runs sharded over GPUs when started by nucleus_rl.nunext.launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

MODES = ("oracle-point", "oracle-box", "pred-point")
MASK_CHOICES = ("score", "smallest")


def best_masks(masks: np.ndarray, scores: np.ndarray, choice: str = "score") -> tuple[np.ndarray, np.ndarray]:
    """(N, C, H, W) masks + (N, C) scores -> one mask per prompt; a squeezed batch of one is restored.

    choice "score": SAM's highest predicted IoU. "smallest": the smallest non-empty candidate
    (falls back to the score when every candidate is empty).
    """
    if choice not in MASK_CHOICES:
        raise ValueError(f"mask choice must be one of {MASK_CHOICES}")
    masks, scores = np.asarray(masks) > 0, np.asarray(scores)
    if masks.ndim == 3:
        masks, scores = masks[None], scores.reshape(1, -1)
    pick = scores.argmax(axis=1)
    if choice == "smallest":
        areas = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2).astype(float)
        areas[areas == 0] = np.inf
        smallest = areas.argmin(axis=1)
        pick = np.where(np.isfinite(areas.min(axis=1)), smallest, pick)
    rows = np.arange(len(pick))
    return masks[rows, pick], scores[rows, pick]


def negative_points(points, count: int, radius: float) -> list[list[tuple[float, float]]] | None:
    """For each point, its ``count`` nearest other points within ``radius`` px (nearest first)."""
    if count <= 0:
        return None
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    result = []
    for index, point in enumerate(array):
        distances = np.hypot(*(array - point).T)
        distances[index] = np.inf
        order = [j for j in np.argsort(distances, kind="stable")[:count] if distances[j] <= radius]
        result.append([tuple(array[j]) for j in order])
    return result


def oracle_boxes(reference: np.ndarray, count: int) -> list[tuple[float, float, float, float]]:
    """Tight box [x0, y0, x1, y1) of labels 1..count (a label lost to an overlap gets a 1-pixel box at 0,0)."""
    boxes = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(reference == label)
        boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)) if len(xs)
                     else (0.0, 0.0, 1.0, 1.0))
    return boxes


def prompts_for(row: dict, mode: str, predictions: dict | None, reference: np.ndarray):
    if mode == "pred-point":
        if row["id"] not in predictions:
            raise KeyError(f"No prediction for {row['id']}")
        return [tuple(p) for p in predictions[row["id"]]], None
    points = [tuple(c) for c in row["centroids"]]
    return points, (oracle_boxes(reference, len(points)) if mode == "oracle-box" else None)


def score_image(row: dict, mode: str, predictions, segment, negatives: int = 0, radius: float = 32.0) -> dict:
    """segment(image (H,W,3) uint8, points [(x,y)], boxes or None, negatives or None) -> (masks (N,H,W), scores (N,))."""
    from PIL import Image
    from nucleus_rl.rewards import instance_metrics, masks_to_instances

    image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    reference = np.asarray(Image.open(row["mask_path"])).astype(np.int64)
    points, boxes = prompts_for(row, mode, predictions, reference)
    height, width = reference.shape
    if points:
        masks, scores = segment(image, points, boxes, negative_points(points, negatives, radius))
        instances = masks_to_instances(np.asarray(masks, dtype=np.float32), scores=np.asarray(scores))
    else:
        instances = np.zeros((height, width), dtype=np.int32)
    metrics = instance_metrics(instances, reference)
    return {"id": row["id"], "prompts": len(points), "gt_count": metrics["gt_count"],
            "pred_count": metrics["pred_count"], "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "iou_sum": float(metrics["sq"] * metrics["tp"]), "pq": float(metrics["pq"])}


def summarize(records: list[dict]) -> dict:
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    iou = sum(r["iou_sum"] for r in records)
    denominator = tp + 0.5 * fp + 0.5 * fn
    annotated = [r["pq"] for r in records if r["gt_count"] > 0]
    return {"images": len(records), "nuclei": sum(r["gt_count"] for r in records),
            "prompts": sum(r["prompts"] for r in records), "tp": tp, "fp": fp, "fn": fn,
            "pq": iou / denominator if denominator else float("nan"),
            "dq": tp / denominator if denominator else float("nan"),
            "sq": iou / tp if tp else float("nan"),
            "mean_image_pq": float(np.mean(annotated)) if annotated else float("nan"),
            "images_with_nuclei": len(annotated)}


class SAMSegmenter:
    """Frozen SAM2 image predictor; prompts are sent in chunks of ``chunk`` per decoder call."""

    def __init__(self, checkpoint: str, config: str, device: str, chunk: int = 64, mask_choice: str = "score"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch, self.device, self.chunk, self.mask_choice = torch, device, chunk, mask_choice
        model = build_sam2(config, str(Path(checkpoint).resolve()), device=device, mode="eval")
        model.requires_grad_(False)
        self.predictor = SAM2ImagePredictor(model)

    def __call__(self, image, points, boxes, negatives=None):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                    enabled=self.device.startswith("cuda")):
            self.predictor.set_image(image.copy())
            all_masks, all_scores = [], []
            for start in range(0, len(points), self.chunk):
                coords, labels = prompt_arrays(points[start:start + self.chunk],
                                               None if negatives is None else negatives[start:start + self.chunk])
                box = None if boxes is None else np.asarray(boxes[start:start + self.chunk], dtype=np.float32)
                masks, scores, _ = self.predictor.predict(point_coords=coords, point_labels=labels, box=box,
                                                          multimask_output=box is None, normalize_coords=True)
                masks, scores = best_masks(masks, scores, self.mask_choice if box is None else "score")
                all_masks.append(masks)
                all_scores.append(scores)
        return np.concatenate(all_masks), np.concatenate(all_scores)


def prompt_arrays(points, negatives=None):
    """(N, 1+K, 2) coords and (N, 1+K) labels: the positive point, its negatives (0), padding (-1)."""
    width = 1 + (max((len(n) for n in negatives), default=0) if negatives else 0)
    coords = np.zeros((len(points), width, 2), dtype=np.float32)
    labels = np.full((len(points), width), -1, dtype=np.int32)
    for index, point in enumerate(points):
        coords[index, 0], labels[index, 0] = point, 1
        for slot, negative in enumerate(negatives[index] if negatives else [], start=1):
            coords[index, slot], labels[index, slot] = negative, 0
    return coords, labels


def run(rows, mode, predictions, segment, rank=0, world=1, negatives=0, radius=32.0) -> list[dict]:
    return [score_image(row, mode, predictions, segment, negatives, radius) for row in rows[rank::world]]


def main(argv=None):
    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.data import read_manifests

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, help="Rows with mask_path (dataset v2)")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--predictions", help="predictions.jsonl of an evaluate run (pred-point mode)")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--negatives", type=int, default=0, help="Nearest other points sent as negative prompts")
    parser.add_argument("--negative-radius", type=float, default=32.0, help="Only neighbours within this many px")
    parser.add_argument("--mask-choice", choices=MASK_CHOICES, default="score")
    args = parser.parse_args(argv)
    if args.mode == "pred-point" and not args.predictions:
        parser.error("--predictions is required for pred-point")

    device = dist.setup() or "cuda"
    rows = read_manifests(args.manifest, args.limit)
    if any("mask_path" not in row for row in rows):
        raise ValueError("Manifest rows lack mask_path: rebuild the dataset with the current prepare_pannuke.py")
    predictions = None
    if args.predictions:
        records = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
        predictions = {r["id"]: r["pred"] if r.get("valid", True) else [] for r in records}
    started = time.time()
    segment = SAMSegmenter(args.sam_checkpoint, args.sam_config, device, args.chunk, args.mask_choice)
    records = dist.gather_objects(run(rows, args.mode, predictions, segment, dist.rank(), dist.world_size(),
                                      args.negatives, args.negative_radius))
    if dist.is_main():
        order = {row["id"]: i for i, row in enumerate(rows)}
        records.sort(key=lambda r: order[r["id"]])
        summary = {"mode": args.mode, "sam_checkpoint": args.sam_checkpoint, "sam_config": args.sam_config,
                   "predictions": args.predictions, "negatives": args.negatives,
                   "negative_radius": args.negative_radius, "mask_choice": args.mask_choice, **summarize(records),
                   "seconds": round(time.time() - started, 1)}
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output / "per_image.jsonl").open("w") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in records)
        print("summary", json.dumps(summary), flush=True)
    dist.teardown()


if __name__ == "__main__":
    main()
