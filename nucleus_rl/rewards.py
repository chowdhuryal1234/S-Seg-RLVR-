"""Strict spatial-prompt parsing and CPU rewards for the first pilot.

The only training rewards are output validity and foreground IoU. Instance
metrics are diagnostics, not additional rewards. All coordinates refer to the
original input crop, before any segmenter-specific resizing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjectPrompt:
    """XYXY box with exclusive right/bottom edges, plus an interior XY point.

    Finite integer and fractional coordinates are accepted; booleans are not.
    Box edges may equal the image width/height. Points must remain below them.
    """

    box: tuple[float, float, float, float]
    point: tuple[float, float]


@dataclass(frozen=True)
class ParseResult:
    valid: bool
    objects: tuple[ObjectPrompt, ...] = ()
    error: str | None = None


_COMPLETION = re.compile(
    r"(?:<think>(?:(?!</?think>).)*</think>\s*)?<answer>(.*?)</answer>", re.DOTALL
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _coordinates(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON array of length {length}")
    result = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{name} coordinates must be numbers, not booleans")
        try:
            number = float(coordinate)
        except OverflowError as error:
            raise ValueError(f"{name} coordinates must be finite") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} coordinates must be finite")
        result.append(number)
    return tuple(result)


def parse_completion(
    text: str, *, width: int, height: int, max_objects: int = 8
) -> ParseResult:
    """Parse one complete answer, optionally preceded by one think block.

    Contract: <answer>{"objects":[{"box":[x0,y0,x1,y1],
    "point":[x,y]}]}</answer>. Unknown keys, duplicate keys, trailing text,
    incomplete JSON and non-finite numbers are rejected. Overlapping boxes are
    allowed. Empty objects is valid syntax (and scores zero IoU for nonempty GT).
    Invalid model text returns a ParseResult; invalid caller dimensions raise.
    """
    for name, value in (("width", width), ("height", height), ("max_objects", max_objects)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(text, str):
        return ParseResult(False, error="Completion must be text")
    if len(text) > 65536:
        return ParseResult(False, error="Completion exceeds 65536 characters")
    match = _COMPLETION.fullmatch(text.strip())
    if match is None:
        return ParseResult(False, error="Expected one complete answer block")
    try:
        payload = json.loads(
            match.group(1), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(payload, dict) or set(payload) != {"objects"}:
            raise ValueError("Answer must contain exactly the objects key")
        entries = payload["objects"]
        if not isinstance(entries, list) or len(entries) > max_objects:
            raise ValueError(f"objects must be an array with at most {max_objects} entries")
        objects = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"box", "point"}:
                raise ValueError("Each object must contain exactly box and point")
            box = _coordinates(entry["box"], 4, "box")
            point = _coordinates(entry["point"], 2, "point")
            x0, y0, x1, y1 = box
            x, y = point
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Box must have positive area and lie inside the image")
            if not (x0 <= x < x1 and y0 <= y < y1):
                raise ValueError("Point must lie inside the half-open box")
            objects.append(ObjectPrompt(box=box, point=point))
        return ParseResult(True, tuple(objects))
    except (ValueError, TypeError, RecursionError) as error:
        return ParseResult(False, error=str(error))


def _label_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty H x W label map")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric instance IDs")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} IDs must be finite and nonnegative")
    if array.dtype.kind == "f" and np.any(array != np.floor(array)):
        raise ValueError(f"{name} IDs must be integer-valued")
    return array


def _paired_labels(pred_instances: Any, gt_instances: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = _label_map(pred_instances, "prediction")
    gt = _label_map(gt_instances, "ground truth")
    if pred.shape != gt.shape:
        raise ValueError("Prediction and ground truth shapes must match")
    return pred, gt


def foreground_iou(pred_instances: Any, gt_instances: Any) -> float:
    """Binary foreground intersection/union, ignoring instance identity.

    Both masks empty -> 1. Exactly one empty -> 0. An invalid completion is
    handled separately by evaluate_prediction and never receives this reward.
    """
    pred, gt = _paired_labels(pred_instances, gt_instances)
    pred_fg, gt_fg = pred > 0, gt > 0
    union = np.count_nonzero(pred_fg | gt_fg)
    return float(np.count_nonzero(pred_fg & gt_fg) / union) if union else 1.0


def masks_to_instances(
    masks: Any, scores: Any = None, *, threshold: float = 0.5
) -> np.ndarray:
    """Resolve N probability/binary masks to a disjoint int32 H x W ID map.

    A pixel is eligible when its probability is strictly above threshold.
    Highest per-pixel probability wins overlaps; ties prefer higher mask score,
    then earlier input index. Output IDs retain input index + 1, even if another
    mask wins every pixel of an object. Never split IDs by connected components.
    For no candidates pass shape (0,H,W), not a shapeless empty list. Logits must
    be converted to probabilities by the caller before invoking this function.
    """
    probabilities = np.asarray(masks)
    if probabilities.ndim != 3 or min(probabilities.shape[1:]) == 0:
        raise ValueError("masks must have shape (N,H,W), with positive H and W")
    if probabilities.dtype.kind not in "biuf":
        raise ValueError("masks must be numeric probabilities or binary masks")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Mask probabilities must be finite and between zero and one")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    count, height, width = probabilities.shape
    if scores is None:
        quality = np.zeros(count, dtype=float)
    else:
        quality = np.asarray(scores, dtype=float)
        if quality.shape != (count,) or not np.all(np.isfinite(quality)):
            raise ValueError("scores must be a finite vector of length N")
    result = np.zeros((height, width), dtype=np.int32)
    best = np.full((height, width), -1.0)
    # Stable sort puts higher-quality masks first, preserving index order on ties.
    for index in np.argsort(-quality, kind="stable"):
        probability = probabilities[index]
        wins = (probability > threshold) & (probability > best)
        result[wins] = int(index) + 1
        best[wins] = probability[wins]
    return result


def instance_metrics(pred_instances: Any, gt_instances: Any) -> dict[str, float | int]:
    """Class-agnostic PQ at strict IoU > 0.5, plus count/detection metrics.

    Counts use unique positive IDs, not connected foreground components.
    Disjoint instance maps guarantee qualifying IoU > 0.5 matches are one-to-one,
    so no assignment solver is needed. Count error is the absolute difference.
    Both empty -> PQ/DQ/SQ/precision/recall=1. Nonempty GT and no predictions ->
    all five are 0. Empty GT with predictions -> PQ/DQ/SQ/precision=0, recall=1
    (no ground-truth objects were missed); false positives remain in fp/count_error.
    """
    pred, gt = _paired_labels(pred_instances, gt_instances)
    pred_ids, pred_inverse, pred_areas = np.unique(pred, return_inverse=True, return_counts=True)
    gt_ids, gt_inverse, gt_areas = np.unique(gt, return_inverse=True, return_counts=True)
    intersections = np.bincount(
        pred_inverse.ravel() * len(gt_ids) + gt_inverse.ravel(),
        minlength=len(pred_ids) * len(gt_ids),
    ).reshape(len(pred_ids), len(gt_ids))
    union = pred_areas[:, None] + gt_areas[None, :] - intersections
    ious = np.divide(intersections, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    ious = ious[np.ix_(pred_ids > 0, gt_ids > 0)]
    qualifying = ious > 0.5
    tp = int(qualifying.sum())
    pred_count, gt_count = int((pred_ids > 0).sum()), int((gt_ids > 0).sum())
    fp, fn = pred_count - tp, gt_count - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    both_empty = pred_count == gt_count == 0
    matched_iou = float(ious[qualifying].sum())
    return {
        "pq": matched_iou / denominator if denominator else 1.0,
        "dq": tp / denominator if denominator else 1.0,
        "sq": matched_iou / tp if tp else float(both_empty),
        "precision": tp / pred_count if pred_count else float(both_empty),
        "recall": tp / gt_count if gt_count else 1.0,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "count_error": abs(pred_count - gt_count),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_prediction(
    text: str, pred_instances: Any, gt_instances: Any, *, max_objects: int = 8,
    seg_weight: float = 1.0, format_weight: float = 1.0,
) -> dict[str, Any]:
    """Return flat training rewards and separate instance diagnostics.

    The caller obtains masks from the frozen segmenter using parsed prompts.
    Invalid completions are treated as empty predictions and receive zero format
    and segmentation rewards, including on empty GT. Empty valid objects likewise
    cannot claim a nonempty prediction. Default total is R_seg + R_fmt (range 0..2).
    Full GT masks are required for IoU: this function is not a weak-label reward.
    """
    for name, weight in (("seg_weight", seg_weight), ("format_weight", format_weight)):
        if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    pred, gt = _paired_labels(pred_instances, gt_instances)
    parsed = parse_completion(text, width=gt.shape[1], height=gt.shape[0], max_objects=max_objects)
    if not parsed.valid or not parsed.objects:
        pred = np.zeros_like(gt)
    fmt = float(parsed.valid)
    seg = foreground_iou(pred, gt) if parsed.valid else 0.0
    return {
        "format_valid": parsed.valid,
        "parse_error": parsed.error,
        "format_reward": fmt,
        "seg_reward": seg,
        "total_reward": format_weight * fmt + seg_weight * seg,
        **instance_metrics(pred, gt),
    }
