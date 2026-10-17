"""CPU-only box/point localization diagnostics, never mask PQ or rewards.

Boxes use continuous half-open XYXY coordinates. GT boxes enclose each positive
instance ID, even if that ID is disconnected. Display IDs follow sorted GT IDs;
prediction IDs follow completion order. This module never repairs predictions.
The runner, not these metrics, selects the fully interior 2–4-nucleus crops.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np


BOX_IOU_THRESHOLD = 0.5


def _labels(mask):
    labels = np.asarray(mask)
    if labels.ndim != 2 or min(labels.shape) == 0 or labels.dtype.kind not in "biuf":
        raise ValueError("mask must be a nonempty numeric H x W instance map")
    if not np.all(np.isfinite(labels)) or np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("Instance IDs must be finite nonnegative integers")
    return labels


def _size(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0 for v in value)):
        raise ValueError("processed_size must be positive integer (width,height)")
    return int(value[0]), int(value[1])


def geometry(mask, *, processed_size=(224, 224)):
    """Return GT geometry in original and processed pixels as JSON-ready data.

    Centroids are arithmetic means of foreground pixel indices (x,y), not box
    centers or guaranteed interior points. Scaling is width/width, height/height.
    """
    labels = _labels(mask)
    height, width = labels.shape
    processed_width, processed_height = _size(processed_size)
    sx, sy = processed_width / width, processed_height / height
    instances = []
    for index, instance_id in enumerate(np.unique(labels[labels > 0]), 1):
        ys, xs = np.nonzero(labels == instance_id)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        centroid = [float(xs.mean()), float(ys.mean())]
        instances.append({"display_id": f"G{index}", "instance_id": int(instance_id),
                          "original_box": box,
                          "processed_box": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
                          "centroid_original": centroid,
                          "centroid_processed": [centroid[0] * sx, centroid[1] * sy],
                          "area": int(len(xs))})
    return {"original_size": [width, height], "processed_size": [processed_width, processed_height],
            "gt_count": len(instances), "instances": instances}


def _coords(value, count):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
        raise ValueError(f"Expected {count} coordinates")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) for v in value):
        raise ValueError("Coordinates must be finite real numbers, never booleans")
    return [float(v) for v in value]


def _object(obj, size):
    box = _coords(obj.get("box") if isinstance(obj, Mapping) else getattr(obj, "box", None), 4)
    point = obj.get("point") if isinstance(obj, Mapping) else getattr(obj, "point", None)
    width, height = size
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Box must have positive area and remain within the declared frame")
    if point is not None:
        point = _coords(point, 2)
        if not (0 <= point[0] < width and 0 <= point[1] < height
                and box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]):
            raise ValueError("Point must be inside its box and image")
    return box, point


def _box_iou(left, right):
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union


def _match_box_iou(matrix):
    """Exact small-set assignment: maximize cardinality, then total box IoU.

    A DP over used-GT bitsets avoids greedy assignment errors when boxes overlap.
    Strict IoU > 0.5 is fixed. Exact objective ties prefer earlier (P,G) pairs.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or max(matrix.shape) > 8 or not np.all(np.isfinite(matrix))
            or np.any((matrix < 0) | (matrix > 1))):
        raise ValueError("Box matching requires an at-most-8 by 8 finite IoU matrix")
    states = {0: (0., ())}
    for pred in range(matrix.shape[0]):
        updated = dict(states)  # Unmatched prediction remains a permissible choice.
        for used, (score, pairs) in states.items():
            for gt in np.flatnonzero(matrix[pred] > BOX_IOU_THRESHOLD):
                gt = int(gt)
                if used & (1 << gt):
                    continue
                target = used | (1 << gt)
                candidate = score + float(matrix[pred, gt]), pairs + ((pred, gt),)
                prior = updated.get(target)
                if prior is None or candidate[0] > prior[0] or (candidate[0] == prior[0] and candidate[1] < prior[1]):
                    updated[target] = candidate
        states = updated
    _, best = min(states.items(), key=lambda item: (-item[0].bit_count(), -item[1][0], item[1][1]))
    return list(best[1])


def _duplicate_groups(predictions, assignment_key, *, box_threshold=False):
    groups = {}
    for pred in predictions:
        gt_id = pred[assignment_key]
        if gt_id is not None and (not box_threshold or pred["best_box_iou"] > BOX_IOU_THRESHOLD):
            groups.setdefault(gt_id, []).append(pred["display_id"])
    return [{"gt_id": gt_id, "prediction_ids": ids, "extra_count": len(ids) - 1}
            for gt_id, ids in groups.items() if len(ids) > 1]


def analyze(raw_objects, original_objects, mask, *, processed_size=(224, 224)):
    """Analyze paired raw/converted prompts against GT, without SAM inference.

    Inputs accept ObjectPrompt instances or mappings with a box and optional
    point. Raw coordinates must use processed pixels; original coordinates must
    be their exact fixed-scale conversion. Invalid completions should be passed
    as two empty lists; the runner must separately retain format validity.
    """
    labels = _labels(mask)
    gt_geometry = geometry(labels, processed_size=processed_size)
    processed_size = tuple(gt_geometry["processed_size"])
    original_size = tuple(gt_geometry["original_size"])
    raw_objects, original_objects = list(raw_objects), list(original_objects)
    if len(raw_objects) != len(original_objects):
        raise ValueError("Raw and original prompt lists must have equal length")
    if len(raw_objects) > 8 or gt_geometry["gt_count"] > 8:
        raise ValueError("This small-crop diagnostic supports at most eight predictions and eight GT IDs")
    gt = gt_geometry["instances"]
    by_instance_id = {item["instance_id"]: item["display_id"] for item in gt}
    scale = original_size[0] / processed_size[0], original_size[1] / processed_size[1]
    predictions, original_boxes = [], []
    for index, (raw, converted) in enumerate(zip(raw_objects, original_objects), 1):
        raw_box, raw_point = _object(raw, processed_size)
        box, point = _object(converted, original_size)
        if (raw_point is None) != (point is None):
            raise ValueError("Raw and converted prompts must agree on point presence")
        expected_box = [raw_box[i] * scale[i % 2] for i in range(4)]
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(expected_box, box)):
            raise ValueError("Original box differs from the declared fixed-scale conversion")
        if point is not None and not all(math.isclose(raw_point[i] * scale[i], point[i], rel_tol=1e-12, abs_tol=1e-12) for i in range(2)):
            raise ValueError("Original point differs from the declared fixed-scale conversion")
        floor_xy = [math.floor(v) for v in point] if point is not None else None
        point_instance = int(labels[floor_xy[1], floor_xy[0]]) if floor_xy is not None else 0
        predictions.append({"display_id": f"P{index}", "raw_box": raw_box, "original_box": box,
                            "raw_point": raw_point, "original_point": point, "point_floor_xy": floor_xy,
                            "point_gt_id": by_instance_id.get(point_instance),
                            "point_gt_instance_id": point_instance if point_instance else None,
                            "point_status": "not_provided" if point is None else "foreground" if point_instance else "background"})
        original_boxes.append(box)
    matrix = np.array([[_box_iou(box, item["original_box"]) for item in gt]
                       for box in original_boxes], dtype=float).reshape(len(predictions), len(gt))
    unique_best = []
    for index, pred in enumerate(predictions):
        best = int(np.argmax(matrix[index])) if len(gt) and np.any(matrix[index] > 0) else None
        best_iou = float(matrix[index, best]) if best is not None else 0.
        edges = [a - b for a, b in zip(pred["raw_box"], gt[best]["processed_box"])] if best is not None else None
        tied = best is not None and sum(math.isclose(float(value), best_iou, rel_tol=1e-12, abs_tol=1e-12) for value in matrix[index]) > 1
        pred.update({"best_box_gt_id": gt[best]["display_id"] if best is not None else None,
                     "best_box_gt_instance_id": gt[best]["instance_id"] if best is not None else None,
                     "best_box_iou": best_iou, "best_box_iou_tied": bool(tied),
                     "center_offset_processed_xy": [(edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2] if edges is not None else None,
                     "edge_errors_processed_xyxy": edges})
        if best is not None and not tied:
            unique_best.append(pred)
    pairs = _match_box_iou(matrix)
    matches = [{"prediction_id": f"P{p + 1}", "gt_id": gt[g]["display_id"], "box_iou": float(matrix[p, g])} for p, g in pairs]
    matched_pred, matched_gt = {p for p, _ in pairs}, {g for _, g in pairs}
    box_duplicates = _duplicate_groups(predictions, "best_box_gt_id", box_threshold=True)
    point_duplicates = _duplicate_groups(predictions, "point_gt_id")
    overlap_pairs = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            iou = _box_iou(original_boxes[left], original_boxes[right])
            if iou > BOX_IOU_THRESHOLD:
                overlap_pairs.append({"prediction_ids": [f"P{left + 1}", f"P{right + 1}"], "box_iou": iou})
    # Collisions are excluded rather than resolved using a favorable assignment.
    associations_per_gt = Counter(pred["best_box_gt_id"] for pred in unique_best)
    associations = [{"prediction_id": p["display_id"], "gt_id": p["best_box_gt_id"],
                     "box_iou": p["best_box_iou"], "center_offset_processed_xy": p["center_offset_processed_xy"]}
                    for p in unique_best if associations_per_gt[p["best_box_gt_id"]] == 1]
    supported = len(associations) >= 2
    offsets = np.array([a["center_offset_processed_xy"] for a in associations], dtype=float).reshape(-1, 2)
    mean = offsets.mean(axis=0) if supported else None
    residual = offsets - mean if supported else None
    translation = {"status": "supported_associations" if supported else "insufficient_associations",
                   "association_rule": "Unique positive-IoU best GT box per prediction; exclude tied bests and all GT collisions; require at least two associations.",
                   "diagnostic_associations_not_true_positive_matches": True,
                   "does_not_establish_global_shift": True,
                   "offset_definition": "Predicted box center minus GT bounding-box center, in processed pixels.",
                   "association_count": len(associations), "associations": associations,
                   "center_offsets_processed_xy": offsets.tolist(),
                   "mean_offset_processed_xy": mean.tolist() if supported else None,
                   "rms_residual_processed": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))) if supported else None,
                   "rms_residual_processed_xy": np.sqrt(np.mean(residual ** 2, axis=0)).tolist() if supported else None}
    return {"metric_scope": "Box/point localization only; no mask PQ or segmentation reward.",
            "box_match_rule": "One-to-one box IoU > 0.5; maximize match count, then summed IoU; exact ties use P/G order.",
            "gt_count": len(gt), "prediction_count": len(predictions),
            "absolute_count_error": abs(len(predictions) - len(gt)), "predictions": predictions,
            "box_iou_matrix": matrix.tolist(),
            "box_iou_matrix_prediction_ids": [p["display_id"] for p in predictions],
            "box_iou_matrix_gt_ids": [item["display_id"] for item in gt],
            "box_matches": matches, "box_true_positive_count": len(matches),
            "matched_gt_ids": [gt[g]["display_id"] for g in range(len(gt)) if g in matched_gt],
            "missed_gt_ids": [gt[g]["display_id"] for g in range(len(gt)) if g not in matched_gt],
            "unmatched_prediction_ids": [f"P{p + 1}" for p in range(len(predictions)) if p not in matched_pred],
            "box_recall": len(matches) / len(gt) if gt else 1.,
            "best_box_duplicate_count": sum(group["extra_count"] for group in box_duplicates),
            "best_box_duplicate_groups": box_duplicates,
            "point_duplicate_count": sum(group["extra_count"] for group in point_duplicates),
            "point_duplicate_groups": point_duplicates,
            "predicted_box_overlap_pair_count": len(overlap_pairs),
            "predicted_box_overlap_pairs": overlap_pairs, "translation_diagnostic": translation}
