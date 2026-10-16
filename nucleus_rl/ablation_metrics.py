"""Inference-only diagnostics for paired prompt ablations; never training rewards.

Duplicate diagnostics consume the separate binary SAM masks BEFORE resolving
their overlaps into the disjoint instance map used by standard PQ/recall.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

import numpy as np


DUPLICATE_DEFINITIONS = {
    "same_gt_duplicate_count": (
        "Assign each nonempty raw mask to its best-IoU reference instance only "
        "when IoU > 0.5; sum max(number assigned - 1, 0) over reference IDs."
    ),
    "raw_overlap_pair_count": (
        "Count unordered pairs of nonempty raw masks with mutual IoU > 0.5. "
        "This is an overlap proxy, not proof that both represent the same nucleus."
    ),
}


def duplicate_metrics(raw_masks: Any, reference: Any) -> dict[str, Any]:
    """Measure redundant raw masks at a fixed strict IoU > 0.5 threshold.

    ``raw_masks`` must contain binary/bool values with shape (N,H,W); a bare []
    is accepted as zero masks of the reference size. Probabilities and logits
    are rejected: the caller must supply SAM's actual binary predictions, not
    choose a different binarization here. Empty masks never match or overlap.
    Reference zero is background; positive integer IDs identify instances even
    when disconnected. Unmatched masks have a null ``raw_best_gt_ids`` entry,
    while ``raw_best_gt_ious`` retains their best IoU (zero when GT is empty).
    These are many-to-one diagnostic assignments, NOT PQ's one-to-one matches.
    """
    gt = np.asarray(reference)
    if gt.ndim != 2 or min(gt.shape) == 0 or gt.dtype.kind not in "biuf":
        raise ValueError("reference must be a nonempty numeric H x W instance map")
    if not np.all(np.isfinite(gt)) or np.any(gt < 0) or np.any(gt != np.floor(gt)):
        raise ValueError("reference IDs must be finite nonnegative integers")
    raw = np.asarray(raw_masks)
    if raw.shape == (0,):
        raw = np.zeros((0, *gt.shape), dtype=bool)
    if raw.ndim != 3 or raw.shape[1:] != gt.shape:
        raise ValueError("raw_masks must have shape (N,H,W) matching reference")
    if raw.dtype.kind not in "biuf" or not np.all(np.isfinite(raw)):
        raise ValueError("raw_masks must contain finite binary values")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("raw_masks must be binary, not probabilities or logits")
    masks = raw.astype(bool, copy=False)
    areas = np.count_nonzero(masks, axis=(1, 2))
    gt_ids = np.unique(gt[gt > 0])
    best_ious = np.zeros(len(masks), dtype=float)
    best_ids: list[int | None] = [None] * len(masks)
    assignments: dict[int, int] = {}
    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        intersection = np.count_nonzero(masks & gt_mask[None], axis=(1, 2))
        union = areas + np.count_nonzero(gt_mask) - intersection
        ious = intersection / union  # A positive reference instance gives union > 0.
        for index in np.flatnonzero(ious > best_ious):
            best_ious[index] = float(ious[index])
            best_ids[index] = int(gt_id) if ious[index] > 0.5 else None
    for gt_id in best_ids:
        if gt_id is not None:
            assignments[gt_id] = assignments.get(gt_id, 0) + 1
    overlapping_pairs = []
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            intersection = int(np.count_nonzero(masks[left] & masks[right]))
            union = int(areas[left] + areas[right] - intersection)
            if union and intersection / union > 0.5:
                overlapping_pairs.append([left, right])
    return {
        "raw_mask_count": len(masks),
        "nonempty_raw_mask_count": int(np.count_nonzero(areas)),
        "same_gt_duplicate_count": sum(max(count - 1, 0) for count in assignments.values()),
        "raw_overlap_pair_count": len(overlapping_pairs),
        "raw_best_gt_ids": best_ids,
        "raw_best_gt_ious": best_ious.tolist(),
        "raw_overlap_pairs": overlapping_pairs,
    }


def _count(record: Mapping[str, Any], key: str) -> int:
    value = record[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return int(value)


def _fraction(record: Mapping[str, Any], key: str) -> float:
    value = record[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a finite number in [0,1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{key} must be a finite number in [0,1]")
    return value


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one flat final-instance metric record per crop/completion.

    Each record must include existing rewards.instance_metrics fields pq,
    recall, count_error, gt_count, pred_count, tp, fp, fn and format_valid, plus
    same_gt_duplicate_count and raw_overlap_pair_count from duplicate_metrics.
    Missing duplicate metrics are an error, never silently replaced with zero.
    Invalid completions remain in all denominators. Macro averages weight each
    record equally; micro recall is total TP / total GT. With records but no GT,
    micro recall is 1, matching existing recall convention; with no records,
    averages/rates are None and totals are zero. Repeated crops count repeatedly:
    callers should summarize each seed separately if a crop occurs per seed.
    """
    rows = list(records)
    totals = {key: 0 for key in ("pred_count", "gt_count", "tp", "fp", "fn",
                                "same_gt_duplicate_count", "raw_overlap_pair_count")}
    pq_sum = recall_sum = count_error_sum = 0.0
    valid_count = 0
    for row in rows:
        values = {key: _count(row, key) for key in totals}
        error = _count(row, "count_error")
        pq, recall = _fraction(row, "pq"), _fraction(row, "recall")
        if values["tp"] + values["fp"] != values["pred_count"] or values["tp"] + values["fn"] != values["gt_count"]:
            raise ValueError("Instance counts must satisfy TP+FP=pred_count and TP+FN=gt_count")
        if error != abs(values["pred_count"] - values["gt_count"]):
            raise ValueError("count_error must equal abs(pred_count-gt_count)")
        expected_recall = values["tp"] / values["gt_count"] if values["gt_count"] else 1.0
        if not math.isclose(recall, expected_recall, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("recall does not agree with TP and GT counts")
        if not isinstance(row["format_valid"], (bool, np.bool_)):
            raise ValueError("format_valid must be boolean")
        valid_count += int(row["format_valid"])
        pq_sum += pq
        recall_sum += recall
        count_error_sum += error
        for key in totals:
            totals[key] += values[key]
    n = len(rows)
    return {
        "n_records": n,
        "format_valid_count": valid_count,
        "format_valid_rate": valid_count / n if n else None,
        "macro_pq": pq_sum / n if n else None,
        "macro_recall": recall_sum / n if n else None,
        "micro_recall": (totals["tp"] / totals["gt_count"] if totals["gt_count"] else 1.0) if n else None,
        "count_mae": count_error_sum / n if n else None,
        **{"total_" + key: value for key, value in totals.items()},
    }
