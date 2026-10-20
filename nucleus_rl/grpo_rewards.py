"""Reward components and group advantages for GRPO from an SFT policy.

Every component lies in [0, 1] and is zero for an unparsable completion. The
training reward is a weighted sum chosen on the command line, e.g.
``format=1,pq=1``. Instance maps and raw masks are in original crop pixels.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nucleus_rl.ablation_metrics import duplicate_metrics
from nucleus_rl.rewards import foreground_iou, instance_metrics

COMPONENTS = ("format", "fg_iou", "pq", "count", "separation")


def parse_reward_spec(spec: str) -> dict[str, float]:
    """Parse ``name=weight`` pairs; unknown names, duplicates and all-zero specs fail."""
    weights: dict[str, float] = {}
    for item in (part.strip() for part in str(spec).split(",")):
        if not item:
            continue
        name, separator, value = item.partition("=")
        name = name.strip()
        if not separator or name not in COMPONENTS:
            raise ValueError(f"Reward items must be name=weight with name in {COMPONENTS}: {item!r}")
        if name in weights:
            raise ValueError(f"Duplicate reward component: {name}")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward weight must be finite and nonnegative: {item!r}")
        weights[name] = weight
    if not weights or not any(weights.values()):
        raise ValueError("At least one reward component needs a positive weight")
    return weights


def merge_extras(raw_masks: Any, reference: Any, coverage: float = 0.5) -> int:
    """Count extra nuclei swallowed by single masks.

    A raw mask "covers" a reference instance when it contains at least
    ``coverage`` of that instance's pixels; a mask covering k instances adds k-1.
    """
    gt = np.asarray(reference)
    masks = np.asarray(raw_masks, dtype=bool)
    if masks.size == 0:
        return 0
    if masks.ndim != 3 or masks.shape[1:] != gt.shape:
        raise ValueError("raw_masks must have shape (N,H,W) matching reference")
    extras = 0
    gt_ids = np.unique(gt[gt > 0])
    areas = {int(gt_id): int(np.count_nonzero(gt == gt_id)) for gt_id in gt_ids}
    for mask in masks:
        if not mask.any():
            continue
        inside, counts = np.unique(gt[mask & (gt > 0)], return_counts=True)
        covered = sum(count >= coverage * areas[int(gt_id)] for gt_id, count in zip(inside, counts))
        extras += max(int(covered) - 1, 0)
    return extras


def reward_components(valid: bool, object_count: int, instances: Any, raw_masks: Any,
                      reference: Any) -> dict[str, float]:
    """All components for one completion; invalid or empty answers earn nothing."""
    if not valid or object_count == 0:
        return {name: (1.0 if name == "format" and valid else 0.0) for name in COMPONENTS}
    metrics = instance_metrics(instances, reference)
    duplicates = duplicate_metrics(raw_masks, reference)["same_gt_duplicate_count"]
    mistakes = duplicates + merge_extras(raw_masks, reference)
    raw_count = max(len(np.asarray(raw_masks)), 1)
    gt_count = metrics["gt_count"]
    return {
        "format": 1.0,
        "fg_iou": float(foreground_iou(instances, reference)),
        "pq": float(metrics["pq"]),
        "count": 1.0 - min(abs(metrics["pred_count"] - gt_count) / max(gt_count, 1), 1.0),
        "separation": max(0.0, 1.0 - mistakes / raw_count),
    }


def total_reward(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(weight * components[name] for name, weight in weights.items()))


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    """GRPO advantages: reward minus group mean, divided by population std."""
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Group advantages need at least two finite rewards")
    spread = float(values.std())
    if spread < 1e-8:
        return [0.0] * len(values)
    return ((values - values.mean()) / (spread + epsilon)).tolist()


def completion_lengths(token_rows: Iterable[Sequence[int]], eos_ids: Iterable[int]) -> list[tuple[int, bool]]:
    """Tokens to train on per sampled row, through the first end token.

    Returns (length, finished). Rows without an end token keep every sampled
    token and are reported as unfinished (truncated at the token limit).
    """
    stops = set(int(token) for token in eos_ids)
    result = []
    for row in token_rows:
        row = list(row)
        end = next((index for index, token in enumerate(row) if int(token) in stops), None)
        result.append((end + 1, True) if end is not None else (len(row), False))
    return result
