"""Hungarian matching of predicted and reference centroids (NuNext Eqs. 13-17).

The F1 score from this matching is both the evaluation metric and the distribution
matching reward. Per-prediction TP/FP labels feed fine-grained advantage shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Match:
    tp: int
    fp: int
    fn: int
    pred_is_tp: tuple[bool, ...] = field(default_factory=tuple)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float(self.fn == 0)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float(self.fp == 0)

    @property
    def f1(self) -> float:
        """2TP / (2TP + FP + FN); 1 when both sets are empty."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 1.0


def match_points(pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]], radius: float) -> Match:
    """Minimum-cost one-to-one assignment, then accept pairs with distance <= radius.

    This follows the paper: solve the assignment on raw distances first, then threshold.
    """
    from scipy.optimize import linear_sum_assignment

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    pred_array = np.asarray(pred, dtype=float).reshape(-1, 2)
    gt_array = np.asarray(gt, dtype=float).reshape(-1, 2)
    if len(pred_array) == 0 or len(gt_array) == 0:
        return Match(0, len(pred_array), len(gt_array), (False,) * len(pred_array))
    distances = np.linalg.norm(pred_array[:, None, :] - gt_array[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    accepted = distances[rows, cols] <= radius
    is_tp = np.zeros(len(pred_array), dtype=bool)
    is_tp[rows[accepted]] = True
    tp = int(accepted.sum())
    return Match(tp, len(pred_array) - tp, len(gt_array) - tp, tuple(bool(v) for v in is_tp))


def aggregate(matches: Sequence[Match]) -> dict[str, float]:
    """Dataset-level (micro) and mean per-image detection scores."""
    tp = sum(m.tp for m in matches)
    fp = sum(m.fp for m in matches)
    fn = sum(m.fn for m in matches)
    micro = Match(tp, fp, fn)
    return {
        "images": len(matches), "tp": tp, "fp": fp, "fn": fn,
        "micro_f1": micro.f1, "micro_precision": micro.precision, "micro_recall": micro.recall,
        "mean_image_f1": float(np.mean([m.f1 for m in matches])) if matches else float("nan"),
    }
