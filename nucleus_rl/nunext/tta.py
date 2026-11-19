"""D4 test-time augmentation: decode each of the eight square views, map back, merge by voting.

The detector is run on every view of an image (data.d4_transform). Each view's nuclei are mapped
back to the original frame and matched across views; a merged nucleus keeps the median centre (and
median box size), and ``votes`` = the number of views that found it. Keeping nuclei with at least
``min_votes`` votes trades recall (low threshold) against precision (high threshold); the threshold
is chosen from a sweep, so every merged nucleus is stored with its vote count.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

VIEWS = tuple(range(8))


def inverse_points(points: Sequence[Sequence[float]], view: int, size: float) -> list[tuple[float, float]]:
    """Undo data.d4_transform for points: view-frame pixels -> original-frame pixels."""
    if not 0 <= view < 8:
        raise ValueError("view must be in 0..7")
    coords = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    for _ in range(view % 4):
        # Inverse of one counter-clockwise quarter turn (x, y) -> (y, size - x).
        coords = np.stack([size - coords[:, 1], coords[:, 0]], axis=1)
    if view >= 4:
        coords[:, 0] = size - coords[:, 0]
    return [tuple(point) for point in coords]


def inverse_objects(objects: Sequence[Sequence[float]], view: int, size: float) -> list[tuple[float, ...]]:
    """(x, y) or (x, y, w, h) per nucleus back to the original frame; w and h swap on odd quarter turns."""
    centres = inverse_points([o[:2] for o in objects], view, size)
    result = []
    for obj, (x, y) in zip(objects, centres):
        if len(obj) == 4:
            w, h = (obj[3], obj[2]) if view % 2 else (obj[2], obj[3])
            result.append((x, y, float(w), float(h)))
        else:
            result.append((x, y))
    return result


def merge_views(views: Sequence[Sequence[Sequence[float]]], radius: float) -> list[dict]:
    """Cluster per-view nuclei (already in the original frame) into merged nuclei with votes.

    Views are added one at a time; each nucleus joins the nearest cluster centre within ``radius``
    through a one-to-one assignment (a view adds at most one member to a cluster), otherwise it starts
    a new cluster, unless it is within the radius of a cluster its own view just joined (a duplicate
    inside one view, dropped). Returns [{"point": (x, y), "size": (w, h) or None, "votes": int}] in raster order.
    """
    from scipy.optimize import linear_sum_assignment

    if radius <= 0:
        raise ValueError("merge radius must be positive")
    clusters: list[list[tuple[float, ...]]] = []
    for objects in views:
        objects = [tuple(float(v) for v in o) for o in objects]
        if not objects:
            continue
        unmatched = list(range(len(objects)))
        if clusters:
            centres = np.array([np.median([m[:2] for m in c], axis=0) for c in clusters])
            points = np.array([o[:2] for o in objects])
            distances = np.linalg.norm(points[:, None] - centres[None], axis=-1)
            rows, cols = linear_sum_assignment(np.where(distances <= radius, distances, 1e9))
            matched, joined = set(), set()
            for r, c in zip(rows, cols):
                if distances[r, c] <= radius:
                    clusters[c].append(objects[r])
                    matched.add(r)
                    joined.add(c)
            # A leftover point within the radius of a cluster this view already joined is a duplicate
            # inside one view; as its own cluster it would split the votes of later views.
            unmatched = [i for i in unmatched if i not in matched
                         and not any(distances[i, c] <= radius for c in joined)]
        clusters.extend([objects[i]] for i in unmatched)
    merged = []
    for members in clusters:
        array = np.array(members)
        x, y = np.median(array[:, 0]), np.median(array[:, 1])
        size = (float(np.median(array[:, 2])), float(np.median(array[:, 3]))) if array.shape[1] == 4 else None
        merged.append({"point": (float(x), float(y)), "size": size, "votes": len(members)})
    merged.sort(key=lambda m: (m["point"][1], m["point"][0]))
    return merged


def select(merged: Sequence[dict], min_votes: int, width: float, height: float):
    """Points and boxes (x0, y0, x1, y1, clipped; None without sizes) of nuclei with >= min_votes votes."""
    kept = [m for m in merged if m["votes"] >= min_votes]
    points = [m["point"] for m in kept]
    if any(m["size"] is None for m in kept):
        return points, None
    boxes = [(max(0.0, x - w / 2), max(0.0, y - h / 2), min(float(width), x + w / 2), min(float(height), y + h / 2))
             for (x, y), (w, h) in ((m["point"], m["size"]) for m in kept)]
    return points, boxes
