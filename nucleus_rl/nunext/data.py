"""PanNuke detection rows, D4 augmentation and the instruction/target text."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from nucleus_rl.nunext.coords import encode_points

# Instruction quoted from the paper (Sec. 3.2).
INSTRUCTION = "Point to nucleus. Output the coordinates in [x, y] format."


def read_manifests(paths: Sequence[str | Path], limit: int | None = None) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No rows in the given manifests")
    for row in rows:
        if not {"id", "image_path", "width", "height", "centroids"} <= set(row):
            raise ValueError(f"Manifest row {row.get('id')} lacks required keys")
    return rows


def d4_transform(pixels: np.ndarray, points: Sequence[Sequence[float]], view: int):
    """One of the eight square symmetries: optional horizontal flip, then view % 4 quarter turns.

    Points are continuous pixel coordinates in [0, size], so pixel centres map to pixel centres.
    """
    if not 0 <= view < 8:
        raise ValueError("view must be in 0..7")
    height, width = pixels.shape[:2]
    if height != width:
        raise ValueError("D4 augmentation needs a square image")
    size = float(width)
    array = np.asarray(pixels)
    coords = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if view >= 4:
        array = array[:, ::-1]
        coords[:, 0] = size - coords[:, 0]
    for _ in range(view % 4):
        # np.rot90 turns counter-clockwise: old (x, y) -> new (y, size - x).
        array = np.rot90(array)
        coords = np.stack([coords[:, 1], size - coords[:, 0]], axis=1)
    return np.ascontiguousarray(array), [tuple(point) for point in coords]


def load_example(row: dict, bins: int, view: int = 0, image_size: int | None = None) -> dict:
    """Image (PIL, optionally resized to image_size square) and its target text.

    Coordinates are normalized by the original width/height before quantization, so
    resizing the image for the vision encoder does not change the target tokens.
    """
    from PIL import Image

    pixels = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    pixels, points = d4_transform(pixels, row["centroids"], view)
    image = Image.fromarray(pixels)
    if image_size is not None and image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BICUBIC)
    return {"id": row["id"], "image": image, "points": points,
            "width": row["width"], "height": row["height"],
            "target": encode_points(points, row["width"], row["height"], bins)}


def epoch_order(row_count: int, epoch: int, seed: int) -> list[int]:
    order = list(range(row_count))
    random.Random(seed * 100003 + epoch).shuffle(order)
    return order
