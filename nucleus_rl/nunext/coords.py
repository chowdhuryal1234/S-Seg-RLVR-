"""Coordinate tokens: K uniform bins over [0, 1], shared by x and y (NuNext Sec. 3.1).

A completion is a flat sequence of coordinate tokens x1 y1 x2 y2 ... in raster order
(top to bottom, then left to right), followed by the end-of-turn token. An image with
no nuclei is the end token alone. Anything else fails the format check.

With boxes (``per_object=4``) every nucleus is x y w h: its centre, then the width and height
of its tight box, quantized on the same bins (w / image width, h / image height). The first
two tokens of each nucleus are exactly the point format, so detection scoring is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

TOKEN_TEMPLATE = "<|coord_{}|>"
_TOKEN_PATTERN = re.compile(r"<\|coord_(\d+)\|>")


def token_strings(bins: int) -> list[str]:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    return [TOKEN_TEMPLATE.format(index) for index in range(bins)]


def quantize(value: float, size: float, bins: int) -> int:
    """Pixel coordinate -> nearest bin index of the normalized value value/size."""
    if size <= 0:
        raise ValueError("size must be positive")
    normalized = min(max(float(value) / float(size), 0.0), 1.0)
    return min(int(round(normalized * (bins - 1))), bins - 1)


def dequantize(index: int, size: float, bins: int) -> float:
    """Bin index -> pixel coordinate (the bin's normalized value times size)."""
    if not 0 <= index < bins:
        raise ValueError("bin index out of range")
    return index / (bins - 1) * float(size)


def raster_order(points: Iterable[Sequence[float]], width: float, height: float, bins: int) -> list[tuple[int, int]]:
    """Quantize points and sort them by (y bin, x bin); duplicates are kept."""
    quantized = [(quantize(x, width, bins), quantize(y, height, bins)) for x, y in points]
    return sorted(quantized, key=lambda item: (item[1], item[0]))


def encode_points(points: Iterable[Sequence[float]], width: float, height: float, bins: int) -> str:
    """Raster-ordered coordinate-token text (without the end token)."""
    return "".join(TOKEN_TEMPLATE.format(x) + TOKEN_TEMPLATE.format(y)
                   for x, y in raster_order(points, width, height, bins))


def encode_objects(objects: Iterable[Sequence[float]], width: float, height: float, bins: int) -> str:
    """(x, y) or (x, y, w, h) per nucleus -> raster-ordered tokens; w, h quantized like x, y."""
    quantized = []
    for item in objects:
        values = [quantize(item[0], width, bins), quantize(item[1], height, bins)]
        if len(item) == 4:
            values += [quantize(item[2], width, bins), quantize(item[3], height, bins)]
        elif len(item) != 2:
            raise ValueError("objects must be (x, y) or (x, y, w, h)")
        quantized.append(values)
    quantized.sort(key=lambda v: (v[1], v[0]))
    return "".join(TOKEN_TEMPLATE.format(v) for values in quantized for v in values)


@dataclass(frozen=True)
class Parsed:
    valid: bool
    bins: tuple[tuple[int, ...], ...] = ()
    error: str | None = None


def _group(values, per_object):
    return tuple(tuple(values[i:i + per_object]) for i in range(0, len(values) - per_object + 1, per_object))


def parse_token_ids(ids: Sequence[int], first_id: int, bins: int, end_ids: Iterable[int],
                    per_object: int = 2) -> Parsed:
    """Format check on generated token ids (the end token and anything after it excluded).

    Valid: every id before the first end token is a coordinate token and their count is a multiple
    of ``per_object`` (2: x y; 4: x y w h). A sequence that never reaches an end token is invalid.
    """
    end_ids = set(end_ids)
    body = []
    for token in ids:
        if token in end_ids:
            break
        body.append(int(token))
    else:
        return Parsed(False, error="no end token (truncated)")
    if any(not first_id <= token < first_id + bins for token in body):
        return Parsed(False, error="non-coordinate token")
    if len(body) % per_object:
        return Parsed(False, error="odd number of coordinate tokens" if per_object == 2
                      else f"coordinate tokens not a multiple of {per_object}")
    return Parsed(True, _group([token - first_id for token in body], per_object))


def salvage_token_ids(ids: Sequence[int], first_id: int, bins: int, end_ids: Iterable[int],
                      per_object: int = 2) -> Parsed:
    """Lenient parse for diagnostics: recover the usable prefix of a malformed or truncated output.

    Takes coordinate tokens up to the first end or non-coordinate token, drops an unpaired last token,
    then keeps points while they stay in raster order ((y, x) non-decreasing, exact repeats dropped).
    A looping model restarts at a smaller y, so the loop is cut off. The strict format check
    (parse_token_ids) is still what training rewards and the headline metric use.
    """
    end_ids = set(end_ids)
    values = []
    for token in ids:
        if token in end_ids or not first_id <= token < first_id + bins:
            break
        values.append(int(token) - first_id)
    groups, previous = [], None
    for group in _group(values, per_object):
        x, y = group[0], group[1]
        if previous is not None and (y, x) <= previous:
            if (y, x) == previous:
                continue
            break
        groups.append(group)
        previous = (y, x)
    return Parsed(True, tuple(groups))


def parse_text(text: str, bins: int) -> Parsed:
    """Format check on decoded text (special tokens kept, end token already stripped)."""
    stripped = text.strip()
    matches = _TOKEN_PATTERN.findall(stripped)
    if _TOKEN_PATTERN.sub("", stripped).strip():
        return Parsed(False, error="non-coordinate text")
    values = [int(value) for value in matches]
    if any(value >= bins for value in values):
        return Parsed(False, error="bin index out of range")
    if len(values) % 2:
        return Parsed(False, error="odd number of coordinate tokens")
    return Parsed(True, tuple(zip(values[0::2], values[1::2])))


def bins_to_points(groups: Iterable[tuple[int, ...]], width: float, height: float, bins: int) -> list[tuple[float, float]]:
    """Centre of each nucleus (the first two bins of each group)."""
    return [(dequantize(g[0], width, bins), dequantize(g[1], height, bins)) for g in groups]


def bins_to_boxes(groups: Iterable[tuple[int, ...]], width: float, height: float, bins: int):
    """(x0, y0, x1, y1) per x y w h group, clipped to the image; groups of 2 have no box."""
    boxes = []
    for g in groups:
        if len(g) != 4:
            raise ValueError("boxes need x y w h groups")
        x, y = dequantize(g[0], width, bins), dequantize(g[1], height, bins)
        w, h = dequantize(g[2], width, bins), dequantize(g[3], height, bins)
        boxes.append((max(0.0, x - w / 2), max(0.0, y - h / 2), min(float(width), x + w / 2), min(float(height), y + h / 2)))
    return boxes
