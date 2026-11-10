"""Coordinate tokens: K uniform bins over [0, 1], shared by x and y (NuNext Sec. 3.1).

A completion is a flat sequence of coordinate tokens x1 y1 x2 y2 ... in raster order
(top to bottom, then left to right), followed by the end-of-turn token. An image with
no nuclei is the end token alone. Anything else fails the format check.
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


@dataclass(frozen=True)
class Parsed:
    valid: bool
    bins: tuple[tuple[int, int], ...] = ()
    error: str | None = None


def parse_token_ids(ids: Sequence[int], first_id: int, bins: int, end_ids: Iterable[int]) -> Parsed:
    """Format check on generated token ids (the end token and anything after it excluded).

    Valid: every id before the first end token is a coordinate token and their count is even.
    A sequence that never reaches an end token (truncated) is invalid.
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
    if len(body) % 2:
        return Parsed(False, error="odd number of coordinate tokens")
    values = [token - first_id for token in body]
    return Parsed(True, tuple(zip(values[0::2], values[1::2])))


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


def bins_to_points(pairs: Iterable[tuple[int, int]], width: float, height: float, bins: int) -> list[tuple[float, float]]:
    return [(dequantize(x, width, bins), dequantize(y, height, bins)) for x, y in pairs]
