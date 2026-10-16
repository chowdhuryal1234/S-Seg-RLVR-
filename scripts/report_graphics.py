"""Small CPU-only graphics helpers with no dependency on experiment renderers."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = ((0, 220, 255), (255, 185, 0), (255, 75, 150), (90, 245, 90),
           (185, 130, 255), (255, 110, 50), (80, 150, 255), (240, 235, 100))


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _fit_text(draw, text, font, max_width):
    text = str(text).replace("\n", " ")
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text + "..."


def instance_panel(image, labels, size=256):
    """Show saved instance IDs, light tint, and four-neighbor inner boundaries."""
    pixels = np.array(image, dtype=float)
    ids = np.unique(labels[labels > 0])
    centers = []
    for index, instance_id in enumerate(ids):
        mask = labels == instance_id
        color = np.array(PALETTE[index % len(PALETTE)])
        pixels[mask] = pixels[mask] * 0.85 + color * 0.15
        padded = np.pad(mask, 1)
        interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                    & padded[1:-1, :-2] & padded[1:-1, 2:])
        pixels[mask & ~interior] = color
        points = np.argwhere(mask)
        center = points[np.argmin(((points - points.mean(axis=0)) ** 2).sum(axis=1))]
        centers.append((int(instance_id), center))
    overlay = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    scale = min(size / image.width, size / image.height)
    dimensions = (round(image.width * scale), round(image.height * scale))
    overlay = overlay.resize(dimensions, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size, size), "#eef1f5")
    offset = ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2)
    panel.paste(overlay, offset)
    draw = ImageDraw.Draw(panel)
    for instance_id, (y, x) in centers:
        draw.text((offset[0] + (x + 0.5) * scale, offset[1] + (y + 0.5) * scale),
                  str(instance_id), anchor="mm", font=_font(15), fill="white",
                  stroke_width=2, stroke_fill="#182235")
    return panel
