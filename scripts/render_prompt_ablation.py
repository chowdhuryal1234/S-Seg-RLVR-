"""Render locally retrieved paired prompt-ablation outputs without loading models.

Example:
    .venv/bin/python scripts/render_prompt_ablation.py --run runs/ablation_a \
        --dataset data/monuseg_iris_pilot_v1 --output reports/ablation_a

Remote absolute paths are rebased explicitly. Original PNG hashes and final-map
metrics are checked before writing anything. Cases retain the preselected order.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.ablation_metrics import DUPLICATE_DEFINITIONS, summarize_records
from nucleus_rl.rewards import instance_metrics


VARIANTS = ("one_example", "three_examples")
TITLES = ("Ground truth", "One-object example", "Three-object example")
PALETTE = ((0, 220, 255), (255, 185, 0), (255, 75, 150), (90, 245, 90),
           (185, 130, 255), (255, 110, 50), (80, 150, 255), (240, 235, 100))
INK, MUTED, RED = "#182235", "#536279", "#b52135"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Retrieved directory containing result.json and paired_records.jsonl")
    parser.add_argument("--dataset", type=Path, required=True, help="Local dataset root containing images/ and masks/")
    parser.add_argument("--output", type=Path, required=True, help="New empty report directory")
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_number(actual, expected):
    if isinstance(actual, (float, int)) and not isinstance(actual, bool):
        return (isinstance(expected, (float, int)) and not isinstance(expected, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12))
    return actual == expected


def load_verified_cases(run, dataset):
    """Return fixed-order cases; refuse missing, mismatched, or ambiguous inputs."""
    result = json.loads((run / "result.json").read_text())
    if result.get("status") != "completed_paired_inference":
        raise ValueError("This renderer requires a completed paired inference run")
    if result.get("optimizer_steps") != 0 or result.get("policy_trainable_parameters") != 0:
        raise ValueError("This report is only for the inference-only experiment")
    if result.get("all_paired_pixels_identical") is not True:
        raise ValueError("The run must confirm identical preprocessed images for paired arms")
    selected = result["selected_crops"]
    ids = [item["id"] for item in selected]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("selected_crops must be nonempty and have unique IDs")
    if any(not isinstance(item, str) or Path(item).name != item or item in (".", "..") for item in ids):
        raise ValueError("Crop IDs must be plain filenames without a directory")
    records = {}
    for line in (run / "paired_records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["id"], record["variant"])
        if key in records or key[0] not in ids or key[1] not in VARIANTS:
            raise ValueError(f"Duplicate or unexpected paired record: {key}")
        records[key] = record
    expected = {(crop_id, variant) for crop_id in ids for variant in VARIANTS}
    if set(records) != expected:
        raise ValueError(f"Missing paired records: {sorted(expected - set(records))}")
    cases, used_predictions = [], set()
    for item in selected:
        crop_id = item["id"]
        image_path, mask_path = dataset / "images" / f"{crop_id}.png", dataset / "masks" / f"{crop_id}.png"
        for path, key in ((image_path, "image_sha256"), (mask_path, "mask_sha256")):
            if _sha256(path) != item[key]:
                raise ValueError(f"SHA-256 mismatch for {path}; refusing to display a different crop")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            reference = np.array(source)
        if reference.shape != (image.height, image.width):
            raise ValueError(f"Image and reference dimensions differ for {crop_id}")
        if int(len(np.unique(reference[reference > 0]))) != item["gt_count"]:
            raise ValueError(f"Selected GT count differs from the actual mask for {crop_id}")
        arms = {}
        for variant in VARIANTS:
            record = records[crop_id, variant]
            prediction_path = run / variant / Path(record["prediction_path"]).name
            if prediction_path.suffix != ".npy" or prediction_path in used_predictions:
                raise ValueError(f"Invalid or reused final prediction path: {prediction_path}")
            used_predictions.add(prediction_path)
            prediction = np.load(prediction_path, allow_pickle=False)
            metrics = instance_metrics(prediction, reference)
            for key, value in metrics.items():
                if not _same_number(value, record[key]):
                    raise ValueError(f"Final mask metric mismatch: {crop_id}/{variant}/{key}")
            if not record["format_valid"] and np.any(prediction):
                raise ValueError(f"Invalid completion unexpectedly has a nonempty final mask: {crop_id}/{variant}")
            arms[variant] = {"record": record, "prediction": prediction}
        cases.append({"selection": item, "image": image, "reference": reference, "arms": arms})
    summaries = {}
    for variant in VARIANTS:
        summary = summarize_records(records[crop_id, variant] for crop_id in ids)
        summary["token_cap_hits"] = sum(bool(records[crop_id, variant]["hit_token_cap"]) for crop_id in ids)
        for key, value in summary.items():
            if not _same_number(value, result["variants"][variant][key]):
                raise ValueError(f"Summary mismatch: {variant}/{key}")
        summaries[variant] = summary
    return result, cases, summaries


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
    """Light tint, four-neighbor inner boundaries, and panel-local instance IDs."""
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
    # Preserve aspect ratio, although the current experiment uses 128 x 128 crops.
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


def render_sheet(cases, first_index, total, destination):
    width, margin, column, row_height, top = 984, 24, 312, 400, 128
    canvas = Image.new("RGB", (width, top + row_height * len(cases) + 64), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Prompt-format comparison | fixed training crops", font=_font(26), fill=INK)
    draw.text((margin, 54), f"Cases {first_index + 1}-{first_index + len(cases)} of {total} | Base Qwen 7B + frozen SAM2 | No training", font=_font(17), fill=MUTED)
    for index, title in enumerate(TITLES):
        draw.text((margin + index * column + column / 2, 100), title, anchor="mm", font=_font(20), fill=INK)
    for index, case in enumerate(cases):
        y = top + index * row_height
        crop_id = case["selection"]["id"]
        draw.line((margin, y, width - margin, y), fill="#dce2ea", width=1)
        draw.text((margin, y + 9), f"{first_index + index + 1:02d}  {crop_id}", font=_font(16), fill=INK)
        maps = [case["reference"]] + [case["arms"][v]["prediction"] for v in VARIANTS]
        for col, labels in enumerate(maps):
            x = margin + col * column + (column - 256) // 2
            canvas.paste(instance_panel(case["image"], labels), (x, y + 38))
            if col == 0:
                lines = [(f"Reference nuclei: {case['selection']['gt_count']}", INK),
                         (f"Original: {case['image'].width} x {case['image'].height} pixels", MUTED),
                         ("Colored outlines + local instance IDs", MUTED)]
            else:
                record = case["arms"][VARIANTS[col - 1]]["record"]
                lines = [(f"PQ {record['pq']:.3f}  |  Recall {record['recall']:.3f}", INK),
                         (f"Pred / GT: {record['pred_count']} / {record['gt_count']}  |  Duplicates: {record['same_gt_duplicate_count']}", INK),
                         (f"Raw overlap pairs: {record['raw_overlap_pair_count']}", MUTED)]
                if not record["format_valid"]:
                    lines.append(("INVALID: " + str(record.get("parse_error") or "format error"), RED))
                    draw.rectangle((x - 2, y + 36, x + 257, y + 295), outline=RED, width=2)
                else:
                    lines.append(("Format valid" + (" | token cap reached" if record["hit_token_cap"] else ""), MUTED))
            for line, (text, color) in enumerate(lines):
                draw.text((margin + col * column + 8, y + 304 + line * 22),
                          _fit_text(draw, text, _font(16), column - 16), font=_font(16), fill=color)
    footer = top + row_height * len(cases) + 8
    draw.text((margin, footer), "Colors and IDs distinguish instances within each panel; they do not match IDs across panels.", font=_font(15), fill=MUTED)
    draw.text((margin, footer + 22), "Duplicates are measured before overlap resolution. These selected training crops are not a held-out benchmark.", font=_font(15), fill=MUTED)
    canvas.save(destination)


def render_html(result, cases, summaries, sheets, destination):
    escape = lambda value: html.escape(str(value), quote=True)
    metrics = (("Macro PQ", "macro_pq"), ("Macro recall", "macro_recall"),
               ("Micro recall", "micro_recall"), ("Count MAE", "count_mae"),
               ("Total predicted nuclei", "total_pred_count"), ("Total GT nuclei", "total_gt_count"),
               ("Same-GT duplicate extras", "total_same_gt_duplicate_count"),
               ("Raw overlap pairs (proxy)", "total_raw_overlap_pair_count"),
               ("Valid completions", "format_valid_count"), ("Token-cap hits", "token_cap_hits"))
    rows = []
    for label, key in metrics:
        cells = []
        for variant in VARIANTS:
            value = summaries[variant][key]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append(f"<tr><th>{escape(label)}</th><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
    details = []
    for index, case in enumerate(cases):
        for variant in VARIANTS:
            record = case["arms"][variant]["record"]
            details.append(f"<details><summary>{index + 1:02d} {escape(case['selection']['id'])} / {escape(variant)}"
                           f" / {'valid' if record['format_valid'] else 'INVALID'}</summary>"
                           f"<p>{escape(record.get('parse_error') or 'No format error')}</p>"
                           f"<pre>{escape(record['text'])}</pre></details>")
    model = result.get("model", {})
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paired prompt-format diagnostic</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#182235;max-width:1120px;margin:32px auto;padding:0 20px}}
h1{{line-height:1.2}}.note{{background:#eef3f9;padding:16px;border-radius:8px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce2ea}}thead{{background:#eef3f9}}td{{font-variant-numeric:tabular-nums}}
img{{width:100%;height:auto;border:1px solid #dce2ea;margin:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:12px}}
details{{margin:10px 0}}dt{{font-weight:650;margin-top:10px}}dd{{margin-left:0}}small{{color:#536279}}</style>
<h1>One-object vs three-object formatting example</h1>
<p class="note">Inference-only comparison on {len(cases)} preselected official-30 training crops. The numeric demonstration changes;
the image, model weights, instructions, and decoding settings remain paired. No optimizer steps or adapter training.
These density-selected examples are a diagnostic, not held-out performance or evidence that one prompt will generalize better.</p>
<p><small>Model: {escape(model.get('model', 'Qwen/Qwen2.5-VL-7B-Instruct'))}<br>
Revision: {escape(model.get('resolved_revision', 'See result.json'))}<br>
All local source image and annotation hashes match selected_crops. Final-mask metrics and summary totals were recomputed and checked.</small></p>
<table><thead><tr><th>Metric ({len(cases)} crops per arm)</th><th>One-object example</th><th>Three-object example</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Every preselected case, in selection order</h2>
<p>Each panel shows the same original image enlarged with nearest-neighbor sampling, with colored nucleus outlines and local instance IDs.
Colors and IDs are panel-local; matching colors across panels do not mean the same nucleus. Invalid responses remain in all metric denominators.</p>
{''.join(f'<img src="{escape(name)}" alt="Paired masks for preselected cases, sheet {index + 1}">' for index, name in enumerate(sheets))}
<h2>Reading the metrics</h2><dl>
<dt>PQ and recall</dt><dd>Computed on final, non-overlapping instance maps. A predicted and reference nucleus match only at IoU &gt; 0.5.
PQ combines matching and outline overlap; recall is matched reference nuclei divided by all reference nuclei.
Macro metrics average crops equally; micro recall is total matched nuclei divided by total reference nuclei.</dd>
<dt>Count MAE</dt><dd>Mean absolute difference between final predicted and reference counts per crop. Accurate counts alone do not establish correct locations or outlines.</dd>
<dt>Same-GT duplicate extras</dt><dd>{escape(DUPLICATE_DEFINITIONS['same_gt_duplicate_count'])} These use separate SAM masks before overlap resolution.</dd>
<dt>Raw overlap pairs</dt><dd>{escape(DUPLICATE_DEFINITIONS['raw_overlap_pair_count'])}</dd>
<dt>Format validity and token cap</dt><dd>Validity is the existing strict parser result. A token-cap hit records that generation used its full allowance; it does not by itself imply invalid output.</dd>
</dl><h2>Full model responses and format errors</h2>{''.join(details)}
</html>"""
    destination.write_text(document, encoding="utf-8")


def render_report(run, dataset, output):
    run, dataset, output = Path(run).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty report directory; previous reports are not overwritten")
    result, cases, summaries = load_verified_cases(run, dataset)
    output.mkdir(parents=True, exist_ok=True)
    sheets = []
    for first_index in range(0, len(cases), 4):
        name = f"contact_sheet_{first_index // 4 + 1:02d}.png"
        render_sheet(cases[first_index:first_index + 4], first_index, len(cases), output / name)
        sheets.append(name)
    render_html(result, cases, summaries, sheets, output / "index.html")
    return {"report": str(output / "index.html"), "sheets": [str(output / name) for name in sheets],
            "verified_cases": len(cases)}


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        print(json.dumps(render_report(args.run, args.dataset, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Cannot render report: {error}") from error
