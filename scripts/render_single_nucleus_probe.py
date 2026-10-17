"""Verify and visualize one saved real-nucleus localization probe.

Reads saved output only. Checks the original crop and annotation hashes, derives
the tight GT box, reparses the completion, and independently scores geometry.
The comparison to a synthetic rectangle is explanatory, not a benchmark.
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import parse_completion
from scripts.render_coordinate_calibration import (
    INK, MUTED, GT_COLOR, PRED_COLOR, _box_iou, _same_number,
    _same_vector, _vec, verify_run as verify_rectangle,
)
from scripts.render_prompt_ablation import _font, _fit_text


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_probe(run):
    run = Path(run).resolve()
    result = json.loads((run / "result.json").read_text())
    if (result.get("status") != "completed_single_nucleus_probe"
            or result.get("sam_used") is not False or result.get("optimizer_steps", 0) != 0):
        raise ValueError("Expected a completed inference-only single-nucleus probe without SAM")
    for filename, key in (("input.png", "image_sha256"), ("gt_mask.png", "mask_sha256")):
        if _sha(run / filename) != result["source"][key]:
            raise ValueError(f"Saved source hash differs: {filename}")
    with Image.open(run / "input.png") as opened:
        if opened.mode != "RGB" or opened.size != (128, 128):
            raise ValueError("Expected the unmodified RGB 128 x 128 crop")
        image = opened.copy()
    with Image.open(run / "gt_mask.png") as opened:
        gt = np.asarray(opened)
    if (gt.shape != (128, 128) or gt.dtype.kind not in "ui"
            or np.any(gt < 0)):
        raise ValueError("GT must be a 128 x 128 nonnegative integer instance map")
    ids = np.unique(gt[gt > 0])
    if len(ids) != 1:
        raise ValueError("Single-nucleus probe requires exactly one positive GT instance ID")
    foreground = gt > 0
    if any(np.any(edge) for edge in (foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1])):
        raise ValueError("The annotated nucleus must be fully inside the crop, not touch an edge")
    selection = result["source"]["selection"]
    source_id = selection["source_image_id"]
    if source_id not in load_official30_cohort()["image_ids"]:
        raise ValueError("Selected source image is outside the official-30 whitelist")
    if selection["patient_id"] != "-".join(source_id.split("-")[:3]):
        raise ValueError("Selected patient ID disagrees with the source image")
    if selection.get("split", "train") != "train":
        raise ValueError("This diagnostic must use a training-source crop")
    if result["source"]["size"] != [128, 128] or result["original_size"] != [128, 128]:
        raise ValueError("Saved image dimensions differ from the actual crop")
    if result["prompt_size"] != [224, 224]:
        raise ValueError("Expected the same 224 x 224 processed-coordinate frame as the rectangle probe")
    positions = np.argwhere(foreground)
    ymin, xmin = positions.min(axis=0)
    ymax, xmax = positions.max(axis=0) + 1
    box = [int(xmin), int(ymin), int(xmax), int(ymax)]
    processed_box = [value * 224 / 128 for value in box]
    known = result["known_geometry"]
    _same_vector(box, known["original_box"], "GT box derived from pixels")
    _same_vector(processed_box, known["processed_box"], "GT processed box")
    _same_number(int(ids[0]), known["instance_id"], "GT instance ID")
    _same_number(1, known["gt_count"], "GT count")
    _same_number(int(foreground.sum()), known["gt_area"], "GT foreground area")
    controls = result["generation_controls"]
    if controls["max_objects"] != 8:
        raise ValueError("Expected parser limit eight; the actual annotated object count is one")
    parsed = parse_completion(result["text"], width=224, height=224, max_objects=8)
    _same_number(parsed.valid, result["format_valid"], "format validity")
    if parsed.error != result["parse_error"]:
        raise ValueError("Saved parse error differs from the current strict parser")
    converted, scales = objects_to_original(parsed.objects, (224, 224), image.size)
    _same_vector(scales, result["scale_to_original"], "scale to original crop")
    for objects, key in ((parsed.objects, "raw_objects"), (converted, "converted_objects")):
        if len(objects) != len(result[key]):
            raise ValueError(f"Saved {key} count differs from parsed completion")
        for index, (obj, saved) in enumerate(zip(objects, result[key]), 1):
            for coordinate in ("box", "point"):
                _same_vector(getattr(obj, coordinate), saved[coordinate], f"{key}/P{index}/{coordinate}")
    _same_number(len(parsed.objects), result["object_count"], "returned object count")
    if len(result["object_checks"]) != len(parsed.objects):
        raise ValueError("Saved object-check count differs from the actual output")
    checks = []
    for index, (obj, original, saved) in enumerate(zip(parsed.objects, converted, result["object_checks"]), 1):
        errors = [float(pred - target) for pred, target in zip(obj.box, processed_box)]
        pred_center = [(obj.box[0] + obj.box[2]) / 2, (obj.box[1] + obj.box[3]) / 2]
        gt_center = [(processed_box[0] + processed_box[2]) / 2, (processed_box[1] + processed_box[3]) / 2]
        x, y = original.point
        point_id = int(gt[int(math.floor(y)), int(math.floor(x))])
        point_in_box = (processed_box[0] <= obj.point[0] < processed_box[2]
                        and processed_box[1] <= obj.point[1] < processed_box[3])
        check = {"index": index, "box_iou": _box_iou(obj.box, processed_box),
                 "edge_errors_processed_px": errors,
                 "max_edge_error_processed_px": max(abs(value) for value in errors),
                 "box_center_error_processed_px": math.dist(pred_center, gt_center),
                 "point_inside_target_box": point_in_box, "point_inside_target": point_id > 0,
                 "point_gt_id_floor_xy": point_id}
        for key, value in check.items():
            if isinstance(value, list):
                _same_vector(value, saved[key], f"P{index}/{key}")
            else:
                _same_number(value, saved[key], f"P{index}/{key}")
        checks.append(check)
    return result, image, gt, checks


def verify_comparison(probe, rectangle):
    for key in ("model", "resolved_revision", "processor_revision", "adapter"):
        if probe["model"][key] != rectangle["model"][key]:
            raise ValueError(f"Checkpoint control differs from rectangle probe: {key}")
    if probe["generation_controls"] != rectangle["generation_controls"]:
        raise ValueError("Generation controls differ from the rectangle probe")
    if probe["prompt_size"] != rectangle["prompt_size"] or probe["original_size"] != rectangle["original_size"]:
        raise ValueError("Image dimensions or coordinate frames differ from rectangle probe")


def _boundary(mask):
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def _verdict(result, checks):
    if not result["format_valid"]:
        return "The output failed the format check.", "The answer fails the format check, leaving no valid box or point to inspect."
    if len(checks) != 1:
        return (f"One annotated nucleus; {len(checks)} objects returned.",
                f"The crop contains one annotated nucleus; the model returned {len(checks)} objects. The table keeps every object without choosing the best one.")
    check = checks[0]
    if check["box_iou"] == 1 and check["point_inside_target"]:
        return "The box matches the GT box, and the point is inside the nucleus.", "The generated box matches the annotated tight box, and the point lies in the actual nucleus."
    if check["box_iou"] == 0:
        return "The generated box does not overlap the GT nucleus box.", "The generated box does not overlap the annotated tight box; localization fails on this image."
    if not check["point_inside_target"]:
        return "The box overlaps the GT box, but the point misses the nucleus.", "The box overlaps the annotated tight box, but the point lies outside the actual nucleus."
    return "The box overlaps the GT box, and the point is inside the nucleus.", "The box overlaps the annotated tight box and the point lies inside the nucleus, but the box edges remain inaccurate."


def _draw_panel(canvas, xy, title, subtitle, source, *, contour=None, gt_box=None, objects=None, source_only=False):
    ox, oy = xy
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((ox, oy, ox + 430, oy + 530), radius=14, fill="white", outline="#dce3eb", width=2)
    draw.text((ox + 20, oy + 16), title, font=_font(23), fill=INK)
    draw.text((ox + 20, oy + 49), subtitle, font=_font(15), fill=MUTED)
    pixels = np.asarray(source).copy()
    if contour is not None:
        pixels[_boundary(contour)] = [0, 188, 168]
    left, top, size = ox + 52, oy + 99, 336
    panel = Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST)
    canvas.paste(panel, (left, top))
    draw.rectangle((left, top, left + size, top + size), outline="#bcc7d5", width=1)
    if not source_only:
        for tick in (0, 56, 112, 168, 224):
            offset = tick * size / 224
            draw.line((left + offset, top + size, left + offset, top + size + 6), fill=INK)
            draw.text((left + offset, top + size + 10), str(tick), anchor="mt", font=_font(14), fill=MUTED)
            draw.line((left - 6, top + offset, left, top + offset), fill=INK)
            draw.text((left - 10, top + offset), str(tick), anchor="rm", font=_font(14), fill=MUTED)
        draw.text((left + size + 16, top + size - 4), "x", font=_font(16), fill=INK)
        draw.text((left - 28, top - 28), "y", font=_font(16), fill=INK)
    if gt_box is not None:
        x0, y0, x1, y1 = gt_box
        draw.rectangle((left + x0 * size / 224, top + y0 * size / 224,
                        left + x1 * size / 224, top + y1 * size / 224), outline=GT_COLOR, width=3)
    for index, obj in enumerate(objects or [], 1):
        x0, y0, x1, y1 = obj["box"]
        draw.rectangle((left + x0 * size / 224, top + y0 * size / 224,
                        left + x1 * size / 224, top + y1 * size / 224), outline=PRED_COLOR, width=3)
        x, y = left + obj["point"][0] * size / 224, top + obj["point"][1] * size / 224
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=PRED_COLOR, outline=INK)
        draw.text((min(left + size - 14, max(left + 14, x)), max(top + 18, y - 11)), f"P{index}",
                  anchor="mb", font=_font(17), fill="white", stroke_width=2, stroke_fill=INK)
    note = "No labels or outlines were supplied in the image." if source_only else "Upper-left origin; x right, y down."
    draw.text((ox + 20, oy + 492), _fit_text(draw, note, _font(15), 390), font=_font(15), fill=MUTED)


def _short_metrics(name, checks, count):
    if len(checks) != 1:
        return f"{name}: {count} returned objects; see each object's geometry in the report."
    c = checks[0]
    return (f"{name}: box IoU {c['box_iou']:.4f} | Point inside target: {'yes' if c['point_inside_target'] else 'no'} | "
            f"Objects: {count} | Max edge error: {c['max_edge_error_processed_px']:.2f} px")


def render_figure(result, image, gt, checks, rectangle, rectangle_checks, destination):
    canvas = Image.new("RGB", (1350, 900), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Single-nucleus localization probe", font=_font(34), fill=INK)
    draw.text((24, 66), "Saved Qwen 32B inference | One annotated nucleus | No SAM | No training", font=_font(21), fill=MUTED)
    _draw_panel(canvas, (18, 112), "A  Actual input crop", "128 x 128 real tissue image", image, source_only=True)
    _draw_panel(canvas, (460, 112), "B  Annotated nucleus", "Cyan: actual GT outline and its tight box", image,
                contour=gt > 0, gt_box=result["known_geometry"]["processed_box"])
    _draw_panel(canvas, (902, 112), "C  Qwen box + point", "Orange: model output; cyan: GT outline", image,
                contour=gt > 0, objects=result["raw_objects"] if result["format_valid"] else [])
    english, _ = _verdict(result, checks)
    draw.text((24, 668), _fit_text(draw, english, _font(23), 1302), font=_font(23), fill=INK)
    draw.text((24, 710), _short_metrics("Earlier rectangle", rectangle_checks, rectangle["object_count"]), font=_font(21), fill=MUTED)
    draw.text((24, 748), _short_metrics("Real nucleus", checks, result["object_count"]), font=_font(21), fill=INK)
    draw.text((24, 801), "Point-inside checks the actual annotated nucleus, not just its box. The point need not be the center.", font=_font(19), fill=MUTED)
    draw.text((24, 832), "Box IoU is not mask IoU or PQ. Both cases are individual diagnostics, not a comparative benchmark.", font=_font(19), fill=MUTED)
    draw.text((24, 863), "The image and target wording changed; checkpoint, decoding settings and coordinate frame stayed fixed.", font=_font(19), fill=MUTED)
    canvas.save(destination)


def render_html(result, checks, rectangle, rectangle_checks, destination):
    e = lambda value: html.escape(str(value), quote=True)
    english, chinese = _verdict(result, checks)
    detailed = []
    for raw, converted, c in zip(result["raw_objects"], result["converted_objects"], checks):
        detailed.append(f"<tr><th>P{c['index']}</th><td>{e(_vec(raw['box']))}<br>{e(_vec(raw['point']))}</td>"
                        f"<td>{e(_vec(converted['box']))}<br>{e(_vec(converted['point']))}</td>"
                        f"<td>{c['box_iou']:.4f}</td><td>{'Yes' if c['point_inside_target_box'] else 'No'}</td>"
                        f"<td>{'Yes' if c['point_inside_target'] else 'No'}</td><td>{c['max_edge_error_processed_px']:.2f}</td>"
                        f"<td>{e(_vec(c['edge_errors_processed_px']))}</td></tr>")
    if not detailed:
        detailed = ["<tr><td colspan='8'>No valid returned objects to score</td></tr>"]
    comparisons = []
    for name, saved, values in (("Synthetic rectangle", rectangle, rectangle_checks),
                               ("Real nucleus", result, checks)):
        scored = values[0] if len(values) == 1 else None
        comparisons.append(f"<tr><th>{name}</th><td>{e(saved['format_valid'])}</td><td>{saved['object_count']}</td>"
                           f"<td>{format(scored['box_iou'], '.4f') if scored else 'See individual objects'}</td>"
                           f"<td>{('Yes' if scored['point_inside_target'] else 'No') if scored else 'See individual objects'}</td>"
                           f"<td>{format(scored['max_edge_error_processed_px'], '.2f') if scored else 'See individual objects'}</td></tr>")
    known = result["known_geometry"]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single-nucleus localization probe</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#172438;background:#eef3f8;max-width:1160px;margin:28px auto;padding:0 20px}}h1{{line-height:1.25}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:16px}}small{{color:#526279}}a{{color:#075da8}}</style>
<h1>From a rectangle to a real nucleus<br>Single-nucleus localization probe</h1>
<section><b>{e(english)}</b><p>{e(chinese)}</p><p>This checks localization of one complete annotated nucleus in a native 128×128 tissue crop. Annotations were used only for selection and post-hoc checks, not drawn on the model input. This page reuses saved answers without inference, training, or SAM calls.</p>
<p>Format valid: <b>{e(result['format_valid'])}</b> · Annotated nuclei: <b>1</b> · Returned objects: <b>{result['object_count']}</b><br>Parse error: {e(result['parse_error'] or 'None')}</p></section>
<a href="single_nucleus_probe.png"><img src="single_nucleus_probe.png" alt="Original real tissue crop, true nucleus outline and box, and Qwen's generated box and point"></a>
<p><a href="single_nucleus_probe.png">Open / save comparison figure</a> · <a href="input.png">Original input image</a> · <a href="gt_mask.png">GT instance mask</a> · <a href="verification.json">Numerical verification</a></p>
<section><h2>Comparison with the earlier rectangle: two individual examples</h2><div class="scroll"><table><tr><th>Input</th><th>Valid format</th><th>Returned objects</th><th>Box IoU</th><th>Point inside actual target</th><th>Max edge error (224 px)</th></tr>{''.join(comparisons)}</table></div>
<p>Both use the same Qwen checkpoint, decoding settings, and 224-coordinate convention. The image and target-description words changed: one is a rectangle and one a nucleus. <b>These are not a shared test set or a performance leaderboard.</b>They only support incremental checks of success or failure on particular inputs.</p></section>
<section><h2>Known geometry and scoring</h2><p>GT instance ID: {known['instance_id']}; foreground area: {known['gt_area']} pixels. It is the only annotated nucleus, and its foreground touches none of the crop edges.</p>
<p>The original-image GT tight box is <b>{e(_vec(known['original_box']))}</b>; after scaling by 224/128 it is <b>{e(_vec(known['processed_box']))}</b> . Qwen coordinates are converted back by multiplying by 128/224.</p>
<p><b>Box IoU</b> compares the predicted box with the GT tight box, not masks. <b>Point inside target</b> checks whether the point lies on actual nucleus foreground. Being inside the tight box alone is insufficient, and the point need not be central. The pixel lookup is GT[floor(y),floor(x)].</p>
<p><small>Panels B/C overlay 224-coordinate axes on the original image; they do not reconstruct Qwen's normalized tensor. Cyan outlines come from GT and orange coordinates from the raw answer, without manual correction.</small></p></section>
<section><h2>Every returned object</h2><div class="scroll"><table><tr><th>Object</th><th>Raw box / point (224)</th><th>Converted box / point (128)</th><th>Box IoU</th><th>Point in GT box</th><th>Point in actual nucleus</th><th>Max edge error (224 px)</th><th>Signed edge errors (224 px)</th></tr>{''.join(detailed)}</table></div>
<p>Signed edge errors are prediction minus annotation in [x0,y0,x1,y1] order. Max edge error is the largest absolute value. All values are rechecked from saved input, GT, and raw output rather than copied from logged scores.</p></section>
<section><h2>Exact initial prompt</h2><pre>{e(json.dumps(result['prompt'],ensure_ascii=False,indent=2))}</pre><h2>Raw model completion</h2><pre>{e(result['text'])}</pre></section>
<section><h2>What this can establish</h2><p>This crop was selected using GT from the official 30 training-source images and contains one complete annotated nucleus. It is not an independent test set or an estimate of dataset performance. Success would not establish multi-object enumeration; failure alone cannot separate visual-recognition and coordinate-generation causes.</p>
<p>SAM was not run, so there are no segmentation masks or PQ scores. The table's box IoU cannot be compared directly with earlier mask IoU or PQ.</p></section>
<section><h2>Source provenance</h2><pre>{e(json.dumps(result['source']['selection'],ensure_ascii=False,indent=2))}</pre><small>Image SHA-256: {e(result['source']['image_sha256'])}<br>GT SHA-256: {e(result['source']['mask_sha256'])}<br>
Model: {e(result['model']['model'])}<br>Revision: {e(result['model']['resolved_revision'])}<br>Completion tokens: {result['completion_tokens']} · Token cap reached: {e(result['hit_token_cap'])}</small>
<details><summary>Generation controls</summary><pre>{e(json.dumps(result['generation_controls'],ensure_ascii=False,indent=2))}</pre></details></section></html>"""
    destination.write_text(document, encoding="utf-8")


def render_report(run, rectangle_run, output):
    run, rectangle_run, output = [Path(path).resolve() for path in (run, rectangle_run, output)]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty directory; previous reports are not overwritten")
    result, image, gt, checks = verify_probe(run)
    rectangle, _, rectangle_checks = verify_rectangle(rectangle_run)
    verify_comparison(result, rectangle)
    output.mkdir(parents=True, exist_ok=True)
    render_figure(result, image, gt, checks, rectangle, rectangle_checks, output / "single_nucleus_probe.png")
    render_html(result, checks, rectangle, rectangle_checks, output / "index.html")
    for filename in ("input.png", "gt_mask.png"):
        (output / filename).write_bytes((run / filename).read_bytes())
    verification = {"image_and_mask_hashes_verified": True, "single_gt_instance_verified": True,
                    "no_gt_border_contact_verified": True, "official30_source_verified": True,
                    "gt_geometry_derived_from_mask": True, "raw_answer_reparsed": True,
                    "point_scored_on_actual_gt_foreground": True, "saved_rectangle_reverified": True,
                    "same_checkpoint_and_generation_controls": True, "sam_used": False,
                    "object_count": result["object_count"], "format_valid": result["format_valid"],
                    "object_checks": checks, "rectangle_object_checks": rectangle_checks,
                    "source_result_sha256": _sha(run / "result.json"),
                    "rectangle_result_sha256": _sha(rectangle_run / "result.json")}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    return {"report": str(output / "index.html"), "figure": str(output / "single_nucleus_probe.png"),
            "object_count": result["object_count"], "format_valid": result["format_valid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rectangle-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(render_report(args.run, args.rectangle_run, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Cannot render single-nucleus probe: {error}") from error
