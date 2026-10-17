"""Verify and display a saved single-rectangle coordinate calibration.

No model, processor or GPU is loaded. The original image is checked pixel by
pixel; every geometric measurement is recomputed from the saved model answer.
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
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import parse_completion
from scripts.render_prompt_ablation import _font, _fit_text


TARGET_RGB = (30, 100, 220)
TARGET_BOX = (72, 20, 112, 52)
INK, MUTED = "#172438", "#526279"
GT_COLOR, PRED_COLOR = "#00bca8", "#eb8500"


def _same_number(actual, saved, context):
    if isinstance(actual, bool):
        matches = isinstance(saved, bool) and actual == saved
    elif isinstance(actual, (int, float)):
        matches = (isinstance(saved, (int, float)) and not isinstance(saved, bool)
                   and math.isfinite(saved) and math.isclose(actual, saved, rel_tol=1e-9, abs_tol=1e-10))
    else:
        matches = actual == saved
    if not matches:
        raise ValueError(f"Saved result disagrees with recomputed value: {context}")


def _same_vector(actual, saved, context):
    if not isinstance(saved, list) or len(actual) != len(saved):
        raise ValueError(f"Saved vector has wrong dimensions: {context}")
    for index, (left, right) in enumerate(zip(actual, saved)):
        _same_number(left, right, f"{context}[{index}]")


def _box_iou(box, target):
    x0, y0 = max(box[0], target[0]), max(box[1], target[1])
    x1, y1 = min(box[2], target[2]), min(box[3], target[3])
    intersection = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (target[2] - target[0]) * (target[3] - target[1]) - intersection)
    return intersection / union


def verify_run(run):
    run = Path(run).resolve()
    result = json.loads((run / "result.json").read_text())
    if result.get("status") != "completed_coordinate_calibration" or result.get("sam_used") is not False:
        raise ValueError("Expected a completed coordinate-only calibration with no SAM call")
    if result.get("optimizer_steps", 0) != 0:
        raise ValueError("This report requires an inference-only calibration")
    source = run / "input.png"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != result["source"]["image_sha256"]:
        raise ValueError("Input image SHA-256 differs from the saved run")
    with Image.open(source) as opened:
        if opened.mode != "RGB" or opened.size != (128, 128):
            raise ValueError("Expected the original RGB 128 x 128 input image")
        image = opened.copy()
    pixels = np.asarray(image)
    expected = np.full((128, 128, 3), 255, dtype=np.uint8)
    expected[20:52, 72:112] = TARGET_RGB
    if not np.array_equal(pixels, expected):
        raise ValueError("Input pixels differ from the single blue rectangle on white")
    if result["source"]["size"] != [128, 128] or result["original_size"] != [128, 128]:
        raise ValueError("Saved source dimensions differ from the actual PNG")
    if result["prompt_size"] != [224, 224]:
        raise ValueError("This diagnostic requires the recorded 224 x 224 coordinate frame")
    if result.get("coordinate_frame", "processed") != "processed":
        raise ValueError("The generated coordinates must be in the processed-image frame")
    processed_box = [value * 224 / 128 for value in TARGET_BOX]
    _same_vector(TARGET_BOX, result["known_geometry"]["original_box"], "known original box")
    _same_vector(processed_box, result["known_geometry"]["processed_box"], "known processed box")
    max_objects = result["generation_controls"]["max_objects"]
    if max_objects != 8:
        raise ValueError("Expected the fixed parser limit of eight objects; target count remains one")
    parsed = parse_completion(result["text"], width=224, height=224, max_objects=max_objects)
    _same_number(parsed.valid, result["format_valid"], "format validity")
    if parsed.error != result["parse_error"]:
        raise ValueError("Saved parse error differs from the current strict parser")
    converted, scales = objects_to_original(parsed.objects, (224, 224), image.size)
    _same_vector(scales, result["scale_to_original"], "scale to original")
    for objects, key in ((parsed.objects, "raw_objects"), (converted, "converted_objects")):
        saved = result[key]
        if len(objects) != len(saved):
            raise ValueError(f"Saved {key} count disagrees with the parsed answer")
        for index, (obj, entry) in enumerate(zip(objects, saved), 1):
            for coordinate in ("box", "point"):
                _same_vector(getattr(obj, coordinate), entry[coordinate], f"{key}/P{index}/{coordinate}")
    _same_number(len(parsed.objects), result["object_count"], "object count")
    if len(result["object_checks"]) != len(parsed.objects):
        raise ValueError("Saved geometric check count differs from the parsed answer")
    checks = []
    for index, (obj, original, saved) in enumerate(zip(parsed.objects, converted, result["object_checks"]), 1):
        error = [float(pred - target) for pred, target in zip(obj.box, processed_box)]
        center = ((obj.box[0] + obj.box[2]) / 2, (obj.box[1] + obj.box[3]) / 2)
        target_center = ((processed_box[0] + processed_box[2]) / 2, (processed_box[1] + processed_box[3]) / 2)
        point_inside = (processed_box[0] <= obj.point[0] < processed_box[2]
                        and processed_box[1] <= obj.point[1] < processed_box[3])
        x, y = original.point
        point_id = int(np.array_equal(pixels[int(math.floor(y)), int(math.floor(x))], TARGET_RGB))
        if bool(point_id) != point_inside:
            raise ValueError("Continuous point containment disagrees with actual target pixels")
        check = {"index": index, "box_iou": _box_iou(obj.box, processed_box),
                 "max_edge_error_processed_px": max(abs(value) for value in error),
                 "point_inside_target": point_inside,
                 "box_center_error_processed_px": math.dist(center, target_center),
                 "edge_errors_processed_px": error, "point_gt_id_floor_xy": point_id}
        for key, value in check.items():
            if key == "edge_errors_processed_px":
                _same_vector(value, saved[key], f"P{index}/{key}")
            else:
                _same_number(value, saved[key], f"P{index}/{key}")
        checks.append(check)
    return result, image, checks


def _verdict(result, checks):
    if not result["format_valid"]:
        return "The output failed the format check.", "The answer fails the format contract, so no valid box or point is available for localization scoring."
    if len(checks) != 1:
        return (f"One target was requested; {len(checks)} objects were returned.",
                f"The image contains one target, but the model returned {len(checks)} objects; inspect every returned location.")
    item = checks[0]
    if item["box_iou"] == 1.0 and item["point_inside_target"]:
        return "This box matches the known rectangle; its point is inside.", "The box matches the known rectangle, and its point lies inside the target."
    if item["box_iou"] == 0:
        return "The generated box has no overlap with the known rectangle.", "The box is disjoint from the known rectangle; localization fails on this simple image."
    if not item["point_inside_target"]:
        return "The box overlaps the target, but its point is outside.", "The box overlaps part of the target, but the point lies outside it."
    return "The box overlaps the target and its point is inside; the box is not exact.", "The box overlaps the target and its point is inside, but the box location or size is still inaccurate."


def _vec(values):
    return "[" + ", ".join(f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else str(int(value)) for value in values) + "]"


def _panel(canvas, xy, title, subtitle, source, *, box=None, objects=None, source_only=False):
    ox, oy = xy
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((ox, oy, ox + 430, oy + 530), radius=14, fill="white", outline="#dce3eb", width=2)
    draw.text((ox + 20, oy + 16), title, font=_font(23), fill=INK)
    draw.text((ox + 20, oy + 49), subtitle, font=_font(15), fill=MUTED)
    left, top, size = ox + 52, oy + 99, 336
    canvas.paste(source.resize((size, size), Image.Resampling.NEAREST), (left, top))
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
    if box is not None:
        draw.rectangle((left + box[0] * size / 224, top + box[1] * size / 224,
                        left + box[2] * size / 224, top + box[3] * size / 224), outline=GT_COLOR, width=4)
    if objects:
        for index, obj in enumerate(objects, 1):
            x0, y0, x1, y1 = obj["box"]
            draw.rectangle((left + x0 * size / 224, top + y0 * size / 224,
                            left + x1 * size / 224, top + y1 * size / 224), outline=PRED_COLOR, width=3)
            x, y = left + obj["point"][0] * size / 224, top + obj["point"][1] * size / 224
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=PRED_COLOR, outline=INK)
            draw.text((min(left + size - 14, max(left + 14, x)), max(top + 18, y - 11)), f"P{index}",
                      anchor="mb", font=_font(17), fill="white", stroke_width=2, stroke_fill=INK)
    footer = "No axes or text were drawn on the input image." if source_only else "Upper-left origin; x right, y down."
    draw.text((ox + 20, oy + 492), footer, font=_font(15), fill=MUTED)


def render_figure(result, image, checks, destination):
    canvas = Image.new("RGB", (1350, 820), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Single-rectangle coordinate calibration", font=_font(34), fill=INK)
    draw.text((24, 66), "Saved Qwen 32B inference | One synthetic image | No SAM | No training", font=_font(21), fill=MUTED)
    _panel(canvas, (18, 112), "A  Actual input image", "128 x 128; blue rectangle on white", image, source_only=True)
    _panel(canvas, (460, 112), "B  Known target box", "Known 224-frame box: [126, 35, 196, 91]", image,
           box=result["known_geometry"]["processed_box"])
    _panel(canvas, (902, 112), "C  Qwen box + point", "224-coordinate frame; orange = model output", image,
           objects=result["raw_objects"] if result["format_valid"] else [])
    verdict, _ = _verdict(result, checks)
    draw.text((24, 668), _fit_text(draw, verdict, _font(23), 1302), font=_font(23), fill=INK)
    if len(checks) == 1:
        check = checks[0]
        metrics = (f"Box IoU: {check['box_iou']:.4f}  |  Point inside: {'yes' if check['point_inside_target'] else 'no'}  |  "
                   f"Max edge error: {check['max_edge_error_processed_px']:.2f} processed pixels")
    else:
        metrics = f"Valid format: {result['format_valid']}  |  Expected objects: 1  |  Returned objects: {result['object_count']}"
    draw.text((24, 707), metrics, font=_font(21), fill=INK)
    draw.text((24, 752), "A valid foreground point can be anywhere inside the rectangle; it does not need to be its center.", font=_font(19), fill=MUTED)
    draw.text((24, 780), "Box IoU measures box overlap. It is not mask IoU or PQ. One toy case does not establish general ability.", font=_font(19), fill=MUTED)
    canvas.save(destination)


def render_html(result, checks, destination):
    e = lambda value: html.escape(str(value), quote=True)
    english, chinese = _verdict(result, checks)
    rows = []
    for obj, converted, check in zip(result["raw_objects"], result["converted_objects"], checks):
        rows.append(f"<tr><th>P{check['index']}</th><td>{e(_vec(obj['box']))}<br>{e(_vec(obj['point']))}</td>"
                    f"<td>{e(_vec(converted['box']))}<br>{e(_vec(converted['point']))}</td><td>{check['box_iou']:.4f}</td>"
                    f"<td>{'Yes' if check['point_inside_target'] else 'No'}</td><td>{check['max_edge_error_processed_px']:.2f}</td>"
                    f"<td>{check['box_center_error_processed_px']:.2f}</td></tr>")
    if not rows:
        rows = ["<tr><td colspan='7'>No valid returned objects to score</td></tr>"]
    model = result.get("model", {})
    model_name = model.get("model", "See saved result.json") if isinstance(model, dict) else model
    revision = model.get("resolved_revision", "See saved result.json") if isinstance(model, dict) else "See saved result.json"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single-rectangle coordinate calibration</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#172438;background:#eef3f8;max-width:1160px;margin:28px auto;padding:0 20px}}h1{{line-height:1.25}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:16px}}small{{color:#526279}}a{{color:#075da8}}</style>
<h1>Single-rectangle coordinate calibration</h1>
<section><b>{e(english)}</b><p>{e(chinese)}</p><p>The input contains one blue rectangle. The task is to return its box and an interior point. This page checks the saved output without new model inference, training, or SAM calls.</p>
<p>Format valid: <b>{e(result['format_valid'])}</b> · Expected objects: <b>1</b> · Returned objects: <b>{result['object_count']}</b><br>Parse error: {e(result['parse_error'] or 'None')}</p></section>
<a href="coordinate_calibration.png"><img src="coordinate_calibration.png" alt="Actual image, known box and generated box and point in the processed coordinate frame"></a>
<p><a href="coordinate_calibration.png">Open / save comparison figure</a> · <a href="input.png">Original input image</a> · <a href="verification.json">Verification results</a></p>
<section><h2>Known geometry</h2><p>The original image is 128×128. Its blue region spans x∈[72,112), y∈[20,52), with RGB [30,100,220]; all other pixels are white. The origin is top-left, x increases rightward, and y downward.</p>
<p>Qwen answers in the 224×224 frame, where the correct box is <b>[126,35,196,91]</b> . Conversion to the original image multiplies coordinates by 128/224 = 4/7. <b>The point only needs to lie inside the rectangle; it need not be its center.</b></p>
<p><small>Panels B/C use the same original image with coordinate axes, not a reconstruction of Qwen's normalized input tensor. Axes, text, and known answers are absent from the model-input PNG.</small></p></section>
<section><h2>Checks recomputed from saved coordinates</h2><div class="scroll"><table><tr><th>Object</th><th>Raw box / point (224)</th><th>Converted box / point (128)</th><th>Box IoU</th><th>Point inside</th><th>Max edge error<br>(224 pixels)</th><th>Box-center error<br>(224 pixels)</th></tr>{''.join(rows)}</table></div>
<p><b>Box IoU</b> is the intersection area of two boxes divided by their union; 1 means identical boxes. It is neither mask IoU nor PQ. Max edge error is the largest absolute error among the four box edges. Box-center error is the distance between box centers, independent of whether the returned point is centered.</p>
<p>These values are recomputed from input pixels and the parsed raw answer, rather than copied from logged scores. Invalid answers are reported as failures, not correct coordinates.</p></section>
<section><h2>Exact initial prompt</h2><pre>{e(json.dumps(result['prompt'], ensure_ascii=False, indent=2))}</pre><h2>Raw model completion</h2><pre>{e(result['text'])}</pre></section>
<section><h2>What can this test establish?</h2><p>It checks localization and coordinate output for one simple geometric target. Success establishes only that this particular example works, not that the model can find nuclei or multiple objects. Failure in one example also cannot isolate image understanding, instruction following, or the coordinate convention as the cause.</p>
<p>This is a synthetic-image inference diagnostic, without pathology images, training, SAM masks, or held-out dataset metrics.</p></section>
<section><small>Model: {e(model_name)}<br>Revision: {e(revision)}<br>Completion tokens: {result['completion_tokens']} · Token cap reached: {e(result['hit_token_cap'])}<br>
Input SHA-256: {e(result['source']['image_sha256'])}<br>Processed pixel tensor SHA-256: {e(result['pixel_values_sha256'])}</small>
<details><summary>Generation controls</summary><pre>{e(json.dumps(result['generation_controls'], ensure_ascii=False, indent=2))}</pre></details></section></html>"""
    destination.write_text(document, encoding="utf-8")


def render_report(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty output directory; existing reports are never overwritten")
    result, image, checks = verify_run(run)
    output.mkdir(parents=True, exist_ok=True)
    render_figure(result, image, checks, output / "coordinate_calibration.png")
    render_html(result, checks, output / "index.html")
    (output / "input.png").write_bytes((run / "input.png").read_bytes())
    verification = {"source_sha256_verified": True, "source_pixels_verified": True,
                    "geometry_recomputed": True, "raw_answer_reparsed": True,
                    "conversion_matches_saved": True, "sam_used": False,
                    "object_checks": checks, "object_count": result["object_count"],
                    "format_valid": result["format_valid"], "source_result_sha256": hashlib.sha256((run / "result.json").read_bytes()).hexdigest()}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    return {"report": str(output / "index.html"), "figure": str(output / "coordinate_calibration.png"),
            "object_count": result["object_count"], "format_valid": result["format_valid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(render_report(args.run, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Cannot render coordinate calibration: {error}") from error
