"""Verify and render one saved multi-nucleus coordinate probe; no model calls.

All geometric diagnostics are recomputed from the saved response and native GT
mask. Point coverage, box matching, count differences and duplicates remain
separate; box-overlap diagnostics are never presented as mask PQ.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import parse_completion
from scripts.render_coordinate_calibration import INK, MUTED, GT_COLOR, PRED_COLOR, _same_number, _same_vector, _vec
from scripts.render_prompt_ablation import _font, _fit_text
from scripts.render_single_nucleus_probe import _boundary, verify_probe as verify_single


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compare(actual, saved, context):
    if isinstance(actual, dict):
        if not isinstance(saved, dict) or set(actual) != set(saved):
            raise ValueError(f"Saved diagnostic has different fields: {context}")
        for key in actual:
            _compare(actual[key], saved[key], f"{context}/{key}")
    elif isinstance(actual, list):
        if not isinstance(saved, list) or len(actual) != len(saved):
            raise ValueError(f"Saved diagnostic has different list length: {context}")
        for index, (left, right) in enumerate(zip(actual, saved)):
            _compare(left, right, f"{context}/{index}")
    else:
        _same_number(actual, saved, context)


def verify_probe(run, single_run):
    from nucleus_rl.localization_metrics import analyze, geometry

    run, single_run = Path(run).resolve(), Path(single_run).resolve()
    result = json.loads((run / "result.json").read_text())
    if (result.get("status") != "completed_multi_nucleus_probe"
            or result.get("sam_used") is not False or result.get("optimizer_steps", 0) != 0):
        raise ValueError("Expected a completed inference-only multi-nucleus probe without SAM")
    for filename, key in (("input.png", "image_sha256"), ("gt_mask.png", "mask_sha256")):
        if _sha(run / filename) != result["source"][key]:
            raise ValueError(f"Input hash disagrees with saved run: {filename}")
    with Image.open(run / "input.png") as opened:
        if opened.mode != "RGB" or opened.size != (128, 128):
            raise ValueError("Expected the original RGB 128 x 128 crop")
        image = opened.copy()
    with Image.open(run / "gt_mask.png") as opened:
        gt = np.asarray(opened)
    if gt.shape != (128, 128) or gt.dtype.kind not in "ui" or np.any(gt < 0):
        raise ValueError("GT must be a nonnegative integer 128 x 128 instance map")
    ids = np.unique(gt[gt > 0])
    if not 2 <= len(ids) <= 4:
        raise ValueError("This few-nucleus probe requires two to four annotated nuclei")
    foreground = gt > 0
    if any(np.any(edge) for edge in (foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1])):
        raise ValueError("Every annotated nucleus must be wholly within the crop")
    selection = result["source"]["selection"]
    source_id = selection["source_image_id"]
    if source_id not in load_official30_cohort()["image_ids"]:
        raise ValueError("Source image is outside the official-30 whitelist")
    if selection["patient_id"] != "-".join(source_id.split("-")[:3]) or selection.get("split") != "train":
        raise ValueError("Source patient identity or training split is inconsistent")
    for selection_key, source_key in (("image_sha256", "image_sha256"), ("mask_sha256", "mask_sha256"),
                                      ("crop_image_png_sha256", "image_sha256"), ("crop_mask_png_sha256", "mask_sha256")):
        if selection_key in selection and selection[selection_key] != result["source"][source_key]:
            raise ValueError(f"Selection hash differs from run input: {selection_key}")
    if result["source"]["size"] != [128, 128] or result["original_size"] != [128, 128] or result["prompt_size"] != [224, 224]:
        raise ValueError("Expected the fixed 128-pixel input and 224-pixel coordinate frame")
    known = geometry(gt, processed_size=(224, 224))
    _compare(known, result["known_geometry"], "known_geometry")
    if result["generation_controls"]["max_objects"] != 8:
        raise ValueError("Expected the unchanged parser limit of eight objects")
    parsed = parse_completion(result["text"], width=224, height=224, max_objects=8)
    _same_number(parsed.valid, result["format_valid"], "format validity")
    if parsed.error != result["parse_error"]:
        raise ValueError("Saved parse error differs from the current parser")
    converted, scales = objects_to_original(parsed.objects, (224, 224), image.size)
    _same_vector(scales, result["scale_to_original"], "coordinate scaling")
    for objects, key in ((parsed.objects, "raw_objects"), (converted, "converted_objects")):
        if len(objects) != len(result[key]):
            raise ValueError(f"Saved {key} count differs from the raw answer")
        for index, (obj, entry) in enumerate(zip(objects, result[key]), 1):
            for coordinate in ("box", "point"):
                _same_vector(getattr(obj, coordinate), entry[coordinate], f"{key}/P{index}/{coordinate}")
    if "object_count" in result:
        _same_number(len(parsed.objects), result["object_count"], "object count")
    analysis = analyze(parsed.objects, converted, gt, processed_size=(224, 224))
    _compare(analysis, result["analysis"], "analysis")
    previous, _, single_gt, _ = verify_single(single_run)
    for key in ("model", "resolved_revision", "processor_revision", "adapter"):
        if result["model"][key] != previous["model"][key]:
            raise ValueError(f"Checkpoint differs from the single-nucleus probe: {key}")
    for key in ("generation_controls", "prompt", "prompt_size", "original_size"):
        if result[key] != previous[key]:
            raise ValueError(f"Paired control differs from the single-nucleus probe: {key}")
    single_analysis = analyze(previous["raw_objects"], previous["converted_objects"], single_gt, processed_size=(224, 224))
    return result, image, gt, known, analysis, previous, single_analysis


def verify_raw_forensic(run, result, known, gt):
    """Inspect a rejected response without repairing it or changing acceptance.

    This path requires the separate independent audit. Geometry is cross-checked
    again against the literal JSON numbers, including negative box edges. It is
    never passed to the pipeline parser or represented as an accepted prediction.
    """
    from nucleus_rl.localization_metrics import _match_box_iou
    from scripts.render_coordinate_calibration import _box_iou

    path = Path(run) / "independent_audit.json"
    audit = json.loads(path.read_text())
    _compare(known, audit["known_geometry"], "independent audit GT geometry")
    if audit["strict_result"]["format_valid"] != result["format_valid"]:
        raise ValueError("Independent audit disagrees with strict response validity")
    if audit["strict_result"]["accepted_object_count"] != result["analysis"]["prediction_count"]:
        raise ValueError("Independent audit disagrees with strict accepted object count")
    match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", result["text"], re.DOTALL)
    if not match:
        raise ValueError("Raw forensic view requires one complete JSON answer block")
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Ambiguous duplicate JSON key in raw forensic answer")
            value[key] = item
        return value
    parsed = json.loads(match.group(1), object_pairs_hook=unique_keys)
    if not isinstance(parsed, dict) or set(parsed) != {"objects"} or not isinstance(parsed["objects"], list):
        raise ValueError("Raw forensic view requires the literal objects array")
    objects = parsed["objects"]
    raw = audit["raw_forensic_analysis"]
    if len(raw["predictions"]) != len(objects) or raw["prediction_count"] != len(objects):
        raise ValueError("Raw emitted object count differs from the independent audit")
    by_id = {item["instance_id"]: item["display_id"] for item in known["instances"]}
    gt_boxes = [item["processed_box"] for item in known["instances"]]
    matrix, entry_validity = [], []
    for index, (obj, pred) in enumerate(zip(objects, raw["predictions"]), 1):
        if not isinstance(obj, dict) or set(obj) != {"box", "point"}:
            raise ValueError("Raw forensic objects must have exactly box and point")
        for key, length in (("box", 4), ("point", 2)):
            if (not isinstance(obj[key], list) or len(obj[key]) != length
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in obj[key])):
                raise ValueError("Raw forensic coordinates must be finite numeric vectors")
        box, point = obj["box"], obj["point"]
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("Raw geometric audit requires positive-area boxes")
        original_box, original_point = [value * 4 / 7 for value in box], [value * 4 / 7 for value in point]
        for actual, key in ((box, "raw_box"), (point, "raw_point"),
                            (original_box, "original_box"), (original_point, "original_point")):
            _same_vector(actual, pred[key], f"forensic/P{index}/{key}")
        if pred["display_id"] != f"P{index}" or pred["accepted_by_strict_parser"] != result["format_valid"]:
            raise ValueError("Raw forensic order or acceptance provenance differs")
        in_frame = 0 <= point[0] < 224 and 0 <= point[1] < 224
        valid = (0 <= box[0] < box[2] <= 224 and 0 <= box[1] < box[3] <= 224 and in_frame
                 and box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])
        entry_validity.append(valid)
        if pred["per_entry_contract_valid"] != valid:
            raise ValueError("Raw per-object contract validity differs")
        point_instance = int(gt[math.floor(original_point[1]), math.floor(original_point[0])]) if in_frame else 0
        if pred["point_gt_id"] != by_id.get(point_instance):
            raise ValueError("Raw point GT membership differs from source pixels")
        row = [_box_iou(box, target) for target in gt_boxes]
        matrix.append(row)
        best_index = int(np.argmax(row)) if row and max(row) > 0 else None
        best_id = known["instances"][best_index]["display_id"] if best_index is not None else None
        if pred["best_box_gt_id"] != best_id:
            raise ValueError("Raw best-positive-overlap association differs; never force zero-overlap matches")
        _same_number(max(row, default=0), pred["best_box_iou"], f"forensic/P{index}/best IoU")
        edges = [left - right for left, right in zip(box, gt_boxes[best_index])] if best_index is not None else None
        offsets = [(edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2] if edges is not None else None
        _compare(edges, pred["edge_errors_processed_xyxy"], f"forensic/P{index}/edge errors")
        _compare(offsets, pred["center_offset_processed_xy"], f"forensic/P{index}/center offset")
    _compare(matrix, raw["box_iou_matrix"], "forensic full IoU matrix")
    matches = _match_box_iou(np.asarray(matrix).reshape(len(objects), known["gt_count"]))
    expected_matches = [{"prediction_id": f"P{p + 1}", "gt_id": known["instances"][g]["display_id"],
                         "box_iou": matrix[p][g]} for p, g in matches]
    _compare(expected_matches, raw["box_matches"], "forensic strict box matches")
    _same_number(len(matches), raw["box_true_positive_count"], "forensic match count")
    _same_number(len(matches) / known["gt_count"], raw["box_recall"], "forensic box recall")
    _same_number(abs(len(objects) - known["gt_count"]), raw["absolute_count_error"], "raw count error")
    return {"objects": objects, "analysis": raw, "entry_validity": entry_validity,
            "audit_sha256": _sha(path), "strict_accepted_count": result["analysis"]["prediction_count"]}


def _draw_panel(canvas, xy, title, subtitle, source, known, gt, *, mode):
    ox, oy = xy
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((ox, oy, ox + 470, oy + 570), radius=14, fill="white", outline="#dce3eb", width=2)
    draw.text((ox + 20, oy + 16), title, font=_font(24), fill=INK)
    draw.text((ox + 20, oy + 51), subtitle, font=_font(16), fill=MUTED)
    pixels = np.asarray(source).copy()
    if mode != "input":
        for instance in known["instances"]:
            pixels[_boundary(gt == instance["instance_id"])] = [0, 188, 168]
    left, top, size = ox + 60, oy + 102, 360
    canvas.paste(Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST), (left, top))
    draw.rectangle((left, top, left + size, top + size), outline="#bcc7d5", width=1)
    if mode != "input":
        for tick in (0, 56, 112, 168, 224):
            offset = tick * size / 224
            draw.line((left + offset, top + size, left + offset, top + size + 6), fill=INK)
            draw.text((left + offset, top + size + 10), str(tick), anchor="mt", font=_font(15), fill=MUTED)
            draw.line((left - 6, top + offset, left, top + offset), fill=INK)
            draw.text((left - 10, top + offset), str(tick), anchor="rm", font=_font(15), fill=MUTED)
        draw.text((left + size + 20, top + size - 4), "x", font=_font(18), fill=INK)
        draw.text((left - 30, top - 30), "y", font=_font(18), fill=INK)
    if mode == "gt":
        for instance in known["instances"]:
            points = np.argwhere(gt == instance["instance_id"])
            y, x = points[np.argmin(((points - points.mean(axis=0)) ** 2).sum(axis=1))]
            draw.text((left + (x + .5) * size / 128, top + (y + .5) * size / 128), instance["display_id"],
                      anchor="mm", font=_font(20), fill="white", stroke_width=2, stroke_fill=INK)
    footer = "Original pixels; no annotations drawn on model input." if mode == "input" else "224-coordinate frame; origin at upper left."
    draw.text((ox + 20, oy + 527), _fit_text(draw, footer, _font(16), 430), font=_font(16), fill=MUTED)
    return left, top, size


def _draw_objects(canvas, panel, objects, entry_validity=None):
    draw = ImageDraw.Draw(canvas)
    left, top, size = panel
    used_labels = []
    for index, obj in enumerate(objects, 1):
        valid = entry_validity[index - 1] if entry_validity is not None else True
        color = PRED_COLOR if valid else "#c6283d"
        x0, y0, x1, y1 = obj["box"]
        draw.rectangle((left + x0 * size / 224, top + y0 * size / 224,
                        left + x1 * size / 224, top + y1 * size / 224), outline=color, width=3)
        x, y = left + obj["point"][0] * size / 224, top + obj["point"][1] * size / 224
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=INK)
        label = None
        for dx, dy in ((0, -12), (0, 26), (-30, -12), (30, 26), (-30, 26), (30, -12), (0, -38), (0, 52)):
            lx = min(left + size - 14, max(left + 14, x + dx))
            ly = min(top + size - 3, max(top + 20, y + dy))
            bounds = (lx - 14, ly - 21, lx + 14, ly + 2)
            if not any(bounds[0] < prior[2] and bounds[2] > prior[0]
                       and bounds[1] < prior[3] and bounds[3] > prior[1] for prior in used_labels):
                label = (lx, ly)
                used_labels.append(bounds)
                break
        if label is None:
            label = (min(left + size - 14, max(left + 14, x)), max(top + 20, y - 12))
        draw.text(label, f"P{index}" + ("!" if not valid else ""),
                  anchor="mb", font=_font(18), fill="white", stroke_width=2, stroke_fill=INK)


def _point_coverage(analysis):
    return sorted({prediction["point_gt_id"] for prediction in analysis["predictions"] if prediction["point_gt_id"] is not None})


def _verdict(result, analysis):
    count, gt = analysis["prediction_count"], analysis["gt_count"]
    objects = "object" if count == 1 else "objects"
    if not result["format_valid"]:
        return "The output failed the format check; no valid objects can be scored."
    if count < gt:
        return f"Qwen returned {count} {objects} for {gt} annotated nuclei; object count is insufficient."
    if count > gt:
        return f"Qwen returned {count} {objects} for {gt} annotated nuclei; check surplus locations and duplicates."
    if analysis["box_true_positive_count"] < gt:
        return "The object count matches GT, but not every nucleus has a matching box."
    if len(_point_coverage(analysis)) < gt:
        return "All nuclei have matching boxes; their foreground points still miss some annotations."
    return "Every nucleus has a matching box and is covered by a foreground point in this one crop."


def _shift_text(analysis):
    shift = analysis["translation_diagnostic"]
    if shift["status"] == "insufficient_associations":
        return f"Common shift: not estimated; only {shift['association_count']} eligible positive-overlap associations."
    return (f"Association mean offset: {_vec(shift['mean_offset_processed_xy'])} px; residual RMS: "
            f"{shift['rms_residual_processed']:.2f} px. This does not establish a global shift.")


def render_figure(result, image, gt, known, analysis, destination, forensic=None):
    strict = analysis
    if forensic:
        analysis = forensic["analysis"]
    canvas = Image.new("RGB", (1470, 1060 if forensic else 1010), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    title = "Raw Qwen output and strict parser acceptance" if forensic else "Few-nucleus localization: count, points and boxes"
    draw.text((24, 20), title, font=_font(34), fill=INK)
    draw.text((24, 67), "Saved Qwen 32B inference | Same prompt as the single-nucleus probe | No SAM | No training", font=_font(21), fill=MUTED)
    _draw_panel(canvas, (18, 118), "A  Actual input crop", "Native 128 x 128 real tissue image", image, known, gt, mode="input")
    _draw_panel(canvas, (500, 118), "B  Annotated nuclei", "Cyan outlines; G labels identify GT instances", image, known, gt, mode="gt")
    panel = _draw_panel(canvas, (982, 118), "C  Raw boxes + points" if forensic else "C  Qwen boxes + points",
                        "All emitted coordinates; red = invalid entry" if forensic else "Orange P labels preserve completion order", image, known, gt, mode="pred")
    _draw_objects(canvas, panel, forensic["objects"] if forensic else result["raw_objects"] if result["format_valid"] else [],
                  forensic["entry_validity"] if forensic else None)
    if forensic:
        invalid = [f"P{i + 1}" for i, valid in enumerate(forensic["entry_validity"]) if not valid]
        # Negative coordinates remain negative. The figure has margin outside the
        # image, so an invalid edge is visible rather than silently clipped.
        if invalid:
            descriptions = []
            for i, obj in enumerate(forensic["objects"], 1):
                if obj["box"][0] < 0:
                    descriptions.append(f"P{i}: x0 = {obj['box'][0]:g}, outside image")
            note = "; ".join(descriptions) or "Out-of-bounds entries: " + ", ".join(invalid)
            draw.text((1002, 618), _fit_text(draw, note, _font(16), 430), font=_font(16), fill="#c6283d")
        draw.text((24, 714),
                  f"{len(forensic['objects'])} objects emitted for {known['gt_count']} GT nuclei; the whole answer is rejected because of {', '.join(invalid)}.",
                  font=_font(23), fill=INK)
        draw.text((24, 757),
                  f"RAW OUTPUT ONLY: count error {analysis['absolute_count_error']} | Box matches {analysis['box_true_positive_count']}/{known['gt_count']} "
                  f"(IoU > 0.5) | GT covered by points {len(_point_coverage(analysis))}/{known['gt_count']}", font=_font(21), fill=INK)
        draw.text((24, 798),
                  f"STRICT PIPELINE: accepted objects {strict['prediction_count']} | Count error {strict['absolute_count_error']} | "
                  f"Format invalid | No entries are salvaged", font=_font(21), fill="#c6283d")
        draw.text((24, 839),
                  f"Raw duplicate extras: {analysis['point_duplicate_count']} by point / {analysis['best_box_duplicate_count']} by box | "
                  f"Overlapping predicted-box pairs: {analysis['predicted_box_overlap_pair_count']}", font=_font(21), fill=MUTED)
        draw.text((24, 880), _fit_text(draw, _shift_text(analysis), _font(20), 1422), font=_font(20), fill=MUTED)
        draw.text((24, 934), "The raw view is forensic inspection of a rejected answer, not accepted pipeline performance. No coordinates were repaired or clipped.", font=_font(19), fill=MUTED)
        draw.text((24, 965), "Positive-overlap associations describe geometry only; zero-overlap boxes remain unassigned. Box recall is not mask PQ.", font=_font(19), fill=MUTED)
        draw.text((24, 996), "No SAM or training was run. This single selected training crop is a diagnostic, not a benchmark.", font=_font(19), fill=MUTED)
        canvas.save(destination)
        return
    draw.text((24, 714), _fit_text(draw, _verdict(result, analysis), _font(23), 1422), font=_font(23), fill=INK)
    coverage = _point_coverage(analysis)
    draw.text((24, 757),
              f"GT: {analysis['gt_count']} | Pred: {analysis['prediction_count']} | Count error: {analysis['absolute_count_error']} | "
              f"Box matches: {analysis['box_true_positive_count']}/{analysis['gt_count']} (IoU > 0.5) | "
              f"GT covered by points: {len(coverage)}/{analysis['gt_count']}", font=_font(21), fill=INK)
    draw.text((24, 798),
              f"Duplicate extras: {analysis['point_duplicate_count']} by point / {analysis['best_box_duplicate_count']} by box | "
              f"Overlapping predicted-box pairs: {analysis['predicted_box_overlap_pair_count']} (proxy only)", font=_font(21), fill=MUTED)
    draw.text((24, 839), _fit_text(draw, _shift_text(analysis), _font(21), 1422), font=_font(21), fill=MUTED)
    draw.text((24, 894), "Points are checked against the actual GT foreground. Correct counts or overlapping boxes alone do not establish detection.", font=_font(19), fill=MUTED)
    draw.text((24, 925), "Box matching is one-to-one at strict box IoU > 0.5. No zero-overlap box is assigned a GT identity for offset analysis.", font=_font(19), fill=MUTED)
    draw.text((24, 956), "No mask prediction was generated. Box recall is not mask PQ; this selected training crop is not a benchmark.", font=_font(19), fill=MUTED)
    canvas.save(destination)


def _gt_label(value):
    return value if value is not None else "None"


def _optional_vec(value):
    return _vec(value) if value is not None else "—"


def _summary_table(previous_analysis, analysis):
    values = (("Annotated nuclei", "gt_count"), ("Returned objects", "prediction_count"),
              ("Absolute count error", "absolute_count_error"), ("Box matches at IoU > 0.5", "box_true_positive_count"),
              ("Box recall", "box_recall"), ("GT instances covered by points", None),
              ("Point duplicate extras", "point_duplicate_count"), ("Best-box duplicate extras", "best_box_duplicate_count"),
              ("Predicted-box overlap pairs (proxy)", "predicted_box_overlap_pair_count"))
    rows = []
    for label, key in values:
        cells = []
        for item in (previous_analysis, analysis):
            if key is None:
                value = f"{len(_point_coverage(item))} / {item['gt_count']}"
            elif key == "box_recall":
                value = f"{item[key]:.2%}"
            else:
                value = str(item[key])
            cells.append(f"<td>{value}</td>")
        rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Metric</th><th>Earlier single-nucleus crop</th><th>This few-nucleus crop</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def render_html(result, known, analysis, previous, single_analysis, destination, forensic=None):
    strict = analysis
    if forensic:
        analysis = forensic["analysis"]
    e = lambda value: html.escape(str(value), quote=True)
    matched = {match["prediction_id"]: match["gt_id"] for match in analysis["box_matches"]}
    rows = []
    for pred in analysis["predictions"]:
        best = f"{_gt_label(pred['best_box_gt_id'])} ({pred['best_box_iou']:.4f})"
        if pred["best_box_iou_tied"]:
            best += " — tied best"
        point = pred["point_gt_id"] or ("Background" if pred["point_status"] == "background" else "No point")
        contract = ("Valid entry; whole answer rejected" if pred.get("per_entry_contract_valid") else "INVALID: " + "; ".join(pred.get("contract_violations", []))) if forensic else "Accepted"
        rows.append(f"<tr><th>{e(pred['display_id'])}</th><td>{e(contract)}</td><td>{e(_vec(pred['raw_box']))}<br>{e(_optional_vec(pred['raw_point']))}</td>"
                    f"<td>{e(_vec(pred['original_box']))}<br>{e(_optional_vec(pred['original_point']))}</td>"
                    f"<td>{e(point)}</td><td>{e(best)}</td><td>{e(_gt_label(matched.get(pred['display_id'])))}</td>"
                    f"<td>{e(_optional_vec(pred['center_offset_processed_xy']))}</td>"
                    f"<td>{e(_optional_vec(pred['edge_errors_processed_xyxy']))}</td></tr>")
    if not rows:
        rows = ["<tr><td colspan='9'>No valid objects to score</td></tr>"]
    matrix_headers = "".join(f"<th>{e(label)}</th>" for label in analysis["box_iou_matrix_gt_ids"])
    matrix_rows = "".join(f"<tr><th>{e(label)}</th>" + "".join(f"<td>{value:.4f}</td>" for value in row) + "</tr>"
                          for label, row in zip(analysis["box_iou_matrix_prediction_ids"], analysis["box_iou_matrix"]))
    if not matrix_rows:
        matrix_rows = f"<tr><td colspan='{known['gt_count'] + 1}'>No returned boxes</td></tr>"
    gt_rows = "".join(f"<tr><th>{e(item['display_id'])}</th><td>{item['instance_id']}</td>"
                      f"<td>{e(_vec(item['original_box']))}</td><td>{e(_vec(item['processed_box']))}</td><td>{item['area']}</td></tr>"
                      for item in known["instances"])
    coverage = _point_coverage(analysis)
    background = sum(pred["point_status"] == "background" for pred in analysis["predictions"])
    shift = analysis["translation_diagnostic"]
    if shift["status"] == "insufficient_associations":
        shift_description = (f"Only {shift['association_count']} eligible positive-overlap associations are available, fewer than two, so <b>no common shift is estimated</b>. "
                             "Individual offsets can be inspected against associated GT boxes, but one box offset cannot establish a global shift.")
    else:
        shift_description = (f"{shift['association_count']} eligible associations give a mean center offset of "
                             f"<b>{e(_vec(shift['mean_offset_processed_xy']))}</b> pixels in the 224-coordinate frame; "
                             f"after subtracting that mean offset, residual RMS is <b>{shift['rms_residual_processed']:.2f}</b> pixels. "
                             "These associations require unique positive box IoU, which may be far below 0.5. <b>They neither prove correct detection nor independently establish a uniform model shift.</b>")
    association_rows = "".join(f"<tr><td>{e(item['prediction_id'])} → {e(item['gt_id'])}</td><td>{item['box_iou']:.4f}</td>"
                               f"<td>{e(_vec(item['center_offset_processed_xy']))}</td></tr>" for item in shift["associations"])
    if not association_rows:
        association_rows = "<tr><td colspan='3'>No eligible association</td></tr>"
    duplicate_json = {key: analysis[key] for key in ("point_duplicate_groups", "best_box_duplicate_groups", "predicted_box_overlap_pairs")}
    if forensic:
        invalid_names = [f"P{i + 1}" for i, valid in enumerate(forensic["entry_validity"]) if not valid]
        title = f"{len(forensic['objects'])} objects emitted; {strict['prediction_count']} accepted by the strict pipeline."
        intro = (f"The raw answer contains <b>{len(forensic['objects'])}</b> box-and-point pairs; GT contains <b>{known['gt_count']}</b> nuclei. "
                 f"<b>{e(', '.join(invalid_names))}</b> has out-of-bounds coordinates, causing the strict parser to reject the whole answer. The pipeline therefore accepts "
                 f"<b>{strict['prediction_count']}</b>. <b>This does not mean the model emitted zero objects.</b>")
        forensic_warning = (f"<section style='border-left:5px solid #c6283d'><h2>Keep raw and accepted results separate</h2>"
                            f"<p><b>Raw-output inspection:</b>{len(forensic['objects'])} objects, with raw absolute count error {analysis['absolute_count_error']}; all coordinates, including negative values, are retained. "
                            "The boxes, points, IoU matrix, duplicate counts, and offset diagnostics below inspect the rejected raw answer; they are not accepted pipeline predictions.</p>"
                            f"<p><b>Strict pipeline:</b>format_valid = false, accepted objects = {strict['prediction_count']}, "
                            f"count error = {strict['absolute_count_error']}, box recall = {strict['box_recall']:.2%}. "
                            "The original result.json, parser, and saved metrics are unchanged. Coordinates are not clipped and remaining entries are not selectively salvaged.</p>"
                            f"<p>Parser reason: {e(result['parse_error'])}</p></section>")
        analysis_scope = "The following geometry diagnostics describe rejected raw output, not the strict pipeline's accepted results."
        comparison_scope = "The current column uses raw geometry from a rejected answer; the earlier single-nucleus answer was valid. Current raw objects are not accepted pipeline objects. Strict results are listed separately above."
    else:
        title = _verdict(result, analysis)
        intro = f"GT contains <b>{known['gt_count']}</b> complete annotated nuclei, and Qwen returns <b>{analysis['prediction_count']}</b> valid objects."
        forensic_warning, analysis_scope, comparison_scope = "", "", ""
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Few-nucleus localization probe</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#172438;background:#eef3f8;max-width:1260px;margin:28px auto;padding:0 20px}}h1{{line-height:1.25}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:16px}}small{{color:#526279}}a{{color:#075da8}}</style>
<h1>Few-nucleus localization probe<br>Separate count, location, and duplication</h1>
<section><b>{e(title)}</b><p>{intro}
One-to-one box matching at IoU > 0.5 matches <b>{analysis['box_true_positive_count']}/{known['gt_count']}</b> GT boxes; returned points cover <b>{len(coverage)}/{known['gt_count']}</b> nuclei. </p>
<p>Format valid: {e(result['format_valid'])} · Parse error: {e(result['parse_error'] or 'None')}. Invalid answers have an empty accepted-object list, and that failure is preserved.</p>
<p>This is a single inference call on a preselected image. It uses <b>exactly the same initial prompt, checkpoint, coordinate convention, and decoding settings as the single-nucleus test</b>; only the real image changes. There is no training, SAM call, or manual output correction.</p></section>
{forensic_warning}
<a href="multi_nucleus_probe.png"><img src="multi_nucleus_probe.png" alt="Actual tissue crop, all annotated nuclei, and every Qwen box and point"></a>
<p><a href="multi_nucleus_probe.png">Open / save full comparison</a> · <a href="input.png">Original input image</a> · <a href="gt_mask.png">GT instance mask</a> · <a href="verification.json">Recomputed verification</a></p>
<section><h2>Four separate checks</h2><p><b>{analysis_scope}</b></p><ol>
<li><b>Count:</b> Predicted {analysis['prediction_count']}; annotated {known['gt_count']}; absolute count error {analysis['absolute_count_error']}. Equal counts do not establish correct locations, and a smaller count does not identify which nucleus was missed.</li>
<li><b>Localization:</b> Strict box matches cover {analysis['box_true_positive_count']}/{known['gt_count']}; GT objects without such a match are {e(', '.join(analysis['missed_gt_ids']) or 'none')}. This means no one-to-one box match meets the threshold; it does not establish that the model never perceived the object. Points cover {e(', '.join(coverage) or 'none')}; another {background} points fall on GT background.</li>
<li><b>Duplicates:</b> Extra points within the same actual nucleus: {analysis['point_duplicate_count']}; extra boxes sharing the same best GT with box IoU > 0.5: {analysis['best_box_duplicate_count']}. Zero duplicates can occur when all boxes and points miss; it does not necessarily indicate good performance.</li>
<li><b>Common shift:</b>{shift_description}</li></ol></section>
<section><h2>Every output: full coordinates and matches, without cherry-picking</h2><div class="scroll"><table><tr><th>Object</th><th>Coordinate contract</th><th>Raw box / point (224)</th><th>Converted box / point (128)</th><th>Point hits actual GT</th><th>Best positive-overlap GT (box IoU)</th><th>One-to-one box match (IoU &gt; 0.5)</th><th>Center offset [dx,dy] (224)</th><th>Edge errors [x0,y0,x1,y1] (224)</th></tr>{''.join(rows)}</table></div>
<p><b>Best positive overlap</b> identifies the GT box with the largest positive overlap, even if IoU is very small. Zero-overlap boxes are unassigned (None), with no center offset. Tied maxima are marked and excluded from common-shift calculations.</p>
<p>Center offset is the predicted box center minus the GT <b>bounding-box center</b>, not the nucleus pixel centroid. It differs from point membership. The origin is top-left, x increases rightward, y downward. Point membership uses GT[floor(y),floor(x)] and requires actual foreground, not just box containment.</p></section>
<section><h2>Full box-IoU matrix</h2><div class="scroll"><table><tr><th>Predicted box</th>{matrix_headers}</tr>{matrix_rows}</table></div>
<p>Strict matching uses one-to-one box IoU > 0.5, maximizing match count first and summed IoU second. Positive overlaps at or below 0.5 remain visible but do not count as successful matches. <b>This is box recall, not mask PQ.</b></p></section>
<section><h2>Which associations enter the offset diagnostic?</h2><p>{shift_description}</p><div class="scroll"><table><tr><th>Diagnostic association</th><th>Box IoU</th><th>Center offset [dx,dy] (224)</th></tr>{association_rows}</table></div>
<p>Rule: each box chooses a unique best positive-IoU GT, excluding ties and contested GTs. At least two associations are required to calculate mean offset and residual RMS. These describe geometric error, do not replace strict matches, and do not move predictions to improve scores.</p>
<details><summary>Individual duplicate and overlap groups</summary><pre>{e(json.dumps(duplicate_json,ensure_ascii=False,indent=2))}</pre></details>
<p>Predicted-box overlap pairs count pairs with mutual IoU > 0.5. This is only a duplication proxy: overlapping boxes do not prove they refer to the same nucleus.</p></section>
<section><h2>Geometry comparison with the earlier single-nucleus case</h2><p><b>{comparison_scope}</b></p>{_summary_table(single_analysis, analysis)}<p>The images and GT counts differ. This is an incremental diagnostic, not a representative sample or a generalization comparison. Both use the same box-IoU, foreground-point, and duplicate definitions. Out-of-bounds raw boxes remain unchanged; the independent audit uses their complete, unclipped areas.</p></section>
<section><h2>Exact initial prompt (same as the single-nucleus test)</h2><pre>{e(json.dumps(result['prompt'],ensure_ascii=False,indent=2))}</pre><h2>Raw completion</h2><pre>{e(result['text'])}</pre></section>
<section><h2>Annotation geometry</h2><table><tr><th>Display label</th><th>Original instance ID</th><th>Tight box (128)</th><th>Tight box (224)</th><th>Foreground pixels</th></tr>{gt_rows}</table>
<p>G1-style labels are display IDs; P1-style labels preserve generation order. Matching numbers do not imply a match. Panels B/C reuse the original image with axes, not reconstructed normalized tensors. The input PNG contains no annotation outlines or display labels.</p></section>
<section><h2>Scope and provenance</h2><p>The native 128×128 crop comes from the official 30 training-source images. Selection used GT and required complete nuclei away from the edges. This is a selected engineering diagnostic, <b>not an independent test set, full-image evaluation, or clinical accuracy estimate</b>. SAM was not run, so segmentation-mask IoU and PQ are not reported.</p>
<details><summary>Complete source provenance</summary><pre>{e(json.dumps(result['source']['selection'],ensure_ascii=False,indent=2))}</pre></details><small>Model: {e(result['model']['model'])}<br>Revision: {e(result['model']['resolved_revision'])}<br>
Input SHA-256: {e(result['source']['image_sha256'])}<br>GT SHA-256: {e(result['source']['mask_sha256'])}<br>Completion tokens: {result['completion_tokens']} · Token cap reached: {e(result['hit_token_cap'])}</small>
<details><summary>Generation controls</summary><pre>{e(json.dumps(result['generation_controls'],ensure_ascii=False,indent=2))}</pre></details></section></html>"""
    destination.write_text(document, encoding="utf-8")


def render_report(run, single_run, output):
    run, single_run, output = [Path(path).resolve() for path in (run, single_run, output)]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty output directory; previous reports are never overwritten")
    result, image, gt, known, analysis, previous, single_analysis = verify_probe(run, single_run)
    forensic = verify_raw_forensic(run, result, known, gt) if (run / "independent_audit.json").exists() else None
    if not result["format_valid"] and forensic is None and '"objects"' in result["text"]:
        raise ValueError("A rejected structured answer needs its independent raw-output audit before rendering")
    output.mkdir(parents=True, exist_ok=True)
    render_figure(result, image, gt, known, analysis, output / "multi_nucleus_probe.png", forensic=forensic)
    render_html(result, known, analysis, previous, single_analysis, output / "index.html", forensic=forensic)
    for filename in ("input.png", "gt_mask.png"):
        (output / filename).write_bytes((run / filename).read_bytes())
    verification = {"image_and_gt_hashes_verified": True, "official30_training_source_verified": True,
                    "two_to_four_gt_instances_without_border_contact": True,
                    "gt_geometry_recomputed": True, "raw_completion_reparsed": True,
                    "all_analysis_fields_recomputed": True, "same_prompt_checkpoint_decoding_as_single": True,
                    "no_zero_overlap_association_for_offset": True, "sam_used": False,
                    "source_result_sha256": _sha(run / "result.json"),
                    "single_result_sha256": _sha(single_run / "result.json"),
                    "format_valid": result["format_valid"], "analysis": analysis,
                    "single_analysis_recomputed": single_analysis}
    if forensic:
        verification.update(raw_forensic_analysis=forensic["analysis"],
                            raw_emitted_object_count=len(forensic["objects"]),
                            strict_accepted_object_count=analysis["prediction_count"],
                            independent_audit_sha256=forensic["audit_sha256"],
                            raw_json_coordinates_and_geometry_crosschecked=True,
                            raw_coordinates_clipped_or_repaired=False)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    return {"report": str(output / "index.html"), "figure": str(output / "multi_nucleus_probe.png"),
            "gt_count": known["gt_count"], "prediction_count": analysis["prediction_count"],
            "raw_emitted_object_count": len(forensic["objects"]) if forensic else analysis["prediction_count"],
            "box_recall": analysis["box_recall"], "format_valid": result["format_valid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--single-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(render_report(args.run, args.single_run, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Cannot render multi-nucleus probe: {error}") from error
