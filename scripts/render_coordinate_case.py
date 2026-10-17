"""Audit and visualize one saved Qwen spatial answer; no model or GPU needed.

The coordinate panels reuse the source image to illustrate coordinate frames.
They are not reconstructions of the normalized Qwen input tensor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import parse_completion
from render_prompt_ablation import _font, _sha256, load_verified_cases

GT_COLOR = (0, 225, 205)
PRED_COLOR = (255, 171, 30)
INK = "#172438"
MUTED = "#526279"


def boundary(mask):
    padded = np.pad(mask, 1)
    inside = (padded[:-2, 1:-1] & padded[2:, 1:-1]
              & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~inside


def gt_geometry(reference):
    rows = []
    for display_id, instance_id in enumerate(np.unique(reference[reference > 0]), 1):
        positions = np.argwhere(reference == instance_id)
        y, x = positions[np.argmin(((positions - positions.mean(axis=0)) ** 2).sum(axis=1))]
        ymin, xmin = positions.min(axis=0)
        ymax, xmax = positions.max(axis=0) + 1
        rows.append({"display_id": f"G{display_id}", "instance_id": int(instance_id),
                     "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                     "interior_point": [int(x), int(y)], "area": len(positions)})
    return rows


def diagnose(case, run, variant):
    record = case["arms"][variant]["record"]
    provenance = json.loads((run / "result.json").read_text())
    if (record["id"] != "TCGA-21-5784-01Z-00-DX1_x512_y640" or variant != "one_example"
            or provenance["model"]["model"] != "Qwen/Qwen2.5-VL-32B-Instruct"
            or provenance["model"]["adapter"] is not None
            or provenance["sam"]["checkpoint_sha256"] != "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"):
        raise ValueError("This explanatory narrative is specific to the 32B/tiny six-object failure")
    reference = case["reference"]
    prediction = case["arms"][variant]["prediction"]
    width, height = record["prompt_size"]
    parsed = parse_completion(record["text"], width=width, height=height)
    if not parsed.valid:
        raise ValueError(f"Selected answer is invalid: {parsed.error}")
    converted, scales = objects_to_original(parsed.objects, (width, height), case["image"].size)
    for actual, saved in ((parsed.objects, record["raw_objects"]),
                          (converted, record["converted_objects"])):
        if len(actual) != len(saved):
            raise ValueError("Object count differs from saved answer")
        for obj, entry in zip(actual, saved):
            for key in ("box", "point"):
                np.testing.assert_allclose(getattr(obj, key), entry[key], rtol=0, atol=1e-10)
    np.testing.assert_allclose(scales, record["scale_to_original"], rtol=0, atol=1e-12)
    raw_path = run / variant / Path(record["raw_masks_path"]).name
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = archive["masks"].astype(bool)
    if raw.shape != (len(converted), *reference.shape):
        raise ValueError("Raw masks do not match the saved image/object count")
    geometry = gt_geometry(reference)
    yy, xx = np.indices(reference.shape)
    objects = []
    for number, (source, obj, mask) in enumerate(zip(parsed.objects, converted, raw), 1):
        x, y = obj.point
        x0, y0, x1, y1 = obj.box
        in_box = ((xx + .5 >= x0) & (xx + .5 < x1)
                  & (yy + .5 >= y0) & (yy + .5 < y1))
        foreground_intersection = int(np.count_nonzero(mask & (reference > 0)))
        ious = []
        for item in geometry:
            gt_mask = reference == item["instance_id"]
            ious.append(float(np.count_nonzero(mask & gt_mask) / np.count_nonzero(mask | gt_mask)))
        objects.append({
            "number": number, "raw": asdict(source), "original": asdict(obj),
            "point_gt_id_floor_xy": int(reference[int(np.floor(y)), int(np.floor(x))]),
            "box_gt_pixels_by_pixel_center": int(np.count_nonzero(in_box & (reference > 0))),
            "raw_sam_area": int(mask.sum()), "raw_sam_gt_intersection": foreground_intersection,
            "raw_sam_best_gt_iou": max(ious, default=0.),
        })
    return {
        "id": record["id"], "variant": variant, "coordinate_frame": record["coordinate_frame"],
        "model": provenance["model"], "sam": provenance["sam"],
        "source_crop": case["selection"],
        "prompt_size": record["prompt_size"], "original_size": record["original_size"],
        "scale_to_original": list(scales), "conversion_matches_saved": True,
        "point_hit_convention": "GT[floor(y), floor(x)]; upper-left origin; x right, y down",
        "box_pixel_convention": "Pixel centers inside half-open XYXY box",
        "objects": objects, "ground_truth": geometry,
        "final_mask_gt_intersection": int(np.count_nonzero((prediction > 0) & (reference > 0))),
        "final_foreground_equals_raw_union": bool(np.array_equal(prediction > 0, raw.any(axis=0))),
        "metrics": {k: record[k] for k in ("gt_count", "pred_count", "tp", "fp", "fn", "pq",
                     "recall", "count_error", "same_gt_duplicate_count", "raw_overlap_pair_count")},
        "verification": "Input PNG hashes and final metrics verified against the saved paired run",
        "pixel_values_sha256": record["pixel_values_sha256"],
        "raw_masks_sha256": _sha256(raw_path),
    }


def annotation_panel(case, geometry, *, predicted=False):
    pixels = np.asarray(case["image"], dtype=float).copy()
    if predicted:
        labels = case["arms"][predicted]["prediction"]
        for instance_id in np.unique(labels[labels > 0]):
            mask = labels == instance_id
            pixels[mask] = .77 * pixels[mask] + .23 * np.array(PRED_COLOR)
            pixels[boundary(mask)] = PRED_COLOR
    for item in geometry:
        mask = case["reference"] == item["instance_id"]
        pixels[boundary(mask)] = GT_COLOR
    return Image.fromarray(np.uint8(pixels)).resize((512, 512), Image.Resampling.NEAREST)


def draw_panel(canvas, origin, title, subtitle, source, frame, *, objects=None, gt=None):
    ox, oy = origin
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((ox, oy, ox + 634, oy + 662), 16, fill="white", outline="#dce3eb", width=2)
    draw.text((ox + 24, oy + 18), title, font=_font(25), fill=INK)
    draw.text((ox + 24, oy + 53), subtitle, font=_font(17), fill=MUTED)
    left, top = ox + 68, oy + 100
    canvas.paste(source, (left, top))
    draw.rectangle((left, top, left + 512, top + 512), outline=INK, width=1)
    width, height = frame
    for index in range(5):
        offset = index * 128
        draw.line((left + offset, top + 512, left + offset, top + 519), fill=INK, width=2)
        draw.text((left + offset, top + 523), f"{width * index / 4:g}", anchor="mt", font=_font(16), fill=MUTED)
        draw.line((left - 7, top + offset, left, top + offset), fill=INK, width=2)
        draw.text((left - 12, top + offset), f"{height * index / 4:g}", anchor="rm", font=_font(16), fill=MUTED)
    draw.text((left + 550, top + 512), "x", font=_font(20), fill=INK)
    draw.text((left - 35, top - 37), "y", font=_font(20), fill=INK)
    if gt:
        for item in gt:
            x, y = item["interior_point"]
            draw.text((left + (x + .5) * 4, top + (y + .5) * 4), item["display_id"],
                      anchor="mm", font=_font(18), fill="white", stroke_width=2, stroke_fill=INK)
    if objects:
        for number, obj in enumerate(objects, 1):
            x0, y0, x1, y1 = obj["box"]
            px, py = obj["point"]
            box = (left + x0 / width * 512, top + y0 / height * 512,
                   left + x1 / width * 512, top + y1 / height * 512)
            draw.rectangle(box, outline=PRED_COLOR, width=3)
            x, y = left + px / width * 512, top + py / height * 512
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=PRED_COLOR, outline=INK, width=1)
            draw.text((x, y - 12), f"P{number}", anchor="mb", font=_font(18),
                      fill="white", stroke_width=2, stroke_fill=INK)


def render_figure(case, audit, variant, path):
    record = case["arms"][variant]["record"]
    canvas = Image.new("RGB", (1320, 1632), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 22), "Qwen 32B: six objects, wrong locations", font=_font(36), fill=INK)
    draw.text((26, 72), audit["id"] + "  |  one_example  |  saved inference", font=_font(20), fill=MUTED)
    draw.line((28, 123, 68, 123), fill=GT_COLOR, width=5)
    draw.text((80, 109), "Cyan: annotated nucleus boundary", font=_font(21), fill=INK)
    draw.line((580, 123, 620, 123), fill=PRED_COLOR, width=5)
    draw.text((632, 109), "Orange: generated prompt / SAM mask", font=_font(21), fill=INK)
    image = annotation_panel(case, audit["ground_truth"])
    draw_panel(canvas, (20, 160), "A  Ground truth", "6 annotated nuclei; G1-G6 are display labels.",
               image, (128, 128), gt=audit["ground_truth"])
    draw_panel(canvas, (666, 160), "B  Raw Qwen box + point", "224-coordinate frame: every point has y = 126.",
               image, (224, 224), objects=record["raw_objects"])
    draw_panel(canvas, (20, 836), "C  Prompts passed to SAM", "128-coordinate frame: every point has y = 72.",
               image, (128, 128), objects=record["converted_objects"])
    sam = annotation_panel(case, audit["ground_truth"], predicted=variant)
    draw_panel(canvas, (666, 836), "D  Saved SAM masks", "6 masks; none overlaps an annotated nucleus.", sam, (128, 128))
    draw.text((28, 1520), "Coordinate conversion: (158, 126) x 128/224 = (90.29, 72). Same position in B and C.",
              font=_font(23), fill=INK)
    draw.text((28, 1564), "Point hits: 0/6   |   PQ: 0   |   Recall: 0/6   |   Count error: 0   |   GT-matched duplicates: 0",
              font=_font(23), fill=INK)
    canvas.save(path)


def vec(value):
    return "[" + ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") if v % 1 else str(int(v)) for v in value) + "]"


def render_html(case, audit, output):
    record = case["arms"][audit["variant"]]["record"]
    rows = "".join(
        f"<tr><td>P{x['number']}</td><td>{vec(x['raw']['box'])}<br>{vec(x['raw']['point'])}</td>"
        f"<td>{vec(x['original']['box'])}<br>{vec(x['original']['point'])}</td>"
        f"<td>{x['point_gt_id_floor_xy']}</td><td>{x['box_gt_pixels_by_pixel_center']}</td>"
        f"<td>{x['raw_sam_best_gt_iou']:.3f}</td></tr>" for x in audit["objects"])
    gt_rows = "".join(f"<tr><td>{x['display_id']} (ID {x['instance_id']})</td><td>{vec(x['box'])}</td>"
                      f"<td>{vec(x['interior_point'])}</td></tr>" for x in audit["ground_truth"])
    escaped = html.escape
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>32B coordinate audit</title>
<style>body{{font-family:system-ui,sans-serif;background:#eef3f8;color:#172438;max-width:1160px;margin:32px auto;padding:0 22px;line-height:1.7}}h1{{line-height:1.25}}section{{background:white;border:1px solid #dce3eb;border-radius:16px;padding:22px;margin:20px 0}}img{{width:100%;height:auto}}a{{color:#075da8}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce3eb;text-align:left;vertical-align:top}}pre{{background:#f1f5f9;overflow:auto;padding:18px;font-size:14px;white-space:pre-wrap}}.tag{{color:#526279;font-size:14px}}.note{{border-left:5px solid #e0a127}}.scroll{{overflow-x:auto}}</style>
<h1>Qwen 32B: correct count, incorrect locations</h1>
<p class="tag">{escaped(audit['id'])} · one_example · Qwen2.5-VL-32B-Instruct → frozen SAM2.1 tiny</p>
<section><b>What does this example show?</b><p>There are 6 annotated nuclei. Qwen generated 6 box-and-point pairs, but placed all points on one horizontal line containing no annotated nuclei. All 6 points fall on background, and the 6 subsequent SAM masks have no overlap with the nuclei.</p>
<p><b>The localization error already exists before SAM is called.</b>The saved 224-to-128 coordinate conversion is correct. This case cannot distinguish failure to recognize nuclei, misunderstanding of coordinates, or generation of a patterned set of locations.</p>
<p>This explanatory case was selected from existing failures on training-source crops; it does not represent held-out performance. This page reuses saved text and masks without new inference, training, or manual prediction changes.</p></section>
<a href="coordinate_comparison.png"><img src="coordinate_comparison.png" alt="Ground truth, raw Qwen prompts, converted prompts, and saved SAM masks"></a>
<p><a href="coordinate_comparison.png">Open / save full figure</a> · <a href="audit.json">Numerical audit JSON</a> · <a href="completion.txt">Raw output</a> · <a href="source.png">Original 128×128 image</a></p>
<section><h2>How to read the figure</h2><ol>
<li><b>A:</b> Cyan lines show annotated nucleus outlines. G1–G6 are display labels, not predictions.</li>
<li><b>B:</b> Orange boxes and points are the raw Qwen output, interpreted in the requested 224×224 frame. P1–P6 preserve generation order. Every point has y=126 and every box has y-range [109,147).</li>
<li><b>C:</b> Coordinates are multiplied by 128/224 before SAM. Every point then has y=72 and every box has y-range [62.29,84). Panels B and C show identical locations with different axis units.</li>
<li><b>D:</b> The saved SAM masks are shown in orange. They do not overlap the cyan ground-truth outlines.</li></ol>
<p class="tag">Every panel uses the same 128×128 PNG enlarged with nearest-neighbor display. Panel B uses a 224-coordinate axis; it does not reconstruct Qwen's normalized input tensor. The origin is top-left, x increases rightward, and y increases downward.</p></section>
<section><h2>Actual generated boxes and points</h2><pre>{escaped(record['text'])}</pre>
<p>For example, P1: point [158,126] × 4/7 = [90.29,72]; box [137,109,187,147] × 4/7 = [78.29,62.29,106.86,84].</p>
<div class="scroll"><table><tr><th>Prompt</th><th>Raw box / point (224)</th><th>SAM box / point (128)</th><th>Point GT ID<br>0 = background</th><th>GT pixels inside box</th><th>Best raw-mask IoU</th></tr>{rows}</table></div>
<p class="tag">Points are checked at GT[floor(y),floor(x)]. Box coverage counts pixels whose centers lie inside the half-open box. P4 and P5 brush the same nucleus G6, containing 56 and 72 annotated pixels respectively, but their points remain on background and their final masks miss it. Thus, not every box is completely disjoint from every nucleus.</p></section>
<section><h2>Results for this crop</h2><p><b>GT = 6 · Pred = 6 · TP = 0 · FP = 6 · FN = 6 · PQ = 0 · Recall = 0/6 · Absolute count error = 0</b></p>
<p>Zero count error only means 6 − 6 = 0; it does not establish correct localization. Zero duplicates based on shared GT matches also does not imply good results: there are no successful matches here. The number of overlapping raw-mask pairs is also zero.</p>
<p>All 6 raw SAM masks are disjoint from GT, and the merged foreground equals their union. This failure therefore cannot be attributed to the subsequent overlap-resolution step.</p></section>
<section class="note"><h2>How could later diagnostics separate recognition and coordinate errors?</h2><ol>
<li><b>Coordinate calibration:</b> First ask the same Qwen model for a box and point on a synthetic image with one obvious target at a known location, checking the 224-coordinate convention.</li>
<li><b>Single-nucleus localization:</b> Then repeat on a real crop with one clear nucleus, controlling the prompt, coordinate convention, and segmenter configuration.</li>
<li><b>Multi-nucleus enumeration:</b> Finally restore several nuclei and check for missed objects, duplicates, or regular-grid coordinates beyond the first object.</li></ol>
<p>These were proposed follow-up diagnostics when this page was created; this page itself did not run them. Passing the first would not establish correct coordinate behavior on all real images.</p></section>
<details><summary>Show annotated nucleus locations (diagnostic only; not supplied to Qwen)</summary><section><table><tr><th>Display / original ID</th><th>GT tight box (128)</th><th>Interior point near centroid (128)</th></tr>{gt_rows}</table><p>Interior points use the foreground pixel nearest the centroid, not necessarily the geometric center. G1 and G4 touch the left crop edge and are counted under the existing evaluation rules.</p></section></details>
<details><summary>Show the exact initial prompt</summary><pre>{escaped(json.dumps(record['prompt'],ensure_ascii=False,indent=2))}</pre></details>
<section class="tag">Verification: source-image and GT hashes match the run manifest; metrics recomputed from the saved final instance map match the records; the generated text is reparsed and raw/converted coordinates are checked. This page does not rerun Qwen's processor or SAM, or claim to reconstruct their internal tensors.</section>
</html>"""
    (output / "index.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--variant", choices=("one_example", "three_examples"), default="one_example")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, cases, _ = load_verified_cases(args.run, args.dataset)
    case = next(item for item in cases if item["selection"]["id"] == args.id)
    audit = diagnose(case, args.run, args.variant)
    # This explanatory narrative is intentionally specific to the audited failure.
    assert audit["prompt_size"] == [224, 224] and audit["original_size"] == [128, 128]
    assert len(audit["objects"]) == 6 and audit["metrics"]["gt_count"] == 6
    assert all(obj["raw"]["point"][1] == 126 and obj["original"]["point"][1] == 72 for obj in audit["objects"])
    assert all(obj["point_gt_id_floor_xy"] == 0 and obj["raw_sam_gt_intersection"] == 0 for obj in audit["objects"])
    assert audit["final_mask_gt_intersection"] == 0 and audit["final_foreground_equals_raw_union"]
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve prior analysis")
    args.output.mkdir(parents=True)
    render_figure(case, audit, args.variant, args.output / "coordinate_comparison.png")
    render_html(case, audit, args.output)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "completion.txt").write_text(case["arms"][args.variant]["record"]["text"] + "\n")
    case["image"].save(args.output / "source.png")
    print(json.dumps({"report": str(args.output / "index.html"), "metrics": audit["metrics"],
                      "conversion_matches_saved": audit["conversion_matches_saved"]}, indent=2))


if __name__ == "__main__":
    main()
