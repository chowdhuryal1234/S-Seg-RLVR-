"""Verify and display two candidate-assisted tasks on one marked tissue image.

Reads saved outputs only. Candidate-label accuracy and square-coordinate errors
remain separate, and neither is described as unaided nucleus segmentation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_coordinate_calibration import INK, MUTED, _vec
from scripts.render_multi_nucleus_probe import _compare
from scripts.render_prompt_ablation import _font, _fit_text
from scripts.render_single_nucleus_probe import _boundary


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_run(run, reference_run):
    from nucleus_rl.recognition_geometry import analyze_recognition, analyze_geometry

    run, reference_run = Path(run).resolve(), Path(reference_run).resolve()
    result = json.loads((run / "result.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    reference = json.loads((reference_run / "result.json").read_text())
    if (result["status"] != "completed_recognition_coordinate_probe"
            or result.get("sam_used") is not False or result.get("optimizer_steps") != 0
            or result.get("policy_trainable_parameters") != 0):
        raise ValueError("Expected completed frozen inference with no SAM or optimizer steps")
    if _sha(reference_run / "result.json") != result["reference_result_sha256"]:
        raise ValueError("Original free-generation reference result differs")
    for key in ("model", "resolved_revision", "processor_revision", "adapter"):
        if result["model"][key] != reference["model"][key]:
            raise ValueError(f"Checkpoint differs from the reference: {key}")
    if result["generation_controls"] != reference["generation_controls"]:
        raise ValueError("Decoding settings differ from the reference")
    manifest = json.loads((run / "candidate_manifest.json").read_text())
    for name, key in (("input.png", "input_sha256"), ("original.png", "original_sha256"),
                      ("candidate_manifest.json", "manifest_sha256")):
        if _sha(run / name) != result[key] or result[key] != plan[key]:
            raise ValueError(f"Locked run hash differs: {name}")
    if _sha(run / "gt_mask.png") != plan["mask_sha256"] or plan["manifest"] != manifest:
        raise ValueError("GT or candidate manifest changed after protocol lock")
    if (manifest["status"] != "locked_pre_inference"
            or manifest["candidate_image_sha256"] != result["input_sha256"]
            or manifest["source_image_sha256"] != result["original_sha256"]
            or manifest["source_mask_sha256"] != _sha(run / "gt_mask.png")):
        raise ValueError("Candidate/source identities differ from the locked manifest")
    if (_sha(reference_run / "input.png") != result["original_sha256"]
            or _sha(reference_run / "gt_mask.png") != _sha(run / "gt_mask.png")):
        raise ValueError("The marked probe is not based on the exact reference image and GT")
    images = []
    for filename in ("original.png", "input.png"):
        with Image.open(run / filename) as opened:
            if opened.mode != "RGB" or opened.size != (128, 128):
                raise ValueError("Both source and marked images must be RGB 128 x 128")
            images.append(opened.copy())
    with Image.open(run / "gt_mask.png") as opened:
        gt = np.asarray(opened)
    if gt.shape != (128, 128) or len(np.unique(gt[gt > 0])) != 3:
        raise ValueError("Expected the unchanged three-nucleus GT")
    if result["prompt_size"] != [224, 224] or result["original_size"] != [128, 128]:
        raise ValueError("The image size or processed coordinate frame differs")
    candidates = manifest["candidates"]
    if [c["id"] for c in candidates] != list("ABCDEF"):
        raise ValueError("Expected six fixed candidates A through F")
    styles = []
    for c in candidates:
        x, y = c["point_original"]
        if not (isinstance(x, int) and isinstance(y, int) and 10 <= x <= 118 and 10 <= y <= 118):
            raise ValueError("Candidate center must be an interior original-image pixel")
        if c["box_original"] != [x - 10, y - 10, x + 10, y + 10]:
            raise ValueError("All candidates must use the same 20-pixel square")
        _compare([value * 1.75 for value in c["box_original"]], c["box_processed"], f"{c['id']}/box scale")
        _compare([x * 1.75, y * 1.75], c["point_processed"], f"{c['id']}/point scale")
        if bool(gt[y, x]) != c["is_nucleus"] or (int(gt[y, x]) or None) != c["gt_instance_id"]:
            raise ValueError("Candidate answer key differs from actual GT foreground")
        if not c["is_nucleus"]:
            x0, y0, x1, y1 = c["box_original"]
            if np.any(gt[y0:y1, x0:x1]):
                raise ValueError("A declared full-square-background candidate overlaps GT")
        styles.append(tuple(tuple(c[key]) if isinstance(c[key], list) else c[key] for key in
                            ("annotation_color_rgb", "square_size_native", "outline_width_native", "cross_radius_native")))
    if len(set(styles)) != 1 or sum(c["is_nucleus"] for c in candidates) != 3:
        raise ValueError("Candidate style or balanced answer-key control differs")
    if not result["same_image_tensor_across_tasks"]:
        raise ValueError("The two tasks must receive identical image tensors")
    pixel_arrays = []
    for task, scorer in (("recognition", analyze_recognition), ("geometry", analyze_geometry)):
        saved = result["tasks"][task]
        if saved["fresh_context"] is not True or saved["prompt"] != plan["prompts"][task]:
            raise ValueError(f"Task prompt or fresh-context control differs: {task}")
        recomputed = scorer(saved["text"], manifest)
        _compare(recomputed, saved["analysis"], f"{task} re-scoring")
        with np.load(run / f"{task}_processor_input.npz", allow_pickle=False) as archive:
            pixels = archive["pixel_values"]
            inputs = archive["input_ids"]
        if hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() != saved["pixel_values_sha256"]:
            raise ValueError(f"Saved image tensor hash differs: {task}")
        if hashlib.sha256(np.ascontiguousarray(inputs).tobytes()).hexdigest() != saved["input_ids_sha256"]:
            raise ValueError(f"Saved input-token hash differs: {task}")
        pixel_arrays.append(pixels)
    if not np.array_equal(*pixel_arrays):
        raise ValueError("Actual saved image tensors differ between tasks")
    return result, manifest, images[0], images[1], gt, reference


def _tile(canvas, origin, title, subtitle, image):
    x, y = origin
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + 430, y + 485), radius=14, fill="white", outline="#dce3eb", width=2)
    draw.text((x + 20, y + 18), title, font=_font(24), fill=INK)
    draw.text((x + 20, y + 54), subtitle, font=_font(15), fill=MUTED)
    left, top, size = x + 39, y + 100, 352
    canvas.paste(image.resize((size, size), Image.Resampling.NEAREST), (left, top))
    return left, top, size


def render_recognition_figure(result, manifest, original, marked, gt, destination):
    analysis = result["tasks"]["recognition"]["analysis"]
    selected = set(analysis["accepted_ids"])
    expected = {c["id"] for c in manifest["candidates"] if c["is_nucleus"]}
    canvas = Image.new("RGB", (1350, 820), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Same image: candidate recognition and coordinate readback", font=_font(30), fill=INK)
    draw.text((24, 66), "Qwen 32B | Both tasks use the identical marked image | Separate fresh contexts | No SAM or training", font=_font(20), fill=MUTED)
    _tile(canvas, (18, 112), "A  Earlier original image", "Used for the earlier free-localization task", original)
    _tile(canvas, (460, 112), "B  Exact shared input", "Six equally styled candidates; no answer key", marked)
    pixels = np.asarray(original).copy()
    pixels[_boundary(gt > 0)] = [0, 188, 168]
    left, top, size = _tile(canvas, (902, 112), "C  Recognition scoring", "Colored annotations below are post-hoc only", Image.fromarray(pixels))
    colors = {"tp": "#168a45", "fp": "#d32740", "fn": "#dd8500", "tn": "#69788a"}
    for c in manifest["candidates"]:
        kind = ("tp" if c["id"] in expected else "fp") if c["id"] in selected else ("fn" if c["id"] in expected else "tn")
        color = colors[kind]
        x0, y0, x1, y1 = c["box_original"]
        draw.rectangle((left + x0 * size / 128, top + y0 * size / 128,
                        left + x1 * size / 128, top + y1 * size / 128), outline=color, width=3)
        x, y = c["point_original"]
        draw.text((left + x * size / 128, top + y * size / 128), c["id"], anchor="mm",
                  font=_font(22), fill="white", stroke_width=3, stroke_fill=color)
    selection_text = ", ".join(sorted(selected)) or "none"
    key_text = ", ".join(sorted(expected))
    line = f"Recognition accepted IDs: {selection_text} | Expected IDs: {key_text} | Exact set correct: {analysis['exact_set_correct']}"
    draw.text((24, 626), _fit_text(draw, line, _font(22), 1302), font=_font(22), fill=INK)
    draw.text((24, 666), "Green = correct selection | Red = wrong selection | Orange = missed positive | Gray = correct negative", font=_font(19), fill=MUTED)
    draw.text((24, 710), "Recognition selects letter IDs whose centers lie in nuclei. Geometry reads ALL six drawn squares, not nucleus outlines.", font=_font(19), fill=MUTED)
    draw.text((24, 744), "Candidate marks and task prompts change the problem. Success here does not establish unaided nucleus detection.", font=_font(19), fill=MUTED)
    draw.text((24, 777), f"Recognition format valid: {analysis['format_valid']}. If rejected, accepted IDs are empty; the raw answer is retained in the report.", font=_font(18), fill=MUTED)
    canvas.save(destination)


def render_geometry_figure(result, manifest, marked, destination):
    analysis = result["tasks"]["geometry"]["analysis"]
    canvas = Image.new("RGB", (910, 780), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Coordinate readback: drawn squares, not nuclei", font=_font(29), fill=INK)
    draw.text((24, 65), "Cyan = supplied markers | Orange/red = unchanged numeric output", font=_font(21), fill=MUTED)
    _tile(canvas, (18, 110), "A  Exact marked input", "Known centers and square boundaries", marked)
    left, top, size = _tile(canvas, (460, 110), "B  Qwen coordinate output", "Arrows: known center to predicted center", marked)
    for row in analysis["per_region"]:
        if not row["box_metrics_available"] or not row["point_metrics_available"]:
            continue
        color = "#ef8a00" if row["per_entry_contract_valid"] else "#d32740"
        x0, y0, x1, y1 = [v * size / 224 for v in row["raw_box"]]
        draw.rectangle((left+x0,top+y0,left+x1,top+y1), outline=color, width=3)
        px, py = [v * size / 224 for v in row["raw_point"]]
        ex, ey = [v * size / 224 for v in row["expected_point_processed"]]
        draw.line((left+ex,top+ey,left+px,top+py), fill=color, width=2)
        draw.ellipse((left+px-4,top+py-4,left+px+4,top+py+4), fill=color, outline="white")
        draw.text((left+x0+3,top+y0+3), row["id"], font=_font(17), fill=color, stroke_width=1, stroke_fill="white")
    draw.text((24, 624), f"Mean center error: {analysis['mean_point_l2_processed']:.2f} / 224 pixels", font=_font(23), fill=INK)
    draw.text((24, 664), f"Mean drawn-box IoU: {analysis['mean_box_iou']:.3f} | Boxes above IoU 0.5: "
              f"{sum(r['box_iou'] is not None and r['box_iou'] > .5 for r in analysis['per_region'])}/6", font=_font(21), fill=INK)
    draw.text((24, 704), "E and F end at y = 235, beyond 224; the whole geometry answer is rejected.", font=_font(20), fill="#c51e3a")
    draw.text((24, 742), "All six raw entries shown without clipping or repair. No SAM, mask PQ or training.", font=_font(19), fill=MUTED)
    canvas.save(destination)


def render_report(run, reference_run, output):
    run, reference_run, output = [Path(v).resolve() for v in (run, reference_run, output)]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve previous reports; choose a new empty directory")
    result, manifest, original, marked, gt, reference = verify_run(run, reference_run)
    output.mkdir(parents=True, exist_ok=True)
    rec = result["tasks"]["recognition"]["analysis"]
    geo = result["tasks"]["geometry"]["analysis"]
    render_recognition_figure(result, manifest, original, marked, gt, output / "recognition_comparison.png")
    render_geometry_figure(result, manifest, marked, output / "geometry_comparison.png")
    e = html.escape
    recognition_rows = "".join(f"<tr><td>{c['id']}</td><td>{'Nucleus' if c['expected_positive'] else 'Background'}</td>"
                               f"<td>{'Selected' if c['predicted_positive'] else 'Not selected'}</td><td>{'Correct' if c['correct'] else 'Wrong'}</td></tr>"
                               for c in rec["per_candidate"])
    geometry_rows = "".join(f"<tr><td>{e(str(c['id']))}</td><td>{e(str(c['expected_box_processed']))}<br>{e(str(c['expected_point_processed']))}</td>"
                            f"<td>{e(str(c['raw_box']))}<br>{e(str(c['raw_point']))}</td><td>{c['box_iou']:.3f}</td>"
                            f"<td>{c['point_l2_processed']:.2f}</td><td>{e('; '.join(c['contract_errors']) or 'In bounds')}</td></tr>"
                            for c in geo["per_region"] if c["box_metrics_available"] and c["point_metrics_available"])
    raw_sections = "".join(f"<details><summary>{task}: exact prompt and raw answer</summary><pre>{e(json.dumps(t['prompt'],ensure_ascii=False,indent=2))}</pre>"
                           f"<pre>{e(t['text'])}</pre><p>Tokens: {t['completion_tokens']}; token cap: {t['hit_token_cap']}; fresh context: {t['fresh_context']}</p></details>"
                           for task,t in result["tasks"].items())
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen 32B: recognition versus coordinate readback</title><style>
body{{font:17px/1.6 system-ui,sans-serif;background:#f0f4f8;color:#1d2b40;margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}
section{{background:white;padding:24px;margin:20px 0;border-radius:14px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px}}
img{{max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce3eb;vertical-align:top}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px;font-size:13px}}.warning{{border-left:5px solid #d32740}}.muted{{color:#64748b}}
</style><main><h1>Recognition and coordinate readback<br>Two tasks on the same image</h1>
<p>Same Qwen2.5-VL-32B checkpoint; two fresh contexts; one completion per task; identical image tensors. No SAM or training.</p>
<section class="warning"><h2>Both tasks fail; coordinate output alone cannot explain the result</h2>
<p><b>Recognition:</b> The expected IDs are B, C, F; the model selects A, B, C, D, E. It selects two nucleus candidates, misses one, and incorrectly selects all three background candidates. Classification accuracy is 2/6 on these candidates, not dataset accuracy.</p>
<p><b>Geometry:</b> All six IDs are emitted, but the artificial squares have a mean box IoU of only {geo['mean_box_iou']:.3f} and mean center error {geo['mean_point_l2_processed']:.2f} processed pixels in the 224-coordinate frame, approximately {geo['mean_point_l2_processed']/1.75:.2f} original-image pixels. Out-of-bounds E/F coordinates cause the entire answer to be rejected.</p>
<p>Under this image and these prompts, both candidate/letter judgments and numeric localization are unreliable. This does not establish that the model first recognized the nuclei correctly and only then wrote wrong coordinates. One example also cannot establish that it has no nucleus-recognition ability.</p></section>
<section><h2>1. Recognition: choose IDs without generating coordinates</h2><img src="recognition_comparison.png" alt="Exact shared marked input and recognition mistakes">
<table><tr><th>ID</th><th>Ground-truth center label</th><th>Qwen selection</th><th>Decision</th></tr>{recognition_rows}</table>
<p>Nucleus-candidate recall is 2/3 and selection precision is 2/5. F is missed; A, D, E are extra selections. The output format is valid.</p></section>
<section><h2>2. Geometry: read every artificial square without classifying nuclei</h2><img src="geometry_comparison.png" alt="Actual drawn squares compared with every unmodified predicted coordinate">
<p>Values below use 224×224 coordinates: boxes are [x0,y0,x1,y1] and points [x,y]. IoU compares artificial squares, not nucleus masks.</p>
<table><tr><th>ID</th><th>Known box / center</th><th>Qwen box / point</th><th>Box IoU</th><th>Center error(px)</th><th>Format</th></tr>{geometry_rows}</table>
<p><b>Six raw entries; zero accepted by the strict parser.</b>E and F have y1=235, exceeding 224. Every unmodified coordinate is retained for error analysis, without clipping bounds or salvaging valid entries into the pipeline.</p>
<p>Predeclared engineering tolerance: six unique IDs, every box IoU > 0.5, and every center error ≤ 7 processed pixels. Here 0/6 entries satisfy it. This is not a medical or literature performance standard.</p></section>
<section><h2>Relationship to earlier free localization</h2><p>Earlier, the same unmarked image was used to request every nucleus with tight boxes and points. Five candidates were emitted, one out of bounds. This experiment supplies six candidate squares and changes the tasks. It is a location-assisted diagnostic, not evidence of improvement or decline on the same task.</p>
<p>The comparison does not fully isolate internal causes: recognition also requires matching letters to regions and judging center locations, while reading artificial squares differs from outlining real nuclei. A possible next check is to crop candidates separately and ask only nucleus/background, removing letter matching and multi-region correspondence.</p></section>
<section><h2>Controls and saved evidence</h2><p>Native 128×128 input, processed as 224×224; seed 42, greedy decoding, 512-token cap, the same image processor, and no LoRA. The two saved image tensors are elementwise identical, and neither output reaches the token cap. This is the same official-30 training-source crop used earlier, not an independent test set.</p>
<p>Markers supply candidate locations only. All six squares share size and style; GT masks, correct letters, true positive counts, and target coordinates are not supplied to the model.</p>
{raw_sections}<details><summary>Predeclared plan</summary><pre>{e((run/'plan.json').read_text())}</pre></details>
<p><a href="../result.json">Original result.json</a> · <a href="verification.json">Local verification</a> · <a href="../candidate_manifest.json">Candidate answer key</a></p></section></main></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    verification = {"status":"verified", "same_checkpoint_decoding_and_processor":True,
                    "same_marked_image_tensor_both_tasks":True, "candidate_key_recomputed_from_mask":True,
                    "raw_answers_rescored":True, "source_result_sha256":_sha(run/'result.json'),
                    "manifest_sha256":_sha(run/'candidate_manifest.json'), "no_coordinate_repair":True,
                    "recognition":rec, "geometry":geo}
    (output / "verification.json").write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"report":str(output/'index.html'),"recognition":str(output/'recognition_comparison.png'),
                      "geometry":str(output/'geometry_comparison.png')},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--reference-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    render_report(args.run,args.reference_run,args.output)
