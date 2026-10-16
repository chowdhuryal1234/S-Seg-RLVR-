"""Read-only report of six saved, unmarked binary-classification crops."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_prompt_ablation import _font


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run, reference_run):
    plan = json.loads((run / "plan.json").read_text())
    reference = json.loads((reference_run / "result.json").read_text())
    manifest = json.loads((reference_run / "candidate_manifest.json").read_text())
    original = np.asarray(Image.open(reference_run / "original.png").convert("RGB"))
    gt = np.asarray(Image.open(reference_run / "gt_mask.png"))
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else None
    for name, key in (("original.png", "source_image_sha256"), ("gt_mask.png", "source_mask_sha256"),
                      ("result.json", "reference_result_sha256"), ("candidate_manifest.json", "reference_manifest_sha256")):
        if sha(reference_run / name) != plan[key]:
            raise ValueError("Reference provenance differs from the locked plan")
    if plan["generation_controls"] != reference["generation_controls"]:
        raise ValueError("Decoding settings differ from the reference")
    if result is not None:
        if result["status"] != "completed_isolated_crop_probe" or result["sam_used"] or result["optimizer_steps"] != 0:
            raise ValueError("Expected completed inference without SAM or optimizer updates")
        if result["generation_controls"] != plan["generation_controls"] or result["prompt"] != plan["prompt"]:
            raise ValueError("Result differs from locked settings or prompt")
        for key in ("model", "resolved_revision", "processor_revision", "adapter"):
            if result["model"][key] != reference["model"][key]:
                raise ValueError("Model differs from the reference")
    planned = plan["records"]
    records = result["records"] if result else planned
    if [r["id"] for r in records] != list("ABCDEF") or [r["id"] for r in planned] != list("ABCDEF"):
        raise ValueError("All six fixed candidates must appear exactly once in order")
    images, masks, token_hashes = [], [], set()
    for record, locked, candidate in zip(records, planned, manifest["candidates"]):
        if any(record.get(key) != value for key, value in locked.items()):
            raise ValueError("Result record differs from the locked crop plan")
        image_path, mask_path = run / record["image_file"], run / record["mask_file"]
        if sha(image_path) != locked["image_sha256"] or sha(mask_path) != locked["mask_sha256"]:
            raise ValueError("Saved crop or mask hash differs")
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path))
        x0, y0, x1, y1 = candidate["box_original"]
        if image.size != (20, 20) or not np.array_equal(np.asarray(image), original[y0:y1, x0:x1]):
            raise ValueError("Input differs from the unmarked original crop")
        expected_mask = gt[y0:y1, x0:x1] > 0
        if mask.shape != (20, 20) or not np.array_equal(mask > 0, expected_mask):
            raise ValueError("Displayed binary mask differs from source GT")
        if locked["crop_box"] != candidate["box_original"] or locked["gt_foreground_fraction"] != float(expected_mask.mean()):
            raise ValueError("Locked crop bounds or foreground fraction differ")
        label = "nucleus" if bool(expected_mask[10, 10]) else "background"
        if record["gt_label"] != label or locked["gt_label"] != label:
            raise ValueError("Saved label differs from the true crop center")
        if label == "background" and expected_mask.any():
            raise ValueError("A background crop contains annotated foreground")
        if result is not None:
            if record["input_size"] != [20, 20] or record["processed_size"] != [224, 224]:
                raise ValueError("Saved input or processed size differs")
            if record["fresh_context"] is not True:
                raise ValueError("Each patch must use a fresh model context")
            with np.load(run / f"{record['id']}_processor_input.npz", allow_pickle=False) as archive:
                for array_key, hash_key in (("pixel_values", "pixel_values_sha256"), ("input_ids", "input_ids_sha256")):
                    if hashlib.sha256(np.ascontiguousarray(archive[array_key]).tobytes()).hexdigest() != record[hash_key]:
                        raise ValueError("Saved processor tensor hash differs")
            token_hashes.add(record["input_ids_sha256"])
            parsed = record["raw_text"].strip()
            parsed = parsed if parsed in ("nucleus", "background") else None
            if (record["predicted_label"] != parsed or record["format_valid"] != (parsed is not None)
                    or record["correct"] != (parsed == label)):
                raise ValueError("Saved classification differs from exact raw-answer parsing")
        images.append(image)
        masks.append(Image.fromarray(expected_mask.astype(np.uint8) * 255))
    if result and (len(token_hashes) != 1 or sum(r["correct"] for r in records) != result["summary"]["correct_count"]):
        raise ValueError("Task tokens or summary count differ")
    return plan, result, records, images, masks


def render(run, reference_run, output):
    run, reference_run, output = map(lambda p: Path(p).resolve(), (run, reference_run, output))
    plan, result, records, images, masks = verify(run, reference_run)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    answers = [r["raw_text"] if result else "Pending inference" for r in records]
    answer_lines = [sum([textwrap.wrap(line, width=41) or [""] for line in answer.splitlines()], []) or [""] for answer in answers]
    card_height = 340 + 23 * max(map(len, answer_lines))
    canvas = Image.new("RGB", (1480, 160 + 2 * (card_height + 18)), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Six isolated patches: nucleus or background?", font=_font(31), fill="#172438")
    status = (f"{sum(r['correct'] for r in records)}/6 correct | " if result else "Prepared inputs; inference pending | ")
    draw.text((24, 63), status + "Each model input contains only one unmarked 20 x 20 crop; A-F are report labels only.", font=_font(21), fill="#526279")
    for index, (record, image, mask, lines) in enumerate(zip(records, images, masks, answer_lines)):
        x, y = 20 + (index % 3) * 486, 108 + (index // 3) * (card_height + 18)
        color = "#168a45" if result and record["correct"] else "#b33146" if result else "#526279"
        draw.rounded_rectangle((x, y, x + 466, y + card_height), radius=12, fill="white", outline="#dce3eb", width=2)
        draw.text((x + 16, y + 12), f"{record['id']}  Ground truth: {record['gt_label']}", font=_font(23), fill="#172438")
        draw.text((x + 16, y + 50), "Actual crop", font=_font(18), fill="#526279")
        draw.text((x + 248, y + 50), "GT mask (not input)", font=_font(18), fill="#526279")
        canvas.paste(image.resize((192, 192), Image.Resampling.NEAREST), (x + 16, y + 80))
        canvas.paste(mask.convert("RGB").resize((192, 192), Image.Resampling.NEAREST), (x + 248, y + 80))
        draw.text((x + 16, y + 282), "Exact raw answer:", font=_font(19), fill=color)
        for line_index, line in enumerate(lines):
            draw.text((x + 16, y + 310 + line_index * 23), line, font=_font(19), fill="#172438")
    draw.text((24, canvas.height - 31), "One selected image | Positive crops contain partial nuclei | 20 -> 224 model resize | No SAM, training or benchmark claim", font=_font(19), fill="#526279")
    canvas.save(output / "isolated_crop_probe.png")
    rows = "".join(f"<tr><td>{r['id']}</td><td>{r['gt_label']}</td><td><pre>{html.escape(a)}</pre></td>"
                   f"<td>{r.get('format_valid', 'pending')}</td><td>{r.get('correct', 'pending')}</td></tr>" for r, a in zip(records, answers))
    details = {"prompt": plan.get("prompt", plan.get("prompts")), "generation_controls": plan.get("generation_controls"),
               "reference_run": str(reference_run), "plan_sha256": sha(run / "plan.json"), "result_sha256": sha(run / "result.json") if result else None}
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Six isolated patches</title>
<style>body{{font:18px/1.6 system-ui,sans-serif;color:#172438;background:#eef3f8;max-width:1320px;margin:30px auto;padding:0 20px}}section{{background:white;padding:24px;border-radius:14px;margin:18px 0}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px}}</style>
<h1>Six isolated patches</h1><section><b>{html.escape(status.rstrip(' | '))}</b><p>Same Qwen 32B checkpoint, one fresh context per patch. No letter IDs, coordinates, GT masks or other answers are supplied to the model.</p><img src="isolated_crop_probe.png" alt="All six actual input crops, their binary ground truth masks and exact raw responses"></section>
<section><p>Each of the same six candidate boxes defines a 20 × 20 crop from the earlier unmarked original. The model sees one crop at a time and answers nucleus or background. A–F identify report entries only; the black-and-white masks are used only to check answers.</p><p>Positive crops contain partial nuclei. The processor enlarges each 20 × 20 image to 224 × 224. Cropping removes surrounding context and enlarging changes the visual presentation. A difference from the earlier full-image result therefore cannot by itself identify coordinates, attention, or recognition as the cause of failure.</p><table><tr><th>Report ID</th><th>Ground truth</th><th>Exact raw answer</th><th>Format valid</th><th>Correct</th></tr>{rows}</table><p>Accuracy here refers only to these six supplied patches from one training-source image. It is not nucleus detection recall, segmentation PQ, or held-out dataset accuracy.</p></section>
<section><details><summary>Exact prompt and provenance</summary><pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details><p>Before rendering, each saved crop and binary mask was compared with its corresponding region of the original unmarked image and GT.</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({"status": "verified", "completed": result is not None,
        "all_six_input_pixels_verified": True, "all_six_gt_masks_verified": True, **details}, ensure_ascii=False, indent=2) + "\n")
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(render(args.run, args.reference_run, args.output))
