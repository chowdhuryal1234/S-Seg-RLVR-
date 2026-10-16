"""Verify and render the frozen-policy (C) and frozen-segmenter (D) diagnostics.

Reads saved predictions only; no model is loaded. Source image / annotation
hashes, final-map metrics, raw-mask duplicate diagnostics and all paired controls
are checked before creating a report. The old experiment-A renderer is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.ablation_metrics import DUPLICATE_DEFINITIONS, duplicate_metrics, summarize_records
from nucleus_rl.rewards import instance_metrics
from scripts.render_prompt_ablation import (
    INK, MUTED, RED, VARIANTS, _fit_text, _font, _same_number,
    instance_panel, load_verified_cases,
)


D_VARIANTS = tuple(f"{size}_{style}" for style in (*VARIANTS, "oracle_tight") for size in ("tiny", "large"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--large-policy-run", type=Path, required=True)
    parser.add_argument("--segmenter-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory")
    return parser


def _compare_values(actual, expected, context):
    """Compare nested JSON diagnostics, tolerating only harmless float rounding."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError(f"Different diagnostic fields: {context}")
        for key in actual:
            _compare_values(actual[key], expected[key], f"{context}/{key}")
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"Different diagnostic list lengths: {context}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_values(left, right, f"{context}/{index}")
    elif not _same_number(actual, expected):
        raise ValueError(f"Saved diagnostic does not match masks: {context}")


def _check_raw(run, variant, record, reference):
    path = run / variant / Path(record["raw_masks_path"]).name
    if path.suffix != ".npz":
        raise ValueError(f"Expected a saved raw-mask archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["masks"]
    computed = duplicate_metrics(raw, reference)
    for key, value in computed.items():
        _compare_values(value, record[key], f"{record['id']}/{variant}/{key}")
    if not record["format_valid"] and len(raw):
        raise ValueError("An invalid completion unexpectedly has raw SAM masks")


def _frozen_sam(audit, context):
    if (audit.get("unchanged") is not True or audit.get("trainable_parameters") != 0
            or audit.get("parameters_before") != audit.get("parameters_after")):
        raise ValueError(f"Missing frozen SAM verification: {context}")


def load_policy_comparison(base_run, large_run, dataset):
    base, base_cases, base_summary = load_verified_cases(base_run, dataset)
    large, large_cases, large_summary = load_verified_cases(large_run, dataset)
    if base["selected_crops"] != large["selected_crops"]:
        raise ValueError("C must use the same selected crops in the same order")
    for result, context in ((base, "C base"), (large, "C larger policy")):
        if result["model"].get("adapter") is not None:
            raise ValueError("C compares unadapted checkpoints, not trained adapters")
        _frozen_sam(result["sam"], context)
    for key in ("checkpoint_sha256", "config", "source_commit"):
        if base["sam"][key] != large["sam"][key]:
            raise ValueError(f"C must hold SAM fixed: {key}")
    plans = [json.loads((run / "plan.json").read_text()) for run in (base_run, large_run)]
    for key in ("seed", "max_completion_length", "min_pixels", "max_pixels", "coordinate_frame"):
        if plans[0]["args"][key] != plans[1]["args"][key]:
            raise ValueError(f"C decoding or image processing settings differ: {key}")
    if plans[0]["decoding"] != plans[1]["decoding"]:
        raise ValueError("C decoding methods differ")
    for left, right in zip(base_cases, large_cases):
        for variant in VARIANTS:
            a, b = left["arms"][variant]["record"], right["arms"][variant]["record"]
            for key in ("prompt", "prompt_size", "pixel_values_sha256", "coordinate_frame"):
                if a[key] != b[key]:
                    raise ValueError(f"C paired control differs: {a['id']}/{variant}/{key}")
            if "input_ids_sha256" in a and "input_ids_sha256" in b:
                if a["input_ids_sha256"] != b["input_ids_sha256"]:
                    raise ValueError("C tokenized prompt inputs differ")
            _check_raw(base_run, variant, a, left["reference"])
            _check_raw(large_run, variant, b, right["reference"])
    return (base, base_cases, base_summary), (large, large_cases, large_summary)


def load_segmenter_comparison(run, base_result, base_cases):
    result = json.loads((run / "result.json").read_text())
    if (result.get("status") != "completed_fixed_prompt_segmenter_ablation"
            or result.get("optimizer_steps") != 0 or result.get("vlm_loaded") is not False
            or result.get("qwen_generation_calls") != 0
            or result.get("fixed_prompts_identical") is not True
            or result.get("source_crop_hashes_verified") is not True):
        raise ValueError("D must be a completed inference-only run with verified fixed prompts")
    for size in ("tiny", "large"):
        _frozen_sam(result["models"][size], f"D {size}")
    for key in ("checkpoint_sha256", "config", "source_commit"):
        if result["models"]["tiny"][key] != base_result["sam"][key]:
            raise ValueError(f"D must replay the same SAM tiny checkpoint used in C: {key}")
    if result["selected_crops"] != base_result["selected_crops"]:
        raise ValueError("D must use the same image and annotation hashes and selection order as C")
    records = {}
    ids = [item["id"] for item in result["selected_crops"]]
    for line in (run / "records.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            key = (record["id"], record["variant"])
            if key in records:
                raise ValueError(f"Duplicate D record: {key}")
            records[key] = record
    expected = {(crop_id, variant) for crop_id in ids for variant in D_VARIANTS}
    if set(records) != expected:
        raise ValueError("D does not have exactly one saved prediction per crop and variant")
    used_paths, cases = set(), []
    for base_case in base_cases:
        crop_id = base_case["selection"]["id"]
        arms = {}
        for variant in D_VARIANTS:
            record = records[crop_id, variant]
            oracle = variant.endswith("oracle_tight")
            if (record.get("oracle_prompts") is not oracle
                    or record.get("ground_truth_used_for_prompting") is not oracle):
                raise ValueError(f"D oracle provenance flag differs: {crop_id}/{variant}")
            prediction_path = run / variant / Path(record["prediction_path"]).name
            if prediction_path.suffix != ".npy" or prediction_path in used_paths:
                raise ValueError(f"Invalid or reused D prediction: {prediction_path}")
            used_paths.add(prediction_path)
            prediction = np.load(prediction_path, allow_pickle=False)
            for key, value in instance_metrics(prediction, base_case["reference"]).items():
                _compare_values(value, record[key], f"{crop_id}/{variant}/{key}")
            if not record["format_valid"] and np.any(prediction):
                raise ValueError("D invalid completion has a nonempty prediction")
            _check_raw(run, variant, record, base_case["reference"])
            arms[variant] = {"record": record, "prediction": prediction}
        for style in VARIANTS:
            original = base_case["arms"][style]
            tiny, large = arms[f"tiny_{style}"], arms[f"large_{style}"]
            # Actual saved model answers must be reused, never regenerated for D.
            for arm in (tiny, large):
                for key in ("text", "format_valid", "raw_objects", "converted_objects", "coordinate_frame", "prompt_size", "original_size", "scale_to_original"):
                    _compare_values(arm["record"][key], original["record"][key], f"D replay/{crop_id}/{style}/{key}")
            if not np.array_equal(tiny["prediction"], original["prediction"]):
                raise ValueError(f"Replayed SAM tiny prediction differs from C: {crop_id}/{style}")
        for key in ("text", "raw_objects", "converted_objects"):
            _compare_values(arms["tiny_oracle_tight"]["record"][key],
                            arms["large_oracle_tight"]["record"][key], f"D identical oracle prompts/{crop_id}/{key}")
        cases.append({**base_case, "arms": arms})
    summaries = {}
    for variant in D_VARIANTS:
        summary = summarize_records(records[crop_id, variant] for crop_id in ids)
        for key, value in summary.items():
            _compare_values(value, result["variants"][variant][key], f"D summary/{variant}/{key}")
        summaries[variant] = summary
    return result, cases, summaries


def render_sheet(cases, arm_keys, titles, first, total, title, subtitle, destination, oracle=False):
    """Three-column, four-case sheets: always keep the predetermined order."""
    width, margin, column, row_height, top = 984, 24, 312, 400, 142
    canvas = Image.new("RGB", (width, top + row_height * len(cases) + 64), "white")
    draw = ImageDraw.Draw(canvas)
    if oracle:
        draw.rectangle((0, 0, width, 85), fill="#fff0cf")
    draw.text((margin, 16), title, font=_font(25), fill=INK)
    draw.text((margin, 52), _fit_text(draw, subtitle, _font(17), width - 2 * margin), font=_font(17), fill=MUTED)
    draw.text((margin, 83), f"Cases {first + 1}-{first + len(cases)} of {total} | Same training crops | Zero optimizer steps", font=_font(16), fill=MUTED)
    for col, label in enumerate(("Ground truth", *titles)):
        draw.text((margin + col * column + column / 2, 121), label, anchor="mm", font=_font(20), fill=INK)
    for index, case in enumerate(cases):
        y = top + index * row_height
        draw.line((margin, y, width - margin, y), fill="#dce2ea")
        draw.text((margin + 8, y + 8), f"{first + index + 1:02d}  {case['selection']['id']}", font=_font(18), fill=INK)
        for col, arm_key in enumerate((None, *arm_keys)):
            x = margin + col * column + (column - 256) // 2
            arm = case["arms"][arm_key] if arm_key else None
            labels = arm["prediction"] if arm else case["reference"]
            canvas.paste(instance_panel(case["image"], labels), (x, y + 38))
            if not arm:
                lines = [(f"Annotated nuclei: {case['selection']['gt_count']}", INK),
                         ("128 x 128 crop, enlarged 2x", MUTED),
                         ("GT supplied to SAM only in oracle panels" if oracle else "GT used only to score predictions", MUTED)]
            else:
                record = arm["record"]
                lines = [(f"PQ {record['pq']:.3f} | Recall {record['recall']:.1%}", INK),
                         (f"Pred / GT {record['pred_count']} / {record['gt_count']} | Count error {record['count_error']}", INK),
                         (f"Duplicates {record['same_gt_duplicate_count']} | Overlap pairs {record['raw_overlap_pair_count']}", MUTED),
                         (("Oracle prompts from annotation" if oracle else "Valid output") if record["format_valid"] else "INVALID: " + str(record.get("parse_error", "format error")),
                          MUTED if record["format_valid"] else RED)]
                if not record["format_valid"]:
                    draw.rectangle((x - 2, y + 36, x + 257, y + 295), outline=RED, width=2)
            for line, (text, color) in enumerate(lines):
                draw.text((margin + col * column + 8, y + 304 + line * 22),
                          _fit_text(draw, text, _font(15), column - 16), font=_font(15), fill=color)
    footer = top + row_height * len(cases) + 8
    draw.text((margin, footer), "Colors and IDs are local to each panel. Matching colors across panels do not indicate matched nuclei.", font=_font(15), fill=MUTED)
    draw.text((margin, footer + 22), "Oracle panels use ground-truth locations and counts; they do not represent automatic nucleus detection." if oracle else
              "All preselected cases are shown, including invalid answers. These are not held-out benchmark results.", font=_font(15), fill=MUTED)
    canvas.save(destination)


def _table(columns, oracle=False):
    e = lambda value: html.escape(str(value), quote=True)
    metrics = (("Macro PQ", "macro_pq"), ("Macro recall", "macro_recall"), ("Micro recall", "micro_recall"),
               ("Count MAE (nuclei per crop)", "count_mae"), ("Matched / GT nuclei", None),
               ("Total predicted nuclei", "total_pred_count"), ("Same-GT duplicate extras", "total_same_gt_duplicate_count"),
               ("Raw overlap pairs (proxy)", "total_raw_overlap_pair_count"), ("Valid completions" if not oracle else "Valid constructed prompts", "format_valid_count"))
    rows = []
    for label, key in metrics:
        cells = []
        for _, summary in columns:
            if key is None:
                text = f"{summary['total_tp']} / {summary['total_gt_count']}"
            elif key == "format_valid_count":
                text = f"{summary[key]} / {summary['n_records']}"
            elif key in ("macro_recall", "micro_recall"):
                text = f"{summary[key]:.2%}"
            elif key in ("macro_pq", "count_mae"):
                text = f"{summary[key]:.4f}"
            else:
                text = str(summary[key])
            cells.append(f"<td>{e(text)}</td>")
        rows.append(f"<tr><th>{e(label)}</th>{''.join(cells)}</tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Metric</th>" +
            "".join(f"<th>{e(label)}</th>" for label, _ in columns) + "</tr></thead><tbody>" +
            "".join(rows) + "</tbody></table></div>")


def render_report(base_run, large_run, segmenter_run, dataset, output):
    base_run, large_run, segmenter_run, dataset, output = [Path(p).resolve() for p in (base_run, large_run, segmenter_run, dataset, output)]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty output directory; previous reports are never overwritten")
    base, large = load_policy_comparison(base_run, large_run, dataset)
    d_result, d_cases, d_summary = load_segmenter_comparison(segmenter_run, base[0], base[1])
    combined_cases = []
    for left, right in zip(base[1], large[1]):
        arms = {f"base_{style}": left["arms"][style] for style in VARIANTS}
        arms.update({f"large_{style}": right["arms"][style] for style in VARIANTS})
        combined_cases.append({**left, "arms": arms})
    output.mkdir(parents=True, exist_ok=True)
    e = lambda value: html.escape(str(value), quote=True)
    base_label = base[0]["model"]["model"].split("/")[-1].replace("-Instruct", "")
    large_label = large[0]["model"]["model"].split("/")[-1].replace("-Instruct", "")
    sections, sheets = [], []
    for experiment, cases, summaries in (("C", combined_cases, None), ("D", d_cases, d_summary)):
        for style in VARIANTS:
            label = "One-object numeric example" if style == "one_example" else "Three-object numeric example"
            keys = (f"base_{style}", f"large_{style}") if experiment == "C" else (f"tiny_{style}", f"large_{style}")
            titles = ("Qwen 7B + SAM tiny", "Qwen 32B + SAM tiny") if experiment == "C" else ("SAM2.1 tiny", "SAM2.1 large")
            columns = [(titles[0], base[2][style]), (titles[1], large[2][style])] if experiment == "C" else [(titles[i], summaries[key]) for i, key in enumerate(keys)]
            heading = f"{experiment}: {label}"
            subtitle = "Only the Qwen checkpoint changes; SAM and inputs remain fixed." if experiment == "C" else "Same saved Qwen 7B answer; only the SAM checkpoint changes."
            block = [f"<section id='{experiment.lower()}-{style}'><h2>{heading}</h2><p>{subtitle}</p>", _table(columns)]
            for first in range(0, len(cases), 4):
                name = f"{experiment.lower()}_{style}_{first // 4 + 1:02d}.png"
                render_sheet(cases[first:first + 4], keys, titles, first, len(cases), heading, subtitle, output / name)
                sheets.append(name)
                block.append(f"<img loading='lazy' src='{name}' alt='{e(heading)}: preselected cases {first + 1} to {min(first + 4, len(cases))}'>")
            block.append("</section>")
            sections.append("".join(block))
    oracle_keys = ("tiny_oracle_tight", "large_oracle_tight")
    oracle = ["<section class='oracle' id='oracle'><h2>D: Oracle prompt control</h2><p><strong>Privileged diagnostic:</strong> each true nucleus supplies a tight box and an interior point. Ground-truth counts and locations are exposed to SAM here. This tests segmentation with good prompts; it is not an automatic detector or a deployment score.</p>",
              _table([("SAM2.1 tiny + GT prompts", d_summary[oracle_keys[0]]), ("SAM2.1 large + GT prompts", d_summary[oracle_keys[1]])], oracle=True)]
    for first in range(0, len(d_cases), 4):
        name = f"d_oracle_tight_{first // 4 + 1:02d}.png"
        render_sheet(d_cases[first:first + 4], oracle_keys, ("SAM tiny + GT prompts", "SAM large + GT prompts"), first, len(d_cases),
                     "D: Oracle prompt control", "GROUND-TRUTH boxes + points | Privileged diagnostic, not automatic detection", output / name, oracle=True)
        sheets.append(name)
        oracle.append(f"<img loading='lazy' src='{name}' alt='Oracle prompt control, preselected cases {first + 1} to {min(first + 4, len(d_cases))}'>")
    oracle.append("</section>")
    sections.append("".join(oracle))
    details = []
    for index, case in enumerate(combined_cases):
        for key, arm in case["arms"].items():
            record = arm["record"]
            details.append(f"<details><summary>{index + 1:02d} {e(case['selection']['id'])} / {e(key)} / {'valid' if record['format_valid'] else 'INVALID'}</summary><p>{e(record.get('parse_error') or 'No format error')}</p><pre>{e(record['text'])}</pre></details>")
    total_gt = sum(case["selection"]["gt_count"] for case in combined_cases)
    model_lines = "<br>".join(f"{e(result['model']['model'])} @ {e(result['model']['resolved_revision'])}" for result in (base[0], large[0]))
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frozen model diagnostics: policy size and SAM size</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#182235;max-width:1120px;margin:32px auto;padding:0 20px}}h1{{line-height:1.2}}.note{{background:#eef3f9;padding:18px;border-radius:8px}}.oracle{{border-left:5px solid #e8ac38;padding-left:18px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce2ea}}thead{{background:#eef3f9}}td{{font-variant-numeric:tabular-nums}}img{{width:100%;height:auto;border:1px solid #dce2ea;margin:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:12px}}details{{margin:10px 0}}small{{color:#536279}}section{{padding-top:24px}}.tablewrap{{overflow-x:auto}}nav a{{display:inline-block;margin:4px 16px 4px 0}}dt{{font-weight:650;margin-top:12px}}dd{{margin-left:0}}</style>
<h1>What changes when the model gets larger?</h1>
<p class="note"><strong>Inference only: {len(combined_cases)} fixed training crops, {total_gt} annotated nuclei, zero optimizer steps.</strong><br>
C compares {e(base_label)} and {e(large_label)} while keeping SAM2.1 tiny fixed. D compares SAM2.1 tiny and large using the exact saved Qwen 7B answers. Each comparison is repeated with the one-object and three-object numeric formatting examples. An oracle control is displayed separately.</p>
<p>The crops come from the official 30-image MoNuSeg training whitelist. These small, density-selected crops support debugging; they are not held-out validation, full-image benchmark results, or evidence of a training improvement. Invalid answers stay in every denominator.</p>
<p>C compares released checkpoints: their pretraining and post-training may also differ, so any change cannot be attributed to parameter count alone.</p>
<nav><a href="#c-one_example">C: one-object example</a><a href="#c-three_examples">C: three-object example</a><a href="#d-one_example">D: one-object example</a><a href="#d-three_examples">D: three-object example</a><a href="#oracle">Oracle control</a><a href="#metrics">Metric definitions</a><a href="#answers">Raw responses</a></nav>
<p><small>{model_lines}<br>Before rendering: verified all source image / annotation hashes, every saved final-mask metric, every raw-mask duplicate diagnostic, fixed crop order, paired prompts and image tensors, and exact D replay of the learned prompts. See <a href="verification.json">verification.json</a> and <a href="per_crop.csv">per_crop.csv</a>.</small></p>
{''.join(sections)}
<h2 id="metrics">How to read the metrics</h2><dl>
<dt>PQ and recall</dt><dd>Final non-overlapping instance maps are scored. A predicted nucleus matches a reference nucleus only at IoU &gt; 0.5. PQ combines successful detection with outline overlap; recall is the fraction of reference nuclei matched.</dd>
<dt>Macro and micro</dt><dd>Macro PQ and macro recall average all crops equally. Micro recall is the total matched nuclei divided by total annotated nuclei.</dd>
<dt>Count MAE</dt><dd>The average absolute difference between predicted and annotated counts per crop. Correct counts alone do not establish correct locations.</dd>
<dt>Same-GT duplicate extras</dt><dd>{e(DUPLICATE_DEFINITIONS['same_gt_duplicate_count'])} Computed before resolving SAM mask overlaps.</dd>
<dt>Raw overlap pairs</dt><dd>{e(DUPLICATE_DEFINITIONS['raw_overlap_pair_count'])}</dd>
<dt>Format validity</dt><dd>The strict parser accepts the complete response or rejects it. Invalid responses receive empty predictions. Oracle prompts are constructed from annotations, so their validity is not a VLM output score.</dd></dl>
<h2 id="answers">Every saved Qwen response</h2><p>D reuses the 7B responses below without another Qwen call.</p>{''.join(details)}</html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    verification = {"verified_cases": len(combined_cases), "gt_nuclei": total_gt, "optimizer_steps": 0,
                    "source_hashes_verified": True, "final_metrics_recomputed": True, "raw_duplicate_metrics_recomputed": True,
                    "paired_policy_controls_verified": True, "saved_7b_prompts_reused_for_d": True,
                    "tiny_predictions_reproduced": True, "oracle_separate": True,
                    "c_base": base[2], "c_large_policy": large[2], "d": d_summary}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    keys = ("id", "variant", "gt_count", "pred_count", "tp", "fp", "fn", "pq", "recall", "count_error", "format_valid", "parse_error", "same_gt_duplicate_count", "raw_overlap_pair_count")
    with (output / "per_crop.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("experiment", *keys))
        writer.writeheader()
        for experiment, cases in (("C", combined_cases), ("D", d_cases)):
            for case in cases:
                for variant, arm in case["arms"].items():
                    writer.writerow({"experiment": experiment, **{key: arm["record"].get(key) for key in keys}, "variant": variant})
    return {"report": str(output / "index.html"), "sheets": [str(output / name) for name in sheets], "verified_cases": len(combined_cases)}


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        print(json.dumps(render_report(args.base_run, args.large_policy_run, args.segmenter_run, args.dataset, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Cannot render follow-up report: {error}") from error
