"""Verify downloaded training artifacts and render English, per-run reports.

No GPU, model loading, training, or source-artifact mutation. Missing artifacts
are labeled unavailable; corrupt or inconsistent artifacts stop verification.
Benchmark and full-validation cohorts are never pooled. Examples use only GT
counts and lexical IDs, with identical selections for runs sharing a cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.ablation_metrics import duplicate_metrics, summarize_records
from nucleus_rl.evaluate import objects_to_original
from nucleus_rl.rewards import evaluate_prediction, parse_completion
from scripts.report_graphics import _font, _fit_text, instance_panel


METRICS = (("macro_pq", "Macro PQ"), ("macro_recall", "Macro recall"),
           ("micro_recall", "Micro recall"), ("count_mae", "Count MAE"),
           ("total_tp", "Matched nuclei (TP)"), ("total_gt_count", "Annotated nuclei"),
           ("total_pred_count", "Final instances"),
           ("total_same_gt_duplicate_count", "Same-GT duplicate extras"),
           ("total_raw_overlap_pair_count", "Raw-mask overlap pairs"),
           ("format_valid_count", "Valid answers"), ("mean_foreground_iou", "Foreground IoU"),
           ("mean_total_reward", "Reward (supporting diagnostic)"))
SUMMARY_FIELDS = {"pq": "macro_pq", "recall": "macro_recall", "count_error": "count_mae",
                  "seg_reward": "mean_foreground_iou", "total_reward": "mean_total_reward"}
CSS = """body{font:17px/1.6 system-ui,sans-serif;color:#172438;background:#eef3f8;
max-width:1450px;margin:28px auto;padding:0 20px}section,details{background:white;
padding:20px;border-radius:12px;margin:18px 0}h1{line-height:1.2}h2{font-size:24px}
img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}
th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3eb;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:16px}
a{color:#075da8}.warning{border-left:5px solid #bc7d22}.scroll{overflow-x:auto}"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).is_file() else []


def same(actual, expected, label):
    """Compare JSON-compatible values without weakening boolean checks."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Different keys: {label}")
        for key in actual:
            same(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError(f"Different list lengths: {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f"{label}/{index}")
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Numeric mismatch: {label}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Mismatch: {label}")


def load_dataset(dataset):
    dataset = Path(dataset).resolve()
    protocol = load(dataset / "frozen_protocol.json")
    if not protocol:
        raise ValueError("A frozen dataset protocol is required")
    cohorts, assets, hashes = {}, {}, {}
    cohort_path = Path(__file__).resolve().parents[1] / "protocols/monuseg_official30.json"
    same(sha(cohort_path), protocol["official30_protocol_sha256"], "official-30 source protocol")
    canonical_ids = set(load(cohort_path)["image_ids"])
    for name, expected in (("train", protocol["train_crops"]),
                           ("validation", protocol["validation_crops"]),
                           ("validation_benchmark8", protocol["benchmark_crops"])):
        path = dataset / f"{name}.jsonl"
        hashes[name] = sha(path)
        same(hashes[name], protocol["manifest_sha256"][name], name + " manifest hash")
        rows = jsonl(path)
        if len(rows) != expected or len({r["id"] for r in rows}) != expected:
            raise ValueError(f"Wrong row count or duplicate IDs in {name}")
        expected_split = "train" if name == "train" else "validation"
        for row in rows:
            if row["split"] != expected_split:
                raise ValueError(f"Unexpected split: {row['id']}")
            if row["source_image_id"] not in canonical_ids:
                raise ValueError(f"Noncanonical source image: {row['id']}")
            if row["id"] in assets:
                same(row, assets[row["id"]]["row"], "shared manifest row")
                continue
            paths = {kind: dataset / row[f"{kind}_path"] for kind in ("image", "mask")}
            for kind, asset in paths.items():
                same(sha(asset), row[f"{kind}_sha256"], f"{row['id']} {kind} hash")
            with Image.open(paths["image"]) as opened:
                image = opened.convert("RGB")
            with Image.open(paths["mask"]) as opened:
                mask = np.asarray(opened).copy()
            if mask.shape != (image.height, image.width) or mask.dtype.kind not in "ui":
                raise ValueError(f"Invalid GT dimensions/IDs: {row['id']}")
            count = len(np.unique(mask[mask > 0]))
            same(count, row["instance_count"], f"{row['id']} annotated count")
            assets[row["id"]] = {"row": row, "image": image, "mask": mask, "count": count}
        cohorts[name] = rows
    if {r["patient_id"] for r in cohorts["train"]} & {r["patient_id"] for r in cohorts["validation"]}:
        raise ValueError("Training and validation patient overlap")
    validation = {r["id"]: r for r in cohorts["validation"]}
    for row in cohorts["validation_benchmark8"]:
        same(row, validation.get(row["id"]), "benchmark must be exact full-validation subset")
    return {"root": dataset, "protocol": protocol, "cohorts": cohorts,
            "assets": assets, "manifest_sha256": hashes}


def select_examples(rows, assets):
    """Same GT-only rule across runs; never use outputs or improvement scores."""
    ordered = sorted(rows, key=lambda row: row["id"])
    return {"single": next((r["id"] for r in ordered if assets[r["id"]]["count"] == 1), None),
            "multiple": next((r["id"] for r in ordered if assets[r["id"]]["count"] > 1), None)}


def aggregate(records):
    if not records:
        return None
    n, total_gt, total_tp = len(records), sum(r["gt_count"] for r in records), sum(r["tp"] for r in records)
    summary = {"n_records": n, "macro_pq": statistics.mean(r["pq"] for r in records),
               "macro_recall": statistics.mean(r["recall"] for r in records),
               "micro_recall": total_tp / total_gt if total_gt else 1.,
               "count_mae": statistics.mean(r["count_error"] for r in records),
               "total_tp": total_tp, "total_gt_count": total_gt,
               "total_pred_count": sum(r["pred_count"] for r in records),
               "format_valid_count": sum(r["format_valid"] for r in records),
               "mean_foreground_iou": statistics.mean(r["seg_reward"] for r in records),
               "mean_total_reward": statistics.mean(r["total_reward"] for r in records)}
    for key in ("same_gt_duplicate_count", "raw_overlap_pair_count"):
        summary["total_" + key] = sum(r[key] for r in records) if all(key in r for r in records) else None
    return summary


def verify_phase(directory, rows, assets, arguments, saved_result=None):
    directory = Path(directory)
    records, saved_summary = jsonl(directory / "completions.jsonl"), load(directory / "summary.json")
    expected_ids = [r["id"] for r in rows]
    actual_ids = [r["id"] for r in records]
    if actual_ids != expected_ids[:len(actual_ids)]:
        raise ValueError(f"Evaluation rows are reordered, repeated, or outside the cohort: {directory}")
    verified, examples, missing, hashes = [], {}, [], {}
    for index, record in enumerate(records, 1):
        asset = assets[record["id"]]
        image, gt = asset["image"], asset["mask"]
        same(record["patient_id"], asset["row"]["patient_id"], "record patient")
        same(record["original_size"], list(image.size), "record original dimensions")
        for key in ("coordinate_frame", "prompt_style"):
            if key in arguments:
                same(record[key], arguments[key], key)
        if Path(record["prediction_path"]).name != f"{index:06d}.npy":
            raise ValueError("Saved prediction basename does not match completion order")
        prediction_file = directory / f"{index:06d}.npy"
        if not prediction_file.is_file():
            missing.append(str(prediction_file))
            continue
        prediction = np.load(prediction_file, allow_pickle=False)
        prompt_size = tuple(record["prompt_size"])
        parsed = parse_completion(record["text"], width=prompt_size[0], height=prompt_size[1],
                                  max_objects=arguments["max_objects"])
        if (not parsed.valid or not parsed.objects) and np.any(prediction):
            raise ValueError("Rejected/empty completion has nonempty saved final instances")
        converted, scales = objects_to_original(parsed.objects, prompt_size, image.size)
        same([asdict(obj) for obj in parsed.objects], record["raw_objects"], "raw parsed objects")
        same([asdict(obj) for obj in converted], record["converted_objects"], "converted objects")
        same(list(scales), record["scale_to_original"], "coordinate scale")
        metrics = evaluate_prediction(record["text"], prediction, gt,
                                      max_objects=arguments["max_objects"], prompt_size=prompt_size)
        for key, value in metrics.items():
            same(value, record[key], f"{directory.name}/{record['id']}/{key}")
        hashes[prediction_file.name] = sha(prediction_file)
        raw_file, raw_masks = directory / f"{index:06d}_raw_masks.npz", None
        if raw_file.is_file():
            with np.load(raw_file, allow_pickle=False) as archive:
                raw_masks = archive["masks"].copy()
            diagnostic = duplicate_metrics(raw_masks, gt)
            for key, value in diagnostic.items():
                same(value, record[key], f"duplicate diagnostic/{key}")
            if not parsed.valid and len(raw_masks):
                raise ValueError("Rejected answer has raw SAM masks")
            same(len(raw_masks), len(parsed.objects), "one raw mask per parsed object")
            if not np.array_equal(np.any(raw_masks, axis=0), prediction > 0):
                raise ValueError("Raw-mask union differs from saved final foreground")
            metrics.update(diagnostic)
            hashes[raw_file.name] = sha(raw_file)
        else:
            missing.append(str(raw_file))
        text_file = directory / f"{index:06d}.txt"
        if text_file.is_file():
            same(text_file.read_text(), record["text"], "saved completion text")
            hashes[text_file.name] = sha(text_file)
        verified.append({"id": record["id"], **metrics})
        examples[record["id"]] = {"record": record, "prediction": prediction, "raw_masks": raw_masks}
    metrics = aggregate(verified)
    complete = actual_ids == expected_ids and len(verified) == len(rows)
    if saved_summary is not None and complete:
        same(saved_summary["completions"], len(rows), "summary crop count")
        for source, target in SUMMARY_FIELDS.items():
            same(saved_summary["mean_" + source], metrics[target], "summary/" + source)
        same(saved_summary["invalid_completions"], len(rows) - metrics["format_valid_count"], "invalid count")
        if all("same_gt_duplicate_count" in row for row in verified) and "instance_diagnostics" in saved_summary:
            same(summarize_records(verified), saved_summary["instance_diagnostics"], "summary duplicates")
        if saved_result is not None:
            same(saved_summary, saved_result, "result embeds exact evaluation summary")
    if saved_summary is not None:
        hashes["summary.json"] = sha(directory / "summary.json")
    if (directory / "completions.jsonl").exists():
        hashes["completions.jsonl"] = sha(directory / "completions.jsonl")
    return {"metrics": metrics, "complete": complete, "expected_crops": len(rows),
            "record_count": len(records), "verified_final_maps": len(verified),
            "verified_raw_masks": sum("same_gt_duplicate_count" in r for r in verified),
            "missing_artifacts": missing, "saved_summary": saved_summary, "file_sha256": hashes,
            "records": records, "examples": examples}


def number(value, digits=4):
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value) if isinstance(value, (int, str)) else f"{value:.{digits}f}"


def metric_rows(before, after):
    b, a = before["metrics"] or {}, after["metrics"] or {}
    return [(label, number(b.get(key)), number(a.get(key))) for key, label in METRICS]


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows) + "</table></div>"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows])


def prompt_panel(image, record, size=288):
    panel = image.resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    sx, sy = size / image.width, size / image.height
    for index, obj in enumerate(record["converted_objects"], 1):
        x0, y0, x1, y1 = obj["box"]
        draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline="#ff9d00", width=3)
        if obj.get("point") is not None:
            x, y = obj["point"]
            draw.ellipse((x*sx-4, y*sy-4, x*sx+4, y*sy+4), fill="#ff9d00")
        draw.text((min(size-30, max(0, x0*sx)), min(size-20, max(0, y0*sy))), f"P{index}",
                  fill="white", font=_font(15), stroke_width=2, stroke_fill="#172438")
    return panel


def raw_panel(masks, size=288):
    panel = Image.new("RGB", (size, size), "#e7edf4")
    draw = ImageDraw.Draw(panel)
    if masks is None or not len(masks):
        draw.text((14, 120), "Raw masks unavailable" if masks is None else "Zero raw SAM masks", font=_font(20), fill="#526279")
        return panel
    columns = math.ceil(math.sqrt(len(masks)))
    cell = size // columns
    for index, mask in enumerate(masks):
        x, y = index % columns * cell, index // columns * cell
        tile = Image.fromarray((mask.astype(np.uint8) * 255)).convert("RGB").resize((cell-4, cell-4), Image.Resampling.NEAREST)
        panel.paste(tile, (x+2, y+2))
        draw.text((x+6, y+4), f"P{index+1}", font=_font(16), fill="#ffb52e", stroke_width=1, stroke_fill="black")
    return panel


def render_example(asset, before, after, destination, run_name):
    canvas = Image.new("RGB", (1380, 880), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), _fit_text(draw, run_name, _font(27), 1336), font=_font(27), fill="#172438")
    noun = "nucleus" if asset["count"] == 1 else "nuclei"
    draw.text((20, 58), f"{asset['row']['id']} | {asset['count']} annotated {noun} | Fixed GT-only example selection", font=_font(20), fill="#526279")
    for phase_index, (phase, saved) in enumerate((("Before", before), ("After", after))):
        y = 106 + phase_index * 368
        panels = [("Shared RGB input" if phase_index == 0 else "Shared GT instances",
                   asset["image"].resize((288, 288), Image.Resampling.NEAREST) if phase_index == 0 else instance_panel(asset["image"], asset["mask"], 288))]
        if saved is not None:
            record = saved["record"]
            panels += [(phase + " parsed prompts", prompt_panel(asset["image"], record)),
                       (phase + " raw masks (each)", raw_panel(saved["raw_masks"])),
                       (phase + " final instances", instance_panel(asset["image"], saved["prediction"], 288))]
        else:
            empty = Image.new("RGB", (288, 288), "#e7edf4")
            ImageDraw.Draw(empty).text((22, 126), "Artifact unavailable", font=_font(22), fill="#526279")
            panels += [(phase + " parsed prompts", empty), (phase + " raw masks", empty), (phase + " final instances", empty)]
        for col, (title, panel) in enumerate(panels):
            x = 20 + col * 344
            draw.text((x+4, y), title, font=_font(21), fill="#172438")
            canvas.paste(panel, (x+4, y+36))
        if saved is not None:
            r = saved["record"]
            caption = (f"{phase}: valid={r['format_valid']} | final instances={r['pred_count']} | TP/GT={r['tp']}/{r['gt_count']} | "
                       f"PQ={r['pq']:.4f} | recall={r['recall']:.3f} | count error={r['count_error']}")
            draw.text((368, y+331), caption, font=_font(17), fill="#526279")
    draw.text((20, 850), "Prompt overlays show strictly parsed objects. Rejected answers remain in metrics; exact raw text is retained in HTML/JSON.", font=_font(19), fill="#526279")
    canvas.save(destination)


def run_report(suite, entry, data, output):
    name = entry["name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Run name must be a safe directory name")
    directory, target = suite / name, output / name
    target.mkdir()
    result, arguments, metadata = (load(directory / f"{item}.json") for item in ("result", "arguments", "metadata"))
    benchmark = entry["benchmark"]
    cohort_name = "validation_benchmark8" if benchmark else "validation"
    rows = data["cohorts"][cohort_name]
    selected = select_examples(rows, data["assets"])
    if arguments is not None:
        for key, expected in (("learning_rate", entry["learning_rate"]), ("seed", entry["seed"]), ("max_steps", entry["planned_steps"])):
            same(arguments[key], expected, "planned argument/" + key)
        if arguments.get("adapter") is not None or arguments.get("resume_from_checkpoint"):
            raise ValueError("This suite requires fresh-base runs")
    if metadata is not None:
        same(metadata["manifest_sha256"], data["manifest_sha256"]["train"], "training manifest identity")
        same(metadata["eval_manifest_sha256"], data["manifest_sha256"][cohort_name], "evaluation cohort identity")
        if arguments is not None:
            same(metadata["model"], arguments["model"], "model identity")
            same(metadata["resolved_revision"], arguments["revision"], "checkpoint revision")
            same(metadata["adapter"], None, "fresh base model")
    if result is not None and entry.get("result") is not None:
        same(result, entry["result"], "downloaded result versus suite snapshot")
    if result is not None:
        same(result["sam"]["unchanged"], result["sam"]["parameters_before"] == result["sam"]["parameters_after"], "SAM freeze proof")
        same(result["sam"]["unchanged"], True, "SAM must stay frozen")
        same(result["sam"]["trainable_parameters"], 0, "SAM trainable parameters")
        same(result["visual_sha256_before"], result["visual_sha256_after"], "vision encoder freeze proof")
        same(result["adapter_changed"], result["adapter_sha256_before"] != result["adapter_sha256_after"], "adapter change proof")
    phases = {}
    for phase in ("before", "after"):
        # Missing arguments are acceptable only for a run that produced no records.
        if arguments is None and (directory / phase / "completions.jsonl").is_file():
            raise ValueError("Evaluation records exist without their arguments")
        phases[phase] = verify_phase(directory / phase, rows, data["assets"], arguments or {},
                                     result.get(phase) if result else None)
    status = result.get("status") if result else entry["status"]
    timing = [("Planned steps", entry["planned_steps"]),
              ("Completed updates this run", (result or {}).get("optimizer_steps_this_run")),
              ("Requested steps completed", (result or {}).get("completed_requested_steps")),
              ("Learning rate", f"{entry['learning_rate']:g}"), ("Seed", entry["seed"]),
              ("Training seconds", (result or {}).get("training_elapsed_seconds")),
              ("Before / after evaluation seconds", f"{number((result or {}).get('before_evaluation_seconds'), 2)} / {number((result or {}).get('after_evaluation_seconds'), 2)}"),
              ("Total runner seconds", (result or {}).get("elapsed_seconds")),
              ("Training-step peak allocated GiB", (result["peak_gpu_allocated_bytes"] / 1024**3) if result and result.get("peak_gpu_allocated_bytes") is not None else None),
              ("Training-step peak reserved GiB", (result["peak_gpu_reserved_bytes"] / 1024**3) if result and result.get("peak_gpu_reserved_bytes") is not None else None),
              ("Adapter changed", (result or {}).get("adapter_changed")),
              ("Nonzero-gradient steps", (result or {}).get("nonzero_gradient_steps")),
              ("Stop reason", (result or {}).get("stop_reason", "Unavailable") if not result or result.get("stop_reason") else "None reported")]
    timing = [(key, number(value, 3)) for key, value in timing]
    coverage = "; ".join(f"{phase}: {p['verified_final_maps']}/{len(rows)} final maps verified, {p['verified_raw_masks']}/{len(rows)} raw-mask files verified" for phase, p in phases.items())
    scope = f"{'Benchmark timing subset' if benchmark else 'Full validation cohort'}: {len(rows)} crops, {len({r['patient_id'] for r in rows})} patients"
    metrics = metric_rows(phases["before"], phases["after"])
    figures, example_sections, example_metadata = [], [], {}
    for kind, crop_id in selected.items():
        if crop_id is None:
            example_sections.append(f"<section><h2>{kind.title()}-nucleus example unavailable</h2><p>This evaluated cohort contains no crop in this GT-count category. No unevaluated example is substituted.</p></section>")
            continue
        asset = data["assets"][crop_id]
        paired = {phase: p["examples"].get(crop_id) for phase, p in phases.items()}
        filename = f"{kind}_example.png"
        render_example(asset, paired["before"], paired["after"], target / filename, name)
        figures.append(filename)
        example_metadata[kind] = {"id": crop_id, "gt_count": asset["count"], "image_sha256": asset["row"]["image_sha256"],
                                  "mask_sha256": asset["row"]["mask_sha256"], "row": asset["row"],
                                  "before": paired["before"]["record"] if paired["before"] else None,
                                  "after": paired["after"]["record"] if paired["after"] else None}
        detail = "".join(f"<h3>{phase.title()} exact completion</h3><pre>{html.escape(saved['record']['text']) if saved else 'Unavailable'}</pre>" for phase, saved in paired.items())
        example_sections.append(f"<section><h2>{kind.title()}-nucleus example</h2><p>{html.escape(crop_id)}; {asset['count']} visible GT instances, including any edge fragments.</p><img src='{filename}' alt='Same input and GT, with before/after parsed prompts, individual raw masks and final instances'><details><summary>Exact answers and provenance</summary>{detail}<pre>{html.escape(json.dumps(asset['row'], indent=2))}</pre></details></section>")
    summary = {"run_name": name, "status": status, "supervisor_status": entry["status"], "benchmark": benchmark,
               "evaluation_scope": scope, "cohort": cohort_name, "cohort_sha256": data["manifest_sha256"][cohort_name],
               "training_manifest_sha256": data["manifest_sha256"]["train"], "examples_selection": selected,
               "selection_rule": "Lexicographically first ID with GT count exactly 1 and first with GT count >1; no output-based selection",
               "result": result, "metadata": metadata, "arguments": arguments, "suite_entry": entry,
               "phases": {phase: {k: v for k, v in p.items() if k not in ("examples", "records")} for phase, p in phases.items()},
               "examples": example_metadata, "failure": load(directory / "failure.json"), "progress": load(directory / "progress.json"),
               "source_sha256": {f.name: sha(f) for f in (directory / "result.json", directory / "arguments.json", directory / "metadata.json", directory / "training_contract.json") if f.is_file()},
               "limitations": "Full-mask engineering validation, selected 1-8-nucleus crops; validation reused and used for model selection; no official test or clinical/SOTA claim. Partial scores are not full-cohort estimates."}
    dump(target / "summary.json", summary)
    for phase, p in phases.items():
        (target / f"{phase}_records.jsonl").write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in p["records"]))
    warnings = "Benchmark8 measures runtime and early behavior; do not pool or rank these scores against validation64." if benchmark else "The validation cohort is reused for setting selection and is not an untouched final test set."
    if arguments is None or metadata is None:
        warnings += " Arguments or model metadata are unavailable; checkpoint provenance is not fully verified."
    if not all(p["complete"] for p in phases.values()):
        warnings += " One or both evaluations are missing or partial. Available scores describe only the verified records, not the full planned cohort."
    page = f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(name)}</title><style>{CSS}</style><h1>{html.escape(name)}</h1><p>{html.escape(scope)} · Status: <b>{html.escape(status)}</b></p><section class='warning'>{html.escape(warnings)}<p>{coverage}</p></section><section><h2>Before and after</h2>{table(('Metric','Before','After'), metrics)}<p>Macro metrics weight crops equally; micro recall is total TP / total GT. Count MAE uses final instance counts. Invalid answers remain in the denominator. Duplicate extras refer to raw SAM masks matching the same GT above IoU 0.5; overlap pairs are a separate proxy.</p></section><section><h2>Training and resources</h2>{table(('Item','Observed or planned value'), timing)}<p>Unavailable means no saved value; it does not mean zero. Training time may include periodic evaluation and checkpoint work; total time also includes load, before/after evaluation and final saving.</p></section>{''.join(example_sections)}<section><h2>Scope and evidence</h2><p>{html.escape(summary['limitations'])}</p><p><a href='summary.json'>Verified summary and provenance</a> · <a href='before_records.jsonl'>Every before answer</a> · <a href='after_records.jsonl'>Every after answer</a> · <a href='../index.html'>Suite overview</a></p><p>No model was run to create this report. Source records remain unchanged; accepted masks and duplicates were recomputed from saved artifacts where available. Raw-mask tiles show every saved mask, including empty masks.</p></section></html>"
    (target / "index.html").write_text(page)
    readme = f"# {name}\n\n**{status}** — {scope}.\n\n{warnings}\n\n{coverage}.\n\n" + markdown_table(("Metric", "Before", "After"), metrics) + "\n\n" + markdown_table(("Training/resources", "Observed or planned value"), timing)
    readme += "\n\nExamples are selected once by GT count and lexical ID, independent of predictions.\n\n" + "\n\n".join(f"![Paired example]({filename})" for filename in figures)
    readme += f"\n\n{summary['limitations']}\n\n[Interactive report](index.html) · [Summary/provenance](summary.json) · [All before answers](before_records.jsonl) · [All after answers](after_records.jsonl).\n"
    (target / "README.md").write_text(readme)
    return summary


def render(suite, dataset, output):
    suite, output = Path(suite).resolve(), Path(output).resolve()
    source = load(suite / "experiment_summary.json")
    if not source:
        raise ValueError("Missing experiment_summary.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty report output directory")
    data = load_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    if len({entry["name"] for entry in source["runs"]}) != len(source["runs"]):
        raise ValueError("Suite has duplicate run names")
    reports = [run_report(suite, entry, data, output) for entry in source["runs"]]
    sections = []
    for benchmark, title in ((False, "Full-validation training runs"), (True, "Separate eight-crop timing checks")):
        rows = []
        for report in reports:
            if report["benchmark"] != benchmark:
                continue
            result = report["result"] or {}
            after = report["phases"]["after"]
            metrics = after["metrics"] or {}
            rows.append((report["run_name"], report["status"], number(result.get("optimizer_steps_this_run")),
                         f"{after['verified_final_maps']}/{after['expected_crops']}", number(metrics.get("macro_pq")),
                         number(metrics.get("micro_recall")), number(metrics.get("count_mae"))))
        sections.append(f"<section><h2>{title}</h2>{table(('Run','Status','Actual updates','Verified after crops','After PQ','After micro recall','After count MAE'),rows)}</section>")
    links = "".join(f"<li><a href='{r['run_name']}/index.html'>{html.escape(r['run_name'])}</a></li>" for r in reports)
    page = f"<!doctype html><html lang='en'><meta charset='utf-8'><title>H200 training suite</title><style>{CSS}</style><h1>H200 training suite</h1><p>Saved supervisor status: <b>{html.escape(source['status'])}</b>. Every run is retained, including failures, skips, partial evaluations and unavailable results.</p><section class='warning'>Planned configuration: fresh language LoRA adapters, full-mask foreground-IoU and format rewards, and frozen SAM2.1 tiny. Saved results below establish which runs actually completed updates. Compare common-cohort scores and actual budgets. Benchmark8 scores are separate. This is validation and setting selection, not a final test.</section>{''.join(sections)}<section><h2>Per-run evidence</h2><ul>{links}</ul><p><a href='suite_summary.json'>Source suite summary</a> · <a href='verification.json'>Report verification</a></p></section></html>"
    (output / "index.html").write_text(page)
    dump(output / "suite_summary.json", source)
    dump(output / "verification.json", {"source_suite_sha256": sha(suite / "experiment_summary.json"),
         "dataset_manifest_sha256": data["manifest_sha256"], "source_assets_verified": len(data["assets"]),
         "runs": [{"name": r["run_name"], "status": r["status"], "cohort": r["cohort"],
                   "before_complete": r["phases"]["before"]["complete"], "after_complete": r["phases"]["after"]["complete"]} for r in reports],
         "selection_is_independent_of_predictions": True, "no_model_inference_or_source_mutation": True})
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.suite, args.dataset, args.output))
