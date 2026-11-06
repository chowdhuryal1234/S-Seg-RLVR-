"""2x2 zoom inference probe: is the policy's miss on "truly unseen" nuclei a resolution problem?

Zero training. For each validation tile (128x128) the finished policy is run greedily on four overlapping
sub-crops (80x80 at offsets 0 and 48, i.e. 32 px overlap), exactly as the normal evaluation runs it on a
tile: same prompt (sft_prompt_messages), same coordinate frame, same processor min/max pixels (200704),
so each 80 px sub-crop is resized to the full 448x448 budget (~1.6x the usual magnification: a SCALE
CONFOUND, since the policy was trained on 128 px tiles at 3.5x). Sub-crop answers are parsed with the same
parser, mapped back to tile pixels, merged across sub-crops, and the merged prompts (cap 32 = 4 x 8, nothing
lower) go through frozen SAM2.1-tiny ONCE on the full tile. Scoring reuses nucleus_rl.evaluate.ArtifactScorer
(the scorer inside nucleus_rl.sft.evaluate_policy), with the merged prompts written as an answer in the
tile's original pixel frame.

The plain (non-zoom) evaluation of the same policy runs in the same process through grpo.evaluate ->
sft.evaluate_policy, as the same-code, same-GPU control.

Merge rule (between DIFFERENT sub-crops only; objects of one sub-crop are never merged with each other):
two objects are duplicates when their boxes have IoU > 0.5, or either one's point lies inside the other's
box. Of a duplicate pair, the object whose box centre is farther from its sub-crop's interior border (a
border that cuts the tile; borders on the tile edge are shared by every sub-crop covering that edge, so
they cannot tell the copies apart) is kept. Implemented greedily: objects sorted by that distance
(descending; ties by sub-crop index, then answer order) are kept unless they duplicate an already kept
object from another sub-crop.

Pre-registered readout: see PREREGISTERED below; analysis() writes results.json and RESULTS.md.

Validation data only. The official test sets are never read.

Example (FASRC):
    python scripts/zoom_eval.py --manifest $TRAIN --eval-manifest $VAL_SUITE --output $W/runs/zoom-s2c256 \
        --model Qwen/Qwen3.6-27B --revision $REV_27B --init-adapter <S2 checkpoint-256> \
        --sam-checkpoint $SAM_CKPT --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 \
        --canary 4
Re-run only the CPU analysis of a finished run:
    python scripts/zoom_eval.py --analyze-only <run dir> --eval-manifest <manifest>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

PREREGISTERED = (
    "Pre-registered readout (written before any result): on the 170-tile suite, zoom vs plain for the same "
    "adapter, same code, same GPU, same frozen SAM2.1-tiny: (1) coverage of GT = fraction of GT nuclei with a "
    "prompt box of box-IoU > 0.5; (2) TP/FP/FN (mask IoU > 0.5) and mean per-tile PQ, pooled and per dataset; "
    "(3) on the GT nuclei for which the PLAIN eval had no box near (best box IoU < 0.1): how many stay FN under "
    "zoom. Zoom PASSES iff the no-box FN count drops by >= 30% relative AND pooled TP rises by >= 15 AND pooled "
    "FP does not double (FP_zoom < 2 x FP_plain). Caveat: sub-crops are seen at ~1.6x the usual magnification "
    "(80 px -> 448 px instead of 128 px -> 448 px), a scale the policy never trained on; a fail does not rule out "
    "resolution as the cause, and a pass mixes resolution with context reduction (fewer nuclei per view)."
)
NO_BOX_IOU = 0.1
MATCH_IOU = 0.5


# ----------------------------------------------------------------------------- geometry (CPU, tested)

def subcrop_offsets(tile, sub):
    """Two offsets per axis: 0 and tile-sub (both equal when sub == tile)."""
    if not (isinstance(tile, int) and isinstance(sub, int) and 0 < sub <= tile):
        raise ValueError("Require integer 0 < sub <= tile")
    return sorted({0, tile - sub})


def subcrop_windows(width, height, sub):
    """(x0, y0, x1, y1) windows in row-major order: (0,0), (dx,0), (0,dy), (dx,dy)."""
    return [(ox, oy, ox + sub, oy + sub)
            for oy in subcrop_offsets(height, sub) for ox in subcrop_offsets(width, sub)]


def to_tile(obj, window, prompt_size):
    """Map one parsed object (box, point in the sub-crop's prompt frame) to tile pixels."""
    x0, y0, x1, y1 = window
    sx, sy = (x1 - x0) / prompt_size[0], (y1 - y0) / prompt_size[1]
    box, point = obj.box, obj.point
    return {"box": [x0 + box[0] * sx, y0 + box[1] * sy, x0 + box[2] * sx, y0 + box[3] * sy],
            "point": [x0 + point[0] * sx, y0 + point[1] * sy]}


def interior_border_distance(box, window, tile_size):
    """Distance from the box centre to the nearest sub-crop border that cuts the tile (inf if none)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0, x1, y1 = window
    width, height = tile_size
    distances = []
    if x0 > 0:
        distances.append(cx - x0)
    if x1 < width:
        distances.append(x1 - cx)
    if y0 > 0:
        distances.append(cy - y0)
    if y1 < height:
        distances.append(y1 - cy)
    return min(distances) if distances else math.inf


def box_iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def point_in_box(point, box):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def duplicates(a, b, iou_threshold=0.5):
    return (box_iou(a["box"], b["box"]) > iou_threshold
            or point_in_box(a["point"], b["box"]) or point_in_box(b["point"], a["box"]))


def merge_objects(objects, iou_threshold=0.5):
    """objects: dicts with box, point, sub (sub-crop index), index (answer order), border_distance.

    Returns (kept, dropped); each dropped item gains "duplicate_of" = (sub, index) of the kept object.
    """
    order = sorted(objects, key=lambda o: (-o["border_distance"], o["sub"], o["index"]))
    kept, dropped = [], []
    for obj in order:
        twin = next((k for k in kept if k["sub"] != obj["sub"] and duplicates(k, obj, iou_threshold)), None)
        if twin is None:
            kept.append(obj)
        else:
            dropped.append({**obj, "duplicate_of": [twin["sub"], twin["index"]]})
    return kept, dropped


def answer_text(objects):
    """Merged prompts as a parseable answer in tile pixels (exact float repr, no rounding)."""
    payload = {"objects": [{"box": list(o["box"]), "point": list(o["point"])} for o in objects]}
    return "<answer>" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + "</answer>"


# ----------------------------------------------------------------------------- per-GT analysis (CPU, tested)

def gt_instances(reference):
    """GT ids, half-open boxes (localization_metrics convention), areas, border contact."""
    import numpy as np
    labels = np.asarray(reference)
    height, width = labels.shape
    result = []
    for gid in np.unique(labels[labels > 0]):
        ys, xs = np.nonzero(labels == gid)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        result.append({"gt_id": int(gid), "box": box, "area": int(len(xs)),
                       "touches_border": bool(box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height)})
    return result


def matched_gt_ids(instances, reference, threshold=MATCH_IOU):
    """GT ids with a predicted instance of mask IoU > threshold (one-to-one for disjoint maps at 0.5)."""
    import numpy as np
    pred, gt = np.asarray(instances), np.asarray(reference)
    pred_ids, pred_inv, pred_area = np.unique(pred, return_inverse=True, return_counts=True)
    gt_ids, gt_inv, gt_area = np.unique(gt, return_inverse=True, return_counts=True)
    inter = np.bincount(pred_inv.ravel() * len(gt_ids) + gt_inv.ravel(),
                        minlength=len(pred_ids) * len(gt_ids)).reshape(len(pred_ids), len(gt_ids))
    union = pred_area[:, None] + gt_area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros(inter.shape, dtype=float), where=union > 0)
    iou = iou[np.ix_(pred_ids > 0, gt_ids > 0)]
    hit = (iou > threshold).any(axis=0)
    return {int(g) for g, h in zip(gt_ids[gt_ids > 0], hit) if h}


def per_gt_status(reference, prompt_boxes, instances):
    """One row per GT nucleus: best prompt-box IoU and whether its mask was matched (TP)."""
    matched = matched_gt_ids(instances, reference)
    rows = []
    for inst in gt_instances(reference):
        best = max((box_iou(inst["box"], box) for box in prompt_boxes), default=0.)
        rows.append({**inst, "best_box_iou": best, "matched": inst["gt_id"] in matched})
    return rows


# ----------------------------------------------------------------------------- GPU inference

def load_policy(args):
    """grpo.evaluate_only's loading path: merged init adapter, optional eval adapter on top."""
    from peft import PeftModel
    from nucleus_rl.grpo import check_init_adapter
    from nucleus_rl.sft import load_model
    from nucleus_rl.training_budget import file_digest
    init = check_init_adapter(args)
    model, processor = load_model(args)
    policy = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=False).merge_and_unload()
    evaluated = None
    if args.eval_adapter:
        path = Path(args.eval_adapter)
        coordinates = json.loads((path / "coordinate_config.json").read_text())
        if coordinates.get("init_adapter_sha256") != init["adapter_sha256"]:
            raise ValueError("--eval-adapter was trained from a different SFT adapter than --init-adapter")
        policy = PeftModel.from_pretrained(policy, str(path), is_trainable=False)
        evaluated = {"path": str(path.resolve()),
                     "adapter_sha256": file_digest(path / "adapter_model.safetensors"),
                     "coordinate_config": coordinates}
    policy.requires_grad_(False)
    return policy, processor, init, evaluated


def generate_answer(model, processor, image, args):
    """One greedy answer for one image, exactly as sft.evaluate_policy generates it."""
    import torch
    from nucleus_rl.evaluate import processed_image_size
    from nucleus_rl.sft import inference_inputs
    from nucleus_rl.sft_data import sft_prompt_messages
    if args.coordinate_frame == "normalized1000":
        prompt_size = (1000, 1000)
    else:
        prompt_size = processed_image_size(processor, processor.image_processor(images=image, return_tensors="pt"))
    example = {"image": image, "messages": sft_prompt_messages(*prompt_size, coordinate_mode=args.coordinate_frame,
                                                               max_objects=args.max_objects)}
    _, inputs = inference_inputs(processor, example)
    processed = processed_image_size(processor, inputs)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.eval_max_completion_length,
                                   use_cache=True)
    tokens = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text, prompt_size, processed, int(tokens.shape[1])


def zoom_evaluate(model, processor, rows, segmenter, directory, args):
    from PIL import Image
    from nucleus_rl.evaluate import ArtifactScorer, processed_image_size
    from nucleus_rl.rewards import parse_completion
    from nucleus_rl.training_budget import preserve_evaluation_state
    directory = Path(directory)
    scorer = ArtifactScorer(segmenter, directory, args.merged_max_objects)
    with preserve_evaluation_state(model):
        model.eval()
        for row in rows:
            started = time.monotonic()
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            tile_processed = processed_image_size(processor, processor.image_processor(images=image, return_tensors="pt"))
            windows = subcrop_windows(image.width, image.height, args.sub_size)
            subs, objects = [], []
            for sub, window in enumerate(windows):
                text, prompt_size, processed, n_tokens = generate_answer(model, processor, image.crop(window), args)
                if processed != tile_processed:
                    raise ValueError(f"Sub-crop processed size {processed} differs from the tile's {tile_processed}")
                parsed = parse_completion(text, width=prompt_size[0], height=prompt_size[1],
                                          max_objects=args.max_objects)
                mapped = []
                for index, obj in enumerate(parsed.objects if parsed.valid else ()):
                    item = {**to_tile(obj, window, prompt_size), "sub": sub, "index": index}
                    item["border_distance"] = interior_border_distance(item["box"], window, image.size)
                    mapped.append(item)
                objects.extend(mapped)
                subs.append({"sub": sub, "window": list(window), "processed_size": list(processed),
                             "text": text, "tokens": n_tokens, "format_valid": parsed.valid,
                             "parse_error": parsed.error, "objects": len(mapped),
                             "at_cap": len(mapped) == args.max_objects})
            kept, dropped = merge_objects(objects)
            text = answer_text(kept)
            record = scorer.score(text, {**row, "coordinate_frame": "original"})
            if not record["format_valid"]:
                raise ValueError(f"Merged answer failed to parse for {row['id']}: {record['parse_error']}")
            scorer.record_group([record])

            def clean(item):
                return {**item, "border_distance": None if math.isinf(item["border_distance"])
                        else item["border_distance"]}
            with (directory / "zoom_details.jsonl").open("a") as stream:
                stream.write(json.dumps({
                    "id": row["id"], "dataset": row.get("dataset", "monuseg"), "subcrops": subs,
                    "all_objects": [clean(o) for o in objects], "kept": [clean(o) for o in kept],
                    "dropped": [clean(o) for o in dropped], "seconds": time.monotonic() - started,
                }, allow_nan=False) + "\n")
    return scorer.summary()


# ----------------------------------------------------------------------------- analysis (CPU)

def _load_mode(directory):
    path = Path(directory) / "completions.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def _sum(records, key):
    return int(sum(r[key] for r in records))


def _block(records, statuses):
    n = len(records)
    gt = sum(len(s) for s in statuses)
    return {"tiles": n, "gt": gt, "tp": _sum(records, "tp"), "fp": _sum(records, "fp"), "fn": _sum(records, "fn"),
            "pred": _sum(records, "pred_count"),
            "mean_pq": sum(r["pq"] for r in records) / n if n else None,
            "invalid": sum(not r["format_valid"] for r in records),
            "coverage": (sum(g["best_box_iou"] > MATCH_IOU for s in statuses for g in s) / gt) if gt else None}


def analysis(run_dir, manifest_rows):
    """Compare zoom vs plain in run_dir; writes results.json and RESULTS.md there."""
    import numpy as np
    from PIL import Image
    run_dir = Path(run_dir)
    plain, zoom = _load_mode(run_dir / "plain"), _load_mode(run_dir / "zoom")
    details = {}
    if (run_dir / "zoom" / "zoom_details.jsonl").exists():
        details = {d["id"]: d for d in map(json.loads, (run_dir / "zoom" / "zoom_details.jsonl").read_text().splitlines())}
    ids = [r["id"] for r in manifest_rows if r["id"] in plain and r["id"] in zoom]
    rows = {r["id"]: r for r in manifest_rows}
    per = {"plain": {}, "zoom": {}, "zoom_unmerged": {}}
    for tid in ids:
        with Image.open(rows[tid]["mask_path"]) as source:
            reference = np.asarray(source)
        for mode, records in (("plain", plain), ("zoom", zoom)):
            rec = records[tid]
            instances = np.load(run_dir / mode / Path(rec["prediction_path"]).name, allow_pickle=False)
            boxes = [o["box"] for o in rec["converted_objects"]]
            per[mode][tid] = per_gt_status(reference, boxes, instances)
        if tid in details:
            boxes = [o["box"] for o in details[tid]["all_objects"]]
            per["zoom_unmerged"][tid] = [{**g, "best_box_iou": max((box_iou(g["box"], b) for b in boxes), default=0.)}
                                         for g in per["zoom"][tid]]
    datasets = sorted({rows[t].get("dataset", "monuseg") for t in ids})
    result = {"preregistered": PREREGISTERED, "tiles": len(ids), "pooled": {}, "per_dataset": {}}
    for scope in ["pooled"] + datasets:
        scoped = [t for t in ids if scope == "pooled" or rows[t].get("dataset", "monuseg") == scope]
        block = {mode: _block([records[t] for t in scoped], [per[mode][t] for t in scoped])
                 for mode, records in (("plain", plain), ("zoom", zoom))}
        if all(t in per["zoom_unmerged"] for t in scoped) and scoped:
            gt = sum(len(per["zoom_unmerged"][t]) for t in scoped)
            block["zoom_unmerged_coverage"] = sum(g["best_box_iou"] > MATCH_IOU for t in scoped
                                                  for g in per["zoom_unmerged"][t]) / gt if gt else None
        # Nuclei the PLAIN eval had no box near.
        nobox = [(t, i) for t in scoped for i, g in enumerate(per["plain"][t]) if g["best_box_iou"] < NO_BOX_IOU]
        plain_fn = sum(not per["plain"][t][i]["matched"] for t, i in nobox)
        zoom_fn = sum(not per["zoom"][t][i]["matched"] for t, i in nobox)
        block["no_box"] = {
            "count": len(nobox), "plain_fn": plain_fn, "zoom_fn": zoom_fn,
            "relative_fn_drop": (plain_fn - zoom_fn) / plain_fn if plain_fn else None,
            "zoom_box_covered": sum(per["zoom"][t][i]["best_box_iou"] > MATCH_IOU for t, i in nobox),
            "zoom_any_box_near": sum(per["zoom"][t][i]["best_box_iou"] >= NO_BOX_IOU for t, i in nobox),
            "touching_border": sum(per["plain"][t][i]["touches_border"] for t, i in nobox),
            "zoom_fn_interior_only": sum(not per["zoom"][t][i]["matched"] and not per["zoom"][t][i]["touches_border"]
                                         for t, i in nobox),
            "plain_fn_interior_only": sum(not per["plain"][t][i]["matched"] and not per["plain"][t][i]["touches_border"]
                                          for t, i in nobox),
        }
        # Nuclei gained/lost by zoom (TP status flips).
        pairs = [(per["plain"][t][i], per["zoom"][t][i]) for t in scoped for i in range(len(per["plain"][t]))]
        block["flips"] = {"gained": sum(z["matched"] and not p["matched"] for p, z in pairs),
                          "lost": sum(p["matched"] and not z["matched"] for p, z in pairs)}
        if scope == "pooled":
            result["pooled"] = block
        else:
            result["per_dataset"][scope] = block
    if details:
        subs = [s for d in details.values() if d["id"] in ids for s in d["subcrops"]]
        result["subcrops"] = {"count": len(subs), "invalid": sum(not s["format_valid"] for s in subs),
                              "at_cap": sum(s["at_cap"] for s in subs),
                              "mean_objects": sum(s["objects"] for s in subs) / len(subs) if subs else None,
                              "objects_before_merge": sum(len(d["all_objects"]) for d in details.values() if d["id"] in ids),
                              "objects_after_merge": sum(len(d["kept"]) for d in details.values() if d["id"] in ids),
                              "max_merged_per_tile": max((len(d["kept"]) for d in details.values() if d["id"] in ids),
                                                         default=0),
                              "processed_sizes": sorted({tuple(s["processed_size"]) for s in subs})}
    p, z = result["pooled"]["plain"], result["pooled"]["zoom"]
    drop = result["pooled"]["no_box"]["relative_fn_drop"]
    result["verdict"] = {
        "no_box_fn_drop_ge_30pct": drop is not None and drop >= 0.30,
        "tp_gain_ge_15": z["tp"] - p["tp"] >= 15,
        "fp_not_doubled": z["fp"] < 2 * p["fp"],
        "tp_delta": z["tp"] - p["tp"], "fp_delta": z["fp"] - p["fp"], "no_box_relative_fn_drop": drop,
    }
    v = result["verdict"]
    v["pass"] = v["no_box_fn_drop_ge_30pct"] and v["tp_gain_ge_15"] and v["fp_not_doubled"]
    from nucleus_rl.training_budget import write_json
    write_json(run_dir / "results.json", result)
    (run_dir / "RESULTS.md").write_text(render_markdown(result, run_dir))
    return result


def render_markdown(result, run_dir):
    def f(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    lines = [f"# Zoom probe results: {Path(run_dir).name}", "", PREREGISTERED, "",
             f"Tiles scored: {result['tiles']}", "",
             "| scope | mode | tiles | GT | TP | FP | FN | mean PQ | box coverage | invalid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    scopes = [("pooled", result["pooled"])] + sorted(result["per_dataset"].items())
    for name, block in scopes:
        for mode in ("plain", "zoom"):
            b = block[mode]
            lines.append(f"| {name} | {mode} | {b['tiles']} | {b['gt']} | {b['tp']} | {b['fp']} | {b['fn']} | "
                         f"{f(b['mean_pq'])} | {f(b['coverage'])} | {b['invalid']} |")
    lines += ["", "No-box nuclei (plain best box IoU < 0.1):", "",
              "| scope | count | plain FN | zoom FN | rel. FN drop | zoom box IoU>0.5 | zoom any box >=0.1 | "
              "touch border | gained | lost | zoom unmerged coverage |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, block in scopes:
        n = block["no_box"]
        lines.append(f"| {name} | {n['count']} | {n['plain_fn']} | {n['zoom_fn']} | {f(n['relative_fn_drop'])} | "
                     f"{n['zoom_box_covered']} | {n['zoom_any_box_near']} | {n['touching_border']} | "
                     f"{block['flips']['gained']} | {block['flips']['lost']} | {f(block.get('zoom_unmerged_coverage'))} |")
    v = result["verdict"]
    lines += ["", f"Verdict: {'PASS' if v['pass'] else 'FAIL'} (no-box FN drop {f(v['no_box_relative_fn_drop'])} "
              f">= 0.30: {v['no_box_fn_drop_ge_30pct']}; TP delta {v['tp_delta']} >= 15: {v['tp_gain_ge_15']}; "
              f"FP delta {v['fp_delta']}, not doubled: {v['fp_not_doubled']})"]
    if "subcrops" in result:
        lines += ["", "Sub-crops: " + json.dumps(result["subcrops"], default=list)]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- driver

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyze-only", help="Run only the CPU analysis of this finished run directory")
    parser.add_argument("--manifest", help="Training manifest (split-disjointness audit only; never evaluated)")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision")
    parser.add_argument("--init-adapter")
    parser.add_argument("--eval-adapter")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--eval-sam-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--coordinate-frame", choices=("processed", "normalized1000"), default="normalized1000")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-objects", type=int, default=8, help="Per-answer cap in the prompt and parser")
    parser.add_argument("--merged-max-objects", type=int, default=32, help="Cap on merged prompts (4 x 8)")
    parser.add_argument("--eval-max-completion-length", type=int, default=1024)
    parser.add_argument("--sub-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, help="Evaluate only the first N validation tiles")
    parser.add_argument("--canary", type=int, default=0,
                        help="First run plain+zoom on the first N tiles into <output>/canary and gate on them")
    return parser


def run_modes(policy, processor, rows, sam, directory, args):
    from nucleus_rl.grpo import evaluate
    timings = {}
    started = time.monotonic()
    plain = evaluate(policy, processor, rows, sam, Path(directory) / "plain", args)
    timings["plain_seconds"] = time.monotonic() - started
    started = time.monotonic()
    zoom = zoom_evaluate(policy, processor, rows, sam, Path(directory) / "zoom", args)
    timings["zoom_seconds"] = time.monotonic() - started
    return plain, zoom, timings


def main(argv=None):
    args = build_parser().parse_args(argv)
    from nucleus_rl.evaluate import read_manifest
    validation = read_manifest(args.eval_manifest)
    if any(row.get("split") != "validation" for row in validation):
        raise ValueError("Validation-only manifest required")
    if args.analyze_only:
        result = analysis(args.analyze_only, validation)
        print(json.dumps(result["verdict"]))
        return result
    for key in ("manifest", "output", "revision", "init_adapter", "sam_checkpoint"):
        if not getattr(args, key):
            raise SystemExit(f"--{key.replace('_', '-')} is required")
    if args.merged_max_objects < 4 * args.max_objects:
        raise ValueError("Merged cap must not cut below 4 x max-objects")
    from nucleus_rl.sft import MAX_OBJECTS, json_write
    from nucleus_rl.sft_data import validate_sft_splits
    audit = validate_sft_splits(read_manifest(args.manifest), validation, verify_files=True,
                                max_objects=args.max_objects, validation_max_objects=MAX_OBJECTS)
    if args.eval_limit:
        validation = validation[:args.eval_limit]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing experiment artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "arguments.json", vars(args))
    json_write(output / "preregistered.json", {"text": PREREGISTERED, "written_at": time.strftime("%FT%T%z")})
    try:
        from nucleus_rl.segmenter import FrozenSAM2
        policy, processor, init, evaluated = load_policy(args)
        sam = FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.eval_sam_device)
        result = {"status": "zoom_probe", "init_adapter": init, "eval_adapter": evaluated, "data_audit": audit,
                  "windows": subcrop_windows(128, 128, args.sub_size)}
        if args.canary:
            rows = validation[:args.canary]
            plain, zoom, timings = run_modes(policy, processor, rows, sam, output / "canary", args)
            canary = analysis(output / "canary", rows)
            subs = canary.get("subcrops", {})
            result["canary"] = {"plain": plain, "zoom": zoom, "timings": timings, "verdict": canary["verdict"],
                                "subcrops": subs}
            json_write(output / "progress.json", {"phase": "canary_done", **timings})
            if plain["invalid_completions"] > len(rows) / 2 or subs.get("invalid", 0) > subs.get("count", 0) / 2:
                raise RuntimeError(f"Canary gate failed: plain invalid {plain['invalid_completions']}, "
                                   f"sub-crop invalid {subs.get('invalid')}/{subs.get('count')}")
        json_write(output / "progress.json", {"phase": "full", "tiles": len(validation)})
        plain, zoom, timings = run_modes(policy, processor, validation, sam, output, args)
        result.update({"plain": plain, "zoom": zoom, "timings": timings,
                       "sam_freeze_verification": sam.verify_unchanged()})
        json_write(output / "summary.json", result)
        final = analysis(output, validation)
        json_write(output / "progress.json", {"phase": "complete", "verdict": final["verdict"]})
        print(json.dumps(final["verdict"]))
        return result
    except Exception as error:
        json_write(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
