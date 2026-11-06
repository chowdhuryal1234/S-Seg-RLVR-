"""Sampled support of an SFT prompt policy on validation tiles: which GT nuclei does at least one of K samples hit?

For each tile, K completions are drawn at temperature T from the policy itself (no top-k/top-p truncation, no
repetition penalty: the GRPO sampling distribution). Boxes are mapped to original pixels and matched one-to-one to
GT instance boxes at box IoU > 0.5 (greedy, highest IoU first), as in the gap analysis (never_found.py).

Reports, pooled and per dataset:
  per_sample_recall  mean over samples of matched GT / GT (what one sample finds)
  union_recall       GT hit by at least one of the K samples (the policy's support)
  never_found        GT hit by none of the K samples, also without edge slivers (truncated and <100 px)
  all_found          GT hit by every sample
No SAM is involved: this measures where the policy points, not mask quality. Validation tiles only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--init-adapter", required=True, help="SFT adapter, merged into the base before sampling")
    parser.add_argument("--coordinate-frame", choices=("processed", "normalized1000"), default="normalized1000")
    parser.add_argument("--min-pixels", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--max-objects", type=int, required=True, help="Same answer cap as in training (prompt text)")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Canary only: first N tiles")
    parser.add_argument("--wall-time-minutes", type=float, help="Stop sampling new tiles after this; summary marks it partial")
    return parser


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_instances(mask, sliver_ids=None, truncated_ids=None):
    """GT boxes [x0, y0, x1, y1) in pixels from an instance mask; sliver flag from the manifest when present."""
    import numpy as np
    labels = np.asarray(mask).astype(np.int64)
    slivers = set(sliver_ids or [])
    truncated = set(truncated_ids or [])
    result = []
    for gid in np.unique(labels[labels > 0]):
        yx = np.argwhere(labels == gid)
        area = int(len(yx))
        sliver = int(gid) in slivers if sliver_ids is not None else (int(gid) in truncated and area < 100)
        result.append({"gid": int(gid), "area": area, "sliver": bool(sliver),
                       "box": [int(yx[:, 1].min()), int(yx[:, 0].min()), int(yx[:, 1].max()) + 1, int(yx[:, 0].max()) + 1]})
    return result


def match(pred_boxes, gt_boxes, threshold=0.5):
    """Indices of GT boxes matched one-to-one by greedy highest-IoU pairing at IoU > threshold."""
    pairs = sorted(((box_iou(p, g), i, j) for i, p in enumerate(pred_boxes) for j, g in enumerate(gt_boxes)),
                   reverse=True)
    used_pred, used_gt = set(), set()
    for value, i, j in pairs:
        if value <= threshold:
            break
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)
    return used_gt


def tile_support(gt, sample_hits):
    """Counts for one tile. sample_hits: one set of matched GT indices per sample (empty for invalid samples)."""
    k = len(sample_hits)
    if k == 0:
        raise ValueError("At least one sample is required")
    counts = [sum(j in hits for hits in sample_hits) for j in range(len(gt))]
    keep = [not item["sliver"] for item in gt]
    return {"gt": len(gt), "gt_no_sliver": sum(keep), "samples": k,
            "matched_per_sample": [len(hits) for hits in sample_hits],
            "never": sum(c == 0 for c in counts),
            "never_no_sliver": sum(c == 0 and kept for c, kept in zip(counts, keep)),
            "all_found": sum(c == k for c in counts),
            "hit_counts": counts}


def aggregate(records):
    """Pooled and per-dataset support summary from tile records (each with 'dataset' and tile_support fields)."""
    groups = {"ALL": records}
    for record in records:
        groups.setdefault(record["dataset"], []).append(record)
    out = {}
    for name, rows in sorted(groups.items()):
        gt = sum(r["gt"] for r in rows)
        gt_ns = sum(r["gt_no_sliver"] for r in rows)
        matched = sum(sum(r["matched_per_sample"]) / r["samples"] for r in rows)
        out[name] = {"tiles": len(rows), "gt": gt, "gt_no_sliver": gt_ns,
                     "per_sample_recall": matched / gt if gt else None,
                     "union_recall": 1 - sum(r["never"] for r in rows) / gt if gt else None,
                     "never_found": sum(r["never"] for r in rows) / gt if gt else None,
                     "never_found_no_sliver": sum(r["never_no_sliver"] for r in rows) / gt_ns if gt_ns else None,
                     "all_found": sum(r["all_found"] for r in rows) / gt if gt else None,
                     "invalid_samples": sum(r.get("invalid", 0) for r in rows),
                     "unfinished_samples": sum(r.get("unfinished", 0) for r in rows),
                     "samples": sum(r["samples"] for r in rows)}
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.samples < 1 or not math.isfinite(args.temperature) or args.temperature <= 0 \
            or not 0 < args.iou_threshold < 1 or args.max_new_tokens < 1:
        raise ValueError("Require samples >= 1, temperature > 0, 0 < iou-threshold < 1, max-new-tokens >= 1")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; existing artifacts are preserved")
    import numpy as np
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import GenerationConfig
    from nucleus_rl.evaluate import (assert_processed_dimensions, objects_to_original, prepare_coordinate_rows,
                                     read_manifest)
    from nucleus_rl.grpo import check_init_adapter, completion_lengths, end_token_ids, repeat_inputs
    from nucleus_rl.rewards import parse_completion
    from nucleus_rl.sft import inference_inputs, json_write, load_model
    from nucleus_rl.sft_data import sft_prompt_messages

    rows = read_manifest(args.eval_manifest)
    if args.limit:
        rows = rows[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "arguments.json", vars(args))
    init = check_init_adapter(args)
    model, processor = load_model(SimpleArgs(args))
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=False).merge_and_unload()
    model.requires_grad_(False)
    model.eval()
    eos_ids = end_token_ids(model, processor)
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_ids[0]
    sampling = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                eos_token_id=eos_ids, pad_token_id=pad_id, use_cache=True)
    prepared = prepare_coordinate_rows(rows, processor, args.coordinate_frame)
    json_write(output / "metadata.json", {"init_adapter": init, "sampling": sampling.to_dict(),
                                          "tiles_requested": len(prepared), "gpu": torch.cuda.get_device_name()})
    started = time.monotonic()
    records, stop_reason = [], None
    for index, row in enumerate(prepared):
        if args.wall_time_minutes and time.monotonic() - started >= args.wall_time_minutes * 60:
            stop_reason = "wall_time_budget"
            break
        tile_started = time.monotonic()
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        example = {"image": image, "messages": sft_prompt_messages(
            row["prompt_width"], row["prompt_height"], coordinate_mode=args.coordinate_frame,
            max_objects=args.max_objects)}
        _, inputs = inference_inputs(processor, example)
        assert_processed_dimensions(processor, inputs, row)
        torch.manual_seed(args.seed * 1_000_003 + index)
        torch.cuda.manual_seed_all(args.seed * 1_000_003 + index)
        with torch.inference_mode():
            generated = model.generate(**{key: value.to(model.device) for key, value in
                                          repeat_inputs(inputs, args.samples).items()}, generation_config=sampling)
        sampled = generated[:, inputs["input_ids"].shape[1]:].cpu().tolist()
        lengths = completion_lengths(sampled, eos_ids)
        texts = processor.batch_decode([tokens[:length] for tokens, (length, _) in zip(sampled, lengths)],
                                       skip_special_tokens=True, clean_up_tokenization_spaces=False)
        with Image.open(row["mask_path"]) as source:
            gt = gt_instances(np.asarray(source), row.get("edge_sliver_instance_ids"), row.get("truncated_instance_ids"))
        prompt_size = (row["prompt_width"], row["prompt_height"])
        hits, objects, invalid = [], [], 0
        for text in texts:
            parsed = parse_completion(text, width=prompt_size[0], height=prompt_size[1], max_objects=args.max_objects)
            if not parsed.valid:
                invalid += 1
                hits.append(set())
                objects.append(0)
                continue
            converted, _ = objects_to_original(parsed.objects, prompt_size, image.size)
            hits.append(match([list(obj.box) for obj in converted], [item["box"] for item in gt], args.iou_threshold))
            objects.append(len(converted))
        record = {"id": row["id"], "dataset": row.get("dataset", "monuseg"), **tile_support(gt, hits),
                  "invalid": invalid, "unfinished": sum(not finished for _, finished in lengths),
                  "objects": objects, "tokens": [length for length, _ in lengths],
                  "seconds": time.monotonic() - tile_started}
        records.append(record)
        with (output / "tiles.jsonl").open("a") as stream:
            stream.write(json.dumps({**record, "gids": [item["gid"] for item in gt],
                                     "slivers": [item["gid"] for item in gt if item["sliver"]]}) + "\n")
        with (output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": row["id"], "texts": texts}) + "\n")
        json_write(output / "progress.json", {"tiles_done": len(records), "tiles_requested": len(prepared),
                                              "elapsed_seconds": time.monotonic() - started})
    summary = {"status": "partial" if stop_reason else "complete", "stop_reason": stop_reason,
               "tiles_done": len(records), "tiles_requested": len(prepared), "samples_per_tile": args.samples,
               "temperature": args.temperature, "iou_threshold": args.iou_threshold,
               "matching": "box IoU > threshold, greedy one-to-one, GT boxes from the instance mask (no SAM)",
               "elapsed_seconds": time.monotonic() - started, "init_adapter": init, "support": aggregate(records)}
    json_write(output / "summary.json", summary)
    return summary


class SimpleArgs:
    """The fields sft.load_model reads, taken from this script's arguments."""
    def __init__(self, args):
        self.model, self.revision, self.seed = args.model, args.revision, args.seed
        self.min_pixels, self.max_pixels = args.min_pixels, args.max_pixels


if __name__ == "__main__":
    main()
