"""Paired one-object versus three-object examples, optionally across checkpoints.

Inference only, on a fixed official-30 training subset. No adapters, optimizer,
validation selection, or test images. All model settings are shared by both arms.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from nucleus_rl.evaluate import prompt_for_image, write_json

VARIANTS = ("one_example", "three_examples")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def paired_prompts(width, height, *, max_objects=8, coordinate_frame="processed"):
    """Change only the numeric demonstration; keep its first object identical.

    Demonstrations depend on dimensions alone, never on image content or masks.
    The common schema and all instructions remain byte-identical.
    """
    if min(width, height) < 8 or max_objects < 3:
        raise ValueError("Paired examples require dimensions >=8 and max_objects >=3")
    single = prompt_for_image(width, height, max_objects, coordinate_frame, "example")
    text = single[-1]["content"]
    marker = "Formatting example with invented coordinates, unrelated to the supplied image:\n"
    start = text.index(marker) + len(marker)
    end = text.index("</answer>", start) + len("</answer>")
    payload = json.loads(text[start + len("<answer>"):end - len("</answer>")])
    for x0, y0, x1, y1 in (
        (5 * width // 8, height // 8, 7 * width // 8, 3 * height // 8),
        (width // 8, 5 * height // 8, 3 * width // 8, 7 * height // 8),
    ):
        payload["objects"].append({"box": [x0, y0, x1, y1],
                                   "point": [(x0 + x1) // 2, (y0 + y1) // 2]})
    replacement = "<answer>" + json.dumps(payload, separators=(",", ":")) + "</answer>"
    multiple = [dict(message) for message in single]
    multiple[-1]["content"] = text[:start] + replacement + text[end:]
    return dict(zip(VARIANTS, (single, multiple)))


def select_fixed_rows(rows, max_images=8):
    """Round-robin visible-count strata, lexicographic IDs, prefer new patients.

    Selection happens before any model output, on training data only. Counts are
    used to cover different densities, not as input to the VLM or SAM.
    """
    import numpy as np
    from PIL import Image
    from nucleus_rl.cohort import validate_training_rows

    validate_training_rows(rows)
    if not 1 <= max_images <= len(rows):
        raise ValueError("max_images must be positive and not exceed the input rows")
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["id"]):
        with Image.open(row["mask_path"]) as source:
            labels = np.asarray(source)
        count = int(len(np.unique(labels[labels > 0])))
        if not 1 <= count <= 8 or row.get("instance_count", count) != count:
            raise ValueError(f"Unexpected visible instance count for {row['id']}: {count}")
        buckets[count].append({**row, "diagnostic_gt_count": count})
    selected, patients = [], set()
    while len(selected) < max_images:
        for count in sorted(buckets):
            candidates = buckets[count]
            if not candidates:
                continue
            index = next((i for i, row in enumerate(candidates) if row["patient_id"] not in patients), 0)
            row = candidates.pop(index)
            selected.append(row)
            patients.add(row["patient_id"])
            if len(selected) == max_images:
                break
    return selected


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest; every row must belong to official30")
    parser.add_argument("--output", required=True, help="New empty output directory")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION, help="Immutable 40-character Hub commit")
    parser.add_argument("--reference-run", help="Completed paired run whose crops and inference controls must match")
    parser.add_argument("--direct-gpu-load", action="store_true", help="Stream checkpoint shards to one GPU to bound host RAM")
    return parser


def verify_reference_run(path, selection, args):
    """Reject altered crops, decoding controls, weights, or missing reference pairs."""
    from nucleus_rl.segmenter import file_sha256
    path = Path(path)
    result = json.loads((path / "result.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    if (result.get("status") != "completed_paired_inference" or result.get("optimizer_steps") != 0
            or result.get("policy_trainable_parameters") != 0 or result.get("model", {}).get("adapter") is not None):
        raise ValueError("Reference must be a completed, unadapted, frozen inference run")
    if result["selected_crops"] != selection:
        raise ValueError("Reference selection or image/mask hashes differ")
    for key in ("max_completion_length", "seed", "min_pixels", "max_pixels", "coordinate_frame", "prompt_style"):
        if plan["args"][key] != getattr(args, key):
            raise ValueError(f"Reference control differs: {key}")
    if (result["sam"]["checkpoint_sha256"] != file_sha256(args.sam_checkpoint)
            or result["sam"]["config"] != args.sam_config):
        raise ValueError("Reference SAM weights/config differ")
    records = {}
    for line in (path / "paired_records.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["id"], row["variant"])
            if key in records:
                raise ValueError("Duplicate reference pair")
            records[key] = row
    expected = {(row["id"], variant) for row in selection for variant in VARIANTS}
    if set(records) != expected:
        raise ValueError("Missing or unexpected reference pair")
    changed_checkpoint = (result["model"]["model"] != args.model
                          or result["model"]["resolved_revision"] != args.revision)
    if changed_checkpoint and any("input_ids_sha256" not in row for row in records.values()):
        raise ValueError("Cross-checkpoint reference requires recorded input token hashes; replay the base first")
    return records


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed
    from nucleus_rl.ablation_metrics import duplicate_metrics, summarize_records
    from nucleus_rl.cohort import load_official30_cohort, validate_training_rows
    from nucleus_rl.evaluate import (ArtifactScorer, assert_processed_dimensions,
                                    prepare_coordinate_rows, read_manifest, prompt_size_for_row)
    from nucleus_rl.segmenter import FrozenSAM2, file_sha256
    from nucleus_rl.train import load_policy
    from scripts.prompt_diagnostic import RawRecordingSegmenter

    started = time.monotonic()
    if args.max_completion_length <= 0:
        raise ValueError("max-completion-length must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; never overwrite an earlier run")
    rows = read_manifest(args.manifest)
    cohort = load_official30_cohort()
    input_cohort_audit = validate_training_rows(rows, cohort)
    selected = select_fixed_rows(rows, args.max_images)
    cohort_audit = validate_training_rows(selected, cohort)
    output.mkdir(parents=True, exist_ok=True)
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Use an immutable 40-character model revision")
    if args.model != MODEL and not args.reference_run:
        raise ValueError("A different checkpoint requires --reference-run for paired controls")
    args.adapter = None
    args.min_pixels, args.max_pixels = 224 ** 2, 448 ** 2
    args.coordinate_frame, args.prompt_style = "processed", "example"
    selection = [{"id": row["id"], "patient_id": row["patient_id"],
                  "source_image_id": row["source_image_id"], "gt_count": row["diagnostic_gt_count"],
                  "image_sha256": file_sha256(row["image_path"]),
                  "mask_sha256": file_sha256(row["mask_path"])} for row in selected]
    reference_records = verify_reference_run(args.reference_run, selection, args) if args.reference_run else None
    write_json(output / "plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "cohort": cohort, "input_cohort_audit": input_cohort_audit,
        "cohort_audit": cohort_audit, "selected_crops": selection,
        "manifest_sha256": file_sha256(args.manifest),
        "scope": "Inference-only TRAIN diagnostic; density-selected crops, not a held-out benchmark",
        "selection": "Round-robin ascending visible-count strata; lexical crop IDs, preferring new patients",
        "comparison": "Only numeric demonstration changes: one object versus three objects",
        "decoding": "Greedy, one completion per arm per crop; seed reset for every completion",
        "weights": "Unadapted Qwen base; no LoRA or optimization; SAM frozen",
        "cross_checkpoint_caveat": "Released checkpoints may differ in pretraining/post-training as well as parameter count",
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
    })
    torch.set_num_threads(8)
    model, processor, metadata = load_policy(args)
    model.eval()
    if reference_records is not None:
        previous_config = json.loads((Path(args.reference_run) / "model.json").read_text())["generation_config"]
        current_config = model.generation_config.to_dict()
        differing = [key for key in set(previous_config) | set(current_config)
                     if key not in ("transformers_version", "_from_model_config")
                     and previous_config.get(key) != current_config.get(key)]
        if differing:
            raise RuntimeError(f"Reference generation defaults differ: {sorted(differing)}")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Inference ablation requires all policy parameters frozen")
    selected = prepare_coordinate_rows(selected, processor, "processed")
    sam = FrozenSAM2(args.sam_checkpoint, args.sam_config)
    recorder = RawRecordingSegmenter(sam)
    scorers = {variant: ArtifactScorer(recorder, output / variant, 8) for variant in VARIANTS}
    records = []
    write_json(output / "model.json", {**metadata, "generation_config": model.generation_config.to_dict()})
    for index, row in enumerate(selected):
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            reference = np.asarray(source)
        prompts = paired_prompts(*prompt_size_for_row(row, image.size))
        pixel_hashes = []
        # Alternate arm order. There is no optimizer or stateful adaptation.
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            set_seed(args.seed)
            prompt = [dict(message) for message in prompts[variant]]
            prompt[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
            chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
            assert_processed_dimensions(processor, inputs, row)
            pixel_hashes.append(tensor_digest(inputs["pixel_values"]))
            input_ids_hash = tensor_digest(inputs["input_ids"])
            if reference_records is not None:
                reference_record = reference_records[(row["id"], variant)]
                if (reference_record["prompt"] != prompts[variant]
                        or reference_record["pixel_values_sha256"] != pixel_hashes[-1]
                        or reference_record["prompt_size"] != list(prompt_size_for_row(row, image.size))):
                    raise RuntimeError("Reference prompts, image tensor, or coordinate frame differ")
                if ("input_ids_sha256" in reference_record
                        and reference_record["input_ids_sha256"] != input_ids_hash):
                    raise RuntimeError("Reference tokenized input differs")
            inputs = inputs.to(model.device)
            generation_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_completion_length,
                                           do_sample=False, use_cache=True)
            answer_ids = generated[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(answer_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
            generation_seconds = time.monotonic() - generation_started
            recorder.last_raw = None  # Invalid outputs skip SAM; do not reuse a previous raw mask.
            result = scorers[variant].score(text, {**row, "prompt_style": variant})
            raw_masks = (recorder.last_raw["masks"] if recorder.last_raw is not None
                         else np.zeros((0, image.height, image.width), dtype=bool))
            raw_path = output / variant / f"{row['id']}_raw_masks.npz"
            np.savez_compressed(raw_path, masks=raw_masks)
            result.update(duplicate_metrics(raw_masks, reference))
            result.update({"variant": variant, "prompt": prompts[variant],
                           "input_tokens": int(inputs["input_ids"].shape[1]),
                           "completion_tokens": int(answer_ids.shape[1]),
                           "hit_token_cap": int(answer_ids.shape[1]) == args.max_completion_length,
                           "generation_seconds": generation_seconds,
                           "pixel_values_sha256": pixel_hashes[-1],
                           "input_ids_sha256": input_ids_hash,
                           "raw_masks_path": str(raw_path),
                           "image_path": row["image_path"], "mask_path": row["mask_path"]})
            records.append(result)
            with (output / "paired_records.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({"crop": index + 1, "variant": variant, "id": row["id"],
                              **{key: result[key] for key in ("pq", "recall", "pred_count", "gt_count",
                                 "count_error", "format_valid", "same_gt_duplicate_count", "raw_overlap_pair_count")}}), flush=True)
        if pixel_hashes[0] != pixel_hashes[1]:
            raise RuntimeError("Paired arms received different preprocessed image tensors")
    summaries = {variant: summarize_records([r for r in records if r["variant"] == variant])
                 for variant in VARIANTS}
    for variant in VARIANTS:
        summaries[variant]["token_cap_hits"] = sum(r["hit_token_cap"] for r in records if r["variant"] == variant)
        scorers[variant].summary()
    result = {"status": "completed_paired_inference", "variants": summaries,
              "selected_crops": selection, "model": metadata, "sam": sam.verify_unchanged(),
              "policy_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "optimizer_steps": 0, "all_paired_pixels_identical": True,
              "reference_run": args.reference_run,
              "reference_controls_verified": reference_records is not None,
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
              "elapsed_seconds": time.monotonic() - started}
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected policy gradients in inference-only experiment")
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
