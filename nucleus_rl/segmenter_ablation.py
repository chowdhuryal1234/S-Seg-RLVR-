"""Experiment D: replay fixed A prompts through frozen SAM2.1 tiny and large.

No VLM is loaded or called. The separate oracle arm uses GT-derived tight boxes
and interior points and is never pooled with the saved-policy prompt arms.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from nucleus_rl.evaluate import write_json
from nucleus_rl.segmenter import file_sha256


POLICY_ARMS = ("one_example", "three_examples")
ORACLE_ARM = "oracle_tight"
ARMS = (*POLICY_ARMS, ORACLE_ARM)
CONFIGS = {"tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
           "large": "configs/sam2.1/sam2.1_hiera_l.yaml"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _size(value, name):
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)):
        raise ValueError(f"Saved {name} must be two positive integer dimensions")
    return tuple(value)


def replay_row(row, record, original_size):
    """Restore the recorded frame without rerunning Qwen preprocessing."""
    from nucleus_rl.evaluate import objects_to_original, prompt_size_for_row
    from nucleus_rl.rewards import parse_completion

    if _size(record.get("original_size"), "original_size") != original_size:
        raise ValueError(f"Saved original dimensions differ for {row['id']}")
    prompt_size = _size(record.get("prompt_size"), "prompt_size")
    frame = record.get("coordinate_frame")
    if frame not in ("original", "processed"):
        raise ValueError("Saved coordinate_frame must be original or processed")
    if frame == "original" and prompt_size != original_size:
        raise ValueError("Original coordinate frame must use original dimensions")
    if not isinstance(record.get("text"), str):
        raise ValueError("Saved completion text is required verbatim")
    if record.get("patient_id") != row["patient_id"]:
        raise ValueError("Saved completion patient does not match the manifest")
    restored = {**row, "coordinate_frame": frame, "prompt_style": record["variant"],
                "original_width": original_size[0], "original_height": original_size[1],
                "prompt_width": prompt_size[0], "prompt_height": prompt_size[1],
                "processed_width": prompt_size[0], "processed_height": prompt_size[1]}
    prompt_size_for_row(restored, original_size)
    parsed = parse_completion(record["text"], width=prompt_size[0], height=prompt_size[1], max_objects=8)
    if not isinstance(record.get("format_valid"), bool) or parsed.valid != record["format_valid"]:
        raise ValueError("Saved format validity differs from the current parser")
    converted, scales = objects_to_original(parsed.objects, prompt_size, original_size)
    saved_scales = record.get("scale_to_original")
    if (not isinstance(saved_scales, list) or len(saved_scales) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not math.isclose(v, s, rel_tol=1e-12, abs_tol=1e-12)
                   for v, s in zip(saved_scales, scales))):
        raise ValueError("Saved coordinate scaling differs from the recorded dimensions")
    for key, objects in (("raw_objects", parsed.objects), ("converted_objects", converted)):
        expected = json.loads(json.dumps([asdict(obj) for obj in objects]))
        if record.get(key) != expected:
            raise ValueError(f"Saved {key} differ from verbatim completion/frame replay")
    return restored


def load_replay_inputs(source_run, manifest):
    """Verify A's fixed eight crops and build all prompts before loading SAM."""
    import numpy as np
    from PIL import Image
    from nucleus_rl.cohort import validate_training_rows
    from nucleus_rl.evaluate import read_manifest
    from scripts.prompt_diagnostic import oracle_prompt_objects

    source = Path(source_run).resolve()
    paths = {name: source / name for name in ("plan.json", "result.json", "paired_records.jsonl")}
    plan = json.loads(paths["plan.json"].read_text())
    result = json.loads(paths["result.json"].read_text())
    if (result.get("status") != "completed_paired_inference" or result.get("optimizer_steps") != 0
            or result.get("all_paired_pixels_identical") is not True):
        raise ValueError("Source must be a completed paired inference-only A run")
    source_model = result.get("model", {})
    if (type(result.get("policy_trainable_parameters")) is not int
            or result["policy_trainable_parameters"] != 0 or not isinstance(source_model, dict)
            or "adapter" not in source_model or source_model["adapter"] is not None):
        raise ValueError("Source A must use a frozen base policy with no adapter")
    source_sam = result.get("sam", {})
    if (not isinstance(source_sam, dict) or source_sam.get("config") != CONFIGS["tiny"]
            or source_sam.get("unchanged") is not True or source_sam.get("trainable_parameters") != 0
            or not source_sam.get("checkpoint_sha256")):
        raise ValueError("Source A must record the unchanged frozen SAM2.1 tiny baseline")
    selection = result.get("selected_crops")
    if not isinstance(selection, list) or len(selection) != 8 or selection != plan.get("selected_crops"):
        raise ValueError("A plan/result must agree on exactly eight selected crops")
    ids = [item["id"] for item in selection]
    if len(set(ids)) != 8:
        raise ValueError("A selection must contain eight distinct crop IDs")
    rows = read_manifest(manifest)
    validate_training_rows(rows)
    by_id = {row["id"]: row for row in rows}
    if not set(ids) <= set(by_id):
        raise ValueError("Selected A crops are missing from the supplied training manifest")
    selected = [by_id[i] for i in ids]
    cohort_audit = validate_training_rows(selected)
    saved = [json.loads(line) for line in paths["paired_records.jsonl"].read_text().splitlines() if line.strip()]
    records = {}
    for line_index, record in enumerate(saved, 1):
        key = record.get("id"), record.get("variant")
        if key[0] not in ids or key[1] not in POLICY_ARMS or key in records:
            raise ValueError("A records must have exactly one completion per selected crop and policy arm")
        records[key] = (line_index, record)
    if len(records) != 16:
        raise ValueError("All sixteen A completions are required, including invalid outputs")
    cases = []
    for row, provenance in zip(selected, selection):
        if any(row.get(key) != provenance.get(key) for key in ("patient_id", "source_image_id")):
            raise ValueError("A selection identity differs from the supplied manifest")
        for key, path_key in (("image_sha256", "image_path"), ("mask_sha256", "mask_path")):
            if file_sha256(row[path_key]) != provenance.get(key):
                raise ValueError(f"Source crop {key} mismatch for {row['id']}")
        with Image.open(row["image_path"]) as image:
            original_size = image.size
        with Image.open(row["mask_path"]) as mask:
            labels = np.asarray(mask)
        if labels.shape != (original_size[1], original_size[0]):
            raise ValueError("Reference mask dimensions do not match the source image")
        oracle_objects = oracle_prompt_objects(labels, padding=0, max_objects=8)
        if len(oracle_objects) != provenance.get("gt_count"):
            raise ValueError("Source crop instance count differs from A's saved selection")
        frames = []
        for arm in POLICY_ARMS:
            line_index, record = records[(row["id"], arm)]
            restored = replay_row(row, record, original_size)
            frames.append((record["coordinate_frame"], record["prompt_size"]))
            cases.append({"row": restored, "text": record["text"], "prompt_arm": arm,
                          "ground_truth_used_for_prompting": False, "source_record_line": line_index})
        if frames[0] != frames[1]:
            raise ValueError("A paired arms disagree on the source prompt coordinate frame")
        oracle_row = {**row, "coordinate_frame": "original", "prompt_style": ORACLE_ARM,
                      "original_width": original_size[0], "original_height": original_size[1],
                      "prompt_width": original_size[0], "prompt_height": original_size[1]}
        cases.append({"row": oracle_row,
                      "text": "<answer>" + json.dumps({"objects": oracle_objects}, separators=(",", ":")) + "</answer>",
                      "prompt_arm": ORACLE_ARM, "ground_truth_used_for_prompting": True,
                      "source_record_line": None})
    for case in cases:
        case["fixed_prompt_sha256"] = _digest({
            "text": case["text"], "coordinate_frame": case["row"]["coordinate_frame"],
            "prompt_size": [case["row"]["prompt_width"], case["row"]["prompt_height"]],
            "original_size": [case["row"]["original_width"], case["row"]["original_height"]],
        })
    return {"cases": cases, "selected_crops": selection, "cohort_audit": cohort_audit,
            "source_run": str(source), "source_artifact_sha256": {k: file_sha256(p) for k, p in paths.items()},
            "source_manifest_sha256": plan.get("manifest_sha256"),
            "supplied_manifest_sha256": file_sha256(manifest), "source_model": result.get("model"),
            "source_sam": result.get("sam"), "source_crop_hashes_verified": True}


def run_one_segmenter(size, segmenter, cases, output):
    """Score all fixed inputs with one model; reusable with a CPU fake in tests."""
    import numpy as np
    from PIL import Image
    from nucleus_rl.ablation_metrics import duplicate_metrics, summarize_records
    from nucleus_rl.evaluate import ArtifactScorer
    from scripts.prompt_diagnostic import RawRecordingSegmenter

    output = Path(output)
    recorder = RawRecordingSegmenter(segmenter)
    scorers = {arm: ArtifactScorer(recorder, output / f"{size}_{arm}", 8) for arm in ARMS}
    records = []
    for case in cases:
        arm, row, text = case["prompt_arm"], case["row"], case["text"]
        scorer = scorers[arm]
        with Image.open(row["mask_path"]) as image:
            reference = np.asarray(image)
        recorder.last_raw = None
        started = time.monotonic()
        record = scorer.score(text, row)
        elapsed = time.monotonic() - started
        raw = recorder.last_raw
        if raw is None:
            raw = {"masks": np.zeros((0, *reference.shape), dtype=bool),
                   "scores": np.zeros((0,), dtype=float),
                   "low_resolution_logits": np.zeros((0, 0, 0), dtype=float)}
        raw_path = scorer.output / f"{scorer.counter:06d}_raw.npz"
        np.savez_compressed(raw_path, **raw)
        record.update(duplicate_metrics(raw["masks"], reference))
        record.update({"variant": f"{size}_{arm}", "segmenter_size": size, "prompt_arm": arm,
                       "ground_truth_used_for_prompting": case["ground_truth_used_for_prompting"],
                       "oracle_prompts": case["ground_truth_used_for_prompting"],
                       "scope": "ORACLE GT-derived training prompts; not learned performance" if case["ground_truth_used_for_prompting"]
                                else "Saved A policy completion replay; training-only diagnostic",
                       "source_record_line": case["source_record_line"],
                       "fixed_prompt_sha256": case["fixed_prompt_sha256"],
                       "raw_masks_path": str(raw_path.resolve()), "raw_sam_outputs_path": str(raw_path.resolve()),
                       "image_path": row["image_path"], "mask_path": row["mask_path"],
                       "scoring_seconds": elapsed, "optimizer_steps": 0, "vlm_loaded": False})
        records.append(record)
        with (output / "records.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        # ArtifactScorer writes before adding diagnostic fields. Rewrite so the
        # standalone oracle completion file also carries its provenance flags.
        (scorer.output / "completions.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in scorer.results))
        print(json.dumps({"variant": record["variant"], "id": row["id"],
                          **{k: record[k] for k in ("format_valid", "seg_reward", "pq", "recall", "pred_count", "gt_count")}}), flush=True)
    variants = {}
    for arm, scorer in scorers.items():
        values = [r for r in records if r["prompt_arm"] == arm]
        if len(values) != 8:
            raise ValueError("Every D arm must retain all eight crops, including invalid completions")
        variants[f"{size}_{arm}"] = {**summarize_records(values),
            **{f"mean_{key}": statistics.mean(float(r[key]) for r in values)
               for key in ("format_reward", "seg_reward", "total_reward")},
            "oracle_prompts": arm == ORACLE_ARM, "ground_truth_used_for_prompting": arm == ORACLE_ARM,
            "optimizer_steps": 0, "vlm_loaded": False}
        write_json(scorer.output / "summary.json", variants[f"{size}_{arm}"])
    freeze = segmenter.verify_unchanged()
    if freeze.get("unchanged") is not True or freeze.get("trainable_parameters") != 0:
        raise RuntimeError("Segmenter freeze verification failed")
    return {"variants": variants, "records": records, "sam": freeze}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Completed A run containing plan/result and paired_records.jsonl")
    parser.add_argument("--manifest", required=True, help="Official30 training manifest with accessible crop files")
    parser.add_argument("--tiny-checkpoint", required=True)
    parser.add_argument("--large-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New empty output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def run(args):
    import torch
    from nucleus_rl.segmenter import FrozenSAM2

    started = time.monotonic()
    if not 1 <= args.threads <= 8:
        raise ValueError("threads must be between one and eight")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    replay = load_replay_inputs(args.source_run, args.manifest)
    checkpoints = {"tiny": args.tiny_checkpoint, "large": args.large_checkpoint}
    checkpoint_hashes = {size: file_sha256(path) for size, path in checkpoints.items()}
    if checkpoint_hashes["tiny"] != replay["source_sam"]["checkpoint_sha256"]:
        raise ValueError("Tiny checkpoint SHA256 differs from the saved A baseline")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no inference performed")
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = list(Path(__file__).parent.glob("*.py"))
    source_paths.append(Path(__file__).resolve().parents[1] / "scripts" / "prompt_diagnostic.py")
    write_json(output / "plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), **replay,
        "checkpoint_sha256": checkpoint_hashes, "sam_configs": CONFIGS,
        "comparison": "Identical saved text/frame per policy arm and identical GT-derived oracle prompts; only SAM2.1 model size changes",
        "scope": "TRAIN diagnostic on A's fixed eight crops; oracle arms separate, not a held-out benchmark",
        "optimizer_steps": 0, "vlm_loaded": False,
        "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p) for p in sorted(source_paths)},
    })
    variants, audits, paired = {}, {}, {}
    for size in ("tiny", "large"):
        # The helper returns only dictionaries/paths. Release every model owner
        # before constructing the next SAM to avoid simultaneous VRAM residency.
        sam = FrozenSAM2(checkpoints[size], CONFIGS[size], device=args.device)
        try:
            observed = run_one_segmenter(size, sam, replay["cases"], output)
            if observed["sam"].get("checkpoint_sha256") != checkpoint_hashes[size]:
                raise RuntimeError("Loaded checkpoint hash differs from the planned checkpoint")
        finally:
            del sam
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        variants.update(observed["variants"])
        audits[size] = observed["sam"]
        paired[size] = {(r["id"], r["prompt_arm"]): r["fixed_prompt_sha256"] for r in observed["records"]}
        write_json(output / f"{size}_result.json", {"variants": observed["variants"], "sam": audits[size]})
        del observed
    if paired["tiny"] != paired["large"] or len(paired["tiny"]) != 24:
        raise RuntimeError("Tiny/large did not replay identical inputs for all 24 crop/arm cases")
    result = {"status": "completed_fixed_prompt_segmenter_ablation", "variants": variants,
              "policy_variants": [f"{size}_{arm}" for size in CONFIGS for arm in POLICY_ARMS],
              "oracle_variants": [f"{size}_{ORACLE_ARM}" for size in CONFIGS],
              "selected_crops": replay["selected_crops"], "cohort_audit": replay["cohort_audit"],
              "models": audits, "fixed_prompts_identical": True, "source_crop_hashes_verified": True,
              "optimizer_steps": 0, "vlm_loaded": False, "qwen_generation_calls": 0,
              "records_path": str(output / "records.jsonl"), "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3 if args.device == "cuda" else None,
              "scope": "Eight training crops; oracle scores are privileged-input diagnostics and are not pooled with saved-policy scores"}
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
