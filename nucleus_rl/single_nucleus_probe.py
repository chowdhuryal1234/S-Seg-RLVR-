"""One preselected real-nucleus coordinate probe, with the same frozen C32 setup.

No SAM, training, prompt repair or repeated generation. Full annotation is used
to select one clear diagnostic image and score it, never as model input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from nucleus_rl.coordinate_calibration import MODEL, REVISION, object_checks, save, sha256
from nucleus_rl.evaluate import prompt_for_image


def geometry(mask):
    import numpy as np
    labels = np.unique(mask[mask > 0])
    if mask.shape != (128, 128) or len(labels) != 1:
        raise ValueError("Require a native 128x128 crop with exactly one annotated instance")
    yy, xx = np.where(mask > 0)
    box = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
    if min(box[:2]) <= 0 or max(box[2:]) >= 128:
        raise ValueError("The annotated nucleus must not touch the crop border")
    return {"original_box": box, "processed_box": [value * 224 / 128 for value in box],
            "instance_id": int(labels[0]), "gt_count": 1, "gt_area": int((mask > 0).sum())}


def score(objects, converted, mask, target):
    import numpy as np
    checks = object_checks(objects, target)
    for item, obj in zip(checks, converted):
        item["point_inside_target_box"] = item.pop("point_inside_target")
        x, y = np.floor(obj.point).astype(int)
        item["point_gt_id_floor_xy"] = int(mask[y, x])
        item["point_inside_target"] = bool(mask[y, x] > 0)
    return checks


def prepare(output, image_path, mask_path, selection_path, rectangle_run):
    import numpy as np
    from PIL import Image
    from nucleus_rl.cohort import validate_training_rows
    if output.exists():
        raise FileExistsError("Preserve existing runs; choose a new output directory")
    selection = json.loads(selection_path.read_text())
    # Selection files expose the source row at top level for independent audit.
    cohort = validate_training_rows([selection])
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path))
    if image.size != (128, 128):
        raise ValueError("Use native128 input, matching the rectangle calibration dimensions")
    known = geometry(mask)
    previous = json.loads((rectangle_run / "result.json").read_text())
    if previous["status"] != "completed_coordinate_calibration":
        raise ValueError("Expected the completed rectangle control")
    output.mkdir(parents=True)
    (output / "input.png").write_bytes(image_path.read_bytes())
    (output / "gt_mask.png").write_bytes(mask_path.read_bytes())
    save(output / "plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One selected training-source nucleus; diagnostic, not held-out performance",
        "selection": selection, "cohort_audit": cohort, "known_geometry": known,
        "source": {"size": [128, 128], "image_sha256": sha256(output / "input.png"),
                   "mask_sha256": sha256(output / "gt_mask.png"), "selection": selection},
        "model": MODEL, "revision": REVISION,
        "generation_controls": previous["generation_controls"],
        "prompt": prompt_for_image(224, 224, 8, "processed", "example"),
        "expected_processed_size": [224, 224],
        "changes_from_rectangle": "Real pathology input; restore original nucleus task nouns. Same checkpoint, image dimensions, numeric example, coordinate frame and decoding controls.",
        "model_sees": "input.png and task prompt only; no true count, GT mask, boxes, points or overlay",
        "scoring": "All returned objects; box IoU against mask-derived tight box; point membership in actual foreground at floor(original XY)",
        "rectangle_result_sha256": sha256(rectangle_run / "result.json"),
        "sam_used": False, "optimizer_steps": 0, "runner_sha256": sha256(__file__),
    })
    print(f"Prepared {output}, GT original box {known['original_box']}", flush=True)


def run(output, rectangle_run):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed
    from nucleus_rl.evaluate import processed_image_size, objects_to_original
    from nucleus_rl.prompt_ablation import tensor_digest
    from nucleus_rl.rewards import parse_completion
    from nucleus_rl.train import load_policy

    started = time.monotonic()
    if (output / "started.json").exists():
        raise FileExistsError("This attempt already started; no selective retries")
    plan = json.loads((output / "plan.json").read_text())
    previous = json.loads((rectangle_run / "result.json").read_text())
    ref_model = json.loads((rectangle_run / "model.json").read_text())
    if (previous["status"] != "completed_coordinate_calibration"
            or previous["model"]["model"] != MODEL
            or previous["model"]["resolved_revision"] != REVISION
            or previous["model"]["adapter"] is not None
            or sha256(rectangle_run / "result.json") != plan["rectangle_result_sha256"]):
        raise ValueError("Rectangle control differs from the predeclared reference")
    for filename, key in (("input.png", "image_sha256"), ("gt_mask.png", "mask_sha256")):
        if sha256(output / filename) != plan["source"][key]:
            raise ValueError(f"Prepared {filename} changed")
    image = Image.open(output / "input.png").convert("RGB")
    mask = np.asarray(Image.open(output / "gt_mask.png"))
    known = geometry(mask)
    if known != plan["known_geometry"] or image.size != (128, 128):
        raise ValueError("Prepared geometry changed")
    controls = plan["generation_controls"]
    if controls != previous["generation_controls"]:
        raise ValueError("Generation controls differ from the rectangle run")
    save(output / "started.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                  "rectangle_run": str(rectangle_run), "runner_sha256": sha256(__file__)})
    args = argparse.Namespace(model=MODEL, revision=REVISION, adapter=None, seed=controls["seed"],
                              min_pixels=controls["min_pixels"], max_pixels=controls["max_pixels"],
                              coordinate_frame="processed", prompt_style="example", direct_gpu_load=True)
    torch.set_num_threads(8)
    model, processor, metadata = load_policy(args)
    model.eval()
    current = model.generation_config.to_dict()
    defaults = ref_model["generation_config"]
    differences = [key for key in set(defaults) | set(current)
                   if key not in ("transformers_version", "_from_model_config")
                   and defaults.get(key) != current.get(key)]
    if differences or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError(f"Inference control mismatch or trainable parameters: {differences}")
    prompt = prompt_for_image(224, 224, controls["max_objects"], "processed", "example")
    if prompt != plan["prompt"]:
        raise ValueError("Prompt changed after selection")
    multimodal = [dict(message) for message in prompt]
    multimodal[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
    chat = processor.apply_chat_template(multimodal, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
    measured = processed_image_size(processor, inputs)
    if measured != (224, 224):
        raise RuntimeError(f"Processed frame differs from the rectangle run: {measured}")
    tensor_hash = tensor_digest(inputs["pixel_values"])
    token_hash = tensor_digest(inputs["input_ids"])
    np.savez_compressed(output / "processor_input.npz", pixel_values=inputs["pixel_values"].numpy(),
                        image_grid_thw=inputs["image_grid_thw"].numpy(), input_ids=inputs["input_ids"].numpy())
    save(output / "processor_config.json", processor.image_processor.to_dict())
    save(output / "model.json", {**metadata, "generation_config": current})
    inputs = inputs.to(model.device)
    set_seed(controls["seed"])
    generation_start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=controls["max_new_tokens"],
                                   do_sample=controls["do_sample"], use_cache=controls["use_cache"])
    generation_seconds = time.monotonic() - generation_start
    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    (output / "completion.txt").write_text(text + "\n")
    parsed = parse_completion(text, width=224, height=224, max_objects=controls["max_objects"])
    converted, scales = objects_to_original(parsed.objects, measured, image.size)
    checks = score(parsed.objects, converted, mask, known["processed_box"])
    result = {
        "status": "completed_single_nucleus_probe", "id": plan["selection"]["id"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": metadata,
        "source": plan["source"], "known_geometry": known,
        "prompt_size": measured, "original_size": image.size, "scale_to_original": scales,
        "generation_controls": controls, "reference_generation_defaults_identical": True,
        "prompt": prompt, "text": text, "format_valid": parsed.valid, "parse_error": parsed.error,
        "raw_objects": [asdict(obj) for obj in parsed.objects],
        "converted_objects": [asdict(obj) for obj in converted],
        "object_count": len(parsed.objects), "object_checks": checks,
        "input_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(answer_ids.shape[1]),
        "hit_token_cap": int(answer_ids.shape[1]) == controls["max_new_tokens"],
        "pixel_values_sha256": tensor_hash, "input_ids_sha256": token_hash,
        "generation_seconds": generation_seconds, "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "optimizer_steps": 0, "policy_trainable_parameters": 0, "sam_used": False,
        "rectangle_result_sha256": plan["rectangle_result_sha256"],
        "scope": plan["scope"],
    }
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradients in inference-only test")
    save(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rectangle-run", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not all((args.image, args.mask, args.selection)):
            parser.error("Preparation requires --image, --mask and --selection")
        prepare(args.output, args.image, args.mask, args.selection, args.rectangle_run)
    else:
        run(args.output, args.rectangle_run)
