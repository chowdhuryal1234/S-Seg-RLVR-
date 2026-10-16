"""One predeclared synthetic-shape inference with the frozen C32 checkpoint.

Prepare the image before inference, run exactly one greedy completion, and keep
the raw answer even when invalid. No SAM, adapter, optimizer, or answer repair.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
REVISION = "7cfb30d71a1f4f49a57592323337a4a4727301da"
SIZE = (128, 128)
BOX = (72, 20, 112, 52)  # Half-open XYXY, intentionally off-center/asymmetric.
BLUE = (30, 100, 220)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def calibration_prompt(width, height):
    from nucleus_rl.evaluate import prompt_for_image
    prompt = prompt_for_image(width, height, 8, "processed", "example")
    prompt[0]["content"] = "You locate individual blue rectangles in images. Return only the requested answer."
    prompt[-1]["content"] = (prompt[-1]["content"]
                             .replace("nuclei", "blue rectangles")
                             .replace("nucleus", "blue rectangle"))
    return prompt


def prepare(output):
    import numpy as np
    from PIL import Image
    if output.exists():
        raise FileExistsError("Choose a new run directory; existing runs are preserved")
    output.mkdir(parents=True)
    pixels = np.full((SIZE[1], SIZE[0], 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = BOX
    pixels[y0:y1, x0:x1] = BLUE
    Image.fromarray(pixels).save(output / "input.png")
    save(output / "plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One synthetic-shape coordinate diagnostic, not a segmentation benchmark",
        "image_size": SIZE, "known_box_original": BOX, "blue_rgb": BLUE,
        "expected_processed_size": [224, 224],
        "expected_box_processed": [126, 35, 196, 91],
        "image_sha256": sha256(output / "input.png"),
        "model": MODEL, "revision": REVISION,
        "generation_controls": {"seed": 42, "do_sample": False, "use_cache": True,
                                "max_new_tokens": 512, "max_objects": 8,
                                "min_pixels": 224**2, "max_pixels": 448**2},
        "prompt": calibration_prompt(224, 224),
        "changes_from_nuclei_run": "Replace target/task nouns with blue rectangles; keep schema, numeric example and decoding controls",
        "model_sees": "Unannotated input.png and prompt only; no true location, axes, labels or masks",
        "scoring": "Report all returned objects: continuous box IoU, signed edge errors, point containment; no answer selection or retries",
        "point_criterion": "Any point inside the rectangle is correct; it need not equal the center",
        "sam_used": False, "optimizer_steps": 0,
        "runner_sha256": sha256(__file__),
    })
    print(f"Prepared {output / 'input.png'}", flush=True)


def object_checks(objects, target):
    import math
    x0, y0, x1, y1 = target
    target_area = (x1 - x0) * (y1 - y0)
    checks = []
    for index, obj in enumerate(objects, 1):
        a, b, c, d = obj.box
        intersection = max(0., min(c, x1) - max(a, x0)) * max(0., min(d, y1) - max(b, y0))
        union = (c - a) * (d - b) + target_area - intersection
        edge_errors = [v - t for v, t in zip(obj.box, target)]
        px, py = obj.point
        checks.append({"index": index, "box_iou": intersection / union,
                       "edge_errors_processed_px": edge_errors,
                       "max_edge_error_processed_px": max(abs(e) for e in edge_errors),
                       "box_center_error_processed_px": math.hypot((a+c-x0-x1)/2, (b+d-y0-y1)/2),
                       "point_inside_target": x0 <= px < x1 and y0 <= py < y1})
    return checks


def run(output, reference_run):
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
        raise FileExistsError("This attempt already started; preserve it without selective retries")
    plan = json.loads((output / "plan.json").read_text())
    reference = json.loads((reference_run / "result.json").read_text())
    ref_plan = json.loads((reference_run / "plan.json").read_text())
    ref_model = json.loads((reference_run / "model.json").read_text())
    if (reference["status"] != "completed_paired_inference"
            or reference["model"]["model"] != MODEL
            or reference["model"]["resolved_revision"] != REVISION
            or reference["model"]["adapter"] is not None):
        raise ValueError("Reference must be the completed, unadapted C32 run")
    if sha256(output / "input.png") != plan["image_sha256"]:
        raise ValueError("Prepared image changed")
    image = Image.open(output / "input.png").convert("RGB")
    expected = np.full((128, 128, 3), 255, dtype=np.uint8)
    expected[20:52, 72:112] = BLUE
    if not np.array_equal(np.asarray(image), expected):
        raise ValueError("Image differs from the predeclared single rectangle")
    controls = plan["generation_controls"]
    for key in ("seed", "min_pixels", "max_pixels"):
        if controls[key] != ref_plan["args"][key]:
            raise ValueError(f"Reference inference control differs: {key}")
    if controls["max_new_tokens"] != ref_plan["args"]["max_completion_length"]:
        raise ValueError("Reference token cap differs")
    save(output / "started.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                  "reference_run": str(reference_run), "runner_sha256": sha256(__file__)})
    args = argparse.Namespace(model=MODEL, revision=REVISION, adapter=None, seed=42,
                              min_pixels=224**2, max_pixels=448**2,
                              coordinate_frame="processed", prompt_style="example", direct_gpu_load=True)
    torch.set_num_threads(8)
    model, processor, metadata = load_policy(args)
    model.eval()
    previous = ref_model["generation_config"]
    current = model.generation_config.to_dict()
    differences = [key for key in set(previous) | set(current)
                   if key not in ("transformers_version", "_from_model_config")
                   and previous.get(key) != current.get(key)]
    if differences:
        raise ValueError(f"Generation defaults changed: {differences}")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("All parameters must be frozen")
    prompt = calibration_prompt(224, 224)
    if prompt != plan["prompt"]:
        raise ValueError("Prompt changed after planning")
    multimodal = [dict(message) for message in prompt]
    multimodal[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
    chat = processor.apply_chat_template(multimodal, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
    measured = processed_image_size(processor, inputs)
    if measured != (224, 224):
        raise RuntimeError(f"Unexpected measured processed frame: {measured}")
    tensor_hash = tensor_digest(inputs["pixel_values"])
    token_hash = tensor_digest(inputs["input_ids"])
    np.savez_compressed(output / "processor_input.npz",
                        pixel_values=inputs["pixel_values"].numpy(),
                        image_grid_thw=inputs["image_grid_thw"].numpy(),
                        input_ids=inputs["input_ids"].numpy())
    save(output / "processor_config.json", processor.image_processor.to_dict())
    save(output / "model.json", {**metadata, "generation_config": current})
    inputs = inputs.to(model.device)
    set_seed(42)
    generation_start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    (output / "completion.txt").write_text(text + "\n")
    parsed = parse_completion(text, width=224, height=224, max_objects=8)
    converted, scales = objects_to_original(parsed.objects, measured, image.size)
    target = [126, 35, 196, 91]
    checks = object_checks(parsed.objects, target)
    for entry, obj in zip(checks, converted):
        x, y = np.floor(obj.point).astype(int)
        entry["point_gt_id_floor_xy"] = int(72 <= x < 112 and 20 <= y < 52)
        if bool(entry["point_gt_id_floor_xy"]) != entry["point_inside_target"]:
            raise RuntimeError("Point scoring differs between declared and original coordinate frames")
    result = {
        "status": "completed_coordinate_calibration", "id": "blue_rectangle_upper_right",
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": metadata,
        "source": {"size": list(image.size), "image_sha256": plan["image_sha256"], "blue_rgb": BLUE},
        "prompt_size": measured, "original_size": image.size, "scale_to_original": scales,
        "known_geometry": {"original_box": BOX, "processed_box": target},
        "generation_controls": controls, "reference_generation_defaults_identical": True,
        "prompt": prompt, "text": text, "format_valid": parsed.valid, "parse_error": parsed.error,
        "raw_objects": [asdict(obj) for obj in parsed.objects],
        "converted_objects": [asdict(obj) for obj in converted],
        "object_count": len(parsed.objects), "object_checks": checks,
        "input_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(answer_ids.shape[1]),
        "hit_token_cap": int(answer_ids.shape[1]) == 512,
        "pixel_values_sha256": tensor_hash, "input_ids_sha256": token_hash,
        "generation_seconds": time.monotonic() - generation_start,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "optimizer_steps": 0, "policy_trainable_parameters": 0, "sam_used": False,
        "scope": "One simple synthetic image; no pathology or general coordinate competence claim",
    }
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradients in inference-only test")
    save(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    else:
        if not args.reference_run:
            parser.error("--reference-run is required for inference")
        run(args.output, args.reference_run)


if __name__ == "__main__":
    main()
