"""Two fixed, cue-assisted probes on the same annotated multi-nucleus crop.

Fresh contexts test letter selection and drawn-square coordinate readback.
This is an engineering diagnostic, not causal identification or a benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from nucleus_rl.coordinate_calibration import MODEL, REVISION, save, sha256


def prompts():
    system = "You inspect images and answer the requested visual task. Return only the requested answer."
    common = ("The image contains six equally styled square markers labeled A, B, C, D, E, F. "
              "The small cross in each square marks its center. Letters and lines are artificial annotations, not tissue. ")
    return {
        "recognition": [
            {"role": "system", "content": system},
            {"role": "user", "content": common +
             "Select every labeled square whose CENTER lies inside a visible cell nucleus in the tissue. "
             "A square is not positive merely because some nucleus lies near it or overlaps its edge. "
             "There may be any number of positive squares; decide from the image. "
             "Return only the selected letter IDs, in alphabetical order, each at most once. "
             'Use exactly <answer>{"nucleus_regions":["REGION_ID",...]}</answer>; REGION_ID means a selected letter from A through F. '
             'If none qualify, use <answer>{"nucleus_regions":[]}</answer>. No coordinates, explanation or markdown.'}],
        "geometry": [
            {"role": "system", "content": system},
            {"role": "user", "content": common +
             "For EVERY drawn square A through F, read back the square outline's bounding box and the center cross position. "
             "Measure the artificial square markers, NOT nuclei and NOT the letter labels. Do not decide which regions contain nuclei. "
             "Use pixel coordinates of the PROCESSED image shown to you, which is 224 pixels wide and 224 pixels high. "
             "The origin [0,0] is at the upper left; x increases rightward and y increases downward. "
             "Boxes are [x0,y0,x1,y1] with 0<=x0<x1<=224 and 0<=y0<y1<=224; centers are [x,y] inside the square. "
             "Do not use normalized coordinates. Return all six IDs exactly once in alphabetical order. "
             'Use exactly <answer>{"regions":[{"id":"REGION_ID","box":[x0,y0,x1,y1],"point":[x,y]},...]}</answer>, '
             "replacing placeholders with actual IDs and numeric coordinates. No explanation or markdown."}],
    }


def prepare(output, selection, reference_run):
    import numpy as np
    from PIL import Image
    if output.exists():
        raise FileExistsError("Choose a new output directory; preserve every prior attempt")
    reference = json.loads((reference_run / "result.json").read_text())
    if (reference["status"] != "completed_multi_nucleus_probe" or reference["model"]["model"] != MODEL
            or reference["model"]["resolved_revision"] != REVISION or reference["model"]["adapter"] is not None):
        raise ValueError("Require the completed original 32B multi-nucleus probe")
    manifest = json.loads((selection / "candidate_manifest.json").read_text())
    marked = Image.open(selection / "candidate_input.png").convert("RGB")
    original = Image.open(reference_run / "input.png").convert("RGB")
    if marked.size != (128, 128) or original.size != (128, 128):
        raise ValueError("Original and marked images must both be native 128 square")
    mask = np.asarray(Image.open(reference_run / "gt_mask.png"))
    if len(np.unique(mask[mask > 0])) != 3:
        raise ValueError("Expected the fixed three-nucleus diagnostic")
    if (manifest["status"] != "locked_pre_inference"
            or manifest["candidate_image_sha256"] != sha256(selection / "candidate_input.png")
            or manifest["source_image_sha256"] != sha256(reference_run / "input.png")
            or manifest["source_mask_sha256"] != sha256(reference_run / "gt_mask.png")):
        raise ValueError("Candidate image must be visually locked against the exact reference")
    if [c["id"] for c in manifest["candidates"]] != list("ABCDEF"):
        raise ValueError("Require six predeclared candidate IDs")
    for candidate in manifest["candidates"]:
        x, y = candidate["point_original"]
        box = candidate["box_original"]
        if box != [x-10, y-10, x+10, y+10] or not (10 <= x <= 118 and 10 <= y <= 118):
            raise ValueError("All candidate squares must be identical in size and wholly within the image")
        if candidate["box_processed"] != [v * 224 / 128 for v in box] or candidate["point_processed"] != [x*224/128,y*224/128]:
            raise ValueError("Candidate coordinate conversion changed")
        if (int(mask[y,x]) or None) != candidate["gt_instance_id"] or bool(mask[y,x]) != candidate["is_nucleus"]:
            raise ValueError("Candidate class key disagrees with the actual mask")
    output.mkdir(parents=True)
    for source, target in ((selection / "candidate_input.png", "input.png"),
                           (selection / "candidate_manifest.json", "candidate_manifest.json"),
                           (reference_run / "input.png", "original.png"),
                           (reference_run / "gt_mask.png", "gt_mask.png")):
        (output / target).write_bytes(source.read_bytes())
    save(output / "plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Two cue-assisted diagnostics on one selected training-source crop; no benchmark or causal identification",
        "model": MODEL, "revision": REVISION,
        "reference_result_sha256": sha256(reference_run / "result.json"),
        "source": reference["source"], "manifest": manifest,
        "input_sha256": sha256(output / "input.png"),
        "original_sha256": sha256(output / "original.png"),
        "mask_sha256": sha256(output / "gt_mask.png"),
        "manifest_sha256": sha256(output / "candidate_manifest.json"),
        "generation_controls": reference["generation_controls"],
        "prompts": prompts(), "task_order": ["recognition", "geometry"],
        "attempts_per_task": 1, "fresh_context_per_task": True,
        "model_sees": "The identical marked image and that task's prompt only; no answer key, GT mask, true class count or coordinates; no response from the other task",
        "intentional_changes_from_reference": "Added six fixed candidate markers and changed task prompts/output schemas. This is assisted identification and artificial-square readback, not an identical free-localization task.",
        "geometry_engineering_criterion": "All six unique IDs, each drawn-square box IoU > 0.5 and center error <= 7 processed pixels (4 native pixels); report continuous errors regardless. This is a diagnostic tolerance, not a clinical standard.",
        "interpretation": {
            "recognition_correct_geometry_wrong": "Supports investigating coordinate readback, not proof of correct unaided nucleus recognition",
            "recognition_wrong_geometry_correct": "Supports investigating nucleus judgment/attention given these cues; not proof the model has no nucleus representation",
            "both_correct": "Cue-assisted component success; original free search/tight nucleus boundary generation may still fail",
            "both_wrong": "Includes marker readability, task understanding, formatting and both visual tasks; cannot isolate cause"},
        "runner_sha256": sha256(__file__),
        "metrics_source_sha256": sha256(Path(__file__).with_name("recognition_geometry.py")),
        "sam_used": False, "optimizer_steps": 0,
    })
    print(f"Prepared two fixed tasks in {output}", flush=True)


def run(output, reference_run):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed
    from nucleus_rl.evaluate import processed_image_size
    from nucleus_rl.prompt_ablation import tensor_digest
    from nucleus_rl.train import load_policy
    from nucleus_rl.recognition_geometry import analyze_recognition, analyze_geometry

    started = time.monotonic()
    if (output / "started.json").exists():
        raise FileExistsError("Attempt already started; no selective retries")
    plan = json.loads((output / "plan.json").read_text())
    if sha256(Path(__file__).with_name("recognition_geometry.py")) != plan["metrics_source_sha256"]:
        raise ValueError("Scoring changed after protocol lock")
    if sha256(__file__) != plan["runner_sha256"]:
        raise ValueError("Runner changed after protocol lock")
    reference = json.loads((reference_run / "result.json").read_text())
    ref_model = json.loads((reference_run / "model.json").read_text())
    if sha256(reference_run / "result.json") != plan["reference_result_sha256"]:
        raise ValueError("Reference result changed")
    for name, key in (("input.png", "input_sha256"), ("original.png", "original_sha256"),
                      ("gt_mask.png", "mask_sha256"), ("candidate_manifest.json", "manifest_sha256")):
        if sha256(output / name) != plan[key]:
            raise ValueError(f"Locked {name} changed")
    if prompts() != plan["prompts"] or plan["generation_controls"] != reference["generation_controls"]:
        raise ValueError("Prompt or generation settings changed")
    controls = plan["generation_controls"]
    save(output / "started.json", {"utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha256(__file__)})
    args = argparse.Namespace(model=MODEL, revision=REVISION, adapter=None, seed=controls["seed"],
                              min_pixels=controls["min_pixels"], max_pixels=controls["max_pixels"],
                              coordinate_frame="processed", prompt_style="example", direct_gpu_load=True)
    torch.set_num_threads(8)
    model, processor, metadata = load_policy(args)
    model.eval()
    metadata = {**metadata, "loader_prompt_style": metadata["prompt_style"],
                "prompt_style": "cue_assisted_task_specific_no_numeric_example"}
    current = model.generation_config.to_dict()
    defaults = ref_model["generation_config"]
    if any(defaults.get(k) != current.get(k) for k in set(defaults) | set(current)
           if k not in ("transformers_version", "_from_model_config")):
        raise ValueError("Generation defaults changed")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("All model parameters must remain frozen")
    if processor.image_processor.to_dict() != json.loads((reference_run / "processor_config.json").read_text()):
        raise ValueError("Processor configuration changed")
    save(output / "model.json", {**metadata, "generation_config": current})
    save(output / "processor_config.json", processor.image_processor.to_dict())
    image = Image.open(output / "input.png").convert("RGB")
    tasks = {}
    shared_pixel_hash = None
    for task in plan["task_order"]:
        # No conversational history from the other task enters these messages.
        prompt = plan["prompts"][task]
        messages = [dict(message) for message in prompt]
        messages[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
        measured = processed_image_size(processor, inputs)
        if measured != (224, 224):
            raise RuntimeError(f"Unexpected processed size: {measured}")
        pixel_hash = tensor_digest(inputs["pixel_values"])
        token_hash = tensor_digest(inputs["input_ids"])
        if shared_pixel_hash is not None and shared_pixel_hash != pixel_hash:
            raise RuntimeError("The two tasks must receive identical image tensors")
        shared_pixel_hash = pixel_hash
        np.savez_compressed(output / f"{task}_processor_input.npz", pixel_values=inputs["pixel_values"].numpy(),
                            image_grid_thw=inputs["image_grid_thw"].numpy(), input_ids=inputs["input_ids"].numpy())
        inputs = inputs.to(model.device)
        set_seed(controls["seed"])
        t0 = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=controls["max_new_tokens"],
                                       do_sample=controls["do_sample"], use_cache=controls["use_cache"])
        duration = time.monotonic() - t0
        answer_ids = generated[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (output / f"{task}_completion.txt").write_text(text + "\n")
        analysis = (analyze_recognition if task == "recognition" else analyze_geometry)(text, plan["manifest"])
        task_result = {"task": task, "prompt": prompt, "text": text, "analysis": analysis,
                       "pixel_values_sha256": pixel_hash, "input_ids_sha256": token_hash,
                       "input_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(answer_ids.shape[1]),
                       "hit_token_cap": int(answer_ids.shape[1]) == controls["max_new_tokens"],
                       "generation_seconds": duration, "fresh_context": True}
        save(output / f"{task}_result.json", task_result)
        tasks[task] = task_result
        print(f"{task}: {text}", flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradients in inference-only diagnostics")
    result = {"status": "completed_recognition_coordinate_probe", "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": metadata, "generation_controls": controls, "tasks": tasks,
              "same_image_tensor_across_tasks": True, "reference_generation_defaults_identical": True,
              "reference_processor_config_identical": True, "reference_result_sha256": plan["reference_result_sha256"],
              "input_sha256": plan["input_sha256"], "original_sha256": plan["original_sha256"],
              "manifest_sha256": plan["manifest_sha256"], "prompt_size": [224,224], "original_size": [128,128],
              "elapsed_seconds": time.monotonic() - started, "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "optimizer_steps": 0, "policy_trainable_parameters": 0, "sam_used": False, "scope": plan["scope"]}
    save(output / "result.json", result)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.selection:
            parser.error("prepare requires --selection")
        prepare(args.output, args.selection, args.reference_run)
    else:
        run(args.output, args.reference_run)
