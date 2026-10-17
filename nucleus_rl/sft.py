"""Single-GPU supervised warm-up of a language-LoRA nucleus-prompt policy.

Reference masks supervise box/point answer tokens on the training split only.
SAM2 is used for paired validation and has no role in the training loss.
Heavy dependencies are imported lazily, so --help and safety tests run on CPU.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cohort-protocol")
    parser.add_argument("--coordinate-frame", choices=("processed", "normalized1000"), default="processed")
    parser.add_argument("--augmentation", choices=("none", "d4"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--train-limit", type=int, help="Explicit canary-only training subset size")
    parser.add_argument("--eval-limit", type=int, help="Explicit canary-only validation subset size")
    parser.add_argument("--fixed-loss-examples", type=int, default=4)
    parser.add_argument("--wall-time-minutes", type=float, help="Training-stage soft limit; reserve time for final evaluation")
    parser.add_argument("--resume-from-checkpoint")
    return parser


def validate_args(args):
    positive = ("max_steps", "save_steps", "gradient_accumulation", "lora_rank", "lora_alpha",
                "max_sequence_length", "max_completion_length", "fixed_loss_examples", "min_pixels")
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Step, batch, rank, token, image, and fixed-loss limits must be positive")
    if args.max_steps > 8192 or args.eval_every < 0 or not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Require max-steps <=8192, eval-every >=0, and 0<=warmup-steps<max-steps")
    if args.max_pixels < args.min_pixels or not 1 <= args.max_objects <= 8:
        raise ValueError("Require max-pixels>=min-pixels and 1<=max-objects<=8")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        raise ValueError("lora-dropout must be finite in [0,1)")
    for key in ("train_limit", "eval_limit"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError("Canary subset limits must be positive")
    if args.wall_time_minutes is not None and (not math.isfinite(args.wall_time_minutes) or args.wall_time_minutes <= 0):
        raise ValueError("wall-time-minutes must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("SFT runner is single-process, single-GPU")


def exposure_for_index(index, row_count, seed, augmentation):
    """Paired base-image order, with each D4 view seen once per eight epochs."""
    if index < 0 or row_count <= 0 or augmentation not in ("none", "d4"):
        raise ValueError("Invalid exposure schedule")
    epoch, offset = divmod(index, row_count)
    order = list(range(row_count))
    random.Random(seed + epoch).shuffle(order)
    return order[offset], (epoch % 8 if augmentation == "d4" else 0)


def assistant_label_mask(prefix_ids, full_ids, max_length):
    """Prove exact prompt alignment; never silently truncate image/answer tokens."""
    if not prefix_ids or not full_ids or len(full_ids) > max_length:
        raise ValueError("Empty or overlength token sequence; truncation is forbidden")
    if len(full_ids) <= len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Teacher-forcing tokens do not preserve the exact inference prefix")
    return [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]


def inference_inputs(processor, example):
    messages = [dict(message) for message in example["messages"]]
    messages[-1] = {**messages[-1], "content": [
        {"type": "image"}, {"type": "text", "text": messages[-1]["content"]},
    ]}
    # New Qwen generations may otherwise insert a reasoning prefix. The same
    # non-thinking inference prefix is reused verbatim for teacher forcing.
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = processor(text=[prefix], images=[example["image"]], padding=False, return_tensors="pt")
    return prefix, inputs


def encode_example(processor, example, max_length):
    import torch
    prefix, prefix_inputs = inference_inputs(processor, example)
    eos = processor.tokenizer.eos_token
    if not isinstance(eos, str) or not eos or eos in example["completion"]:
        raise ValueError("Expected an unambiguous assistant end token")
    # Construct from the actual generation prefix. Rendering a separate full
    # conversation can change Qwen3.x's empty-think prefix and misalign labels.
    full = processor(text=[prefix + example["completion"] + eos], images=[example["image"]],
                     padding=False, return_tensors="pt")
    prefix_ids = prefix_inputs["input_ids"][0].tolist()
    full_ids = full["input_ids"][0].tolist()
    labels = assistant_label_mask(prefix_ids, full_ids, max_length)
    if full_ids[-1] != processor.tokenizer.eos_token_id:
        raise ValueError("Assistant training target must end with the assistant end token")
    # Image grids must be identical between prompt-only and answer-bearing input.
    for key in ("image_grid_thw", "pixel_values"):
        if key in prefix_inputs and (key not in full or not torch.equal(prefix_inputs[key], full[key])):
            raise ValueError("Image preprocessing changed between prefix and teacher-forcing input")
    full["labels"] = torch.tensor([labels], dtype=torch.long)
    return full, {"prompt_tokens": len(prefix_ids), "assistant_tokens": len(labels) - len(prefix_ids),
                  "total_tokens": len(labels)}


def language_lora_targets(model):
    """Adapt language-layer linear attention and MLP projections only."""
    import torch
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and ".layers." in name
               and not any(word in name.lower() for word in ("visual", "vision", "merger", "lm_head"))]
    if not targets:
        raise ValueError("No language-layer Linear modules found; refusing global all-linear targeting")
    return targets


def trainable_language_adapters(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any("lora_" not in name or any(word in name.lower() for word in ("vision", "visual", "merger"))
                        for name in names):
        raise RuntimeError("Only language LoRA adapter parameters may require gradients")
    return names


def frozen_versions(model):
    return {name: parameter._version for name, parameter in model.named_parameters() if not parameter.requires_grad}


def assert_frozen(model, versions):
    actual = frozen_versions(model)
    if actual != versions or any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise RuntimeError("A frozen base-model parameter changed or received a gradient")


def json_write(path, value):
    from nucleus_rl.training_budget import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def checkpoint_manifest(path, contract, step):
    from nucleus_rl.training_budget import contract_digest, file_digest
    path = Path(path)
    required = ("adapter_config.json", "adapter_model.safetensors", "safe_training_state.json",
                "safe_training_state.safetensors", "trainer_state.json", "coordinate_config.json")
    return {"schema_version": 1, "kind": "nucleus_prompt_sft", "global_step": step,
            "contract": contract, "contract_sha256": contract_digest(contract),
            "files": {name: file_digest(path / name) for name in required}}


def verify_sft_checkpoint(path, expected_contract=None):
    from nucleus_rl.training_budget import contract_digest, file_digest
    path = Path(path)
    marker = json.loads((path / "sft_checkpoint.json").read_text())
    required = {"adapter_config.json", "adapter_model.safetensors", "safe_training_state.json",
                "safe_training_state.safetensors", "trainer_state.json", "coordinate_config.json"}
    if (marker.get("schema_version") != 1 or marker.get("kind") != "nucleus_prompt_sft"
            or set(marker.get("files", {})) != required or type(marker.get("global_step")) is not int
            or marker["global_step"] < 1 or marker.get("contract_sha256") != contract_digest(marker["contract"])):
        raise ValueError("Invalid or incomplete SFT checkpoint")
    for name, digest in marker["files"].items():
        if file_digest(path / name) != digest:
            raise ValueError(f"SFT checkpoint content changed: {name}")
    if json.loads((path / "trainer_state.json").read_text()).get("global_step") != marker["global_step"]:
        raise ValueError("SFT checkpoint step mismatch")
    if expected_contract is not None and marker["contract_sha256"] != contract_digest(expected_contract):
        raise ValueError("SFT resume contract mismatch")
    return marker


def save_checkpoint(path, model, processor, optimizer, scheduler, contract, step, coordinates):
    from nucleus_rl.training_budget import save_training_state
    path = Path(path)
    temporary = path.with_name(".writing-" + path.name)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    json_write(temporary / "coordinate_config.json", coordinates)
    json_write(temporary / "trainer_state.json", {"global_step": step})
    save_training_state(temporary, optimizer, scheduler)
    json_write(temporary / "sft_checkpoint.json", checkpoint_manifest(temporary, contract, step))
    temporary.rename(path)
    verify_sft_checkpoint(path, contract)
    return str(path.resolve())


def _package_versions():
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "huggingface-hub", "SAM-2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_model(args):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required; SFT has not started")
    set_seed(args.seed)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        args.revision = model_info(args.model, revision=args.revision).sha
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                             min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                                             trust_remote_code=False)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cuda"}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.requires_grad_(False)
    model.config.use_cache = False
    return model, processor


def evaluate_policy(model, processor, rows, segmenter, directory, args):
    import numpy as np
    import torch
    from PIL import Image
    from nucleus_rl.evaluate import ArtifactScorer, assert_processed_dimensions, prepare_coordinate_rows
    from nucleus_rl.sft_data import sft_prompt_messages
    from nucleus_rl.localization_metrics import analyze
    from nucleus_rl.training_budget import preserve_evaluation_state
    scorer = ArtifactScorer(segmenter, directory, args.max_objects)
    localizations = []
    prepared_rows = prepare_coordinate_rows(rows, processor, args.coordinate_frame)
    with preserve_evaluation_state(model):
        model.eval()
        for row in prepared_rows:
            # Validation generation reads only the image and its measured frame.
            # The reference mask is opened later inside ArtifactScorer.
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            example = {"image": image, "messages": sft_prompt_messages(
                row["prompt_width"], row["prompt_height"], coordinate_mode=args.coordinate_frame,
                max_objects=args.max_objects)}
            _, inputs = inference_inputs(processor, example)
            assert_processed_dimensions(processor, inputs, row)
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_completion_length,
                                           use_cache=True)
            tokens = generated[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            record = scorer.score(raw, row)
            scorer.record_group([record])
            with Image.open(row["mask_path"]) as source:
                reference = np.asarray(source)
            localization = analyze(record["raw_objects"], record["converted_objects"], reference,
                                   processed_size=tuple(record["prompt_size"]))
            localization.update({"id": row["id"], "format_valid": record["format_valid"],
                                 "coordinate_frame": args.coordinate_frame,
                                 "geometry_frame_note": "Analyzer's processed-coordinate fields use the declared prompt frame"})
            localizations.append(localization)
            with (Path(directory) / "localizations.jsonl").open("a") as stream:
                stream.write(json.dumps(localization, allow_nan=False) + "\n")
    summary = scorer.summary()
    predictions = sum(item["prediction_count"] for item in localizations)
    foreground = sum(pred["point_status"] == "foreground" for item in localizations for pred in item["predictions"])
    summary["localization"] = {
        "macro_box_recall": sum(item["box_recall"] for item in localizations) / len(localizations),
        "micro_box_recall": sum(item["box_true_positive_count"] for item in localizations) / sum(item["gt_count"] for item in localizations),
        "predicted_points": predictions, "foreground_points": foreground,
        "point_foreground_rate": foreground / predictions if predictions else None,
        "point_duplicate_count": sum(item["point_duplicate_count"] for item in localizations),
        "box_duplicate_count": sum(item["best_box_duplicate_count"] for item in localizations),
    }
    json_write(Path(directory) / "summary.json", summary)
    return summary


def fixed_training_loss(model, processor, rows, args):
    import torch
    from nucleus_rl.sft_data import prepare_example
    from nucleus_rl.training_budget import preserve_evaluation_state
    values = []
    with preserve_evaluation_state(model):
        model.eval()
        for row in rows[:args.fixed_loss_examples]:
            example = prepare_example(row, processor, coordinate_frame=args.coordinate_frame,
                                      max_objects=args.max_objects)
            encoded, _ = encode_example(processor, example, args.max_sequence_length)
            with torch.no_grad():
                value = float(model(**encoded.to(model.device)).loss.detach().float().cpu())
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite fixed-example training loss")
            values.append(value)
    return {"ids": [row["id"] for row in rows[:args.fixed_loss_examples]], "losses": values,
            "mean_loss": sum(values) / len(values), "augmentation_id": 0,
            "scope": "Teacher-forcing loss on fixed training examples; not a validation metric"}


def run(args, rows, validation, audit, output):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from nucleus_rl.segmenter import FrozenSAM2, parameter_sha256
    from nucleus_rl.sft_data import prepare_example
    from nucleus_rl.training_budget import (contract_digest, dataset_identity, file_digest,
                                            load_training_state, restore_rng)

    model, processor = load_model(args)
    targets = language_lora_targets(model)
    contract_keys = ("model", "revision", "seed", "max_steps", "gradient_accumulation", "learning_rate",
                     "warmup_steps", "lora_rank", "lora_alpha", "lora_dropout", "coordinate_frame",
                     "augmentation", "max_sequence_length", "max_completion_length", "max_objects",
                     "min_pixels", "max_pixels", "eval_every", "save_steps", "train_limit", "eval_limit")
    contract = {"config": {key: getattr(args, key) for key in contract_keys},
                "train_data": dataset_identity(rows), "validation_data": dataset_identity(validation),
                "data_audit": audit, "sam_checkpoint_sha256": file_digest(args.sam_checkpoint),
                "sam_config": args.sam_config, "packages": _package_versions(),
                "source_sha256": {name: file_digest(Path(__file__).parent / name)
                                  for name in ("sft.py", "sft_data.py", "evaluate.py", "rewards.py")}}
    start_step = 0
    if args.resume_from_checkpoint:
        marker = verify_sft_checkpoint(args.resume_from_checkpoint, contract)
        start_step = marker["global_step"]
        if start_step >= args.max_steps:
            raise ValueError("Resume checkpoint already reached planned training length")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=targets, bias="none", revision=args.revision,
        ))
    trainable_names = trainable_language_adapters(model)
    visual_modules = [(name, module) for name, module in model.named_modules()
                      if name.endswith(".visual") or name.endswith(".vision_tower")]
    if len(visual_modules) != 1:
        raise ValueError(f"Expected one vision encoder for freeze audit, found {[name for name, _ in visual_modules]}")
    visual_name, visual = visual_modules[0]
    initial_visual_hash = parameter_sha256(visual)
    initial_adapter_hash = parameter_sha256(model, trainable_only=True)
    versions = frozen_versions(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))
    if args.resume_from_checkpoint:
        state = load_training_state(args.resume_from_checkpoint)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
    coordinates = {"coordinate_frame": args.coordinate_frame, "prompt_style": "schema",
                   "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                   "conversion": "x_original=x_prompt*original_width/prompt_width; likewise y",
                   "training_kind": "supervised_assistant_tokens"}
    metadata = {"resolved_revision": args.revision, "model": args.model, "packages": _package_versions(),
                "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "contract_sha256": contract_digest(contract), "trainable_parameters": sum(p.numel() for p in trainable_parameters),
                "lora_targets": targets, "trainable_names": trainable_names, "vision_encoder": visual_name,
                "vision_sha256_before": initial_visual_hash, "adapter_sha256_before": initial_adapter_hash,
                "microbatch": 1, "effective_batch": args.gradient_accumulation,
                "loss": "per-example mean assistant-answer-and-end-token cross entropy",
                "optimizer": "AdamW", "scheduler": "linear warmup then constant LR", "thinking": False,
                "scope": "canary" if args.train_limit or args.eval_limit else "main",
                "before_is_resume_checkpoint": bool(args.resume_from_checkpoint)}
    json_write(output / "metadata.json", metadata)
    json_write(output / "contract.json", contract)
    json_write(output / "arguments.json", vars(args))
    d4_audit = []
    for row in rows[:4]:
        for transform in range(8):
            example = prepare_example(row, processor, augmentation_id=transform,
                                      coordinate_frame=args.coordinate_frame, max_objects=args.max_objects)
            _, token_counts = encode_example(processor, example, args.max_sequence_length)
            d4_audit.append({"id": row["id"], "transform": transform, **token_counts})
    d4_collator_audit = {"passed": True, "transforms": list(range(8)), "records": d4_audit,
                         "collations": len(d4_audit),
                         "max_total_tokens": max(item["total_tokens"] for item in d4_audit)}
    json_write(output / "d4_collator_audit.json", d4_collator_audit)
    # SAM stays frozen. CPU mode leaves its parameters off the training GPU.
    segmenter = FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.sam_device)
    eval_started = time.monotonic()
    before = evaluate_policy(model, processor, validation, segmenter, output / "before", args)
    before_seconds = time.monotonic() - eval_started
    fixed_before = fixed_training_loss(model, processor, rows, args)
    json_write(output / "fixed_training_loss_before.json", fixed_before)
    if not args.resume_from_checkpoint:
        from transformers import set_seed
        set_seed(args.seed)
    else:
        # Constructing frozen SAM instantiates modules and consumes CPU RNG.
        # Restore only after setup and initial evaluation to continue exactly.
        restore_rng(state["rng"])
    model.train()
    # A frozen vision encoder must also avoid training-time dropout.
    visual.eval()
    step = start_step
    checkpoint_steps = set()
    evaluations = []
    started = time.monotonic()
    stop_reason = None
    step_seconds = []
    peak_allocated = peak_reserved = 0
    json_write(output / "progress.json", {"phase": "training", "step": step, "max_steps": args.max_steps})
    for step_index in range(start_step, args.max_steps):
        step_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        losses, tokens, exposures = [], [], []
        for micro in range(args.gradient_accumulation):
            index = step_index * args.gradient_accumulation + micro
            row_index, transform = exposure_for_index(index, len(rows), args.seed, args.augmentation)
            row = rows[row_index]
            example = prepare_example(row, processor, augmentation_id=transform,
                                      coordinate_frame=args.coordinate_frame, max_objects=args.max_objects)
            encoded, token_counts = encode_example(processor, example, args.max_sequence_length)
            encoded = encoded.to(model.device)
            loss = model(**encoded).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite SFT loss at step {step_index + 1}")
            losses.append(float(loss.detach().float().cpu()))
            tokens.append(token_counts)
            exposures.append({"id": row["id"], "augmentation_id": transform, "exposure_index": index})
            (loss / args.gradient_accumulation).backward()
            del loss, encoded
        gradients = [p.grad for p in trainable_parameters if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Missing or nonfinite LoRA gradients")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError("All language-LoRA gradients are zero")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0).cpu())
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        step = step_index + 1
        assert_frozen(model, versions)
        torch.cuda.synchronize()
        duration = time.monotonic() - step_started
        step_seconds.append(duration)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        log = {"step": step, "loss": sum(losses) / len(losses), "example_losses": losses,
               "learning_rate": lr_used, "gradient_norm": gradient_norm, "duration_seconds": duration,
               "training_elapsed_seconds": time.monotonic() - started, "tokens": tokens, "exposures": exposures,
               "peak_gpu_allocated_bytes": peak_allocated, "peak_gpu_reserved_bytes": peak_reserved}
        with (output / "training_log.jsonl").open("a") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        elapsed = time.monotonic() - started
        if args.wall_time_minutes and elapsed >= args.wall_time_minutes * 60:
            stop_reason = "wall_time_budget"
        due_eval = bool(args.eval_every and step % args.eval_every == 0 and step < args.max_steps and not stop_reason)
        if step % args.save_steps == 0 or due_eval or step == args.max_steps or stop_reason:
            save_checkpoint(output / "checkpoints" / f"checkpoint-{step}", model, processor, optimizer,
                            scheduler, contract, step, coordinates)
            checkpoint_steps.add(step)
        if due_eval:
            eval_started = time.monotonic()
            summary = evaluate_policy(model, processor, validation, segmenter,
                                      output / "evaluations" / f"step-{step}", args)
            record = {"step": step, "duration_seconds": time.monotonic() - eval_started, "summary": summary}
            evaluations.append(record)
            with (output / "validation_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if args.wall_time_minutes and time.monotonic() - started >= args.wall_time_minutes * 60:
                stop_reason = "wall_time_budget"
        json_write(output / "progress.json", {"phase": "training", "step": step, "max_steps": args.max_steps,
                                               "stop_reason": stop_reason, "latest": log})
        if stop_reason:
            break
    training_seconds = time.monotonic() - started
    if step not in checkpoint_steps:
        save_checkpoint(output / "checkpoints" / f"checkpoint-{step}", model, processor, optimizer,
                        scheduler, contract, step, coordinates)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    processor.save_pretrained(output / "adapter")
    json_write(output / "adapter" / "coordinate_config.json", coordinates)
    json_write(output / "progress.json", {"phase": "final_evaluation", "step": step})
    fixed_after = fixed_training_loss(model, processor, rows, args)
    json_write(output / "fixed_training_loss_after.json", fixed_after)
    eval_started = time.monotonic()
    after = evaluate_policy(model, processor, validation, segmenter, output / "after", args)
    after_seconds = time.monotonic() - eval_started
    assert_frozen(model, versions)
    final_visual_hash = parameter_sha256(visual)
    if final_visual_hash != initial_visual_hash:
        raise RuntimeError("Qwen vision encoder hash changed")
    final_adapter_hash = parameter_sha256(model, trainable_only=True)
    if final_adapter_hash == initial_adapter_hash:
        raise RuntimeError("SFT adapter did not change")
    result = {"status": "verified_sft_run", "scope": metadata["scope"], "model": args.model,
              "resolved_revision": args.revision, "optimizer_steps": step,
              "optimizer_steps_this_run": step - start_step, "requested_optimizer_steps": args.max_steps,
              "nonzero_gradient_steps": step - start_step, "d4_collator_audit": d4_collator_audit,
              "completed_requested_steps": step == args.max_steps, "stop_reason": stop_reason,
              "train_examples": len(rows), "validation_examples": len(validation),
              "training_exposures_this_run": (step - start_step) * args.gradient_accumulation,
              "training_elapsed_seconds": training_seconds, "optimizer_step_seconds": step_seconds,
              "mean_optimizer_step_seconds": sum(step_seconds) / len(step_seconds),
              "peak_gpu_allocated_bytes": peak_allocated, "peak_gpu_reserved_bytes": peak_reserved,
              "before_evaluation_seconds": before_seconds, "after_evaluation_seconds": after_seconds,
              "adapter_changed": True, "adapter_sha256_before": initial_adapter_hash,
              "adapter_sha256_after": final_adapter_hash, "vision_sha256_before": initial_visual_hash,
              "vision_sha256_after": final_visual_hash, "frozen_base_versions_unchanged": True,
              "sam_freeze_verification": segmenter.verify_unchanged(),
              "fixed_training_loss_before": fixed_before, "fixed_training_loss_after": fixed_after,
              "periodic_evaluations": evaluations, "before": before, "after": after}
    json_write(output / "summary.json", result)
    json_write(output / "progress.json", {"phase": "complete", "step": step,
                                          "status": result["status"], "stop_reason": stop_reason})
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    owns_output = False
    try:
        validate_args(args)
        from nucleus_rl.evaluate import read_manifest
        from nucleus_rl.sft_data import validate_sft_splits
        rows, validation = read_manifest(args.manifest), read_manifest(args.eval_manifest)
        # Validate the entire source manifests before applying canary limits.
        audit = validate_sft_splits(rows, validation, verify_files=True)
        if args.cohort_protocol:
            from nucleus_rl.cohort import load_official30_cohort
            from nucleus_rl.training_budget import check_cohort
            check_cohort(rows, validation, load_official30_cohort(args.cohort_protocol))
        if args.train_limit:
            rows = rows[:args.train_limit]
        if args.eval_limit:
            validation = validation[:args.eval_limit]
        if args.resume_from_checkpoint:
            marker = verify_sft_checkpoint(args.resume_from_checkpoint)
            config = marker["contract"]["config"]
            if args.model != config["model"] or args.revision not in ("main", config["revision"]):
                raise ValueError("Resume requires identical base model and resolved revision")
            args.revision = config["revision"]
        if not Path(args.sam_checkpoint).is_file():
            raise FileNotFoundError(args.sam_checkpoint)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty output directory; existing experiment artifacts are preserved")
        output.mkdir(parents=True, exist_ok=True)
        owns_output = True
        json_write(output / "preflight.json", {"status": "passed_before_gpu_load", "audit": audit,
                                               "selected_train": len(rows), "selected_validation": len(validation)})
        return run(args, rows, validation, audit, output)
    except Exception as error:
        if owns_output:
            json_write(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
