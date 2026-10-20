"""Single-GPU GRPO, or a continued-SFT control, starting from a finished SFT adapter.

The SFT LoRA is merged into the frozen base weights, so turning the new adapter
off gives exactly the SFT policy; that is the KL reference. A fresh language
LoRA is trained on top. GRPO rewards score frozen-SAM2 masks of sampled
box/point answers against training masks. The control objective (--objective
sft) trains the same fresh LoRA on the same images with answer-token cross
entropy. Validation reuses sft.evaluate_policy, so metrics are directly
comparable with the SFT runs. Heavy dependencies load lazily: --help and the
CPU tests need no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from types import SimpleNamespace

from nucleus_rl.grpo_rewards import (completion_lengths, group_advantages, parse_reward_spec,
                                     reward_components, total_reward)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("grpo", "sft"), default="grpo")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Pinned 40-character base commit of the SFT run")
    parser.add_argument("--init-adapter", required=True, help="Finished SFT adapter; merged into the base")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--eval-sam-device", choices=("cpu", "cuda"), default="cpu",
                        help="cpu matches the SFT validation runs")
    parser.add_argument("--reward-sam-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cohort-protocol")
    parser.add_argument("--coordinate-frame", choices=("processed", "normalized1000"), required=True)
    parser.add_argument("--min-pixels", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--augmentation", choices=("none", "d4"), default="d4")
    parser.add_argument("--reward", default="format=1,pq=1",
                        help="Weighted components: format, fg_iou, pq, count, separation")
    parser.add_argument("--group-size", type=int, default=8, help="Sampled answers per image")
    parser.add_argument("--prompts-per-step", type=int, default=2, help="Training images per optimizer step")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04, help="KL weight toward the SFT policy")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=512, help="Sampling limit during training")
    parser.add_argument("--eval-max-completion-length", type=int, default=1024,
                        help="Greedy validation limit; 1024 matches the SFT runs")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, help="Canary-only training subset size")
    parser.add_argument("--eval-limit", type=int, help="Canary-only validation subset size")
    parser.add_argument("--skip-before-eval", action="store_true", help="Canary-only: skip initial validation")
    parser.add_argument("--wall-time-minutes", type=float, help="Training-stage soft limit; final evaluation is extra")
    parser.add_argument("--resume-from-checkpoint", help="Checkpoint written by this runner; use a new --output")
    return parser


def validate_args(args):
    positive = ("group_size", "prompts_per_step", "max_steps", "save_steps", "lora_rank", "lora_alpha",
                "max_completion_length", "eval_max_completion_length", "max_sequence_length", "min_pixels")
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Group, batch, step, rank, token and pixel settings must be positive")
    if args.objective == "grpo" and args.group_size < 2:
        raise ValueError("GRPO needs at least two sampled answers per image")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Pin --revision to the 40-character base commit used for SFT")
    if args.max_pixels < args.min_pixels or not 1 <= args.max_objects <= 8:
        raise ValueError("Require max-pixels >= min-pixels and 1 <= max-objects <= 8")
    for key in ("temperature", "learning_rate"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if args.eval_every < 0 or not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Require eval-every >= 0 and 0 <= warmup-steps < max-steps")
    for key in ("train_limit", "eval_limit"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError("Canary subset limits must be positive")
    if args.wall_time_minutes is not None and (not math.isfinite(args.wall_time_minutes) or args.wall_time_minutes <= 0):
        raise ValueError("wall-time-minutes must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner is single-process, single-GPU")
    args.reward_weights = parse_reward_spec(args.reward)


def check_init_adapter(args):
    """The SFT adapter must match the requested base model, revision and input contract."""
    from nucleus_rl.training_budget import file_digest
    path = Path(args.init_adapter)
    config = json.loads((path / "adapter_config.json").read_text())
    coordinates = json.loads((path / "coordinate_config.json").read_text())
    if config.get("base_model_name_or_path") != args.model:
        raise ValueError(f"SFT adapter base is {config.get('base_model_name_or_path')!r}, not {args.model!r}")
    if config.get("revision") not in (None, args.revision):
        raise ValueError(f"SFT adapter revision {config.get('revision')!r} differs from --revision")
    for key in ("coordinate_frame", "min_pixels", "max_pixels"):
        if coordinates.get(key) != getattr(args, key):
            flag = "--" + key.replace("_", "-")
            raise ValueError(f"{flag}={getattr(args, key)!r} differs from the SFT adapter ({coordinates.get(key)!r})")
    return {"path": str(path.resolve()), "adapter_sha256": file_digest(path / "adapter_model.safetensors"),
            "coordinate_config": coordinates, "lora_rank": config.get("r")}


def exposure(index, row_count, seed, augmentation):
    """Shuffled base-image order per epoch; with d4, an independent seeded view per exposure."""
    if index < 0 or row_count <= 0 or augmentation not in ("none", "d4"):
        raise ValueError("Invalid exposure schedule")
    epoch, offset = divmod(index, row_count)
    order = list(range(row_count))
    random.Random(seed + epoch).shuffle(order)
    transform = random.Random(f"{seed}:{index}").randrange(8) if augmentation == "d4" else 0
    return order[offset], transform


def training_view(row, transform, processor, frame, max_objects):
    """Augmented image, its matching reference mask, the prompt frame and messages."""
    from nucleus_rl.evaluate import processed_image_size
    from nucleus_rl.sft_data import _load_training_pair, apply_d4, coordinate_size, sft_prompt_messages
    image, mask = apply_d4(*_load_training_pair(row), transform)
    processed = processed_image_size(processor, processor.image_processor(images=image, return_tensors="pt"))
    prompt_size = coordinate_size(image.size, coordinate_mode=frame,
                                  prompt_size=processed if frame == "processed" else None)
    messages = sft_prompt_messages(*prompt_size, coordinate_mode=frame, max_objects=max_objects)
    return image, mask, prompt_size, messages


def repeat_inputs(inputs, count):
    """One prompt stacked `count` times: token-aligned tensors repeat rows, pixels concatenate."""
    import torch
    batch = {}
    for key, value in inputs.items():
        if not hasattr(value, "shape"):
            batch[key] = value
        elif key == "pixel_values":
            batch[key] = torch.cat([value] * count, dim=0)
        elif value.shape[0] == 1:
            batch[key] = value.repeat(count, *([1] * (value.dim() - 1)))
        else:
            raise ValueError(f"Cannot replicate processor output {key} with shape {tuple(value.shape)}")
    return batch


def completion_batch(inputs, rows, pad_id):
    """Prompt plus sampled tokens, right-padded; returns the batch and a completion-token mask."""
    import torch
    prompt_length = inputs["input_ids"].shape[1]
    longest = max(len(row) for row in rows)
    batch = repeat_inputs(inputs, len(rows))
    ids = torch.full((len(rows), prompt_length + longest), pad_id, dtype=batch["input_ids"].dtype)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), longest), dtype=torch.float32)
    ids[:, :prompt_length] = batch["input_ids"]
    attention[:, :prompt_length] = batch["attention_mask"]
    for index, row in enumerate(rows):
        ids[index, prompt_length:prompt_length + len(row)] = torch.as_tensor(row, dtype=ids.dtype)
        attention[index, prompt_length:prompt_length + len(row)] = 1
        mask[index, :len(row)] = 1
    for key, value in list(batch.items()):
        # Other token-aligned processor outputs (e.g. token type IDs) are zero on answer tokens.
        if key not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") \
                and hasattr(value, "dim") and value.dim() == 2 and value.shape[1] == prompt_length:
            padded = torch.zeros((len(rows), prompt_length + longest), dtype=value.dtype)
            padded[:, :prompt_length] = value
            batch[key] = padded
    batch["input_ids"], batch["attention_mask"] = ids, attention
    return batch, mask, longest


def token_logprobs(model, batch, completion_length, keep_logits):
    """Log-probabilities of the last `completion_length` tokens, one row at a time."""
    import torch
    extra = {"logits_to_keep": completion_length + 1} if keep_logits else {}
    logits = model(**batch, use_cache=False, **extra).logits[:, -(completion_length + 1):-1, :]
    targets = batch["input_ids"][:, -completion_length:]
    values = []
    for row_logits, row_targets in zip(logits, targets):
        row_logits = row_logits.float()
        values.append(row_logits.gather(-1, row_targets[:, None]).squeeze(-1) - torch.logsumexp(row_logits, dim=-1))
    return torch.stack(values)


def end_token_ids(model, processor):
    configured = model.generation_config.eos_token_id
    ids = set(configured if isinstance(configured, (list, tuple)) else [] if configured is None else [configured])
    if processor.tokenizer.eos_token_id is not None:
        ids.add(processor.tokenizer.eos_token_id)
    if not ids:
        raise ValueError("No end-of-answer token is configured")
    return sorted(int(token) for token in ids)


def load_policy(args):
    """Base model with the SFT adapter merged in, plus a fresh (or resumed) language LoRA."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from nucleus_rl.sft import language_lora_targets, load_model
    model, processor = load_model(args)
    merged = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=False).merge_and_unload()
    merged.requires_grad_(False)
    targets = language_lora_targets(merged)
    if args.resume_from_checkpoint:
        policy = PeftModel.from_pretrained(merged, args.resume_from_checkpoint, is_trainable=True)
    else:
        policy = get_peft_model(merged, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=targets, bias="none", revision=args.revision,
        ))
    return policy, processor, targets


def evaluate(policy, processor, rows, segmenter, directory, args):
    from nucleus_rl.sft import evaluate_policy
    settings = SimpleNamespace(coordinate_frame=args.coordinate_frame, max_objects=args.max_objects,
                               max_completion_length=args.eval_max_completion_length)
    return evaluate_policy(policy, processor, rows, segmenter, directory, settings)


def grpo_micro(policy, visual, processor, sam, row, transform, args, sampling, eos_ids, pad_id, keep_logits):
    """Sample a group for one image, score it with SAM, and backpropagate the GRPO loss."""
    import numpy as np
    import torch
    from nucleus_rl.evaluate import objects_to_original
    from nucleus_rl.rewards import parse_completion
    from nucleus_rl.sft import inference_inputs
    image, mask, prompt_size, messages = training_view(row, transform, processor, args.coordinate_frame,
                                                       args.max_objects)
    _, inputs = inference_inputs(processor, {"image": image, "messages": messages})
    device = policy.device
    policy.eval()
    with torch.inference_mode():
        generated = policy.generate(**{key: value.to(device) for key, value in
                                       repeat_inputs(inputs, args.group_size).items()},
                                    generation_config=sampling)
    sampled = generated[:, inputs["input_ids"].shape[1]:].cpu().tolist()
    lengths = completion_lengths(sampled, eos_ids)
    rows = [tokens[:length] for tokens, (length, _) in zip(sampled, lengths)]
    texts = processor.batch_decode(rows, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    records = []
    for text, (length, finished) in zip(texts, lengths):
        parsed = parse_completion(text, width=prompt_size[0], height=prompt_size[1], max_objects=args.max_objects)
        objects = parsed.objects if parsed.valid else ()
        if objects:
            converted, _ = objects_to_original(objects, prompt_size, image.size)
            instances = sam.predict(image, converted, image_key=f"{row['id']}#{transform}")
            raw = sam.last_raw_masks
        else:
            instances = np.zeros(mask.shape, dtype=np.int32)
            raw = np.zeros((0, *mask.shape), dtype=bool)
        components = reward_components(parsed.valid, len(objects), instances, raw, mask)
        records.append({"text": text, "tokens": length, "finished": finished, "valid": parsed.valid,
                        "objects": len(objects), "gt": int(np.count_nonzero(np.unique(mask))),
                        "components": components, "reward": total_reward(components, args.reward_weights)})
    rewards = [record["reward"] for record in records]
    advantages = group_advantages(rewards)
    info = {"id": row["id"], "transform": transform, "rewards": rewards,
            "reward_mean": sum(rewards) / len(rewards), "reward_std": float(np.std(rewards)),
            "components": {name: sum(r["components"][name] for r in records) / len(records)
                           for name in records[0]["components"]},
            "tokens": [record["tokens"] for record in records],
            "unfinished": sum(not record["finished"] for record in records),
            "invalid": sum(not record["valid"] for record in records),
            "objects": [record["objects"] for record in records], "gt": records[0]["gt"],
            "skipped": not any(advantages), "records": records}
    if info["skipped"]:
        return info
    policy.train()
    visual.eval()
    batch, completion_mask, longest = completion_batch(inputs, rows, pad_id)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    completion_mask = completion_mask.to(device)
    logprobs = token_logprobs(policy, batch, longest, keep_logits)
    with torch.no_grad(), policy.disable_adapter():
        reference = token_logprobs(policy, batch, longest, keep_logits)
    advantage = torch.tensor(advantages, device=device, dtype=torch.float32)[:, None]
    delta = reference - logprobs
    kl = torch.exp(delta) - delta - 1
    per_token = -torch.exp(logprobs - logprobs.detach()) * advantage + args.beta * kl
    tokens = completion_mask.sum(1)
    loss = ((per_token * completion_mask).sum(1) / tokens).mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite GRPO loss")
    (loss / args.prompts_per_step).backward()
    info["loss"] = float(loss.detach().cpu())
    info["kl"] = float(((kl.detach() * completion_mask).sum(1) / tokens).mean().cpu())
    return info


def sft_micro(policy, visual, processor, row, transform, args):
    """Continued-SFT control: answer-token cross entropy on the same image view."""
    import torch
    from nucleus_rl.sft import encode_example
    from nucleus_rl.sft_data import prepare_example
    policy.train()
    visual.eval()
    example = prepare_example(row, processor, augmentation_id=transform,
                              coordinate_frame=args.coordinate_frame, max_objects=args.max_objects)
    encoded, token_counts = encode_example(processor, example, args.max_sequence_length)
    loss = policy(**encoded.to(policy.device)).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite SFT loss")
    (loss / args.prompts_per_step).backward()
    return {"id": row["id"], "transform": transform, "loss": float(loss.detach().cpu()), **token_counts}


def save_checkpoint(path, policy, optimizer, scheduler, state):
    from nucleus_rl.sft import json_write
    from nucleus_rl.training_budget import save_training_state
    path = Path(path)
    temporary = path.with_name(".writing-" + path.name)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
    temporary.mkdir(parents=True)
    policy.save_pretrained(temporary, safe_serialization=True)
    save_training_state(temporary, optimizer, scheduler)
    json_write(temporary / "trainer_state.json", state)
    temporary.rename(path)
    return str(path.resolve())


def run(args, rows, validation, audit, output):
    import inspect
    import torch
    from transformers import GenerationConfig, set_seed
    from nucleus_rl.segmenter import FrozenSAM2, parameter_sha256
    from nucleus_rl.sft import (_package_versions, assert_frozen, frozen_versions, json_write,
                                trainable_language_adapters)
    from nucleus_rl.training_budget import load_training_state, restore_rng

    init = check_init_adapter(args)
    policy, processor, targets = load_policy(args)
    trainable_names = trainable_language_adapters(policy)
    visual_modules = [(name, module) for name, module in policy.named_modules()
                      if name.endswith(".visual") or name.endswith(".vision_tower")]
    if len(visual_modules) != 1:
        raise ValueError(f"Expected one vision encoder, found {[name for name, _ in visual_modules]}")
    visual_name, visual = visual_modules[0]
    visual_hash = parameter_sha256(visual)
    adapter_hash = parameter_sha256(policy, trainable_only=True)
    versions = frozen_versions(policy)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))
    start_step = 0
    if args.resume_from_checkpoint:
        resumed = json.loads((Path(args.resume_from_checkpoint) / "trainer_state.json").read_text())
        if resumed.get("init_adapter_sha256") != init["adapter_sha256"] or resumed.get("objective") != args.objective:
            raise ValueError("Resume checkpoint was trained from a different SFT adapter or objective")
        state = load_training_state(args.resume_from_checkpoint)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = resumed["global_step"]
        if start_step >= args.max_steps:
            raise ValueError("Resume checkpoint already reached --max-steps")
    eos_ids = end_token_ids(policy, processor)
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_ids[0]
    # Sample from the policy itself: no top-k/top-p truncation or repetition penalty,
    # so the log-probabilities in the loss describe the distribution that was sampled.
    sampling = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                repetition_penalty=1.0, max_new_tokens=args.max_completion_length,
                                eos_token_id=eos_ids, pad_token_id=pad_id, use_cache=True)
    keep_logits = "logits_to_keep" in inspect.signature(policy.get_base_model().forward).parameters
    coordinates = {**init["coordinate_config"], "training_kind": f"{args.objective}_from_sft",
                   "init_adapter": init["path"], "init_adapter_sha256": init["adapter_sha256"],
                   "usage": "Merge init_adapter into the base model first, then apply this adapter"}
    metadata = {"model": args.model, "resolved_revision": args.revision, "objective": args.objective,
                "init_adapter": init, "reward_weights": args.reward_weights, "packages": _package_versions(),
                "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "lora_targets": len(targets), "trainable_names": len(trainable_names),
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "vision_encoder": visual_name, "vision_sha256_before": visual_hash,
                "adapter_sha256_before": adapter_hash, "eos_token_ids": eos_ids, "pad_token_id": pad_id,
                "sampling": sampling.to_dict(), "validation_generation_config": policy.generation_config.to_dict(),
                "logits_to_keep": keep_logits, "data_audit": audit, "reference": "adapter-disabled merged SFT policy",
                "loss": "GRPO, one update per sample, per-sequence token mean, k3 KL to reference"
                if args.objective == "grpo" else "answer-token cross entropy (continued SFT control)",
                "scope": "canary" if args.train_limit or args.eval_limit else "main",
                "resumed_from_step": start_step}
    json_write(output / "metadata.json", metadata)
    reward_sam = (FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.reward_sam_device)
                  if args.objective == "grpo" else None)
    eval_sam = FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.eval_sam_device)
    before = None
    before_seconds = 0.0
    if not args.skip_before_eval:
        started = time.monotonic()
        before = evaluate(policy, processor, validation, eval_sam, output / "before", args)
        before_seconds = time.monotonic() - started
    set_seed(args.seed + start_step)
    if args.resume_from_checkpoint:
        restore_rng(state["rng"])
    step = start_step
    evaluations, step_seconds, saved = [], [], []
    stop_reason = None
    peak_allocated = peak_reserved = 0
    groups_with_signal = groups_seen = 0
    started = time.monotonic()
    json_write(output / "progress.json", {"phase": "training", "step": step, "max_steps": args.max_steps})
    for step_index in range(start_step, args.max_steps):
        step_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        micro = []
        for offset in range(args.prompts_per_step):
            index = step_index * args.prompts_per_step + offset
            row_index, transform = exposure(index, len(rows), args.seed, args.augmentation)
            if args.objective == "grpo":
                info = grpo_micro(policy, visual, processor, reward_sam, rows[row_index], transform, args,
                                  sampling, eos_ids, pad_id, keep_logits)
                groups_seen += 1
                groups_with_signal += not info["skipped"]
                with (output / "samples.jsonl").open("a") as stream:
                    for record in info.pop("records"):
                        stream.write(json.dumps({"step": step_index + 1, "id": info["id"],
                                                 "transform": transform, **record}, allow_nan=False) + "\n")
            else:
                info = sft_micro(policy, visual, processor, rows[row_index], transform, args)
            micro.append(info)
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if gradients and any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError("Nonfinite LoRA gradients")
        updated = bool(gradients) and any(torch.count_nonzero(gradient).item() for gradient in gradients)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).cpu()) if updated else 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        if updated:
            optimizer.step()
        scheduler.step()
        step = step_index + 1
        assert_frozen(policy, versions)
        torch.cuda.synchronize()
        duration = time.monotonic() - step_started
        step_seconds.append(duration)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        log = {"step": step, "updated": updated, "learning_rate": learning_rate, "gradient_norm": gradient_norm,
               "duration_seconds": duration, "training_elapsed_seconds": time.monotonic() - started,
               "peak_gpu_allocated_bytes": peak_allocated, "peak_gpu_reserved_bytes": peak_reserved,
               "micro": micro}
        if args.objective == "grpo":
            log["reward_mean"] = sum(item["reward_mean"] for item in micro) / len(micro)
            log["groups_with_signal"] = sum(not item["skipped"] for item in micro)
        with (output / "training_log.jsonl").open("a") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        if args.objective == "grpo" and step - start_step >= 20 and groups_with_signal == 0:
            stop_reason = "no_reward_variation"
        elif args.wall_time_minutes and time.monotonic() - started >= args.wall_time_minutes * 60:
            stop_reason = "wall_time_budget"
        due_eval = bool(args.eval_every and step % args.eval_every == 0 and step < args.max_steps and not stop_reason)
        if step % args.save_steps == 0 or due_eval or step == args.max_steps or stop_reason:
            saved.append((step, due_eval, save_checkpoint(
                output / "checkpoints" / f"checkpoint-{step}", policy, optimizer, scheduler,
                {"global_step": step, "objective": args.objective, "init_adapter_sha256": init["adapter_sha256"]})))
            # Keep evaluation-step checkpoints and the newest one.
            for old_step, kept, path in saved[:-1]:
                if not kept and Path(path).exists():
                    shutil.rmtree(path)
            saved = [item for item in saved[:-1] if item[1]] + saved[-1:]
        if due_eval:
            eval_started = time.monotonic()
            summary = evaluate(policy, processor, validation, eval_sam, output / "evaluations" / f"step-{step}", args)
            record = {"step": step, "duration_seconds": time.monotonic() - eval_started, "summary": summary}
            evaluations.append(record)
            with (output / "validation_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if args.wall_time_minutes and time.monotonic() - started >= args.wall_time_minutes * 60:
                stop_reason = "wall_time_budget"
        json_write(output / "progress.json", {"phase": "training", "step": step, "max_steps": args.max_steps,
                                               "stop_reason": stop_reason, "groups_with_signal": groups_with_signal,
                                               "groups_seen": groups_seen, "last_duration_seconds": duration})
        if stop_reason:
            break
    training_seconds = time.monotonic() - started
    policy.save_pretrained(output / "adapter", safe_serialization=True)
    processor.save_pretrained(output / "adapter")
    json_write(output / "adapter" / "coordinate_config.json", coordinates)
    json_write(output / "progress.json", {"phase": "final_evaluation", "step": step})
    eval_started = time.monotonic()
    after = evaluate(policy, processor, validation, eval_sam, output / "after", args)
    after_seconds = time.monotonic() - eval_started
    assert_frozen(policy, versions)
    if parameter_sha256(visual) != visual_hash:
        raise RuntimeError("Qwen vision encoder hash changed")
    final_adapter_hash = parameter_sha256(policy, trainable_only=True)
    result = {"status": "verified_run" if final_adapter_hash != adapter_hash else "adapter_unchanged",
              "objective": args.objective, "model": args.model, "resolved_revision": args.revision,
              "init_adapter": init, "reward_weights": args.reward_weights, "optimizer_steps": step,
              "optimizer_steps_this_run": step - start_step, "requested_optimizer_steps": args.max_steps,
              "stop_reason": stop_reason, "groups_seen": groups_seen, "groups_with_signal": groups_with_signal,
              "train_examples": len(rows), "validation_examples": len(validation),
              "training_elapsed_seconds": training_seconds,
              "mean_step_seconds": sum(step_seconds) / len(step_seconds) if step_seconds else None,
              "peak_gpu_allocated_bytes": peak_allocated, "peak_gpu_reserved_bytes": peak_reserved,
              "before_evaluation_seconds": before_seconds, "after_evaluation_seconds": after_seconds,
              "adapter_sha256_before": adapter_hash, "adapter_sha256_after": final_adapter_hash,
              "vision_sha256_unchanged": True, "frozen_base_versions_unchanged": True,
              "sam_freeze_verification": eval_sam.verify_unchanged(),
              "reward_sam_freeze_verification": reward_sam.verify_unchanged() if reward_sam else None,
              "periodic_evaluations": evaluations, "before": before, "after": after}
    json_write(output / "summary.json", result)
    json_write(output / "progress.json", {"phase": "complete", "step": step, "status": result["status"],
                                          "stop_reason": stop_reason})
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    owns_output = False
    try:
        validate_args(args)
        from nucleus_rl.evaluate import read_manifest
        from nucleus_rl.sft import json_write
        from nucleus_rl.sft_data import validate_sft_splits
        rows, validation = read_manifest(args.manifest), read_manifest(args.eval_manifest)
        audit = validate_sft_splits(rows, validation, verify_files=True)
        if args.cohort_protocol:
            from nucleus_rl.cohort import load_official30_cohort
            from nucleus_rl.training_budget import check_cohort
            check_cohort(rows, validation, load_official30_cohort(args.cohort_protocol))
        if args.train_limit:
            rows = rows[:args.train_limit]
        if args.eval_limit:
            validation = validation[:args.eval_limit]
        check_init_adapter(args)
        if not Path(args.sam_checkpoint).is_file():
            raise FileNotFoundError(args.sam_checkpoint)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty output directory; existing experiment artifacts are preserved")
        output.mkdir(parents=True, exist_ok=True)
        owns_output = True
        json_write(output / "arguments.json", vars(args))
        json_write(output / "preflight.json", {"status": "passed_before_gpu_load", "selected_train": len(rows),
                                               "selected_validation": len(validation)})
        return run(args, rows, validation, audit, output)
    except Exception as error:
        if owns_output:
            from nucleus_rl.sft import json_write
            json_write(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
