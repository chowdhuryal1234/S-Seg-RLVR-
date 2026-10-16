"""Bounded single-GPU Qwen text-policy LoRA GRPO; SAM2 stays frozen.

Use --help without installing GPU dependencies. This is engineering validation
with full reference masks, not a weak-supervision benchmark or clinical model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time


REPLAY_SETTINGS = ("coordinate_frame", "prompt_style", "min_pixels", "max_pixels")


class ReplaySettingAction(argparse.Action):
    """Remember explicit CLI settings so adapter defaults cannot override them."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        explicit = dict(getattr(namespace, "_explicit_replay_settings", None) or {})
        explicit[self.dest] = values
        namespace._explicit_replay_settings = explicit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rollout", "train", "evaluate"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-manifest", help="Disjoint validation manifest; required for training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main", help="Resolved to an immutable Hub commit before loading")
    parser.add_argument("--adapter", help="Saved adapter for rollout/evaluation; training starts from the base model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate the fixed --eval-manifest every N steps; 0 disables intermediate evaluation")
    parser.add_argument("--resume-from-checkpoint",
                        help="Complete checkpoint from this runner; use a NEW output directory and the same planned max-steps")
    parser.add_argument("--wall-time-minutes", type=float,
                        help="Soft training-stage limit checked between optimizer steps; final saving/evaluation require extra time")
    parser.add_argument("--cohort-protocol", help="Verified official30 membership JSON; defaults to the tracked protocol")
    parser.add_argument("--direct-gpu-load", action="store_true",
                        help="Stream model shards to the single CUDA device to reduce host RAM demand")
    parser.add_argument("--min-pixels", type=int, default=224 * 224, action=ReplaySettingAction)
    parser.add_argument("--max-pixels", type=int, default=448 * 448, action=ReplaySettingAction)
    parser.add_argument("--coordinate-frame", choices=("processed", "original"), default="processed",
                        action=ReplaySettingAction,
                        help="Model output pixel frame; processed follows Qwen2.5-VL's native grounding convention")
    parser.add_argument("--prompt-style", choices=("schema", "example"), default="schema",
                        action=ReplaySettingAction,
                        help="Optional invented numeric formatting example; schema preserves the original pilot prompt")
    return parser


def restore_adapter_settings(args):
    """Replay the saved input contract, rejecting explicit CLI disagreements.

    Older adapters may lack some metadata. Require those settings explicitly;
    defaults are not evidence of the prompt or processor used for training.
    This runs before GPU imports and before writing the effective arguments.
    """
    if not args.adapter:
        return
    path = Path(args.adapter) / "coordinate_config.json"
    explicit = getattr(args, "_explicit_replay_settings", None) or {}
    try:
        saved = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read adapter coordinate settings at {path}: {error}") from error
    if not isinstance(saved, dict):
        raise ValueError("Adapter coordinate_config.json must contain an object")
    missing = [key for key in REPLAY_SETTINGS if key not in saved and key not in explicit]
    if missing:
        flags = ", ".join("--" + key.replace("_", "-") for key in missing)
        raise ValueError(f"Adapter metadata is missing {flags}; supply these explicitly for legacy replay")
    effective = {}
    for key in REPLAY_SETTINGS:
        if key in saved:
            value = saved[key]
            valid = (
                value in ("processed", "original") if key == "coordinate_frame"
                else value in ("schema", "example") if key == "prompt_style"
                else type(value) is int and value > 0
            )
            if not valid:
                raise ValueError(f"Invalid saved adapter setting {key}={value!r}")
            if key in explicit and explicit[key] != value:
                flag = "--" + key.replace("_", "-")
                raise ValueError(f"{flag}={explicit[key]!r} conflicts with saved adapter value {value!r}")
            effective[key] = value
        else:
            effective[key] = explicit[key]
    if effective["max_pixels"] < effective["min_pixels"]:
        raise ValueError("Saved adapter pixel budgets require min_pixels <= max_pixels")
    for key, value in effective.items():
        setattr(args, key, value)
    args.adapter_replay_settings = {
        "config_path": str(path.resolve()) if path.is_file() else None,
        "restored_fields": [key for key in REPLAY_SETTINGS if key in saved],
        "explicit_legacy_fields": [key for key in REPLAY_SETTINGS if key not in saved],
        "effective": effective,
    }


def validate_args(args):
    if args.mode == "train" and not args.eval_manifest:
        raise ValueError("Training requires --eval-manifest for paired before/after evaluation")
    if args.mode == "train" and args.adapter:
        raise ValueError("This pilot starts new adapters; --adapter is only for rollout/evaluation")
    if args.group_size != 2:
        raise ValueError("This bounded single-GPU runner fixes group-size=2 and accumulation=2")
    if not 1 <= args.max_steps <= 5000:
        raise ValueError("Use 1–5000 optimizer steps for this bounded runner")
    if args.eval_every < 0:
        raise ValueError("Evaluation interval must be nonnegative")
    if args.wall_time_minutes is not None and (not math.isfinite(args.wall_time_minutes) or args.wall_time_minutes <= 0):
        raise ValueError("Wall-time limit must be finite and positive")
    if args.resume_from_checkpoint and args.mode != "train":
        raise ValueError("--resume-from-checkpoint is only for training; use --adapter for evaluation")
    if not 1 <= args.max_objects <= 8:
        raise ValueError("This crop schema supports 1–8 objects")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("Require 0 < min_pixels <= max_pixels")
    if args.max_completion_length <= 0 or args.save_steps <= 0 or args.lora_rank <= 0:
        raise ValueError("Token limit, save interval, and LoRA rank must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError("Require finite positive learning-rate and finite nonnegative beta")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner is intentionally single-process, single-GPU")


def restore_resume_revision(args):
    """Check completed payloads before importing GPU libraries or loading weights."""
    if not args.resume_from_checkpoint:
        return None
    from nucleus_rl.training_budget import verify_checkpoint
    marker = verify_checkpoint(args.resume_from_checkpoint)
    config = marker["contract"]["config"]
    if args.model != config["model"] or args.revision not in ("main", config["revision"]):
        raise ValueError("Resume requires the checkpoint's exact base model and revision")
    if marker["global_step"] >= args.max_steps:
        raise ValueError("Checkpoint already reached --max-steps; changing the planned schedule is not supported")
    args.revision = config["revision"]
    return marker


def training_contract(args, rows, validation, metadata, segmenter, cohort):
    from nucleus_rl.training_budget import dataset_identity, file_digest
    keys = ("model", "revision", "seed", "max_steps", "group_size", "max_completion_length", "max_objects",
            "lora_rank", "learning_rate", "beta", "save_steps", "eval_every", "min_pixels", "max_pixels",
            "coordinate_frame", "prompt_style")
    source = Path(__file__).resolve().parent
    return {
        "config": {key: getattr(args, key) for key in keys},
        "fixed_training": {"microbatch": 1, "gradient_accumulation": 2, "num_iterations": 1,
                           "dtype": "bfloat16", "loss_type": "grpo", "optimizer": "adamw_torch"},
        "train_data": dataset_identity(rows), "validation_data": dataset_identity(validation),
        "cohort_protocol_sha256": cohort["protocol_sha256"],
        "sam": {key: value for key, value in segmenter.metadata.items() if key != "checkpoint"},
        "packages": metadata["packages"],
        "source_sha256": {name: file_digest(source / name) for name in (
            "train.py", "training_budget.py", "evaluate.py", "rewards.py", "segmenter.py", "cohort.py")},
    }


def text_lora_targets(model):
    targets = [
        name for name, _ in model.named_modules()
        if name.endswith((".q_proj", ".v_proj"))
        and ".layers." in name and "visual" not in name and "vision" not in name
    ]
    if not targets:
        raise ValueError("No text-layer q_proj/v_proj modules found; refusing broad LoRA targeting")
    return targets


def assert_text_adapters_only(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any("lora_" not in name or "visual" in name or "vision" in name for name in names):
        raise RuntimeError(f"Only text LoRA adapters may be trainable: {names}")
    return names


def load_policy(args):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no training has been performed")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configured pilot requires BF16-capable CUDA hardware")
    if args.adapter:
        adapter_config = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        adapter_revision = adapter_config.get("revision")
        if adapter_revision:
            if args.revision != "main" and args.revision != adapter_revision:
                raise ValueError("Requested base revision differs from the saved adapter revision")
            args.revision = adapter_revision
        adapter_base = adapter_config.get("base_model_name_or_path")
        if adapter_base and adapter_base != args.model:
            raise ValueError(f"Adapter expects base {adapter_base}; use the matching --model")
    set_seed(args.seed)
    revision = args.revision if re.fullmatch(r"[0-9a-f]{40}", args.revision) else model_info(args.model, revision=args.revision).sha
    args.revision = revision
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    processor.tokenizer.padding_side = "left"
    # Large inference checkpoints can exceed the job's host-RAM allocation.
    # Stream shards directly to one GPU only when explicitly requested.
    load_options = {"device_map": {"": "cuda"}, "low_cpu_mem_usage": True} if getattr(args, "direct_gpu_load", False) else {}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        **load_options,
    )
    if not load_options:
        model = model.to("cuda")
    model.requires_grad_(False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    metadata = {
        "model": args.model, "resolved_revision": revision,
        "processor_revision": revision, "adapter": args.adapter,
        "direct_gpu_load": bool(load_options),
        "image_coordinate_system": args.coordinate_frame,
        "prompt_style": args.prompt_style,
        "adapter_replay_settings": getattr(args, "adapter_replay_settings", None),
        "sam_coordinate_system": "original crop pixels, obtained by one fixed scale conversion",
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "trl", "peft", "accelerate", "datasets", "SAM-2"
        )},
    }
    return model, processor, metadata


def train_policy(args, model, processor, rows, validation, segmenter, output, contract):
    import torch
    from datasets import Dataset, Image as DatasetImage
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from nucleus_rl.evaluate import ArtifactScorer, completion_text, prompt_for_image, run_evaluation, write_json
    from nucleus_rl.segmenter import parameter_sha256
    from nucleus_rl.training_budget import (
        load_training_state, preserve_evaluation_state, restore_rng, save_training_state,
        seal_checkpoint, verify_checkpoint, wall_budget_reached,
    )

    targets = text_lora_targets(model)
    resumed_step = 0
    if args.resume_from_checkpoint:
        resumed_step = verify_checkpoint(args.resume_from_checkpoint, contract)["global_step"]
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0, target_modules=targets, bias="none", revision=args.revision,
        ))
    trainable = assert_text_adapters_only(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    adapter_hash_before = parameter_sha256(model, trainable_only=True)
    visual = next((module for name, module in model.named_modules() if name.endswith(".visual")), None)
    if visual is None:
        raise RuntimeError("Could not identify Qwen visual encoder for freeze audit")
    visual_hash_before = parameter_sha256(visual)
    write_json(output / "trainable_parameters.json", {
        "names": trainable, "targets": targets,
        "count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "adapter_sha256_before": adapter_hash_before, "visual_sha256_before": visual_hash_before,
    })
    def evaluate_at(directory):
        # Evaluation seeds RNGs; restore them and every submodule's train/eval
        # mode so validation cannot change the following training sample.
        with preserve_evaluation_state(model):
            return run_evaluation(model, processor, validation, ArtifactScorer(segmenter, directory, args.max_objects),
                                  args.seed, args.max_completion_length)

    evaluation_started = time.monotonic()
    before = evaluate_at(output / "before")
    before_evaluation_seconds = time.monotonic() - evaluation_started
    set_seed(args.seed)
    scorer = ArtifactScorer(segmenter, output / "training_completions", args.max_objects)
    lookup = {row["id"]: row for row in rows}

    def segmentation_and_format_reward(completions, id, trainer_state=None, **kwargs):
        step = trainer_state.global_step if trainer_state is not None else None
        results = [scorer.score(completion_text(text), lookup[record_id], step) for text, record_id in zip(completions, id)]
        for start in range(0, len(results), args.group_size):
            group = results[start:start + args.group_size]
            if len(group) != args.group_size or len({row["id"] for row in group}) != 1:
                raise RuntimeError("GRPO reward callback received incomplete or mixed-image groups")
            scorer.record_group(group, step)
        return [float(result["total_reward"]) for result in results]

    dataset_rows = []
    for row in rows:
        prompt = prompt_for_image(
            row["prompt_width"], row["prompt_height"], args.max_objects,
            coordinate_frame=row["coordinate_frame"],
            prompt_style=row.get("prompt_style", "schema"),
        )
        # TRL's multimodal helper inserts the image token for the 'image' column.
        dataset_rows.append({"id": row["id"], "image": row["image_path"], "prompt": prompt})
    dataset = Dataset.from_list(dataset_rows).cast_column("image", DatasetImage())
    proof = {"finite_gradient_steps": 0, "nonzero_gradient_steps": 0, "adapter_changed_steps": 0, "stop_reason": None}
    progress = {"last_checkpoint": None, "periodic_evaluations": [], "resumed_from_step": resumed_step,
                "peak_gpu_allocated_bytes": 0, "peak_gpu_reserved_bytes": 0}
    coordinate_config = {key: getattr(args, key) for key in REPLAY_SETTINGS}
    coordinate_config["conversion"] = "x_original=x_prompt*original_width/prompt_width; likewise y"
    runner_args = args

    class ProofCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.step_started = self.started
            torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                raise RuntimeError("Missing or nonfinite adapter gradients; stopping pilot")
            proof["finite_gradient_steps"] += 1
            if any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
                proof["nonzero_gradient_steps"] += 1

        def on_step_end(self, args, state, control, model=None, **kwargs):
            assert_text_adapters_only(model)
            changed = any(not torch.equal(parameter.detach().cpu(), initial[name]) for name, parameter in model.named_parameters() if parameter.requires_grad)
            proof["adapter_changed_steps"] += int(changed)
            elapsed = time.monotonic() - self.started
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            progress["peak_gpu_allocated_bytes"] = max(progress["peak_gpu_allocated_bytes"], allocated)
            progress["peak_gpu_reserved_bytes"] = max(progress["peak_gpu_reserved_bytes"], reserved)
            if state.global_step - resumed_step >= 10 and (
                not any(group["reward_std"] > 0 for group in scorer.groups)
                or not changed or all(group["all_zero"] for group in scorer.groups)
            ):
                proof["stop_reason"] = "No useful group reward variation or adapter update in the first 10 new steps"
                control.should_training_stop = True
            elif wall_budget_reached(elapsed, runner_args.wall_time_minutes):
                proof["stop_reason"] = "wall_time_budget"
                control.should_training_stop = True
            due_eval = runner_args.eval_every and state.global_step % runner_args.eval_every == 0
            if control.should_training_stop or state.global_step == runner_args.max_steps or due_eval:
                control.should_save = True
            row = {
                "step": state.global_step, "elapsed_seconds": elapsed,
                "step_duration_seconds": time.monotonic() - self.step_started,
                "adapter_differs_from_initial": changed,
                "step_peak_gpu_allocated_bytes": allocated, "step_peak_gpu_reserved_bytes": reserved,
                "peak_gpu_allocated_bytes": progress["peak_gpu_allocated_bytes"],
                "peak_gpu_reserved_bytes": progress["peak_gpu_reserved_bytes"], **proof,
            }
            with (output / "optimizer_proof.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return control

        def on_save(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "coordinate_config.json", coordinate_config)
            save_training_state(checkpoint, optimizer, lr_scheduler)
            seal_checkpoint(checkpoint, contract, {**proof, "resumed_from_step": resumed_step})
            progress["last_checkpoint"] = str(checkpoint.resolve())
            write_json(output / "progress.json", {"step": state.global_step, **progress, **proof})
            due_eval = runner_args.eval_every and state.global_step % runner_args.eval_every == 0
            if due_eval and not control.should_training_stop and state.global_step < runner_args.max_steps:
                eval_started = time.monotonic()
                summary = evaluate_at(output / "evaluations" / f"step-{state.global_step}")
                record = {"step": state.global_step, "duration_seconds": time.monotonic() - eval_started,
                          "summary": summary}
                progress["periodic_evaluations"].append(record)
                with (output / "validation_history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if wall_budget_reached(time.monotonic() - self.started, runner_args.wall_time_minutes):
                    proof["stop_reason"] = "wall_time_budget"
                    control.should_training_stop = True
                write_json(output / "progress.json", {"step": state.global_step, **progress, **proof})
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for name, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite logged metric: {name}={value}")
            with (output / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")

    class SafeResumeGRPOTrainer(GRPOTrainer):
        """Keep Trainer's adapter/step/data resume, loading no pickle payloads.

        Transformers 4.56 refuses torch.load under Torch 2.5. The additional
        safetensors/JSON bundle restores only optimizer, scheduler and RNG.
        Fixed accumulation=2, generations=2, iterations=1 puts each optimizer
        checkpoint at a generation-buffer boundary in TRL 0.23.1.
        """
        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint:
                verify_checkpoint(checkpoint, contract)
                bundle = load_training_state(checkpoint)
                self.optimizer.load_state_dict(bundle["optimizer"])
                self.lr_scheduler.load_state_dict(bundle["scheduler"])

        def _load_rng_state(self, checkpoint):
            if checkpoint:
                restore_rng(load_training_state(checkpoint)["rng"])

    config = GRPOConfig(
        output_dir=str(output / "checkpoints"), max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        generation_batch_size=2, num_generations=2, num_iterations=1,
        learning_rate=args.learning_rate, beta=args.beta, loss_type="grpo",
        max_prompt_length=None, max_completion_length=args.max_completion_length,
        use_vllm=False, bf16=True, gradient_checkpointing=True,
        # TRL 0.23.1 restores checkpointing with Transformers' reentrant default
        # after generation. Matching it here also enables embedding OUTPUT
        # gradients in prepare_peft_model, necessary with frozen base weights.
        # This does not unfreeze embedding or image-encoder parameters.
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="adamw_torch", report_to="none", logging_steps=1,
        save_steps=args.save_steps, save_total_limit=2, seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0, disable_dropout=True,
        temperature=1.0, scale_rewards="group", mask_truncated_completions=True,
    )
    write_json(output / "grpo_config.json", config.to_dict())
    callback = ProofCallback()
    trainer = SafeResumeGRPOTrainer(
        model=model, args=config, processing_class=processor, train_dataset=dataset,
        reward_funcs=segmentation_and_format_reward, callbacks=[callback],
    )
    if trainer.ref_model is not None:
        raise RuntimeError("Expected adapter-disabled reference; refusing an extra model copy")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    training_elapsed = time.monotonic() - callback.started
    trainer.save_model(str(output / "adapter"))
    processor.save_pretrained(output / "adapter")
    write_json(output / "adapter" / "coordinate_config.json", coordinate_config)
    scorer.summary()
    evaluation_started = time.monotonic()
    after = evaluate_at(output / "after")
    after_evaluation_seconds = time.monotonic() - evaluation_started
    visual_hash_after = parameter_sha256(visual)
    if visual_hash_after != visual_hash_before:
        raise RuntimeError("Qwen image encoder changed despite freeze requirement")
    changed = parameter_sha256(model, trainable_only=True) != adapter_hash_before
    status = "verified_training_pilot" if changed and proof["nonzero_gradient_steps"] and proof["stop_reason"] in (None, "wall_time_budget") else "no_useful_update"
    return {
        "status": status, "optimizer_steps": trainer.state.global_step, **proof, **progress,
        "optimizer_steps_this_run": trainer.state.global_step - resumed_step,
        "requested_optimizer_steps": args.max_steps,
        "completed_requested_steps": trainer.state.global_step == args.max_steps,
        "training_elapsed_seconds": training_elapsed, "wall_time_minutes": args.wall_time_minutes,
        "before_evaluation_seconds": before_evaluation_seconds,
        "after_evaluation_seconds": after_evaluation_seconds,
        "adapter_changed": changed, "adapter_sha256_before": adapter_hash_before,
        "adapter_sha256_after": parameter_sha256(model, trainable_only=True),
        "visual_sha256_before": visual_hash_before, "visual_sha256_after": visual_hash_after,
        "reference_uses_disabled_adapter": True, "before": before, "after": after,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        restore_adapter_settings(args)
        validate_args(args)
        restore_resume_revision(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    from nucleus_rl.evaluate import ArtifactScorer, prepare_coordinate_rows, read_manifest, run_evaluation, write_json
    from nucleus_rl.segmenter import FrozenSAM2, file_sha256
    from nucleus_rl.cohort import load_official30_cohort
    from nucleus_rl.training_budget import check_cohort, verify_checkpoint

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Use a new empty --output directory to keep run artifacts unambiguous")
    write_json(output / "arguments.json", vars(args))
    started = time.monotonic()
    try:
        rows = read_manifest(args.manifest)
        validation = read_manifest(args.eval_manifest) if args.eval_manifest else []
        cohort = None
        if args.mode == "train":
            cohort = load_official30_cohort(args.cohort_protocol)
            check_cohort(rows, validation, cohort)
        model, processor, metadata = load_policy(args)
        rows = prepare_coordinate_rows(rows, processor, args.coordinate_frame)
        validation = prepare_coordinate_rows(validation, processor, args.coordinate_frame)
        for row in rows + validation:
            row["prompt_style"] = args.prompt_style
        frame_keys = ("id", "split", "coordinate_frame", "prompt_style", "original_width", "original_height",
                      "processed_width", "processed_height", "prompt_width", "prompt_height")
        write_json(output / "coordinate_frames.json", [
            {key: row[key] for key in frame_keys} for row in rows + validation
        ])
        metadata["manifest_sha256"] = file_sha256(args.manifest)
        metadata["eval_manifest_sha256"] = file_sha256(args.eval_manifest) if args.eval_manifest else None
        metadata["cohort_protocol_sha256"] = cohort["protocol_sha256"] if cohort else None
        write_json(output / "metadata.json", metadata)
        segmenter = FrozenSAM2(args.sam_checkpoint, args.sam_config)
        write_json(output / "sam_before.json", segmenter.metadata)
        if args.mode == "train":
            contract = training_contract(args, rows, validation, metadata, segmenter, cohort)
            if args.resume_from_checkpoint:
                verify_checkpoint(args.resume_from_checkpoint, contract)
            write_json(output / "training_contract.json", contract)
            result = train_policy(args, model, processor, rows, validation, segmenter, output, contract)
        else:
            scorer = ArtifactScorer(segmenter, output / "predictions", args.max_objects)
            result = run_evaluation(
                model, processor, rows, scorer, args.seed, args.max_completion_length,
                group_size=args.group_size if args.mode == "rollout" else 1,
            )
            result["status"] = "completed_inference_only"
        result["sam"] = segmenter.verify_unchanged()
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] != "no_useful_update" else 2
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed", "type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
