"""Single-GPU Qwen text-policy LoRA GRPO pilot; SAM2 stays frozen.

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
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    return parser


def validate_args(args):
    if args.mode == "train" and not args.eval_manifest:
        raise ValueError("Training requires --eval-manifest for paired before/after evaluation")
    if args.mode == "train" and args.adapter:
        raise ValueError("This pilot starts new adapters; --adapter is only for rollout/evaluation")
    if args.group_size != 2:
        raise ValueError("This bounded single-GPU runner fixes group-size=2 and accumulation=2")
    if not 1 <= args.max_steps <= 100:
        raise ValueError("Use 1–100 optimizer steps for this pilot")
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.requires_grad_(False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    metadata = {
        "model": args.model, "resolved_revision": revision,
        "processor_revision": revision, "adapter": args.adapter,
        "image_coordinate_system": "original crop pixels; processor resizing does not change output coordinate contract",
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "trl", "peft", "accelerate", "datasets", "SAM-2"
        )},
    }
    return model, processor, metadata


def train_policy(args, model, processor, rows, validation, segmenter, output):
    import numpy as np
    import torch
    from datasets import Dataset, Image as DatasetImage
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from nucleus_rl.evaluate import ArtifactScorer, completion_text, prompt_for_image, run_evaluation, write_json
    from nucleus_rl.segmenter import parameter_sha256

    targets = text_lora_targets(model)
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
    before = run_evaluation(model, processor, validation, ArtifactScorer(segmenter, output / "before", args.max_objects), args.seed, args.max_completion_length)
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
        with Image.open(row["image_path"]) as image:
            prompt = prompt_for_image(*image.size, args.max_objects)
        # TRL's multimodal helper inserts the image token for the 'image' column.
        dataset_rows.append({"id": row["id"], "image": row["image_path"], "prompt": prompt})
    dataset = Dataset.from_list(dataset_rows).cast_column("image", DatasetImage())
    proof = {"finite_gradient_steps": 0, "nonzero_gradient_steps": 0, "adapter_changed_steps": 0, "stop_reason": None}

    class ProofCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
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
            row = {
                "step": state.global_step, "elapsed_seconds": time.monotonic() - self.started,
                "adapter_differs_from_initial": changed,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), **proof,
            }
            with (output / "optimizer_proof.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if state.global_step >= 10 and (
                not any(group["reward_std"] > 0 for group in scorer.groups)
                or not changed or all(group["all_zero"] for group in scorer.groups)
            ):
                proof["stop_reason"] = "No useful group reward variation or adapter update in the first 10 steps"
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for name, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite logged metric: {name}={value}")
            with (output / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")

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
    trainer = GRPOTrainer(
        model=model, args=config, processing_class=processor, train_dataset=dataset,
        reward_funcs=segmentation_and_format_reward, callbacks=[ProofCallback()],
    )
    if trainer.ref_model is not None:
        raise RuntimeError("Expected adapter-disabled reference; refusing an extra model copy")
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    processor.save_pretrained(output / "adapter")
    scorer.summary()
    after = run_evaluation(model, processor, validation, ArtifactScorer(segmenter, output / "after", args.max_objects), args.seed, args.max_completion_length)
    visual_hash_after = parameter_sha256(visual)
    if visual_hash_after != visual_hash_before:
        raise RuntimeError("Qwen image encoder changed despite freeze requirement")
    changed = parameter_sha256(model, trainable_only=True) != adapter_hash_before
    status = "verified_training_pilot" if changed and proof["nonzero_gradient_steps"] and not proof["stop_reason"] else "no_useful_update"
    return {
        "status": status, "optimizer_steps": trainer.state.global_step, **proof,
        "adapter_changed": changed, "adapter_sha256_before": adapter_hash_before,
        "adapter_sha256_after": parameter_sha256(model, trainable_only=True),
        "visual_sha256_before": visual_hash_before, "visual_sha256_after": visual_hash_after,
        "reference_uses_disabled_adapter": True, "before": before, "after": after,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    from nucleus_rl.evaluate import ArtifactScorer, read_manifest, run_evaluation, write_json
    from nucleus_rl.segmenter import FrozenSAM2, file_sha256

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Use a new empty --output directory to keep run artifacts unambiguous")
    write_json(output / "arguments.json", vars(args))
    started = time.monotonic()
    try:
        rows = read_manifest(args.manifest)
        validation = read_manifest(args.eval_manifest) if args.eval_manifest else []
        if args.mode == "train":
            if any(row["split"] != "train" for row in rows):
                raise ValueError("Training manifest must contain only train records")
            if any(row["split"] != "validation" for row in validation):
                raise ValueError("Evaluation manifest must contain only validation records")
            if {row["patient_id"] for row in rows} & {row["patient_id"] for row in validation}:
                raise ValueError("Train/validation patient overlap")
        model, processor, metadata = load_policy(args)
        metadata["manifest_sha256"] = file_sha256(args.manifest)
        metadata["eval_manifest_sha256"] = file_sha256(args.eval_manifest) if args.eval_manifest else None
        write_json(output / "metadata.json", metadata)
        segmenter = FrozenSAM2(args.sam_checkpoint, args.sam_config)
        write_json(output / "sam_before.json", segmenter.metadata)
        if args.mode == "train":
            result = train_policy(args, model, processor, rows, validation, segmenter, output)
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
