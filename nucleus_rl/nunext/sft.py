"""NuNext stage 1: supervised fine-tuning to emit raster-ordered coordinate tokens.

``--sigma 0`` is naive SFT (NSFT, one-hot next-token loss). ``--sigma > 0`` adds spatial-aware
soft supervision (SASS). Chain-of-visual-thought is not implemented (see README.md).
The language model, its embeddings and output head are fully fine-tuned; the vision tower is frozen.
Micro-batch of one image with gradient accumulation. Under torchrun (single node) every rank
processes different images and gradients are averaged, so the effective batch is
--gradient-accumulation x world size; validation is sharded across ranks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest", action="append", required=True, help="Repeatable (fold1.jsonl, fold2.jsonl)")
    p.add_argument("--eval-manifest", action="append", required=True, help="Repeatable (fold3.jsonl)")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--bins", type=int, default=256, help="K coordinate bins over [0, 1]")
    p.add_argument("--sigma", type=float, default=0.0, help="SASS Gaussian width in bins; 0 = naive SFT")
    p.add_argument("--image-size", type=int, default=448, help="Square input size (multiple of 28)")
    p.add_argument("--augmentation", choices=("none", "d4"), default="d4")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, help="Optimizer steps; overrides --epochs")
    p.add_argument("--gradient-accumulation", type=int, default=16, help="Per rank; effective batch = this x GPUs")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--train-merger", action="store_true", help="Also train the vision-language merger")
    p.add_argument("--max-sequence-length", type=int, default=2048)
    p.add_argument("--radius", type=float, default=12.0, help="Detection match threshold in pixels")
    p.add_argument("--eval-every", type=int, default=250, help="Optimizer steps; 0 = only before/after")
    p.add_argument("--eval-limit", type=int, help="Validation subset for periodic evaluation (final uses all)")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--train-limit", type=int, help="Canary-only training subset")
    p.add_argument("--final-eval-limit", type=int, help="Canary-only final validation subset")
    p.add_argument("--skip-before-eval", action="store_true")
    p.add_argument("--skip-save", action="store_true", help="Do not write the final model (canaries)")
    p.add_argument("--save-every", type=int, default=0, help="Optimizer steps between full checkpoints; 0 = only final")
    p.add_argument("--wall-time-minutes", type=float, help="Stop training after this, then evaluate and save")
    p.add_argument("--seed", type=int, default=42)
    return p


def validate(args):
    if args.image_size % 28:
        raise ValueError("--image-size must be a multiple of 28 (Qwen2.5-VL patch 14 x merge 2)")
    if args.bins < 2 or args.sigma < 0 or args.gradient_accumulation < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid bins, sigma, accumulation or learning rate")
    if not 0 <= args.warmup_ratio < 1 or args.radius <= 0:
        raise ValueError("Invalid warmup ratio or radius")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise FileExistsError(f"{args.output} is not empty; use a new output directory")


def exposure_index(micro_step: int, rank: int, world: int) -> int:
    """Global sample index of a rank's micro-step: ranks interleave, so no image is seen twice per epoch."""
    return micro_step * world + rank


def lr_at(step, total, warmup, peak):
    """Linear warm-up, then cosine decay to zero."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run(args):
    import torch
    from transformers import set_seed

    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.data import epoch_order, load_example, read_manifests
    from nucleus_rl.nunext.evaluate import evaluate
    from nucleus_rl.nunext.modeling import (completion_logits, encode_training_example, finite_or_raise,
                                            gaussian_targets, load_policy, soft_label_loss)

    device = dist.setup()
    world, rank, main = dist.world_size(), dist.rank(), dist.is_main()
    output = Path(args.output)
    if main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "arguments.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2) + "\n")
    set_seed(args.seed)
    random.seed(args.seed)
    train = read_manifests(args.train_manifest, args.train_limit)
    validation = read_manifests(args.eval_manifest, args.final_eval_limit)
    periodic = validation[:args.eval_limit] if args.eval_limit else validation

    model, processor, first_id = load_policy(args.model, args.revision, bins=args.bins, dtype="float32",
                                             device=device, train_merger=args.train_merger)
    dist.broadcast_parameters(model)
    kernel = gaussian_targets(args.bins, args.sigma, device=model.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if main:
        print(f"trainable parameters: {sum(p.numel() for p in trainable) / 1e9:.3f} B; GPUs: {world}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    per_step = args.gradient_accumulation * world
    total = args.max_steps or max(1, int(args.epochs * len(train) / per_step))
    warmup = max(1, int(args.warmup_ratio * total))
    eval_kwargs = dict(first_id=first_id, bins=args.bins, image_size=args.image_size, radius=args.radius,
                       batch_size=args.eval_batch_size, max_new_tokens=args.max_new_tokens)
    history = {"steps": total, "train_images": len(train), "world_size": world, "effective_batch": per_step,
               "losses": [], "evaluations": []}
    if not args.skip_before_eval:
        history["before"] = evaluate(model, processor, periodic, output=output / "eval-before", **eval_kwargs)
        if main:
            print("before", json.dumps(history["before"]), flush=True)

    started, micro, step, window = time.time(), 0, 0, []
    stopped_early = False
    model.train()
    while step < total:
        if dist.any_true(bool(args.wall_time_minutes) and time.time() - started > 60 * (args.wall_time_minutes or 0)):
            stopped_early = True
            break
        for group_lr in optimizer.param_groups:
            group_lr["lr"] = lr_at(step, total, warmup, args.learning_rate)
        for _ in range(args.gradient_accumulation):
            exposure = exposure_index(micro, rank, world)
            micro += 1
            epoch, offset = divmod(exposure, len(train))
            row = train[epoch_order(len(train), epoch, args.seed)[offset]]
            view = random.Random(args.seed * 7919 + exposure).randrange(8) if args.augmentation == "d4" else 0
            example = load_example(row, args.bins, view, args.image_size)
            inputs, length = encode_training_example(processor, example["image"], example["target"],
                                                     args.max_sequence_length)
            inputs = inputs.to(model.device)
            with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                logits = completion_logits(model, inputs, length)[0]
            loss = soft_label_loss(logits, inputs["labels"][0, -length:], first_id, args.bins, kernel)
            (loss / args.gradient_accumulation).backward()
            window.append(float(finite_or_raise(loss.detach(), "loss")))
        dist.average_gradients(trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            entry = {"step": step, "loss": dist.mean(sum(window) / len(window)), "grad_norm": float(norm),
                     "lr": optimizer.param_groups[0]["lr"], "epoch": round(micro * world / len(train), 3),
                     "minutes": round((time.time() - started) / 60, 1)}
            history["losses"].append(entry)
            if main:
                print(json.dumps(entry), flush=True)
            window = []
        if args.eval_every and step % args.eval_every == 0 and step < total:
            summary = evaluate(model, processor, periodic, output=output / f"eval-step{step}", **eval_kwargs)
            history["evaluations"].append({"step": step, **summary})
            if main:
                print(f"eval step {step}", json.dumps(summary), flush=True)
            model.train()
        if args.save_every and step % args.save_every == 0 and step < total:
            save(model, processor, output / "checkpoints" / f"step-{step}")

    history.update(completed_steps=step, stopped_early=stopped_early,
                   training_minutes=round((time.time() - started) / 60, 1))
    if not args.skip_save:
        save(model, processor, output / "model")
    history["after"] = evaluate(model, processor, validation, output=output / "eval-final", **eval_kwargs)
    if main:
        print("final", json.dumps(history["after"]), flush=True)
        (output / "summary.json").write_text(json.dumps(history, indent=2) + "\n")
    dist.teardown()


def save(model, processor, path: Path):
    """Rank 0 writes; other ranks wait so nobody reads a half-written model."""
    from nucleus_rl.nunext import dist

    if not dist.is_main():
        dist.barrier()
        return
    import torch

    path.mkdir(parents=True, exist_ok=True)
    # bf16 copy on the CPU in <=2 GB shards: Databricks Files API uploads fail for files over 5 GB, and an
    # fp32 3B model is ~15 GB. Tied tensors stay shared so they are written once.
    copies, state = {}, {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.dtype, tuple(tensor.shape))
        if key not in copies:
            copies[key] = tensor.detach().to("cpu", dtype=torch.bfloat16 if tensor.is_floating_point() else tensor.dtype)
        state[name] = copies[key]
    model.config.use_cache = True
    model.save_pretrained(path, state_dict=state, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(path)
    model.config.use_cache = False
    del copies, state
    dist.barrier()


def main(argv=None):
    from nucleus_rl.nunext import dist

    args = build_parser().parse_args(argv)
    if dist.is_main():   # Other ranks would see rank 0's files and refuse.
        validate(args)
    run(args)


if __name__ == "__main__":
    main()
