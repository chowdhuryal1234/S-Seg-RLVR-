"""NuNext stage 2: GRPO from a NuNext SFT model (paper Sec. 3.4).

Reward: distribution matching F1 (Hungarian matching within --radius pixels); malformed or
truncated rollouts get zero. Optional low-variance group filtering (LVGF, --delta) and
fine-grained advantage shaping (FGAS, --fgas-decay). One policy update per sampled batch
(fully on-policy, so the PPO ratio is 1 and clipping is inactive), k3 KL to the frozen SFT model.
The task-guided PQ reward is out of scope here (see README.md).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- pure functions (CPU-tested)

def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> np.ndarray:
    """(r - mean) / (std + eps) within one group (unbiased std, as in TRL)."""
    values = np.asarray(rewards, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return (values - values.mean()) / (std + eps)


def keep_group(rewards: Sequence[float], delta: float) -> bool:
    """LVGF: drop groups whose reward std is below delta. delta=0 keeps every group with any spread."""
    values = np.asarray(rewards, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return bool(std > 1e-8 and std >= delta)  # 1e-8: identical rewards can give a float-noise std


def fgas_token_advantages(advantage: float, length: int, valid: bool, pred_is_tp: Sequence[bool],
                          decay: float) -> np.ndarray:
    """Per-token advantages for one rollout of ``length`` tokens (paper Eq. 18).

    Point n owns completion tokens 2n and 2n+1. With A > 0, false-positive points get decay*A;
    with A < 0, true-positive points get decay*A. The end token and invalid rollouts keep A.
    decay=1 disables FGAS.
    """
    shaped = np.full(length, float(advantage))
    if not valid or decay == 1.0 or advantage == 0:
        return shaped
    if 2 * len(pred_is_tp) > length:
        raise ValueError("More predicted points than completion tokens")
    for index, is_tp in enumerate(pred_is_tp):
        if (advantage > 0 and not is_tp) or (advantage < 0 and is_tp):
            shaped[2 * index:2 * index + 2] = decay * advantage
    return shaped


# ---------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-model", required=True, help="NuNext SFT output folder (…/model)")
    p.add_argument("--train-manifest", action="append", required=True)
    p.add_argument("--eval-manifest", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--augmentation", choices=("none", "d4"), default="d4")
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--rollout-batch", type=int, default=4, help="Rollouts per forward/backward micro-batch")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04, help="k3 KL coefficient to the SFT model")
    p.add_argument("--delta", type=float, default=0.0, help="LVGF threshold on group reward std; 0 = off")
    p.add_argument("--fgas-decay", type=float, default=1.0, help="FGAS decay in (0,1]; 1 = off")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-limit", type=int)
    p.add_argument("--final-eval-limit", type=int)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--skip-before-eval", action="store_true")
    p.add_argument("--skip-save", action="store_true", help="Do not write the final model (canaries)")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--wall-time-minutes", type=float)
    p.add_argument("--seed", type=int, default=42)
    return p


def validate(args):
    if args.group_size < 2 or args.prompts_per_step < 1 or args.rollout_batch < 1:
        raise ValueError("Need group size >= 2 and positive batch sizes")
    if not 0 < args.fgas_decay <= 1 or args.delta < 0 or args.beta < 0 or args.temperature <= 0:
        raise ValueError("Invalid FGAS decay, delta, beta or temperature")
    if args.image_size % 28:
        raise ValueError("--image-size must be a multiple of 28")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise FileExistsError(f"{args.output} is not empty; use a new output directory")


def run(args):
    import torch
    from transformers import set_seed

    from nucleus_rl.nunext.data import epoch_order, load_example, read_manifests
    from nucleus_rl.nunext.evaluate import evaluate, score_completion
    from nucleus_rl.nunext.modeling import (coordinate_first_id, end_ids, finite_or_raise, generate, load_policy,
                                            rollout_batch, token_logprobs)
    from nucleus_rl.nunext.sft import save

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    set_seed(args.seed)
    train = read_manifests(args.train_manifest, args.train_limit)
    validation = read_manifests(args.eval_manifest, args.final_eval_limit)
    periodic = validation[:args.eval_limit] if args.eval_limit else validation

    policy, processor, _ = load_policy(args.init_model, "main", dtype="float32")
    reference, _, _ = load_policy(args.init_model, "main", dtype="bfloat16", trainable=False)
    first_id = coordinate_first_id(processor.tokenizer, args.bins)
    stop_ids = end_ids(processor, policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    eval_kwargs = dict(first_id=first_id, bins=args.bins, image_size=args.image_size, radius=args.radius,
                       batch_size=args.eval_batch_size, max_new_tokens=args.max_new_tokens)
    history = {"steps": [], "evaluations": []}
    if not args.skip_before_eval:
        history["before"] = evaluate(policy, processor, periodic, output=output / "eval-before", **eval_kwargs)
        print("before", json.dumps(history["before"]), flush=True)

    started, exposure, step, stopped_early = time.time(), 0, 0, False
    autocast = dict(device_type=policy.device.type, dtype=torch.bfloat16, enabled=policy.device.type == "cuda")
    while step < args.max_steps:
        if args.wall_time_minutes and time.time() - started > 60 * args.wall_time_minutes:
            stopped_early = True
            break
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * min(1.0, (step + 1) / max(args.warmup_steps, 1))
        # ---- sample
        examples, rows = [], []
        for _ in range(args.prompts_per_step):
            epoch, offset = divmod(exposure, len(train))
            row = train[epoch_order(len(train), epoch, args.seed)[offset]]
            view = random.Random(args.seed * 7919 + exposure).randrange(8) if args.augmentation == "d4" else 0
            exposure += 1
            example = load_example(row, args.bins, view, args.image_size)
            # Rewards compare against the augmented centroids, in the original pixel frame.
            rows.append({**row, "centroids": [list(p) for p in example["points"]]})
            examples.append(example)
        policy.eval()
        sequences = generate(policy, processor, [e["image"] for e in examples], do_sample=True,
                             temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                             num_return_sequences=args.group_size)
        policy.train()
        # ---- score, filter, shape advantages
        stats = {"rewards": [], "invalid": 0, "filtered_groups": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        work = []  # (example, completions, token advantages)
        for index, (example, row) in enumerate(zip(examples, rows)):
            group = sequences[index * args.group_size:(index + 1) * args.group_size]
            scored = [score_completion(ids, row, first_id, args.bins, stop_ids, args.radius) for ids in group]
            rewards = [match.f1 if parsed.valid else 0.0 for parsed, _, match in scored]
            stats["rewards"].extend(rewards)
            for parsed, points, match in scored:
                stats["invalid"] += not parsed.valid
                stats["pred"] += len(points)
                stats["tp"] += match.tp
                stats["fp"] += match.fp
                stats["fn"] += match.fn
            if not keep_group(rewards, args.delta):
                stats["filtered_groups"] += 1
                continue
            advantages = group_advantages(rewards)
            shaped = [fgas_token_advantages(a, len(ids), parsed.valid, match.pred_is_tp, args.fgas_decay)
                      for a, ids, (parsed, _, match) in zip(advantages, group, scored)]
            work.append((example, group, shaped))
        # ---- update
        kept = sum(len(group) for _, group, _ in work)
        loss_total, kl_total = 0.0, 0.0
        if kept:
            for example, group, shaped in work:
                for start in range(0, len(group), args.rollout_batch):
                    completions = group[start:start + args.rollout_batch]
                    if not any(completions):
                        continue
                    completions = [c if c else [sorted(stop_ids)[0]] for c in completions]
                    batch, ids, mask = rollout_batch(processor, example["image"], completions, policy.device)
                    advantage = torch.zeros(ids.shape, dtype=torch.float32, device=policy.device)
                    for row_index, values in enumerate(shaped[start:start + args.rollout_batch]):
                        advantage[row_index, :len(values)] = torch.as_tensor(values, dtype=torch.float32)
                    with torch.no_grad(), torch.autocast(**autocast):
                        ref_logp = token_logprobs(reference, batch, ids, args.temperature)
                    with torch.autocast(**autocast):
                        logp = token_logprobs(policy, batch, ids, args.temperature)
                    ratio = torch.exp(logp - logp.detach())
                    delta_ref = ref_logp - logp
                    kl = torch.exp(delta_ref) - delta_ref - 1
                    per_token = -advantage * ratio + args.beta * kl
                    sequence_loss = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    loss = sequence_loss.sum() / kept
                    loss.backward()
                    loss_total += float(finite_or_raise(loss.detach(), "loss"))
                    kl_total += float(((kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).sum()) / kept
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
            optimizer.step()
        else:
            norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        step += 1
        entry = {"step": step, "reward": float(np.mean(stats["rewards"])), "kept_rollouts": kept,
                 "filtered_groups": stats["filtered_groups"], "invalid": stats["invalid"],
                 "pred_per_image": stats["pred"] / len(stats["rewards"]), "tp": stats["tp"], "fp": stats["fp"],
                 "fn": stats["fn"], "loss": loss_total, "kl": kl_total, "grad_norm": norm,
                 "lr": optimizer.param_groups[0]["lr"], "minutes": round((time.time() - started) / 60, 1)}
        history["steps"].append(entry)
        print(json.dumps(entry), flush=True)
        if not math.isfinite(entry["reward"]):
            raise FloatingPointError("reward became non-finite")
        if args.eval_every and step % args.eval_every == 0 and step < args.max_steps:
            summary = evaluate(policy, processor, periodic, output=output / f"eval-step{step}", **eval_kwargs)
            history["evaluations"].append({"step": step, **summary})
            print(f"eval step {step}", json.dumps(summary), flush=True)
        if args.save_every and step % args.save_every == 0 and step < args.max_steps:
            save(policy, processor, output / "checkpoints" / f"step-{step}")

    history.update(completed_steps=step, stopped_early=stopped_early,
                   training_minutes=round((time.time() - started) / 60, 1))
    if not args.skip_save:
        save(policy, processor, output / "model")
    history["after"] = evaluate(policy, processor, validation, output=output / "eval-final", **eval_kwargs)
    print("final", json.dumps(history["after"]), flush=True)
    (output / "summary.json").write_text(json.dumps(history, indent=2) + "\n")


def main(argv=None):
    from nucleus_rl.nunext import dist

    if dist.enabled():
        raise SystemExit("The NuNext GRPO runner is single-GPU; launch it without torchrun")
    args = build_parser().parse_args(argv)
    validate(args)
    run(args)


if __name__ == "__main__":
    main()
