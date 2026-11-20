"""NuNext stage 2: GRPO from a NuNext SFT model (paper Sec. 3.4).

Reward: distribution matching F1 (Hungarian matching within --radius pixels); malformed or
truncated rollouts get zero. Options: --capped-matching (pairs beyond the radius unassignable,
so no correct point is lost to the global assignment), --reward-beta (F-beta; > 1 weighs recall
more) and, for box-token models, --box-iou-weight w: reward = (1 - w) F-beta + w * mean IoU between
each matched nucleus's predicted box and its reference box (centroid-centred tight-box size).
Task reward: --pq-weight p mixes in panoptic quality: every valid rollout's boxes + centres prompt a
frozen SAM2 (--sam-checkpoint), the masks are scored against the instance mask of the same D4 view,
and reward = (1 - p) * detection reward + p * PQ (p = 1: PQ alone). --rollout-end-bias samples
rollouts with a bias on the end token's logit; the update still uses the unbiased policy's
log-probs, so this makes the update slightly off-policy. Optional low-variance group filtering (LVGF, --delta) and
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

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two (x0, y0, x1, y1) boxes; 0 when either has no area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rollout_reward(valid: bool, match, beta: float = 1.0, iou_weight: float = 0.0,
                   pred_boxes: Sequence | None = None, gt_boxes: Sequence | None = None) -> tuple[float, float | None]:
    """(reward, mean matched-box IoU or None). Invalid rollouts get 0.

    With iou_weight > 0: (1 - w) * F-beta + w * mean IoU over matched pairs (0 when nothing matched).
    """
    if not valid:
        return 0.0, (0.0 if iou_weight else None)
    score = match.f_beta(beta)
    if not iou_weight:
        return score, None
    ious = [box_iou(pred_boxes[p], gt_boxes[g]) for p, g in match.pairs]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return (1 - iou_weight) * score + iou_weight * mean_iou, mean_iou


def rollout_pq(points, boxes, reference, predict, negatives: int = 0, radius: float = 32.0) -> float:
    """PQ of one rollout: predict(points, boxes, negatives) -> SAM masks, resolved and scored against ``reference``.

    No predictions on an image without nuclei scores 1; anything else without matches scores 0.
    """
    from nucleus_rl.nunext.pq_eval import negative_points
    from nucleus_rl.rewards import instance_metrics, masks_to_instances

    reference = np.asarray(reference).astype(np.int64)
    if points:
        masks, scores = predict(list(points), list(boxes), negative_points(points, negatives, radius))
        instances = masks_to_instances(np.asarray(masks, dtype=np.float32), scores=np.asarray(scores))
    else:
        instances = np.zeros(reference.shape, dtype=np.int32)
    return float(instance_metrics(instances, reference)["pq"])


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
                          decay: float, per_object: int = 2) -> np.ndarray:
    """Per-token advantages for one rollout of ``length`` tokens (paper Eq. 18).

    Nucleus n owns completion tokens k*n .. k*n+k-1 (k = 2 for x y, 4 for x y w h; its box size is
    credited with its centre). With A > 0, false-positive points get decay*A;
    with A < 0, true-positive points get decay*A. The end token and invalid rollouts keep A.
    decay=1 disables FGAS.
    """
    shaped = np.full(length, float(advantage))
    if not valid or decay == 1.0 or advantage == 0:
        return shaped
    if per_object * len(pred_is_tp) > length:
        raise ValueError("More predicted points than completion tokens")
    for index, is_tp in enumerate(pred_is_tp):
        if (advantage > 0 and not is_tp) or (advantage < 0 and is_tp):
            shaped[per_object * index:per_object * (index + 1)] = decay * advantage
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
    p.add_argument("--box-tokens", action="store_true", help="The SFT model outputs x y w h per nucleus")
    p.add_argument("--capped-matching", action="store_true",
                   help="Reward and FGAS labels from capped matching (pairs beyond --radius unassignable)")
    p.add_argument("--reward-beta", type=float, default=1.0, help="F-beta reward; > 1 weighs recall more")
    p.add_argument("--box-iou-weight", type=float, default=0.0,
                   help="Box-token models: weight of the matched-box IoU term in the reward (0 = off)")
    p.add_argument("--pq-weight", type=float, default=0.0,
                   help="Box-token models: weight of SAM panoptic quality in the reward (1 = PQ alone, 0 = off)")
    p.add_argument("--sam-checkpoint", help="SAM2 checkpoint for the PQ reward")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam-negatives", type=int, default=4, help="PQ reward: nearest other centres as negative points")
    p.add_argument("--rollout-end-bias", type=float, default=0.0,
                   help="Added to the end tokens' logits while sampling rollouts (negative = longer outputs)")
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
    if args.reward_beta <= 0 or not 0 <= args.box_iou_weight < 1:
        raise ValueError("Need --reward-beta > 0 and 0 <= --box-iou-weight < 1")
    if args.box_iou_weight and not args.box_tokens:
        raise ValueError("--box-iou-weight needs a box-token model (--box-tokens)")
    if not 0 <= args.pq_weight <= 1:
        raise ValueError("--pq-weight must be in [0, 1]")
    if args.pq_weight and not (args.box_tokens and args.sam_checkpoint):
        raise ValueError("--pq-weight needs a box-token model (--box-tokens) and --sam-checkpoint")
    if args.image_size % 28:
        raise ValueError("--image-size must be a multiple of 28")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise FileExistsError(f"{args.output} is not empty; use a new output directory")


def run(args):
    import torch
    from transformers import set_seed

    from nucleus_rl.nunext.coords import bins_to_boxes
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
    segmenter = None
    if args.pq_weight:
        from nucleus_rl.nunext.pq_eval import SAMSegmenter
        segmenter = SAMSegmenter(args.sam_checkpoint, args.sam_config, str(policy.device))
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    per_object = 4 if args.box_tokens else 2
    eval_kwargs = dict(first_id=first_id, bins=args.bins, image_size=args.image_size, radius=args.radius,
                       per_object=per_object,
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
            example = load_example(row, args.bins, view, args.image_size, boxes=args.box_iou_weight > 0,
                                   arrays=args.pq_weight > 0)
            # Rewards compare against the augmented centroids, in the original pixel frame.
            target = {**row, "centroids": [list(p) for p in example["points"]]}
            if args.pq_weight:
                target["pixels"], target["mask"] = example["pixels"], example["mask"]
            if args.box_iou_weight:
                target["boxes"] = [(max(0.0, x - w / 2), max(0.0, y - h / 2), min(float(row["width"]), x + w / 2),
                                    min(float(row["height"]), y + h / 2)) for x, y, w, h in example["objects"]]
            rows.append(target)
            examples.append(example)
        policy.eval()
        sequences = generate(policy, processor, [e["image"] for e in examples], do_sample=True,
                             temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                             num_return_sequences=args.group_size, end_bias=args.rollout_end_bias)
        policy.train()
        # ---- score, filter, shape advantages
        stats = {"rewards": [], "box_iou": [], "pq": [], "invalid": 0, "filtered_groups": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        work = []  # (example, completions, token advantages)
        for index, (example, row) in enumerate(zip(examples, rows)):
            group = sequences[index * args.group_size:(index + 1) * args.group_size]
            scored = [score_completion(ids, row, first_id, args.bins, stop_ids, args.radius, per_object,
                                       args.capped_matching)
                      for ids in group]
            rewards = []
            if segmenter is not None:
                segmenter.set_image(row["pixels"])     # one image embedding for the whole group
            for parsed, points, match in scored:
                pred_boxes = (bins_to_boxes(parsed.bins, row["width"], row["height"], args.bins)
                              if (args.box_iou_weight or args.pq_weight) and parsed.valid else None)
                reward, mean_iou = rollout_reward(parsed.valid, match, args.reward_beta, args.box_iou_weight,
                                                  pred_boxes, row.get("boxes"))
                if mean_iou is not None:
                    stats["box_iou"].append(mean_iou)
                if args.pq_weight:
                    pq = (rollout_pq(points, pred_boxes, row["mask"], segmenter.predict, args.sam_negatives)
                          if parsed.valid else 0.0)
                    stats["pq"].append(pq)
                    reward = (1 - args.pq_weight) * reward + args.pq_weight * pq
                rewards.append(reward)
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
            shaped = [fgas_token_advantages(a, len(ids), parsed.valid, match.pred_is_tp, args.fgas_decay, per_object)
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
                    kl_total += float(((kl.detach() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).sum()) / kept
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
                 **({"box_iou": float(np.mean(stats["box_iou"]))} if stats["box_iou"] else {}),
                 **({"pq": float(np.mean(stats["pq"]))} if stats["pq"] else {}),
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
