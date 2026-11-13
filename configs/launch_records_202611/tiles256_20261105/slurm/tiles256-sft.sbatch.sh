#!/bin/bash
# tiles256 lane SFT screen (written 2026-11-05; submitted 2026-11-06 by coordinator after canary 58185622 was verified). Code: $W/code-tiles256 (claude/tiles256-20261105 @ 744f408).
# S2 recipe (27B, init = base model, rank-16 language LoRA, lr 1e-4, warmup 16, GA 4, D4 rotating, normalized1000, seed 42,
# 512 steps = 2048 exposures) on trustmix256_v1/train.jsonl (689 tiles: MoNuSeg 189 + CoNSeP 192 + PUMA 308, all nuclei).
# Changes vs S2 (all forced by the 256 px protocol): input 896 px (min/max pixels 802816, same source px per visual token
# as 128->448), --max-objects 176 (densest train tile 168), --max-sequence-length 12288 (longest D4 target 10617 tokens
# in the canary), eval on validation_quick.jsonl (48 tiles, 1719 GT), SAM on cuda for eval, eval-every 0.
# Canary 58185622 (3 steps on the 4 densest + 8 tiles): 55.8/62.6/94.7 s per step, peak 96.5 GB allocated / 120.3 GB
# reserved. Predicted mean step on the real mix (mean 2841 tokens/example): ~28 s -> 512 steps ~4.0 h.
# Safety: --wall-time-minutes 330 stops training (and still saves + evaluates) if steps run slower than predicted.
# Coordinator change: eval completion cap 6144 (val max 98 nuclei x ~57 tokens = 5.6k) so a looping tile costs less;
# the S2@256 reference moved to its own job (tiles256-s2ref).
#SBATCH --job-name=sseg-nx-tiles256-sft
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/env.sh
source $T/slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # canary reserved 120 GB vs 96.5 GB allocated
CODE=$W/code-tiles256
D=$W/data-tiles256/trustmix256_v1
S2CKPT=$T/runs/s2-27b448-trustmix-sft-seed42/checkpoints/checkpoint-256
OUT_A=$W/runs/t256s1-27b896-trustmix256-sft-seed42
ARGS_A=(--manifest "$D/train.jsonl" --eval-manifest "$D/validation_quick.jsonl" --output "$OUT_A"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --sam-device cuda
  --coordinate-frame normalized1000 --augmentation d4 --d4-schedule rotating --seed 42
  --max-steps 512 --eval-every 0 --save-steps 128 --gradient-accumulation 4
  --learning-rate 1e-4 --warmup-steps 16 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --max-sequence-length 12288 --max-completion-length 6144 --max-objects 176
  --min-pixels 802816 --max-pixels 802816 --fixed-loss-examples 4
  --skip-before-eval --wall-time-minutes 330)
refuse_existing "$OUT_A"
cd "$CODE"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date +%FT%T%z) lane_commit=$(cat $CODE/COMMIT)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
rc=0
echo "[A] $(date +%FT%T%z) start"
"$PY" -m nucleus_rl.sft "${ARGS_A[@]}" || rc=1
echo "[A] $(date +%FT%T%z) end rc=$rc"
exit $rc
