#!/bin/bash
# G6: as G2 but G=16 x 1 prompt (same 16 samples per step; G7 recipe otherwise: format=1,pq=1; T=1; beta 0.04; lr 2e-5; 512 tokens; cap 8),
# 300 steps, eval every 150 on MoNuSeg val, save every 100. Init = S2 checkpoint-256. Prompts = trustmix_v2 train.
# Submitted by the S2 watcher (or slurm/submit_children.sh s2 on the login node).
#SBATCH --job-name=sseg-tm-g6
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/env.sh
source "$W/slurm/common.sh"
# >>> lane (assignments only; tools/preflight.py evaluates this block)
NAME=g6-27b448-trustmix-pq-g16x1-from-s2c256
OUT=$W/runs/$NAME
INIT=$W/runs/s2-27b448-trustmix-sft-seed42/checkpoints/checkpoint-256
MODULE=nucleus_rl.grpo
ARGS=(--objective grpo --manifest "$TRAIN" --eval-manifest "$VAL_MONUSEG" --output "$OUT"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B" --init-adapter "$INIT"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml
  --eval-sam-device cpu --reward-sam-device cuda
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 --augmentation d4
  --reward format=1,pq=1 --group-size 16 --prompts-per-step 1 --temperature 1.0 --beta 0.04
  --learning-rate 2e-5 --warmup-steps 10 --lora-rank 16 --lora-alpha 32
  --max-steps 300 --eval-every 150 --save-steps 100
  --max-completion-length 512 --eval-max-completion-length 1024 --max-sequence-length 2048 --max-objects 8
  --seed 42 --wall-time-minutes 540)
# <<< lane
refuse_existing "$OUT"
cd "$W/code"
lane_header
verify_gate_checkpoint "$INIT"
exec "$PY" -m $MODULE "${ARGS[@]}"
