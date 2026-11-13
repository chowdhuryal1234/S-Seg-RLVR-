#!/bin/bash
# G4: GRPO, G7 recipe (format=1,pq=1; G=8 x 2 prompts; T=1; beta 0.04; lr 2e-5; 512 tokens; cap 8),
# 300 steps, eval every 150 on MoNuSeg val, save every 100. Init = S4 checkpoint-256, seed 43 (seed replicate of G2). Prompts = trustmix_v2 train.
# Submitted by the S4 watcher (or slurm/submit_children.sh s4 on the login node).
#SBATCH --job-name=sseg-tm-g4
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
NAME=g4-27b448-trustmix-pq-from-s4c256-seed43
OUT=$W/runs/$NAME
INIT=$W/runs/s4-27b448-trustmix-sft-seed43/checkpoints/checkpoint-256
MODULE=nucleus_rl.grpo
ARGS=(--objective grpo --manifest "$TRAIN" --eval-manifest "$VAL_MONUSEG" --output "$OUT"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B" --init-adapter "$INIT"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml
  --eval-sam-device cpu --reward-sam-device cuda
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 --augmentation d4
  --reward format=1,pq=1 --group-size 8 --prompts-per-step 2 --temperature 1.0 --beta 0.04
  --learning-rate 2e-5 --warmup-steps 10 --lora-rank 16 --lora-alpha 32
  --max-steps 300 --eval-every 150 --save-steps 100
  --max-completion-length 512 --eval-max-completion-length 1024 --max-sequence-length 2048 --max-objects 8
  --seed 43 --wall-time-minutes 540)
# <<< lane
refuse_existing "$OUT"
cd "$W/code"
lane_header
verify_gate_checkpoint "$INIT"
exec "$PY" -m $MODULE "${ARGS[@]}"
