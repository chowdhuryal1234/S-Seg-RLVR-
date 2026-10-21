#!/bin/bash
#SBATCH --job-name=sseg-h2-27b-sft448
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/env.sh
cd $R/code
echo "host=$(hostname) start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec $R/.venv/bin/python -m nucleus_rl.sft --manifest $DATA/train.jsonl --eval-manifest $DATA/validation.jsonl \
  --output /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/h2-27b-d4-sft448 --model Qwen/Qwen3.6-27B --revision $REV_27B \
  --sam-checkpoint $SAM_CKPT --sam-device cpu --cohort-protocol $PROTOCOL \
  --coordinate-frame normalized1000 --augmentation d4 --seed 42 \
  --max-steps 512 --eval-every 128 --save-steps 128 --min-pixels 200704 --max-pixels 200704 \
  --wall-time-minutes 540
