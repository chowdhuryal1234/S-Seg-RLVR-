#!/bin/bash
#SBATCH --job-name=sseg-c1-9b-sftcont
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
cd /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/code
echo "host=$(hostname) start=$(date -Is) commit=$(cat SOURCE_COMMIT)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec $R/.venv/bin/python -m nucleus_rl.grpo --objective sft \
  --manifest $DATA/train.jsonl --eval-manifest $DATA/validation.jsonl \
  --output /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/c1-9b-sftcont --model Qwen/Qwen3.5-9B --revision $REV_9B --init-adapter /n/holylabs/mooney_lab/Lab/junyizhou/sseg-sft-20261017/runs/h200-53586843-2/9b/d4/adapter \
  --sam-checkpoint $SAM_CKPT --cohort-protocol $PROTOCOL \
  --coordinate-frame normalized1000 --min-pixels 50176 --max-pixels 200704 \
  --augmentation d4 --reward format=1,pq=1 --group-size 8 --prompts-per-step 2 --temperature 1.0 \
  --learning-rate 1e-4 --beta 0.04 --max-steps 800 --eval-every 200 --save-steps 50 --wall-time-minutes 570
