#!/bin/bash
#SBATCH --job-name=sseg-g9-27b448c128-pqcount
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/env.sh
export OMP_NUM_THREADS=${GRPO_THREADS:-${SLURM_CPUS_PER_TASK:-2}}
cd /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/code
echo "host=$(hostname) start=$(date -Is) commit=$(cat SOURCE_COMMIT)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/h2-27b-d4-sft448/checkpoints/checkpoint-128/adapter_model.safetensors
exec $R/.venv/bin/python -m nucleus_rl.grpo --objective grpo \
  --manifest $DATA/train.jsonl --eval-manifest $DATA/validation.jsonl \
  --output /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/g9-27b448c128-pqcount --model Qwen/Qwen3.6-27B --revision $REV_27B --init-adapter /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/h2-27b-d4-sft448/checkpoints/checkpoint-128 \
  --sam-checkpoint $SAM_CKPT --cohort-protocol $PROTOCOL \
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 \
  --augmentation d4 --reward format=1,pq=1,count=1 --group-size 8 --prompts-per-step 2 --temperature 1.0 \
  --learning-rate 2e-5 --beta 0.04 --max-steps 800 --eval-every 150 --save-steps 50 --wall-time-minutes 600
