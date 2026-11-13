#!/bin/bash
# vision-LoRA canary: arm B (--vision-lora all), 3 steps on canary_train (4 densest + 8 tiles), eval on 2 quick tiles. Checks memory, step time, vision gradients, base-hash audit, save and generate.
#SBATCH --job-name=sseg-vl-canary
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/env.sh
source $T/slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CODE=$W/code-vislora
D=$W/data-tiles256/trustmix256_v1
cd "$CODE"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date +%FT%T%z) commit=$(cat $CODE/COMMIT)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
ARGS=(--manifest "$D/canary_train.jsonl" --eval-manifest "$D/validation_quick.jsonl" --output "$W/runs/canary-vlb-27b896-3steps"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --sam-device cuda
  --coordinate-frame normalized1000 --augmentation d4 --d4-schedule rotating --seed 42
  --max-steps 3 --eval-every 0 --save-steps 3 --gradient-accumulation 4
  --learning-rate 1e-4 --warmup-steps 1 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --vision-lora all
  --max-sequence-length 12288 --max-completion-length 6144 --max-objects 176
  --min-pixels 802816 --max-pixels 802816 --fixed-loss-examples 4
  --skip-before-eval --wall-time-minutes 60 --eval-limit 2)
refuse_existing "$W/runs/canary-vlb-27b896-3steps"
exec "$PY" -m nucleus_rl.sft "${ARGS[@]}"
