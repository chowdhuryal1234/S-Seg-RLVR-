#!/bin/bash
# vision-LoRA study arm B (--vision-lora all), seed 44. Recipe identical to t256s1 (tiles256-sft.sbatch: 27B, rank-16 LoRA, lr 1e-4, 512 steps, 896 px, cap 176) except seed/vision-lora/code.
#SBATCH --job-name=sseg-vl-bs44
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CODE=$W/code-vislora
D=$W/data-tiles256/trustmix256_v1
cd "$CODE"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date +%FT%T%z) commit=$(cat $CODE/COMMIT)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
ARGS=(--manifest "$D/train.jsonl" --eval-manifest "$D/validation_quick.jsonl" --output "$W/runs/vlb-27b896-trustmix256-sft-visionall-seed44"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --sam-device cuda
  --coordinate-frame normalized1000 --augmentation d4 --d4-schedule rotating --seed 44
  --max-steps 512 --eval-every 0 --save-steps 128 --gradient-accumulation 4
  --learning-rate 1e-4 --warmup-steps 16 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --vision-lora all
  --max-sequence-length 12288 --max-completion-length 6144 --max-objects 176
  --min-pixels 802816 --max-pixels 802816 --fixed-loss-examples 4
  --skip-before-eval --wall-time-minutes 420)
refuse_existing "$W/runs/vlb-27b896-trustmix256-sft-visionall-seed44"
exec "$PY" -m nucleus_rl.sft "${ARGS[@]}"
