#!/bin/bash
# full-96 validation, merged eval-only path, for vla-27b896-trustmix256-sft-visionnone-seed44 (adapter merged into the base, greedy, cap 176).
#SBATCH --job-name=sseg-vl-ea44
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=10:00:00
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
OUT=$W/runs/vla-27b896-trustmix256-sft-visionnone-seed44-eval96
ARGS=(--objective grpo --eval-only --manifest "$D/train.jsonl" --eval-manifest "$D/validation.jsonl" --output "$OUT"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B" --init-adapter "$W/runs/vla-27b896-trustmix256-sft-visionnone-seed44/adapter"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cuda
  --coordinate-frame normalized1000 --min-pixels 802816 --max-pixels 802816
  --max-objects 176 --eval-max-completion-length 6144 --seed 42)
refuse_existing "$OUT"
exec "$PY" -m nucleus_rl.grpo "${ARGS[@]}"
