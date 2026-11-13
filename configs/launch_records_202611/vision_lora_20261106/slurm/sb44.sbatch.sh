#!/bin/bash
# sampled support (8 samples/tile, T=1, box IoU>0.5 matching, no SAM) on the 96 val tiles for vlb-27b896-trustmix256-sft-visionall-seed44
#SBATCH --job-name=sseg-vl-sb44
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
OUT=$W/runs/vlb-27b896-trustmix256-sft-visionall-seed44-support96
ARGS=(--eval-manifest "$D/validation.jsonl" --output "$OUT" --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --init-adapter "$W/runs/vlb-27b896-trustmix256-sft-visionall-seed44/adapter" --coordinate-frame normalized1000 --min-pixels 802816 --max-pixels 802816
  --max-objects 176 --samples 8 --temperature 1.0 --max-new-tokens 6144 --seed 42 --wall-time-minutes 420)
refuse_existing "$OUT"
exec "$PY" scripts/support_eval.py "${ARGS[@]}"
