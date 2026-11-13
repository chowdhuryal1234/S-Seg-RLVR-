#!/bin/bash
# tiles256-s2ref: LOWER-BOUND reference only. Existing S2@256 adapter (trained on 128 px crops, prompt 'at most 8 objects')
# evaluated at 448 px on the 48 validation_quick 256 px tiles with --max-objects 176: each nucleus gets half the pixels S2
# was trained on, so its recall is limited by design. A fair baseline is S2 on the four 128 px quadrants stitched back.
#SBATCH --job-name=sseg-nx-tiles256-s2ref
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/env.sh
source $T/slurm/common.sh
CODE=$W/code-tiles256
D=$W/data-tiles256/trustmix256_v1
S2CKPT=$T/runs/s2-27b448-trustmix-sft-seed42/checkpoints/checkpoint-256
OUT_B=$W/runs/e-s2c256-448-tiles256quick48-cap176
ARGS_B=(--objective grpo --eval-only --manifest "$D/train.jsonl" --eval-manifest "$D/validation_quick.jsonl" --output "$OUT_B"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B" --init-adapter "$S2CKPT"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cuda
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704
  --max-objects 176 --eval-max-completion-length 6144 --seed 42)
refuse_existing "$OUT_B"
cd "$CODE"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date +%FT%T%z) lane_commit=$(cat $CODE/COMMIT)"
exec "$PY" -m nucleus_rl.grpo "${ARGS_B[@]}"
