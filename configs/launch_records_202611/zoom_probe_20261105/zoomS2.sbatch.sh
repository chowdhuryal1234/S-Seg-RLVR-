#!/bin/bash
# zoomS2: 2x2 zoom inference probe (ZERO training), adapter (a) S2@256 trustmix SFT (checkpoint-256 merged, no eval adapter).
# Question: are the "truly unseen" nuclei (3.6-5.7% of GT; pale/vesicular) a resolution problem?
# Per 128x128 validation tile: 4 sub-crops 80x80 (offsets 0/48, 32 px overlap), greedy policy on each with the normal
# eval prompt, normalized1000 frame, min=max pixels 200704 (each sub-crop gets the full 448x448 budget), map to tile px,
# merge across sub-crops (box IoU>0.5 or point-in-other-box -> keep the copy farther from its interior sub-crop border),
# cap 32, frozen SAM2.1-tiny ONCE on the full tile, scored with ArtifactScorer (= sft.evaluate_policy's scorer).
# Control in the SAME job/GPU/code: plain eval of the same policy via grpo.evaluate -> sft.evaluate_policy.
# Canary (first 4 tiles, plain+zoom) runs first in-process and gates the full 170-tile run.
#
# PRE-REGISTERED READOUT (written 2026-11-05 before any result):
#   coverage of GT (box IoU>0.5 by any prompt); TP/FP/FN (mask IoU>0.5) and mean per-tile PQ, pooled + per dataset,
#   zoom vs plain; and on the GT nuclei the PLAIN eval had NO box near (best box IoU<0.1): FN plain vs zoom.
#   Zoom PASSES iff no-box FN drops >= 30% relative AND pooled TP +15 or more AND FP_zoom < 2 x FP_plain.
#   Scale confound: sub-crops are seen at ~1.6x the usual magnification (80->448 vs 128->448 px), never trained on.
#   Output: $OUT/results.json and $OUT/RESULTS.md (also canary/). Validation suite only; test sets never read.
#SBATCH --job-name=sseg-nx-zoom-S2
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
CODE=$W/code-zoom
OUT=$W/runs/zoom-s2c256-val170
INIT=$T/runs/s2-27b448-trustmix-sft-seed42/checkpoints/checkpoint-256
refuse_existing "$OUT"
cd "$CODE"
echo "job=${SLURM_JOB_ID:-none} name=${SLURM_JOB_NAME:-none} host=$(hostname) start=$(date +%Y-%m-%dT%H:%M:%S%z) lane_commit=$(cat "$CODE/COMMIT")"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
TM_GATE_STEP=256 verify_gate_checkpoint "$INIT"
"$PY" -m scripts.zoom_eval --manifest "$TRAIN" --eval-manifest "$VAL_SUITE" --output "$OUT" \
  --model Qwen/Qwen3.6-27B --revision "$REV_27B" --init-adapter "$INIT" \
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cpu \
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 \
  --max-objects 8 --merged-max-objects 32 --eval-max-completion-length 1024 --sub-size 80 --seed 42 --canary 4
echo "end=$(date +%Y-%m-%dT%H:%M:%S%z)"
