#!/bin/bash
# S4: SFT on trustmix_v2/train.jsonl (1146 tiles), H2 recipe, seed 43 (seed replicate of S2).
# In-run eval on MoNuSeg val only: step 256 (eval-every 256) and the final "after" pass at 512.
# Starts a watcher that submits G4 once checkpoint-256 is verified.
#SBATCH --job-name=sseg-tm-s4
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/env.sh
source "$W/slurm/common.sh"
TAG=s4
# >>> lane (assignments only; tools/preflight.py evaluates this block)
NAME=s4-27b448-trustmix-sft-seed43
OUT=$W/runs/$NAME
MODULE=nucleus_rl.sft
ARGS=(--manifest "$TRAIN" --eval-manifest "$VAL_MONUSEG" --output "$OUT"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --sam-device cpu
  --coordinate-frame normalized1000 --augmentation d4 --d4-schedule rotating --seed 43
  --max-steps 512 --eval-every 256 --save-steps 128 --gradient-accumulation 4
  --learning-rate 1e-4 --warmup-steps 16 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --max-sequence-length 2048 --max-completion-length 1024 --max-objects 8
  --min-pixels 200704 --max-pixels 200704 --fixed-loss-examples 4
  --skip-before-eval --wall-time-minutes 270)
# <<< lane
refuse_existing "$OUT"
cd "$W/code"
lane_header
WATCH_LOG=$W/logs/watch-$TAG-${SLURM_JOB_ID:-0}.log
"$PY" -m $MODULE "${ARGS[@]}" &
TRAINER=$!
watch_and_submit "$OUT" "$TRAINER" "$TAG" >> "$WATCH_LOG" 2>&1 &
WATCHER=$!
rc=0
wait "$TRAINER" || rc=$?
echo "trainer exit=$rc at $(date +%Y-%m-%dT%H:%M:%S%z)"
for _ in $(seq 1 30); do kill -0 "$WATCHER" 2>/dev/null || break; sleep 10; done
kill "$WATCHER" 2>/dev/null || true
if wait "$WATCHER" 2>/dev/null; then echo "watcher ok"; else echo "watcher rc=$? (see $WATCH_LOG)"; fi
exit $rc
