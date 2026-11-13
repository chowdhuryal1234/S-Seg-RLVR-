#!/bin/bash
# R0: transfer baseline. grpo.py --eval-only for H2 final and G7 final on
#   trustmix_v2/validation.jsonl       (170 tiles: MoNuSeg 64 + CoNSeP 42 + PUMA 64) at cap 8, 1024 tokens
#   trustmix_v2/validation_dense.jsonl (192 tiles) at cap 32, 2048 tokens
# Two 27B processes share one H200 (~55-60 GB each); each runs its two evals one after the other.
# Adapter logic (grpo.evaluate_only): --init-adapter is merged into the base first.
#   SFT adapter (H2 final): --init-adapter H2/adapter, no --eval-adapter.
#   GRPO adapter (G7 final): --init-adapter = G7's own init (H2 checkpoint-128), --eval-adapter G7/adapter;
#   the trainer refuses a mismatch with coordinate_config.json's init_adapter_sha256.
#SBATCH --job-name=sseg-tm-r0
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/env.sh
source "$W/slurm/common.sh"
# >>> lane (assignments only; tools/preflight.py evaluates this block)
MODULE=nucleus_rl.grpo
COMMON=(--objective grpo --eval-only --manifest "$TRAIN"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cpu
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 --seed 42)
SPARSE=(--eval-manifest "$VAL_SUITE" --max-objects 8 --eval-max-completion-length 1024)
DENSE=(--eval-manifest "$VAL_DENSE" --max-objects 32 --eval-max-completion-length 2048)
H2_FINAL=(--init-adapter "$H2_RUN/adapter")
G7_FINAL=(--init-adapter "$H2_RUN/checkpoints/checkpoint-128" --eval-adapter "$G7_RUN/adapter")
ARGS_A1=("${COMMON[@]}" "${SPARSE[@]}" "${H2_FINAL[@]}" --output "$W/runs/r0-h2final-val170-cap8")
ARGS_A2=("${COMMON[@]}" "${DENSE[@]}" "${H2_FINAL[@]}" --output "$W/runs/r0-h2final-dense192-cap32")
ARGS_B1=("${COMMON[@]}" "${SPARSE[@]}" "${G7_FINAL[@]}" --output "$W/runs/r0-g7final-val170-cap8")
ARGS_B2=("${COMMON[@]}" "${DENSE[@]}" "${G7_FINAL[@]}" --output "$W/runs/r0-g7final-dense192-cap32")
# <<< lane
refuse_existing "$W/runs/r0-h2final-val170-cap8" "$W/runs/r0-h2final-dense192-cap32" \
  "$W/runs/r0-g7final-val170-cap8" "$W/runs/r0-g7final-dense192-cap32"
cd "$W/code"
lane_header
run_chain A ARGS_A1 ARGS_A2 &
PA=$!
run_chain B ARGS_B1 ARGS_B2 &
PB=$!
rc=0
wait "$PA" || rc=1
wait "$PB" || rc=1
echo "end=$(date +%Y-%m-%dT%H:%M:%S%z) rc=$rc"
exit $rc
