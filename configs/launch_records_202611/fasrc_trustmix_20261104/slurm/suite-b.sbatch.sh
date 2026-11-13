#!/bin/bash
# Suite B (submit with slurm/submit_suite.sh b once S4, G4, G5, G6 are verified): sparse 170 @ cap 8 only.
#   process A: S4 final, then G4 final      process B: G5 final, then G6 final
# Each GRPO adapter is evaluated on top of its own merged init (S2 or S4 checkpoint-256).
#SBATCH --job-name=sseg-tm-suite-b
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/logs/%x-%j.out
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/env.sh
source "$W/slurm/common.sh"
# >>> lane (assignments only; tools/preflight.py evaluates this block)
MODULE=nucleus_rl.grpo
S2=$W/runs/s2-27b448-trustmix-sft-seed42
S4=$W/runs/s4-27b448-trustmix-sft-seed43
G4=$W/runs/g4-27b448-trustmix-pq-from-s4c256-seed43
G5=$W/runs/g5-27b448-trustmix-wpq-fp03-fn07-from-s2c256
G6=$W/runs/g6-27b448-trustmix-pq-g16x1-from-s2c256
COMMON=(--objective grpo --eval-only --manifest "$TRAIN"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cpu
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 --seed 42
  --eval-manifest "$VAL_SUITE" --max-objects 8 --eval-max-completion-length 1024)
ARGS_A1=("${COMMON[@]}" --init-adapter "$S4/adapter" --output "$W/runs/e-s4final-val170-cap8")
ARGS_A2=("${COMMON[@]}" --init-adapter "$S4/checkpoints/checkpoint-256" --eval-adapter "$G4/adapter"
  --output "$W/runs/e-g4final-val170-cap8")
ARGS_B1=("${COMMON[@]}" --init-adapter "$S2/checkpoints/checkpoint-256" --eval-adapter "$G5/adapter"
  --output "$W/runs/e-g5final-val170-cap8")
ARGS_B2=("${COMMON[@]}" --init-adapter "$S2/checkpoints/checkpoint-256" --eval-adapter "$G6/adapter"
  --output "$W/runs/e-g6final-val170-cap8")
# <<< lane
refuse_existing "$W/runs/e-s4final-val170-cap8" "$W/runs/e-g4final-val170-cap8" \
  "$W/runs/e-g5final-val170-cap8" "$W/runs/e-g6final-val170-cap8"
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
