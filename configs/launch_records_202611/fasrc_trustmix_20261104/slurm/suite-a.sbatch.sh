#!/bin/bash
# Suite A (submit with slurm/submit_suite.sh a once S2 and G2 are verified):
#   S2 final: sparse 170 @ cap 8, then dense 192 @ cap 32      (process A)
#   G2 final: sparse 170 @ cap 8, then dense 192 @ cap 32      (process B)
# G2's --init-adapter is its own init (S2 checkpoint-256); the trainer checks init_adapter_sha256.
#SBATCH --job-name=sseg-tm-suite-a
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
S2=$W/runs/s2-27b448-trustmix-sft-seed42
G2=$W/runs/g2-27b448-trustmix-pq-from-s2c256
COMMON=(--objective grpo --eval-only --manifest "$TRAIN"
  --model Qwen/Qwen3.6-27B --revision "$REV_27B"
  --sam-checkpoint "$SAM_CKPT" --sam-config configs/sam2.1/sam2.1_hiera_t.yaml --eval-sam-device cpu
  --coordinate-frame normalized1000 --min-pixels 200704 --max-pixels 200704 --seed 42)
SPARSE=(--eval-manifest "$VAL_SUITE" --max-objects 8 --eval-max-completion-length 1024)
DENSE=(--eval-manifest "$VAL_DENSE" --max-objects 32 --eval-max-completion-length 2048)
ARGS_A1=("${COMMON[@]}" "${SPARSE[@]}" --init-adapter "$S2/adapter" --output "$W/runs/e-s2final-val170-cap8")
ARGS_A2=("${COMMON[@]}" "${DENSE[@]}" --init-adapter "$S2/adapter" --output "$W/runs/e-s2final-dense192-cap32")
ARGS_B1=("${COMMON[@]}" "${SPARSE[@]}" --init-adapter "$S2/checkpoints/checkpoint-256" --eval-adapter "$G2/adapter"
  --output "$W/runs/e-g2final-val170-cap8")
ARGS_B2=("${COMMON[@]}" "${DENSE[@]}" --init-adapter "$S2/checkpoints/checkpoint-256" --eval-adapter "$G2/adapter"
  --output "$W/runs/e-g2final-dense192-cap32")
# <<< lane
refuse_existing "$W/runs/e-s2final-val170-cap8" "$W/runs/e-s2final-dense192-cap32" \
  "$W/runs/e-g2final-val170-cap8" "$W/runs/e-g2final-dense192-cap32"
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
