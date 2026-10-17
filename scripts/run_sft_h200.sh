#!/usr/bin/env bash
# Submit this four-task array only after all A100 canaries have passed.
# Each task runs two augmentation arms sequentially on one H200.
#SBATCH --job-name=sseg-sft
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
set -euo pipefail
: "${SFT_ROOT:?Set SFT_ROOT to the staged immutable experiment workspace}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a four-task Slurm array}"
TASK_MODELS=(7b 32b 9b 27b)
TASK_MODEL=${TASK_MODELS[$SLURM_ARRAY_TASK_ID]}
export HF_HOME="$SFT_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$SFT_ROOT/code"
# Leave a final hour outside the supervisor's budget for scheduler teardown.
TASK_DEADLINE=$(( $(date +%s) + 23 * 3600 ))
exec "$SFT_ROOT/.venv/bin/python" scripts/run_sft_experiments.py \
  --stage main --model-key "$TASK_MODEL" --code "$SFT_ROOT/code" \
  --root "$SFT_ROOT/runs/h200-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID" \
  --python "$SFT_ROOT/.venv/bin/python" \
  --sam-checkpoint "$SFT_ROOT/checkpoints/sam2.1_hiera_tiny.pt" \
  --canary-root "$SFT_ROOT/runs/a100-canaries" \
  --deadline-epoch "$TASK_DEADLINE"
