#!/usr/bin/env bash
# Execute inside the user's existing A100 allocation; never requests a GPU.
set -euo pipefail
: "${SFT_ROOT:?Set SFT_ROOT to the staged experiment workspace}"
: "${SFT_DEADLINE_EPOCH:?Set the existing allocation's actual end timestamp}"
export HF_HOME="$SFT_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$SFT_ROOT/code"
for TASK_MODEL in 7b 9b 32b 27b; do
  # Download receipts are written only after complete immutable snapshots exist.
  while [[ ! -f "$SFT_ROOT/environment-ready" || ! -s "$SFT_ROOT/model-$TASK_MODEL.json" ]]; do
    if (( $(date +%s) >= SFT_DEADLINE_EPOCH - 7800 )); then
      echo 'Environment/download preparation exhausted the canary time budget.' >&2
      exit 1
    fi
    sleep 20
  done
  "$SFT_ROOT/.venv/bin/python" scripts/run_sft_experiments.py \
    --stage canary --model-key "$TASK_MODEL" --code "$SFT_ROOT/code" \
    --root "$SFT_ROOT/runs/a100-canaries" \
    --python "$SFT_ROOT/.venv/bin/python" \
    --sam-checkpoint "$SFT_ROOT/checkpoints/sam2.1_hiera_tiny.pt" \
    --deadline-epoch "$SFT_DEADLINE_EPOCH"
done
date -Is > "$SFT_ROOT/all-a100-canaries-passed"
