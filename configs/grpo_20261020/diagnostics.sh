#!/bin/bash
# As run on 2026-10-20 (A100). In this repository diagnostics.py is scripts/grpo_diagnostics.py.
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/env.sh
cd $W/code
export PYTHONPATH=$W/code
$R/.venv/bin/python $W/diagnostics.py oracle --eval-manifest $DATA/validation.jsonl --output $W/runs/diag-oracle \
  --sam-tiny $SAM_CKPT --sam-large /n/home01/jzhou1125/rlvr-followup-20261015/checkpoints/sam2.1_hiera_large.pt --device cuda
$R/.venv/bin/python $W/diagnostics.py merge --eval-manifest $DATA/validation.jsonl --output $W/runs/diag-merge \
  --sam-tiny $SAM_CKPT --model Qwen/Qwen3.5-9B --revision $REV_9B \
  --adapter $R/runs/h200-53586843-2/9b/d4/adapter --eval-limit 8
