#!/bin/bash
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/env.sh
cd $W/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec $R/.venv/bin/python -m nucleus_rl.grpo --objective grpo \
  --manifest $DATA/train.jsonl --eval-manifest $DATA/validation.jsonl \
  --output $W/runs/canary-grpo-9b --model Qwen/Qwen3.5-9B --revision $REV_9B \
  --init-adapter $R/runs/h200-53586843-2/9b/d4/adapter \
  --sam-checkpoint $SAM_CKPT --cohort-protocol $PROTOCOL \
  --coordinate-frame normalized1000 --min-pixels 50176 --max-pixels 200704 \
  --augmentation d4 --reward format=1,pq=1 --group-size 8 --prompts-per-step 2 \
  --max-steps 10 --eval-every 0 --save-steps 10 --warmup-steps 2 --train-limit 8 --eval-limit 8
