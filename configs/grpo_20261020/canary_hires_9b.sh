#!/bin/bash
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/env.sh
cd $R/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec $R/.venv/bin/python -m nucleus_rl.sft --manifest $DATA/train.jsonl --eval-manifest $DATA/validation.jsonl \
  --output $W/runs/canary-hires-9b --model Qwen/Qwen3.5-9B --revision $REV_9B \
  --sam-checkpoint $SAM_CKPT --sam-device cpu --cohort-protocol $PROTOCOL \
  --coordinate-frame normalized1000 --augmentation d4 --seed 42 \
  --max-steps 20 --eval-every 20 --save-steps 20 --min-pixels 200704 --max-pixels 200704 \
  --train-limit 4 --eval-limit 8
