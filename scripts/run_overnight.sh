#!/usr/bin/env bash
# Bounded pilot inside an existing allocation; does not request another GPU.
set -euo pipefail
TASK_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${RLVR_VENV:?Set the isolated GPU environment path}"
: "${RLVR_CACHE:?Set the task cache directory}"
export HF_HOME="$RLVR_CACHE/huggingface"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
cd "$TASK_WORKSPACE"
TASK_PYTHON="$RLVR_VENV/bin/python"
TASK_DEADLINE=$(( $(date +%s) + 5400 ))
while ! grep -q 'GPU imports and GRPO configuration check passed' runs/bootstrap.log; do
  if [[ -n "${RLVR_BOOTSTRAP_PID:-}" ]] && ! kill -0 "$RLVR_BOOTSTRAP_PID" 2>/dev/null; then
    echo 'Bootstrap ended before its GPU checks passed; inspect runs/bootstrap.log.' >&2
    exit 1
  fi
  if (( $(date +%s) >= TASK_DEADLINE )); then
    echo 'Dependency setup exceeded the 90-minute bound.' >&2
    exit 1
  fi
  sleep 20
done
test -s runs/policy_download.json
TASK_REVISION=$("$TASK_PYTHON" -c 'import json; print(json.load(open("runs/policy_download.json"))["revision"])')
TASK_DATA=data/monuseg_iris_pilot_v1
TASK_SAM=checkpoints/sam2.1_hiera_tiny.pt

"$TASK_PYTHON" scripts/prompt_diagnostic.py \
  --manifest "$TASK_DATA/train.jsonl" --sam-checkpoint "$TASK_SAM" \
  --output runs/oracle_iris_split_cuda --device cuda --max-images 4 --threads 8 \
  --include-full-image-boxes

# Keep this manifest beside the original so relative image paths remain valid.
head -n 4 "$TASK_DATA/train.jsonl" > "$TASK_DATA/rollout_train4.jsonl"
"$TASK_PYTHON" -m nucleus_rl.train --mode rollout \
  --manifest "$TASK_DATA/rollout_train4.jsonl" --sam-checkpoint "$TASK_SAM" \
  --revision "$TASK_REVISION" --output runs/base_policy_rollouts

"$TASK_PYTHON" -m nucleus_rl.train --mode train \
  --manifest "$TASK_DATA/train.jsonl" --eval-manifest "$TASK_DATA/validation.jsonl" \
  --sam-checkpoint "$TASK_SAM" --revision "$TASK_REVISION" \
  --output runs/grpo_smoke_10 --max-steps 10

# A longer independent run starts from the same base, not from Iris's decoder.
if "$TASK_PYTHON" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('runs/grpo_smoke_10/result.json').read_text())
scores=[json.loads(line) for line in Path('runs/grpo_smoke_10/training_completions/completions.jsonl').read_text().splitlines()]
passed=(r['status']=='verified_training_pilot' and r['adapter_changed']
        and r['nonzero_gradient_steps']>0 and r['sam']['unchanged']
        and any(x['seg_reward']>0 for x in scores))
raise SystemExit(0 if passed else 1)
PY
then
  "$TASK_PYTHON" -m nucleus_rl.train --mode train \
    --manifest "$TASK_DATA/train.jsonl" --eval-manifest "$TASK_DATA/validation.jsonl" \
    --sam-checkpoint "$TASK_SAM" --revision "$TASK_REVISION" \
    --output runs/grpo_pilot_50 --max-steps 50
else
  echo 'No useful segmentation-policy update established; retaining the diagnostic run.'
fi
