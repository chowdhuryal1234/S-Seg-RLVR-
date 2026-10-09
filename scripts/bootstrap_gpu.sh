#!/usr/bin/env bash
# Run inside an existing GPU allocation. Never submits a new Slurm job.
set -euo pipefail
TASK_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${RLVR_VENV:?Set RLVR_VENV to an isolated Python 3.11 environment}"
: "${RLVR_CACHE:?Set RLVR_CACHE to a writable task cache directory}"
mkdir -p "$RLVR_CACHE/pip" "$RLVR_CACHE/huggingface"
export PIP_CACHE_DIR="$RLVR_CACHE/pip"
export HF_HOME="$RLVR_CACHE/huggingface"
export SAM2_BUILD_CUDA=0
cd "$TASK_WORKSPACE"
"$RLVR_VENV/bin/python" -m pip install 'wheel==0.45.1'
"$RLVR_VENV/bin/python" -m pip install 'torch==2.5.1' 'torchvision==0.20.1'
"$RLVR_VENV/bin/python" -m pip install --no-build-isolation -r requirements-gpu.txt
"$RLVR_VENV/bin/python" -m pip check
"$RLVR_VENV/bin/python" -m pip freeze > requirements-gpu-lock.txt
"$RLVR_VENV/bin/python" -m unittest discover -s tests -v
"$RLVR_VENV/bin/python" - <<'PY'
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import Qwen2_5_VLForConditionalGeneration
from sam2.build_sam import build_sam2
assert torch.cuda.is_available(), 'CUDA is unavailable in this job'
assert torch.cuda.is_bf16_supported(), 'GPU does not support BF16'
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
       'vram_bytes': torch.cuda.get_device_properties(0).total_memory})
GRPOConfig(output_dir='/tmp/rlvr-config-only', bf16=True,
           per_device_train_batch_size=1, gradient_accumulation_steps=2,
           generation_batch_size=2, num_generations=2, use_vllm=False,
           report_to='none')
print('GPU imports and GRPO configuration check passed; no training claimed.')
PY
