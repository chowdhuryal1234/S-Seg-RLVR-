# Environment for the trustmix lanes (2026-11-04).
# Copied from sseg-grpo-20261020/env.sh (the env of H2 and G7). Differences:
#   - W points at the new workspace; OLD names the 10/20 workspace (read only: H2/G7 adapters).
#   - OMP_NUM_THREADS follows the CPUs Slurm grants (make_grpo.sh did the same after right-sizing).
#   - DATA is the synced copy in $W/data. No PROTOCOL: --cohort-protocol is NOT used on mixed data.
#   - PY names the SFT venv python that H2 and G7 ran with ($R/.venv/bin/python).
# The old env set no modules, no PYTHONPATH and no PYTORCH_CUDA_ALLOC_CONF; neither does this one.
export R=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-sft-20261017
export OLD=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020
export W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104
export PY=$R/.venv/bin/python
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export SAM_CKPT=$R/checkpoints/sam2.1_hiera_tiny.pt
export REV_27B=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
export DATA=$W/data
export TRAIN=$DATA/trustmix_v2/train.jsonl
export VAL_MONUSEG=$DATA/monuseg_official30_grpo_v1/validation.jsonl
export VAL_SUITE=$DATA/trustmix_v2/validation.jsonl
export VAL_DENSE=$DATA/trustmix_v2/validation_dense.jsonl
export H2_RUN=$OLD/runs/h2-27b-d4-sft448
export G7_RUN=$OLD/runs/g7-27b448c128-pq
