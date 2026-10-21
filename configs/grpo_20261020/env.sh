export R=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-sft-20261017
export W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export DATA=$R/code/data/monuseg_official30_grpo_v1 SAM_CKPT=$R/checkpoints/sam2.1_hiera_tiny.pt PROTOCOL=$R/code/protocols/monuseg_official30.json
export REV_9B=c202236235762e1c871ad0ccb60c8ee5ba337b9a REV_27B=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
