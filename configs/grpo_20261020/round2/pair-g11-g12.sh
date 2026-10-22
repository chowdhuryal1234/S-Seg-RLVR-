#!/bin/bash
#SBATCH --job-name=sseg-pair-g11-g12
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/logs/%x-%j.out
# Two 9B GRPO runs share one H200 (about 60 GB each); each writes its own log.
set -uo pipefail
echo "host=$(hostname) start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GRPO_THREADS=2 bash /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/slurm/g11-9b448final-pqcount.sbatch > /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/logs/g11-9b448final-pqcount.log 2>&1 &
GRPO_THREADS=2 bash /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/slurm/g12-9b448c128-pqcount-t13.sbatch > /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/logs/g12-9b448c128-pqcount-t13.log 2>&1 &
wait
echo "both finished $(date -Is)"
