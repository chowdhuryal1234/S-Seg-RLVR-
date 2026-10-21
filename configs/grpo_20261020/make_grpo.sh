#!/bin/bash
# usage: make_grpo.sh ID OBJECTIVE MODEL REVVAR ADAPTER REWARD LR STEPS EVERY WALL PIXELS
set -euo pipefail
W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020
[ $# -eq 11 ] || { echo "need 11 args" >&2; exit 2; }
ID=$1; OUT=$W/runs/$ID; S=$W/slurm/$ID.sbatch
[ ! -e "$OUT" ] || { echo "output $OUT already exists" >&2; exit 3; }
[ ! -e "$S" ] || { echo "script $S already exists" >&2; exit 3; }
cat > $S <<SH
#!/bin/bash
#SBATCH --job-name=sseg-$ID
#SBATCH --account=mooney_lab
#SBATCH --partition=gpu_h200,seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=$W/logs/%x-%j.out
set -euo pipefail
source $W/env.sh
export OMP_NUM_THREADS=\${GRPO_THREADS:-\${SLURM_CPUS_PER_TASK:-2}}
cd $W/code
echo "host=\$(hostname) start=\$(date -Is) commit=\$(cat SOURCE_COMMIT)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f $5/adapter_model.safetensors
exec \$R/.venv/bin/python -m nucleus_rl.grpo --objective $2 \\
  --manifest \$DATA/train.jsonl --eval-manifest \$DATA/validation.jsonl \\
  --output $OUT --model $3 --revision \$$4 --init-adapter $5 \\
  --sam-checkpoint \$SAM_CKPT --cohort-protocol \$PROTOCOL \\
  --coordinate-frame normalized1000 --min-pixels ${11} --max-pixels ${11} \\
  --augmentation d4 --reward $6 --group-size 8 --prompts-per-step 2 --temperature 1.0 \\
  --learning-rate $7 --beta 0.04 --max-steps $8 --eval-every $9 --save-steps 50 --wall-time-minutes ${10}
SH
bash -n $S
echo "$S"; grep -o -E -- "--output [^ ]+|--init-adapter [^ ]+|--reward [^ ]+|--learning-rate [^ ]+|--min-pixels [0-9]+" $S | tr "\n" " "; echo
