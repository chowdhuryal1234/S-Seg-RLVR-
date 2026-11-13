#!/bin/bash
# Run locally after `ssh fasrc` (Duo) is restored. Checks the canary first; submits the SFT main job exactly once.
set -euo pipefail
W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105
ssh -o BatchMode=yes fasrc "sacct -n -X -j 58185622 -o JobID,State,Elapsed; tail -n 4 $W/logs/sseg-nx-tiles256-canary-58185622.out; ls $W/runs/canary-tiles256-sft896-3steps $W/runs/canary-tiles256-e-s2c256-448-val4 2>&1 | head -30"
read -r -p "Canary OK (memory passed already; check eval completions)? submit tiles256-sft? [y/N] " ok
[ "$ok" = y ] || exit 0
scp "$(dirname "$0")/tiles256-sft.sbatch" fasrc:$W/slurm/tiles256-sft.sbatch
ssh -o BatchMode=yes fasrc "cd $W && bash -n slurm/tiles256-sft.sbatch && sbatch --test-only slurm/tiles256-sft.sbatch && slurm/submit.sh tiles256-sft && cat receipts/tiles256-sft.submitted.json"
