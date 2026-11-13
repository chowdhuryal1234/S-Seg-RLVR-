#!/bin/bash
# Vision-LoRA study: stage code + sbatch on FASRC, run the login-node module check, then submit exactly once.
#   ./submit.sh stage    -> copy code (refuses to overwrite a different commit) and sbatch files; inspect modules; quota test
#   ./submit.sh submit   -> canary + arms (A s43/s44, B s42/s43/s44); per arm eval-96 (afterok on the SFT job), then
#                           support-96 (afterok on eval-96). ea42/sa42 for the existing t256s1 = A seed 42 start now.
#                           Peak concurrency: 8 GPUs (canary, ea42, sa42, 5 arms).
# Receipts: $W/receipts/vl-<name>.submitted.json; an existing receipt means "already submitted", never resubmitted.
set -euo pipefail
W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105
WT=$HOME/Desktop/RLVR/vision-lora-worktree-20261106
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=$(git -C "$WT" rev-parse HEAD)
[ -z "$(git -C "$WT" status --porcelain)" ] || { echo "worktree has uncommitted changes"; exit 2; }
S="ssh -o BatchMode=yes fasrc"

stage() {
  existing=$($S "cat $W/code-vislora/COMMIT 2>/dev/null || true")
  if [ -n "$existing" ] && [ "$existing" != "$COMMIT" ]; then echo "code-vislora holds $existing, not $COMMIT: refusing"; exit 3; fi
  rsync -az --exclude .git --exclude __pycache__ --exclude '*.pyc' "$WT"/ fasrc:$W/code-vislora/
  $S "echo $COMMIT > $W/code-vislora/COMMIT && touch $W/.quota-test-vl && rm $W/.quota-test-vl && echo quota-write-ok"
  for f in "$HERE"/slurm/*.sbatch "$HERE"/slurm/submit-one.sh; do scp -q "$f" fasrc:$W/slurm/vl-$(basename "$f"); done
  scp -q "$HERE/inspect_vision.py" fasrc:$W/code-vislora/.inspect_vision.py
  $S "source $W/env.sh && cd $W/code-vislora && \$PY .inspect_vision.py $W/code-vislora"
}

submit_one() {  # name [dependency-job-id]
  $S "bash $W/slurm/vl-submit-one.sh $1 ${2:-}"
}

submit() {
  canary=$(submit_one canary); echo "canary $canary"
  ea=$(submit_one ea42); echo "ea42 $ea"
  sa=$(submit_one sa42); echo "sa42 $sa"
  for arm in as43 as44 bs42 bs43 bs44; do
    id=$(submit_one $arm); echo "$arm $id"
    ev=$(submit_one e${arm:0:1}${arm:2} "$id"); echo "  e${arm:0:1}${arm:2} $ev (afterok:$id)"
    sp=$(submit_one s${arm:0:1}${arm:2} "$ev"); echo "  s${arm:0:1}${arm:2} $sp (afterok:$ev, keeps <=8 GPUs at once)"
  done
  $S "squeue -u jzhou1125 -o '%.10i %.24j %.8T %.12r %.20S'"
}

case "${1:-}" in stage) stage ;; submit) submit ;; *) echo "usage: $0 stage|submit"; exit 1 ;; esac
