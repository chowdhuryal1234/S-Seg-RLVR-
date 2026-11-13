#!/usr/bin/env bash
# Pull the A/B study outputs from FASRC into ./results, without weights or per-tile mask artifacts.
# Read-only on FASRC. Safe to re-run: rsync only adds or updates files locally.
set -euo pipefail
W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/results"
SSH="ssh -o BatchMode=yes"
$SSH -O check fasrc >/dev/null 2>&1 || { echo "FASRC not connected: run 'ssh fasrc' (Duo) first" >&2; exit 1; }
mkdir -p "$DEST"
RUNS="t256s1-27b896-trustmix256-sft-seed42 vla-27b896-trustmix256-sft-visionnone-seed43 vla-27b896-trustmix256-sft-visionnone-seed44 vlb-27b896-trustmix256-sft-visionall-seed42 vlb-27b896-trustmix256-sft-visionall-seed43 vlb-27b896-trustmix256-sft-visionall-seed44"
for run in $RUNS; do
  for dir in "$run" "$run-eval96" "$run-support96"; do
    if $SSH fasrc "test -d $W/runs/$dir"; then
      rsync -az -e "$SSH" --exclude checkpoints/ --exclude adapter/ --exclude '*.npy' --exclude '*.npz' --exclude '*.png' \
        "fasrc:$W/runs/$dir/" "$DEST/$dir/"
      echo "pulled $dir"
    else
      echo "not on FASRC yet: $dir"
    fi
  done
done
if $SSH fasrc "test -d $W/runs/canary-vlb-27b896-3steps"; then
  rsync -az -e "$SSH" --exclude checkpoints/ --exclude adapter/ --exclude '*.npy' --exclude '*.npz' --exclude '*.png' \
    "fasrc:$W/runs/canary-vlb-27b896-3steps/" "$DEST/canary-vlb-27b896-3steps/"
  echo "pulled canary-vlb-27b896-3steps"
fi
