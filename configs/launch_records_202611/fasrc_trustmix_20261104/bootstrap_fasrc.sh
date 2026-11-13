#!/bin/bash
# Run on the Mac once `ssh fasrc` (password + Duo) is open in another Terminal.
# Creates and fills /n/holylabs/.../sseg-trustmix-20261104, then runs the login-node preflight and
# sbatch --test-only. It never submits a job and never writes outside the new workspace.
# Safe to re-run: code is extracted only into an empty code/, data uses rsync --ignore-existing and is
# then checked file-by-file with sha256, launch files are refreshed only while no job has a receipt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/conny/Desktop/RLVR/grpo-worktree-20261020
LDATA=/Users/conny/Desktop/RLVR/S-Seg-RLVR-/data
LRUNS=/Users/conny/Desktop/RLVR/S-Seg-RLVR-/runs/grpo_20261020
COMMIT=$(git -C "$REPO" rev-parse 1bc1b5e)
LAB=/n/holylabs/mooney_lab/Lab/junyizhou
W=$LAB/sseg-trustmix-20261104
OLD=$LAB/sseg-grpo-20261020
R=$LAB/sseg-sft-20261017
PY=$R/.venv/bin/python
DATASETS=(trustmix_v2 monuseg_official30_grpo_v1 monuseg_official30_dense_v1 consep_v1 consep_v2 puma_v1 puma_v2)
SSH=(ssh -o BatchMode=yes fasrc)
STAMP=$(date +%Y%m%d-%H%M%S)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ssh -O check fasrc || { echo "No ControlMaster: run 'ssh fasrc' in another Terminal (password + Duo) and leave it open." >&2; exit 1; }

echo "== 1. old launch tooling (the env these lanes copy)"
"${SSH[@]}" "cat $OLD/env.sh; echo '--- G7 sbatch'; cat $OLD/slurm/g7-27b448c128-pq.sbatch; echo '--- H2 sbatch'; cat $OLD/slurm/h2-27b-sft-448.sbatch 2>/dev/null || ls $OLD/slurm | grep -i h2"
for run in h2-27b-d4-sft448 g7-27b448c128-pq; do
  remote=$("${SSH[@]}" "sha256sum $OLD/runs/$run/arguments.json" | cut -d' ' -f1)
  local_sha=$(shasum -a 256 "$LRUNS/$run/arguments.json" | cut -d' ' -f1)
  [ "$remote" = "$local_sha" ] && echo "$run/arguments.json: FASRC == local copy" || echo "WARNING $run/arguments.json differs from the local copy"
done
"${SSH[@]}" "ls -d $OLD/runs/h2-27b-d4-sft448/adapter $OLD/runs/h2-27b-d4-sft448/checkpoints/checkpoint-* $OLD/runs/g7-27b448c128-pq/adapter; ls -la $OLD/runs/g7-27b448c128-pq/adapter/adapter_model.safetensors $OLD/runs/h2-27b-d4-sft448/adapter/adapter_model.safetensors"

echo "== 2. workspace, write test, space"
"${SSH[@]}" "set -e; mkdir -p $W/code $W/data $W/runs $W/logs $W/slurm $W/receipts $W/tools
  touch $W/receipts/.write-test && rm $W/receipts/.write-test && echo 'write test ok'
  df -h $W | tail -n1; (quota -s 2>/dev/null | tail -n3) || true
  timeout 120 du -sh $LAB 2>/dev/null | tail -n1 || echo 'du of the lab folder timed out (not fatal)'"

echo "== 3. code @ $COMMIT"
existing=$("${SSH[@]}" "cat $W/code/COMMIT 2>/dev/null || true")
if [ -z "$existing" ]; then
  "${SSH[@]}" "[ -z \"\$(ls -A $W/code)\" ]" || { echo "code/ is not empty but has no COMMIT; refusing" >&2; exit 1; }
  git -C "$REPO" archive "$COMMIT" | "${SSH[@]}" "tar -x -C $W/code && echo $COMMIT > $W/code/COMMIT"
  echo "extracted"
elif [ "$existing" != "$COMMIT" ]; then
  echo "code/COMMIT is $existing, expected $COMMIT; refusing to mix code" >&2; exit 1
else
  echo "already present"
fi

echo "== 4. data (sibling layout kept; trustmix_v2 rows use ../ paths)"
for d in "${DATASETS[@]}"; do
  rsync -a --ignore-existing -e "ssh -o BatchMode=yes" "$LDATA/$d" "fasrc:$W/data/"
done
(cd "$LDATA" && find "${DATASETS[@]}" -type f -print0 | sort -z | xargs -0 shasum -a 256) > "$TMP/data.sha256"
echo "$(wc -l < "$TMP/data.sha256") files hashed locally"
scp -q -o BatchMode=yes "$TMP/data.sha256" "fasrc:$W/receipts/data-$STAMP.sha256"
"${SSH[@]}" "cd $W/data && sha256sum -c --quiet $W/receipts/data-$STAMP.sha256 && echo 'all synced data files match local sha256'"
# MoNuSeg rows must be byte-identical to the copy H2 and G7 trained and validated on.
"${SSH[@]}" "cd $R/code/data && grep '  monuseg_official30_grpo_v1/' $W/receipts/data-$STAMP.sha256 | sha256sum -c --quiet && echo 'MoNuSeg sparse folder == the H2/G7 copy in the SFT workspace'" \
  || echo "WARNING: synced MoNuSeg folder differs from $R/code/data (inspect before submitting)"

echo "== 5. launch files"
if "${SSH[@]}" "ls $W/receipts/*.submitted.json >/dev/null 2>&1"; then
  echo "jobs already have receipts: launch files are NOT refreshed (copy by hand if a change is intended)"
else
  rsync -a -e "ssh -o BatchMode=yes" "$HERE/env.sh" "fasrc:$W/"
  rsync -a -e "ssh -o BatchMode=yes" "$HERE/slurm/" "fasrc:$W/slurm/"
  rsync -a -e "ssh -o BatchMode=yes" "$HERE/tools/" "fasrc:$W/tools/"
  "${SSH[@]}" "chmod +x $W/slurm/submit.sh; for f in $W/slurm/*.sbatch $W/slurm/*.sh; do bash -n \$f || echo SYNTAX \$f; done; echo 'launch files synced'"
fi

echo "== 6. preflight on the login node (CPU only; manifests, splits, argparse, init adapters)"
"${SSH[@]}" "cd $W && $PY tools/preflight.py"

echo "== 7. sbatch --test-only (one job of each kind; nothing is submitted)"
"${SSH[@]}" "cd $W && for s in s2 g2 r0; do echo \"-- \$s\"; sbatch --test-only slurm/\$s.sbatch 2>&1 | tail -n2; done"

echo "== 8. queue context"
"${SSH[@]}" "sshare -U -u jzhou1125 2>/dev/null | tail -n3; squeue -u jzhou1125 -o '%.10i %.28j %.8T %.10M %R' | head -n 20"
echo "bootstrap done (nothing submitted)"
