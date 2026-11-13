#!/bin/bash
# Login-node submitter for the trustmix lanes. Every submission is exactly-once
# (atomic mkdir lock + receipts/<stem>.submitted.json) and needs a passing preflight
# whose recorded hashes still match env.sh, slurm/ and code/COMMIT.
#
#   slurm/submit.sh jobs r0 s2 s4     submit these sbatch scripts now
#   slurm/submit.sh children s2       fallback if the S2 watcher could not sbatch: submits G2 G5 G6
#                                     once S2 checkpoint-256 verifies (safe to repeat; never double-submits)
#   slurm/submit.sh children s4       same for G4
#   slurm/submit.sh suite a|b         suite evals, only after their runs finished verified
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-trustmix-20261104/env.sh
source "$W/slurm/common.sh"

(cd "$W" && sha256sum -c --quiet receipts/preflight.sha256) || {
  echo "preflight hashes missing or stale: rerun tools/preflight.py before submitting" >&2; exit 2; }

require_verified() {  # run directories whose summary.json must show a finished, verified run
  (cd "$W/code" && "$PY" - "$@" <<'PY'
import json, sys
from pathlib import Path
bad = []
for run in sys.argv[1:]:
    path = Path(run) / "summary.json"
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        bad.append(f"{run}: no readable summary.json ({error.__class__.__name__})")
        continue
    ok = (summary.get("status") == "verified_sft_run" and summary.get("completed_requested_steps")) \
        or summary.get("status") == "verified_run"
    if not ok or not (Path(run) / "adapter" / "adapter_model.safetensors").is_file():
        bad.append(f"{run}: status={summary.get('status')} steps={summary.get('optimizer_steps')} stop={summary.get('stop_reason')}")
    else:
        print(f"ok {Path(run).name}: {summary['status']} steps={summary.get('optimizer_steps')} stop={summary.get('stop_reason')}")
if bad:
    print("not ready:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
  )
}

mode=${1:-}; shift || true
case "$mode" in
  jobs)
    [ $# -gt 0 ] || { echo "usage: submit.sh jobs r0 s2 s4" >&2; exit 2; }
    for stem in "$@"; do submit_child "$stem" login; done ;;
  children)
    parent=${1:?usage: submit.sh children s2|s4}
    tm_children "$parent" >/dev/null || { echo "unknown parent $parent" >&2; exit 2; }
    run=$(ls -d "$W/runs/$parent-"* 2>/dev/null | head -n1) || run=""
    [ -n "$run" ] || { echo "$parent has no run directory yet" >&2; exit 1; }
    ckpt=$run/checkpoints/checkpoint-$TM_GATE_STEP
    [ -f "$ckpt/sft_checkpoint.json" ] || { echo "$parent: $ckpt not written yet; try later" >&2; exit 1; }
    verify_gate_checkpoint "$ckpt"
    for child in $(tm_children "$parent"); do submit_child "$child" "login:$parent"; done ;;
  suite)
    case "${1:-}" in
      a) require_verified "$W/runs/s2-27b448-trustmix-sft-seed42" "$W/runs/g2-27b448-trustmix-pq-from-s2c256"
         submit_child suite-a login ;;
      b) require_verified "$W/runs/s4-27b448-trustmix-sft-seed43" "$W/runs/g4-27b448-trustmix-pq-from-s4c256-seed43" \
           "$W/runs/g5-27b448-trustmix-wpq-fp03-fn07-from-s2c256" "$W/runs/g6-27b448-trustmix-pq-g16x1-from-s2c256"
         submit_child suite-b login ;;
      *) echo "usage: submit.sh suite a|b" >&2; exit 2 ;;
    esac ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
echo "receipts:"; ls -1 "$W/receipts" | grep -E 'submitted|lock' || true
