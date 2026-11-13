# Shared helpers for the trustmix lanes. Source after env.sh (needs W, PY).
# Used inside jobs (watcher) and on the login node (submit_children.sh, submit_suite.sh).

# Which GRPO jobs start from which SFT run's checkpoint-256. (Plain bash 3.2+ throughout.)
tm_children() {
  case "$1" in
    s2) echo "g2 g5 g6" ;;
    s4) echo "g4" ;;
    *) return 1 ;;
  esac
}
TM_GATE_STEP=256

lane_header() {
  echo "job=${SLURM_JOB_ID:-none} name=${SLURM_JOB_NAME:-none} host=$(hostname) start=$(date +%Y-%m-%dT%H:%M:%S%z) commit=$(cat "$W/code/COMMIT")"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
}

refuse_existing() {
  local path
  for path in "$@"; do
    if [ -e "$path" ]; then echo "refusing: $path already exists (runs are never overwritten)" >&2; exit 3; fi
  done
}

# Run sbatch without the parent job's Slurm/GPU variables, so a submission made from
# inside a running job does not inherit its allocation settings.
clean_sbatch() {
  (
    # SLURM_CONF is client configuration (where slurm.conf lives), not job state: keep it.
    while IFS= read -r name; do unset "$name"; done < <(compgen -e | grep -E '^(SLURM_|SBATCH_|SRUN_|SALLOC_)|^(CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|ROCR_VISIBLE_DEVICES)$' | grep -vx 'SLURM_CONF')
    sbatch --parsable "$@"
  )
}

# Job ID of a queued/running/recent job with this name, if any (guards a retry after an sbatch timeout).
# A failing squeue/sacct must not abort a `set -e` caller after it took the lock (that would strand the lock).
existing_job() {
  local name=$1 found
  found=$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null | head -n1) || found=""
  if [ -z "$found" ]; then
    found=$(sacct -n -X -P -u "$USER" --name "$name" -S now-6hours -o JobID 2>/dev/null | head -n1) || found=""
  fi
  echo "$found"
}

# Exactly-once submission of $W/slurm/<child>.sbatch: an atomic mkdir lock, then a receipt file.
#   receipts/<child>.submitted.json  exists  -> done, never submit again
#   receipts/<child>.lock            exists  -> another submitter holds it (or died mid-way: inspect)
submit_child() {
  local child=$1 origin=$2 script lock receipt jobid="" attempt
  script=$W/slurm/$child.sbatch
  lock=$W/receipts/$child.lock
  receipt=$W/receipts/$child.submitted.json
  [ -f "$script" ] || { echo "$child: missing $script" >&2; return 1; }
  if [ -e "$receipt" ]; then echo "$child: already submitted: $(cat "$receipt")"; return 0; fi
  if ! mkdir "$lock" 2>/dev/null; then
    echo "$child: lock $lock is held (another submitter is active, or one died mid-way; check squeue, then remove the lock)" >&2
    return 0
  fi
  for attempt in 1 2 3 4 5; do
    jobid=$(existing_job "sseg-tm-$child")
    [ -n "$jobid" ] && { echo "$child: found existing job $jobid, not resubmitting"; break; }
    if jobid=$(clean_sbatch "$script" 2>>"$lock/sbatch.err") && [ -n "$jobid" ]; then break; fi
    jobid=""
    echo "$child: sbatch attempt $attempt failed: $(tail -n1 "$lock/sbatch.err" 2>/dev/null)" >&2
    sleep "${TM_SBATCH_RETRY_SECONDS:-30}"
  done
  if [ -z "$jobid" ]; then
    mv "$lock" "$lock.failed-$(date +%s)"   # frees the lock so the login-node fallback can retry
    echo "$child: sbatch failed 5 times" >&2
    return 1
  fi
  printf '{"child": "%s", "job_id": "%s", "origin": "%s", "submitted_at": "%s", "host": "%s", "submitter_job": "%s"}\n' \
    "$child" "${jobid%%;*}" "$origin" "$(date +%Y-%m-%dT%H:%M:%S%z)" "$(hostname)" "${SLURM_JOB_ID:-login}" > "$receipt.tmp"
  mv "$receipt.tmp" "$receipt"
  echo "$child: submitted job ${jobid%%;*}"
}

# Verify an SFT checkpoint with the trainer's own check (marker + file hashes + step).
verify_gate_checkpoint() {
  local ckpt=$1
  (cd "$W/code" && "$PY" -c '
import sys
from nucleus_rl.sft import verify_sft_checkpoint
marker = verify_sft_checkpoint(sys.argv[1])
if marker["global_step"] != int(sys.argv[2]):
    raise SystemExit("unexpected step %s" % marker["global_step"])
print("verified SFT checkpoint step", marker["global_step"], sys.argv[1])
' "$ckpt" "$TM_GATE_STEP")
}

# Background watcher run inside an SFT job: wait for checkpoint-256 (sft.py renames
# .writing-checkpoint-256 -> checkpoint-256 only after sft_checkpoint.json is written),
# then submit that parent's GRPO children once.
watch_and_submit() {
  local run=$1 trainer_pid=$2 parent=$3 ckpt marker deadline child rc=0
  ckpt=$run/checkpoints/checkpoint-$TM_GATE_STEP
  marker=$ckpt/sft_checkpoint.json
  deadline=$(( $(date +%s) + ${TM_WATCH_TIMEOUT_SECONDS:-18000} ))
  echo "watch: parent=$parent waiting for $marker (children: $(tm_children "$parent"))"
  while [ ! -f "$marker" ]; do
    if [ -f "$run/failure.json" ]; then echo "watch: parent failed: $(cat "$run/failure.json")"; return 1; fi
    if ! kill -0 "$trainer_pid" 2>/dev/null; then
      sleep 5
      [ -f "$marker" ] && break
      echo "watch: trainer exited before checkpoint-$TM_GATE_STEP; nothing submitted"; return 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "watch: timed out; nothing submitted"; return 1; fi
    sleep "${TM_WATCH_POLL_SECONDS:-60}"
  done
  verify_gate_checkpoint "$ckpt" || { echo "watch: checkpoint failed verification; nothing submitted"; return 1; }
  for child in $(tm_children "$parent"); do submit_child "$child" "watcher:$parent" || rc=1; done
  echo "watch: done rc=$rc"
  return $rc
}

# Two processes on one GPU: run a list of argument arrays (by name) one after another.
run_chain() {
  local tag=$1 name rc=0 log
  shift
  for name in "$@"; do
    eval "local -a chain_args=(\"\${${name}[@]}\")"
    log=$W/logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-0}-$tag-$name.log
    echo "[$tag] $(date +%Y-%m-%dT%H:%M:%S%z) start $name -> $log"
    if OMP_NUM_THREADS=2 "$PY" -m nucleus_rl.grpo "${chain_args[@]}" > "$log" 2>&1; then
      echo "[$tag] $(date +%Y-%m-%dT%H:%M:%S%z) ok $name"
    else
      rc=1; echo "[$tag] $(date +%Y-%m-%dT%H:%M:%S%z) FAILED $name (see $log)"
    fi
  done
  return $rc
}
