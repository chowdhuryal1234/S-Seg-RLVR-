#!/bin/bash
# Watch the vision-LoRA jobs; exit (and so wake the agent) on the first event that needs attention.
#   ./watch.sh clip   -> exit once all 3 B arms have >= 64 logged steps; print gradient/clip stats of all 5 arms
#   ./watch.sh train  -> exit once all 5 training jobs are COMPLETED; print their summaries
#   ./watch.sh eval   -> exit once the five new eval96 jobs are COMPLETED (ea42 finished earlier)
#   ./watch.sh all    -> exit once all 17 jobs are COMPLETED
# Every mode also exits on: a job in a failure state, a dependency that can never be satisfied, or SSH lost 3 polls in a row.
# Read-only on FASRC.
set -uo pipefail
MODE=${1:-train}
POLL=${POLL:-300}
W=/n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105
TRAIN="58643366 58643388 58643395 58643398 58643401"
ALL="58643358 58643360 58643366 58643369 58643377 58643388 58643389 58643390 58643395 58643396 58643397 58643398 58643399 58643400 58643401 58643402 58643404"
RUNS="vla-27b896-trustmix256-sft-visionnone-seed43 vla-27b896-trustmix256-sft-visionnone-seed44 vlb-27b896-trustmix256-sft-visionall-seed42 vlb-27b896-trustmix256-sft-visionall-seed43 vlb-27b896-trustmix256-sft-visionall-seed44"
S="ssh -o BatchMode=yes fasrc"
lost=0
for i in $(seq 1 200); do
  states=$($S "sacct -X -n -P -j $(echo $ALL | tr ' ' ',') -o JobID,JobName,State" 2>/dev/null)
  if [ -z "$states" ]; then
    lost=$((lost + 1)); [ $lost -ge 3 ] && { echo "SSH_LOST $(date +%H:%M): FASRC not reachable for 3 polls"; exit 0; }
    sleep "$POLL"; continue
  fi
  lost=0
  bad=$(echo "$states" | grep -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE")
  if [ -n "$bad" ]; then
    echo "BAD_STATE $(date +%H:%M):"; echo "$bad"
    $S "for id in \$(echo '$bad' | cut -d'|' -f1); do f=\$(ls $W/logs/*-\$id.out 2>/dev/null | head -1); echo == \$f; tail -25 \$f | cut -c1-300; done" 2>/dev/null
    exit 0
  fi
  never=$($S "squeue -u jzhou1125 -h -o '%i %j %r' | grep -i DependencyNeverSatisfied" 2>/dev/null)
  [ -n "$never" ] && { echo "DEPENDENCY_NEVER_SATISFIED $(date +%H:%M):"; echo "$never"; exit 0; }
  case "$MODE" in
    clip)
      out=$($S "cd $W/runs && python3 -c \"
import json, sys
runs = sys.argv[1:]
stats, ready = [], True
for run in runs:
    rows = []
    try:
        for l in open(run + '/training_log.jsonl'):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass  # a line still being written
    except OSError:
        pass
    if 'visionall' in run and len(rows) < 64:
        ready = False
    norms = [r['gradient_norm'] for r in rows]
    vision = [r['vision_gradient_norm'] for r in rows if r.get('vision_gradient_norm') is not None]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float('nan')
    stats.append('%s steps %d s/step %.1f median total %.3f clipped %d (%.0f%%) median vision %.3f' % (
        run[:44], len(rows), med([r['duration_seconds'] for r in rows]), med(norms), sum(n > 1 for n in norms),
        100 * sum(n > 1 for n in norms) / max(1, len(norms)), med(vision)))
print('READY' if ready else 'WAIT')
print('\n'.join(stats))
\" $RUNS 2>&1" 2>/dev/null)
      first=$(echo "$out" | head -1)
      if [ "$first" = "READY" ]; then echo "CLIP_CHECK $(date +%H:%M):"; echo "$out" | tail -n +2; exit 0; fi
      if [ -n "$out" ] && [ "$first" != "WAIT" ]; then echo "CLIP_QUERY_ERROR $(date +%H:%M):"; echo "$out" | head -30; exit 0; fi ;;
    train)
      done_n=$(echo "$states" | grep -E "^($(echo $TRAIN | tr ' ' '|'))\|" | grep -c "|COMPLETED$")
      if [ "$done_n" = "5" ]; then
        echo "TRAIN_DONE $(date +%H:%M):"
        $S "cd $W/runs && for r in $RUNS; do python3 -c \"
import json, sys
d = json.load(open(sys.argv[1] + '/summary.json'))
print(sys.argv[1][:44], d['status'], 'vision', d.get('vision_lora'), 'steps', d['optimizer_steps'], 'complete', d['completed_requested_steps'],
      'stop', d['stop_reason'], 's/step %.1f' % d['mean_optimizer_step_seconds'], 'peak %.1f GB' % (d['peak_gpu_allocated_bytes'] / 1e9),
      'quick48 PQ %.3f' % d['after']['mean_pq'], 'invalid', d['after']['invalid_completions'])
\" \$r; done" 2>/dev/null
        exit 0
      fi ;;
    eval)
      done_n=$(echo "$states" | grep -E "^(58643369|58643389|58643396|58643399|58643402)\|" | grep -c "|COMPLETED$")
      [ "$done_n" = "5" ] && { echo "EVAL96_DONE $(date +%H:%M)"; exit 0; } ;;
    all)
      done_n=$(echo "$states" | grep -c "|COMPLETED$")
      [ "$done_n" = "17" ] && { echo "ALL_DONE $(date +%H:%M)"; exit 0; } ;;
  esac
  sleep "$POLL"
done
echo "WATCH_TIMEOUT $(date +%H:%M) mode $MODE"
