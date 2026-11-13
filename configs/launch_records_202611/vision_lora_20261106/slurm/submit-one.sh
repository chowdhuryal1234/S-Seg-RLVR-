#!/bin/bash
# Exactly-once submit of one vision-LoRA job: vl-submit-one.sh <name> [afterok-job-id]. Prints the job id.
# An existing receipt ($W/receipts/vl-<name>.submitted.json) means the job was already submitted: print its id, submit nothing.
set -euo pipefail
source /n/holylabs/mooney_lab/Lab/junyizhou/sseg-next-20261105/env.sh
name=$1
dep=${2:-}
receipt=$W/receipts/vl-$name.submitted.json
if [ -e "$receipt" ]; then
  echo "vl-$name: already submitted: $(cat "$receipt")" >&2
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$receipt"
  exit 0
fi
if [ -n "$dep" ]; then
  id=$(sbatch --parsable --dependency=afterok:"$dep" "$W/slurm/vl-$name.sbatch")
else
  id=$(sbatch --parsable "$W/slurm/vl-$name.sbatch")
fi
printf '{"stem":"vl-%s","job_id":"%s","dependency":"%s","submitted_at":"%s","commit":"%s"}\n' \
  "$name" "$id" "$dep" "$(date +%FT%T%z)" "$(cat "$W/code-vislora/COMMIT")" > "$receipt"
echo "$id"
