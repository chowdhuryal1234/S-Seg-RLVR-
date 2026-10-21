"""Compact read-only status of the GRPO-from-SFT runs and the Slurm queue; prints JSON.

Runs on the FASRC login node:
    python scripts/grpo_status.py --workspace /n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020
"""
import argparse
import json
import subprocess
from pathlib import Path


def last_line(path):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 200_000))
            lines = handle.read().decode("utf-8", "replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def metrics(summary):
    if not summary:
        return None
    diag = summary.get("instance_diagnostics", {})
    return {"pq": summary.get("mean_pq"), "tp": diag.get("total_tp"), "pred": diag.get("total_pred_count"),
            "gt": diag.get("total_gt_count"), "count_mae": summary.get("mean_count_error"),
            "invalid": summary.get("invalid_completions")}


def run_status(run):
    status = {"run": run.name}
    if run.name.startswith("diag-"):
        status["results"] = sorted(path.name for path in run.glob("*.json"))
    for name in ("progress.json", "failure.json"):
        path = run / name
        if path.exists():
            try:
                status[name.split(".")[0]] = json.loads(path.read_text())
            except ValueError:
                status[name.split(".")[0]] = "unreadable"
    log = last_line(run / "training_log.jsonl")
    if log:
        status["last_step"] = {key: log.get(key) for key in ("step", "duration_seconds", "reward_mean",
                                                             "groups_with_signal", "updated", "gradient_norm")}
        micro = log.get("micro") or []
        if micro and "kl" in micro[0]:
            status["last_step"]["kl"] = sum(item.get("kl", 0) for item in micro) / len(micro)
        if micro and "loss" in micro[0]:
            status["last_step"]["loss"] = sum(item.get("loss", 0) for item in micro) / len(micro)
    before = run / "before" / "summary.json"
    if before.exists():
        status["before"] = metrics(json.loads(before.read_text()))
    history = run / "validation_history.jsonl"
    if history.exists():
        status["evals"] = [{"step": item["step"], **metrics(item["summary"])}
                           for item in map(json.loads, history.read_text().splitlines())]
    after = run / "after" / "summary.json"
    if after.exists():
        status["after"] = metrics(json.loads(after.read_text()))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True, help="Directory that contains runs/")
    args = parser.parse_args()
    queue = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%l|%R"], capture_output=True, text=True)
    jobs = [dict(zip(("id", "name", "state", "elapsed", "limit", "where"), line.split("|")))
            for line in queue.stdout.splitlines() if line.strip()]
    runs = [run_status(run) for run in sorted((args.workspace / "runs").iterdir()) if run.is_dir()]
    print(json.dumps({"jobs": jobs, "squeue_ok": queue.returncode == 0, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
