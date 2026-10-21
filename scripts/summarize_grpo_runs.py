#!/usr/bin/env python3
"""Print one line per GRPO/SFT run: status, start, best, final PQ, final TP, count MAE and the PQ curve.

    python scripts/summarize_grpo_runs.py --runs runs/grpo_20261020
    python scripts/summarize_grpo_runs.py --runs runs/grpo_20261020 \
        --sync-from fasrc:/n/holylabs/mooney_lab/Lab/junyizhou/sseg-grpo-20261020/runs/

`--sync-from` first copies only JSON/JSONL files (no checkpoints or adapters) with rsync.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def sync(remote, local):
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--prune-empty-dirs", "--include=*/", "--include=*.json", "--include=*.jsonl",
                    "--exclude=checkpoints/***", "--exclude=adapter/***", "--exclude=*", remote, str(local) + "/"],
                   check=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def point(summary):
    if not summary:
        return None
    diag = summary.get("instance_diagnostics", {})
    return {"pq": summary["mean_pq"], "tp": diag.get("total_tp"), "pred": diag.get("total_pred_count"),
            "count_mae": summary.get("mean_count_error"), "recall": summary.get("mean_recall")}


def curve(run):
    """[(step, metrics)] including the starting point (step 0) and the final evaluation."""
    points = []
    before = point(read_json(run / "before" / "summary.json"))
    if before:
        points.append((0, before))
    history = run / "validation_history.jsonl"
    if history.exists():
        points += [(item["step"], point(item["summary"])) for item in map(json.loads, history.read_text().splitlines())]
    summary = read_json(run / "summary.json")
    after = point(read_json(run / "after" / "summary.json"))
    if after:
        steps = summary.get("optimizer_steps") if summary else None
        points.append((steps or "final", after))
    return points


def table(runs):
    print(f"{'run':30s} {'status':16s} {'start PQ':>8s} {'best PQ':>8s} {'final PQ':>8s} {'final TP':>8s} {'cnt MAE':>7s}  curve")
    for run in runs:
        points = curve(run)
        progress = read_json(run / "progress.json") or {}
        status = (read_json(run / "summary.json") or {}).get("status") or progress.get("phase", "?")
        if (run / "failure.json").exists():
            status = "FAILED"
        if not points:
            print(f"{run.name:30s} {status:16s}")
            continue
        final = points[-1][1] if (run / "after" / "summary.json").exists() else None
        best = max(p["pq"] for _, p in points)
        line = " ".join(f"{step}:{p['pq']:.3f}" for step, p in points)
        print(f"{run.name:30s} {status:16s} {points[0][1]['pq']:8.3f} {best:8.3f} "
              f"{(final or {}).get('pq', float('nan')):8.3f} {str((final or {}).get('tp', '-')):>8s} "
              f"{(final or {}).get('count_mae', float('nan')):7.2f}  {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Local directory of run folders")
    parser.add_argument("--sync-from", help="rsync source (remote runs directory, with trailing slash)")
    args = parser.parse_args()
    if args.sync_from:
        sync(args.sync_from, args.runs)
    runs = sorted(path for path in args.runs.iterdir() if path.is_dir() and not path.name.startswith(("canary", "diag")))
    table(runs)


if __name__ == "__main__":
    main()
