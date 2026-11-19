"""Re-score an evaluate run's predictions.jsonl without a GPU: both matching protocols, and the TTA vote sweep.

    python -m nucleus_rl.nunext.rescore --predictions run/eval-final/predictions.jsonl \\
        --manifest data/pannuke_nunext_v2/fold3.jsonl [--output rescore.json]

``paper``: assignment on raw distances, then the radius cut (the headline F1 so far).
``capped``: pairs beyond the radius are unassignable before solving (no correct point is lost to the
global assignment). Records written with ``--tta d4`` also get F1 for every vote threshold 1-8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nucleus_rl.nunext.matching import aggregate, match_points


def _scores(pairs, radius, capped):
    summary = aggregate([match_points(points, gt, radius, capped) for points, gt in pairs])
    return {key: summary[key] for key in ("tp", "fp", "fn", "micro_precision", "micro_recall", "micro_f1")}


def rescore(records, rows, radius: float = 12.0) -> dict:
    """Both protocols for the records' ``pred`` points against the rows' centroids."""
    centroids = {row["id"]: row["centroids"] for row in rows}
    missing = [r["id"] for r in records if r["id"] not in centroids]
    if missing:
        raise KeyError(f"{len(missing)} predictions have no manifest row, e.g. {missing[0]}")
    pairs = [(r["pred"] if r.get("valid", True) else [], centroids[r["id"]]) for r in records]
    result = {"images": len(records), "radius": radius,
              "paper": _scores(pairs, radius, False), "capped": _scores(pairs, radius, True)}
    if any("tta" in r for r in records):
        result["tta_vote_sweep"] = vote_sweep(records, rows, radius)
    return result


def vote_sweep(records, rows, radius: float = 12.0) -> dict:
    """{min_votes: {"paper": ..., "capped": ...}} from the merged nuclei stored by a TTA run."""
    centroids = {row["id"]: row["centroids"] for row in rows}
    sweep = {}
    for min_votes in range(1, 9):
        pairs = []
        for record in records:
            tta = record.get("tta", {"points": [], "votes": []})
            points = [p for p, v in zip(tta["points"], tta["votes"]) if v >= min_votes]
            pairs.append((points, centroids[record["id"]]))
        sweep[str(min_votes)] = {"paper": _scores(pairs, radius, False), "capped": _scores(pairs, radius, True)}
    return sweep


def main(argv=None):
    from nucleus_rl.nunext.data import read_manifests

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, action="append", help="One or more predictions.jsonl")
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--output", help="Write the results as JSON here")
    args = parser.parse_args(argv)
    rows = read_manifests(args.manifest)
    results = {}
    for path in args.predictions:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        results[path] = rescore(records, rows, args.radius)
        paper, capped = results[path]["paper"], results[path]["capped"]
        print(f"{path}\n  paper  F1 {paper['micro_f1']:.4f}  P {paper['micro_precision']:.4f}  "
              f"R {paper['micro_recall']:.4f}\n  capped F1 {capped['micro_f1']:.4f}  "
              f"P {capped['micro_precision']:.4f}  R {capped['micro_recall']:.4f}", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
