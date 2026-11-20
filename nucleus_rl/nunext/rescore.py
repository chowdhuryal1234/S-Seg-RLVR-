"""Re-score an evaluate run's predictions.jsonl without a GPU: both matching protocols, and the TTA vote sweep.

    python -m nucleus_rl.nunext.rescore --predictions run/eval-final/predictions.jsonl \\
        --manifest data/pannuke_nunext_v2/fold3.jsonl [--output rescore.json]

``paper``: assignment on raw distances, then the radius cut (the headline F1 so far).
``capped``: pairs beyond the radius are unassignable before solving (no correct point is lost to the
global assignment). Records written with ``--tta d4`` also get F1 for every vote threshold 1-8.
``--positions`` adds where the misses are (raster position, y, after the last prediction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def miss_positions(records, rows, radius: float = 12.0, bands: int = 8) -> dict:
    """Where the misses are, with capped matching.

    If the model stops too early, misses pile up at the end of the raster order (the bottom of the
    image) and after its last predicted point; if they are scattered, recall is flat across bands.

    - ``by_rank``: recall per band of each nucleus's raster position (sorted by y, then x) within its image.
    - ``by_y``: recall per band of the nucleus's y coordinate.
    - ``tail``: misses after the last predicted point in raster order (all misses when nothing was predicted),
      against the misses before it; ``tail_if_random`` is the expected tail count if the same number of
      misses per image fell on uniformly random nuclei.
    - ``count_ratio``: predictions / ground truth per density band (nuclei per image).
    """
    centroids = {row["id"]: (row["centroids"], row["height"]) for row in rows}
    rank_hits, rank_all = np.zeros(bands), np.zeros(bands)
    y_hits, y_all = np.zeros(bands), np.zeros(bands)
    tail = interior = 0
    tail_if_random = 0.0
    density = {}   # band -> [predictions, ground truth, images]
    for record in records:
        gt, height = centroids[record["id"]]
        pred = record["pred"] if record.get("valid", True) else []
        key = "0-10" if len(gt) < 10 else "10-25" if len(gt) < 25 else "25-50" if len(gt) < 50 else \
            "50-100" if len(gt) < 100 else "100+"
        entry = density.setdefault(key, [0, 0, 0])
        entry[0] += len(pred)
        entry[1] += len(gt)
        entry[2] += 1
        if not gt:
            continue
        hit = match_points(pred, gt, radius, capped=True).gt_is_tp
        order = sorted(range(len(gt)), key=lambda i: (gt[i][1], gt[i][0]))
        last = max(((p[1], p[0]) for p in pred), default=None)
        misses_here, after_last = 0, 0
        for rank, i in enumerate(order):
            band = min(int(rank / len(gt) * bands), bands - 1)
            rank_all[band] += 1
            rank_hits[band] += hit[i]
            y_band = min(int(gt[i][1] / height * bands), bands - 1)
            y_all[y_band] += 1
            y_hits[y_band] += hit[i]
            beyond = last is None or (gt[i][1], gt[i][0]) > last
            after_last += beyond
            if not hit[i]:
                misses_here += 1
                tail += beyond
                interior += not beyond
        tail_if_random += misses_here * after_last / len(gt)
    recall = lambda hits, total: [round(float(h / t), 4) if t else None for h, t in zip(hits, total)]
    return {"bands": bands, "by_rank": recall(rank_hits, rank_all), "by_y": recall(y_hits, y_all),
            "tail": {"after_last_prediction": tail, "before_it": interior,
                     "after_last_if_random": round(tail_if_random, 1)},
            "count_ratio": {k: {"pred_over_gt": round(v[0] / v[1], 3) if v[1] else None, "images": v[2]}
                            for k, v in density.items()}}


def main(argv=None):
    from nucleus_rl.nunext.data import read_manifests

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, action="append", help="One or more predictions.jsonl")
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--output", help="Write the results as JSON here")
    parser.add_argument("--positions", action="store_true", help="Also report where the misses are (miss_positions)")
    args = parser.parse_args(argv)
    rows = read_manifests(args.manifest)
    results = {}
    for path in args.predictions:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        results[path] = rescore(records, rows, args.radius)
        if args.positions:
            results[path]["miss_positions"] = miss_positions(records, rows, args.radius)
            print(json.dumps(results[path]["miss_positions"], indent=1), flush=True)
        paper, capped = results[path]["paper"], results[path]["capped"]
        print(f"{path}\n  paper  F1 {paper['micro_f1']:.4f}  P {paper['micro_precision']:.4f}  "
              f"R {paper['micro_recall']:.4f}\n  capped F1 {capped['micro_f1']:.4f}  "
              f"P {capped['micro_precision']:.4f}  R {capped['micro_recall']:.4f}", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
