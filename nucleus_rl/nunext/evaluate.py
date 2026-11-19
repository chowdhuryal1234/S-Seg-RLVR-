"""Greedy-decoding detection evaluation (F1 at a distance threshold) on a manifest.

Every run reports two matching protocols (nucleus_rl.nunext.matching.match_points): the paper's
(assignment on raw distances, then the radius cut; ``micro_*``) and the capped one (pairs beyond the
radius unassignable; ``capped_micro_*``). With ``--tta d4`` each image is decoded in all eight D4 views
and the results are merged by voting (nucleus_rl.nunext.tta); the summary then also holds a sweep over
the vote threshold, and each prediction record keeps every merged nucleus with its votes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nucleus_rl.nunext.coords import bins_to_boxes, bins_to_points, parse_token_ids, salvage_token_ids
from nucleus_rl.nunext.data import load_example, read_manifests
from nucleus_rl.nunext.matching import Match, aggregate, match_points


def score_completion(ids, row, first_id, bins, stop_ids, radius, per_object=2, capped=False):
    """(parse result, match) for one generated sequence; invalid output counts as no predictions."""
    parsed = parse_token_ids(ids, first_id, bins, stop_ids, per_object)
    points = bins_to_points(parsed.bins, row["width"], row["height"], bins) if parsed.valid else []
    return parsed, points, match_points(points, row["centroids"], radius, capped)


def salvaged_match(ids, row, first_id, bins, stop_ids, radius, per_object=2):
    """Match for the lenient parse (salvage_token_ids); used only for the diagnostic salvaged_* metrics."""
    parsed = salvage_token_ids(ids, first_id, bins, stop_ids, per_object)
    points = bins_to_points(parsed.bins, row["width"], row["height"], bins)
    return points, match_points(points, row["centroids"], radius)


def _counts(match: Match) -> dict:
    return {"tp": match.tp, "fp": match.fp, "fn": match.fn}


def _rounded(values):
    return [[round(v, 1) for v in item] for item in values]


def _single_view_record(row, ids, first_id, bins, stop_ids, radius, per_object):
    parsed, points, match = score_completion(ids, row, first_id, bins, stop_ids, radius, per_object)
    record = {"id": row["id"], "valid": parsed.valid, "error": parsed.error, "tokens": len(ids),
              "pred": _rounded(points), **_counts(match),
              "capped": _counts(match_points(points, row["centroids"], radius, capped=True))}
    if per_object == 4:
        record["pred_boxes"] = (_rounded(bins_to_boxes(parsed.bins, row["width"], row["height"], bins))
                                if parsed.valid else [])
    if parsed.valid:
        record["salvaged"] = {**_counts(match), "points": len(points)}
    else:
        salvaged_points, salvaged = salvaged_match(ids, row, first_id, bins, stop_ids, radius, per_object)
        record["salvaged"] = {**_counts(salvaged), "points": len(salvaged_points)}
    return record


def tta_record(row, view_ids, first_id, bins, stop_ids, radius, per_object, merge_radius, min_votes):
    """Merge the eight views' outputs of one image; invalid views contribute nothing."""
    from nucleus_rl.nunext import tta

    size = float(row["width"])
    per_view, valid_views, tokens = [], 0, 0
    for view, ids in zip(tta.VIEWS, view_ids):
        tokens += len(ids)
        parsed = parse_token_ids(ids, first_id, bins, stop_ids, per_object)
        if not parsed.valid:
            per_view.append([])
            continue
        valid_views += 1
        objects = [tuple(v / (bins - 1) * size for v in group) for group in parsed.bins]
        per_view.append(tta.inverse_objects(objects, view, size))
    merged = tta.merge_views(per_view, merge_radius)
    points, boxes = tta.select(merged, min_votes, row["width"], row["height"])
    match = match_points(points, row["centroids"], radius)
    record = {"id": row["id"], "valid": valid_views > 0, "error": None if valid_views else "no valid view",
              "tokens": tokens, "valid_views": valid_views, "pred": _rounded(points), **_counts(match),
              "capped": _counts(match_points(points, row["centroids"], radius, capped=True)),
              "salvaged": {**_counts(match), "points": len(points)},
              "tta": {"points": _rounded([m["point"] for m in merged]), "votes": [m["votes"] for m in merged]}}
    if per_object == 4:
        record["pred_boxes"] = _rounded(boxes or [])
        record["tta"]["sizes"] = _rounded([m["size"] for m in merged])
    return record


def summarize(records, rows, radius, started=None) -> dict:
    from nucleus_rl.nunext.rescore import vote_sweep

    matches = [Match(r["tp"], r["fp"], r["fn"]) for r in records]
    capped = aggregate([Match(**r["capped"]) for r in records])
    salvaged = aggregate([Match(r["salvaged"]["tp"], r["salvaged"]["fp"], r["salvaged"]["fn"]) for r in records])
    errors = {}
    for record in records:
        if record["error"]:
            errors[record["error"]] = errors.get(record["error"], 0) + 1
    summary = {**aggregate(matches), "invalid": sum(not r["valid"] for r in records), "errors": errors,
               "radius": radius,
               "capped_micro_f1": capped["micro_f1"], "capped_micro_precision": capped["micro_precision"],
               "capped_micro_recall": capped["micro_recall"],
               # Diagnostic only: malformed/truncated outputs re-read up to their first raster-order break.
               "salvaged_micro_f1": salvaged["micro_f1"], "salvaged_micro_precision": salvaged["micro_precision"],
               "salvaged_micro_recall": salvaged["micro_recall"]}
    if any("tta" in r for r in records):
        summary["tta_vote_sweep"] = vote_sweep(records, rows, radius)
    if started is not None:
        summary["seconds"] = round(time.time() - started, 1)
    return summary


def evaluate(model, processor, rows, *, first_id, bins, image_size, radius, batch_size, max_new_tokens,
             output: Path | None = None, keep_predictions: bool = True, per_object: int = 2,
             tta: str | None = None, merge_radius: float = 8.0, min_votes: int = 4) -> dict:
    """Under torchrun each rank scores rows[rank::world]; every rank returns the full summary, rank 0 writes."""
    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.modeling import end_ids, generate
    from nucleus_rl.nunext.tta import VIEWS

    if tta not in (None, "d4"):
        raise ValueError("tta must be None or 'd4'")
    started = time.time()
    stop_ids = end_ids(processor, model)
    model.eval()
    records = []
    shard = rows[dist.rank()::dist.world_size()]
    views = VIEWS if tta else (0,)
    step = max(1, batch_size // len(views))   # batch_size counts generated sequences
    for start in range(0, len(shard), step):
        chunk = shard[start:start + step]
        examples = [load_example(row, bins, view, image_size) for row in chunk for view in views]
        sequences = generate(model, processor, [e["image"] for e in examples], do_sample=False,
                             temperature=1.0, max_new_tokens=max_new_tokens)
        for index, row in enumerate(chunk):
            own = sequences[index * len(views):(index + 1) * len(views)]
            if tta:
                records.append(tta_record(row, own, first_id, bins, stop_ids, radius, per_object,
                                          merge_radius, min_votes))
            else:
                records.append(_single_view_record(row, own[0], first_id, bins, stop_ids, radius, per_object))
    records = dist.gather_objects(records)
    order = {row["id"]: index for index, row in enumerate(rows)}
    records.sort(key=lambda record: order[record["id"]])
    summary = summarize(records, rows, radius, started)
    if tta:
        summary.update({"tta": tta, "merge_radius": merge_radius, "min_votes": min_votes})
    if output is not None and dist.is_main():
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if keep_predictions:
            with (output / "predictions.jsonl").open("w") as stream:
                stream.writelines(json.dumps(record) + "\n" for record in records)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="A NuNext SFT/GRPO output folder (model + processor)")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=16, help="Generated sequences per batch (8 per image with TTA)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--box-tokens", action="store_true", help="The model outputs x y w h per nucleus")
    parser.add_argument("--tta", choices=["d4"], help="Decode all eight D4 views and merge them by voting")
    parser.add_argument("--merge-radius", type=float, default=8.0, help="TTA: px within which views' nuclei merge")
    parser.add_argument("--min-votes", type=int, default=4, help="TTA: views that must find a nucleus (1-8)")
    args = parser.parse_args(argv)
    if not 1 <= args.min_votes <= 8:
        parser.error("--min-votes must be in 1..8")
    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.modeling import coordinate_first_id, load_policy

    device = dist.setup()   # sharded across GPUs when started by nucleus_rl.nunext.launch
    model, processor, _ = load_policy(args.model, "main", dtype="bfloat16", device=device, trainable=False)
    first_id = coordinate_first_id(processor.tokenizer, args.bins)
    rows = read_manifests(args.manifest, args.limit)
    summary = evaluate(model, processor, rows, first_id=first_id, bins=args.bins, image_size=args.image_size,
                       radius=args.radius, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                       output=Path(args.output), per_object=4 if args.box_tokens else 2,
                       tta=args.tta, merge_radius=args.merge_radius, min_votes=args.min_votes)
    if dist.is_main():
        print(json.dumps(summary, indent=2), flush=True)
    dist.teardown()


if __name__ == "__main__":
    main()
