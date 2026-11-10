"""Greedy-decoding detection evaluation (F1 at a distance threshold) on a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nucleus_rl.nunext.coords import bins_to_points, parse_token_ids, salvage_token_ids
from nucleus_rl.nunext.data import load_example, read_manifests
from nucleus_rl.nunext.matching import aggregate, match_points


def score_completion(ids, row, first_id, bins, stop_ids, radius):
    """(parse result, match) for one generated sequence; invalid output counts as no predictions."""
    parsed = parse_token_ids(ids, first_id, bins, stop_ids)
    points = bins_to_points(parsed.bins, row["width"], row["height"], bins) if parsed.valid else []
    return parsed, points, match_points(points, row["centroids"], radius)


def salvaged_match(ids, row, first_id, bins, stop_ids, radius):
    """Match for the lenient parse (salvage_token_ids); used only for the diagnostic salvaged_* metrics."""
    parsed = salvage_token_ids(ids, first_id, bins, stop_ids)
    points = bins_to_points(parsed.bins, row["width"], row["height"], bins)
    return points, match_points(points, row["centroids"], radius)


def evaluate(model, processor, rows, *, first_id, bins, image_size, radius, batch_size, max_new_tokens,
             output: Path | None = None, keep_predictions: bool = True) -> dict:
    """Under torchrun each rank scores rows[rank::world]; every rank returns the full summary, rank 0 writes."""
    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.matching import Match
    from nucleus_rl.nunext.modeling import end_ids, generate

    started = time.time()
    stop_ids = end_ids(processor, model)
    model.eval()
    records = []
    shard = rows[dist.rank()::dist.world_size()]
    for start in range(0, len(shard), batch_size):
        chunk = shard[start:start + batch_size]
        examples = [load_example(row, bins, 0, image_size) for row in chunk]
        sequences = generate(model, processor, [e["image"] for e in examples], do_sample=False,
                             temperature=1.0, max_new_tokens=max_new_tokens)
        for row, ids in zip(chunk, sequences):
            parsed, points, match = score_completion(ids, row, first_id, bins, stop_ids, radius)
            record = {"id": row["id"], "valid": parsed.valid, "error": parsed.error, "tokens": len(ids),
                      "pred": [[round(x, 1), round(y, 1)] for x, y in points],
                      "tp": match.tp, "fp": match.fp, "fn": match.fn}
            if parsed.valid:
                record["salvaged"] = {"tp": match.tp, "fp": match.fp, "fn": match.fn, "points": len(points)}
            else:
                salvaged_points, salvaged = salvaged_match(ids, row, first_id, bins, stop_ids, radius)
                record["salvaged"] = {"tp": salvaged.tp, "fp": salvaged.fp, "fn": salvaged.fn,
                                      "points": len(salvaged_points)}
            records.append(record)
    records = dist.gather_objects(records)
    order = {row["id"]: index for index, row in enumerate(rows)}
    records.sort(key=lambda record: order[record["id"]])
    matches = [Match(r["tp"], r["fp"], r["fn"]) for r in records]
    invalid = sum(not r["valid"] for r in records)
    salvaged = aggregate([Match(r["salvaged"]["tp"], r["salvaged"]["fp"], r["salvaged"]["fn"]) for r in records])
    errors = {}
    for record in records:
        if record["error"]:
            errors[record["error"]] = errors.get(record["error"], 0) + 1
    summary = {**aggregate(matches), "invalid": invalid, "errors": errors, "radius": radius,
               # Diagnostic only: malformed/truncated outputs re-read up to their first raster-order break.
               "salvaged_micro_f1": salvaged["micro_f1"], "salvaged_micro_precision": salvaged["micro_precision"],
               "salvaged_micro_recall": salvaged["micro_recall"],
               "seconds": round(time.time() - started, 1)}
    if output is not None and dist.is_main():
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if keep_predictions:
            with (output / "predictions.jsonl").open("w") as stream:
                stream.writelines(json.dumps(record) + "\n" for record in records)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="A NuNext SFT/GRPO output folder (model + processor)")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    from nucleus_rl.nunext import dist
    from nucleus_rl.nunext.modeling import coordinate_first_id, load_policy

    device = dist.setup()   # sharded across GPUs when started by nucleus_rl.nunext.launch
    model, processor, _ = load_policy(args.model, "main", dtype="bfloat16", device=device, trainable=False)
    first_id = coordinate_first_id(processor.tokenizer, args.bins)
    rows = read_manifests(args.manifest, args.limit)
    summary = evaluate(model, processor, rows, first_id=first_id, bins=args.bins, image_size=args.image_size,
                       radius=args.radius, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                       output=Path(args.output))
    if dist.is_main():
        print(json.dumps(summary, indent=2), flush=True)
    dist.teardown()


if __name__ == "__main__":
    main()
