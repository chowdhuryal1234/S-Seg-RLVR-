"""Greedy-decoding detection evaluation (F1 at a distance threshold) on a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nucleus_rl.nunext.coords import bins_to_points, parse_token_ids
from nucleus_rl.nunext.data import load_example, read_manifests
from nucleus_rl.nunext.matching import aggregate, match_points


def score_completion(ids, row, first_id, bins, stop_ids, radius):
    """(parse result, match) for one generated sequence; invalid output counts as no predictions."""
    parsed = parse_token_ids(ids, first_id, bins, stop_ids)
    points = bins_to_points(parsed.bins, row["width"], row["height"], bins) if parsed.valid else []
    return parsed, points, match_points(points, row["centroids"], radius)


def evaluate(model, processor, rows, *, first_id, bins, image_size, radius, batch_size, max_new_tokens,
             output: Path | None = None, keep_predictions: bool = True) -> dict:
    from nucleus_rl.nunext.modeling import end_ids, generate

    started = time.time()
    stop_ids = end_ids(processor, model)
    model.eval()
    matches, invalid, records = [], 0, []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        examples = [load_example(row, bins, 0, image_size) for row in chunk]
        sequences = generate(model, processor, [e["image"] for e in examples], do_sample=False,
                             temperature=1.0, max_new_tokens=max_new_tokens)
        for row, ids in zip(chunk, sequences):
            parsed, points, match = score_completion(ids, row, first_id, bins, stop_ids, radius)
            invalid += not parsed.valid
            matches.append(match)
            if keep_predictions:
                records.append({"id": row["id"], "valid": parsed.valid, "error": parsed.error,
                                "pred": [[round(x, 1), round(y, 1)] for x, y in points],
                                "tp": match.tp, "fp": match.fp, "fn": match.fn})
    summary = {**aggregate(matches), "invalid": invalid, "radius": radius, "seconds": round(time.time() - started, 1)}
    if output is not None:
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
    from nucleus_rl.nunext.modeling import coordinate_first_id, load_policy

    model, processor, _ = load_policy(args.model, "main", dtype="bfloat16", trainable=False)
    first_id = coordinate_first_id(processor.tokenizer, args.bins)
    rows = read_manifests(args.manifest, args.limit)
    summary = evaluate(model, processor, rows, first_id=first_id, bins=args.bins, image_size=args.image_size,
                       radius=args.radius, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                       output=Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
