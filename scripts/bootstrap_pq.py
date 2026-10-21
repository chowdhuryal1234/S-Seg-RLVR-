#!/usr/bin/env python3
"""Paired bootstrap over the shared validation crops for the GRPO-from-SFT comparisons.

Each comparison reads two evaluations of the same crops (`completions.jsonl`, one record per crop with
`id` and `pq`), resamples the per-crop PQ differences and reports the mean with a 95% interval.
A difference is "clear" when the interval excludes zero.

    python scripts/bootstrap_pq.py --grpo-runs runs/grpo_20261020 --sft-runs runs/sft_20261017 \
        --output bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (label, (run, phase), (run, phase)); phase is "before", "after" or "evaluations/step-N".
COMPARISONS = [
    ("Resolution: 9B SFT 448 vs 224 (final)", ("h1-9b-d4-sft448", "after"), ("h200-53586843-2/9b/d4", "after")),
    ("Resolution: 27B SFT 448 vs 224 (final)", ("h2-27b-d4-sft448", "after"), ("h200-53586843-3/27b/d4", "after")),
    ("Resolution: 9B SFT 672 (H3) vs 448 (H1), final", ("h3-9b-d4-sft672", "after"), ("h1-9b-d4-sft448", "after")),
    ("224: GRPO pq (G2) vs continued SFT (C1), step 200", ("g2-9b-pq", "evaluations/step-200"), ("c1-9b-sftcont", "evaluations/step-200")),
    ("224: GRPO pq (G2) vs continued SFT (C1), step 400", ("g2-9b-pq", "evaluations/step-400"), ("c1-9b-sftcont", "evaluations/step-400")),
    ("224: GRPO pq (G2) vs continued SFT (C1), final 800", ("g2-9b-pq", "after"), ("c1-9b-sftcont", "after")),
    ("224: GRPO fg_iou (G1) vs C1, final", ("g1-9b-fgiou", "after"), ("c1-9b-sftcont", "after")),
    ("224: GRPO mentor set (G3) vs C1, final", ("g3-9b-mentor", "after"), ("c1-9b-sftcont", "after")),
    ("224: GRPO fg_iou (G1) vs GRPO pq (G2), final", ("g1-9b-fgiou", "after"), ("g2-9b-pq", "after")),
    ("27B 224: GRPO pq (G4) final vs its start", ("g4-27b-pq", "after"), ("g4-27b-pq", "before")),
    ("448: GRPO pq (G6x) vs continued SFT (C2x), final", ("g6x-9b448c128-pq", "after"), ("c2x-9b448c128-sftcont", "after")),
    ("448: GRPO pq+count (G8x) vs C2x, final", ("g8x-9b448c128-pqcount", "after"), ("c2x-9b448c128-sftcont", "after")),
    ("448: GRPO pq+count seed 43 (g8r) vs C2x, final", ("g8r-9b448c128-pqcount-seed43", "after"), ("c2x-9b448c128-sftcont", "after")),
    ("27B 448: GRPO pq (G7) final vs its start", ("g7-27b448c128-pq", "after"), ("g7-27b448c128-pq", "before")),
    ("27B 448: G7 final vs SFT final 512 (H2)", ("g7-27b448c128-pq", "after"), ("h2-27b-d4-sft448", "after")),
    ("27B 448: G7 final vs SFT step 384 (H2 best)", ("g7-27b448c128-pq", "after"), ("h2-27b-d4-sft448", "evaluations/step-384")),
]


def per_crop(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {record["id"]: record["pq"] for record in records}


def compare(label, a_path, b_path, draws=10000, seed=0):
    a, b = per_crop(a_path), per_crop(b_path)
    ids = sorted(a)
    if sorted(b) != ids:
        raise ValueError(f"{label}: different crop sets")
    diffs = [a[i] - b[i] for i in ids]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(draws))
    low, high = boots[int(0.025 * draws)], boots[int(0.975 * draws)]
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    verdict = "clear" if low > 0 or high < 0 else "within noise"
    return {"label": label, "mean": mean, "low": low, "high": high, "better": better, "worse": worse, "verdict": verdict}


def completions_path(roots, run, phase):
    base = next((root / run for root in roots if (root / run).exists()), roots[0] / run)
    return base / phase / "completions.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grpo-runs", type=Path, required=True, help="Directory holding the GRPO-from-SFT run folders")
    parser.add_argument("--sft-runs", type=Path, required=True, help="Directory holding the 2026-10-17 SFT suite runs")
    parser.add_argument("--output", type=Path, help="Write the comparisons as JSON")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    roots = [args.grpo_runs, args.sft_runs]
    rows = []
    for label, (a_run, a_phase), (b_run, b_phase) in COMPARISONS:
        try:
            row = compare(label, completions_path(roots, a_run, a_phase), completions_path(roots, b_run, b_phase), args.draws)
        except (OSError, ValueError, KeyError) as error:
            print(f"{label:56s} skipped: {error}")
            continue
        rows.append(row)
        print(f"{label:56s} {row['mean']:+.3f}  95% CI [{row['low']:+.3f}, {row['high']:+.3f}]  "
              f"crops better/worse {row['better']}/{row['worse']}  -> {row['verdict']}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
