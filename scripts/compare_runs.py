#!/usr/bin/env python3
"""One table for any set of evaluations: PQ next to SQ, DQ, TP, FP, FN, precision and recall.

    python scripts/compare_runs.py --reference runs/h2/after runs/g7/after runs/g7r/after [--labels H2 G7 G7r]
        [--json out.json]

Each argument is an evaluation folder holding completions.jsonl (for a run folder, its after/ folder is used).
PQ, SQ and DQ are tile means, as in summary.json; TP / FP / FN are totals; precision and recall are micro
(TP / (TP + FP), TP / (TP + FN)). Every run after the reference also gets a paired bootstrap over tiles
(95 % interval) of the difference in mean PQ and in total TP, FP and FN. Runs must cover the same tile IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


FIELDS = ('pq', 'sq', 'dq', 'tp', 'fp', 'fn')


def load(path):
    path = Path(path)
    if (path / 'after' / 'completions.jsonl').is_file():
        path = path / 'after'
    rows = {}
    for line in (path / 'completions.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['id']] = {key: float(row.get(key) or 0.0) for key in FIELDS}
    if not rows:
        raise ValueError(f'No completions in {path}')
    return rows


def totals(rows, ids):
    n = len(ids)
    out = {key: sum(rows[i][key] for i in ids) / n for key in ('pq', 'sq', 'dq')}
    out.update({key: int(sum(rows[i][key] for i in ids)) for key in ('tp', 'fp', 'fn')})
    out['precision'] = out['tp'] / max(1, out['tp'] + out['fp'])
    out['recall'] = out['tp'] / max(1, out['tp'] + out['fn'])
    out['tiles'] = n
    return out


def paired_bootstrap(reference, other, ids, samples=10000, seed=0):
    rng = random.Random(seed)
    diffs = {key: [other[i][key] - reference[i][key] for i in ids] for key in ('pq', 'tp', 'fp', 'fn')}
    n = len(ids)
    draws = {key: [] for key in diffs}
    for _ in range(samples):
        pick = [rng.randrange(n) for _ in range(n)]
        draws['pq'].append(sum(diffs['pq'][j] for j in pick) / n)
        for key in ('tp', 'fp', 'fn'):
            draws[key].append(sum(diffs[key][j] for j in pick))
    out = {}
    for key, values in draws.items():
        values.sort()
        point = sum(diffs[key]) / n if key == 'pq' else sum(diffs[key])
        out[key] = {'delta': point, 'ci95': [values[int(.025 * samples)], values[int(.975 * samples) - 1]]}
    return out


def compare(paths, labels=None, samples=10000):
    labels = labels or [Path(p).parent.name if Path(p).name == 'after' else Path(p).name for p in paths]
    runs = [load(p) for p in paths]
    ids = sorted(runs[0])
    for label, rows in zip(labels, runs):
        if sorted(rows) != ids:
            raise ValueError(f'{label} covers different tiles than {labels[0]}')
    table = []
    for index, (label, rows) in enumerate(zip(labels, runs)):
        entry = {'label': label, **totals(rows, ids)}
        if index:
            entry['vs_reference'] = paired_bootstrap(runs[0], rows, ids, samples)
        table.append(entry)
    return table


def markdown(table):
    lines = ['| Run | PQ | SQ | DQ | TP | FP | FN | Precision | Recall | ΔPQ vs ref [95% CI] | ΔTP | ΔFP | ΔFN |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in table:
        cells = [row['label'], f"{row['pq']:.3f}", f"{row['sq']:.3f}", f"{row['dq']:.3f}", str(row['tp']),
                 str(row['fp']), str(row['fn']), f"{row['precision']:.3f}", f"{row['recall']:.3f}"]
        ref = row.get('vs_reference')
        if ref:
            pq = ref['pq']
            cells.append(f"{pq['delta']:+.3f} [{pq['ci95'][0]:+.3f}, {pq['ci95'][1]:+.3f}]")
            cells += [f"{ref[k]['delta']:+.0f} [{ref[k]['ci95'][0]:+.0f}, {ref[k]['ci95'][1]:+.0f}]" for k in ('tp', 'fp', 'fn')]
        else:
            cells += ['reference', '', '', '']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True, help='Evaluation folder that the others are compared with')
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    paths = [args.reference, *args.runs]
    if args.labels and len(args.labels) != len(paths):
        parser.error('--labels needs one label per evaluation, reference first')
    table = compare(paths, args.labels, args.samples)
    print(markdown(table))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
