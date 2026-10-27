#!/usr/bin/env python3
"""Add dense 128 px crops (9-32 visible nuclei) to the frozen official30 GRPO dataset.

The sparse dataset keeps only crops with 1-8 nuclei, which hold about a sixth of all
annotated nuclei. This builds a second dataset from the same patients, the same
patient split, rasterizer and grid, selecting crops with min..max nuclei instead:

    validation.jsonl    dense validation crops (validation patients only)
    train.jsonl         dense training crops
    train_mixed.jsonl   the frozen sparse training crops plus the dense ones
    validation_combined.jsonl  the frozen sparse validation crops plus the dense ones

The sparse validation set stays where it is, so old and new runs remain comparable.
No test data is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_monuseg import prepare


SEED = 20261027


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def main(args):
    root = Path(__file__).resolve().parents[1]
    base, out = (root / args.base).resolve(), (root / args.output).resolve()
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new destination')
    base_report = json.loads((base / 'preparation_report.json').read_text())
    groups = base_report['patient_groups']
    report = prepare(argparse.Namespace(
        train_archive=Path(args.train_archive), out=out, cohort='official30', crop_size=128,
        min_nuclei=args.min_nuclei, max_nuclei=args.max_nuclei, border_policy='visible',
        train_patient_ids=','.join(groups['train']), val_patient_ids=','.join(groups['validation']),
        train_patients=len(groups['train']), val_patients=len(groups['validation']),
        train_crops=args.train_crops, val_crops=args.val_crops, seed=args.seed))
    if report['patient_groups'] != groups:
        raise ValueError('Dense dataset patient groups differ from the frozen sparse dataset')

    rows = [json.loads(line) for line in (out / 'manifest.jsonl').read_text().splitlines()]
    for row in rows:
        if not args.min_nuclei <= row['instance_count'] <= args.max_nuclei:
            raise ValueError(f"{row['id']}: {row['instance_count']} nuclei outside {args.min_nuclei}-{args.max_nuclei}")
        # Centroids are GT-derived; portable manifests omit them, as in the sparse dataset.
        row.pop('visible_centroids_xy', None)
        row.pop('visible_centroid_definition', None)
        row.update(image_sha256=digest(out / row['image_path']), mask_sha256=digest(out / row['mask_path']))
    base_rows = [json.loads(line) for line in (base / 'manifest.jsonl').read_text().splitlines()]
    if {row['id'] for row in rows} & {row['id'] for row in base_rows}:
        raise ValueError('Dense crops overlap the sparse dataset')
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    relative_base = Path('..') / base.name
    sparse = {split: [{**row, **{key: (relative_base / row[key]).as_posix() for key in ('image_path', 'mask_path', 'gt_mask')}}
                      for row in base_rows if row['split'] == split] for split in ('train', 'validation')}
    for name, subset in (('manifest', rows), ('train', train), ('validation', validation),
                         ('train_mixed', sparse['train'] + train),
                         ('validation_combined', sparse['validation'] + validation)):
        write_rows(out / f'{name}.jsonl', subset)

    summary = {
        'dataset_id': out.name, 'base_dataset': base.name, 'seed': args.seed,
        'nuclei_per_crop': [args.min_nuclei, args.max_nuclei], 'patient_groups': groups,
        'selected': {'train': len(train), 'validation': len(validation), 'train_mixed': len(sparse['train']) + len(train),
                     'validation_combined': len(sparse['validation']) + len(validation)},
        'candidates': report['candidate_crop_counts'],
        'histograms': {split: dict(sorted(Counter(r['instance_count'] for r in subset).items()))
                       for split, subset in (('train', train), ('validation', validation))},
        'by_patient': {split: dict(sorted(Counter(r['patient_id'] for r in subset).items()))
                       for split, subset in (('train', train), ('validation', validation))},
        'manifest_sha256': {name: digest(out / f'{name}.jsonl') for name in ('train', 'validation', 'train_mixed', 'validation_combined', 'manifest')},
        'note': f'Crops with more than {args.max_nuclei} nuclei are still excluded; grid crops only, no overlap with the sparse dataset.',
    }
    (out / 'dense_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('selected', 'candidates', 'histograms', 'by_patient')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-archive', required=True)
    parser.add_argument('--base', default='data/monuseg_official30_grpo_v1')
    parser.add_argument('--output', default='data/monuseg_official30_dense_v1')
    parser.add_argument('--min-nuclei', type=int, default=9)
    parser.add_argument('--max-nuclei', type=int, default=32)
    parser.add_argument('--train-crops', type=int, default=256)
    parser.add_argument('--val-crops', type=int, default=64)
    parser.add_argument('--seed', type=int, default=SEED)
    main(parser.parse_args())
