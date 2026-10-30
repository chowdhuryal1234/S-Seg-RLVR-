#!/usr/bin/env python3
"""Whole-image evaluation set: every validation source image cut into overlapping 128 px tiles.

    python scripts/prepare_stitch_eval.py monuseg --source MoNuSeg_2018_Training_data.zip \
        --split-from data/monuseg_official30_grpo_v1 --output data/monuseg_stitch_v1
    python scripts/prepare_stitch_eval.py consep --source consep.zip --split-from data/consep_v1 --output ...
    python scripts/prepare_stitch_eval.py puma --source <PUMA dir> --split-from data/puma_v1 --output ... --max-images 6

Writes tiles.jsonl (validation rows the normal --eval-only runner scores, one per tile, each with a `stitch` record
of its source image and offset) and whole.jsonl (one row per source image with its full image and full reference
mask). Only the validation source images listed in --split-from's preparation report are read. Tiles may hold no
nuclei or more than 32: they are marked `stitch` and only the stitched whole-image score counts
(scripts/stitch_eval.py). Tile grid: stride --stride (default 96), the last row and column aligned to the image edge.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_external import consep_sources, puma_sources


CROP = 128


def positions(length, crop, stride):
    starts = list(range(0, max(1, length - crop + 1), stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


def monuseg_sources(archive, keep_ids):
    from PIL import Image
    from scripts.prepare_monuseg import audit_archive, parse_regions, rasterize_regions

    audit = audit_archive(Path(archive), expected_images=30)
    with ZipFile(archive) as zf:
        for pair in audit['pairs']:
            if pair['id'] not in keep_ids:
                continue
            with Image.open(io.BytesIO(zf.read(pair['image_member']))) as source:
                image = source.convert('RGB')
            labels, _ = rasterize_regions(parse_regions(zf.read(pair['xml_member'])), *image.size)
            yield pair['id'], pair['patient_id'], image, labels, {'source_image_member': pair['image_member'],
                                                                  'source_label_member': pair['xml_member']}


def validation_sources(dataset, split_from):
    report = json.loads((Path(split_from) / 'preparation_report.json').read_text())
    rows = [json.loads(line) for line in (Path(split_from) / 'validation.jsonl').read_text().splitlines() if line]
    ids = sorted({row['source_image_id'] for row in rows})
    if dataset == 'monuseg':
        groups = set(report['patient_groups']['validation'])
        if {row['patient_id'] for row in rows} - groups:
            raise ValueError('Validation rows outside the validation patients')
    return ids


def main(argv=None):
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=('monuseg', 'consep', 'puma'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--split-from', required=True, help='Dataset folder whose validation images to use')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stride', type=int, default=96)
    parser.add_argument('--max-images', type=int, help='Use only the first N validation images (sorted IDs)')
    args = parser.parse_args(argv)
    if 'test' in Path(args.source).name.lower():
        parser.error('Refusing a test archive')
    out = Path(args.output).resolve()
    if out.exists():
        parser.error(f'{out} exists; choose a new destination')
    keep = validation_sources(args.dataset, args.split_from)[:args.max_images]
    if args.dataset == 'monuseg':
        sources = monuseg_sources(args.source, set(keep))
    elif args.dataset == 'consep':
        sources = (s for s in consep_sources(args.source) if s[0] in keep)
    else:
        sources = puma_sources(args.source, set(keep))
    (out / 'tiles').mkdir(parents=True)
    (out / 'whole').mkdir()
    tiles, wholes = [], []
    for source_id, group, image, labels, meta in sources:
        width, height = image.size
        image.save(out / 'whole' / f'{source_id}.png')
        Image.fromarray(labels.astype(np.uint16)).save(out / 'whole' / f'{source_id}_mask.png')
        xs, ys = positions(width, CROP, args.stride), positions(height, CROP, args.stride)
        wholes.append({'id': source_id, 'dataset': args.dataset, 'patient_id': group, 'width': width,
                       'height': height, 'image_path': f'whole/{source_id}.png',
                       'mask_path': f'whole/{source_id}_mask.png', 'instances': int(len(np.unique(labels)) - 1),
                       'tile_x': xs, 'tile_y': ys, 'crop': CROP, 'stride': args.stride, **meta})
        for y in ys:
            for x in xs:
                tile_id = f'{source_id}_x{x}_y{y}'
                pixels = image.crop((x, y, x + CROP, y + CROP))
                mask = labels[y:y + CROP, x:x + CROP].astype(np.uint16)
                pixels.save(out / 'tiles' / f'{tile_id}.png')
                Image.fromarray(mask).save(out / 'tiles' / f'{tile_id}_mask.png')
                row = {'id': tile_id, 'split': 'validation', 'source_image_id': source_id, 'patient_id': group,
                       'image_path': f'tiles/{tile_id}.png', 'mask_path': f'tiles/{tile_id}_mask.png',
                       'gt_mask': f'tiles/{tile_id}_mask.png', 'path_base': 'manifest_directory',
                       'instance_count': int(len(np.unique(mask[mask > 0]))),
                       'image_pixels_sha256': hashlib.sha256(np.asarray(pixels).tobytes()).hexdigest(),
                       'stitch': {'source_image_id': source_id, 'x': x, 'y': y, 'crop': CROP}, **meta}
                if args.dataset != 'monuseg':
                    row['dataset'] = args.dataset
                tiles.append(row)
    (out / 'tiles.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tiles))
    (out / 'whole.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in wholes))
    report = {'dataset': args.dataset, 'split_from': str(Path(args.split_from)), 'images': [w['id'] for w in wholes],
              'tiles': len(tiles), 'stride': args.stride, 'crop': CROP,
              'nuclei': sum(w['instances'] for w in wholes),
              'tiles_with_more_than_32': sum(t['instance_count'] > 32 for t in tiles),
              'empty_tiles': sum(t['instance_count'] == 0 for t in tiles)}
    (out / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
