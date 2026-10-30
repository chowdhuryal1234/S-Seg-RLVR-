#!/usr/bin/env python3
"""Build 128 px crop datasets from CoNSeP and PUMA in the same manifest format as monuseg_official30_grpo_v1.

    python scripts/prepare_external.py consep --source consep.zip --output data/consep_v1
    python scripts/prepare_external.py puma --source <dir with 01_training_dataset_tif_ROIs and
        01_training_dataset_geojson_nuclei> --output data/puma_v1
    python scripts/prepare_external.py merge --inputs data/consep_v1 data/puma_v1 --output data/cleanmix_v1

Each dataset folder gets sparse crops (1-8 visible nuclei; train.jsonl, validation.jsonl) and dense validation crops
(9-32; validation_dense.jsonl), cut on the same 128 px grid with the same "visible" border policy as the MoNuSeg
datasets. Images are not resampled: CoNSeP is 0.25 um/px like MoNuSeg, PUMA 0.226 um/px.

Splits are by source group and are fixed by the seed before any crop is chosen:
- CoNSeP: the official Test folder is never opened. The 27 Train images are split by image (CoNSeP publishes no
  patient or slide IDs, so two images from the same slide can land in different splits).
- PUMA: primary_roi_k and metastatic_roi_k form one group (conservative: the release does not say whether they share
  a patient). Groups are split train / validation / held-out; held-out images are listed in the report and never read.
CoNSeP .mat labels need scipy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_monuseg import candidate_crops, select_balanced


SEED = 20261030
SPARSE, DENSE = (1, 8), (9, 32)
CROP = 128


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def split_groups(groups, fractions, seed):
    """Shuffle sorted group keys with the seed and cut them by the given fractions (last split takes the rest)."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    start, out = 0, {}
    for name, fraction in fractions[:-1]:
        count = max(1, round(fraction * len(keys)))
        out[name] = sorted(keys[start:start + count])
        start += count
    out[fractions[-1][0]] = sorted(keys[start:])
    return out


def consep_sources(archive):
    """(source_id, group, image, labels, meta) for CoNSeP Train only."""
    import numpy as np
    from PIL import Image
    import scipy.io as sio

    with ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if '__MACOSX' not in n]
        images = sorted(n for n in names if '/Train/Images/' in n and n.endswith('.png'))
        if not images:
            raise ValueError(f'No CoNSeP Train images in {archive}')
        for member in images:
            stem = Path(member).stem
            label_member = member.replace('/Images/', '/Labels/').replace('.png', '.mat')
            image_bytes, label_bytes = zf.read(member), zf.read(label_member)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            labels = sio.loadmat(io.BytesIO(label_bytes))['inst_map'].astype(np.int32)
            yield (f'consep_{stem}', f'consep_{stem}', image, labels,
                   {'source_image_member': member, 'source_label_member': label_member,
                    'source_image_sha256': sha256_bytes(image_bytes), 'source_label_sha256': sha256_bytes(label_bytes),
                    'um_per_pixel': 0.25})


def consep_groups(archive):
    with ZipFile(archive) as zf:
        stems = sorted(Path(n).stem for n in zf.namelist()
                       if '__MACOSX' not in n and '/Train/Images/' in n and n.endswith('.png'))
    return {f'consep_{stem}': [f'consep_{stem}'] for stem in stems}


def puma_paths(root):
    root = Path(root)
    rois = sorted(p for p in (root / '01_training_dataset_tif_ROIs').iterdir() if p.suffix in ('.tif', '.tiff'))
    if not rois:
        raise ValueError(f'No PUMA ROIs under {root}')
    return rois


def puma_group(stem):
    match = re.fullmatch(r'training_set_(primary|metastatic)_roi_(\d+)', stem)
    if not match:
        raise ValueError(f'Unexpected PUMA ROI name: {stem}')
    return f'puma_roi_{match.group(2)}'


def puma_groups(root):
    groups = {}
    for path in puma_paths(root):
        groups.setdefault(puma_group(path.stem), []).append('puma_' + path.stem.removeprefix('training_set_'))
    return groups


def rasterize_geojson(features, size):
    """Instance map from GeoJSON polygons: Pillow inclusive fill, vertices floor(x+0.5), earliest feature wins
    (the MoNuSeg convention). Holes are ignored; MultiPolygon parts share one ID."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new('I', size, 0)
    draw = ImageDraw.Draw(canvas)
    for index, feature in reversed(list(enumerate(features, start=1))):
        geometry = feature['geometry']
        rings = ([geometry['coordinates'][0]] if geometry['type'] == 'Polygon' else
                 [polygon[0] for polygon in geometry['coordinates']] if geometry['type'] == 'MultiPolygon' else [])
        for ring in rings:
            points = [(int(np.floor(x + 0.5)), int(np.floor(y + 0.5))) for x, y in ring]
            if len(set(points)) >= 3:
                draw.polygon(points, fill=index, outline=index)
    return np.asarray(canvas).astype(np.int32)


def puma_sources(root, keep_ids):
    from PIL import Image

    root = Path(root)
    for path in puma_paths(root):
        source_id = 'puma_' + path.stem.removeprefix('training_set_')
        if source_id not in keep_ids:
            continue
        geojson = root / '01_training_dataset_geojson_nuclei' / f'{path.stem}_nuclei.geojson'
        image_bytes, label_bytes = path.read_bytes(), geojson.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert('RGB')
            resolution = source.tag_v2.get(282)
        features = json.loads(label_bytes)
        features = features['features'] if isinstance(features, dict) else features
        labels = rasterize_geojson(features, image.size)
        yield (source_id, puma_group(path.stem), image, labels,
               {'source_image_member': path.name, 'source_label_member': geojson.name,
                'source_image_sha256': sha256_bytes(image_bytes), 'source_label_sha256': sha256_bytes(label_bytes),
                'um_per_pixel': round(1e4 / float(resolution), 4) if resolution else None})


def crop_rows(sources, split_of, out, dataset, args):
    """Cut every source image, select sparse train/validation and dense validation crops, write PNGs."""
    import numpy as np
    from PIL import Image

    pools = {('train', SPARSE): [], ('validation', SPARSE): [], ('validation', DENSE): []}
    cache, audits = {}, []
    for source_id, group, image, labels, meta in sources:
        split = split_of[group]
        audit = {'id': source_id, 'group': group, 'split': split, 'width': image.size[0], 'height': image.size[1],
                 'instances': int(len(np.unique(labels)) - 1), **meta}
        for (pool_split, (low, high)), pool in pools.items():
            if pool_split != split:
                continue
            found = candidate_crops(labels, CROP, high, (), 'visible', low)
            for crop in found:
                crop.update(source_image_id=source_id, patient_id=group, split=split, **meta)
            pool.extend(found)
            audit[f'eligible_{low}_{high}'] = len(found)
        cache[source_id] = (image, labels)
        audits.append(audit)
    wanted = {('train', SPARSE): args.train_crops, ('validation', SPARSE): args.val_crops,
              ('validation', DENSE): args.dense_val_crops}
    chosen = {key: select_balanced(pool, min(wanted[key], len(pool)), args.seed) for key, pool in pools.items()}
    (out / 'images').mkdir(parents=True)
    (out / 'masks').mkdir()
    written, files = {}, {}
    for (split, (low, high)), crops in chosen.items():
        rows = []
        for crop in crops:
            x0, y0, x1, y1 = crop['bbox_xyxy']
            sample_id = f"{crop['source_image_id']}_x{x0}_y{y0}"
            image, labels = cache[crop['source_image_id']]
            if sample_id not in written:
                pixels = image.crop((x0, y0, x1, y1))
                mask = labels[y0:y1, x0:x1].astype(np.uint16)
                pixels.save(out / 'images' / f'{sample_id}.png')
                Image.fromarray(mask).save(out / 'masks' / f'{sample_id}.png')
                written[sample_id] = {'image_pixels_sha256': hashlib.sha256(np.asarray(pixels).tobytes()).hexdigest(),
                                      'mask_pixels_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
            rows.append({**crop, 'id': sample_id, 'dataset': dataset,
                         'image_path': f'images/{sample_id}.png', 'mask_path': f'masks/{sample_id}.png',
                         'gt_mask': f'masks/{sample_id}.png', 'path_base': 'manifest_directory',
                         **written[sample_id],
                         'instance_count_definition': 'Distinct nonzero IDs with at least one visible pixel; '
                                                      'includes partial objects.',
                         'sampling_condition': {'crop_size': CROP, 'grid_stride': CROP, 'min_nuclei': low,
                                                'max_nuclei': high, 'border_policy': 'visible'}})
        name = {('train', SPARSE): 'train', ('validation', SPARSE): 'validation',
                ('validation', DENSE): 'validation_dense'}[(split, (low, high))]
        write_rows(out / f'{name}.jsonl', rows)
        files[name] = rows
    write_rows(out / 'manifest.jsonl', files['train'] + files['validation'])
    return files, {f'{s}_{lo}_{hi}': len(p) for (s, (lo, hi)), p in pools.items()}, audits


def build(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new destination')
    if args.dataset == 'consep':
        if 'test' in Path(args.source).name.lower():
            raise ValueError('Refusing a test archive')
        groups = consep_groups(args.source)
        splits = split_groups(groups, [('validation', args.val_fraction), ('train', None)], args.seed)
        heldout = []
    else:
        groups = puma_groups(args.source)
        splits = split_groups(groups, [('validation', args.val_fraction), ('heldout', args.heldout_fraction),
                                       ('train', None)], args.seed)
        heldout = sorted(s for g in splits['heldout'] for s in groups[g])
    split_of = {group: split for split, keys in splits.items() for group in keys}
    keep = {s for group, ids in groups.items() if split_of[group] != 'heldout' for s in ids}
    sources = (consep_sources(args.source) if args.dataset == 'consep' else puma_sources(args.source, keep))
    files, candidates, audits = crop_rows(sources, split_of, out, args.dataset, args)
    report = {'schema_version': 1, 'dataset': args.dataset, 'seed': args.seed, 'source': str(Path(args.source)),
              'crop_size': CROP, 'sparse_nuclei': SPARSE, 'dense_nuclei': DENSE, 'border_policy': 'visible',
              'group_splits': splits, 'heldout_source_images_not_read': heldout,
              'official_test_accessed': False,
              'selected_crops': {name: len(rows) for name, rows in files.items()},
              'candidate_crops': candidates, 'source_images': audits,
              'nuclei': {name: sum(r['instance_count'] for r in rows) for name, rows in files.items()},
              'caveats': ['CoNSeP splits are by image: no patient or slide IDs are published.'
                          if args.dataset == 'consep' else
                          'PUMA groups pair primary_roi_k with metastatic_roi_k; patient identity is not published.']}
    (out / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def merge(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new destination')
    out.mkdir(parents=True)
    names = ('train', 'validation', 'validation_dense')
    merged = {name: [] for name in names}
    for folder in map(Path, args.inputs):
        for name in names:
            for line in (folder / f'{name}.jsonl').read_text().splitlines():
                row = json.loads(line)
                for key in ('image_path', 'mask_path', 'gt_mask'):
                    row[key] = (Path('..') / folder.name / row[key]).as_posix()
                merged[name].append(row)
    for name, rows in merged.items():
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate ids in merged {name}')
        write_rows(out / f'{name}.jsonl', rows)
    write_rows(out / 'manifest.jsonl', merged['train'] + merged['validation'])
    report = {'inputs': [Path(p).name for p in args.inputs], 'paths': 'relative to this folder, pointing into the inputs',
              'rows': {name: len(rows) for name, rows in merged.items()}}
    (out / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='dataset', required=True)
    for name in ('consep', 'puma'):
        p = sub.add_parser(name)
        p.add_argument('--source', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--train-crops', type=int, default=256)
        p.add_argument('--val-crops', type=int, default=64)
        p.add_argument('--dense-val-crops', type=int, default=64)
        p.add_argument('--val-fraction', type=float, default=0.2 if name == 'consep' else 0.15)
        p.add_argument('--heldout-fraction', type=float, default=0.15)
        p.add_argument('--seed', type=int, default=SEED)
    p = sub.add_parser('merge')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = merge(args) if args.dataset == 'merge' else build(args)
    summary = report.get('selected_crops', report.get('rows'))
    print(json.dumps({'output': args.output, 'crops': summary, 'nuclei': report.get('nuclei')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
