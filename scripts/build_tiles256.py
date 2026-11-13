#!/usr/bin/env python3
"""Build trustmix256: 256x256 grid tiles with ALL visible nuclei from the trustmix_v2 source images and splits.

    python scripts/build_tiles256.py build --output data/trustmix256_v1 \\
        --monuseg-archive <MoNuSeg_2018_Training_data.zip> --monuseg-reference data/monuseg_official30_grpo_v1 \\
        --consep-archive <consep.zip> --consep-reference data/consep_v2 \\
        --puma-root <dir with 01_training_dataset_tif_ROIs, 01_training_dataset_geojson_nuclei> \\
        --puma-reference data/puma_v2 --trustmix-reference data/trustmix_v2
    python scripts/build_tiles256.py verify data/trustmix256_v1

Same sources, same splits. The source images and their train/validation group assignment are read from the folders
trustmix_v2 was built from (MoNuSeg: frozen_protocol.json patient_groups and source_member_hashes.json; CoNSeP and PUMA:
preparation_report.json group_splits). Labels are rasterized by the same functions (prepare_monuseg.rasterize_regions,
prepare_external.consep_sources / puma_sources), and `build` checks that every trustmix_v2 128 px crop lying inside a
256 tile has identical image pixels and identical instance IDs there. Official test sets are never opened; PUMA held-out
groups are never read (puma_sources skips them).

Tiles. Non-overlapping 256x256 grid from (0, 0) (the 128 px grid of the old crops nests inside it); right/bottom tails
that do not fill a tile are dropped (recorded per source). Every tile with >= 1 visible nucleus is a candidate; there
is NO upper count limit and no 1-8 selection. Partial nuclei are kept and listed in truncated_instance_ids;
edge_sliver_instance_ids lists the truncated ones with < 100 visible pixels.

Splits.
- train.jsonl: every train tile of MoNuSeg and CoNSeP; PUMA train tiles subsampled (seeded) to --puma-train-tiles, so the
  PUMA tile share matches trustmix_v2 (512 of 1146 tiles). --max-objects is checked against the densest train tile.
- validation.jsonl: up to --val-per-dataset tiles per dataset from the validation groups only, stratified by nucleus count
  (sort by count, cut into equal bins, one seeded pick per bin). validation_quick.jsonl: every other tile of that list in
  count order (nested, still count-stratified). Rows are interleaved across datasets so an --eval-limit prefix is mixed.
- canary_train.jsonl: the 4 densest train tiles plus 8 seeded train tiles (a memory/step-time canary; train rows only).
The report holds count histograms and edge-sliver statistics for these tiles, for the 128 px grid of the same source
images, and for the trustmix_v2 128 px crops themselves.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TILE = 256
SLIVER_PIXELS = 100
SEED = 20261105
# trustmix_v2 train tiles: MoNuSeg 256, CoNSeP 378, PUMA 512 (cap). PUMA share kept: 512 / (256 + 378).
PUMA_SHARE = 512 / (256 + 378)
DATASETS = ('monuseg', 'consep', 'puma')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    return sha256(Path(path).read_bytes())


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tile_stats(labels, size: int):
    """Every non-overlapping size x size grid cell with >= 1 visible nucleus: (x, y, ids, areas, truncated, border)."""
    import numpy as np

    height, width = labels.shape
    full = np.bincount(labels.ravel())
    out = []
    for y in range(0, height - size + 1, size):
        for x in range(0, width - size + 1, size):
            crop = labels[y:y + size, x:x + size]
            ids, areas = np.unique(crop[crop > 0], return_counts=True)
            if not len(ids):
                continue
            truncated = [int(i) for i, a in zip(ids, areas) if a != full[i]]
            border = sorted({int(i) for i in np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1])) if i})
            out.append((x, y, [int(i) for i in ids], [int(a) for a in areas], truncated, border))
    return out


def sliver_ids(ids, areas, truncated) -> list[int]:
    cut = set(truncated)
    return [i for i, a in zip(ids, areas) if i in cut and a < SLIVER_PIXELS]


def gt_summary(entries) -> dict:
    """entries: iterable of (count, truncated_count, sliver_count)."""
    entries = list(entries)
    counts = sorted(e[0] for e in entries)
    nuclei = sum(counts)

    def pct(q):
        return counts[min(len(counts) - 1, int(q * (len(counts) - 1) + 0.5))] if counts else None
    return {'tiles': len(entries), 'nuclei': nuclei,
            'count_min': counts[0] if counts else None, 'count_p10': pct(.10), 'count_p25': pct(.25),
            'count_median': pct(.5), 'count_p75': pct(.75), 'count_p90': pct(.90), 'count_p95': pct(.95),
            'count_p99': pct(.99), 'count_max': counts[-1] if counts else None,
            'count_mean': nuclei / len(counts) if counts else None,
            'truncated': sum(e[1] for e in entries), 'edge_slivers': sum(e[2] for e in entries),
            'truncated_fraction': sum(e[1] for e in entries) / nuclei if nuclei else None,
            'edge_sliver_fraction': sum(e[2] for e in entries) / nuclei if nuclei else None}


def histogram(counts, edges=(1, 9, 17, 33, 49, 65, 97, 129, 177, 257, 10 ** 9)) -> dict:
    out = {}
    for low, high in zip(edges[:-1], edges[1:]):
        label = f'{low}-{high - 1}' if high < 10 ** 9 else f'>={low}'
        out[label] = sum(low <= c < high for c in counts)
    return out


# ---------------------------------------------------------------- sources

def monuseg_sources(archive_path, reference):
    """(source_id, group, split, image, labels, meta, forbidden_ids) for the trustmix MoNuSeg train/val patients."""
    import numpy as np
    from PIL import Image
    from scripts.audit_monuseg import parse_regions
    from scripts.prepare_monuseg import rasterize_regions

    reference = Path(reference)
    protocol = json.loads((reference / 'frozen_protocol.json').read_text())
    members = json.loads((reference / 'source_member_hashes.json').read_text())
    if 'test' in Path(archive_path).name.lower():
        raise ValueError('Only the MoNuSeg training archive is permitted')
    if file_sha256(archive_path) != protocol['source_archive_sha256']:
        raise ValueError('MoNuSeg archive differs from the one monuseg_official30_grpo_v1 was frozen from')
    split_of = {p: s for s, ps in protocol['patient_groups'].items() for p in ps}
    with ZipFile(archive_path) as archive:
        for source_id in sorted(members):
            patient = '-'.join(source_id.split('-')[:3])
            if patient not in split_of:
                continue  # excluded canonical images (negative/degenerate regions), as in the 128 px dataset
            entry = members[source_id]
            image_bytes, xml_bytes = archive.read(entry['image_member']), archive.read(entry['xml_member'])
            if sha256(image_bytes) != entry['image_sha256'] or sha256(xml_bytes) != entry['xml_sha256']:
                raise ValueError(f'{source_id}: archive member hash differs from source_member_hashes.json')
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            labels, log = rasterize_regions(parse_regions(xml_bytes), image.width, image.height)
            forbidden = {r['instance_id'] for r in log if r['clipped_to_image']}
            meta = {'source_image_member': entry['image_member'], 'source_xml_member': entry['xml_member'],
                    'source_image_sha256': entry['image_sha256'], 'source_xml_sha256': entry['xml_sha256']}
            yield source_id, patient, split_of[patient], image, labels.astype(np.int32), meta, forbidden


def group_splits(reference) -> dict[str, str]:
    report = json.loads((Path(reference) / 'preparation_report.json').read_text())
    return {g: s for s, gs in report['group_splits'].items() for g in gs}


def external_sources(dataset, source, reference):
    from scripts.prepare_external import consep_sources, puma_groups, puma_sources

    split_of = group_splits(reference)
    if dataset == 'consep':
        if 'test' in Path(source).name.lower():
            raise ValueError('Refusing a test archive')
        iterator = consep_sources(source)
    else:
        groups = puma_groups(source)
        keep = {s for g, ids in groups.items() if split_of[g] in ('train', 'validation') for s in ids}
        iterator = puma_sources(source, keep)
    for source_id, group, image, labels, meta in iterator:
        split = split_of[group]
        if split not in ('train', 'validation'):
            raise ValueError(f'{source_id}: group {group} is {split}; it must never be read')
        yield source_id, group, split, image, labels, meta, set()


# ---------------------------------------------------------------- build

def stratified(candidates, k, rng):
    """k candidates, one from each of k equal bins of the count-sorted list (random tie order)."""
    if len(candidates) <= k:
        return list(candidates)
    keyed = sorted(candidates, key=lambda c: (c['instance_count'], rng.random()))
    n, chosen = len(keyed), []
    for b in range(k):
        lo, hi = (b * n) // k, ((b + 1) * n) // k
        chosen.append(keyed[rng.randrange(lo, hi)])
    return chosen


def interleave(groups: dict[str, list[dict]], rng) -> list[dict]:
    queues = {}
    for name in sorted(groups):
        rows = list(groups[name])
        rng.shuffle(rows)
        queues[name] = rows
    out = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                out.append(queues[name].pop(0))
    return out


def crosscheck_128(reference_rows, tiles_by_source, reference_dir) -> dict:
    """Every trustmix_v2 128 crop inside a written 256 tile: same RGB pixels and same instance IDs there."""
    import numpy as np
    from PIL import Image

    checked = outside = 0
    for row in reference_rows:
        x0, y0, x1, y1 = row['bbox_xyxy']
        tile = tiles_by_source.get(row['source_image_id'], {}).get((x0 // TILE * TILE, y0 // TILE * TILE))
        if tile is None:
            outside += 1
            continue
        image, labels = tile
        ox, oy = x0 % TILE, y0 % TILE
        old_rgb = np.asarray(Image.open(reference_dir / row['image_path']).convert('RGB'))
        old_mask = np.asarray(Image.open(reference_dir / row['mask_path']))
        if not np.array_equal(old_rgb, image[oy:oy + 128, ox:ox + 128]):
            raise ValueError(f'{row["id"]}: RGB pixels differ from the 256 tile')
        if not np.array_equal(old_mask.astype(np.int64), labels[oy:oy + 128, ox:ox + 128].astype(np.int64)):
            raise ValueError(f'{row["id"]}: instance IDs differ from the 256 tile')
        checked += 1
    return {'trustmix_v2_crops_checked_pixel_and_id_identical': checked,
            'trustmix_v2_crops_outside_256_grid': outside}


def build(args) -> dict:
    import numpy as np
    from PIL import Image
    from scripts.build_trustmix import heldout_keys

    out = Path(args.output).expanduser().resolve()
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError(f'{out} (or {partial.name}) exists; choose a new destination')
    trustmix = Path(args.trustmix_reference).expanduser().resolve()
    tm_report = json.loads((trustmix / 'preparation_report.json').read_text())
    sources = [('monuseg', monuseg_sources(args.monuseg_archive, args.monuseg_reference)),
               ('consep', external_sources('consep', args.consep_archive, args.consep_reference)),
               ('puma', external_sources('puma', args.puma_root, args.puma_reference))]
    (partial / 'images').mkdir(parents=True)
    (partial / 'masks').mkdir()
    try:
        candidates = {(d, s): [] for d in DATASETS for s in ('train', 'validation')}
        grid128 = {(d, s): [] for d in DATASETS for s in ('train', 'validation')}
        source_audit, arrays = [], {}
        for dataset, iterator in sources:
            for source_id, group, split, image, labels, meta, forbidden in iterator:
                rgb = np.asarray(image)
                cells = tile_stats(labels, TILE)
                for x, y, ids, areas, truncated, border in tile_stats(labels, 128):
                    grid128[(dataset, split)].append((len(ids), len(truncated), len(sliver_ids(ids, areas, truncated))))
                arrays[source_id] = {}
                for x, y, ids, areas, truncated, border in cells:
                    sample_id = f'{source_id}_x{x}_y{y}_s{TILE}'
                    crop_rgb, crop_mask = rgb[y:y + TILE, x:x + TILE], labels[y:y + TILE, x:x + TILE].astype(np.uint16)
                    arrays[source_id][(x, y)] = (crop_rgb, crop_mask)
                    candidates[(dataset, split)].append({
                        'id': sample_id, 'dataset': dataset, 'split': split, 'patient_id': group,
                        'source_image_id': source_id, 'bbox_xyxy': [x, y, x + TILE, y + TILE],
                        'instance_ids': ids, 'instance_count': len(ids),
                        'truncated_instance_ids': truncated, 'border_touching_instance_ids': border,
                        'edge_sliver_instance_ids': sliver_ids(ids, areas, truncated),
                        'source_clipped_instance_ids': sorted(set(ids) & forbidden), **meta})
                source_audit.append({'source_image_id': source_id, 'dataset': dataset, 'group': group, 'split': split,
                                     'width': image.width, 'height': image.height, 'tiles_with_nuclei': len(cells),
                                     'grid_cells': (image.width // TILE) * (image.height // TILE),
                                     'uncovered_tail_fraction': 1 - (image.width // TILE) * (image.height // TILE)
                                     * TILE * TILE / (image.width * image.height)})
        # --- selection
        rng = random.Random(f'{args.seed}:select')
        train, val_by_dataset, quick_by_dataset = [], {}, {}
        selection = {}
        for dataset in DATASETS:
            pool = candidates[(dataset, 'train')]
            if dataset == 'puma':
                cap = args.puma_train_tiles
                if cap is None:
                    cap = round(PUMA_SHARE * (len(candidates[('monuseg', 'train')]) + len(candidates[('consep', 'train')])))
                keep = sorted(random.Random(f'{args.seed}:puma_train').sample(range(len(pool)), min(cap, len(pool))))
                pool = [pool[i] for i in keep]
            train.extend(pool)
            selection[dataset] = {'train_available': len(candidates[(dataset, 'train')]), 'train_selected': len(pool)}
            val_pool = candidates[(dataset, 'validation')]
            chosen = stratified(val_pool, args.val_per_dataset, random.Random(f'{args.seed}:val:{dataset}'))
            chosen = sorted(chosen, key=lambda r: (r['instance_count'], r['id']))
            val_by_dataset[dataset] = chosen
            quick_by_dataset[dataset] = chosen[::2]
            selection[dataset].update(validation_available=len(val_pool), validation_selected=len(chosen),
                                      validation_quick_selected=len(chosen[::2]))
        densest = max(r['instance_count'] for r in train)
        if densest > args.max_objects:
            raise ValueError(f'Densest train tile holds {densest} nuclei > --max-objects {args.max_objects}')
        validation = interleave(val_by_dataset, random.Random(f'{args.seed}:order'))
        quick_ids = {r['id'] for rows in quick_by_dataset.values() for r in rows}
        quick = [r for r in validation if r['id'] in quick_ids]
        by_count = sorted(train, key=lambda r: (-r['instance_count'], r['id']))
        rest = [r for r in train if r['id'] not in {x['id'] for x in by_count[:4]}]
        canary = by_count[:4] + random.Random(f'{args.seed}:canary').sample(rest, 8)
        # --- leakage: groups and source images of every held-out file trustmix_v2 checked, plus our validation
        held_files = [Path(item['file']) for item in tm_report['leakage_check']['heldout_files']]
        held_groups, held_images = heldout_keys(held_files)
        val_groups = {r['patient_id'] for d in DATASETS for r in candidates[(d, 'validation')]}
        val_images = {r['source_image_id'] for d in DATASETS for r in candidates[(d, 'validation')]}
        train_groups, train_images = {r['patient_id'] for r in train}, {r['source_image_id'] for r in train}
        for name, overlap in (('groups vs trustmix held-out', train_groups & held_groups),
                              ('images vs trustmix held-out', train_images & held_images),
                              ('groups vs validation', train_groups & val_groups),
                              ('images vs validation', train_images & val_images)):
            if overlap:
                raise ValueError(f'Train overlap ({name}): {sorted(overlap)[:5]}')
        tm_val_groups = {r['patient_id'] for r in read_rows(trustmix / 'validation.jsonl')}
        if not val_groups <= held_groups or not tm_val_groups <= val_groups | {None}:
            raise ValueError('Validation groups differ from the trustmix_v2 validation/held-out groups')
        # --- write files
        written = {}
        for row in {r['id']: r for r in train + validation}.values():
            crop_rgb, crop_mask = arrays[row['source_image_id']][tuple(row['bbox_xyxy'][:2])]
            image_path, mask_path = f'images/{row["id"]}.png', f'masks/{row["id"]}.png'
            Image.fromarray(crop_rgb).save(partial / image_path)
            Image.fromarray(crop_mask).save(partial / mask_path)
            written[row['id']] = {
                'image_path': image_path, 'mask_path': mask_path, 'gt_mask': mask_path,
                'path_base': 'manifest_directory',
                'image_sha256': file_sha256(partial / image_path), 'mask_sha256': file_sha256(partial / mask_path),
                'image_pixels_sha256': sha256(np.ascontiguousarray(crop_rgb).tobytes()),
                'mask_pixels_sha256': sha256(np.ascontiguousarray(crop_mask).tobytes()),
                'instance_count_definition': 'Distinct nonzero IDs with at least one visible pixel; includes partial '
                                             'objects (no centroid-in-tile filtering, no upper limit).',
                'sampling_condition': {'crop_size': TILE, 'grid_stride': TILE, 'min_nuclei': 1, 'max_nuclei': None,
                                       'border_policy': 'visible',
                                       'edge_sliver_definition': f'truncated and < {SLIVER_PIXELS} visible pixels'}}
        finish = lambda rows: [{**row, **written[row['id']]} for row in rows]  # noqa: E731
        files = {'train.jsonl': finish(train), 'validation.jsonl': finish(validation),
                 'validation_quick.jsonl': finish(quick), 'canary_train.jsonl': finish(canary)}
        for name, rows in files.items():
            write_rows(partial / name, rows)
        write_rows(partial / 'manifest.jsonl', files['train.jsonl'] + files['validation.jsonl'])
        # --- checks against trustmix_v2 crops
        tm_rows = read_rows(trustmix / 'train.jsonl') + read_rows(trustmix / 'validation.jsonl')
        cross = crosscheck_128(tm_rows, arrays, trustmix)
        tm_stats = {}
        for split in ('train', 'validation'):
            for dataset in DATASETS:
                entries = []
                for row in read_rows(trustmix / f'{split}.jsonl'):
                    if row['dataset'] != dataset:
                        continue
                    mask = np.asarray(Image.open(trustmix / row['mask_path']))
                    ids, areas = np.unique(mask[mask > 0], return_counts=True)
                    entries.append((len(ids), len(row['truncated_instance_ids']),
                                    len(sliver_ids(ids.tolist(), areas.tolist(), row['truncated_instance_ids']))))
                tm_stats[f'{dataset}/{split}'] = gt_summary(entries)
        stats = {}
        for (dataset, split), pool in candidates.items():
            key = f'{dataset}/{split}'
            entry = lambda r: (r['instance_count'], len(r['truncated_instance_ids']), len(r['edge_sliver_instance_ids']))  # noqa: E731
            stats[key] = {
                'tiles256_all_candidates': gt_summary(map(entry, pool)),
                'tiles256_all_candidates_histogram': histogram([r['instance_count'] for r in pool]),
                'tiles256_selected': gt_summary(map(entry, [r for r in (train if split == 'train' else validation)
                                                            if r['dataset'] == dataset])),
                'grid128_all_cells_with_nuclei': gt_summary(grid128[(dataset, split)]),
                'grid128_cells_1_to_8': gt_summary([e for e in grid128[(dataset, split)] if e[0] <= 8]),
                'trustmix_v2_selected_128_crops': tm_stats[key]}
        pooled = {}
        for name, rows in (('train', train), ('validation', validation), ('validation_quick', quick)):
            counts = [r['instance_count'] for r in rows]
            pooled[name] = {**gt_summary((r['instance_count'], len(r['truncated_instance_ids']),
                                          len(r['edge_sliver_instance_ids'])) for r in rows),
                            'histogram': histogram(counts),
                            'by_dataset': dict(Counter(r['dataset'] for r in rows)),
                            'groups': len({r['patient_id'] for r in rows})}
        sft_audit = audit(partial, args.max_objects)
    except BaseException:
        shutil.rmtree(partial)
        raise
    outputs = {p.name: {'rows': len(p.read_text().splitlines()), 'sha256': file_sha256(p)}
               for p in sorted(partial.glob('*.jsonl'))}
    report = {'schema_version': 1, 'kind': 'tiles256', 'seed': args.seed, 'tile': TILE,
              'max_objects': args.max_objects, 'densest_train_tile': densest,
              'sources': {'monuseg_archive': str(args.monuseg_archive), 'monuseg_reference': str(args.monuseg_reference),
                          'consep_archive': str(args.consep_archive), 'consep_reference': str(args.consep_reference),
                          'puma_root': str(args.puma_root), 'puma_reference': str(args.puma_reference),
                          'trustmix_reference': str(trustmix)},
              'official_test_accessed': False, 'puma_heldout_groups_read': False,
              'selection': selection, 'pooled': pooled, 'per_dataset_split': stats,
              'crosscheck_vs_trustmix_v2': cross,
              'leakage_check': {'heldout_files_from_trustmix_v2_report': [str(p) for p in held_files],
                                'heldout_groups': len(held_groups), 'heldout_source_images': len(held_images),
                                'train_groups': len(train_groups), 'validation_groups': len(val_groups),
                                'overlap': 0},
              'source_audit': source_audit, 'sft_split_audit': sft_audit, 'outputs': outputs,
              'usage': {'sft --manifest': 'train.jsonl', 'sft --eval-manifest': 'validation.jsonl or '
                        'validation_quick.jsonl', 'max_objects': args.max_objects}}
    (partial / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    partial.rename(out)
    return report


def audit(folder: Path, max_objects: int) -> dict:
    """The split audit sft.py runs before loading a model (validation tiles are never capped)."""
    import math
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.sft_data import validate_sft_splits

    train = read_manifest(folder / 'train.jsonl')
    result = {}
    for name in ('validation.jsonl', 'validation_quick.jsonl'):
        found = validate_sft_splits(train, read_manifest(folder / name), verify_files=True,
                                    max_objects=max_objects, validation_max_objects=math.inf)
        result[name] = {k: v for k, v in found.items() if k != 'all_files_sha256'}
    canary = read_manifest(folder / 'canary_train.jsonl')
    if not {r['id'] for r in canary} <= {r['id'] for r in train}:
        raise ValueError('canary rows must be train rows')
    return result


def verify(folder) -> dict:
    folder = Path(folder).expanduser().resolve()
    report = json.loads((folder / 'preparation_report.json').read_text())
    for name, entry in report['outputs'].items():
        if file_sha256(folder / name) != entry['sha256']:
            raise ValueError(f'{name} differs from its recorded sha256')
    checked = 0
    for name in report['outputs']:
        for row in read_rows(folder / name):
            for key in ('image', 'mask'):
                if file_sha256(folder / row[f'{key}_path']) != row[f'{key}_sha256']:
                    raise ValueError(f'{row["id"]}: {key} file changed')
            checked += 1
    return {'status': 'verified', 'folder': str(folder), 'rows_checked': checked,
            'audit': audit(folder, report['max_objects'])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--output', required=True)
    p.add_argument('--monuseg-archive', required=True)
    p.add_argument('--monuseg-reference', required=True)
    p.add_argument('--consep-archive', required=True)
    p.add_argument('--consep-reference', required=True)
    p.add_argument('--puma-root', required=True)
    p.add_argument('--puma-reference', required=True)
    p.add_argument('--trustmix-reference', required=True)
    p.add_argument('--puma-train-tiles', type=int, help='Default keeps the trustmix_v2 PUMA tile share')
    p.add_argument('--val-per-dataset', type=int, default=32)
    p.add_argument('--max-objects', type=int, default=176)
    p.add_argument('--seed', type=int, default=SEED)
    p = sub.add_parser('verify')
    p.add_argument('folder')
    args = parser.parse_args(argv)
    if args.command == 'build':
        report = build(args)
        print(json.dumps({'output': args.output, 'selection': report['selection'], 'pooled': report['pooled'],
                          'crosscheck': report['crosscheck_vs_trustmix_v2']}, indent=2))
    else:
        print(json.dumps(verify(args.folder), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
