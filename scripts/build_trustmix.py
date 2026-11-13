#!/usr/bin/env python3
"""Combine prepared crop datasets (MoNuSeg, CoNSeP, PUMA) into one training mix without copying images.

    python scripts/build_trustmix.py build --output data/trustmix_v2 \\
        --source monuseg_sparse=data/monuseg_official30_grpo_v1:256 \\
        --source consep_sparse=data/consep_v2 --source puma_sparse=data/puma_v2:512 \\
        --validation monuseg=data/monuseg_official30_grpo_v1 --validation consep=data/consep_v1 \\
        --validation puma=data/puma_v1 \\
        --validation-dense monuseg=data/monuseg_official30_dense_v1/validation.jsonl \\
        --validation-dense consep=data/consep_v1 --validation-dense puma=data/puma_v1 \\
        --heldout data/monuseg_stitch_v1 data/consep_stitch_v1 data/puma_stitch_v1
    python scripts/build_trustmix.py verify data/trustmix_v2
    python scripts/build_trustmix.py compare data/consep_v1/validation.jsonl data/consep_v2/validation.jsonl

Specs are NAME=PATH[:CAP]. NAME starts with the dataset (monuseg, consep or puma), optionally followed by _tag
(monuseg_sparse, puma_dense). PATH is a .jsonl file or a folder (--source reads train.jsonl, --validation
validation.jsonl, --validation-dense validation_dense.jsonl). CAP (--source only) is a row count or "all"; a source
with more rows is subsampled with a seeded RNG keyed by --seed and NAME, keeping the source's row order.

Every output row is its source row unchanged except: image_path/mask_path/gt_mask are rewritten relative to the
output folder (the `merge` convention of prepare_external.py), and a `dataset` field is appended when the source row
has none (MoNuSeg). `verify` rebuilds every source line from the output and checks it is identical.

Before anything is written, train groups (patient_id) and source images must not occur in any held-out file: every
validation input, every validation*.jsonl / tiles.jsonl / whole.jsonl / *stitch*.jsonl next to any input, and every
--heldout file or folder. Train rows must also lie in the train groups of their folder's preparation_report.json.
The written folder is then checked with nucleus_rl.sft_data.validate_sft_splits (what sft.py and grpo.py run without
--cohort-protocol); dense training rows need --max-objects 32 here and in SFT/GRPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = ('monuseg', 'consep', 'puma')
PATH_KEYS = ('image_path', 'mask_path', 'gt_mask')
SPLITS = {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'validation_dense': 'validation_dense.jsonl'}
SEED = 20261104
# nucleus_rl.sft.MAX_OBJECTS: validation crops are only scored, so they may be dense.
VALIDATION_MAX_OBJECTS = 32


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_spec(text: str, split: str, allow_cap: bool) -> dict[str, Any]:
    """NAME=PATH[:CAP] -> {name, dataset, path, cap}; a folder PATH means its <split>.jsonl."""
    name, sep, rest = text.partition('=')
    if not sep or not name or not rest:
        raise ValueError(f'Expected NAME=PATH[:CAP], got {text!r}')
    cap = None
    head, colon, tail = rest.rpartition(':')
    if allow_cap and colon and (tail.isdigit() or tail == 'all'):
        rest, cap = head, None if tail == 'all' else int(tail)
        if cap == 0:
            raise ValueError(f'Cap must be positive or "all": {text!r}')
    dataset = name.split('_')[0]
    if dataset not in DATASETS:
        raise ValueError(f'Spec name must start with one of {DATASETS}: {name!r}')
    path = Path(rest).expanduser().resolve()
    if path.is_dir():
        path = path / SPLITS[split]
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'name': name, 'dataset': dataset, 'path': path, 'cap': cap, 'split': split}


def read_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    return [(line, json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def subsample(items: list, cap: int | None, seed: int, name: str) -> list:
    """Seeded subset in source order; the RNG is keyed by name so adding a source never changes another."""
    if cap is None or cap >= len(items):
        return list(items)
    keep = sorted(random.Random(f'{seed}:{name}').sample(range(len(items)), cap))
    return [items[i] for i in keep]


def relocate(row: dict[str, Any], source_dir: Path, out: Path) -> dict[str, Any]:
    """Copy of a row whose relative file paths point from `out` into the source folder."""
    row = dict(row)
    for key in PATH_KEYS:
        if key in row and not Path(row[key]).is_absolute():
            row[key] = Path(os.path.relpath(source_dir / row[key], out)).as_posix()
    return row


def restore(row: dict[str, Any], source_dir: Path, out: Path, had_dataset: bool) -> dict[str, Any]:
    """Inverse of relocate plus the dataset tag: the source row as it was."""
    row = dict(row)
    for key in PATH_KEYS:
        if key in row and not Path(row[key]).is_absolute():
            row[key] = Path(os.path.relpath(out / row[key], source_dir)).as_posix()
    if not had_dataset:
        row.pop('dataset', None)
    return row


def tag_dataset(row: dict[str, Any], dataset: str, where: str) -> dict[str, Any]:
    if row.get('dataset', dataset) != dataset:
        raise ValueError(f'{where}: row {row.get("id")!r} has dataset {row["dataset"]!r}, spec says {dataset!r}')
    return {**row, 'dataset': dataset}


def load_spec_rows(spec: dict[str, Any], out: Path, seed: int) -> list[dict[str, Any]]:
    lines = read_lines(spec['path'])
    spec['available_rows'] = len(lines)
    chosen = subsample([row for _, row in lines], spec['cap'], seed, spec['name'])
    expected_split = 'train' if spec['split'] == 'train' else 'validation'
    rows = []
    for row in chosen:
        if row.get('split') != expected_split:
            raise ValueError(f'{spec["path"]}: row {row.get("id")!r} has split {row.get("split")!r}, '
                             f'expected {expected_split!r}')
        rows.append(tag_dataset(relocate(row, spec['path'].parent, out), spec['dataset'], str(spec['path'])))
    return rows


def heldout_files(specs: list[dict[str, Any]], extra: list[str]) -> list[Path]:
    """Validation and stitch files next to every input, every validation input, and --heldout files/folders."""
    found = {spec['path'] for spec in specs if spec['split'] != 'train'}
    for folder in {spec['path'].parent for spec in specs}:
        for path in folder.glob('*.jsonl'):
            if (path.name.startswith('validation') or path.name in ('tiles.jsonl', 'whole.jsonl')
                    or 'stitch' in path.name):
                found.add(path)
    for item in map(lambda p: Path(p).expanduser().resolve(), extra):
        if item.is_dir():
            found.update(item.glob('*.jsonl'))
        elif item.is_file():
            found.add(item)
        else:
            raise FileNotFoundError(item)
    return sorted(found)


def heldout_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    """Patient groups and source image IDs of every held-out row."""
    groups, images = set(), set()
    for path in paths:
        for _, row in read_lines(path):
            if not isinstance(row.get('patient_id'), str):
                raise ValueError(f'{path}: held-out row {row.get("id")!r} has no patient_id')
            groups.add(row['patient_id'])
            source = row.get('source_image_id') or (row.get('stitch') or {}).get('source_image_id')
            if path.name == 'whole.jsonl':
                source = source or row.get('id')
            if source:
                images.add(source)
    return groups, images


def check_group_splits(spec: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    """Train rows must lie in the train groups the source folder's preparation report declares."""
    report_path = spec['path'].parent / 'preparation_report.json'
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    splits = report.get('group_splits') or report.get('patient_groups')
    if not isinstance(splits, dict) or 'train' not in splits:
        return None
    allowed = set(splits['train'])
    outside = sorted({row['patient_id'] for row in rows} - allowed)
    if outside:
        raise ValueError(f'{spec["name"]}: train rows outside the declared train groups: {outside}')
    return str(report_path)


def check_leakage(train: list[dict[str, Any]], held_groups: set[str], held_images: set[str]) -> None:
    groups = {row['patient_id'] for row in train} & held_groups
    if groups:
        raise ValueError(f'Train/held-out group overlap: {sorted(groups)}')
    images = {row['source_image_id'] for row in train} & held_images
    if images:
        raise ValueError(f'Train/held-out source image overlap: {sorted(images)}')


def check_unique_ids(merged: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in merged.items():
        for row in rows:
            if row['id'] in seen:
                raise ValueError(f'Duplicate id {row["id"]!r} in {seen[row["id"]]} and {split}')
            seen[row['id']] = split


def counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = out.setdefault(row['dataset'], {'tiles': 0, 'nuclei': 0})
        entry['tiles'] += 1
        entry['nuclei'] += int(row['instance_count'])
    return {key: out[key] for key in sorted(out)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def audit_output(folder: Path, max_objects: int) -> dict[str, Any]:
    """Run the same split audit sft.py and grpo.py run (without --cohort-protocol) on the written files."""
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.sft_data import validate_sft_splits

    train = read_manifest(folder / 'train.jsonl')
    audits = {}
    for split in ('validation', 'validation_dense'):
        if (folder / SPLITS[split]).is_file():
            audit = validate_sft_splits(train, read_manifest(folder / SPLITS[split]), verify_files=True,
                                        max_objects=max_objects, validation_max_objects=VALIDATION_MAX_OBJECTS)
            audits[split] = {key: value for key, value in audit.items() if key != 'all_files_sha256'}
    return audits


def source_entry(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row['id'] for row in rows]
    return {'name': spec['name'], 'dataset': spec['dataset'], 'file': str(spec['path']),
            'file_sha256': file_sha256(spec['path']), 'cap': spec['cap'], 'available_rows': spec['available_rows'],
            'selected_rows': len(rows), 'nuclei': sum(int(r['instance_count']) for r in rows),
            'selected_ids_sha256': text_sha256('\n'.join(ids) + '\n'), 'selected_ids': ids}


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output).expanduser().resolve()
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError(f'{out} (or {partial.name}) exists; choose a new destination')
    specs = ([parse_spec(t, 'train', True) for t in args.source]
             + [parse_spec(t, 'validation', False) for t in args.validation]
             + [parse_spec(t, 'validation_dense', False) for t in args.validation_dense])
    names = [(spec['split'], spec['name']) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError('Spec names must be unique within each split')
    if not args.source or not args.validation:
        raise ValueError('Need at least one --source and one --validation')
    merged: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    entries: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    reports_checked = []
    for spec in specs:
        rows = load_spec_rows(spec, out, args.seed)
        if spec['split'] == 'train':
            checked = check_group_splits(spec, rows)
            if checked and checked not in reports_checked:
                reports_checked.append(checked)
        merged[spec['split']].extend(rows)
        entries[spec['split']].append(source_entry(spec, rows))
    if not merged['validation_dense']:
        del merged['validation_dense'], entries['validation_dense']
    check_unique_ids(merged)
    held = heldout_files(specs, args.heldout)
    held_groups, held_images = heldout_keys(held)
    check_leakage(merged['train'], held_groups, held_images)
    densest = max(int(row['instance_count']) for row in merged['train'])
    if densest > args.max_objects:
        raise ValueError(f'Train rows hold up to {densest} nuclei; pass --max-objects {densest} or more '
                         '(and the same --max-objects to SFT/GRPO)')

    partial.mkdir(parents=True)
    try:
        for split, rows in merged.items():
            write_rows(partial / SPLITS[split], rows)
        write_rows(partial / 'manifest.jsonl', merged['train'] + merged['validation'])
        sft_audit = audit_output(partial, args.max_objects)
    except BaseException:
        shutil.rmtree(partial)
        raise
    outputs = {path.name: {'rows': len(path.read_text().splitlines()), 'sha256': file_sha256(path)}
               for path in sorted(partial.glob('*.jsonl'))}
    report = {'schema_version': 1, 'kind': 'trustmix', 'seed': args.seed, 'max_objects': args.max_objects,
              'paths': 'image_path/mask_path/gt_mask are relative to this folder, pointing into the source folders',
              'row_contract': 'source row unchanged except relocated paths and an appended dataset field '
                              '(only where the source row had none); check with `build_trustmix.py verify`',
              'sources': entries,
              'counts': {split: counts(rows) for split, rows in merged.items()},
              'totals': {split: {'tiles': len(rows), 'nuclei': sum(int(r['instance_count']) for r in rows)}
                         for split, rows in merged.items()},
              'outputs': outputs,
              'leakage_check': {'heldout_files': [{'file': str(p), 'sha256': file_sha256(p)} for p in held],
                                'heldout_groups': len(held_groups), 'heldout_source_images': len(held_images),
                                'train_groups': len({r['patient_id'] for r in merged['train']}),
                                'group_overlap': 0, 'source_image_overlap': 0,
                                'train_group_splits_checked_against': reports_checked},
              'sft_split_audit': sft_audit,
              'usage': {'sft/grpo --manifest': 'train.jsonl', 'sft/grpo --eval-manifest': 'validation.jsonl',
                        'max_objects': args.max_objects}}
    (partial / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    partial.rename(out)
    return report


def verify(folder: str | Path) -> dict[str, Any]:
    """Rebuild every source line from the output rows; check file hashes and per-source content identity."""
    out = Path(folder).expanduser().resolve()
    report = json.loads((out / 'preparation_report.json').read_text())
    for name, entry in report['outputs'].items():
        if file_sha256(out / name) != entry['sha256']:
            raise ValueError(f'{name} differs from its recorded sha256')
    result = {}
    for split, sources in report['sources'].items():
        rows = [row for _, row in read_lines(out / SPLITS[split])]
        if len(rows) != sum(entry['selected_rows'] for entry in sources):
            raise ValueError(f'{split}: row count differs from the recorded sources')
        start, per_source = 0, {}
        for entry in sources:
            source_path = Path(entry['file'])
            if file_sha256(source_path) != entry['file_sha256']:
                raise ValueError(f'{source_path} changed since the build')
            lines = {row['id']: (line, row) for line, row in read_lines(source_path)}
            block = rows[start:start + entry['selected_rows']]
            start += entry['selected_rows']
            if [row['id'] for row in block] != entry['selected_ids']:
                raise ValueError(f'{split}/{entry["name"]}: ids or order differ from the report')
            identical = byte_identical = 0
            for row in block:
                line, original = lines[row['id']]
                if row.get('dataset') != entry['dataset']:
                    raise ValueError(f'{row["id"]}: dataset field is not {entry["dataset"]!r}')
                rebuilt = restore(row, source_path.parent, out, 'dataset' in original)
                if rebuilt != original or list(rebuilt) != list(original):
                    raise ValueError(f'{split}/{entry["name"]}: {row["id"]} differs from its source row')
                identical += 1
                byte_identical += json.dumps(rebuilt, allow_nan=False) == line
                for key in PATH_KEYS:
                    if key in row and not Path(out / row[key]).is_file():
                        raise FileNotFoundError(out / row[key])
            per_source[entry['name']] = {'rows': len(block), 'content_identical': identical,
                                         'source_line_bytes_identical': byte_identical}
        result[split] = per_source
    return {'status': 'verified', 'folder': str(out), 'splits': result}


def without_paths(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in PATH_KEYS}


def compare(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Two manifests select the same crops: same ids in the same order, same fields apart from the file paths,
    and byte-identical image and mask files."""
    sides = []
    for path in map(lambda p: Path(p).expanduser().resolve(), (first, second)):
        sides.append([(row, path.parent) for _, row in read_lines(path)])
    ids = [[row['id'] for row, _ in side] for side in sides]
    same_order = ids[0] == ids[1]
    same_ids = sorted(ids[0]) == sorted(ids[1])
    content = files = 0
    if same_ids:
        other = {row['id']: (row, base) for row, base in sides[1]}
        for row, base in sides[0]:
            twin, twin_base = other[row['id']]
            content += without_paths(row) == without_paths(twin)
            files += all(file_sha256(base / row[key]) == file_sha256(twin_base / twin[key])
                         for key in ('image_path', 'mask_path'))
    identical = same_order and content == files == len(ids[0])
    return {'first': str(first), 'second': str(second), 'rows': [len(i) for i in ids], 'same_ids': same_ids,
            'same_order': same_order, 'content_identical_except_paths': content, 'files_identical': files,
            'identical': identical}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--output', required=True)
    p.add_argument('--source', action='append', default=[], help='Training rows: NAME=PATH[:CAP]')
    p.add_argument('--validation', action='append', default=[], help='Validation rows: NAME=PATH')
    p.add_argument('--validation-dense', action='append', default=[], help='Dense validation rows: NAME=PATH')
    p.add_argument('--heldout', nargs='+', action='extend', default=[],
                   help='Extra held-out .jsonl files or folders (e.g. stitch); may be repeated')
    p.add_argument('--max-objects', type=int, default=8, help='Largest instance_count allowed in train rows')
    p.add_argument('--seed', type=int, default=SEED)
    p = sub.add_parser('verify')
    p.add_argument('folder')
    p = sub.add_parser('compare')
    p.add_argument('first')
    p.add_argument('second')
    args = parser.parse_args(argv)
    if args.command == 'build':
        report = build(args)
        print(json.dumps({'output': args.output, 'counts': report['counts'], 'totals': report['totals'],
                          'outputs': report['outputs'], 'verify': verify(args.output)['splits']}, indent=2))
        return 0
    if args.command == 'verify':
        print(json.dumps(verify(args.folder), indent=2))
        return 0
    result = compare(args.first, args.second)
    print(json.dumps(result, indent=2))
    return 0 if result['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
