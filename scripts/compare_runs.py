#!/usr/bin/env python3
"""One table for any set of evaluations: PQ next to SQ, DQ, TP, FP, FN, precision and recall.

    python scripts/compare_runs.py --reference runs/h2/after runs/g7/after runs/g7r/after [--labels H2 G7 G7r]
        [--json out.json]
    python scripts/compare_runs.py --reference runs/h2 runs/g7 --labels H2 G7 \\
        --manifest data/monuseg_official30_grpo_v1/validation.jsonl \\
        --by-dataset --per-group --exclude-edge-slivers --fn-causes

Each argument is an evaluation folder holding completions.jsonl (for a run folder, its after/ folder is used).
PQ, SQ and DQ are tile means, as in summary.json; TP / FP / FN are totals; precision and recall are micro
(TP / (TP + FP), TP / (TP + FN)). Every run after the reference also gets a paired bootstrap over tiles
(95 % interval) of the difference in mean PQ and in total TP, FP and FN. Runs must cover the same tile IDs.

--manifest names the evaluation manifest(s) the runs were scored on (several are merged by tile ID). It supplies
each tile's dataset, group and reference mask; without it the completions' own fields are used (completions have
no 'dataset' field, so a consep_ / puma_ patient_id prefix names the dataset there).

--by-dataset    the same table (and bootstrap) within each dataset; rows without a 'dataset' field are 'monuseg'.
--per-group     per group (the manifest's patient_id: a MoNuSeg patient, a CoNSeP image, a PUMA ROI pair), the
                sign of dTP and dFP vs the reference, with an exact two-sided sign test over the nonzero groups.
--exclude-edge-slivers
                rescores without the reference nuclei that are cut by the tile edge (truncated_instance_ids, or
                touching the border when the row lacks it) and have fewer than 100 visible pixels: such nuclei
                leave the FN count, and a predicted mask matching one (IoU > 0.5) counts as neither TP nor FP.
--fn-causes     splits every missed reference nucleus by the best box IoU of any predicted box
                (converted_objects): no box (<= 0.1), box near (0.1-0.5], good box but mask failed (> 0.5).
Both need --manifest (reference masks). Matching uses the run's NNNNNN.npy instance maps when they sit in the
evaluation folder (exact, the same one-to-one IoU > 0.5 matching as the stored TP); otherwise it falls back to
the raw SAM masks' recorded best-GT matches (raw_best_gt_ids / raw_best_gt_ious, before overlap resolution), so
its all-nuclei TP / FP can differ from the stored ones by a nucleus or two. Each section reports which was used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


FIELDS = ('pq', 'sq', 'dq', 'tp', 'fp', 'fn')
MATCH_IOU = 0.5
NEAR_BOX_IOU = 0.1
SLIVER_AREA = 100
CAUSES = ('no_box', 'box_near', 'good_box_mask_failed')
EXTERNAL_DATASETS = ('consep', 'puma')  # keys of nucleus_rl.cohort.EXTERNAL_DATASETS


def eval_folder(path) -> Path:
    path = Path(path)
    if (path / 'after' / 'completions.jsonl').is_file():
        path = path / 'after'
    return path


def load_records(path) -> dict[str, dict]:
    path = eval_folder(path)
    rows = {}
    for line in (path / 'completions.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['id']] = row
    if not rows:
        raise ValueError(f'No completions in {path}')
    return rows


def metrics(records: dict[str, dict]) -> dict[str, dict]:
    return {i: {key: float(row.get(key) or 0.0) for key in FIELDS} for i, row in records.items()}


def load(path):
    return metrics(load_records(path))


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


# ---------------------------------------------------------------- manifests, datasets and groups

def read_manifests(paths) -> dict[str, dict]:
    """Merge manifest rows by tile ID (first wins); relative mask paths resolve against the manifest's folder."""
    rows = {}
    for path in paths or ():
        path = Path(path).resolve()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('mask_path'):
                mask = Path(row['mask_path']).expanduser()
                row['mask_path'] = str(mask if mask.is_absolute() else path.parent / mask)
            rows.setdefault(row['id'], row)
    return rows


def row_dataset(row: dict) -> str:
    """The row's 'dataset' field; completions lack it, so an external group ID prefix (consep_ / puma_, as
    nucleus_rl.cohort.validate_external_rows requires) names the dataset; anything else is MoNuSeg."""
    if row.get('dataset'):
        return row['dataset']
    group = str(row.get('patient_id') or '')
    return next((name for name in EXTERNAL_DATASETS if group.startswith(name + '_')), 'monuseg')


def tile_meta(ids, manifest: dict[str, dict], records: dict[str, dict]) -> dict[str, dict]:
    """Dataset and group of every tile, from the manifest when one is given, else from the completions."""
    out = {}
    for i in ids:
        if manifest and i not in manifest:
            raise ValueError(f'Tile {i} is not in the manifest')
        row = manifest.get(i) or records.get(i, {})
        out[i] = {'dataset': row_dataset(row),
                  'group': row.get('patient_id') or row.get('source_image_id') or i}
    return out


def by_dataset_ids(ids, meta) -> dict[str, list[str]]:
    groups = {}
    for i in ids:
        groups.setdefault(meta[i]['dataset'], []).append(i)
    return dict(sorted(groups.items()))


def sign_test(up: int, down: int) -> float:
    """Exact two-sided binomial sign test over the nonzero differences."""
    n = up + down
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def group_signs(reference, other, ids, meta) -> dict:
    groups = {}
    for i in ids:
        info = meta[i]
        entry = groups.setdefault(info['group'], {'dataset': info['dataset'], 'tiles': 0, 'dtp': 0, 'dfp': 0})
        entry['tiles'] += 1
        entry['dtp'] += int(other[i]['tp'] - reference[i]['tp'])
        entry['dfp'] += int(other[i]['fp'] - reference[i]['fp'])
    signs = {}
    for key in ('dtp', 'dfp'):
        values = [g[key] for g in groups.values()]
        up, same, down = sum(v > 0 for v in values), sum(v == 0 for v in values), sum(v < 0 for v in values)
        signs[key] = {'up': up, 'same': same, 'down': down, 'sign_test_p': sign_test(up, down)}
    return {'groups': dict(sorted(groups.items())), 'signs': signs}


# ---------------------------------------------------------------- reference nuclei and per-object matching

def gt_objects(mask, truncated_ids=None) -> dict[int, dict]:
    """Box [x0, y0, x1, y1) (exclusive end), visible area and edge flag of every reference nucleus in a tile.

    Edge means cut by the tile edge: in truncated_ids when the manifest lists them, else touching the border.
    """
    import numpy as np
    mask = np.asarray(mask)
    height, width = mask.shape
    out = {}
    for gid in np.unique(mask[mask > 0]):
        ys, xs = np.nonzero(mask == gid)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if truncated_ids is not None:
            edge = int(gid) in truncated_ids
        else:
            edge = box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height
        out[int(gid)] = {'box': box, 'area': int(len(xs)), 'edge': bool(edge)}
    return out


def load_gt(row: dict):
    import numpy as np
    from PIL import Image
    mask = np.asarray(Image.open(row['mask_path'])).astype(np.int64)
    truncated = row.get('truncated_instance_ids')
    return mask, gt_objects(mask, None if truncated is None else {int(t) for t in truncated})


def exact_matches(pred, gt) -> list[tuple[int | None, float]]:
    """(matched reference ID or None, best IoU) for every predicted instance of a disjoint instance map."""
    import numpy as np
    pred, gt = np.asarray(pred), np.asarray(gt)
    out = []
    for pid in np.unique(pred[pred > 0]):
        own = pred == pid
        best_id, best_iou = None, 0.0
        for gid in np.unique(gt[own & (gt > 0)]):
            other = gt == gid
            inter = np.count_nonzero(own & other)
            iou = inter / (np.count_nonzero(own) + np.count_nonzero(other) - inter)
            if iou > best_iou:
                best_id, best_iou = int(gid), float(iou)
        out.append((best_id if best_iou > MATCH_IOU else None, best_iou))
    return out


def prediction_matches(record: dict, folder: Path, gt_mask) -> tuple[list[tuple[int | None, float]], str]:
    name = record.get('prediction_path')
    if name and (folder / Path(name).name).is_file():
        import numpy as np
        return exact_matches(np.load(folder / Path(name).name), gt_mask), 'exact'
    if 'raw_best_gt_ious' in record:
        ids = record.get('raw_best_gt_ids') or [None] * len(record['raw_best_gt_ious'])
        return [(None if g is None else int(g), float(iou or 0.0))
                for g, iou in zip(ids, record['raw_best_gt_ious'])], 'raw'
    raise ValueError(f"{folder}: tile {record['id']} has neither an instance map nor raw_best_gt_* matches")


def pq_of(iou_sum, tp, fp, fn):
    if tp + fp + fn == 0:
        return 1.0
    return iou_sum / (tp + 0.5 * fp + 0.5 * fn)


def score_tile(preds, gt: dict, ignore=frozenset()) -> dict:
    """One-to-one matches at IoU > 0.5; predictions matching an ignored nucleus count as neither TP nor FP."""
    matched, fp = {}, 0
    for gid, iou in preds:
        hit = gid is not None and iou > MATCH_IOU
        if hit and gid in ignore:
            continue
        if hit and gid not in matched:
            matched[gid] = iou
        else:
            fp += 1
    tp = len(matched)
    fn = len(set(gt) - set(ignore)) - tp
    return {'tp': tp, 'fp': fp, 'fn': fn, 'pq': pq_of(sum(matched.values()), tp, fp, fn), 'matched': set(matched)}


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_sliver(info: dict) -> bool:
    return info['edge'] and info['area'] < SLIVER_AREA


def fn_causes(record: dict, gt: dict, matched) -> dict | None:
    """Missed nuclei by best box IoU of any predicted box; None when the run kept no per-object boxes."""
    if 'converted_objects' not in record:
        return None
    boxes = [obj['box'] for obj in record['converted_objects'] or [] if obj.get('box')]
    counts = {cause: 0 for cause in CAUSES}
    slivers = {cause: 0 for cause in CAUSES}
    for gid, info in gt.items():
        if gid in matched:
            continue
        best = max((box_iou(box, info['box']) for box in boxes), default=0.0)
        cause = 'good_box_mask_failed' if best > MATCH_IOU else 'box_near' if best > NEAR_BOX_IOU else 'no_box'
        counts[cause] += 1
        slivers[cause] += is_sliver(info)
    return {'counts': counts, 'edge_slivers': slivers}


def tile_extras(record, folder, gt_mask, gt, edge_slivers=True, causes=True) -> dict:
    preds, method = prediction_matches(record, folder, gt_mask)
    every = score_tile(preds, gt)
    out = {'method': method, 'all': {k: every[k] for k in ('tp', 'fp', 'fn', 'pq')}}
    if edge_slivers:
        ignore = {gid for gid, info in gt.items() if is_sliver(info)}
        kept = score_tile(preds, gt, ignore)
        out['no_slivers'] = {**{k: kept[k] for k in ('tp', 'fp', 'fn', 'pq')}, 'excluded': len(ignore)}
    if causes:
        out['fn_causes'] = fn_causes(record, gt, every['matched'])
    return out


def counts(tiles: dict[str, dict], ids) -> dict:
    tp, fp, fn = (sum(tiles[i][k] for i in ids) for k in ('tp', 'fp', 'fn'))
    return {'pq': sum(tiles[i]['pq'] for i in ids) / len(ids), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn)}


def sliver_summary(extras, reference, ids, samples, compare_with_reference) -> dict:
    kept = {i: extras[i]['no_slivers'] for i in ids}
    out = {'method': sorted({extras[i]['method'] for i in ids}),
           'excluded': sum(kept[i]['excluded'] for i in ids),
           'all_nuclei': counts({i: extras[i]['all'] for i in ids}, ids), 'without_slivers': counts(kept, ids)}
    if compare_with_reference:
        out['vs_reference'] = paired_bootstrap({i: reference[i]['no_slivers'] for i in ids}, kept, ids, samples)
    return out


def cause_summary(extras, ids) -> dict | None:
    per_tile = [extras[i]['fn_causes'] for i in ids]
    if any(t is None for t in per_tile):
        return None
    out = {'method': sorted({extras[i]['method'] for i in ids}),
           'fn': sum(sum(t['counts'].values()) for t in per_tile)}
    for part in ('counts', 'edge_slivers'):
        out[part] = {cause: sum(t[part][cause] for t in per_tile) for cause in CAUSES}
    return out


# ---------------------------------------------------------------- assembly

def compare(paths, labels=None, samples=10000, manifest=None, by_dataset=False, per_group=False,
            edge_slivers=False, causes=False):
    """Rows of the comparison table, reference first. manifest: tile ID -> manifest row (see read_manifests)."""
    labels = labels or [Path(p).parent.name if Path(p).name == 'after' else Path(p).name for p in paths]
    records = [load_records(p) for p in paths]
    runs = [metrics(r) for r in records]
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
    if not (by_dataset or per_group or edge_slivers or causes):
        return table

    manifest = manifest or {}
    meta = tile_meta(ids, manifest, records[0])
    extras = None
    if edge_slivers or causes:
        if not manifest:
            raise ValueError('Edge-sliver and FN-cause reports need the evaluation manifest (reference masks)')
        gts = {i: load_gt(manifest[i]) for i in ids}
        extras = [{i: tile_extras(recs[i], eval_folder(path), *gts[i], edge_slivers, causes) for i in ids}
                  for recs, path in zip(records, paths)]

    def details(index, subset):
        out = {}
        if edge_slivers:
            out['edge_slivers'] = sliver_summary(extras[index], extras[0], subset, samples, index > 0)
        if causes:
            out['fn_causes'] = cause_summary(extras[index], subset)
        return out

    for index, entry in enumerate(table):
        entry.update(details(index, ids))
        if by_dataset:
            entry['by_dataset'] = {}
            for dataset, subset in by_dataset_ids(ids, meta).items():
                part = {**totals(runs[index], subset), **details(index, subset)}
                if index:
                    part['vs_reference'] = paired_bootstrap(runs[0], runs[index], subset, samples)
                entry['by_dataset'][dataset] = part
        if per_group and index:
            entry['per_group'] = group_signs(runs[0], runs[index], ids, meta)
    return table


# ---------------------------------------------------------------- markdown

def metric_cells(row):
    return [f"{row['pq']:.3f}", f"{row['sq']:.3f}", f"{row['dq']:.3f}", str(row['tp']), str(row['fp']),
            str(row['fn']), f"{row['precision']:.3f}", f"{row['recall']:.3f}"]


def delta_cells(ref, keys=('pq', 'tp', 'fp', 'fn')):
    if not ref:
        return ['reference'] + [''] * (len(keys) - 1)
    cells = []
    for key in keys:
        d = ref[key]
        if key == 'pq':
            cells.append(f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        else:
            cells.append(f"{d['delta']:+.0f} [{d['ci95'][0]:+.0f}, {d['ci95'][1]:+.0f}]")
    return cells


def markdown(table):
    lines = ['| Run | PQ | SQ | DQ | TP | FP | FN | Precision | Recall | ΔPQ vs ref [95% CI] | ΔTP | ΔFP | ΔFN |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in table:
        cells = [row['label'], *metric_cells(row), *delta_cells(row.get('vs_reference'))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def markdown_by_dataset(table):
    lines = ['| Dataset | Run | Tiles | PQ | SQ | DQ | TP | FP | FN | Precision | Recall | ΔPQ vs ref [95% CI] '
             '| ΔTP | ΔFP | ΔFN |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for dataset in table[0].get('by_dataset', {}):
        for row in table:
            part = row['by_dataset'][dataset]
            cells = [dataset, row['label'], str(part['tiles']), *metric_cells(part),
                     *delta_cells(part.get('vs_reference'))]
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def markdown_groups(table):
    runs = [row for row in table if 'per_group' in row]
    if not runs:
        return ''
    lines = ['| Run | Groups | dTP > 0 / = 0 / < 0 | sign-test p | dFP > 0 / = 0 / < 0 | sign-test p |',
             '|---|---|---|---|---|---|']
    for row in runs:
        signs = row['per_group']['signs']
        cells = [row['label'], str(len(row['per_group']['groups']))]
        for key in ('dtp', 'dfp'):
            s = signs[key]
            cells += [f"{s['up']} / {s['same']} / {s['down']}", f"{s['sign_test_p']:.3f}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    header = ['Group', 'Dataset', 'Tiles'] + [f"{row['label']} {k}" for row in runs for k in ('dTP', 'dFP')]
    lines += ['', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for group, info in runs[0]['per_group']['groups'].items():
        cells = [group, info['dataset'], str(info['tiles'])]
        for row in runs:
            g = row['per_group']['groups'][group]
            cells += [f"{g['dtp']:+d}", f"{g['dfp']:+d}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _sections(table, key):
    """(dataset label, run label, summary) for the whole set and, when present, each dataset."""
    out = [('all', row['label'], row.get(key)) for row in table]
    for dataset in table[0].get('by_dataset', {}):
        out += [(dataset, row['label'], row['by_dataset'][dataset].get(key)) for row in table]
    return [item for item in out if item[2] is not None]


def markdown_slivers(table):
    items = _sections(table, 'edge_slivers')
    if not items:
        return ''
    lines = [f'Edge slivers: reference nuclei cut by the tile edge with < {SLIVER_AREA} visible px. Matching: '
             + ', '.join(sorted({m for _, _, s in items for m in s['method']})) + '.', '',
             '| Dataset | Run | Slivers excluded | TP / FP / FN (all) | Recall (all) | PQ (all) '
             '| TP / FP / FN (no slivers) | Recall (no slivers) | PQ (no slivers) | ΔPQ | ΔTP | ΔFP | ΔFN |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for dataset, label, s in items:
        a, k = s['all_nuclei'], s['without_slivers']
        cells = [dataset, label, str(s['excluded']), f"{a['tp']} / {a['fp']} / {a['fn']}", f"{a['recall']:.3f}",
                 f"{a['pq']:.3f}", f"{k['tp']} / {k['fp']} / {k['fn']}", f"{k['recall']:.3f}", f"{k['pq']:.3f}",
                 *delta_cells(s.get('vs_reference'))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def markdown_causes(table):
    items = _sections(table, 'fn_causes')
    if not items:
        return ''
    lines = [f'FN causes by best predicted-box IoU: no box (<= {NEAR_BOX_IOU}), box near ({NEAR_BOX_IOU}-{MATCH_IOU}], '
             f'good box but mask failed (> {MATCH_IOU}); in parentheses, how many are edge slivers.', '',
             '| Dataset | Run | FN | No box | Box near | Good box, mask failed |', '|---|---|---|---|---|---|']
    for dataset, label, s in items:
        cells = [dataset, label, str(s['fn'])] + [f"{s['counts'][c]} ({s['edge_slivers'][c]})" for c in CAUSES]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True, help='Evaluation folder that the others are compared with')
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--manifest', nargs='+', help='Evaluation manifest(s) the runs were scored on')
    parser.add_argument('--by-dataset', action='store_true', help='Also report each dataset separately')
    parser.add_argument('--per-group', action='store_true', help='Sign counts of dTP / dFP per patient or image group')
    parser.add_argument('--exclude-edge-slivers', action='store_true',
                        help=f'Rescore without edge-cut reference nuclei under {SLIVER_AREA} px (needs --manifest)')
    parser.add_argument('--fn-causes', action='store_true',
                        help='Split missed nuclei by best predicted-box IoU (needs --manifest)')
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    paths = [args.reference, *args.runs]
    if args.labels and len(args.labels) != len(paths):
        parser.error('--labels needs one label per evaluation, reference first')
    if (args.exclude_edge_slivers or args.fn_causes) and not args.manifest:
        parser.error('--exclude-edge-slivers and --fn-causes need --manifest')
    table = compare(paths, args.labels, args.samples, manifest=read_manifests(args.manifest),
                    by_dataset=args.by_dataset, per_group=args.per_group,
                    edge_slivers=args.exclude_edge_slivers, causes=args.fn_causes)
    sections = [markdown(table)]
    if args.by_dataset:
        sections.append(markdown_by_dataset(table))
    sections += [markdown_groups(table), markdown_slivers(table), markdown_causes(table)]
    print('\n\n'.join(s for s in sections if s))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
