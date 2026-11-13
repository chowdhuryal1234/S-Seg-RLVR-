#!/usr/bin/env python3
"""Stitch per-tile predictions back into whole images and score them against the full reference masks.

    python scripts/stitch_eval.py --stitch data/monuseg_stitch_v1 --eval runs/<run>/after [--json out.json]
    python scripts/stitch_eval.py --stitch data/monuseg_stitch_v1 --oracle     # tile reference masks as predictions

--eval is the folder an --eval-only run on <stitch>/tiles.jsonl wrote (completions.jsonl and NNNNNN.npy instance
maps). Merge rule: every pixel of the image is owned by exactly one tile (cuts at the middle of each overlap); a
predicted instance is kept only by the tile that owns its centroid, and pixels already taken by an earlier kept
instance stay with it; a kept instance that overlaps an earlier one by at least MERGE_FRACTION of its area
only extends that one (a nucleus cut by a tile edge). Scoring: one-to-one matches at IoU > 0.5; TP, FP, FN, SQ (mean matched IoU), DQ and PQ per
image, plus pooled over all images (TP / FP / FN summed). --oracle stitches the reference tiles themselves, which
measures what the merge rule alone loses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MERGE_FRACTION = 0.2


def owned_ranges(starts, crop, length):
    cuts = [0] + [(b + a + crop) // 2 for a, b in zip(starts, starts[1:])] + [length]
    return list(zip(cuts, cuts[1:]))


def stitch(tile_maps, whole):
    """tile_maps: {(x, y): int instance map of the tile}. Returns the image-size instance map."""
    import numpy as np

    crop = whole['crop']
    canvas = np.zeros((whole['height'], whole['width']), dtype=np.int32)
    xr = dict(zip(whole['tile_x'], owned_ranges(whole['tile_x'], crop, whole['width'])))
    yr = dict(zip(whole['tile_y'], owned_ranges(whole['tile_y'], crop, whole['height'])))
    next_id = 1
    for (x, y), labels in sorted(tile_maps.items(), key=lambda item: (item[0][1], item[0][0])):
        (x0, x1), (y0, y1) = xr[x], yr[y]
        for instance in np.unique(labels[labels > 0]):
            ys, xs = np.nonzero(labels == instance)
            cx, cy = x + xs.mean(), y + ys.mean()
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                continue
            gy, gx = ys + y, xs + x
            taken = canvas[gy, gx]
            free = taken == 0
            if not free.any():
                continue
            owners, overlap = np.unique(taken[~free], return_counts=True)
            if len(owners) and overlap.max() >= MERGE_FRACTION * len(gy):
                # Mostly the same object as an instance kept from a neighbouring tile: extend it.
                canvas[gy[free], gx[free]] = owners[overlap.argmax()]
            else:
                canvas[gy[free], gx[free]] = next_id
                next_id += 1
    return canvas


def match(pred, gt):
    """One-to-one IoU > 0.5 matching; returns tp, fp, fn and the summed IoU of matches."""
    import numpy as np

    pred_ids, pred_area = np.unique(pred[pred > 0], return_counts=True)
    gt_ids, gt_area = np.unique(gt[gt > 0], return_counts=True)
    both = (pred > 0) & (gt > 0)
    pairs, inter = np.unique(np.stack([gt[both], pred[both]]), axis=1, return_counts=True)
    pa, ga = dict(zip(pred_ids, pred_area)), dict(zip(gt_ids, gt_area))
    tp, iou_sum = 0, 0.0
    for (g, p), overlap in zip(pairs.T, inter):
        iou = overlap / (ga[g] + pa[p] - overlap)
        if iou > 0.5:
            tp += 1
            iou_sum += iou
    return {'tp': tp, 'fp': len(pred_ids) - tp, 'fn': len(gt_ids) - tp, 'iou_sum': iou_sum}


def quality(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    sq = counts['iou_sum'] / tp if tp else 0.0
    dq = tp / (tp + .5 * fp + .5 * fn) if tp + fp + fn else 1.0
    return {**counts, 'sq': sq, 'dq': dq, 'pq': sq * dq,
            'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn)}


def tile_predictions(stitch_dir, eval_dir, oracle):
    import numpy as np
    from PIL import Image

    tiles = {r['id']: r for r in map(json.loads, (stitch_dir / 'tiles.jsonl').read_text().splitlines())}
    maps = {}
    if oracle:
        for row in tiles.values():
            with Image.open(stitch_dir / row['mask_path']) as source:
                maps[row['id']] = np.asarray(source).astype(np.int32)
    else:
        for line in (eval_dir / 'completions.jsonl').read_text().splitlines():
            record = json.loads(line)
            maps[record['id']] = np.load(eval_dir / Path(record['prediction_path']).name)
        missing = set(tiles) - set(maps)
        if missing:
            raise ValueError(f'{len(missing)} tiles have no prediction, e.g. {sorted(missing)[:3]}')
    grouped = {}
    for tile_id, labels in maps.items():
        s = tiles[tile_id]['stitch']
        grouped.setdefault(s['source_image_id'], {})[(s['x'], s['y'])] = labels
    return grouped


def evaluate(stitch_dir, eval_dir=None, oracle=False):
    import numpy as np
    from PIL import Image

    stitch_dir = Path(stitch_dir)
    grouped = tile_predictions(stitch_dir, Path(eval_dir) if eval_dir else None, oracle)
    images, pooled = [], {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0}
    for whole in map(json.loads, (stitch_dir / 'whole.jsonl').read_text().splitlines()):
        with Image.open(stitch_dir / whole['mask_path']) as source:
            gt = np.asarray(source).astype(np.int32)
        counts = match(stitch(grouped[whole['id']], whole), gt)
        for key in pooled:
            pooled[key] += counts[key]
        images.append({'id': whole['id'], **quality(counts)})
    return {'stitch': str(stitch_dir), 'eval': str(eval_dir) if eval_dir else 'oracle', 'images': images,
            'pooled': quality(pooled), 'mean_image_pq': sum(i['pq'] for i in images) / len(images)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stitch', required=True)
    parser.add_argument('--eval')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    if bool(args.eval) == args.oracle:
        parser.error('Give exactly one of --eval and --oracle')
    result = evaluate(args.stitch, args.eval, args.oracle)
    p = result['pooled']
    print(f"pooled PQ {p['pq']:.3f} SQ {p['sq']:.3f} DQ {p['dq']:.3f} TP {p['tp']} FP {p['fp']} FN {p['fn']} "
          f"precision {p['precision']:.3f} recall {p['recall']:.3f}; mean per-image PQ {result['mean_image_pq']:.3f}")
    for image in result['images']:
        print(f"  {image['id']}: PQ {image['pq']:.3f} TP {image['tp']} FP {image['fp']} FN {image['fn']}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
