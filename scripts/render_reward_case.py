"""Render an audited, post-hoc teaching case from saved GRPO outputs. CPU only."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_h200_training import load_dataset, verify_phase, sha
from scripts.report_graphics import _font

INK = '#172b43'
MUTED = '#50637a'
BG = '#f1f5f9'
ORANGE = '#ffad00'
PINK = (244, 62, 136)
CYAN = (0, 180, 214)
SIZE = 352
RULE = ('Post-hoc illustrative selection: first lexicographic validation ID with at least '
        'three complete nuclei, valid answers in both phases, one predicted instance and '
        'zero matched instances in both phases, and increased foreground IoU. '
        'This is a case explanation, not a representative sample or a new experiment.')


def boundary(mask):
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def panel(image, gt, record=None, prediction=None, show_gt=False, labels=False):
    pixels = np.asarray(image).astype(float)
    if prediction is not None:
        foreground = prediction > 0
        pixels[foreground] = pixels[foreground] * .65 + np.array(CYAN) * .35
        pixels[boundary(foreground)] = CYAN
    if show_gt:
        for value in np.unique(gt[gt > 0]):
            pixels[boundary(gt == value)] = PINK
    out = Image.fromarray(pixels.astype(np.uint8)).resize((SIZE, SIZE), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(out)
    scale = SIZE / image.width
    if labels:
        for index, value in enumerate(np.unique(gt[gt > 0]), 1):
            points = np.argwhere(gt == value)
            center = points[np.argmin(((points-points.mean(axis=0))**2).sum(axis=1))]
            y, x = center
            draw.text(((x+.5)*scale, (y+.5)*scale), f'G{index}', anchor='mm',
                      font=_font(21), fill='white', stroke_width=2, stroke_fill=INK)
    if record is not None:
        for obj in record['converted_objects']:
            x0, y0, x1, y1 = [v * scale for v in obj['box']]
            draw.rectangle((x0, y0, min(SIZE-1, x1), min(SIZE-1, y1)), outline=ORANGE, width=4)
            x, y = [v * scale for v in obj['point']]
            draw.ellipse((x-7, y-7, x+7, y+7), fill=ORANGE, outline=INK, width=2)
    return out


def stats(saved, gt):
    pred = saved['prediction'] > 0
    target = gt > 0
    intersection = int(np.count_nonzero(pred & target))
    union = int(np.count_nonzero(pred | target))
    record = saved['record']
    assert np.isclose(intersection/union, record['seg_reward'])
    per_instance = []
    for value in np.unique(gt[target]):
        nucleus = gt == value
        per_instance.append({'gt_id': int(value), 'iou': float(np.count_nonzero(pred & nucleus)
                             / np.count_nonzero(pred | nucleus))})
    point = record['converted_objects'][0]['point']
    point_id = int(gt[int(np.floor(point[1])), int(np.floor(point[0]))])
    return {'foreground_pixels': int(pred.sum()), 'gt_foreground_pixels': int(target.sum()),
            'intersection': intersection, 'union': union,
            'false_positive_pixels': int(np.count_nonzero(pred & ~target)),
            'image_coverage_fraction': float(pred.mean()),
            'point_gt_id': point_id, 'per_instance_ious': per_instance}


def render_slide(asset, phases, metrics, destination):
    canvas = Image.new('RGB', (1920, 1080), BG)
    d = ImageDraw.Draw(canvas)
    def text(x, y, value, size=26, color=INK):
        assert d.textlength(value, font=_font(size)) < 1920-x-24, value
        d.text((x, y), value, font=_font(size), fill=color)
    text(40, 25, 'Higher foreground reward, still no correctly segmented nucleus', 45)
    text(40, 85, 'Qwen2.5-VL-7B | language LoRA | 1,000 steps | LR 5e-5 | seed 42 | frozen SAM2.1 tiny', 26, MUTED)
    text(40, 139, 'REFERENCE', 26)
    text(412, 139, 'BEFORE TRAINING', 26)
    text(1156, 139, 'AFTER TRAINING', 26)
    xs = [40, 412, 784, 1156, 1528]
    titles = ['GT: 5 annotated objects', 'Qwen box + point', 'SAM mask + GT outline',
              'Qwen box + point', 'SAM mask + GT outline']
    tiles = [panel(asset['image'], asset['mask'], show_gt=True, labels=True)]
    for phase in ['before', 'after']:
        saved = phases[phase]
        tiles.extend([panel(asset['image'], asset['mask'], record=saved['record']),
                      panel(asset['image'], asset['mask'], prediction=saved['prediction'], show_gt=True)])
    for x, title, tile in zip(xs, titles, tiles):
        text(x, 182, title, 24)
        canvas.paste(tile, (x, 220))
    text(40, 594, '3 complete nuclei', 25)
    text(40, 628, '+ 2 edge fragments', 25)
    text(40, 684, 'Orange: Qwen prompt', 24)
    text(40, 718, 'Cyan: SAM foreground', 24)
    text(40, 752, 'Pink: true boundaries', 24)
    text(40, 806, 'Same real crop in all panels.', 22, MUTED)
    text(40, 837, 'Native size: 128 x 128.', 22, MUTED)
    for phase, x in [('before', 412), ('after', 1156)]:
        r, m = phases[phase]['record'], metrics[phase]
        d.rounded_rectangle((x, 590, x+724, 874), radius=16, fill='white')
        obj = r['raw_objects'][0]
        box = ', '.join(str(int(v)) for v in obj['box'])
        point = ', '.join(str(int(v)) for v in obj['point'])
        text(x+18, 606, f'Qwen: box [{box}]  point [{point}]', 23)
        text(x+18, 642, 'Raw coordinates: 224 x 224; overlays scaled to the same crop.', 21, MUTED)
        text(x+18, 683, f"R_seg = {r['seg_reward']:.4f}     R_fmt = {r['format_reward']:.0f}     Total = {r['total_reward']:.4f}", 28)
        text(x+18, 731, f"Matched: {r['tp']}/5     PQ: {r['pq']:.0f}     Recall: {r['recall']:.0%}", 28)
        text(x+18, 776, 'Predictions: 1   |   Count error: 4   |   Duplicates: 0', 24)
        text(x+18, 818, f"SAM covers {m['foreground_pixels']:,} / 16,384 image pixels", 24, MUTED)
    d.rounded_rectangle((40, 900, 1880, 978), radius=16, fill='#e0eeeb')
    text(60, 917, 'More true pixels are covered, but the after-mask is almost the whole image: still one incorrect object.', 29, '#12584e')
    text(40, 995, 'Selected after inspecting results to explain this failure; not a representative sample. No inference was rerun.', 23, MUTED)
    text(40, 1032, asset['row']['id'] + ' | Match requires individual-mask IoU > 0.5; best IoU is only about 0.106 in both phases.', 22, MUTED)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    arguments = json.loads((args.run/'arguments.json').read_text())
    verified = {phase: verify_phase(args.run/phase, data['cohorts']['validation'], data['assets'], arguments)
                for phase in ['before', 'after']}
    assert all(v['complete'] and not v['missing_artifacts'] for v in verified.values())
    eligible = []
    for key in sorted(verified['before']['examples']):
        asset = data['assets'][key]
        b, a = [verified[p]['examples'][key]['record'] for p in ['before', 'after']]
        complete = asset['count'] - len(asset['row']['truncated_instance_ids'])
        if (complete >= 3 and b['format_valid'] and a['format_valid']
                and b['tp'] == a['tp'] == 0 and b['pred_count'] == a['pred_count'] == 1
                and a['seg_reward'] > b['seg_reward']):
            eligible.append(key)
    assert eligible
    key = eligible[0]
    assert key == 'TCGA-AR-A1AS-01Z-00-DX1_x256_y0', 'Saved-case selection changed'
    asset = data['assets'][key]
    phases = {p: verified[p]['examples'][key] for p in ['before', 'after']}
    metrics = {p: stats(phases[p], asset['mask']) for p in phases}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    asset['image'].save(out/'input.png')
    for phase in phases:
        (out/f'{phase}.txt').write_text(phases[phase]['record']['text'])
    render_slide(asset, phases, metrics, out/'comparison.png')
    proof = {'case_id': key, 'selection_rule': RULE, 'eligible_case_count': len(eligible),
             'eligible_ids': eligible, 'source_run': str(args.run),
             'source_result_sha256': sha(args.run/'result.json'),
             'manifest_sha256': data['manifest_sha256']['validation'],
             'image_sha256': asset['row']['image_sha256'], 'mask_sha256': asset['row']['mask_sha256'],
             'source_instance_ids': asset['row']['instance_ids'],
             'partial_instance_ids': asset['row']['truncated_instance_ids'],
             'display_ids': {f'G{i}': int(v) for i, v in enumerate(np.unique(asset['mask'][asset['mask'] > 0]), 1)},
             'metrics': metrics, 'records': {p: phases[p]['record'] for p in phases},
             'mask_sha256': {p: sha(args.run/p/Path(phases[p]['record']['prediction_path']).name) for p in phases},
             'verification': 'Both full64 phases reverified from saved outputs; no model execution or source edits.'}
    (out/'case.json').write_text(json.dumps(proof, indent=2)+'\n')
    b, a = phases['before']['record'], phases['after']['record']
    note = f'''# Why a higher reward still misses every nucleus

![Paired Qwen prompts and SAM masks](comparison.png)

Same real validation crop: `{key}`. Three complete nuclei and two partial border nuclei are annotated. The comparison is Qwen2.5-VL-7B before and after 1,000 language-LoRA GRPO steps (LR 5e-5, seed 42), with frozen SAM2.1 tiny. Evaluation uses greedy decoding. No new inference or training was run to create this figure.

| Per-crop metric | Before | After |
| --- | ---: | ---: |
| R_seg: foreground IoU | {b['seg_reward']:.6f} | {a['seg_reward']:.6f} |
| R_fmt | 1 | 1 |
| Total reward | {b['total_reward']:.6f} | {a['total_reward']:.6f} |
| Matched instances | 0 / 5 | 0 / 5 |
| PQ / recall | 0 / 0 | 0 / 0 |
| Predicted objects | 1 | 1 |
| Absolute count error | 4 | 4 |
| Same-GT duplicate extras | 0 | 0 |

Before training, the point lies inside true nucleus 106 (display G3). The SAM mask contains only 184 pixels, all inside that nucleus, which occupies 1,738 pixels. It is a small fragment, not a correctly delineated nucleus. Foreground IoU is 184 / 4,259 = 0.0432.

After training, the point lies on annotated background. The almost full-frame box produces one mask covering 16,374 of 16,384 pixels (99.94%). It includes all 4,259 true foreground pixels and 12,115 background pixels. Foreground IoU is 4,259 / 16,374 = 0.2601. This is numerically better than the tiny initial fragment, despite the large false-positive area.

Foreground IoU **does penalize background** through its union denominator. However, it ignores which nucleus each foreground pixel belongs to. The largest IoU with any individual annotated nucleus is only about 0.106 in either phase, below the strict IoU > 0.5 matching threshold. Neither phase correctly segments an individual nucleus. “Unmatched” does not mean the after-mask covers none of the nucleus pixels.

R_fmt checks answer structure and coordinate validity, not whether the point is actually inside a real nucleus. Both answers receive R_fmt = 1. These are rewards recomputed on the same validation image, not sampled training-group rewards.

## Selection and limits

{RULE} There are {len(eligible)} qualifying crops. The existing fixed multi-nucleus example in the parent report is unchanged; that other crop's R_seg decreases from 0.28435 to 0.21687. Do not confuse the present illustration with that fixed example or with cohort-average improvement.

Prompts use a 224 x 224 processed coordinate frame. All overlay positions are converted to the same 128 x 128 native crop using a 128/224 scale, then enlarged only for display. G1-G5 are display labels; original IDs are preserved in case.json. Pink outlines are true boundaries, orange marks are Qwen box/point outputs, and cyan is SAM foreground. The two edge fragments remain in the five-instance denominator.

[Exact raw before answer](before.txt) · [Exact raw after answer](after.txt) · [Audited coordinates, metrics and provenance](case.json) · [Unannotated input](input.png) · [Original full64 report](../README.md)

Reproduce from the repository root:

```sh
.venv/bin/python scripts/render_reward_case.py --run runs/h200_overnight_20261016/main_7b_lr5e-5_seed42 --dataset data/monuseg_official30_grpo_v1 --output reports/2026-10-16-main-7b-lr5e-5/reward-case
```
'''
    (out/'README.md').write_text(note)
    rows = [('R_seg', f"{b['seg_reward']:.6f}", f"{a['seg_reward']:.6f}"),
            ('R_fmt', '1', '1'), ('Total reward', f"{b['total_reward']:.6f}", f"{a['total_reward']:.6f}"),
            ('Matched / GT', '0 / 5', '0 / 5'), ('PQ / recall', '0 / 0', '0 / 0'),
            ('Predicted objects', '1', '1'), ('Count error', '4', '4'), ('Duplicate extras', '0', '0')]
    table = ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in rows)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Higher reward, missed nuclei: paired real crop</title><style>
body{{font:18px/1.6 system-ui,sans-serif;color:{INK};background:{BG};max-width:1700px;margin:30px auto;padding:0 24px}}
h1{{line-height:1.2}}img{{width:100%;height:auto}}section{{background:white;padding:24px;border-radius:14px;margin:20px 0}}
table{{border-collapse:collapse;min-width:550px}}td,th{{padding:10px 24px;border-bottom:1px solid #ddd;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f7;padding:16px}}a{{color:#126494}}
</style><h1>Higher foreground reward, still no correctly segmented nucleus</h1>
<p>Same real validation crop, same frozen SAM. Qwen2.5-VL-7B before / after 1,000 LoRA GRPO steps.</p>
<a href="comparison.png"><img src="comparison.png" alt="Five panels show ground truth, before prompts and masks, and after prompts and masks. Reward rises, matched nuclei remain zero."></a>
<p><a href="comparison.png">Slide PNG (1920 x 1080)</a> · <a href="README.md">Full explanation</a> · <a href="case.json">Verified data</a> · <a href="input.png">Original crop</a> · <a href="../index.html">Full64 report</a></p>
<section><h2>What changed?</h2><p><b>Before:</b> one tiny mask inside a nucleus, covering 184 pixels. <b>After:</b> one mask covers 99.94% of the image, including all five annotated objects and 12,115 background pixels.</p>
<p>The foreground overlap score rises from 184 / 4,259 to 4,259 / 16,374. It penalizes background, but the large region still outperforms the initial fragment. No individual nucleus passes IoU &gt; 0.5, so instance recall and PQ stay zero. The after point is on background, although its coordinates are format-valid.</p>
<table><tr><th>Per-crop metric</th><th>Before</th><th>After</th></tr>{table}</table></section>
<section><h2>Exact Qwen completions</h2><p>Numbers use the 224 x 224 processed frame; overlays are scaled to the same native 128 x 128 crop.</p>
<h3>Before</h3><pre>{html.escape(b['text'])}</pre><h3>After</h3><pre>{html.escape(a['text'])}</pre></section>
<section><h2>Scope</h2><p>{html.escape(RULE)} There are {len(eligible)} qualifying cases. This crop has three complete nuclei and two edge fragments; both fragments count as annotated instances.</p>
<p>This case was selected to explain a reward increase. The separate, original fixed multi-nucleus example has a reward decrease and is preserved in the parent report. Cohort summaries and raw experimental evidence are unchanged. No model was rerun.</p></section></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps({'case': key, 'eligible_count': len(eligible), 'metrics': metrics, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
