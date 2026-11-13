#!/usr/bin/env python3
"""Databricks AI Runtime canary: did the Files-API code and data staging give the node everything a run needs?

    python -m scripts.air.canary --manifest <staged jsonl> [--manifest ...] --output <dir>
        [--expect monuseg=64 consep=42 puma=64] [--no-require-cuda] [--fuse-probe <Volume folder>]

Checks: nucleus_rl imports from the unpacked code; every manifest reads with nucleus_rl.evaluate.read_manifest (all
image and mask files present), every image and mask decodes and sizes match; row counts per dataset equal --expect;
torch sees CUDA; transformers, peft, huggingface_hub and sam2 import. Writes <output>/canary.json and an empty marker
<output>/CANARY_OK or <output>/CANARY_FAILED (run_arm.py uploads both to the Volume run folder). Exit 1 on failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

DISTRIBUTED = ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')


def check_manifest(path):
    from nucleus_rl.evaluate import read_manifest
    import numpy as np
    from PIL import Image
    rows = read_manifest(path)  # Resolves relative paths and raises if a file is missing.
    problems = []
    for row in rows:
        try:
            with Image.open(row['image_path']) as image:
                image.load()
                size = image.size
            mask_path = Path(row['mask_path'])
            if mask_path.suffix.lower() == '.npy':
                mask = np.load(mask_path, allow_pickle=False)
            else:
                with Image.open(mask_path) as source:
                    mask = np.array(source)
            if (mask.shape[1], mask.shape[0]) != size:
                problems.append(f"{row['id']}: image {size} vs mask {mask.shape[1::-1]}")
            gt = row.get('gt_mask')
            if isinstance(gt, str) and not (Path(path).resolve().parent / gt).is_file():
                problems.append(f"{row['id']}: gt_mask {gt} missing")
        except Exception as error:
            problems.append(f"{row['id']}: {type(error).__name__}: {error}")
    return {'rows': len(rows), 'by_dataset': dict(Counter(row.get('dataset', 'unknown') for row in rows)),
            'problems': problems[:20], 'problem_count': len(problems)}


def package_report(require_cuda):
    report, problems = {}, []
    for name in ('torch', 'transformers', 'peft', 'huggingface_hub', 'sam2', 'numpy', 'PIL'):
        try:
            module = importlib.import_module(name)
            report[name] = getattr(module, '__version__', 'imported')
        except Exception as error:
            report[name] = f'FAILED {type(error).__name__}: {error}'
            problems.append(f'import {name} failed')
    try:
        import torch
        report['cuda'] = {'available': torch.cuda.is_available(), 'devices': torch.cuda.device_count(),
                          'names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        if require_cuda and not torch.cuda.is_available():
            problems.append('CUDA is not available')
    except Exception as error:
        report['cuda'] = f'FAILED {type(error).__name__}: {error}'
        if require_cuda:
            problems.append('CUDA check failed')
    return report, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expect', nargs='*', default=[], help='DATASET=ROWS summed over all manifests')
    parser.add_argument('--no-require-cuda', dest='require_cuda', action='store_false')
    parser.add_argument('--fuse-probe', help='Volume folder to list through /Volumes (informational only)')
    args = parser.parse_args(argv)
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'python': sys.version.split()[0], 'host': platform.node(), 'cwd': os.getcwd(),
              'cpus': os.cpu_count(), 'env': {key: os.environ.get(key) for key in
                                               ('CUDA_VISIBLE_DEVICES', 'HF_HOME', 'SEGRLVR_LOCAL', *DISTRIBUTED)}}
    problems = []
    try:
        import nucleus_rl
        report['nucleus_rl'] = str(Path(nucleus_rl.__file__).resolve().parent)
        code_info = Path.cwd() / 'air_code.json'
        report['code'] = json.loads(code_info.read_text()) if code_info.is_file() else None
    except Exception as error:
        problems.append(f'import nucleus_rl failed: {type(error).__name__}: {error}')
    report['manifests'], counts = {}, Counter()
    for manifest in args.manifest:
        try:
            result = check_manifest(manifest)
            counts.update(result['by_dataset'])
            if result['problem_count']:
                problems.append(f"{manifest}: {result['problem_count']} unreadable rows")
        except Exception as error:
            result = {'error': f'{type(error).__name__}: {error}'}
            problems.append(f'{manifest}: {result["error"]}')
        report['manifests'][manifest] = result
    report['counts'] = dict(counts)
    for item in args.expect:
        dataset, expected = item.split('=')
        if counts.get(dataset, 0) != int(expected):
            problems.append(f'{dataset}: {counts.get(dataset, 0)} rows, expected {expected}')
    report['packages'], package_problems = package_report(args.require_cuda)
    problems += package_problems
    if args.fuse_probe:
        try:
            report['fuse_probe'] = {'path': args.fuse_probe, 'entries': sorted(os.listdir(args.fuse_probe))[:20]}
        except OSError as error:
            report['fuse_probe'] = {'path': args.fuse_probe, 'error': f'{type(error).__name__}: {error}'}
    usage = shutil.disk_usage(output)
    report['local_disk_free_gb'] = round(usage.free / 1e9, 1)
    report['problems'], report['ok'] = problems, not problems
    report['seconds'] = round(time.time() - started, 1)
    (output / 'canary.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / ('CANARY_OK' if not problems else 'CANARY_FAILED')).write_text('')
    print(json.dumps({'counts': report['counts'], 'ok': report['ok'], 'problems': problems}, indent=2), flush=True)
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
