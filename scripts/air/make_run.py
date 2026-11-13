#!/usr/bin/env python3
"""Submit one arm spec (configs/air/<name>.json) to Databricks AI Runtime with `databricks air run`.

    python scripts/air/make_run.py configs/air/<name>.json --receipts <file outside git> [--dry-run]

The YAML is generated next to the receipts file. The code archive is this repository's nucleus_rl/, scripts/,
configs/ and protocols/ folders as they are on disk; the commit is recorded in the receipt. The run name is also the
idempotency key, so submitting the same spec twice returns the first run instead of starting a second one, and a
name that already has a receipt is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


DEPENDENCIES = ['torch==2.8.0', 'torchvision==0.23.0', 'transformers==5.7.0', 'peft==0.19.0', 'accelerate==1.12.0',
                'huggingface-hub==1.5.0', 'tokenizers==0.22.2', 'safetensors==0.7.0', 'numpy==2.2.6',
                'pillow==11.3.0', 'hydra-core', 'iopath', 'tqdm', 'scipy']
INCLUDE = ['nucleus_rl', 'scripts', 'configs', 'protocols']


def yaml_text(spec, spec_relative, root, args):
    lines = [f"experiment_name: {spec['name']}",
             'compute:', f"  accelerator_type: {spec.get('accelerator', args.accelerator)}",
             f"  num_accelerators: {spec.get('num_accelerators', 1)}",
             'environment:', '  version: 6', '  dependencies:', *[f'    - {d}' for d in DEPENDENCIES],
             'code_source:', f'  root_path: {root}', f"  remote_volume: {spec['volume']}/code",
             '  include_paths:', *[f'    - {p}' for p in INCLUDE],
             f'command: python "$CODE_SOURCE_PATH/scripts/air/run_arm.py" --spec "{spec_relative}"',
             f"timeout_minutes: {spec.get('timeout_minutes', args.timeout_minutes)}",
             'max_retries: 0']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec')
    parser.add_argument('--receipts', required=True, help='JSONL file outside the repository')
    parser.add_argument('--accelerator', default='GPU_1xH100')
    parser.add_argument('--timeout-minutes', type=int, default=1440)
    parser.add_argument('--profile', default='dfci')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    receipts = Path(args.receipts).resolve()
    if root in receipts.parents:
        parser.error('Keep receipts outside the repository')
    if receipts.exists() and any(json.loads(line)['name'] == spec['name']
                                 for line in receipts.read_text().splitlines() if line.strip()):
        parser.error(f"{spec['name']} already has a receipt; choose a new name")
    yaml_path = receipts.parent / 'yaml' / f"{spec['name']}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text(spec, spec_path.relative_to(root).as_posix(), root, args))
    command = ['databricks', 'air', 'run', '-p', args.profile, '-f', str(yaml_path), '-o', 'json']
    command += ['--dry-run'] if args.dry_run else ['--idempotency-key', spec['name']]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode or args.dry_run:
        return result.returncode
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    data = json.loads('\n'.join(lines[lines.index('{'):])) if '{' in lines else {}
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain', *INCLUDE],
                                capture_output=True, text=True).stdout.strip())
    with receipts.open('a') as stream:
        stream.write(json.dumps({'name': spec['name'], 'run_id': data.get('data', {}).get('run_id'),
                                 'submitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 'commit': commit, 'dirty_tree': dirty, 'spec': spec,
                                 'yaml': str(yaml_path)}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
