#!/usr/bin/env python3
"""Submit one arm spec (configs/air/<name>.json) to Databricks AI Runtime with `databricks air run`.

    python scripts/air/make_run.py configs/air/<name>.json --receipts <file outside git> [--dry-run] [--skip-preflight]

How the code gets to the GPU node. The AI Runtime `code_source` field cannot be used: it unpacks a tarball from
<volume>/code/.internal through the /Volumes FUSE mount, and on the GPU nodes that mount fails below the catalog
("Invalid argument"), so every such run died with "Tarball not found". The Files API works there. So this script
  1. packs nucleus_rl/, scripts/, configs/ and protocols/ as they are on disk (no __pycache__ or hidden files) into a
     reproducible tar.gz named by its sha256, plus air_code.json with the commit and whether the tree was dirty;
  2. uploads it with the Files API (`databricks fs cp`) to <volume>/code/files-api/, unless that name is already there;
  3. writes a YAML whose command downloads that exact tarball with databricks.sdk, checks the sha256, unpacks it under
     /tmp/segrlvr/code/<sha> and runs scripts/air/run_arm.py from there (run_arm stages inputs and syncs outputs
     with the Files API too).
The alternative, `databricks workspace import-dir` to /Workspace, was not chosen: the code would be a mutable folder
that a later import overwrites while an earlier run is still queued (queues of 45+ min were seen), every import of
the training code would go through another FUSE mount, and import-dir turns notebook-style .py files into notebooks.
A content-addressed tarball is immutable, so the receipt's sha256 is exactly what ran.

Before submitting (skip with --skip-preflight; --dry-run never touches the Volume) every run folder
<volume>/runs/<name> must be absent and every staged input must exist. The run name is the idempotency key, and a
name that already has a receipt is refused.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import posixpath
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_arm  # noqa: E402  (same folder; shared spec and path checks)


DEPENDENCIES = ['torch==2.8.0', 'torchvision==0.23.0', 'transformers==5.7.0', 'peft==0.19.0', 'accelerate==1.12.0',
                'huggingface-hub==1.5.0', 'tokenizers==0.22.2', 'safetensors==0.7.0', 'numpy==2.2.6',
                'pillow==11.3.0', 'hydra-core', 'iopath', 'tqdm', 'scipy']
INCLUDE = ['nucleus_rl', 'scripts', 'configs', 'protocols']
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 20 << 20, 200 << 20  # Code only; anything bigger is data that slipped in.
LOCAL = '/tmp/segrlvr'

# Runs on the node inside `python - <<'PY'`; keep it standard library + databricks.sdk only.
BOOTSTRAP = r'''
import hashlib, io, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
remote, expected = os.environ['SEGRLVR_CODE_TARBALL'], os.environ['SEGRLVR_CODE_SHA256']
target = Path(os.environ['SEGRLVR_CODE_DIR'])
if (target / '.complete').is_file():
    print('code already unpacked at', target, flush=True)
    raise SystemExit(0)
try:
    from databricks.sdk import WorkspaceClient
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'databricks-sdk'], check=True)
    from databricks.sdk import WorkspaceClient
files = WorkspaceClient().files
for attempt in range(5):
    try:
        stream = files.download(remote).contents
        try:
            data = stream.read()
        finally:
            stream.close()
        break
    except Exception as error:
        if attempt == 4 or type(error).__name__ in ('NotFound', 'ResourceDoesNotExist', 'PermissionDenied'):
            raise
        print(f'code download failed ({type(error).__name__}: {error}); retrying', flush=True)
        time.sleep(10 * (attempt + 1))
digest = hashlib.sha256(data).hexdigest()
if digest != expected:
    raise SystemExit(f'{remote} has sha256 {digest}, expected {expected}')
partial = target.with_name(target.name + '.partial')
shutil.rmtree(partial, ignore_errors=True)
partial.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
            raise SystemExit(f'unsafe member {member.name} in {remote}')
    if hasattr(tarfile, 'data_filter'):
        archive.extractall(partial, filter='data')
    else:
        archive.extractall(partial)
(partial / '.complete').write_text(expected + '\n')
shutil.rmtree(target, ignore_errors=True)
partial.rename(target)
print(f'code {digest[:16]} ({len(data)} bytes) from {remote} unpacked at {target}', flush=True)
'''.strip('\n')


# ---------- Code archive ----------

def code_files(root):
    files = []
    for top in INCLUDE:
        for path in sorted((root / top).rglob('*')):
            relative = path.relative_to(root)
            if (path.is_file() and not any(part.startswith('.') or part == '__pycache__' for part in relative.parts)
                    and path.suffix != '.pyc'):
                files.append(relative)
    return files


def build_code_tarball(root, commit, dirty):
    """Reproducible tar.gz bytes of the code folders: same files, commit and dirty flag give the same sha256."""
    buffer, total = io.BytesIO(), 0
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            def add(name, data, mode=0o644):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime, info.uid, info.gid, info.uname, info.gname = \
                    len(data), mode, 0, 0, 0, '', ''
                archive.addfile(info, io.BytesIO(data))
            for relative in code_files(root):
                data = (root / relative).read_bytes()
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError(f'{relative} is {len(data)} bytes; the code archive is for code only')
                total += len(data)
                add(relative.as_posix(), data, 0o755 if (root / relative).stat().st_mode & 0o111 else 0o644)
            add('air_code.json', (json.dumps({'commit': commit, 'dirty_tree': dirty, 'include': INCLUDE},
                                             sort_keys=True) + '\n').encode())
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f'code archive holds {total} bytes; the code archive is for code only')
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def remote_tarball(spec, sha256):
    return posixpath.join(spec['volume'], 'code', 'files-api', f'segrlvr-code-{sha256[:16]}.tar.gz')


# ---------- YAML ----------

def bootstrap_command(tarball, sha256, spec_relative, sync_minutes):
    q = shlex.quote
    return '\n'.join([
        'set -euo pipefail',
        f'export SEGRLVR_LOCAL="${{SEGRLVR_LOCAL:-{LOCAL}}}"',
        'export DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT=true',
        f'export SEGRLVR_CODE_TARBALL={q(tarball)}',
        f'export SEGRLVR_CODE_SHA256={q(sha256)}',
        f'export SEGRLVR_CODE_DIR="$SEGRLVR_LOCAL/code/{sha256[:16]}"',
        "python - <<'PY'", BOOTSTRAP, 'PY',
        'cd "$SEGRLVR_CODE_DIR"',
        f'exec python scripts/air/run_arm.py --spec {q(spec_relative)} --sync-minutes {q(str(sync_minutes))}'])


def yaml_text(spec, spec_relative, tarball, sha256, args):
    command = bootstrap_command(tarball, sha256, spec_relative, spec.get('sync_minutes', 10))
    lines = [f"experiment_name: {spec['name']}",
             'compute:', f"  accelerator_type: {spec.get('accelerator', args.accelerator)}",
             f"  num_accelerators: {spec.get('num_accelerators', 1)}",
             'environment:', '  version: 6', '  dependencies:', *[f'    - {d}' for d in DEPENDENCIES],
             'command: |', *[f'  {line}' if line else '' for line in command.splitlines()],
             f"timeout_minutes: {spec.get('timeout_minutes', args.timeout_minutes)}",
             'max_retries: 0']
    return '\n'.join(lines) + '\n'


# ---------- Volume checks from the laptop (Files API through the CLI) ----------

def databricks(args, profile):
    return subprocess.run(['databricks', *args, '-p', profile], capture_output=True, text=True)


def volume_exists(path, profile):
    """Whether a Volume file or folder exists: its name is listed in its parent folder."""
    parent, name = posixpath.split(path.rstrip('/'))
    result = databricks(['fs', 'ls', f'dbfs:{parent}'], profile)
    if result.returncode:
        message = (result.stderr + result.stdout).lower()
        if any(text in message for text in ('no such', 'not found', 'does not exist', 'not_found')):
            return False
        raise RuntimeError(f'databricks fs ls dbfs:{parent} failed: {result.stderr.strip() or result.stdout.strip()}')
    return name in {line.strip().rstrip('/') for line in result.stdout.splitlines()}


def stage_checks(lane):
    """Volume paths that must exist before the lane can start."""
    paths = []
    for entry in lane.get('stage', {}).values():
        entry = run_arm.stage_entry(entry)
        if 'manifests' in entry or 'files' in entry:
            paths += [run_arm.resolve_remote(lane['volume'], posixpath.join(entry['path'], item))
                      for item in entry.get('manifests', []) + entry.get('files', [])]
        else:
            paths.append(run_arm.resolve_remote(lane['volume'], entry['path']))
    return paths


def preflight(spec, lanes, profile):
    problems = []
    runs = [posixpath.join(lane['volume'], 'runs', lane['name']) for lane in lanes]
    if 'lanes' in spec:
        runs.append(posixpath.join(spec['volume'], 'runs', spec['name']))
    for path in runs:
        if volume_exists(path, profile):
            problems.append(f'{path} exists; choose a new run name')
    for lane in lanes:
        for path in stage_checks(lane):
            if not volume_exists(path, profile):
                problems.append(f"{lane['name']}: input {path} is not on the Volume")
    return problems


def upload_tarball(data, local_path, remote, profile):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    if volume_exists(remote, profile):
        return 'already on the Volume'
    for args in (['fs', 'mkdir', f'dbfs:{posixpath.dirname(remote)}'], ['fs', 'cp', str(local_path), f'dbfs:{remote}']):
        result = databricks(args, profile)
        if result.returncode:
            raise RuntimeError(f"databricks {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return 'uploaded'


# ---------- Main ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec')
    parser.add_argument('--receipts', required=True, help='JSONL file outside the repository')
    parser.add_argument('--accelerator', default='GPU_1xH100')
    parser.add_argument('--timeout-minutes', type=int, default=1440)
    parser.add_argument('--profile', default='dfci')
    parser.add_argument('--dry-run', action='store_true', help='build and validate, `air run --dry-run`, no upload')
    parser.add_argument('--skip-preflight', action='store_true', help='do not check run folders and inputs')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    receipts = Path(args.receipts).resolve()
    if root in receipts.parents:
        parser.error('Keep receipts outside the repository')
    if spec_path.parent != root / 'configs' / 'air':
        parser.error('The spec must be in configs/air/ so that it is inside the code archive')
    try:
        lanes = run_arm.load_lanes(spec, root)
    except (ValueError, PermissionError, OSError) as error:
        parser.error(f'invalid spec: {error}')
    accelerators = spec.get('num_accelerators', 1)
    if 'lanes' in spec and len(lanes) > accelerators:
        parser.error(f'{len(lanes)} lanes need num_accelerators >= {len(lanes)} (got {accelerators})')
    if receipts.exists() and any(json.loads(line)['name'] == spec['name']
                                 for line in receipts.read_text().splitlines() if line.strip()):
        parser.error(f"{spec['name']} already has a receipt; choose a new name")

    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain', *INCLUDE],
                                capture_output=True, text=True).stdout.strip())
    data, sha256 = build_code_tarball(root, commit, dirty)
    tarball = remote_tarball(spec, sha256)
    local_tarball = receipts.parent / 'code' / posixpath.basename(tarball)
    if args.dry_run:
        local_tarball.parent.mkdir(parents=True, exist_ok=True)
        local_tarball.write_bytes(data)
        print(f'code archive {local_tarball} ({len(data)} bytes, sha256 {sha256}); not uploaded (dry run)')
    else:
        if not args.skip_preflight:
            problems = preflight(spec, lanes, args.profile)
            if problems:
                print('\n'.join(['Preflight refused the submission:', *problems]), file=sys.stderr)
                return 2
        state = upload_tarball(data, local_tarball, tarball, args.profile)
        print(f'code archive {tarball} ({len(data)} bytes, sha256 {sha256}): {state}')

    yaml_path = receipts.parent / 'yaml' / f"{spec['name']}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text(spec, spec_path.relative_to(root).as_posix(), tarball, sha256, args))
    command = ['databricks', 'air', 'run', '-p', args.profile, '-f', str(yaml_path), '-o', 'json']
    command += ['--dry-run'] if args.dry_run else ['--idempotency-key', spec['name']]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode or args.dry_run:
        return result.returncode
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    data_out = json.loads('\n'.join(lines[lines.index('{'):])) if '{' in lines else {}
    with receipts.open('a') as stream:
        stream.write(json.dumps({'name': spec['name'], 'run_id': data_out.get('data', {}).get('run_id'),
                                 'submitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 'commit': commit, 'dirty_tree': dirty, 'code_tarball': tarball,
                                 'code_sha256': sha256, 'lanes': [lane['name'] for lane in lanes],
                                 'spec': spec, 'yaml': str(yaml_path)}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
