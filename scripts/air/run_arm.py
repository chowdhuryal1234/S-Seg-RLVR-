#!/usr/bin/env python3
"""Run one SFT / GRPO / evaluation arm on Databricks AI Runtime (serverless GPU), the Slurm wrapper's counterpart.

Submitted by scripts/air/make_run.py as `python $CODE_SOURCE_PATH/scripts/air/run_arm.py --spec <spec.json>`.
The spec names the runner module (nucleus_rl.sft or nucleus_rl.grpo), its arguments, and the Volume inputs to stage.
Arguments may use {stage:<name>} for a staged input's local path and {output} for the local output folder.
A spec with `steps` (a list of {"name", "module", "args"}) runs them one after another in one allocation, each
writing to <output>/<step name>; the model download is shared. A failed step does not stop the later ones.

Steps: install SAM-2 at the pinned commit -> copy inputs from the Volume to local disk -> run the module with its
output on local disk -> copy the output to <volume>/runs/<name>/ every --sync-minutes and once more at the end,
also after a failure. An existing <volume>/runs/<name>/ is refused (use a new name, as on FASRC).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


SAM2 = 'https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.zip'


def log(message):
    print(f'[run_arm {time.strftime("%H:%M:%S")}] {message}', flush=True)


def local_root():
    for candidate in (os.environ.get('SEGRLVR_LOCAL'), '/local_disk0', '/tmp'):
        if candidate and os.access(candidate, os.W_OK):
            return Path(candidate) / 'segrlvr'
    raise RuntimeError('No writable local disk')


def copy_in(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def sync_out(source, target):
    """Copy new or changed files (size or mtime) from local output to the Volume."""
    copied = 0
    for path in sorted(Path(source).rglob('*')):
        if not path.is_file():
            continue
        destination = Path(target) / path.relative_to(source)
        stat = path.stat()
        if destination.exists():
            existing = destination.stat()
            if existing.st_size == stat.st_size and existing.st_mtime >= stat.st_mtime:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        os.utime(destination, (stat.st_atime, stat.st_mtime))
        copied += 1
    return copied


def sync_run(output, stdout_log, remote):
    copied = sync_out(output, remote) if output.exists() else 0
    shutil.copyfile(stdout_log, remote / 'air_stdout.log')
    return copied


def periodic_sync(output, stdout_log, remote, minutes, stop):
    while not stop.wait(minutes * 60):
        try:
            log(f'sync: {sync_run(output, stdout_log, remote)} files')
        except OSError as error:
            log(f'sync failed, will retry: {error}')


def expand(argument, staged, output):
    for name, path in staged.items():
        argument = argument.replace('{stage:' + name + '}', str(path))
    return argument.replace('{output}', str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, help='JSON spec, relative to the code root or absolute')
    parser.add_argument('--sync-minutes', type=float, default=10)
    args = parser.parse_args(argv)
    code = Path(__file__).resolve().parents[2]
    spec_path = Path(args.spec) if Path(args.spec).is_absolute() else code / args.spec
    spec = json.loads(spec_path.read_text())
    volume = Path(spec['volume'])
    remote = volume / 'runs' / spec['name']
    if remote.exists():
        raise FileExistsError(f'{remote} exists; choose a new run name')
    remote.mkdir(parents=True)
    (remote / 'air_spec.json').write_text(json.dumps(spec, indent=2) + '\n')

    root = local_root()
    os.environ.setdefault('HF_HOME', str(root / 'hf'))
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    for key, value in spec.get('env', {}).items():
        os.environ[key] = str(value)
    log(f'code {code}, local {root}, remote {remote}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', '--no-deps', SAM2],
                   check=True, env={**os.environ, 'SAM2_BUILD_CUDA': '0'})

    staged = {}
    for name, relative in spec.get('stage', {}).items():
        target = root / 'inputs' / name
        started = time.time()
        copy_in(volume / relative, target)
        staged[name] = target
        log(f'staged {name} from {relative} in {time.time() - started:.0f}s')
    output = root / 'runs' / spec['name']
    steps = spec.get('steps') or [{'name': None, 'module': spec['module'], 'args': spec['args']}]
    commands = []
    for step in steps:
        target = output / step['name'] if step['name'] else output
        commands.append((step['name'], [sys.executable, '-m', step['module'],
                                        *(expand(a, staged, target) for a in step['args'])]))
    (remote / 'command.json').write_text(json.dumps(commands, indent=2) + '\n')

    stdout_log = root / 'logs' / f"{spec['name']}.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.touch()
    stop = threading.Event()
    syncer = threading.Thread(target=periodic_sync, args=(output, stdout_log, remote, args.sync_minutes, stop),
                              daemon=True)
    syncer.start()
    codes = {}
    with open(stdout_log, 'w') as stream:
        for name, command in commands:
            log('running: ' + ' '.join(command))
            stream.write(f'### step {name}: ' + ' '.join(command) + '\n')
            if name:
                output.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(command, cwd=code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                sys.stdout.write(line)
                stream.write(line)
                stream.flush()
            codes[name or spec['name']] = process.wait()
            log(f'step {name} exit {codes[name or spec["name"]]}')
    stop.set()
    syncer.join(timeout=5)
    code_ = next((c for c in codes.values() if c), 0)
    log(f'final sync: {sync_run(output, stdout_log, remote)} files; exits {codes}')
    (remote / 'air_exit.json').write_text(json.dumps({'returncode': code_, 'steps': codes,
                                                      'finished_utc': time.time()}) + '\n')
    return code_


if __name__ == '__main__':
    raise SystemExit(main())
