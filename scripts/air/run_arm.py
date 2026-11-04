#!/usr/bin/env python3
"""Run SFT / GRPO / evaluation arms on Databricks AI Runtime (serverless GPU), the Slurm wrapper's counterpart.

Started by the command that scripts/air/make_run.py writes: it downloads the code tarball with the Files API, unpacks
it on local disk and runs `python scripts/air/run_arm.py --spec <spec.json>` from there.

Every Volume access uses the Databricks Files API (databricks.sdk WorkspaceClient().files), never the /Volumes FUSE
mount: on the GPU nodes the mount lists the catalog but fails below it ("Invalid argument"), while the Files API works.

Spec (configs/air/<name>.json):
  name, volume         run name and the Volume folder everything is relative to (e.g. .../processed/conny)
  stage                {name: entry}; an entry is a path (file or folder, copied whole) or
                       {"path": folder, "manifests": [jsonl relative to folder], "files": [extra files]}, which copies
                       only the manifests and the image_path / mask_path / gt_mask files their rows name
  module + args        the runner (python -m <module> <args>), or
  steps                [{"name", "module", "args"}] run one after another, each writing to <output>/<step name>;
                       a failed step does not stop the later ones
  env                  extra environment for the runner processes
  install_sam2         default true: pip install SAM-2 at the pinned commit first
  lanes                multi-lane mode (GPU_8xH100): a list of arm specs (paths relative to the code root, or inline
                       objects) run at the same time, lane i on GPU i only (CUDA_VISIBLE_DEVICES=i, torch.distributed
                       variables removed); one shared HF cache; lane i starts i * stagger_seconds (default 60) after
                       lane 0; the model snapshots the lanes name are downloaded once first (prefetch_models: false
                       turns that off). Each lane keeps its own Volume run folder; the exit code is 1 if any lane
                       failed.
Arguments may use {stage:<name>} for a staged input's local path and {output} for the local output folder.

Flow: refuse if any <volume>/runs/<name> exists -> install SAM-2 -> stage inputs to local disk (the local copy mirrors
the Volume layout, so relative manifest paths such as ../consep_v1/images/x.png keep working) -> run -> upload new or
changed output files to <volume>/runs/<name>/ every --sync-minutes and once more at the end, also after a failure or
SIGTERM. Uploads never delete. Inputs must stay inside `volume`, and test / held-out paths are refused before any read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time


SAM2 = 'https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.zip'
CODE = Path(__file__).resolve().parents[2]
MANIFEST_PATH_KEYS = ('image_path', 'mask_path', 'gt_mask')
# torch.distributed / torchrun variables the AI Runtime may set on a multi-GPU node; a lane is one process on one GPU.
DISTRIBUTED_ENV = ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_WORLD_SIZE', 'NODE_RANK',
                   'GROUP_RANK', 'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID')
# Never read: MoNuSeg official test, CoNSeP Test, PUMA held-out (or anything else named like a test split).
FORBIDDEN = (re.compile(r'(^|/)raw/monuseg/test(/|$)', re.I), re.compile(r'(^|/)(test|tests|testing)(/|$)', re.I),
             re.compile(r'held[-_ ]?out', re.I), re.compile(r'monuseg[-_ ]*2018[-_ ]*test', re.I))
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')

_print_lock = threading.Lock()
_sleep = time.sleep  # Patched in tests.


def log(message):
    with _print_lock:
        print(f'[run_arm {time.strftime("%H:%M:%S")}] {message}', flush=True)


def local_root():
    """$SEGRLVR_LOCAL (set by make_run's command to /tmp/segrlvr), else /local_disk0/segrlvr or /tmp/segrlvr."""
    if os.environ.get('SEGRLVR_LOCAL'):
        return Path(os.environ['SEGRLVR_LOCAL'])
    for candidate in ('/local_disk0', '/tmp'):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return Path(candidate) / 'segrlvr'
    raise RuntimeError('No writable local disk')


# ---------- Volume paths ----------

def check_allowed(path):
    for pattern in FORBIDDEN:
        if pattern.search(path):
            raise PermissionError(f'{path}: test / held-out data is never read')
    return path


def resolve_remote(base, relative):
    """Normalized Volume path for `relative` (to `base`); it must stay inside `base` and not be test data."""
    base = posixpath.normpath(base)
    path = posixpath.normpath(posixpath.join(base, relative))
    if path != base and not path.startswith(base + '/'):
        raise PermissionError(f'{relative!r} leaves the spec volume {base}')
    return check_allowed(path)


def local_mirror(root, remote):
    return Path(root) / 'inputs' / remote.lstrip('/')


def stage_entry(entry):
    entry = {'path': entry} if isinstance(entry, str) else dict(entry)
    if not isinstance(entry.get('path'), str):
        raise ValueError(f'stage entry needs a path: {entry}')
    unknown = set(entry) - {'path', 'manifests', 'files'}
    if unknown:
        raise ValueError(f'unknown stage keys {sorted(unknown)}')
    return entry


def check_stage(spec):
    """Path checks that need no network (also run by make_run before submission)."""
    for name, entry in spec.get('stage', {}).items():
        entry = stage_entry(entry)
        resolve_remote(spec['volume'], entry['path'])
        for item in entry.get('manifests', []) + entry.get('files', []):
            resolve_remote(spec['volume'], posixpath.join(entry['path'], item))


# ---------- Files API ----------

def _missing(error):
    return (isinstance(error, FileNotFoundError) or type(error).__name__ in ('NotFound', 'ResourceDoesNotExist')
            or getattr(error, 'error_code', None) in ('NOT_FOUND', 'RESOURCE_DOES_NOT_EXIST'))


class Volume:
    """Unity Catalog Volume through the Files API. `files` is WorkspaceClient().files or a test double."""

    def __init__(self, files=None, retries=4, backoff=5.0):
        self._files, self.retries, self.backoff = files, retries, backoff
        self._local, self._made, self._lock = threading.local(), set(), threading.Lock()

    @property
    def files(self):
        if self._files is not None:
            return self._files
        client = getattr(self._local, 'files', None)
        if client is None:  # One client per thread: staging downloads in parallel.
            # Keep transfers on the plain Files API endpoints; the SDK's presigned-URL client cannot reach cloud
            # storage from serverless GPU (Iris, scripts/dbx/dbx_common.py on origin/databricks-sft-grpo).
            os.environ.setdefault('DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT', 'true')
            from databricks.sdk import WorkspaceClient
            client = self._local.files = WorkspaceClient().files
        return client

    def _retry(self, what, function):
        for attempt in range(self.retries):
            try:
                return function()
            except Exception as error:
                if _missing(error) or attempt == self.retries - 1:
                    raise
                log(f'{what}: {type(error).__name__}: {str(error)[:200]}; retry {attempt + 1}')
                _sleep(self.backoff * 2 ** attempt)

    def kind(self, path):
        """'file', 'dir' or None."""
        try:
            self._retry(f'stat {path}', lambda: self.files.get_metadata(path))
            return 'file'
        except Exception as file_error:
            try:
                self._retry(f'stat {path}', lambda: self.files.get_directory_metadata(path))
                return 'dir'
            except Exception as dir_error:
                if _missing(dir_error) and (_missing(file_error) or type(file_error).__name__ == 'BadRequest'):
                    return None
                raise

    def walk(self, path):
        """(Volume path, size) of every file below `path`."""
        entries = self._retry(f'list {path}', lambda: list(self.files.list_directory_contents(path)))
        for entry in entries:
            child = entry.path.rstrip('/')
            if entry.is_directory:
                yield from self.walk(child)
            else:
                yield child, entry.file_size

    def mkdir(self, path):
        with self._lock:
            if path in self._made:
                return
        self._retry(f'mkdir {path}', lambda: self.files.create_directory(path))
        with self._lock:
            self._made.add(path)

    def download(self, path, local_path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        partial = local_path.with_name(f'.{local_path.name}.partial')

        def attempt():
            stream = self.files.download(path).contents
            try:
                with partial.open('wb') as output:
                    shutil.copyfileobj(stream, output, 1 << 20)
            finally:
                stream.close()
        self._retry(f'download {path}', attempt)
        partial.replace(local_path)  # A file under its final name is always complete.

    def upload(self, local_path, path):
        self.mkdir(posixpath.dirname(path))

        def attempt():
            with Path(local_path).open('rb') as stream:
                self.files.upload(path, stream, overwrite=True)
        self._retry(f'upload {path}', attempt)

    def write_text(self, path, text):
        self.mkdir(posixpath.dirname(path))
        self._retry(f'upload {path}', lambda: self.files.upload(path, io.BytesIO(text.encode()), overwrite=True))

    def upload_tree(self, local_dir, path, state):
        """Upload new or changed files (size, mtime); hidden in-progress names (.writing-*) wait. Never deletes."""
        local_dir, sent = Path(local_dir), 0
        if not local_dir.is_dir():
            return 0
        for file in sorted(local_dir.rglob('*')):
            relative = file.relative_to(local_dir).as_posix()
            if any(part.startswith('.') for part in PurePosixPath(relative).parts) or not file.is_file():
                continue
            try:
                stat = file.stat()
                key = (stat.st_size, stat.st_mtime_ns)
                if state.get(relative) == key:
                    continue
                self.upload(file, f'{path}/{relative}')
            except FileNotFoundError:
                continue  # Replaced or removed (old checkpoint) mid-pass; the Volume copy is kept.
            state[relative] = key
            sent += 1
        return sent


def sync_out(source, target):
    """Local-folder counterpart of Volume.upload_tree (copy new or changed files); not used for the Volume."""
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


# ---------- Staging ----------

def manifest_files(volume_root, manifest_remote, manifest_local):
    """Volume paths of the files that a manifest's rows name, relative to the manifest's folder."""
    folder, wanted = posixpath.dirname(manifest_remote), set()
    for number, line in enumerate(Path(manifest_local).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in MANIFEST_PATH_KEYS:
            value = row.get(key)
            if not isinstance(value, str):
                continue
            if value.startswith('/'):
                raise ValueError(f'{manifest_remote}:{number}: absolute {key} {value} cannot be staged')
            wanted.add(resolve_remote(volume_root, posixpath.relpath(posixpath.join(folder, value), volume_root)))
    return wanted


def stage_inputs(specs, root, volume, workers=16):
    """Copy every stage entry of `specs` to local disk once; returns {spec name: {stage name: local path}}."""
    jobs, staged = {}, {}  # Volume path -> (local path, size or None)
    for spec in specs:
        staged[spec['name']] = {}
        for name, entry in spec.get('stage', {}).items():
            entry, started = stage_entry(entry), time.time()
            remote = resolve_remote(spec['volume'], entry['path'])
            local = local_mirror(root, remote)
            staged[spec['name']][name] = local
            if 'manifests' in entry or 'files' in entry:
                wanted = set()
                for manifest in entry.get('manifests', []):
                    manifest_remote = resolve_remote(spec['volume'], posixpath.join(entry['path'], manifest))
                    manifest_local = local_mirror(root, manifest_remote)
                    if not manifest_local.is_file():
                        volume.download(manifest_remote, manifest_local)
                    wanted |= manifest_files(spec['volume'], manifest_remote, manifest_local)
                for item in entry.get('files', []):
                    wanted.add(resolve_remote(spec['volume'], posixpath.join(entry['path'], item)))
                found = {path: None for path in wanted}
            else:
                kind = volume.kind(remote)
                if kind is None:
                    raise FileNotFoundError(f'stage {name}: {remote} is not on the Volume')
                found = {remote: None} if kind == 'file' else dict(volume.walk(remote))
                for path in found:  # A folder may hold something it must not; refuse before reading any of it.
                    resolve_remote(spec['volume'], posixpath.relpath(path, spec['volume']))
            for path, size in found.items():
                jobs[path] = (local_mirror(root, path), size)
            log(f'stage {spec["name"]}:{name} <- {remote} ({len(found)} files listed in {time.time() - started:.0f}s)')

    def fetch(item):
        path, (local, size) = item
        if local.is_file() and (size is None or local.stat().st_size == size):
            return 0
        volume.download(path, local)
        return 1
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        copied = sum(pool.map(fetch, sorted(jobs.items())))
    log(f'staged {copied} new files ({len(jobs)} needed) in {time.time() - started:.0f}s')
    return staged


# ---------- Specs and commands ----------

def load_lanes(spec, code=CODE):
    """The arm specs this spec runs: itself, or its lanes (paths relative to the code root, or inline)."""
    if 'lanes' not in spec:
        lanes = [spec]
    else:
        lanes = []
        for item in spec['lanes']:
            lane = json.loads((code / item).read_text()) if isinstance(item, str) else dict(item)
            lane.setdefault('volume', spec['volume'])
            if 'lanes' in lane:
                raise ValueError(f"lane {lane.get('name')} has lanes of its own")
            lanes.append(lane)
        if not lanes:
            raise ValueError('lanes is empty')
    names = [lane.get('name', '') for lane in lanes] + ([spec['name']] if 'lanes' in spec else [])
    for name in names:
        if not NAME.fullmatch(name):
            raise ValueError(f'run name {name!r} must match {NAME.pattern}')
    if len(set(names)) != len(names):
        raise ValueError(f'run names must be unique: {names}')
    for lane in lanes:
        if not lane.get('volume', '').startswith('/Volumes/'):
            raise ValueError(f"{lane['name']}: volume must be a /Volumes/... path")
        if not lane.get('steps') and not lane.get('module'):
            raise ValueError(f"{lane['name']}: needs module + args or steps")
        check_stage(lane)
    return lanes


def expand(argument, staged, output):
    for name, path in staged.items():
        argument = argument.replace('{stage:' + name + '}', str(path))
    return argument.replace('{output}', str(output))


def lane_commands(lane, staged, output):
    steps = lane.get('steps') or [{'name': None, 'module': lane['module'], 'args': lane.get('args', [])}]
    commands = []
    for step in steps:
        target = output / step['name'] if step['name'] else output
        command = [sys.executable, '-m', step['module'], *(expand(a, staged, target) for a in step['args'])]
        left = [a for a in command if '{stage:' in a]
        if left:
            raise ValueError(f"{lane['name']}: unknown stage in {left}")
        commands.append((step['name'], command))
    return commands


def model_snapshots(lanes):
    """(model, revision) pairs named by --model / --revision in any lane's arguments."""
    pairs = set()
    for lane in lanes:
        for step in lane.get('steps') or [lane]:
            args = list(step.get('args', []))
            values = {flag: args[args.index(flag) + 1] for flag in ('--model', '--revision')
                      if flag in args and args.index(flag) + 1 < len(args)}
            if '--model' in values:
                pairs.add((values['--model'], values.get('--revision')))
    return sorted(pairs, key=str)


def gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sum(line.startswith('GPU ') for line in result.stdout.splitlines()) if result.returncode == 0 else None


def lane_environment(lane, index, count, base=None):
    env = dict(os.environ if base is None else base)
    if count > 1:
        for key in DISTRIBUTED_ENV:
            env.pop(key, None)
        env['CUDA_VISIBLE_DEVICES'] = str(index)
        env.setdefault('OMP_NUM_THREADS', str(max(1, (os.cpu_count() or count) // count)))
    env.update({key: str(value) for key, value in lane.get('env', {}).items()})
    return env


# ---------- Running ----------

class Lane:
    def __init__(self, spec, index, count, root, staged):
        self.spec, self.index, self.name = spec, index, spec['name']
        self.output = root / 'runs' / self.name
        self.log_path = root / 'logs' / f'{self.name}.log'
        self.remote = posixpath.join(spec['volume'], 'runs', self.name)
        self.commands = lane_commands(spec, staged, self.output)
        self.env = lane_environment(spec, index, count)
        self.prefix = f'[{self.name}] ' if count > 1 else ''
        self.codes, self.process, self.state = {}, None, {}
        self.status, self.started, self.finished = 'waiting', None, None
        self.stop = threading.Event()

    def run(self):
        self.status, self.started = 'running', time.time()
        try:
            self._run_steps()
        except Exception as error:  # Never leave a lane looking healthy after a launcher error.
            log(f'{self.prefix}lane error: {type(error).__name__}: {error}')
            self.codes.setdefault('launcher', 1)
        self.status, self.finished = ('failed' if self.failed() else 'done'), time.time()

    def _run_steps(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as stream:
            for name, command in self.commands:
                if self.stop.is_set():
                    break
                key = name or self.name
                text = ' '.join(command)
                log(f'{self.prefix}running: {text}')
                stream.write(f'### step {name}: {text}\n')
                stream.flush()
                try:
                    self.process = subprocess.Popen(command, cwd=CODE, env=self.env, stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT, text=True, errors='replace')
                except OSError as error:
                    stream.write(f'### could not start: {error}\n')
                    self.codes[key] = 127
                    continue
                with self.process.stdout:
                    for line in self.process.stdout:
                        stream.write(line)
                        stream.flush()
                        with _print_lock:
                            sys.stdout.write(self.prefix + line)
                            sys.stdout.flush()
                self.codes[key] = self.process.wait()
                log(f'{self.prefix}step {name} exit {self.codes[key]}')

    def failed(self):
        return any(self.codes.values()) or (self.stop.is_set() and len(self.codes) < len(self.commands))

    def terminate(self):
        self.stop.set()
        process = self.process
        if process and process.poll() is None:
            process.terminate()

    def summary(self):
        return {'name': self.name, 'gpu': self.index if self.env.get('CUDA_VISIBLE_DEVICES') == str(self.index)
                else None, 'status': self.status, 'steps': self.codes, 'started_utc': self.started,
                'finished_utc': self.finished, 'remote': self.remote}

    def sync(self, volume):
        sent = volume.upload_tree(self.output, self.remote, self.state)
        if self.log_path.is_file():
            volume.upload(self.log_path, f'{self.remote}/air_stdout.log')
        return sent


def _terminate_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def main(argv=None, files=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, help='JSON spec, relative to the code root or absolute')
    parser.add_argument('--sync-minutes', type=float, default=10)
    parser.add_argument('--stage-workers', type=int, default=16)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec) if Path(args.spec).is_absolute() else CODE / args.spec
    spec = json.loads(spec_path.read_text())
    lanes = load_lanes(spec)
    multi = 'lanes' in spec
    volume = Volume(files)
    if multi:
        gpus = gpu_count()
        if gpus is not None and len(lanes) > gpus:
            raise RuntimeError(f'{len(lanes)} lanes but {gpus} GPUs')
    remotes = [posixpath.join(lane['volume'], 'runs', lane['name']) for lane in lanes]
    parent_remote = posixpath.join(spec['volume'], 'runs', spec['name']) if multi else None
    for remote in remotes + ([parent_remote] if multi else []):
        if volume.kind(remote) is not None:
            raise FileExistsError(f'{remote} exists; choose a new run name')

    root = local_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    os.environ.setdefault('HF_HOME', str(root / 'hf'))  # Shared by every lane.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for lane, remote in zip(lanes, remotes):
        volume.write_text(f'{remote}/air_spec.json', json.dumps(lane, indent=2) + '\n')
    if multi:
        volume.write_text(f'{parent_remote}/air_spec.json', json.dumps({**spec, 'lanes': lanes}, indent=2) + '\n')
    log(f'code {CODE}, local {root}, {len(lanes)} lane(s): {", ".join(remotes)}')

    try:
        if spec.get('install_sam2', True):
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', '--no-deps', SAM2],
                           check=True, env={**os.environ, 'SAM2_BUILD_CUDA': '0'})
        staged = stage_inputs(lanes, root, volume, args.stage_workers)
        if multi and spec.get('prefetch_models', True):
            for model, revision in model_snapshots(lanes):  # Once, before the lanes share the cache.
                started = time.time()
                result = subprocess.run([sys.executable, '-c', 'import sys; from huggingface_hub import '
                                         'snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)',
                                         model, revision or ''])
                log(f'prefetch {model}@{revision}: exit {result.returncode} in {time.time() - started:.0f}s')
        runners = [Lane(lane, index, len(lanes), root, staged[lane['name']]) for index, lane in enumerate(lanes)]
        for runner in runners:
            volume.write_text(f'{runner.remote}/command.json', json.dumps(runner.commands, indent=2) + '\n')
    except Exception as error:  # Leave a readable reason in every run folder, then fail.
        for remote in remotes + ([parent_remote] if multi else []):
            try:
                volume.write_text(f'{remote}/air_exit.json', json.dumps(
                    {'returncode': 1, 'setup_error': f'{type(error).__name__}: {error}',
                     'finished_utc': time.time()}) + '\n')
            except Exception as upload_error:
                log(f'could not record the setup error in {remote}: {upload_error}')
        raise

    sync_lock = threading.Lock()

    def sync_all():
        with sync_lock:
            sent = sum(runner.sync(volume) for runner in runners)
            if multi:
                volume.write_text(f'{parent_remote}/lanes.json',
                                  json.dumps([runner.summary() for runner in runners], indent=2) + '\n')
            return sent

    stop = threading.Event()

    def periodic():
        while not stop.wait(args.sync_minutes * 60):
            try:
                log(f'sync: {sync_all()} files')
            except Exception as error:  # A transient upload failure must not stop training.
                log(f'sync failed, will retry: {type(error).__name__}: {error}')

    syncer = threading.Thread(target=periodic, daemon=True)
    syncer.start()
    threads, interrupted, previous = [], None, None
    try:
        if threading.current_thread() is threading.main_thread():
            # A job timeout or cancel sends SIGTERM: stop the lanes, then the final sync below still runs.
            previous = signal.signal(signal.SIGTERM, _terminate_on_signal)
        for runner in runners:
            if threads and spec.get('stagger_seconds', 60):
                _sleep(spec.get('stagger_seconds', 60))
            thread = threading.Thread(target=runner.run, name=runner.name, daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            while thread.is_alive():
                thread.join(5)
    except BaseException as error:
        interrupted = error
        log(f'interrupted ({type(error).__name__}); stopping lanes and syncing')
        for runner in runners:
            runner.terminate()
        deadline = time.time() + 120
        for thread in threads:
            thread.join(max(0.0, deadline - time.time()))
        for runner in runners:
            if runner.process and runner.process.poll() is None:
                runner.process.kill()
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        stop.set()
        syncer.join(timeout=60)
        for runner in runners:
            if runner.status in ('waiting', 'running'):
                runner.status = 'failed'
        try:
            log(f'final sync: {sync_all()} files')
        except Exception as error:
            log(f'final sync failed: {type(error).__name__}: {error}')
        for runner in runners:
            code = next((c for c in runner.codes.values() if c), 1 if runner.failed() else 0)
            try:
                volume.write_text(f'{runner.remote}/air_exit.json', json.dumps(
                    {'returncode': code, 'steps': runner.codes, 'interrupted': interrupted is not None,
                     'finished_utc': time.time()}) + '\n')
            except Exception as error:
                log(f'could not write {runner.remote}/air_exit.json: {type(error).__name__}: {error}')
    if interrupted is not None and not isinstance(interrupted, Exception):
        raise interrupted
    if multi:
        failed = [runner.name for runner in runners if runner.status != 'done']
        log(f'lanes finished; failed: {failed or "none"}')
        return 1 if failed or interrupted else 0
    runner = runners[0]
    log(f'exits {runner.codes}')
    return next((c for c in runner.codes.values() if c), 1 if runner.failed() or interrupted else 0)


if __name__ == '__main__':
    raise SystemExit(main())
