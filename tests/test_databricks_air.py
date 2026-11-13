"""Databricks AI Runtime launcher: Files-API staging and sync, multi-lane runs, code tarball bootstrap (CPU only)."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import types
import unittest
from unittest import mock

import yaml

from nucleus_rl.evaluate import read_manifest
from scripts.air import canary, make_run, run_arm

ROOT = Path(__file__).resolve().parents[1]
VOLUME = '/Volumes/cat/schema/processed/conny'
CONNY = '/Volumes/aistudio_dev_catalog/aios_segrlvr_dev/processed/conny'  # The canary config's volume.


class NotFound(Exception):
    """Name matches databricks.sdk.errors.NotFound."""


class FakeFiles:
    """WorkspaceClient().files backed by a local folder; records every call."""

    def __init__(self, base):
        self.base, self.calls, self.fail_next = Path(base), [], {}

    def local(self, path):
        return self.base / path.lstrip('/')

    def _record(self, name, path):
        self.calls.append((name, path))
        if self.fail_next.get(name):
            self.fail_next[name] -= 1
            raise ConnectionError('transient')

    def get_metadata(self, path):
        self._record('get_metadata', path)
        if not self.local(path).is_file():
            raise NotFound(path)
        return types.SimpleNamespace(content_length=self.local(path).stat().st_size)

    def get_directory_metadata(self, path):
        self._record('get_directory_metadata', path)
        if not self.local(path).is_dir():
            raise NotFound(path)

    def list_directory_contents(self, path):
        self._record('list', path)
        if not self.local(path).is_dir():
            raise NotFound(path)
        for item in sorted(self.local(path).iterdir()):
            remote = path.rstrip('/') + '/' + item.name
            yield types.SimpleNamespace(path=remote + ('/' if item.is_dir() else ''), is_directory=item.is_dir(),
                                        file_size=None if item.is_dir() else item.stat().st_size, name=item.name)

    def download(self, path):
        self._record('download', path)
        if not self.local(path).is_file():
            raise NotFound(path)
        return types.SimpleNamespace(contents=io.BytesIO(self.local(path).read_bytes()))

    def upload(self, path, stream, overwrite=False):
        self._record('upload', path)
        target = self.local(path)
        if target.exists() and not overwrite:
            raise FileExistsError(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stream.read())

    def create_directory(self, path):
        self._record('create_directory', path)
        self.local(path).mkdir(parents=True, exist_ok=True)

    def read(self, path):
        return self.local(path).read_text()

    def touched(self, prefix):
        return [path for name, path in self.calls if path.startswith(prefix)]


def write(path, text=''):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return path


def png(path, size=(8, 8), value=0):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('L', size, value).save(path)
    return path


FAKE_ARM = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    output, code = Path(sys.argv[sys.argv.index('--output') + 1]), int(sys.argv[sys.argv.index('--exit') + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / '.writing-checkpoint-1').mkdir(exist_ok=True)
    (output / '.writing-checkpoint-1' / 'partial.bin').write_text('half')
    keys = ('CUDA_VISIBLE_DEVICES', 'HF_HOME', 'OMP_NUM_THREADS', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
            'MASTER_ADDR', 'MASTER_PORT', 'LANE_TAG')
    (output / 'env.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}))
    for arg in sys.argv[1:]:
        if arg.startswith('/') and arg.endswith('.txt'):
            (output / 'input_seen.txt').write_text(Path(arg).read_text())
    print('fake arm says hello', flush=True)
    sys.exit(code)
''')


class AirTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', str(self.temp)]))
        self.files = FakeFiles(self.temp / 'remote')
        self.volume = run_arm.Volume(self.files, backoff=0)
        self.vol = self.files.local(VOLUME)
        self.local = self.temp / 'local'
        environ = mock.patch.dict(os.environ, {'SEGRLVR_LOCAL': str(self.local)})
        environ.start()
        self.addCleanup(environ.stop)
        for key in ('HF_HOME', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'OMP_NUM_THREADS',
                    'CUDA_VISIBLE_DEVICES'):
            os.environ.pop(key, None)


class StagingTests(AirTestCase):
    def make_mix(self):
        rows = []
        for dataset, crop in [('consep_v1', 'a'), ('puma_v1', 'b')]:
            png(self.vol / f'data/{dataset}/images/{crop}.png')
            png(self.vol / f'data/{dataset}/masks/{crop}.png', value=1)
            png(self.vol / f'data/{dataset}/images/unused.png')
            rows.append({'id': crop, 'patient_id': crop, 'split': 'validation', 'dataset': dataset.split('_')[0],
                         'image_path': f'../{dataset}/images/{crop}.png', 'mask_path': f'../{dataset}/masks/{crop}.png',
                         'gt_mask': f'../{dataset}/masks/{crop}.png'})
        write(self.vol / 'data/trustmix_v2/validation.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
        write(self.vol / 'data/trustmix_v2/train.jsonl', '{}\n')
        return rows

    def spec(self, stage):
        return {'name': 'arm', 'volume': VOLUME, 'module': 'x', 'stage': stage}

    def test_manifest_stage_copies_only_the_files_its_rows_name(self):
        self.make_mix()
        with redirect_stdout(io.StringIO()):
            staged = run_arm.stage_inputs([self.spec({'data': {'path': 'data',
                                                                'manifests': ['trustmix_v2/validation.jsonl']}})],
                                          self.local, self.volume, workers=4)
        local = staged['arm']['data']
        self.assertEqual(local, self.local / 'inputs' / VOLUME.lstrip('/') / 'data')
        copied = sorted(p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file())
        self.assertEqual(copied, ['consep_v1/images/a.png', 'consep_v1/masks/a.png', 'puma_v1/images/b.png',
                                  'puma_v1/masks/b.png', 'trustmix_v2/validation.jsonl'])
        rows = read_manifest(local / 'trustmix_v2/validation.jsonl')  # Relative ../ paths resolve locally.
        self.assertEqual([Path(r['image_path']).name for r in rows], ['a.png', 'b.png'])
        self.assertFalse([p for p in self.files.touched(VOLUME + '/data/consep_v1/images/unused')])
        self.assertFalse([name for name, _ in self.files.calls if name == 'upload'])

    def test_folder_and_file_stage_are_copied_once(self):
        write(self.vol / 'checkpoints/sam.pt', 'weights')
        write(self.vol / 'runs/c1/checkpoints/checkpoint-128/adapter.safetensors', 'lora')
        write(self.vol / 'runs/c1/checkpoints/checkpoint-128/sub/config.json', '{}')
        spec = self.spec({'sam': 'checkpoints/sam.pt', 'init': 'runs/c1/checkpoints/checkpoint-128'})
        with redirect_stdout(io.StringIO()):
            staged = run_arm.stage_inputs([spec], self.local, self.volume)
            self.assertEqual(staged['arm']['sam'].read_text(), 'weights')
            self.assertEqual((staged['arm']['init'] / 'sub/config.json').read_text(), '{}')
            downloads = len([c for c in self.files.calls if c[0] == 'download'])
            self.assertEqual(downloads, 3)
            run_arm.stage_inputs([spec, {**spec, 'name': 'other'}], self.local, self.volume)
        self.assertEqual(len([c for c in self.files.calls if c[0] == 'download']), downloads)

    def test_test_and_held_out_data_are_refused_before_any_read(self):
        bad = ['../../../raw/monuseg/test', 'raw/monuseg/test/x.tif', 'data/consep/Test', 'data/puma_heldout',
               '../other_user/data', '/Volumes/cat/schema/raw/monuseg/train']
        for path in bad:
            with self.subTest(path=path), self.assertRaises(PermissionError), redirect_stdout(io.StringIO()):
                run_arm.stage_inputs([self.spec({'x': path})], self.local, self.volume)
        self.assertEqual(self.files.calls, [])
        with self.assertRaises(PermissionError):
            run_arm.load_lanes(self.spec({'x': {'path': 'data', 'manifests': ['../../raw/monuseg/test/m.jsonl']}}))

    def test_manifest_rows_pointing_at_test_data_or_absolute_paths_are_refused(self):
        rows = [{'id': 'a', 'image_path': '../../../../raw/monuseg/test/a.png', 'mask_path': 'm.png'}]
        write(self.vol / 'data/mix/validation.jsonl', json.dumps(rows[0]) + '\n')
        write(self.vol / 'data/abs/validation.jsonl', json.dumps({'image_path': '/n/fasrc/a.png'}) + '\n')
        for folder, error in [('mix', PermissionError), ('abs', ValueError)]:
            with self.subTest(folder=folder), self.assertRaises(error), redirect_stdout(io.StringIO()):
                run_arm.stage_inputs([self.spec({'d': {'path': f'data/{folder}', 'manifests': ['validation.jsonl']}})],
                                     self.local, self.volume)
        self.assertEqual({name for name, _ in self.files.calls}, {'download'})
        self.assertTrue(all(path.endswith('validation.jsonl') for _, path in self.files.calls))

    def test_folder_holding_a_test_split_is_refused_before_downloading(self):
        write(self.vol / 'data/consep/train/a.png', 'x')
        write(self.vol / 'data/consep/test/b.png', 'x')
        with self.assertRaises(PermissionError), redirect_stdout(io.StringIO()):
            run_arm.stage_inputs([self.spec({'d': 'data/consep'})], self.local, self.volume)
        self.assertNotIn('download', {name for name, _ in self.files.calls})

    def test_missing_input_fails_and_transient_errors_are_retried(self):
        with self.assertRaises(FileNotFoundError), redirect_stdout(io.StringIO()):
            run_arm.stage_inputs([self.spec({'x': 'checkpoints/none.pt'})], self.local, self.volume)
        write(self.vol / 'checkpoints/sam.pt', 'weights')
        self.files.fail_next['download'] = 2
        with redirect_stdout(io.StringIO()):
            staged = run_arm.stage_inputs([self.spec({'sam': 'checkpoints/sam.pt'})], self.local, self.volume)
        self.assertEqual(staged['arm']['sam'].read_text(), 'weights')

    def test_upload_tree_skips_hidden_and_unchanged_files_and_never_deletes(self):
        out = self.temp / 'out'
        write(out / 'metrics.json', '1')
        write(out / '.writing-checkpoint-2/model.bin', 'partial')
        write(out / 'checkpoints/checkpoint-1/model.bin', 'done')
        state = {}
        self.assertEqual(self.volume.upload_tree(out, VOLUME + '/runs/r', state), 2)
        self.assertEqual(self.volume.upload_tree(out, VOLUME + '/runs/r', state), 0)
        write(out / 'metrics.json', '22')
        subprocess.run(['rm', '-rf', str(out / 'checkpoints')])
        self.assertEqual(self.volume.upload_tree(out, VOLUME + '/runs/r', state), 1)
        self.assertEqual(self.files.read(VOLUME + '/runs/r/metrics.json'), '22')
        self.assertTrue(self.files.local(VOLUME + '/runs/r/checkpoints/checkpoint-1/model.bin').is_file())
        self.assertFalse(self.files.local(VOLUME + '/runs/r/.writing-checkpoint-2').exists())


class RunArmTests(AirTestCase):
    def setUp(self):
        super().setUp()
        package = self.temp / 'pkg'
        write(package / 'fakearm.py', FAKE_ARM)
        write(self.vol / 'inputs/hello.txt', 'staged input')
        self.base_env = {'PYTHONPATH': str(package)}

    def arm(self, name, exit_code=0, **extra):
        return {'name': name, 'volume': VOLUME, 'module': 'fakearm', 'install_sam2': False,
                'env': self.base_env, 'stage': {'hello': 'inputs/hello.txt'},
                'args': ['--output', '{output}', '--exit', str(exit_code), '{stage:hello}'], **extra}

    def run_spec(self, spec):
        path = write(self.temp / f"{spec['name']}.json", json.dumps(spec))
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_arm.main(['--spec', str(path), '--sync-minutes', '60'], files=self.files)
        return code, out.getvalue()

    def remote(self, name, file):
        return self.files.local(f'{VOLUME}/runs/{name}/{file}')

    def test_single_arm_stages_runs_and_syncs_with_the_files_api(self):
        code, out = self.run_spec(self.arm('s1'))
        self.assertEqual(code, 0)
        self.assertIn('fake arm says hello', out)
        self.assertEqual(self.remote('s1', 'input_seen.txt').read_text(), 'staged input')
        self.assertIn('fake arm says hello', self.remote('s1', 'air_stdout.log').read_text())
        self.assertEqual(json.loads(self.remote('s1', 'air_exit.json').read_text())['returncode'], 0)
        self.assertEqual(json.loads(self.remote('s1', 'air_spec.json').read_text())['name'], 's1')
        self.assertTrue(self.remote('s1', 'command.json').is_file())
        self.assertFalse(self.remote('s1', '.writing-checkpoint-1').exists())
        env = json.loads(self.remote('s1', 'env.json').read_text())
        self.assertEqual(env['HF_HOME'], str(self.local / 'hf'))

    def test_existing_run_folder_is_refused_without_writing(self):
        write(self.remote('s2', 'old.txt'), 'keep')
        with self.assertRaises(FileExistsError):
            self.run_spec(self.arm('s2'))
        self.assertNotIn('upload', {name for name, _ in self.files.calls})
        self.assertEqual(self.remote('s2', 'old.txt').read_text(), 'keep')

    def test_failed_step_returns_its_code_runs_later_steps_and_still_syncs(self):
        spec = self.arm('s3')
        step_args = lambda code: ['--output', '{output}', '--exit', str(code)]
        spec.update(module=None, steps=[{'name': 'bad', 'module': 'fakearm', 'args': step_args(3)},
                                        {'name': 'good', 'module': 'fakearm', 'args': step_args(0)}])
        code, _ = self.run_spec(spec)
        self.assertEqual(code, 3)
        exit_info = json.loads(self.remote('s3', 'air_exit.json').read_text())
        self.assertEqual(exit_info['steps'], {'bad': 3, 'good': 0})
        self.assertTrue(self.remote('s3', 'bad/env.json').is_file())
        self.assertTrue(self.remote('s3', 'good/env.json').is_file())

    def test_lanes_get_one_gpu_each_no_distributed_env_shared_cache_and_stagger(self):
        lane_file = write(self.temp / 'lane-c.json', json.dumps(self.arm('lane-c', env={**self.base_env,
                                                                                         'LANE_TAG': 'c'})))
        spec = {'name': 'multi', 'volume': VOLUME, 'install_sam2': False, 'prefetch_models': False,
                'stagger_seconds': 7, 'lanes': [self.arm('lane-a'), self.arm('lane-b', exit_code=5), str(lane_file)]}
        sleeps = []
        distributed = {'WORLD_SIZE': '8', 'RANK': '3', 'LOCAL_RANK': '3', 'MASTER_ADDR': 'h', 'MASTER_PORT': '1'}
        with mock.patch.dict(os.environ, distributed), mock.patch.object(run_arm, 'gpu_count', return_value=8), \
                mock.patch.object(run_arm, '_sleep', side_effect=sleeps.append):
            code, out = self.run_spec(spec)
        self.assertEqual(code, 1)
        self.assertEqual(sleeps, [7, 7])
        self.assertIn('[lane-b] fake arm says hello', out)
        for index, name in enumerate(['lane-a', 'lane-b', 'lane-c']):
            env = json.loads(self.remote(name, 'env.json').read_text())
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], str(index))
            self.assertEqual([env[k] for k in distributed], [None] * 5)
            self.assertEqual(env['HF_HOME'], str(self.local / 'hf'))
            self.assertIsNotNone(env['OMP_NUM_THREADS'])
            self.assertIn('fake arm says hello', self.remote(name, 'air_stdout.log').read_text())
            self.assertEqual(self.remote(name, 'input_seen.txt').read_text(), 'staged input')
        self.assertEqual(json.loads(self.remote('lane-c', 'env.json').read_text())['LANE_TAG'], 'c')
        lanes = json.loads(self.remote('multi', 'lanes.json').read_text())
        self.assertEqual([lane['status'] for lane in lanes], ['done', 'failed', 'done'])
        self.assertEqual(json.loads(self.remote('lane-b', 'air_exit.json').read_text())['returncode'], 5)
        self.assertEqual(len([p for n, p in self.files.calls if n == 'download' and p.endswith('hello.txt')]), 1)

    def test_setup_failure_is_recorded_in_the_run_folder(self):
        spec = self.arm('s4')
        spec['stage'] = {'hello': 'inputs/missing.txt'}
        with self.assertRaises(FileNotFoundError):
            self.run_spec(spec)
        self.assertIn('FileNotFoundError', json.loads(self.remote('s4', 'air_exit.json').read_text())['setup_error'])

    def test_sigterm_stops_the_lane_and_still_syncs(self):
        spec = self.arm('s5')
        spec['args'] = ['--output', '{output}', '--exit', '0']
        package = Path(self.base_env['PYTHONPATH'])
        write(package / 'slowarm.py', FAKE_ARM.replace("print('fake arm says hello', flush=True)",
                                                        "print('started', flush=True); time.sleep(60)"))
        spec['module'] = 'slowarm'
        timer = __import__('threading').Timer(1.5, lambda: os.kill(os.getpid(), __import__('signal').SIGTERM))
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(SystemExit) as raised:
            self.run_spec(spec)
        self.assertEqual(raised.exception.code, 128 + 15)
        exit_info = json.loads(self.remote('s5', 'air_exit.json').read_text())
        self.assertTrue(exit_info['interrupted'])
        self.assertNotEqual(exit_info['returncode'], 0)
        self.assertTrue(self.remote('s5', 'env.json').is_file())

    def test_lanes_are_refused_when_gpus_or_names_do_not_fit(self):
        spec = {'name': 'multi', 'volume': VOLUME, 'install_sam2': False,
                'lanes': [self.arm('a'), self.arm('b'), self.arm('c')]}
        with mock.patch.object(run_arm, 'gpu_count', return_value=2), self.assertRaises(RuntimeError):
            self.run_spec(spec)
        self.assertEqual(self.files.calls, [])
        for lanes in ([self.arm('a'), self.arm('a')], [{**self.arm('a'), 'lanes': []}], [self.arm('multi')],
                      [self.arm('bad name')], [{**self.arm('a'), 'volume': '/dbfs/x'}]):
            with self.subTest(lanes=[lane['name'] for lane in lanes]), self.assertRaises(ValueError):
                run_arm.load_lanes({**spec, 'lanes': lanes})

    def test_lane_environment_and_model_prefetch_pairs(self):
        env = run_arm.lane_environment({'env': {'X': 1}}, 3, 8, base={'RANK': '1', 'MASTER_PORT': '9', 'PATH': 'p'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
        self.assertNotIn('RANK', env)
        self.assertEqual((env['PATH'], env['X']), ('p', '1'))
        single = run_arm.lane_environment({}, 0, 1, base={'RANK': '0'})
        self.assertEqual(single, {'RANK': '0'})
        lanes = [{'args': ['--model', 'Q/27B', '--revision', 'abc']},
                 {'steps': [{'args': ['--model', 'Q/27B', '--revision', 'abc']}, {'args': ['--model', 'Q/9B']}]}]
        self.assertEqual(run_arm.model_snapshots(lanes), [('Q/27B', 'abc'), ('Q/9B', None)])


class MakeRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', str(self.temp)]))

    def fake_repo(self):
        repo = self.temp / 'repo'
        write(repo / 'nucleus_rl/__init__.py', 'VALUE = 1\n')
        write(repo / 'nucleus_rl/__pycache__/x.pyc', b'\0')
        write(repo / 'scripts/air/run_arm.py', textwrap.dedent('''
            import os, sys
            print('stub run_arm', sys.argv[1:], os.getcwd(), flush=True)
        '''))
        write(repo / 'configs/air/x.json', '{}')
        write(repo / 'protocols/p.json', '{}')
        write(repo / 'data/big.bin', b'\0' * 10)
        return repo

    def test_code_tarball_is_reproducible_and_holds_only_code(self):
        repo = self.fake_repo()
        first, sha = make_run.build_code_tarball(repo, 'abc', False)
        os.utime(repo / 'nucleus_rl/__init__.py', (1, 1))
        self.assertEqual(make_run.build_code_tarball(repo, 'abc', False)[1], sha)
        self.assertNotEqual(make_run.build_code_tarball(repo, 'abc', True)[1], sha)
        with tarfile.open(fileobj=io.BytesIO(first), mode='r:gz') as archive:
            names = sorted(archive.getnames())
        self.assertEqual(names, ['air_code.json', 'configs/air/x.json', 'nucleus_rl/__init__.py', 'protocols/p.json',
                                 'scripts/air/run_arm.py'])
        self.assertTrue(make_run.remote_tarball({'volume': VOLUME}, sha).endswith(f'/code/files-api/segrlvr-code-{sha[:16]}.tar.gz'))

    def test_yaml_command_downloads_checks_and_runs_the_tarball_without_code_source(self):
        repo = self.fake_repo()
        data, sha = make_run.build_code_tarball(repo, 'abc', False)
        spec = {'name': 'arm-1', 'volume': VOLUME, 'timeout_minutes': 30}
        tarball = make_run.remote_tarball(spec, sha)
        args = types.SimpleNamespace(accelerator='GPU_1xA10', timeout_minutes=60)
        config = yaml.safe_load(make_run.yaml_text(spec, 'configs/air/x.json', tarball, sha, args))
        self.assertNotIn('code_source', config)
        self.assertEqual((config['experiment_name'], config['timeout_minutes']), ('arm-1', 30))
        self.assertEqual(config['compute']['accelerator_type'], 'GPU_1xA10')
        sdk = self.temp / 'sdk'
        write(sdk / 'databricks/__init__.py')
        write(sdk / 'databricks/sdk/__init__.py', textwrap.dedent('''
            import io, os
            from pathlib import Path
            class NotFound(Exception):
                pass
            class _Files:
                def download(self, path):
                    local = Path(os.environ['FAKE_VOLUME']) / path.lstrip('/')
                    if not local.is_file():
                        raise NotFound(path)
                    return type('D', (), {'contents': io.BytesIO(local.read_bytes())})()
            class WorkspaceClient:
                def __init__(self):
                    self.files = _Files()
        '''))
        bin_dir = self.temp / 'bin'
        write(bin_dir / 'python', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n').chmod(0o755)
        remote = self.temp / 'remote'
        env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'PYTHONPATH': str(sdk),
               'FAKE_VOLUME': str(remote), 'SEGRLVR_LOCAL': str(self.temp / 'node')}
        run = lambda: subprocess.run(['bash', '-c', config['command']], env=env, capture_output=True, text=True)
        missing = run()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('NotFound', missing.stderr)
        write(remote / tarball.lstrip('/'), data[:-4] + b'oops')
        corrupt = run()
        self.assertNotEqual(corrupt.returncode, 0)
        write(remote / tarball.lstrip('/'), data)
        result = run()
        self.assertEqual(result.returncode, 0, result.stderr)
        code_dir = self.temp / 'node' / 'code' / sha[:16]
        self.assertIn(f"stub run_arm ['--spec', 'configs/air/x.json', '--sync-minutes', '10'] {code_dir}",
                      result.stdout)
        self.assertEqual(json.loads((code_dir / 'air_code.json').read_text())['commit'], 'abc')
        self.assertIn('already unpacked', run().stdout)

    def fake_cli(self, existing=()):
        calls = []

        def fake_run(command, capture_output=True, text=True):
            calls.append(command)
            ok = lambda out='': subprocess.CompletedProcess(command, 0, out, '')
            if command[0] == 'git':
                return ok('abc123\n' if 'rev-parse' in command else '')
            if command[1:3] == ['fs', 'ls']:
                parent = command[3][len('dbfs:'):]
                names = [path.rsplit('/', 1)[1] for path in existing if path.rsplit('/', 1)[0] == parent]
                if not names:
                    return subprocess.CompletedProcess(command, 1, '', 'Error: no such directory')
                return ok('\n'.join(names) + '\n')
            if command[1:3] == ['air', 'run']:
                return ok('{\n"data": {"run_id": "42"}\n}\n')
            return ok()
        return calls, fake_run

    def submit(self, argv, existing=()):
        calls, fake_run = self.fake_cli(existing)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(make_run.subprocess, 'run', side_effect=fake_run), redirect_stdout(out), \
                redirect_stderr(err):
            code = make_run.main([str(ROOT / 'configs/air/canary-files-api.json'),
                                  '--receipts', str(self.temp / 'receipts.jsonl'), *argv])
        return code, calls, out.getvalue() + err.getvalue()

    def test_submit_uploads_code_after_preflight_and_writes_receipt(self):
        manifest = CONNY + '/data/trustmix_v2/validation.jsonl'
        code, calls, out = self.submit([], existing=[manifest])
        self.assertEqual(code, 0, out)
        cli = [c[1:] for c in calls if c[0] == 'databricks']
        copies = [c for c in cli if c[:2] == ['fs', 'cp']]
        self.assertEqual(len(copies), 1)
        self.assertTrue(copies[0][3].startswith(f'dbfs:{CONNY}/code/files-api/segrlvr-code-'))
        air = next(c for c in cli if c[:2] == ['air', 'run'])
        self.assertIn('--idempotency-key', air)
        receipt = json.loads((self.temp / 'receipts.jsonl').read_text())
        self.assertEqual((receipt['run_id'], receipt['lanes'], receipt['commit']),
                         ('42', ['canary-files-api-a10'], 'abc123'))
        self.assertEqual(len(receipt['code_sha256']), 64)
        config = yaml.safe_load(Path(receipt['yaml']).read_text())
        self.assertEqual(config['compute']['accelerator_type'], 'GPU_1xA10')
        self.assertIn(receipt['code_sha256'], config['command'])
        with self.assertRaises(SystemExit):  # The name already has a receipt.
            self.submit([], existing=[manifest])

    def test_preflight_refuses_missing_inputs_and_existing_run_folders(self):
        run_folder = CONNY + '/runs/canary-files-api-a10'
        code, calls, out = self.submit([], existing=[run_folder])
        self.assertEqual(code, 2)
        self.assertIn('exists; choose a new run name', out)
        self.assertIn('trustmix_v2/validation.jsonl is not on the Volume', out)
        self.assertFalse([c for c in calls if c[1:3] in (['fs', 'cp'], ['air', 'run'])])

    def test_dry_run_touches_no_volume(self):
        code, calls, out = self.submit(['--dry-run'])
        self.assertEqual(code, 0, out)
        self.assertEqual([c[1:3] for c in calls if c[0] == 'databricks'], [['air', 'run']])
        self.assertFalse((self.temp / 'receipts.jsonl').exists())


class ConfigTests(unittest.TestCase):
    def test_every_air_config_is_valid_and_reads_no_test_data(self):
        for path in sorted((ROOT / 'configs/air').glob('*.json')):
            with self.subTest(config=path.name):
                spec = json.loads(path.read_text())
                lanes = run_arm.load_lanes(spec, ROOT)
                for lane in lanes:
                    self.assertTrue(lane['volume'].endswith('/processed/conny'))
                    staged = {name: Path('/staged') / name for name in lane.get('stage', {})}
                    run_arm.lane_commands(lane, staged, Path('/out'))

    def test_canary_stages_only_trustmix_validation_and_checks_counts(self):
        spec = json.loads((ROOT / 'configs/air/canary-files-api.json').read_text())
        self.assertEqual(spec['accelerator'], 'GPU_1xA10')
        self.assertEqual(spec['stage'], {'data': {'path': 'data', 'manifests': ['trustmix_v2/validation.jsonl']}})
        self.assertEqual(spec['module'], 'scripts.air.canary')
        self.assertLessEqual(spec['timeout_minutes'], 720)


class CanaryTests(unittest.TestCase):
    def test_canary_reports_counts_and_writes_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / 'data'
            rows = []
            for i, dataset in enumerate(['monuseg', 'puma', 'puma']):
                png(data / f'{dataset}_v1/images/{i}.png')
                png(data / f'{dataset}_v1/masks/{i}.png', value=1)
                rows.append({'id': str(i), 'patient_id': str(i), 'split': 'validation', 'dataset': dataset,
                             'image_path': f'../{dataset}_v1/images/{i}.png',
                             'mask_path': f'../{dataset}_v1/masks/{i}.png', 'gt_mask': f'../{dataset}_v1/masks/{i}.png'})
            manifest = write(data / 'mix/validation.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
            out = Path(temp) / 'out'
            with mock.patch.object(canary, 'package_report', return_value=({}, [])), redirect_stdout(io.StringIO()):
                self.assertEqual(canary.main(['--manifest', str(manifest), '--output', str(out),
                                              '--expect', 'monuseg=1', 'puma=2']), 0)
            report = json.loads((out / 'canary.json').read_text())
            self.assertEqual(report['counts'], {'monuseg': 1, 'puma': 2})
            self.assertTrue((out / 'CANARY_OK').is_file())
            with mock.patch.object(canary, 'package_report', return_value=({}, [])), redirect_stdout(io.StringIO()):
                self.assertEqual(canary.main(['--manifest', str(manifest), '--output', str(out / 'b'),
                                              '--expect', 'consep=42']), 1)
            self.assertTrue((out / 'b/CANARY_FAILED').is_file())


if __name__ == '__main__':
    unittest.main()
