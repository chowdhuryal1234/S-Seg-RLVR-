import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import load_official30_cohort
from nucleus_rl.evaluate import read_manifest
from nucleus_rl.sft_data import validate_sft_splits
from scripts import build_trustmix
from scripts.prepare_external import crop_cap, crop_rows


MONUSEG_TRAIN, MONUSEG_VAL = load_official30_cohort()['image_ids'][:2]


def write_pair(folder, sample_id, nuclei):
    """8x8 RGB crop and a uint16 mask with `nuclei` single-pixel instances."""
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    (folder / 'masks').mkdir(exist_ok=True)
    mask = np.zeros((8, 8), dtype=np.uint16)
    for index in range(nuclei):
        mask[index // 8, index % 8] = index + 1
    rgb = np.full((8, 8, 3), sum(sample_id.encode()) % 200, dtype=np.uint8)
    rgb[0, 0] = [len(sample_id), nuclei, 7]
    Image.fromarray(rgb).save(folder / 'images' / f'{sample_id}.png')
    Image.fromarray(mask).save(folder / 'masks' / f'{sample_id}.png')


def monuseg_row(folder, image_id, split, x, nuclei=3):
    sample_id = f'{image_id}_x{x}_y0'
    write_pair(folder, sample_id, nuclei)
    return {'id': sample_id, 'source_image_id': image_id, 'patient_id': '-'.join(image_id.split('-')[:3]),
            'split': split, 'instance_count': nuclei, 'image_path': f'images/{sample_id}.png',
            'mask_path': f'masks/{sample_id}.png', 'gt_mask': f'masks/{sample_id}.png',
            'path_base': 'manifest_directory'}


def consep_row(folder, number, split, x, nuclei=2):
    sample_id = f'consep_train_{number}_x{x}_y0'
    write_pair(folder, sample_id, nuclei)
    return {'id': sample_id, 'dataset': 'consep', 'source_image_id': f'consep_train_{number}',
            'patient_id': f'consep_train_{number}', 'split': split,
            'source_image_member': f'CoNSeP/Train/Images/train_{number}.png', 'instance_count': nuclei,
            'image_path': f'images/{sample_id}.png', 'mask_path': f'masks/{sample_id}.png',
            'gt_mask': f'masks/{sample_id}.png', 'path_base': 'manifest_directory'}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def build_args(output, sources, validation, **extra):
    values = {'output': str(output), 'source': sources, 'validation': validation, 'validation_dense': [],
              'heldout': [], 'max_objects': 8, 'seed': 1}
    values.update(extra)
    return argparse.Namespace(**values)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        mon, con = self.root / 'monuseg', self.root / 'consep'
        write_jsonl(mon / 'train.jsonl', [monuseg_row(mon, MONUSEG_TRAIN, 'train', x) for x in (0, 128, 256, 384)])
        write_jsonl(mon / 'validation.jsonl', [monuseg_row(mon, MONUSEG_VAL, 'validation', 0)])
        write_jsonl(con / 'train.jsonl', [consep_row(con, 1, 'train', x) for x in (0, 128)])
        write_jsonl(con / 'validation.jsonl', [consep_row(con, 2, 'validation', 0)])
        self.mon, self.con = mon, con

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, name='mix', sources=None, **extra):
        sources = sources or [f'monuseg_sparse={self.mon}:3', f'consep_sparse={self.con}']
        args = build_args(self.root / name, sources, [f'monuseg={self.mon}', f'consep={self.con}'], **extra)
        return build_trustmix.build(args), self.root / name


class SpecTest(unittest.TestCase):
    def test_parse_folder_cap_and_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'puma_v2'
            write_jsonl(folder / 'train.jsonl', [])
            write_jsonl(folder / 'validation.jsonl', [])
            spec = build_trustmix.parse_spec(f'puma_sparse={folder}:512', 'train', True)
            self.assertEqual((spec['dataset'], spec['cap'], spec['path'].name), ('puma', 512, 'train.jsonl'))
            self.assertIsNone(build_trustmix.parse_spec(f'puma={folder}:all', 'train', True)['cap'])
            self.assertEqual(build_trustmix.parse_spec(f'puma={folder}', 'validation', False)['path'].name,
                             'validation.jsonl')
            with self.assertRaises(ValueError):
                build_trustmix.parse_spec(f'pannuke={folder}', 'train', True)

    def test_subsample_is_seeded_order_preserving_and_capped(self):
        items = list(range(50))
        a = build_trustmix.subsample(items, 10, 7, 'puma')
        self.assertEqual(a, build_trustmix.subsample(items, 10, 7, 'puma'))
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, build_trustmix.subsample(items, 10, 8, 'puma'))
        self.assertEqual(build_trustmix.subsample(items, 99, 7, 'puma'), items)
        self.assertEqual(build_trustmix.subsample(items, None, 7, 'puma'), items)

    def test_relocate_and_restore_round_trip(self):
        row = {'id': 'a', 'image_path': 'images/a.png', 'mask_path': 'masks/a.png', 'gt_mask': 'masks/a.png'}
        source, out = Path('/data/consep_v2'), Path('/data/trustmix')
        moved = build_trustmix.relocate(row, source, out)
        self.assertEqual(moved['image_path'], '../consep_v2/images/a.png')
        self.assertEqual(build_trustmix.restore({**moved, 'dataset': 'consep'}, source, out, False), row)


class BuildTest(FixtureCase):
    def test_build_tags_relocates_and_verifies(self):
        report, out = self.build()
        rows = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
        self.assertEqual([r['dataset'] for r in rows], ['monuseg'] * 3 + ['consep'] * 2)
        self.assertTrue(rows[0]['image_path'].startswith('../monuseg/images/'))
        self.assertEqual(report['counts']['train'], {'consep': {'tiles': 2, 'nuclei': 4},
                                                     'monuseg': {'tiles': 3, 'nuclei': 9}})
        self.assertEqual(report['outputs']['manifest.jsonl']['rows'], 7)
        result = build_trustmix.verify(out)
        self.assertEqual(result['splits']['validation']['monuseg'],
                         {'rows': 1, 'content_identical': 1, 'source_line_bytes_identical': 1})
        # The manifest goes through the same audit sft.py/grpo.py run without --cohort-protocol.
        audit = validate_sft_splits(read_manifest(out / 'train.jsonl'), read_manifest(out / 'validation.jsonl'))
        self.assertEqual(audit['train_crops'], 5)

    def test_same_arguments_give_identical_files(self):
        first, _ = self.build('a')
        second, _ = self.build('b')
        self.assertEqual(first['outputs']['train.jsonl'], second['outputs']['train.jsonl'])

    def test_verify_detects_an_edited_row(self):
        _, out = self.build()
        path = out / 'validation.jsonl'
        path.write_text(path.read_text().replace('"instance_count": 3', '"instance_count": 4', 1))
        with self.assertRaises(ValueError):
            build_trustmix.verify(out)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate id'):
            self.build(sources=[f'consep_a={self.con}', f'consep_b={self.con}'])
        self.assertFalse((self.root / 'mix').exists())

    def test_heldout_group_overlap_is_rejected(self):
        stitch = self.root / 'consep_stitch'
        write_jsonl(stitch / 'tiles.jsonl', [{'id': 't', 'split': 'validation', 'patient_id': 'consep_train_1',
                                              'source_image_id': 'consep_train_1'}])
        with self.assertRaisesRegex(ValueError, 'group overlap'):
            self.build(heldout=[str(stitch)])
        self.assertFalse((self.root / 'mix').exists())
        self.assertFalse((self.root / 'mix.partial').exists())

    def test_rows_outside_declared_train_groups_are_rejected(self):
        (self.con / 'preparation_report.json').write_text(json.dumps({'group_splits': {'train': ['consep_train_9']}}))
        with self.assertRaisesRegex(ValueError, 'outside the declared train groups'):
            self.build()

    def test_dense_training_rows_need_a_higher_object_cap(self):
        dense = self.root / 'consep_dense'
        write_jsonl(dense / 'train.jsonl', [consep_row(dense, 3, 'train', 0, nuclei=12)])
        sources = [f'consep_sparse={self.con}', f'consep_dense={dense}']
        with self.assertRaisesRegex(ValueError, 'max-objects'):
            self.build(sources=sources)
        report, _ = self.build(sources=sources, max_objects=32)
        self.assertEqual(report['totals']['train'], {'tiles': 3, 'nuclei': 16})

    def test_compare_reports_identity_and_differences(self):
        copy = self.root / 'consep_copy'
        write_jsonl(copy / 'validation.jsonl', [consep_row(copy, 2, 'validation', 0)])
        self.assertTrue(build_trustmix.compare(self.con / 'validation.jsonl', copy / 'validation.jsonl')['identical'])
        write_jsonl(copy / 'validation.jsonl', [consep_row(copy, 2, 'validation', 0, nuclei=3)])
        self.assertFalse(build_trustmix.compare(self.con / 'validation.jsonl', copy / 'validation.jsonl')['identical'])


class PrepareExternalPoolsTest(unittest.TestCase):
    @staticmethod
    def labels(dense_corner):
        """256x256 instance map: three 128 px crops with 2 nuclei and one with 12."""
        labels = np.zeros((256, 256), dtype=np.int32)
        next_id = 1
        for y0, x0 in ((0, 0), (0, 128), (128, 0), (128, 128)):
            count = 12 if (y0, x0) == dense_corner else 2
            for k in range(count):
                y, x = y0 + 10 + 20 * (k // 4), x0 + 10 + 20 * (k % 4)
                labels[y:y + 5, x:x + 5] = next_id
                next_id += 1
        return labels

    def run_rows(self, out, dense_train_crops):
        sources = [(f'consep_train_{i}', f'consep_train_{i}', Image.new('RGB', (256, 256)), self.labels((128, 128)), {})
                   for i in (1, 2)]
        split_of = {'consep_train_1': 'train', 'consep_train_2': 'validation'}
        args = argparse.Namespace(train_crops=None, val_crops=64, dense_val_crops=64, seed=3,
                                  dense_train_crops=dense_train_crops)
        return crop_rows(iter(sources), split_of, out, 'consep', args)

    def test_dense_train_pool_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, candidates, _ = self.run_rows(Path(tmp) / 'a', 0)
            self.assertNotIn('train_dense', files)
            self.assertNotIn('train_9_32', candidates)
            self.assertFalse((Path(tmp) / 'a' / 'train_dense.jsonl').exists())
            self.assertEqual(len(files['train']), 3)  # "all": every eligible sparse crop

    def test_dense_train_pool_writes_its_own_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, candidates, _ = self.run_rows(Path(tmp) / 'b', None)
            self.assertEqual(candidates['train_9_32'], 1)
            self.assertEqual([r['instance_count'] for r in files['train_dense']], [12])
            manifest = (Path(tmp) / 'b' / 'manifest.jsonl').read_text().splitlines()
            self.assertEqual(len(manifest), len(files['train']) + len(files['validation']))

    def test_dense_train_pool_leaves_other_selections_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            without, _, _ = self.run_rows(Path(tmp) / 'a', 0)
            with_dense, _, _ = self.run_rows(Path(tmp) / 'b', None)
            for name in ('train', 'validation', 'validation_dense'):
                self.assertEqual(without[name], with_dense[name])
                self.assertEqual((Path(tmp) / 'a' / f'{name}.jsonl').read_bytes(),
                                 (Path(tmp) / 'b' / f'{name}.jsonl').read_bytes())
            self.assertEqual((Path(tmp) / 'a' / 'manifest.jsonl').read_bytes(),
                             (Path(tmp) / 'b' / 'manifest.jsonl').read_bytes())

    def test_heldout_option_accumulates_when_repeated(self):
        from unittest import mock

        class Stop(Exception):
            pass

        with mock.patch.object(build_trustmix, 'build', side_effect=Stop) as build:
            with self.assertRaises(Stop):
                build_trustmix.main(['build', '--output', 'x', '--heldout', 'a', 'b', '--heldout', 'c'])
        self.assertEqual(build.call_args.args[0].heldout, ['a', 'b', 'c'])

    def test_crop_cap_parses_all_and_rejects_negative(self):
        self.assertIsNone(crop_cap('all'))
        self.assertEqual(crop_cap('378'), 378)
        with self.assertRaises(argparse.ArgumentTypeError):
            crop_cap('-1')


if __name__ == '__main__':
    unittest.main()
