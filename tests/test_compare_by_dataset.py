import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compare_runs


def reference_mask():
    """32 x 32 tile: 1 interior (100 px), 2 edge sliver (20 px), 3 large edge nucleus (144 px)."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[0:5, 0:4] = 2
    mask[20:32, 0:12] = 3
    return mask


def completion(tile, tp, fp, fn, pq=.5, patient='p1', **extra):
    return {'id': tile, 'patient_id': patient, 'pq': pq, 'sq': .8, 'dq': .6, 'tp': tp, 'fp': fp, 'fn': fn, **extra}


class Fixture:
    def __init__(self, root: Path):
        self.root = root

    def run(self, name, rows):
        folder = self.root / name / 'after'
        folder.mkdir(parents=True)
        (folder / 'completions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return folder

    def manifest(self, rows):
        (self.root / 'masks').mkdir(exist_ok=True)
        lines = []
        for row in rows:
            mask = row.pop('mask')
            Image.fromarray(mask).save(self.root / 'masks' / f"{row['id']}.png")
            lines.append({'mask_path': f"masks/{row['id']}.png", **row})
        path = self.root / 'validation.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in lines))
        return compare_runs.read_manifests([path])


def single_nucleus_mask():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 8:20] = 7
    return mask


class ByDatasetTest(unittest.TestCase):
    def test_tiles_split_by_dataset_with_monuseg_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            a = fx.run('a', [completion('m1', 2, 2, 1, .4), completion('c1', 1, 0, 0, .9)])
            b = fx.run('b', [completion('m1', 3, 1, 0, .6), completion('c1', 1, 1, 0, .7)])
            manifest = fx.manifest([
                {'id': 'm1', 'patient_id': 'TCGA-A', 'mask': reference_mask()},
                {'id': 'c1', 'patient_id': 'consep_train_1', 'dataset': 'consep', 'mask': single_nucleus_mask()}])
            table = compare_runs.compare([a, b], ['A', 'B'], samples=200, manifest=manifest, by_dataset=True)
            self.assertEqual(sorted(table[0]['by_dataset']), ['consep', 'monuseg'])
            mon = table[1]['by_dataset']['monuseg']
            self.assertEqual((mon['tiles'], mon['tp'], mon['fp'], mon['fn']), (1, 3, 1, 0))
            self.assertAlmostEqual(mon['vs_reference']['pq']['delta'], .2)
            self.assertEqual(table[1]['by_dataset']['consep']['vs_reference']['fp']['delta'], 1)
            self.assertNotIn('vs_reference', table[0]['by_dataset']['consep'])
            # Overall row is unchanged by the split.
            self.assertEqual((table[1]['tp'], table[1]['fp']), (4, 2))
            text = compare_runs.markdown_by_dataset(table)
            self.assertIn('| consep | B | 1 |', text)

    def test_dataset_falls_back_to_completions_without_manifest(self):
        meta = compare_runs.tile_meta(['x'], {}, {'x': {'id': 'x', 'patient_id': 'TCGA-1'}})
        self.assertEqual(meta['x'], {'dataset': 'monuseg', 'group': 'TCGA-1'})
        with self.assertRaises(ValueError):
            compare_runs.tile_meta(['x'], {'y': {'id': 'y'}}, {})

    def test_external_dataset_from_completion_group_prefix(self):
        # Completions carry patient_id but no 'dataset'; a mixed eval must not collapse into 'monuseg'.
        records = {'c': {'patient_id': 'consep_train_5'}, 'p': {'patient_id': 'puma_roi_100'},
                   'm': {'patient_id': 'TCGA-B0-5710'}, 'x': {'patient_id': 'puma_roi_1', 'dataset': 'other'}}
        meta = compare_runs.tile_meta(sorted(records), {}, records)
        self.assertEqual({i: m['dataset'] for i, m in meta.items()},
                         {'c': 'consep', 'p': 'puma', 'm': 'monuseg', 'x': 'other'})
        from nucleus_rl.cohort import EXTERNAL_DATASETS
        self.assertEqual(sorted(compare_runs.EXTERNAL_DATASETS), sorted(EXTERNAL_DATASETS))


class PerGroupTest(unittest.TestCase):
    def test_sign_counts_per_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            a = fx.run('a', [completion('t1', 2, 3, 1, patient='g1'), completion('t2', 2, 3, 1, patient='g1'),
                             completion('t3', 2, 3, 1, patient='g2'), completion('t4', 2, 3, 1, patient='g3')])
            b = fx.run('b', [completion('t1', 3, 1, 0, patient='g1'), completion('t2', 1, 3, 2, patient='g1'),
                             completion('t3', 1, 2, 2, patient='g2'), completion('t4', 2, 3, 1, patient='g3')])
            table = compare_runs.compare([a, b], ['A', 'B'], samples=50, per_group=True)
            groups = table[1]['per_group']['groups']
            self.assertEqual((groups['g1']['tiles'], groups['g1']['dtp'], groups['g1']['dfp']), (2, 0, -2))
            self.assertEqual((groups['g2']['dtp'], groups['g2']['dfp']), (-1, -1))
            signs = table[1]['per_group']['signs']
            self.assertEqual((signs['dtp']['up'], signs['dtp']['same'], signs['dtp']['down']), (0, 2, 1))
            self.assertEqual((signs['dfp']['up'], signs['dfp']['same'], signs['dfp']['down']), (0, 1, 2))
            self.assertNotIn('per_group', table[0])
            self.assertIn('| g1 | monuseg | 2 | +0 | -2 |', compare_runs.markdown_groups(table))

    def test_sign_test(self):
        self.assertEqual(compare_runs.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(compare_runs.sign_test(0, 4), .125)
        self.assertAlmostEqual(compare_runs.sign_test(3, 1), .625)


class EdgeSliverAndCauseTest(unittest.TestCase):
    def runs(self, fx):
        boxes = [{'box': [10, 10, 20, 20], 'point': [15, 15]}, {'box': [0, 0, 4, 5], 'point': [1, 2]},
                 {'box': [0, 20, 12, 32], 'point': [6, 26]}]
        a = fx.run('a', [completion('m1', 2, 1, 1, raw_best_gt_ids=[1, 2, None], raw_best_gt_ious=[.8, .7, .2],
                                    converted_objects=boxes)])
        b = fx.run('b', [completion('m1', 1, 0, 2, raw_best_gt_ids=[1], raw_best_gt_ious=[.9],
                                    converted_objects=boxes[:1])])
        manifest = fx.manifest([{'id': 'm1', 'patient_id': 'TCGA-A', 'truncated_instance_ids': [2, 3],
                                 'mask': reference_mask()}])
        return a, b, manifest

    def test_gt_objects_flags_slivers(self):
        gt = compare_runs.gt_objects(reference_mask(), {2, 3})
        self.assertEqual(gt[3]['box'], [0, 20, 12, 32])
        self.assertEqual([compare_runs.is_sliver(gt[g]) for g in (1, 2, 3)], [False, True, False])
        # Without truncated IDs, touching the border counts as cut by the edge.
        border = compare_runs.gt_objects(reference_mask())
        self.assertEqual([border[g]['edge'] for g in (1, 2, 3)], [False, True, True])

    def test_sliver_rescoring_with_raw_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, manifest = self.runs(Fixture(Path(tmp)))
            table = compare_runs.compare([a, b], ['A', 'B'], samples=50, manifest=manifest, edge_slivers=True)
            ref, other = table[0]['edge_slivers'], table[1]['edge_slivers']
            self.assertEqual(ref['method'], ['raw'])
            self.assertEqual(ref['excluded'], 1)
            self.assertEqual([ref['all_nuclei'][k] for k in ('tp', 'fp', 'fn')], [2, 1, 1])
            # The mask on the sliver is neither TP nor FP once the sliver is ignored.
            self.assertEqual([ref['without_slivers'][k] for k in ('tp', 'fp', 'fn')], [1, 1, 1])
            self.assertEqual([other['without_slivers'][k] for k in ('tp', 'fp', 'fn')], [1, 0, 1])
            self.assertEqual(other['vs_reference']['fp']['delta'], -1)
            self.assertEqual(other['vs_reference']['tp']['delta'], 0)
            self.assertIn('Slivers excluded', compare_runs.markdown_slivers(table))

    def test_fn_causes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, manifest = self.runs(Fixture(Path(tmp)))
            table = compare_runs.compare([a, b], ['A', 'B'], samples=50, manifest=manifest, causes=True)
            ref, other = table[0]['fn_causes'], table[1]['fn_causes']
            self.assertEqual(ref['counts'], {'no_box': 0, 'box_near': 0, 'good_box_mask_failed': 1})
            self.assertEqual(other['counts'], {'no_box': 2, 'box_near': 0, 'good_box_mask_failed': 0})
            self.assertEqual(other['edge_slivers']['no_box'], 1)
            self.assertIn('| all | B | 2 | 2 (1) | 0 (0) | 0 (0) |', compare_runs.markdown_causes(table))

    def test_box_near_band(self):
        gt = compare_runs.gt_objects(reference_mask(), {2, 3})
        record = {'id': 'm1', 'converted_objects': [{'box': [10, 10, 15, 20]}]}  # IoU 0.5 with nucleus 1
        causes = compare_runs.fn_causes(record, gt, matched=set())
        self.assertEqual(causes['counts'], {'no_box': 2, 'box_near': 1, 'good_box_mask_failed': 0})
        self.assertIsNone(compare_runs.fn_causes({'id': 'm1'}, gt, set()))

    def test_exact_matching_from_instance_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            pred = np.zeros((32, 32), dtype=np.int32)
            pred[10:20, 10:20] = 1          # nucleus 1
            pred[0:5, 0:4] = 2              # the sliver
            pred[24:30, 24:30] = 3          # nothing there
            folder = fx.run('a', [completion('m1', 2, 1, 1, prediction_path='/remote/run/after/000001.npy')])
            np.save(folder / '000001.npy', pred)
            manifest = fx.manifest([{'id': 'm1', 'patient_id': 'TCGA-A', 'truncated_instance_ids': [2, 3],
                                     'mask': reference_mask()}])
            table = compare_runs.compare([folder], ['A'], samples=10, manifest=manifest, edge_slivers=True)
            slivers = table[0]['edge_slivers']
            self.assertEqual(slivers['method'], ['exact'])
            self.assertEqual([slivers['all_nuclei'][k] for k in ('tp', 'fp', 'fn')], [2, 1, 1])
            self.assertEqual([slivers['without_slivers'][k] for k in ('tp', 'fp', 'fn')], [1, 1, 1])

    def test_cli_requires_manifest_and_prints_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            a, b, _ = self.runs(fx)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                compare_runs.main(['--reference', str(a), str(b), '--fn-causes'])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                compare_runs.main(['--reference', str(a), str(b), '--samples', '20', '--manifest',
                                   str(Path(tmp) / 'validation.jsonl'), '--by-dataset', '--per-group',
                                   '--exclude-edge-slivers', '--fn-causes', '--json', str(Path(tmp) / 'out.json')])
            text = out.getvalue()
            for needle in ('| monuseg | b | 1 |', 'sign-test p', 'Slivers excluded', 'Good box, mask failed'):
                self.assertIn(needle, text)
            saved = json.loads((Path(tmp) / 'out.json').read_text())
            self.assertEqual(saved[1]['by_dataset']['monuseg']['fn_causes']['fn'], 2)


if __name__ == '__main__':
    unittest.main()
