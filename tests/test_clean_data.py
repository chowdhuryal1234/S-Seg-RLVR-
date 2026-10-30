import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import validate_external_rows, validate_source_rows
from scripts import compare_runs, stitch_eval
from scripts.air.run_arm import expand, sync_out
from scripts.prepare_external import puma_group, rasterize_geojson, split_groups


def external_row(**overrides):
    row = {'id': 'consep_train_3_x0_y0', 'dataset': 'consep', 'split': 'train', 'source_image_id': 'consep_train_3',
           'patient_id': 'consep_train_3', 'source_image_member': 'CoNSeP/Train/Images/train_3.png'}
    row.update(overrides)
    return row


class ExternalRowsTest(unittest.TestCase):
    def test_accepts_training_release_rows(self):
        audit = validate_external_rows([external_row()])
        self.assertEqual(audit['datasets'], ['consep'])

    def test_rejects_test_folder_unknown_dataset_and_foreign_group(self):
        with self.assertRaises(ValueError):
            validate_external_rows([external_row(source_image_member='CoNSeP/Test/Images/test_1.png')])
        with self.assertRaises(ValueError):
            validate_external_rows([external_row(dataset='pannuke')])
        with self.assertRaises(ValueError):
            validate_external_rows([external_row(patient_id='TCGA-21-5786')])

    def test_source_rows_require_train_split(self):
        with self.assertRaises(ValueError):
            validate_source_rows([external_row(split='validation')])


class SplitTest(unittest.TestCase):
    def test_puma_primary_and_metastatic_share_a_group(self):
        self.assertEqual(puma_group('training_set_primary_roi_007'), puma_group('training_set_metastatic_roi_007'))

    def test_group_split_is_seeded_disjoint_and_complete(self):
        groups = {f'g{i}': [f's{i}'] for i in range(20)}
        a = split_groups(groups, [('validation', .2), ('heldout', .2), ('train', None)], 1)
        self.assertEqual(a, split_groups(groups, [('validation', .2), ('heldout', .2), ('train', None)], 1))
        keys = [k for part in a.values() for k in part]
        self.assertEqual(sorted(keys), sorted(groups))
        self.assertEqual((len(a['validation']), len(a['heldout']), len(a['train'])), (4, 4, 12))

    def test_geojson_rasterizer_earliest_feature_wins(self):
        square = lambda x0, y0, s: {'geometry': {'type': 'Polygon', 'coordinates': [[[x0, y0], [x0 + s, y0],
                                                                                   [x0 + s, y0 + s], [x0, y0 + s]]]}}
        labels = rasterize_geojson([square(2, 2, 6), square(5, 5, 6)], (16, 16))
        self.assertEqual(labels[6, 6], 1)
        self.assertEqual(labels[10, 10], 2)


class StitchTest(unittest.TestCase):
    def test_owned_ranges_partition_the_image(self):
        ranges = stitch_eval.owned_ranges([0, 96, 192, 272], 128, 400)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 400)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))

    def test_nucleus_cut_by_a_tile_edge_is_one_instance(self):
        gt = np.zeros((200, 200), dtype=np.int32)
        gt[60:80, 120:140] = 1  # spans the right edge of the first tile (0..128)
        gt[10:20, 10:20] = 2
        whole = {'width': 200, 'height': 200, 'crop': 128, 'tile_x': [0, 72], 'tile_y': [0, 72]}
        tiles = {(x, y): gt[y:y + 128, x:x + 128] for x in whole['tile_x'] for y in whole['tile_y']}
        stitched = stitch_eval.stitch(tiles, whole)
        result = stitch_eval.quality(stitch_eval.match(stitched, gt))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 0, 0))

    def test_match_counts_false_positives_and_misses(self):
        gt = np.zeros((20, 20), dtype=np.int32)
        gt[0:5, 0:5] = 1
        gt[10:15, 10:15] = 2
        pred = np.zeros_like(gt)
        pred[0:5, 0:5] = 1
        pred[15:20, 0:5] = 2
        self.assertEqual(stitch_eval.match(pred, gt)['tp'], 1)
        q = stitch_eval.quality(stitch_eval.match(pred, gt))
        self.assertEqual((q['fp'], q['fn']), (1, 1))
        self.assertAlmostEqual(q['pq'], 0.5)


class CompareRunsTest(unittest.TestCase):
    def write(self, folder, values):
        folder.mkdir(parents=True)
        (folder / 'completions.jsonl').write_text(''.join(
            json.dumps({'id': f't{i}', 'pq': pq, 'sq': .8, 'dq': .5, 'tp': tp, 'fp': fp, 'fn': fn}) + '\n'
            for i, (pq, tp, fp, fn) in enumerate(values)))

    def test_totals_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a' / 'after', Path(tmp) / 'b'
            self.write(a, [(.4, 2, 2, 1), (.6, 3, 0, 1)])
            self.write(b, [(.5, 2, 1, 1), (.7, 3, 0, 1)])
            table = compare_runs.compare([a.parent, b], ['A', 'B'], samples=200)
            self.assertEqual((table[0]['tp'], table[0]['fp'], table[0]['fn']), (5, 2, 2))
            self.assertAlmostEqual(table[1]['vs_reference']['pq']['delta'], .1)
            self.assertEqual(table[1]['vs_reference']['fp']['delta'], -1)
            self.assertIn('| B |', compare_runs.markdown(table))

    def test_rejects_different_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp) / 'a', [(.4, 1, 0, 0)])
            self.write(Path(tmp) / 'b', [(.4, 1, 0, 0), (.5, 1, 0, 0)])
            with self.assertRaises(ValueError):
                compare_runs.compare([Path(tmp) / 'a', Path(tmp) / 'b'])


class AirRunnerTest(unittest.TestCase):
    def test_expand_placeholders(self):
        self.assertEqual(expand('{stage:data}/train.jsonl', {'data': Path('/l/in/data')}, Path('/l/out')),
                         '/l/in/data/train.jsonl')
        self.assertEqual(expand('{output}', {}, Path('/l/out')), '/l/out')

    def test_sync_copies_only_new_or_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 's', Path(tmp) / 't'
            (source / 'a').mkdir(parents=True)
            (source / 'a' / 'x.txt').write_text('1')
            self.assertEqual(sync_out(source, target), 1)
            self.assertEqual(sync_out(source, target), 0)
            (source / 'a' / 'x.txt').write_text('22')
            self.assertEqual(sync_out(source, target), 1)
            self.assertEqual((target / 'a' / 'x.txt').read_text(), '22')


if __name__ == '__main__':
    unittest.main()
