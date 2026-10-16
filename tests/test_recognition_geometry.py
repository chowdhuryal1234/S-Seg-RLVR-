"""Strict acceptance and unchanged-raw readback diagnostics, without models."""
import copy
import json
import unittest

from nucleus_rl.recognition_geometry import analyze_geometry, analyze_recognition


def answer(key, entries):
    return '<answer>' + json.dumps({key: entries}) + '</answer>'


class RecognitionGeometryTests(unittest.TestCase):
    def setUp(self):
        points = [('A', (45, 20)), ('B', (88, 20)), ('C', (18, 40)),
                  ('D', (110, 60)), ('E', (45, 93)), ('F', (82, 93))]
        self.manifest = {'original_size': [128, 128], 'processed_size': [224, 224], 'candidates': []}
        self.regions = []
        for key, (x, y) in points:
            box = [x-10, y-10, x+10, y+10]
            candidate = {'id': key, 'box_original': box, 'point_original': [x, y],
                         'box_processed': [v*1.75 for v in box], 'point_processed': [x*1.75, y*1.75],
                         'is_nucleus': key in 'BCF'}
            self.manifest['candidates'].append(candidate)
            self.regions.append({'id': key, 'box': candidate['box_processed'], 'point': candidate['point_processed']})

    def test_recognition_exact_set_and_errors_are_candidate_classification(self):
        result = analyze_recognition(answer('nucleus_regions', ['F', 'C', 'B']), self.manifest)
        self.assertTrue(result['format_valid'])
        self.assertTrue(result['exact_set_correct'])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), (3, 0, 0, 3))
        wrong = analyze_recognition(answer('nucleus_regions', ['A', 'B']), self.manifest)
        self.assertTrue(wrong['format_valid'])
        self.assertEqual((wrong['tp'], wrong['fp'], wrong['fn'], wrong['tn']), (1, 1, 2, 2))
        self.assertFalse(wrong['exact_set_correct'])

    def test_invalid_recognition_never_salvages_valid_ids(self):
        for entries in (['B', 'B', 'C', 'F'], ['B', 'Z'], ['B', True]):
            with self.subTest(entries=entries):
                result = analyze_recognition(answer('nucleus_regions', entries), self.manifest)
                self.assertFalse(result['format_valid'])
                self.assertEqual(result['raw_entries'], entries)
                self.assertEqual(result['accepted_ids'], [])
                self.assertEqual((result['tp'], result['fn']), (0, 3))
        empty = analyze_recognition(answer('nucleus_regions', []), self.manifest)
        self.assertTrue(empty['format_valid'])
        self.assertEqual((empty['precision'], empty['recall']), (0., 0.))

    def test_both_tasks_reject_duplicate_json_prose_and_truncation(self):
        for fn, key in ((analyze_recognition, 'nucleus_regions'), (analyze_geometry, 'regions')):
            texts = [f'<answer>{{"{key}":[],"{key}":[]}}</answer>',
                     'Here is my answer: ' + answer(key, []), '<answer>{',
                     '<answer>{"wrong":[]}</answer>', '<answer>{"' + key + '":NaN}</answer>']
            for text in texts:
                with self.subTest(fn=fn.__name__, text=text):
                    result = fn(text, self.manifest)
                    self.assertFalse(result['format_valid'])
                    self.assertIsNotNone(result['parse_error'])

    def test_geometry_exact_readback_order_independent_and_label_independent(self):
        result = analyze_geometry(answer('regions', list(reversed(self.regions))), self.manifest)
        self.assertTrue(result['format_valid'])
        self.assertTrue(result['engineering_pass'])
        self.assertEqual(result['accepted_region_count'], 6)
        self.assertEqual((result['mean_box_iou'], result['max_point_l2_processed']), (1., 0.))
        changed = copy.deepcopy(self.manifest)
        for item in changed['candidates']:
            item['is_nucleus'] = not item['is_nucleus']
        self.assertEqual(result, analyze_geometry(answer('regions', list(reversed(self.regions))), changed))

    def test_out_of_bounds_raw_geometry_is_retained_without_clipping(self):
        regions = copy.deepcopy(self.regions)
        regions[0]['box'][0] = -1
        result = analyze_geometry(answer('regions', regions), self.manifest)
        self.assertFalse(result['format_valid'])
        self.assertEqual(result['accepted_regions'], [])
        self.assertEqual(result['accepted_region_count'], 0)
        self.assertEqual(result['per_region'][0]['raw_box'][0], -1)
        self.assertTrue(result['per_region'][0]['box_metrics_available'])
        self.assertFalse(result['per_region'][0]['accepted_by_strict_parser'])
        self.assertAlmostEqual(result['per_region'][0]['box_iou'], 35 / 97.25)
        self.assertFalse(result['engineering_pass'])
        self.assertEqual(len(result['raw_entries']), 6)

    def test_missing_duplicate_and_unknown_region_ids_reject_whole_answer(self):
        for modification in ('missing', 'duplicate', 'unknown'):
            regions = copy.deepcopy(self.regions)
            if modification == 'missing': regions.pop()
            else: regions[-1]['id'] = 'A' if modification == 'duplicate' else 'Z'
            result = analyze_geometry(answer('regions', regions), self.manifest)
            self.assertFalse(result['format_valid'])
            self.assertEqual(result['accepted_region_count'], 0)
            self.assertIn('F', result['missing_ids'])
            if modification == 'duplicate': self.assertEqual(result['duplicate_ids'], ['A'])

    def test_engineering_thresholds_are_fixed_and_continuous_errors_remain(self):
        regions = copy.deepcopy(self.regions)
        regions[0]['point'][0] += 7
        at_limit = analyze_geometry(answer('regions', regions), self.manifest)
        self.assertTrue(at_limit['engineering_pass'])
        self.assertEqual(at_limit['per_region'][0]['point_l2_original'], 4.)
        regions[0]['point'][0] += .01
        above_limit = analyze_geometry(answer('regions', regions), self.manifest)
        self.assertTrue(above_limit['format_valid'])
        self.assertFalse(above_limit['engineering_pass'])
        self.assertAlmostEqual(above_limit['per_region'][0]['point_l2_processed'], 7.01)
        regions = copy.deepcopy(self.regions)
        regions[0]['box'][2] += 35
        half_iou = analyze_geometry(answer('regions', regions), self.manifest)
        self.assertEqual(half_iou['per_region'][0]['box_iou'], .5)
        self.assertFalse(half_iou['engineering_pass'])

    def test_invalid_numbers_and_points_are_nonthrowing_forensic_entries(self):
        for value in (True, '17', 10**400):
            regions = copy.deepcopy(self.regions)
            regions[0]['box'][0] = value
            result = analyze_geometry(answer('regions', regions), self.manifest)
            self.assertFalse(result['format_valid'])
            self.assertFalse(result['per_region'][0]['box_metrics_available'])
        regions = copy.deepcopy(self.regions)
        regions[0]['point'][0] = regions[0]['box'][2]
        result = analyze_geometry(answer('regions', regions), self.manifest)
        self.assertFalse(result['format_valid'])
        self.assertTrue(result['per_region'][0]['point_metrics_available'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
