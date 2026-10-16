"""Keep Databricks preparation independent of generated reports."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import prepare_monuseg

ROOT = Path(__file__).resolve().parents[1]


def config_referenced_by_notebook():
    # Inspect the source without executing Spark/Databricks notebook cells.
    tree = ast.parse((ROOT / 'scripts/prep_monuseg_databricks.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'protocol' for t in node.targets))
    paths = [node.value for node in ast.walk(assignment.value)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.endswith('.json')]
    if len(paths) != 1:
        raise AssertionError('Expected one readable preparation configuration')
    return ROOT / paths[0]


class DatabricksConfigTests(unittest.TestCase):
    def test_notebook_input_exists_without_any_report_files(self):
        path = config_referenced_by_notebook()
        self.assertEqual(path.parent, ROOT / 'configs')
        config = json.loads(path.read_text())
        self.assertEqual(set(config), {'preparation_argv', 'expected_crop_ids'})
        self.assertEqual(set(config['expected_crop_ids']), {'train', 'validation'})

    def test_configuration_matches_cli_and_disjoint_patient_assignments(self):
        config = json.loads(config_referenced_by_notebook().read_text())
        stub_report = dict(status='checked', selected_crop_counts={}, candidate_crop_counts={}, patient_groups={})
        with patch.object(prepare_monuseg, 'prepare', return_value=stub_report) as prepare, redirect_stdout(StringIO()):
            self.assertEqual(prepare_monuseg.main(config['preparation_argv'][1:]), 0)
        args = prepare.call_args.args[0]
        patients = {'train': set(args.train_patient_ids.split(',')),
                    'validation': set(args.val_patient_ids.split(','))}
        self.assertFalse(patients['train'] & patients['validation'])
        for split, expected_count in [('train', args.train_crops), ('validation', args.val_crops)]:
            crop_ids = config['expected_crop_ids'][split]
            self.assertEqual(len(crop_ids), expected_count)
            self.assertEqual(len(set(crop_ids)), expected_count)
            self.assertTrue(all('-'.join(crop.split('-')[:3]) in patients[split] for crop in crop_ids))


if __name__ == '__main__':
    unittest.main()
