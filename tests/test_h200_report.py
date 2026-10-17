"""CPU artifact tests; synthetic masks are never presented as experiment results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from nucleus_rl.evaluate import ArtifactScorer
from nucleus_rl.rewards import masks_to_instances
from scripts.render_h200_training import (
    dump, jsonl, load_dataset, render, run_report, select_examples, sha, verify_phase,
)


def answer(objects):
    return '<answer>' + json.dumps({"objects": objects}) + '</answer>'


OBJECT = {"box": [4, 4, 12, 12], "point": [8, 8]}
SECOND = {"box": [20, 20, 28, 28], "point": [24, 24]}


class SavedArtifactFixture:
    def __init__(self, root):
        self.root = root
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        protocol_file = Path(__file__).resolve().parents[1] / 'protocols/monuseg_official30.json'
        source_ids = json.loads(protocol_file.read_text())["image_ids"]
        self.rows = []
        for index, (crop_id, count, split, patient) in enumerate((
                ('train', 1, 'train', 'patient_train'),
                ('a_single', 1, 'validation', 'patient_val'),
                ('b_multiple', 2, 'validation', 'patient_val'))):
            image = np.full((32, 32, 3), 230, dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:12, 4:12] = 1
            if count == 2:
                mask[20:28, 20:28] = 2
            image[mask > 0] = [80, 70, 130]
            image_path = self.dataset / f'{crop_id}.png'
            mask_path = self.dataset / f'{crop_id}_mask.png'
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            self.rows.append({"id": crop_id, "patient_id": patient, "source_image_id": source_ids[index],
                              "split": split, "image_path": image_path.name, "mask_path": mask_path.name,
                              "image_sha256": sha(image_path), "mask_sha256": sha(mask_path),
                              "instance_count": count})
        manifests = {"train": self.rows[:1], "validation": self.rows[1:],
                     "validation_benchmark8": self.rows[2:]}
        for name, rows in manifests.items():
            (self.dataset / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        dump(self.dataset / 'frozen_protocol.json', {
            "official30_protocol_sha256": sha(protocol_file), "train_crops": 1,
            "validation_crops": 2, "benchmark_crops": 1,
            "manifest_sha256": {name: sha(self.dataset / f'{name}.jsonl') for name in manifests}})
        self.data = load_dataset(self.dataset)
        self.suite = root / 'suite'
        self.run = self.suite / 'main_fixture'
        self.run.mkdir(parents=True)
        self.arguments = {"max_objects": 8, "coordinate_frame": "original", "prompt_style": "schema",
                          "learning_rate": 1e-5, "seed": 42, "max_steps": 2,
                          "model": "fixture_only", "revision": "fixture_revision", "adapter": None}
        dump(self.run / 'arguments.json', self.arguments)
        dump(self.run / 'metadata.json', {
            "model": "fixture_only", "resolved_revision": "fixture_revision", "adapter": None,
            "manifest_sha256": self.data["manifest_sha256"]["train"],
            "eval_manifest_sha256": self.data["manifest_sha256"]["validation"]})
        self.entry = {"name": self.run.name, "benchmark": False, "model_size": "7b", "planned_steps": 2,
                      "learning_rate": 1e-5, "seed": 42, "fresh_base": True, "status": "fixture_only"}
        for phase, texts in (("before", [answer([OBJECT, OBJECT]), 'invalid answer']),
                             ("after", [answer([OBJECT]), answer([OBJECT, SECOND])])):
            self.write_phase(phase, texts)
        self.result = {"status": "fixture_only", "sam": {"unchanged": True, "parameters_before": "frozen",
                       "parameters_after": "frozen", "trainable_parameters": 0},
                       "visual_sha256_before": "visual", "visual_sha256_after": "visual",
                       "adapter_changed": True, "adapter_sha256_before": "before", "adapter_sha256_after": "after",
                       "before": json.loads((self.run / 'before/summary.json').read_text()),
                       "after": json.loads((self.run / 'after/summary.json').read_text())}
        dump(self.run / 'result.json', self.result)
        dump(self.suite / 'experiment_summary.json', {"status": "synthetic_fixture_only", "runs": [self.entry]})

    def write_phase(self, phase, texts):
        masks = {str(self.dataset / r['image_path']): self.data['assets'][r['id']]['mask'] for r in self.rows}

        class SavedMaskSegmenter:
            def predict(self, image, objects, image_key=None):
                gt = masks[image_key]
                ids = np.unique(gt[gt > 0])
                self.last_raw_masks = np.stack([gt == ids[min(i, len(ids)-1)] for i in range(len(objects))])
                return masks_to_instances(self.last_raw_masks)

        scorer = ArtifactScorer(SavedMaskSegmenter(), self.run / phase)
        for row, text in zip(self.rows[1:], texts):
            scorer.score(text, {**row, "image_path": str(self.dataset / row['image_path']),
                               "mask_path": str(self.dataset / row['mask_path'])})
        dump(self.run / phase / 'summary.json', scorer.summary())
        # Mimic remote absolute paths after download. Renderer must use only
        # matching local ordinal basenames, never dereference the source host.
        records = jsonl(self.run / phase / 'completions.jsonl')
        for i, record in enumerate(records, 1):
            record['prediction_path'] = f'/remote/unavailable/{phase}/{i:06d}.npy'
        (self.run / phase / 'completions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))

    def phase(self, phase):
        return verify_phase(self.run / phase, self.rows[1:], self.data['assets'], self.arguments)


class H200ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SavedArtifactFixture(Path(self.temp.name))

    def test_fixed_selection_is_gt_only_and_benchmark_has_no_single(self):
        f = self.fixture
        selected = select_examples(list(reversed(f.rows[1:])), f.data['assets'])
        self.assertEqual(selected, {'single': 'a_single', 'multiple': 'b_multiple'})
        self.assertIsNone(select_examples(f.rows[2:], f.data['assets'])['single'])

    def test_renderer_import_does_not_require_other_report_pr(self):
        script = '''
import importlib.abc
import sys
class NoPriorRenderer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts.render_prompt_ablation":
            raise ImportError("Earlier report PR is deliberately unavailable")
sys.meta_path.insert(0, NoPriorRenderer())
import scripts.render_h200_training
'''
        completed = subprocess.run([sys.executable, '-c', script],
                                   cwd=Path(__file__).resolve().parents[1],
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_recompute_final_and_raw_metrics_including_invalid_answer(self):
        before, after = self.fixture.phase('before'), self.fixture.phase('after')
        self.assertTrue(before['complete'])
        self.assertEqual(before['metrics']['macro_pq'], .5)
        self.assertEqual(before['metrics']['micro_recall'], 1/3)
        self.assertEqual(before['metrics']['count_mae'], 1)
        self.assertEqual(before['metrics']['format_valid_count'], 1)
        self.assertEqual(before['metrics']['total_same_gt_duplicate_count'], 1)
        self.assertEqual(after['metrics']['micro_recall'], 1.)
        self.assertEqual(after['metrics']['total_gt_count'], 3)

    def test_missing_raw_masks_is_unknown_duplicates_not_zero(self):
        (self.fixture.run / 'before/000001_raw_masks.npz').unlink()
        result = self.fixture.phase('before')
        self.assertTrue(result['complete'])
        self.assertIsNone(result['metrics']['total_same_gt_duplicate_count'])
        self.assertEqual(result['verified_raw_masks'], 1)

    def test_missing_prediction_is_partial_not_a_full_cohort_score(self):
        (self.fixture.run / 'after/000001.npy').unlink()
        result = self.fixture.phase('after')
        self.assertFalse(result['complete'])
        self.assertEqual(result['verified_final_maps'], 1)
        self.assertEqual(result['metrics']['total_gt_count'], 2)
        self.assertNotIn('a_single', result['examples'])

    def test_corrupt_nonempty_prediction_for_invalid_text_is_rejected(self):
        np.save(self.fixture.run / 'before/000002.npy', np.ones((32, 32), dtype=np.int32))
        with self.assertRaisesRegex(ValueError, 'Rejected/empty'):
            self.fixture.phase('before')

    def test_saved_score_mismatch_is_rejected(self):
        path = self.fixture.run / 'after/completions.jsonl'
        records = jsonl(path)
        records[0]['pq'] = .8
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaisesRegex(ValueError, 'pq'):
            self.fixture.phase('after')

    def test_image_hash_mismatch_is_rejected(self):
        Image.new('RGB', (32, 32), 'red').save(self.fixture.dataset / 'a_single.png')
        with self.assertRaisesRegex(ValueError, 'image hash'):
            load_dataset(self.fixture.dataset)

    def test_changed_sam_cannot_be_reported_as_frozen(self):
        f = self.fixture
        f.result['sam'].update(unchanged=False, parameters_after='changed')
        dump(f.run / 'result.json', f.result)
        output = f.root / 'report'
        output.mkdir()
        with self.assertRaisesRegex(ValueError, 'SAM must stay frozen'):
            run_report(f.suite, f.entry, f.data, output)

    def test_completed_and_missing_runs_render_without_inventing_updates(self):
        f = self.fixture
        source = json.loads((f.suite / 'experiment_summary.json').read_text())
        source['runs'].append({**f.entry, 'name': 'planned_fixture', 'status': 'planned'})
        dump(f.suite / 'experiment_summary.json', source)
        output = f.root / 'report'
        render(f.suite, f.dataset, output)
        summary = json.loads((output / 'main_fixture/summary.json').read_text())
        self.assertEqual(summary['phases']['after']['metrics']['macro_pq'], 1.)
        pending = json.loads((output / 'planned_fixture/summary.json').read_text())
        self.assertIsNone(pending['phases']['after']['metrics'])
        self.assertFalse(pending['phases']['after']['complete'])
        self.assertIsNone(pending['result'])
        self.assertIn('missing or partial', (output / 'planned_fixture/index.html').read_text())
        with Image.open(output / 'main_fixture/single_example.png') as image:
            self.assertEqual(image.size, (1380, 880))


if __name__ == '__main__':
    unittest.main()
