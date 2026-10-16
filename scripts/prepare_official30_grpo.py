#!/usr/bin/env python3
"""Freeze the official30-only GRPO dataset, without model loading or inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.cohort import load_official30_cohort, select_cohort_pairs
from scripts.audit_monuseg import audit_archive, parse_regions
from scripts.prepare_monuseg import prepare, rasterize_regions, select_balanced


SEED = 20261016
BENCHMARK_SEED = 20261017


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def grouped_duplicates(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row['id'])
    return [ids for ids in grouped.values() if len(ids) > 1]


def distribution(rows):
    return {'crops': len(rows), 'patients': len({row['patient_id'] for row in rows}),
            'by_patient': dict(sorted(Counter(row['patient_id'] for row in rows).items())),
            'visible_instance_count_histogram': dict(sorted(Counter(row['instance_count'] for row in rows).items())),
            'total_visible_instances': sum(row['instance_count'] for row in rows),
            'crops_with_truncated_instances': sum(bool(row['truncated_instance_ids']) for row in rows),
            'truncated_visible_instances': sum(len(row['truncated_instance_ids']) for row in rows),
            'crops_with_border_touching_instances': sum(bool(row['border_touching_instance_ids']) for row in rows),
            'crops_with_source_clipped_instances': sum(bool(row['source_clipped_instance_ids']) for row in rows)}


def main(args):
    root = Path(__file__).resolve().parents[1]
    out, report_dir, protocol_path = (root / args.output).resolve(), (root / args.report).resolve(), (root / args.protocol).resolve()
    if out.exists() or report_dir.exists() or protocol_path.exists():
        raise FileExistsError('Preserve existing datasets/reports/protocols; choose new destinations')
    archive_path = Path(args.train_archive).resolve()
    if 'test' in archive_path.name.lower():
        raise ValueError('Only the original training archive is permitted')
    cohort = load_official30_cohort()
    iris_path = root / 'baselines/iris/split.json'
    iris = json.loads(iris_path.read_text())
    original_audit = audit_archive(archive_path, expected_images=30)
    if not original_audit['pair_validation_passed']:
        raise ValueError('Training archive pair audit failed')
    canonical, membership = select_cohort_pairs(original_audit['pairs'], cohort)
    canonical_ids = {p['id'] for p in canonical}
    iris_groups = {'train': set(iris['train']), 'validation': set(iris['validation'])}
    if iris_groups['train'] & iris_groups['validation'] or canonical_ids - set.union(*iris_groups.values()):
        raise ValueError('Iris split is overlapping or missing a canonical source')
    excluded = [p for p in canonical if p['negative_region_entries'] or p['degenerate_region_entries']]
    eligible = [p for p in canonical if p not in excluded]
    patient_groups = {split: sorted(p['patient_id'] for p in eligible if p['id'] in ids)
                      for split, ids in iris_groups.items()}
    if set(patient_groups['train']) & set(patient_groups['validation']):
        raise ValueError('Patient groups overlap')
    preparation = argparse.Namespace(train_archive=archive_path, out=out, cohort='official30',
        crop_size=128, max_nuclei=8, border_policy='visible', train_patient_ids=','.join(patient_groups['train']),
        val_patient_ids=','.join(patient_groups['validation']), train_patients=len(patient_groups['train']),
        val_patients=len(patient_groups['validation']), train_crops=256, val_crops=64, seed=SEED)
    report = prepare(preparation)
    rows = [json.loads(line) for line in (out / 'manifest.jsonl').read_text().splitlines()]
    rows_by_source = defaultdict(list)
    for row in rows:
        rows_by_source[row['source_image_id']].append(row)
    source_audits, candidates_by_split = [], {'train': [], 'validation': []}
    source_hashes = {}
    # Independent reopening verifies saved crop pixels and source containment;
    # the established rasterizer is reused without changing annotation semantics.
    with ZipFile(archive_path) as archive:
        for pair in canonical:
            image_bytes, xml_bytes = archive.read(pair['image_member']), archive.read(pair['xml_member'])
            source_hashes[pair['id']] = {'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                                        'xml_sha256': hashlib.sha256(xml_bytes).hexdigest(),
                                        'image_member': pair['image_member'], 'xml_member': pair['xml_member']}
            if pair in excluded:
                continue
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            labels, raster_log = rasterize_regions(parse_regions(xml_bytes), image.width, image.height)
            full_counts = np.bincount(labels.ravel())
            source_clipped = {r['instance_id'] for r in raster_log if r['clipped_to_image']}
            split = next(split for split, ids in iris_groups.items() if pair['id'] in ids)
            all_grid = []
            for y in range(0, image.height - 127, 128):
                for x in range(0, image.width - 127, 128):
                    ids = np.unique(labels[y:y+128, x:x+128]); count = int(np.count_nonzero(ids))
                    all_grid.append(count)
            source_audits.append({'source_image_id': pair['id'], 'patient_id': pair['patient_id'], 'split': split,
                'source_size': list(image.size), 'full_grid_crops': len(all_grid),
                'grid_visible_count_histogram': dict(sorted(Counter(all_grid).items())),
                'eligible_1_to_8': sum(1 <= n <= 8 for n in all_grid),
                'excluded_empty': sum(n == 0 for n in all_grid), 'excluded_dense_over8': sum(n > 8 for n in all_grid),
                'grid_covered_pixels': len(all_grid) * 128**2,
                'uncovered_right_bottom_pixels': image.width * image.height - len(all_grid) * 128**2,
                'selected_crops': len(rows_by_source[pair['id']])})
            candidates_by_split[split].extend(all_grid)
            for row in rows_by_source[pair['id']]:
                x0, y0, x1, y1 = row['bbox_xyxy']
                assert 0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height
                assert x1-x0 == y1-y0 == 128 and x0 % 128 == y0 % 128 == 0
                saved_rgb = np.asarray(Image.open(out / row['image_path']).convert('RGB'))
                saved_mask = np.asarray(Image.open(out / row['mask_path']))
                assert np.array_equal(saved_rgb, np.asarray(image)[y0:y1, x0:x1])
                assert np.array_equal(saved_mask, labels[y0:y1, x0:x1])
                ids, counts = np.unique(saved_mask[saved_mask > 0], return_counts=True)
                assert 1 <= len(ids) <= 8 and row['instance_ids'] == ids.tolist() and row['instance_count'] == len(ids)
                assert row['truncated_instance_ids'] == [int(i) for i, count in zip(ids, counts) if count != full_counts[i]]
                assert row['source_clipped_instance_ids'] == sorted(set(ids.tolist()) & source_clipped)
                border = np.concatenate((saved_mask[0], saved_mask[-1], saved_mask[:,0], saved_mask[:,-1]))
                assert row['border_touching_instance_ids'] == sorted(int(i) for i in np.unique(border) if i)
                row.update(image_sha256=digest(out / row['image_path']), mask_sha256=digest(out / row['mask_path']),
                           image_pixels_sha256=hashlib.sha256(saved_rgb.tobytes()).hexdigest(),
                           source_image_sha256=source_hashes[pair['id']]['image_sha256'],
                           source_xml_sha256=source_hashes[pair['id']]['xml_sha256'])
                # Centroids are not required by the learner or scorer and are
                # omitted from portable manifests to reduce accidental prompt leakage.
                row.pop('visible_centroids_xy', None)
                row.pop('visible_centroid_definition', None)
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    assert len(train) == 256 and len(validation) == 64
    assert {row['patient_id'] for row in train}.isdisjoint({row['patient_id'] for row in validation})
    assert {row['source_image_id'] for row in rows} <= canonical_ids
    assert len({row['id'] for row in rows}) == len(rows)
    overlap_pairs = []
    for source, source_rows in rows_by_source.items():
        for index, left in enumerate(source_rows):
            for right in source_rows[index+1:]:
                a, b = left['bbox_xyxy'], right['bbox_xyxy']
                if min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1]):
                    overlap_pairs.append([left['id'], right['id']])
    duplicates = grouped_duplicates(rows, 'image_pixels_sha256')
    assert not duplicates and not overlap_pairs
    benchmark = select_balanced(validation, 8, BENCHMARK_SEED)
    assert len({r['id'] for r in benchmark}) == 8 and Counter(r['patient_id'] for r in benchmark) == dict.fromkeys(patient_groups['validation'], 2)
    for name, subset in (('manifest', rows), ('train', train), ('validation', validation), ('validation_benchmark8', benchmark)):
        write_rows(out / f'{name}.jsonl', subset)
    prior_overlap = {}
    for prior in ('monuseg_iris_pilot_v1', 'monuseg_engineering_v1'):
        prior_rows = [json.loads(line) for line in (root / 'data' / prior / 'manifest.jsonl').read_text().splitlines()]
        prior_lookup = {r['id']: r for r in prior_rows}
        prior_overlap[prior] = {'identical_crop_ids': sorted({r['id'] for r in rows} & set(prior_lookup)),
            'current_validation_prior_train_patient_overlap': sorted({r['patient_id'] for r in validation} & {r['patient_id'] for r in prior_rows if r['split']=='train'}),
            'current_train_prior_validation_patient_overlap': sorted({r['patient_id'] for r in train} & {r['patient_id'] for r in prior_rows if r['split']=='validation'}),
            'note': 'Historical reuse is disclosed; this is frozen engineering validation, not a newly untouched test set. New train/validation are patient-disjoint; fresh base-policy starts must not reuse adapters trained under conflicting older splits.'}
    now = datetime.now(timezone.utc).isoformat()
    protocol = {'schema_version': 1, 'dataset_id': 'monuseg_official30_grpo_v1', 'frozen_utc': now,
        'scope': 'Full-mask training/reward engineering dataset from official30 training sources only; no official test and no benchmark-equivalence claim.',
        'source_archive_name': archive_path.name, 'source_archive_sha256': digest(archive_path),
        'official30_protocol_sha256': cohort['protocol_sha256'], 'official30_source_pdf_sha256': cohort['source']['sha256'],
        'iris_split_path': 'baselines/iris/split.json', 'iris_split_sha256': digest(iris_path),
        'iris_assignment_preserved_for_all_retained_patients': True, 'patient_groups': patient_groups,
        'seed': SEED, 'benchmark_seed': BENCHMARK_SEED, 'train_crops': 256, 'validation_crops': 64, 'benchmark_crops': 8,
        'sampling': {'crop_size': 128, 'stride': 128, 'min_visible_nuclei': 1, 'max_visible_nuclei': 8,
                     'patient_balanced_round_robin': True, 'count_stratification': False, 'border_policy': 'visible',
                     'partial_objects': 'Retain and supervise every visible annotated pixel and ID, including tiny fragments; record truncation/source-clipping metadata; no ignored pixels or GT-based prompt locations.',
                     'selection_bias': 'The 1-8 capacity excludes dense crops; no additional clarity/size/color/complete-nucleus filtering. Patient balancing reweights source prevalence. Grid excludes uncovered right/bottom tails.'},
        'benchmark_selection': 'Deterministic patient-balanced sample of frozen validation64, two crops from each of four patients, chosen before overnight model outputs. Same exact rows/pixels for 7B and 32B.',
        'model_input_contract': 'Only RGB image and fixed task instructions/format example. Full masks, counts, instance IDs, coordinates, patient metadata and validation annotations never enter Qwen prompts. Masks are used only by the reward/scorer. GT centroids are omitted from manifests.',
        'excluded_noncanonical_image_ids': membership['excluded_noncanonical_image_ids'],
        'excluded_canonical_annotation_images': [{'id':p['id'],'negative_region_entries':p['negative_region_entries'],'degenerate_region_entries':p['degenerate_region_entries']} for p in excluded],
        'canonical_sources_with_zero_eligible_crops': [p['source_image_id'] for p in source_audits if p['eligible_1_to_8']==0],
        'manifest_sha256': {name:digest(out/f'{name}.jsonl') for name in ('train','validation','validation_benchmark8','manifest')},
        'source_code_sha256': {name:digest(root/name) for name in ('scripts/prepare_official30_grpo.py','scripts/prepare_monuseg.py','scripts/audit_monuseg.py','nucleus_rl/cohort.py')},
        'historical_reuse_limit': 'The validation patient cohort and some crops have appeared in earlier engineering evaluation; not a new untouched test set. Do not initialize from an old adapter that trained on a current validation patient.'}
    audit = {'status': 'passed', 'created_utc': now, 'original_test_accessed': False,
        'dataset': str(out), 'protocol': str(protocol_path), 'cohort_membership': membership,
        'patient_disjoint': True, 'unique_crop_ids': True, 'same_source_spatial_overlap_pairs': overlap_pairs,
        'duplicate_rgb_pixel_hash_groups': duplicates, 'all_rgb_crops_pixel_exact': True,
        'all_masks_exact_source_raster_crops': True, 'all_crop_windows_contained_in_sources': True,
        'all_partial_border_source_clipped_metadata_recomputed': True,
        'portable_paths': 'image_path, mask_path and gt_mask are relative to the manifest directory; absolute source_archive is provenance only.',
        'splits': {name:distribution(subset) for name,subset in (('train',train),('validation',validation),('validation_benchmark8',benchmark))},
        'candidate_population': {split:{'grid_crops':len(counts),'eligible_1_to_8':sum(1<=n<=8 for n in counts),
            'excluded_empty':sum(n==0 for n in counts),'excluded_dense_over8':sum(n>8 for n in counts),
            'visible_count_histogram':dict(sorted(Counter(counts).items()))} for split,counts in candidates_by_split.items()},
        'source_audits':source_audits,'source_member_hashes':source_hashes,'prior_manifest_overlap':prior_overlap,
        'manifest_sha256':protocol['manifest_sha256']}
    save(protocol_path, protocol)
    save(out / 'frozen_protocol.json', protocol)
    save(report_dir / 'data_audit.json', audit)
    save(out / 'source_member_hashes.json', source_hashes)
    save(out / 'frozen_validation.json', {'frozen_utc':now, 'validation_sha256':protocol['manifest_sha256']['validation'],
         'benchmark8_sha256':protocol['manifest_sha256']['validation_benchmark8'],
         'validation_ids':[r['id'] for r in validation], 'benchmark8_ids':[r['id'] for r in benchmark],
         'validation_files':[{key:r[key] for key in ('id','image_sha256','mask_sha256')} for r in validation]})
    print(json.dumps({'status':'passed','output':str(out),'counts':report['selected_crop_counts'],
                      'candidate_counts':report['candidate_crop_counts'],'splits':audit['splits'],
                      'manifest_sha256':protocol['manifest_sha256']},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-archive', required=True)
    parser.add_argument('--output', default='data/monuseg_official30_grpo_v1')
    parser.add_argument('--report', default='reports/2026-10-16-data')
    parser.add_argument('--protocol', default='protocols/monuseg_official30_grpo_v1.json')
    main(parser.parse_args())
