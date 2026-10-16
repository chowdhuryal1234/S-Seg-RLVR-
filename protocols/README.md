# MoNuSeg official training membership

`monuseg_official30.json` records the exact 30 full image IDs extracted from the
official three-page `Training_Patient_Organ_information.pdf`. It includes the
source URL and SHA-256 of the locally verified PDF. The protocol also records
the seven additional image IDs found by comparing the local 37-image training
archive with this list; image count alone was not used to identify them.

Use `scripts/prepare_monuseg.py --cohort official30` together with its usual
archive, output and sampling arguments to enforce this membership. All 30
canonical archive pairs must be present. Noncanonical images are excluded by
exact ID and recorded in `preparation_report.json` under `cohort`. Existing
NegativeROA/degenerate-polygon exclusions and crop eligibility checks then run
on the canonical images and are reported separately. An official source cohort
does not make the resulting selected crops an official benchmark.

Omitting `--cohort` preserves the original engineering preparation behavior.
The old audit's `expected_images=30` is only a count diagnostic, not a membership
filter. Existing pilot outputs are not relabeled or regenerated.

For training-only prompt experiments:

```python
from nucleus_rl.cohort import load_official30_cohort, validate_training_rows

cohort = load_official30_cohort()
allowed_ids = set(cohort["image_ids"])
# Choose fixed crops from train rows whose source_image_id is in allowed_ids.
# Do this before generation; retain and report every excluded source ID.
selection_audit = validate_training_rows(selected_rows, cohort)
```

The validator rejects validation/test rows, unknown source IDs, inconsistent
patient IDs, and duplicate row IDs. Its JSON-safe return value records the
selected row/image/patient IDs plus source and protocol hashes. It permits a
canonical subset from an older manifest without requiring new metadata, but
checks only identity and split fields, not image pixels or mask correctness.
Save this audit with the experiment. Full masks used for scoring must never
enter the VLM formatting example.
