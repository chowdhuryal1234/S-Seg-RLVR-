# Frozen official30 GRPO data — 2026-10-16

Prepared 2026-10-16T03:32:10.983344+00:00 before overnight model outputs. This is a full-mask engineering training/validation protocol using only official MoNuSeg training sources. The official test archive was not opened.

| Manifest | Crops | Contributing patients | Visible instance occurrences |
|---|---:|---:|---:|
| `train.jsonl` | 256 | 19 | 1,584 |
| `validation.jsonl` | 64 | 4 | 384 |
| `validation_benchmark8.jsonl` | 8 | 4 | 51 |

Copy the entire `data/monuseg_official30_grpo_v1/` directory (approximately 14.6 MB) to the remote repository. Image/mask paths are relative to each manifest directory. The eight-crop comparison set is an exact subset of validation64, with two crops per validation patient; use the identical saved rows and pixels for 7B and 32B. It contains 4–8 visible nuclei per crop and is a small timing/comparison set, not a representative benchmark.

Source identity is enforced by the exact 30-ID protocol in `protocols/monuseg_official30.json`. Seven additional archive images are excluded by ID. Five canonical images remain excluded under the established rasterizer policy: `TCGA-50-5931`, `TCGA-E2-A1B5`, `TCGA-HE-7128`, `TCGA-HE-7129`, and `TCGA-HE-7130` (degenerate polygons or NegativeROA semantics). The other 25 source patients retain Iris's original train/validation assignments: 21 train and four validation. `TCGA-KB-A93J` and `TCGA-RD-A8N9` contribute no crops under the eight-instance limit, leaving 19 contributing training patients.

Every crop is an unresized 128×128 RGB/mask window from a non-overlapping grid. Selection is seeded patient-balanced round-robin, without clarity, color, minimum-instance-area, or complete-nucleus filters. Capacity filtering retains 408 of 1,029 training grid crops and 72 of 196 validation grid crops; 619 training and 124 validation crops are excluded for exceeding eight visible instances. Two training crops are empty. Each 1000×1000 source's grid covers 896×896 pixels; its remaining right/bottom strips are not sampled. These restrictions and patient balancing limit representativeness.

Partial nuclei are retained with every visible pixel supervised: 250/256 training and 61/64 validation crops contain truncated source instances. No pixels are erased or ignored. Metadata identifies truncated, border-touching and source-clipped IDs. Source instances crossing adjacent crop boundaries may occur in multiple crops within one split: the training set's 1,584 occurrences correspond to 1,466 unique source IDs; validation has 384 occurrences from 340 unique source IDs. No patient or source instance crosses the new train/validation split.

Qwen must receive only the RGB image and fixed task prompt/format example. Masks, image-specific counts, instance IDs, patient metadata, and GT coordinates are scorer/reward information, never prompt inputs. GT centroid fields were removed from portable manifests. This remains full-mask supervision for the overlap reward.

All 320 saved RGB crops and masks match their source raster windows exactly. Checks passed for official30 membership, patient separation, containment, file hashes, unique crop IDs, duplicate RGB pixels, non-overlapping windows, and partial-instance metadata. The independent post-save audit also verified every manifest/hash and validation8 membership. All 26 existing cohort/preparation tests passed.

Historical reuse is explicit: 22 training and 14 validation crops also appear in the earlier Iris-aligned pilot. Validation is frozen engineering validation, not a new untouched test set. The older `monuseg_engineering_v1` used conflicting patient assignments, including training patient `TCGA-G9-6363` now in validation. Start from fresh base policies; do not silently reuse adapters trained under that conflicting split.

The frozen protocol and hashes are in `protocols/monuseg_official30_grpo_v1.json` and the dataset's `frozen_protocol.json` / `frozen_validation.json`. Detailed exclusions, distributions, source-member hashes and historical overlaps are in `reports/2026-10-16-data/data_audit.json`; independent checks are in `independent_manifest_audit.json`. Prior datasets and run artifacts were preserved.

Reproduce into new, empty destinations using `scripts/prepare_official30_grpo.py --train-archive /path/to/MoNuSeg_2018_Training_data.zip --output data/new_name --report reports/new_name --protocol protocols/new_name.json`. Seeds are 20261016 for preparation and 20261017 for validation8. The script refuses to overwrite existing destinations.
