# Iris baseline notebook audit

Read-only inspection on 2026-10-09; **the notebook was not executed**. Source: `Baseline1_and_2.ipynb` (original filename `Baseline1&2.ipynb`), provided Git ref `1d8b70e4b8b09f7b8db6cb2d3b44bbe999c0b554`, SHA256 `9efc9237435fbebbb8f177f6fa51728909521856c4a5afec6962778b2780fe76`. References below use **zero-based notebook cell indices**. Saved outputs are evidence of reported runs, not an independently reproduced result or verified checkpoint. All code-cell execution counts are null.

## Corrections that affect the meeting

- Iris uses **37 image/XML pairs, not 30**: cell 21 saved output explicitly says 37. Cell 24's 37 patient tokens match all 37 patient tokens in our local training ZIP. This establishes patient membership agreement, not image-byte/annotation-hash identity. Withdraw the previous “Iris used 30” warning.
- The saved split is **31 training images / 6 validation images**, seed 0, all from the raw training directory. Our 8/4-patient, 32/16-crop engineering set is different.
- **0.492 is full-mask + watershed; 0.299 is weak-label + connected components.** Holding postprocessing fixed gives full/weak PQ **0.464/0.299 (CC)** or **0.492/0.289 (watershed)** (cells 42, 52).
- Training optimizes a **custom convolutional head with 387,777 parameters**, calculated from its declared layers. The printed **4.05834 million** “trainable SAM parameters” describes SAM's unused mask decoder, not the parameters optimized in either baseline (cells 10, 13, 31).

## Data source, cohort, and split

Cell 4 mounts Google Drive. Cell 21 reads existing `ROOT/data/monuseg/raw/train/images/*.tif` and `annotations/*.xml`, where `ROOT=/content/drive/MyDrive/s_seg_rlvr`. It asserts exact stem matching. **No dataset download URL, extraction command, file checksum, or acquisition record appears in the notebook.** The only model download is SAM's checkpoint in cell 29.

Cell 25 uses `random.Random(0)`, groups stems by the second TCGA field (TSS), samples six groups with at least two images, then chooses **one image per sampled group** for validation. The remaining images stay in training. This is **not TSS-disjoint validation**: the same six TSS groups retain training images. Saved patient tokens are unique, so the observed cohort is patient-disjoint at the TCGA patient-ID level.

Exact validation IDs (cell 25 saved output):

1. `TCGA-HE-7129-01Z-00-DX1`
2. `TCGA-UZ-A9PN-01Z-00-DX1`
3. `TCGA-B0-5710-01Z-00-DX1`
4. `TCGA-21-5786-01Z-00-DX1`
5. `TCGA-AR-A1AS-01Z-00-DX1`
6. `TCGA-G9-6363-01Z-00-DX1`

The 31 training IDs are the other stems from the sorted 37-image raw directory. They are recorded with the six validation IDs in `split.json`: independently applying the documented seed-0 split to local ZIP stems exactly reproduces the saved validation list **including order**, without executing notebook code. Validation filenames are directly saved evidence; training filenames are a deterministic reconstruction, not an export of Iris's Drive `split.json`. Saved counts: **24,133 rasterized instances total; 19,784 train / 4,349 validation**. These differ from XML Region totals and should not be conflated with them.

## Masks, weak labels, and image preprocessing

- **Rasterization (cell 21):** `skimage.draw.polygon(ys, xs, shape=(h,w))` uses floating XML coordinates. Skip fewer than three vertices and empty rasterizations. Earlier regions claim overlapping pixels first. IDs increment per nonempty polygon; fully occluded IDs may disappear from the mask. **`NegativeROA` is never inspected.** Distinct-vertex degeneracy is not checked.
- **Weak labels (cell 21):** derive each instance's center of mass from its full rasterized mask; round to a point. Initialize all pixels as ignore=255. Nearest-point/Voronoi boundaries, dilated by a 3×3 kernel, become background=0. Radius-4-pixel filled discs become foreground=1 afterward, overriding background where they overlap. Thus these are **synthetic weak labels derived from complete annotations**, not independently collected point labels.
- **Tiles (cells 26–27):** 512×512, stride 256, final start flush with image edge; 1000-pixel axes use starts `[0,256,488]`. Saved output: **279 train tiles / 54 validation tiles**. Train uses eight rotations/flips; validation uses identity only. The 2,286 cached embeddings are 279×8 + 54.
- **Encoder input (cell 33):** OpenCV BGR→RGB; apply tile augmentation; bilinearly resize 512→1024 (`align_corners=False`); subtract `[123.675,116.28,103.53]` and divide by `[58.395,57.12,57.375]` on the 0–255 scale. No separate stain normalization appears. Frozen SAM embeddings are saved as float16, shape 256×64×64.

## Model, optimization, and checkpoints

SAM checkpoint: `sam_vit_b_01ec64.pth`, downloaded from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`; saved size 375.042383 MB, saved whole-SAM count 93.735472 million parameters (cells 29–30). No checksum is verified. Dependencies are installed without pinned versions/commits.

Only `sam.image_encoder` supplies cached features, under eval/no-grad. The new `SegHead` upsamples three times bilinearly; each stage applies 3×3 convolution, GroupNorm(8), GELU: channels 256→128→64→32, followed by a 1×1 32→1 convolution. It outputs **one semantic foreground logit per pixel**, not an instance map. There are no SAM prompts and the SAM mask decoder is not used for these predictions (cells 10, 13, 33).

Both runs: seed 0; 5,000 optimizer steps; batch 8; AdamW learning rate 1e-3, weight decay 0.01; OneCycleLR `pct_start=0.1`; FP16 autocast and GradScaler. Loss is BCE-with-logits plus soft Dice over all valid pixels in the batch. Full masks supervise foreground/background everywhere. Weak labels omit 255 pixels and pass **`pos_weight=6.6`**; the saved background/foreground estimate from the first 60 training tiles is 6.633624 (cells 11, 13, 48, 50).

Saved training logs reach step **5000** for both runs (cells 40 and 50). Code saves `head`, optimizer, scheduler, and step every 500 steps to:

- `/content/drive/MyDrive/s_seg_rlvr/checkpoints/full_s0.pt`
- `/content/drive/MyDrive/s_seg_rlvr/checkpoints/weak_s0.pt`

Those checkpoint files were **not supplied or inspected here**. Resume does not save/restore GradScaler or RNG state; “resume” is not a claim of bitwise replay. Cache/preprocessing skip conditions check file existence/count, not content hashes (cell 7). Some early cells read pre-existing Drive artifacts, so the saved notebook is not proof of a clean fresh-runtime execution.

## Evaluation contract and saved results

Cell 15 averages overlapping **probabilities** back onto each complete validation image before postprocessing. Metrics are computed on those six full images, not averaged over individual tiles.

Postprocessing (cell 14): threshold is strictly `prob > 0.5`; discard predicted components with area <15 pixels. CC uses scipy's default 4-connectivity. Watershed uses Euclidean-distance peaks, `min_distance=5`, `exclude_border=False`, then watershed on negative distance within the foreground mask, followed by the same small-object removal.

PQ (cell 14): enumerate intersections of nonzero GT/prediction IDs; **IoU >0.5** defines unique matches. There is no Hungarian implementation; strict >0.5 is sufficient for unique matching of disjoint instance maps. SQ is matched-pair mean IoU (0 if no matches), DQ is `TP/(TP+0.5FP+0.5FN)`, and PQ=DQ×SQ. Cell 15 reports the **unweighted mean of per-image PQ**, not pooled TP/FP/FN or nucleus-weighted PQ. Its all-empty GT/prediction special case gives DQ=1 but SQ/PQ=0, a minor convention difference to document if reusing it elsewhere.

Merge rate: fraction of predicted IDs covering at least two rounded GT centroids. Split rate: fraction of GT IDs overlapped by at least two predicted IDs, each covering ≥20% of the GT area. Count error is mean absolute instance-count difference per image; prediction/GT count ratio is pooled across images.

| Condition | CC PQ | Watershed PQ | Evidence |
|---|---:|---:|---|
| Full-mask trained head | 0.464 | 0.4922643805 | Saved output, cell 42 |
| Weak-label trained head | 0.299 | 0.2893513811 | Saved output, cell 52 |
| GT foreground passed to postprocessor | 0.7793208376 | 0.8662497326 | Saved diagnostic outputs, cells 37–38 |

**No evaluation Dice or global foreground-IoU results are saved** for either postprocessor. Dice appears as a training-loss term (cell 11). SQ is matched-instance mean IoU averaged per image, not foreground IoU: full CC **0.738**, full watershed **0.7387045542**, weak CC **0.625**, weak watershed **0.6207910647**. The CC numbers are available only rounded to three decimals in saved output. Both foreground thresholding and PQ matching use strict **>0.5**, not ≥0.5; area filtering retains exactly 15-pixel predictions (drops <15).

GT→GT saved PQ=1 (cell 37). GT foreground rows are **oracle diagnostics**, not deployable predictions or formal universal performance ceilings. A learned foreground can trade pixel agreement for separation. “Full-mask upper bound” and “weak lower bound” are informal baseline names, not mathematical bounds.

## What must match before numerical comparison

1. Use Iris's exact 31/6 image split, six-image reconstruction, and per-image aggregation; our selected 128×128 sparse crops are an engineering test, not a comparison to her six whole images.
2. Reconcile rasterizers and label exclusions. Our Pillow rounding and conservative six-image exclusions differ from Iris's floating-coordinate skimage rasterizer and ignored NegativeROA flags. In particular, her validation image `TCGA-HE-7129` is excluded by our current conservative preparation.
3. Separate backbone/head, supervision, and optimizer effects. Her model is frozen SAM ViT-B features + trained convolutional head; our pipeline is a token policy + frozen SAM2. Full-mask reward does not constitute a weak-label-budget match.
4. Hold postprocessing and PQ threshold/aggregation fixed, or report all variants. Keep GT-derived prompt diagnostics separate from image-only held-out policy evaluation. A single seed and six validation images do not establish statistical superiority; the notebook's suggested 0.02–0.03 “noise” band is not backed by an uncertainty calculation in the supplied code.

**Meeting-safe statement:** “Iris's saved notebook reports 37 source images, a 31/6 seed-0 split, and whole-image PQ for a frozen SAM ViT-B encoder plus a learned convolutional head; our current token-policy/SAM2 crop experiment validates a different training pipeline and is not yet a like-for-like baseline comparison.”

## Follow-up during this audit

The pilot now preserves Iris's patient assignments using `split.json`. Conservative annotation checks retain26 training patients and 5 validation patients, then select32 training and 16 validation small crops. This still differs from her complete 31/6 whole-image evaluation. An earlier 8/4 patient engineering split is retained only as an initial setup diagnostic.
