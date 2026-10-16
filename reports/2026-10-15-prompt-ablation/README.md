# Experiment A: one-object versus three-object formatting example

**The three-object example changes output cardinality, but reliable nucleus localization remains unresolved.** On the same eight crops, the one-object example produced one object every time. The three-object example emitted multiple objects on six crops, but three responses failed the unchanged strict validator. Only two final predicted instances matched annotated nuclei at IoU > 0.5.

This is a completed inference diagnostic, not a GRPO training result or a held-out benchmark. No parameters were updated. Both arms used the unadapted Qwen2.5-VL-7B-Instruct base and frozen SAM2.1 tiny.

## Paired results

All eight crops remain in every metric denominator, including invalid responses.

| Metric | One-object example | Three-object example |
|---|---:|---:|
| Mean crop PQ | 0.0000 | 0.0253 |
| Overall recall, TP / 41 annotated nuclei | 0.00% (0/41) | 4.88% (2/41) |
| Mean crop recall | 0.00% | 4.06% |
| Count MAE, nuclei per crop | 4.125 | 3.375 |
| Final predicted instances, including incorrect predictions | 8 | 26 |
| False positives / false negatives | 8 / 41 | 24 / 39 |
| Same-GT duplicate extras before mask merging | 0 | 0 |
| Raw-mask pairs with mutual IoU > 0.5 (overlap proxy) | 0 | 1 |
| Valid responses | 8/8 | 5/8 |
| Responses reaching the 512-token cap | 0/8 | 0/8 |

Emitted object-list lengths in the fixed crop order were `1,1,1,1,1,1,1,1` versus `1,9,8,8,9,8,1,8`. Of the three invalid responses, two were first rejected for exceeding eight objects and one for an out-of-bounds box. No repair, clipping, truncation to eight objects, or selective regeneration was applied. An invalid response yields an empty final prediction under the existing scorer.

The 26 final predictions must not be described as 26 correctly found nuclei: 24 did not meet the matching criterion. Several generated box lists form regular spatial grids; the saved outputs and overlays show poor alignment with real nuclei. Zero same-GT duplicates also does not mean the predictions are good: poorly localized masks can fail to match any annotated nucleus.

The unchanged diagnostic reward decreased from 1.0599 to 0.6634 on average. Mean foreground IoU decreased from 0.0599 to 0.0384, with invalid outputs scored as zero. This experiment did not optimize either reward.

## Fixed protocol

- Source membership: the [official 30-image allowlist](../../protocols/monuseg_official30.json), derived from the official patient PDF with its hash recorded. All 32 candidate training crops passed membership checks; eight were selected before generation.
- Eight crops from eight patients, 128×128 original pixels, 41 visible annotated nuclei. Counts are `2,4,5,6,7,8,4,5`. Selection uses a deterministic round-robin over available count strata, lexical crop IDs, and preference for new patients. Counts select diagnostic densities; they never enter model prompts.
- These are training-source engineering crops, including edge-truncated nuclei. They do not cover all 30 original images. Neither validation nor the official 14-image test set was evaluated or used to choose prompts.
- Qwen revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`; no LoRA adapter loaded. SAM checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
- The two prompts have identical instructions and share the same first numeric example object. The second arm appends two invented objects to that example. Both still request every visible nucleus, at most eight. Coordinates in the example depend only on frame dimensions.
- Both arms: greedy decoding, seed 42, at most 512 new tokens, Qwen processed coordinates, image budgets 224²–448². Each crop was processed at 224×224; the actual image tensor hashes agree across arms. Arm execution order alternates across crops.
- Final-instance PQ/recall use strict IoU > 0.5. Count MAE is the mean absolute final-count error per crop. Duplicate diagnostics inspect separate SAM masks before overlap resolution: repeated matches to the same annotated instance are distinguished from mutually overlapping mask pairs.

The existing preparation code now supports `--cohort official30`. The seven noncanonical sources are filtered by ID before separate annotation checks. Five canonical sources remain excluded by the existing conservative polygon rules; this is a rasterizer limitation, not a claim that the organizers rejected those images.

## Execution and checks

Executed on one Harvard A100 80GB in allocation `53253742`. The paired runner completed in **52.31 seconds**, with **16.27 GiB** peak reserved GPU memory; environment setup and model download are excluded from this timing. Exit status was zero. SAM parameter hashes were identical before/after, the policy had zero trainable parameters or gradients, and optimizer steps were zero.

The 88-test CPU suite passed locally and in the GPU environment. After retrieval, all 16 final maps were independently rescored; all raw-mask duplicate diagnostics, image/mask hashes, pair membership, summaries, and executing Python source hashes were checked. Every selected case was rendered and visually inspected. The runtime dependency snapshot is retained separately from the October 9 environment.

```bash
python -m nucleus_rl.prompt_ablation \
  --manifest data/monuseg_iris_pilot_v1/train.jsonl \
  --sam-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --output runs/a_prompt_ablation_new --max-images 8 \
  --max-completion-length 512 --seed 42

python scripts/render_prompt_ablation.py \
  --run runs/a_prompt_ablation_new \
  --dataset data/monuseg_iris_pilot_v1 \
  --output runs/a_prompt_ablation_new/review
```

## Artifacts and interpretation

- [Exact summary and provenance](summary.json)
- [Per-crop metrics](per_crop.csv)
- [The two exact prompts](prompts.json)
- [Every raw completion and its scores](completions.jsonl)
- [Executed runtime packages](requirements-runtime.txt)

Full local artifacts, masks, and the two contact sheets are under `runs/a_prompt_ablation_20261015/`; the HTML viewer is `review/index.html`. Raw source image pixels and checkpoints remain outside Git. The remote copy is `/n/home01/jzhou1125/rlvr-prompt-ablation-20261015/runs/a_greedy8`.

This supports a narrow finding: prompt-example structure affects how many objects this base model emits on these eight crops. It does not establish that single-object prompting was the only bottleneck, that the model now identifies nuclei reliably, or that the result generalizes. Before investing in longer GRPO, output validity and spatial localization still need attention; simply generating more boxes is insufficient.
