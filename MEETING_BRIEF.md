# First prompt-policy pilot: meeting notes

**Goal:** check whether a VLM can learn to place nucleus prompts using `R_seg + R_fmt`, while SAM stays frozen. We are building the RL pipeline, not claiming a new benchmark score.

## What we have checked

- Iris's notebook is in the shared repository with its original source and credit. It uses 37 images, a 31/6 patient split, a frozen SAM ViT-B image encoder, and a small trained convolutional head.
- Her saved whole-image PQ scores are full/weak **0.464/0.299 with connected components**, or **0.492/0.289 with watershed**. Keep postprocessing fixed when comparing them. These are saved results, not our reproduction.
- The new pilot preserves her patient assignments, excluding six slides with annotation issues. Of the 26/5 eligible source patients, the selected 32/16 crops cover 19 training and 4 validation patients because dense regions exceed the eight-object limit. It does not use the official test images.
- With annotation-derived prompts on four training crops, frozen SAM2 achieved mean foreground IoU **0.848 with tight boxes**, **0.717 with boxes padded by five pixels**, and **0.180 with full-image boxes**. SAM's parameter hash stayed unchanged. This is an oracle diagnostic: no VLM, no optimizer updates, and no held-out performance claim.
- The code passes 61 CPU tests, including coordinate-conversion checks. The 80 GB A100 environment passed GPU imports. The initial ten-step original-coordinate test has logged nonzero gradients; its complete update/freeze audit remains pending in this note. The longer pilot uses Qwen's measured processed coordinates, converted back once for SAM.

## Experiment

**Image → Qwen2.5-VL-3B text LoRA → boxes and points → frozen SAM2.1 tiny → masks → segmentation/format rewards → GRPO.**

First run ten optimizer steps. Check valid outputs, varied rewards, nonzero adapter gradients, changed adapter weights, and unchanged SAM weights. If these pass, run a separate 50-step pilot from the same base model. Save before/after validation predictions and metrics.

`R_seg` is foreground IoU and `R_fmt` checks the output schema and coordinates. PQ and count errors are evaluation diagnostics. Foreground IoU can miss merge/split mistakes, so we should not judge instance segmentation by IoU alone.

## Questions for Alex

1. Should the first segmentation reward remain foreground IoU, or use an instance-aware score once the basic pipeline works?
2. Before comparing with Iris, should we standardize her mask conversion and whole-image evaluation, including how ambiguous annotations are handled?
3. What result should define next week's milestone: a reliable training pipeline, an improvement on a matched validation setup, or a first weak-label experiment?

For the 30-minute meeting: 5 minutes on data/baselines, 10 on the architecture and run evidence, and 15 on decisions and next steps.
