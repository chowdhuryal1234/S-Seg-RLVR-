# One-night MoNuSeg prompt-policy pilot

This is a preparation plan. A completed GPU run and performance improvement have not been established.

## Question for tomorrow

Can a trainable vision-language policy produce valid spatial prompts, receive segmentation rewards from a frozen SAM2 model, and complete genuine GRPO updates on nucleus images?

This is the first `R_seg + R_fmt` engineering milestone. It does not establish weak-label efficiency, automatic full-slide instance segmentation, or a new benchmark result.

## Connection to Iris's work

- Reuse her MoNuSeg data protocol, split manifest, evaluation implementation and saved baseline results once those artifacts are available.
- Do not repeat supervised decoder training.
- Her reported full-mask PQ 0.492 and weak-label PQ 0.299 are reference results, not verified reproductions in this checkout.
- Her SAM encoder plus custom convolutional decoder is not assumed to accept spatial prompts. The new policy starts from pretrained Qwen; frozen, promptable SAM2 supplies masks.
- The shared repository was initially empty. Iris's notebook has now been supplied and copied with source provenance under `baselines/iris/`. Saved decoder weights and a matched evaluation run are still needed for a reproduced comparison.

## Narrow scope

Use only small MoNuSeg nucleus crops from the original training archive. Keep original patient/image groups separate before cropping. A provisional engineering split is permitted for a smoke test, with that status visible; it is not a substitute for Iris's split.

The data-analysis document describes 30 training images, while both the local archive and Iris's actual notebook use 37 real TIFF/XML pairs. The earlier mismatch with Iris's cohort is resolved. Her notebook uses 31 training images and 6 validation images; our small crop split and conservative label conversion are engineering choices and remain different from her full-image protocol.

For the bounded engineering task, select 128-pixel crops with 1–8 visible annotated nucleus IDs. Run preparation with the explicit `--border-policy visible` option: include every visible labeled pixel, record truncated IDs, and ignore zero pixels. Requiring only complete nuclei gave too few crops (9 training and 1 validation), so that stricter default is not used for this pilot. Selecting low-count regions is not representative of all MoNuSeg tissue. The target is full-mask supervision for this first run. Patient-grouped validation crops are distinct from training crops; the official test archive is not used to tune or train.

## Model and rewards

- Policy: Qwen2.5-VL-3B-Instruct with language-layer LoRA adapters.
- Segmenter: frozen SAM2.1 tiny, with image and prompt preprocessing checked.
- Action: structured text containing a list of object boxes and foreground points.
- `R_fmt`: valid syntax and usable spatial coordinates.
- `R_seg`: foreground IoU against dense reference masks.
- Report instance PQ, precision/recall and count error separately. These do not become additional training rewards in this first experiment.

Foreground IoU cannot distinguish all merges and splits: equal foreground pixels can have different object identities. Preserve instance IDs in diagnostics and document this limitation.

## Stages and stop conditions

1. Audit archive/image IDs and produce a reproducible crop manifest.
2. Run frozen-SAM training-only diagnostics with ground-truth-derived prompts. These are explicitly oracle diagnostics, not automatic test performance. Check the segmenter can produce usable outlines before training its prompt policy.
3. Save baseline VLM completions, parsed prompts, masks, rewards and validation metrics.
4. Run 5–10 optimizer steps. Require finite gradients, changed adapter weights, unchanged SAM weights, nonconstant group rewards, recorded peak VRAM and seconds per step.
5. If those checks pass and time remains, extend to roughly 50–100 steps. Save checkpoints and reserve at least one hour for the same before/after validation examples and a meeting summary.

Stop rather than run overnight blindly if data membership is unresolved for the claimed comparison, all rewards are identical, the output schema regularly truncates, the segmenter cannot respond to prompts, memory is insufficient, or runtime leaves no evaluation window.

## Resource request

The user's existing Harvard allocation was verified on October 9: one A100 with 80 GB VRAM, 8 CPU cores, 64 GB system RAM, and a 14-hour allocation. This supplies the requested resources; measured model memory and throughput still require a GPU run. Ordinary Transformers generation avoids a separate inference GPU.

Harvard is a user-authorized temporary pilot environment. Keep paths/configuration portable for the mentor's planned Databricks environment.

## Meeting deliverables

- Verified dataset audit and explicit exclusions/split status.
- One architecture diagram and runnable command/configuration.
- If GPU execution succeeds: training evidence, saved policy adapter, before/after validation results and original-image overlays.
- If access or model integration blocks execution: report the exact completed checks and blocker, without claiming training or improved performance.

## Sources

- Supplied data analysis: https://docs.google.com/document/d/1qQ26lzwtXxggEivNzh0cKp5kHuNwUmR_ry0S30JJ7jI/edit
- Seg-R1: https://arxiv.org/html/2506.22624v1
- TRL GRPO: https://huggingface.co/docs/trl/grpo_trainer
- Qwen policy: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- SAM2: https://github.com/facebookresearch/sam2
