# Prompt-policy pilot: meeting brief

**We implemented and exercised the GRPO pipeline on an A100, including verified adapter updates and frozen-model checks. We have not demonstrated a held-out segmentation improvement. The immediate blocker is getting enough valid structured outputs to produce a useful reward signal.**

## Pipeline and data

**Image → Qwen2.5-VL-3B text LoRA → boxes and points → frozen SAM2.1 tiny → masks → `R_seg + R_fmt` → GRPO.** Only the text adapters are trainable. `R_seg` uses foreground IoU against full reference masks; `R_fmt` checks the answer format and coordinates. Invalid outputs receive zero reward. PQ and count error are additional diagnostics: foreground IoU alone can hide incorrect instance separation. This pilot uses full-mask rewards; it is not yet a weak-supervision result.

The current data preserve Iris's patient assignments while excluding six images with ambiguous/degenerate annotations. The selected **32 training / 16 validation crops** cover **19 / 4 patients**, respectively. Crops are 128×128 with at most eight visible nuclei; partial instances remain supervised. The official test set is untouched. These selected crops are not representative whole-image benchmarks.

## What the saved runs show

| Run | Verified outcome | Interpretation |
|---|---|---|
| Original-coordinate, 10-step smoke run | 10 optimizer steps; 9 nonzero-gradient steps; adapter hash changed. Only **1/20** training completions was valid, with reward variation in **1/10** groups. | Real optimization ran, but useful rewards were very sparse. |
| Numeric-example prompt, processed coordinates | **0/8** valid completions across four training crops; no optimization. | Adding a concrete output example did not resolve formatting in this small diagnostic. |
| Processed-coordinate pilot, configured for 50 steps | Automatically **stopped at step 10**: **0/20** valid training completions, all rewards zero, zero nonzero-gradient steps, adapter hash unchanged. | It did not complete 50 steps or produce a useful update. |

Both training runs have unchanged **SAM and Qwen vision-encoder parameter hashes**. Their 16-crop validation evaluations were **0/16 valid before and after**, with pipeline PQ=0 and foreground IoU=0. These zeros reflect rejected outputs; they do not establish that a correctly prompted SAM cannot segment nuclei. **No held-out quality gain was observed.**

The coordinate handling now measures Qwen's processed frame (224×224 for these 128×128 crops) and converts predicted coordinates back once for SAM. Format failures remain, including missing answer wrappers, malformed JSON, and coordinate violations. With all group rewards equal to zero, GRPO has no relative task-reward preference to learn from.

**Qwen2.5-VL-7B format diagnostic: pending.** No outcome is included in this brief.

Evidence: [original smoke result](runs/grpo_smoke_10/result.json), [training completions](runs/grpo_smoke_10/training_completions/summary.json), [numeric-example diagnostic](runs/base_policy_example/result.json), [stopped processed-coordinate run](runs/grpo_pilot_50/result.json).

## What the oracle diagnostic establishes

On **four training crops**, annotation-derived prompts gave frozen SAM2 mean foreground IoU **0.848 with tight boxes**, **0.717 with five-pixel padding**, and **0.180 with full-image boxes**. SAM stayed unchanged. This shows prompt sensitivity with privileged label information. **It is not learned-policy performance, held-out evaluation, or a performance ceiling:** no VLM or optimizer was used. [Saved oracle diagnostic](runs/oracle_iris_split_cpu/summary.json)

## How this relates to Iris

Iris's saved notebook uses **37 images, split 31/6 with seed 0**, a frozen SAM ViT-B image encoder, and a trained convolutional head. It reconstructs six full validation images from overlapping 512×512 tiles and averages per-image PQ with matching **IoU >0.5**.

| Same postprocessing | Full-mask PQ | Weak-label PQ |
|---|---:|---:|
| Connected components | 0.464 | 0.299 |
| Watershed | 0.492 | 0.289 |

These are **Iris's saved results, not our reproduction**. Our crop selection, rasterizer, annotation exclusions, backbone, and training method differ. We cannot compare the pilot's numbers directly with hers. [Notebook audit](baselines/iris/AUDIT.md)

## Decisions for Alex

1. Should we add a small **training-only supervised format warmup** before GRPO, reporting it separately from a pure-GRPO start?
2. Should the prompt schema accept the model's native JSON style, or retain the strict answer wrapper and train toward it? Fix the acceptance rule before comparing runs.
3. For the next milestone, should we first require reliable valid outputs and reward variation, then a paired validation improvement under an agreed rasterizer and evaluation protocol before adding weak/structural rewards?
