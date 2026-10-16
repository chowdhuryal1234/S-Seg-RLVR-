# C and D: larger checkpoints did not resolve prompt localization

**Neither switching Qwen 7B to 32B nor switching SAM tiny to large improved
automatic instance recall in this diagnostic.** Supplying correct annotation-derived
boxes and points produced much stronger segmentation with either SAM. This
points toward prompt localization and instruction following as priorities for
further investigation under the current input contract.

All comparisons use the same eight official-30 training crops as Experiment A:
eight source patients, 128×128 pixels, and 41 visible annotated nuclei. These
density-selected crops include partial nuclei at the image borders. This is
inference only, with **zero optimizer steps**, not new GRPO training or a held-out
MoNuSeg benchmark. R_fmt + foreground-IoU scoring was unchanged.

## C: Qwen 7B versus 32B, with SAM tiny fixed

The example contains either one or three invented objects; both instructions
still request every visible nucleus, at most eight. Every row retains all eight
crops, including invalid and valid-empty responses.

| Policy | Numeric example | Macro PQ | Macro recall | Micro recall | Count MAE | Valid responses | Predicted / matched |
|---|---|---:|---:|---:|---:|---:|---:|
| 7B | One object | 0.0000 | 0.00% | 0/41 = 0.00% | 4.125 | 8/8 | 8 / 0 |
| 32B | One object | 0.0000 | 0.00% | 0/41 = 0.00% | 2.500 | 7/8 | 21 / 0 |
| 7B | Three objects | 0.0253 | 4.06% | 2/41 = 4.88% | 3.375 | 5/8 | 26 / 2 |
| 32B | Three objects | 0.0000 | 0.00% | 0/41 = 0.00% | 4.000 | 6/8 | 19 / 0 |

Same-reference duplicate extras were zero in all four conditions. There was
one raw overlapping-mask pair in the 7B three-object condition and zero elsewhere.
Zero duplicates does not imply good detection: poorly placed masks can fail to
match any reference nucleus. No completion reached the 512-token cap.

The lower count error for 32B with the one-object example is **not a localization
improvement**. For example, crop `TCGA-21-5784-01Z-00-DX1_x512_y640` has six
annotated nuclei; 32B produced six final instances but zero qualified matches.
Its boxes form a horizontal row rather than following the annotated nuclei.

The 32B one-object condition had one invalid response (missing the required
objects key) and one valid empty list. Its three-object condition had two invalid
out-of-bounds boxes and two valid empty lists. These were retained without repair
or selective regeneration. Poor recall is therefore not explained by formatting
errors alone.

Zero PQ/recall means no final instance exceeded the matching threshold, not no
overlap anywhere. The highest individual final-mask IoU across C32 was 0.4627.

## D: SAM tiny versus large, with saved Qwen 7B prompts fixed

No Qwen generation occurs in D. Both segmenters receive A's exact saved text,
parsed coordinates and coordinate conversion.

| Numeric example | SAM | Macro PQ | Micro recall | Count MAE | Same-GT duplicates | Raw overlap pairs |
|---|---|---:|---:|---:|---:|---:|
| One object | Tiny | 0.0000 | 0/41 = 0.00% | 4.125 | 0 | 0 |
| One object | Large | 0.0000 | 0/41 = 0.00% | 4.125 | 0 | 0 |
| Three objects | Tiny | 0.0253 | 2/41 = 4.88% | 3.375 | 0 | 1 |
| Three objects | Large | 0.0247 | 2/41 = 4.88% | 3.375 | 0 | 0 |

Increasing SAM size did not recover additional true nuclei from the existing
prompts. Tiny replay reproduced A's final maps and raw binary masks exactly.

## Separate oracle diagnostic: correct boxes and points supplied

This uses a tight GT box and an actual interior GT pixel for every annotated
instance. It supplies the true number and locations of nuclei, so these are
**privileged-input scores, not automatic detection performance**.

| SAM | Macro PQ | Macro recall | Micro recall | Count MAE |
|---|---:|---:|---:|---:|
| Tiny | 0.7571 | 96.88% | 40/41 = 97.56% | 0.000 |
| Large | 0.7958 | 100.00% | 41/41 = 100.00% | 0.000 |

All 41 matching does not mean perfect contours: the matching rule is IoU > 0.5.
Large was not better on every measure; oracle mean foreground IoU was 0.8203
for tiny and 0.8162 for large. Oracle scores are never pooled with policy scores.

## Controls and execution evidence

- Fixed crop order, source image and annotation hashes, official-30 membership,
  training-only split, strict parser, SAM overlap handling, and IoU > 0.5 matching.
- C: identical actual image tensors, tokenized inputs, prompts, generation
  defaults, processed-coordinate geometry, seed 42, greedy decoding, and
  512-token cap. BF16 and SDPA, no LoRA, no quantization or CPU offload.
- The 7B replay first reproduced A's 16 answers, masks and all aggregate scores
  exactly. The 32B run then passed the replay's token/image pairing checks.
- Qwen 7B revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
  32B revision `7cfb30d71a1f4f49a57592323337a4a4727301da`.
  These are different released checkpoints, not a pure parameter-count experiment:
  the [official 32B model card](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
  describes additional post-training changes.
- SAM2 implementation `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
  Tiny checkpoint SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`;
  large `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
  Both parameter hashes stayed unchanged; no trainable SAM parameters.
- Harvard A100 80GB, allocation 53253742. C7B: 100.80 seconds / 16.12 GiB
  peak reserved memory; C32B: 234.98 seconds / 63.12 GiB; D: 18.53 seconds /
  1.47 GiB. Times include loading/scoring and exclude downloads; concurrent
  download activity differed, so these are execution records, not speed benchmarks.
- All three runs exited successfully. 111 CPU tests passed locally and remotely.
  The report independently recomputes every final-map metric and raw-mask duplicate
  diagnostic. Source hashes, replay controls and all eight cases were checked.

## What to investigate next

Prioritize the connection between the image and the generated coordinates. Check
prompt/coordinate understanding and visual resolution on training images before
investing in longer GRPO. A small supervised localization warm-up is another
candidate, but its annotation use must be explicit and compared as warm-up-only
versus warm-up-plus-GRPO. None of those follow-up experiments has run here.

This small negative result does not establish that Qwen 32B is generally worse,
that alternative prompts cannot work, or that GRPO cannot help. It shows that
larger released checkpoints alone did not solve the current protocol on these
eight crops. The official test set and validation crops were not used for this
selection or evaluation.

## Artifacts and reproduction

- [Aggregate results, hashes and runtime metadata](summary.json)
- [All per-crop metrics](per_crop.csv)
- [Every Qwen completion](completions.jsonl)
- [Recomputed report verification](verification.json)
- [Exact A replay verification](a_replay_verification.json)
- [Runtime packages](requirements-runtime.txt)
- [Protocol and controls](../../protocols/followup_cd.md)

Full local artifacts: `runs/followup_cd_20261015/`; English visual report:
`runs/followup_cd_20261015/review/index.html` (all eight crops, ten contact sheets).
Remote raw runs: `/n/home01/jzhou1125/rlvr-followup-20261015/runs/`.
Images, masks and checkpoints remain outside Git.

Use `python -m nucleus_rl.prompt_ablation --help` for C. First replay the 7B
checkpoint with `--direct-gpu-load --reference-run <completed-A>`; then use the
32B model/revision above with the new 7B run as `--reference-run`. Keep the same
manifest and SAM tiny checkpoint. `python -m nucleus_rl.segmenter_ablation --help`
runs D from the original A artifacts with both SAM checkpoints. Every run requires
a fresh output directory. Render saved outputs with:

```bash
.venv/bin/python scripts/render_followup_comparison.py \
  --base-run runs/followup_cd_20261015/c_7b \
  --large-policy-run runs/followup_cd_20261015/c_32b \
  --segmenter-run runs/followup_cd_20261015/d_fixed_prompts \
  --dataset data/monuseg_iris_pilot_v1 \
  --output runs/followup_cd_20261015/review_new
```
