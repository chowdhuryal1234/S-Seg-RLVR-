# Follow-up C and D: locate the current bottleneck

These are inference diagnostics on the same eight official-30 training crops
selected for Experiment A (41 visible annotated nuclei). No validation/test
selection, optimizer updates, new reward components, or annotation-policy
changes are part of this stage. This is not a held-out MoNuSeg benchmark.

## C: compare released Qwen checkpoints

Repeat both numerical formatting examples with frozen Qwen2.5-VL-7B-Instruct
and Qwen2.5-VL-32B-Instruct, keeping SAM2.1 tiny fixed. Each checkpoint produces
one greedy completion for each of the eight crops and each prompt condition.
The 7B replay first checks the old A controls and records input token hashes;
32B then checks against that replay. The old A outputs remain preserved.

- 7B revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- 32B revision: `7cfb30d71a1f4f49a57592323337a4a4727301da`.
- No adapters; BF16, SDPA, one GPU, direct shard loading, seed 42.
- Greedy decoding, 512 new-token cap, maximum eight objects, processed pixel
  coordinates, image budgets 224²–448², original 128×128 crops.
- Require identical crop/annotation hashes, instructions, actual image tensors,
  coordinate frames, tokenized input (between the C checkpoints), and generation
  defaults. Only the checkpoint changes within each formatting condition.
- Whole invalid responses remain empty predictions. No selective retries,
  coordinate repair, truncation of object lists, or selecting a better answer.
- If the 32B model cannot fit, report the failure; quantization or CPU offload
  would require a separately named comparison.

The released 32B model also differs in post-training. This comparison cannot
isolate parameter count alone. See the [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct).

## D: compare segmenters with identical prompts

Replay the exact saved A completions into frozen SAM2.1 tiny and large. This
holds Qwen's decisions fixed and measures the effect of the segmenter. Both
formatting conditions retain all eight crops, including invalid responses.

Separately, provide one tight annotation-derived box and an actual interior
foreground pixel per reference instance. This **oracle** diagnostic checks
segmentation when location information is supplied correctly. It exposes the
answer's location/count and must never be presented as automatic detection or
mixed into learned-prompt performance. Oracle prompts are generated once per
crop and shared by tiny and large.

Both segmenters use the same pinned SAM2 implementation, single-mask output,
coordinate conversion and overlap resolution. Load one segmenter at a time.
Record checkpoint SHA-256 and verify parameter hashes unchanged after inference.

## Common measurements and interpretation

Report mean crop PQ, mean crop recall, pooled recall (TP / 41), count MAE,
TP/FP/FN, format validity, duplicate matches to the same reference nucleus, and
overlapping raw-mask pairs. Matching uses strict IoU > 0.5. Count error and
duplicate diagnostics do not change the existing R_fmt + foreground-IoU reward.

Keep every raw answer and final/raw mask. Recompute metrics from saved masks
before rendering every selected crop. Use these small training diagnostics to
choose what to investigate next, not to claim generalization or statistical
superiority. Larger learned-prompt counts alone do not count as improvement.
