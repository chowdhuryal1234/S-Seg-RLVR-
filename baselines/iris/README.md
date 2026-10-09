# Iris's Baseline 1 and Baseline 2

[Baseline1_and_2.ipynb](Baseline1_and_2.ipynb) is Iris's original notebook, including saved outputs. It was copied without changing its contents from [xukanz/RLVR](https://github.com/xukanz/RLVR/blob/1d8b70e4b8b09f7b8db6cb2d3b44bbe999c0b554/Baseline1%262.ipynb).

- Original author: Iris, GitHub `xukanz`.
- Original commit: `1d8b70e4b8b09f7b8db6cb2d3b44bbe999c0b554`.
- Verified Git blob: `cb640479c4092cce2add3286bd6fcbf3b4bca466`.
- Status: source and saved outputs reviewed; not executed or reproduced in this repository.

The notebook trains a custom convolutional decoder on cached features from a frozen SAM ViT-B image encoder, using BCE plus Dice. The trained head has 387,777 parameters. It uses 37 MoNuSeg training image/XML pairs and a 31-image training / 6-image validation split, with seed 0. [split.json](split.json) records the saved validation IDs and independently reconstructed training IDs. [AUDIT.md](AUDIT.md) records the label conversion, tiling, metrics and limitations.

The saved results under a common postprocessor are:

| Supervision | Connected components PQ | Watershed PQ |
|---|---:|---:|
| Full masks | 0.464 | 0.492 |
| Weak labels | 0.299 | 0.289 |

The postprocessor differs between the two headline numbers (0.492 and 0.299). The weak labels were simulated from full annotations; these results do not measure real annotation-cost savings. The full-mask result is a supervised reference, not a mathematical performance ceiling.

The notebook contains Colab-specific setup, paths and reconnect guards. Read its setup instructions before running it. The new GRPO pilot uses a different promptable segmenter, frozen SAM2; it does not assume this custom decoder accepts point or box prompts.
