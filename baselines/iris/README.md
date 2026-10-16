# Iris's supervised baselines

Author: Iris, GitHub [`xukanz`](https://github.com/xukanz).

[Baseline1_and_2.ipynb](Baseline1_and_2.ipynb) comes from [the original notebook](https://github.com/xukanz/RLVR/blob/1d8b70e4b8b09f7b8db6cb2d3b44bbe999c0b554/Baseline1%262.ipynb), commit `1d8b70e4b8b09f7b8db6cb2d3b44bbe999c0b554` (original Git blob `cb640479c4092cce2add3286bd6fcbf3b4bca466`). All imported code-cell sources are preserved. Saved outputs, execution metadata, and the results-only markdown section are removed for source-only publication; this copy has not been rerun here.

The notebook trains a convolutional decoder using BCE plus Dice on cached features from a frozen SAM ViT-B image encoder. It includes full-mask and simulated weak-label training. [split.json](split.json) preserves the imported image assignments as a reproducibility input; it is not an official challenge split.

Run the notebook in a GPU notebook environment after reviewing its Colab-specific setup, paths, dependency installation, and reconnect guards. Keep downloaded data, embeddings, and checkpoints outside Git. Clear notebook outputs before committing. The separate GRPO code uses a promptable SAM2 segmenter and does not assume this convolutional decoder accepts point or box prompts.
