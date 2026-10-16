# S-Seg-RLVR

Research code for nucleus segmentation with a vision-language prompt policy and a frozen SAM segmenter. The GRPO runner updates Qwen language LoRA adapters using format and segmentation rewards.

This repository contains source, tests, dependency files, executable configuration, and short setup instructions. Keep datasets, checkpoints, model outputs, figures, reports, and meeting notes in ignored local directories. Notebooks are stored without execution outputs. Check the Git index before publishing:

```bash
python scripts/check_repository_contents.py
```

## Setup and tests

Use an isolated Python 3.11 environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
```

For GPU training, install Torch and the pinned training dependencies:

```bash
.venv/bin/python -m pip install wheel==0.45.1 torch==2.5.1 torchvision==0.20.1
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation -r requirements-gpu.txt
```

Place the official [SAM2 checkpoint](https://github.com/facebookresearch/sam2#download-checkpoints) under `checkpoints/`. Qwen model weights are downloaded through Hugging Face.

## Run

Prepare a local MoNuSeg training archive, then provide separate patient-disjoint training and validation manifests:

```bash
.venv/bin/python scripts/prepare_monuseg.py --help
.venv/bin/python -m nucleus_rl.train --help
.venv/bin/python -m nucleus_rl.train --mode train \
  --manifest data/monuseg/train.jsonl \
  --eval-manifest data/monuseg/validation.jsonl \
  --sam-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --output runs/pilot --max-steps 10
```

Use the data preparation options to set patient assignments and crop policy before training. Keep the official test set separate from training and model selection. Training writes artifacts locally; do not add them to Git.

## Baseline attribution

[Iris's supervised baseline notebook](baselines/iris/README.md), authored by Iris (`xukanz`), trains a convolutional head on frozen SAM ViT-B features. Its code is preserved with source attribution; imported outputs and result tables are excluded. The prompt-policy implementation is a separate training approach.
