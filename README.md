# S-Seg-RLVR

Nucleus segmentation research: learn a vision-language prompt policy while keeping the mask-generating model frozen.

## Current work

- [Iris's supervised baselines](baselines/iris/README.md): original notebook and saved outputs, copied with source provenance. They have not been rerun here.
- [Prompt-policy pilot](OVERNIGHT_PLAN.md): Qwen text-layer LoRA adapters, frozen SAM2.1 tiny, and only segmentation plus format rewards. The first 7B pilot completed 50 GPU optimizer steps with frozen-model checks. Validation instance PQ stayed unchanged; segmentation quality remains weak.
- [Meeting brief and roadmap](MEETING_BRIEF.md), with [verified aggregate results](reports/2026-10-09/pilot_metrics.json). The initial 3B runs exposed an output-format failure; the 7B run produced a usable reward signal.
- Small MoNuSeg crops are an engineering task. Their scores are not directly comparable with Iris's full-image scores.

```mermaid
flowchart LR
    I["Tissue image + instruction"] --> P["Qwen policy\nTrain text LoRA adapters"]
    P --> O["Sample box + point lists"]
    O --> S["Frozen SAM2 → instance masks"]
    O --> F["Format check"]
    S --> R["Foreground IoU vs full mask"]
    F --> G["R_fmt + R_seg → GRPO"]
    R --> G
    G --> P
```

## Data

Keep archives, derived crops, reference masks, checkpoints and run artifacts outside Git. Preparation reads only the training archive, separates patient groups before cropping, preserves instance IDs, and writes portable manifests with explicit exclusions. The first pilot uses 32 training and 16 validation crops, 128 pixels square, with at most eight visible nuclei.

```bash
python scripts/audit_monuseg.py \
  --train-archive /path/MoNuSeg_2018_Training_data.zip \
  --test-archive /path/MoNuSeg_2018_Testing_data.zip \
  --verify-crc --output runs/data_audit.json

python scripts/prepare_monuseg.py \
  --train-archive /path/MoNuSeg_2018_Training_data.zip \
  --out data/monuseg_engineering_v1 --border-policy visible
```

The explicit `visible` policy keeps partial nuclei, logs their IDs, and supervises all visible labeled pixels. The stricter default requires complete nuclei and may fail when too few eligible crops exist. The command above makes a provisional split. The overnight pilot instead preserves Iris's assignments from [split.json](baselines/iris/split.json), supplying eligible patient lists through `--train-patient-ids` and `--val-patient-ids`: 26 training / 5 validation source patients remain after conservative annotation exclusions. Crop scores still differ from her whole-image evaluation.

## Environment and checks

Use an isolated Python 3.11 environment. CPU data/reward tests need only `requirements-cpu.txt`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
```

For CUDA training, install Torch first, then the pinned GPU dependencies. The optional SAM2 CUDA postprocessing extension is disabled for this pilot.

```bash
.venv/bin/python -m pip install wheel==0.45.1 torch==2.5.1 torchvision==0.20.1
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation -r requirements-gpu.txt
```

Obtain the official `sam2.1_hiera_tiny.pt` checkpoint from [SAM2's checkpoint instructions](https://github.com/facebookresearch/sam2#download-checkpoints). Qwen downloads through Hugging Face; the runner records its resolved commit.

```bash
.venv/bin/python -m nucleus_rl.train --mode train \
  --manifest data/monuseg_iris_pilot_v1/train.jsonl \
  --eval-manifest data/monuseg_iris_pilot_v1/validation.jsonl \
  --sam-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --output runs/pilot_10_steps --max-steps 10
```

The runner saves before/after completions, masks and overlays, reward-group variation, gradients, adapter changes, frozen-model hashes, peak GPU memory and timing. A failed or constant-reward run is not an improvement result. Foreground IoU does not detect every merge or split; instance PQ and count errors are separate diagnostics, not additional training rewards.

Qwen's spatial prompts use its measured processed-image pixels by default. The scorer converts them once to original crop pixels before calling SAM, and saves both representations. For these 128-pixel crops the processor produces 224-pixel images. This follows [Qwen's official grounding convention](https://github.com/QwenLM/Qwen3-VL/blob/2f25a646fb0f329647428eb8dacf19293de6f5d4/cookbooks/spatial_understanding.ipynb). `--coordinate-frame original` is available for an explicitly separate diagnostic.

The official test archive is not used for training or model selection. This is research code, not a clinical system.
