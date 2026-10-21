# Databricks Serverless GPU notebooks

These notebooks reproduce the Harvard SFT → GRPO workflow on Databricks Serverless GPU. They call the repository's own
runners (`scripts/prepare_official30_grpo.py`, `nucleus_rl.sft`, `nucleus_rl.grpo`) with the same arguments that the
Slurm supervisor builds. Only the parts tied to Harvard are replaced: Slurm, its deadlines, and the virtual-environment
and cache paths. `tests/test_dbx_notebooks.py` checks that the SFT arguments match `scripts/run_sft_experiments.py`.

## Setup

1. Add the repository as a Databricks Git folder and check out this branch.
2. Edit `PATHS` in `dbx_common.py`:
   - `train_dir` is the extracted MoNuSeg 2018 training folder (30 official images).
   - `artifacts` is a Unity Catalog Volume folder for the dataset, checkpoints, model mirror and runs.
3. Attach **Serverless GPU → 1×H100** and set the Environment panel to **Standard v6**. An A10 (24 GB) is not
   recommended for 7B GRPO.

## Run order

| Notebook | Compute | What it does |
|---|---|---|
| `00_check_environment` | GPU | Reports GPU, Python, disk, network reachability (Hugging Face, PyPI, GitHub, SAM2 host) and Volume behavior. Installs nothing. |
| `01_setup_assets` | GPU | Installs `requirements-databricks.txt`, downloads and hash-checks SAM2.1 tiny, and mirrors the pinned Qwen2.5-VL-7B snapshot to the Volume. |
| `02_prepare_data` | any | Zips the training folder to local disk and builds `monuseg_official30_grpo_v1`: 256 train, 64 validation and 8 benchmark crops. |
| `03_sft` | GPU | SFT canary (20 steps), then the `none` and `d4` arms (512 steps each), validated with the Harvard criteria. |
| `04_grpo` | GPU | GRPO from the SFT `d4` adapter: a 10-step canary first, then the full 600-step run, optionally with the continued-SFT control. |

Run 03 and 04 as notebook **jobs** for full runs, so a closed browser tab does not stop training. Sessions are limited
to two days. Runs write to fast local disk and are mirrored to `<artifacts>/runs/<name>-<timestamp>/` every 10 minutes
and at the end, including on failure. If a session ends mid-run, resume with the runner's `--resume-from-checkpoint`
pointing at the latest mirrored `checkpoints/checkpoint-N`, and give it a new output directory.

## Network access

`00_check_environment` prints one row per host. If every row is `OK`, nothing else is needed. If a row fails, either
ask the workspace admin to allow the host or a mirror, or stage the files on the Volume:

- **Hugging Face blocked:** on a connected machine, run
  `hf download Qwen/Qwen2.5-VL-7B-Instruct --revision <sha> --cache-dir hub`. Upload
  `hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<sha>/` as plain files to the same path under
  `<artifacts>/cache/huggingface/`. The notebooks then copy it to local disk and set `HF_HUB_OFFLINE=1`.
- **PyPI or GitHub blocked:** build a Linux x86_64 / Python 3.12 wheelhouse with
  `pip download -r requirements-databricks.txt -d wheels` and `pip wheel <SAM-2 URL> -w wheels`, then upload it to the
  Volume. In the notebooks, add `--no-index --find-links <artifacts>/wheels` to the `%pip` lines.
- **SAM2 host blocked:** upload `sam2.1_hiera_tiny.pt` to `<artifacts>/checkpoints/`. Notebook 01 still checks its hash.

## Differences from the Harvard runs

- Python 3.12 (Standard v6) instead of 3.11. Package pins are the same as `requirements-sft.txt`; each run records the
  versions it actually used.
- The data comes from an extracted folder that is zipped locally, not Harvard's original zip. Manifests record the
  absolute archive path, so whole-file manifest hashes will differ from Harvard's. To compare crop IDs and pixel hashes
  row by row, set `REFERENCE_DIR` in 02 to a copy of Harvard's `data/monuseg_official30_grpo_v1/*.jsonl`.
- Only the 7B model is configured by default. `MODEL_KEY` accepts the other keys in `configs/sft_experiments.json`,
  but 32B and 27B need more than one H100.
