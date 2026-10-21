# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Supervised warm-up (SFT) of Qwen2.5-VL-7B
# MAGIC
# MAGIC Serverless GPU, **1×H100**, environment Standard v6. Runs the unchanged `nucleus_rl.sft` CLI with the settings in
# MAGIC `configs/sft_experiments.json`, the same arguments the Harvard supervisor (`scripts/run_sft_experiments.py`) builds:
# MAGIC
# MAGIC 1. **Canary**: 20 steps on 4 training / 8 validation crops with D4 augmentation, checked with the Harvard canary criteria.
# MAGIC 2. **Main arms**: `none` and `d4`, 512 optimizer steps × 4 accumulation = 2048 exposures over the 256 training crops,
# MAGIC    evaluated before and after on all 64 validation crops.
# MAGIC
# MAGIC Each run writes to local disk and is mirrored to `<artifacts>/runs/` every 10 minutes and at the end. Run it as a
# MAGIC notebook **job** for the main arms so a closed browser tab does not stop training (sessions are limited to 2 days).

# COMMAND ----------

# MAGIC %pip install -q setuptools==75.8.0 wheel==0.45.1 torch==2.8.0 torchvision==0.23.0

# COMMAND ----------

import os
os.environ["SAM2_BUILD_CUDA"] = "0"

# COMMAND ----------

# MAGIC %pip install -q --no-build-isolation -r ../../requirements-databricks.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, sys, json
from pathlib import Path
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
from scripts.run_sft_experiments import canary_errors, main_run_errors

MODEL_KEY = "7b"                 # Harvard compared 7b/32b/9b/27b; 7b fits one H100 comfortably.
STAGES = ["canary", "main"]      # Drop "canary" once it has passed on this environment.
ARMS = ["none", "d4"]            # Main-stage augmentation arms, run one after another.

config = dbx.sft_config()
model = config["models"][MODEL_KEY]
data = dbx.dataset_dir()
assert (data / "frozen_protocol.json").is_file(), "Run 02_prepare_data first"
assert dbx.sam_checkpoint().is_file(), "Run 01_setup_assets first"
print(dbx.ensure_model(model["model"], model["revision"]))



def show(summary):
    keys = ("completions", "invalid_completions", "mean_seg_reward", "mean_pq", "mean_precision", "mean_recall", "mean_count_error")
    for phase in ("before", "after"):
        values = summary.get(phase, {})
        row = {k: round(values[k], 4) if isinstance(values.get(k), float) else values.get(k) for k in keys}
        row["micro_box_recall"] = values.get("localization", {}).get("micro_box_recall")
        print(f"{phase:6}", row)

# COMMAND ----------

runs = {}
if "canary" in STAGES:
    local, volume = dbx.new_run_dirs(f"sft-{MODEL_KEY}-canary-d4")
    dbx.run_cli(dbx.sft_argv(MODEL_KEY, "canary", config["canary"]["augmentation"], local, config), local, volume)
    summary = json.loads((local / "summary.json").read_text())
    errors = canary_errors(summary, json.loads((local / "metadata.json").read_text()), model["model"], model["revision"])
    show(summary)
    assert not errors, f"Canary failed the Harvard criteria: {errors}"
    print("Canary passed:", volume)
    runs["canary"] = str(volume)

# COMMAND ----------

if "main" in STAGES:
    for arm in ARMS:
        local, volume = dbx.new_run_dirs(f"sft-{MODEL_KEY}-main-{arm}")
        dbx.run_cli(dbx.sft_argv(MODEL_KEY, "main", arm, local, config), local, volume)
        summary = json.loads((local / "summary.json").read_text())
        print(f"== {arm}: {volume}")
        show(summary)
        errors = main_run_errors(summary, config, MODEL_KEY)
        print("contract:", errors or "passed")
        runs[arm] = str(volume)

# Record which runs belong together; 04_grpo reads the d4 adapter from here by default.
index = dbx.artifact("runs", f"sft-{MODEL_KEY}-latest.json")
merged = {**(json.loads(index.read_text()) if index.is_file() else {}), **runs}
index.write_text(json.dumps(merged, indent=2) + "\n")
print(index, json.dumps(merged, indent=2))
