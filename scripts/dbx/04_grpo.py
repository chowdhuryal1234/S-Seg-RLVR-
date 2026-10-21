# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · GRPO starting from the SFT adapter
# MAGIC
# MAGIC Serverless GPU, **1×H100**, environment Standard v6. Runs the unchanged `nucleus_rl.grpo` CLI: the finished SFT
# MAGIC adapter is merged into the frozen Qwen base (that merged model is also the KL reference), and a fresh language LoRA is
# MAGIC trained with frozen-SAM2 rewards. Validation uses the same evaluator as SFT, so the numbers are directly comparable.
# MAGIC
# MAGIC `OBJECTIVES = ["grpo", "sft"]` also runs the continued-SFT control on the same images and steps.

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

import os, sys, json, shutil
from pathlib import Path
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()

MODEL_KEY = "7b"
SFT_ARM = "d4"                 # Which SFT run to start from (key in runs/sft-<model>-latest.json).
INIT_ADAPTER = None            # Or an explicit Volume path to an SFT run's "adapter" folder.
CANARY = True                  # 10 steps on 4 train / 8 validation crops; set False for the full run.
OBJECTIVES = ["grpo"]          # Add "sft" for the continued-SFT control.
# Overrides of the runner defaults in dbx.GRPO_SETTINGS (canary limits: dbx.GRPO_CANARY).
SETTINGS = {}
print("GRPO settings:", {**dbx.GRPO_SETTINGS, **SETTINGS, **(dbx.GRPO_CANARY if CANARY else {})})

config = dbx.sft_config()
model = config["models"][MODEL_KEY]
data = dbx.dataset_dir()
if INIT_ADAPTER is None:
    index = json.loads(dbx.artifact("runs", f"sft-{MODEL_KEY}-latest.json").read_text())
    INIT_ADAPTER = Path(index[SFT_ARM]) / "adapter"
INIT_ADAPTER = Path(INIT_ADAPTER)
assert (INIT_ADAPTER / "adapter_model.safetensors").is_file(), f"No finished SFT adapter at {INIT_ADAPTER}"
print("SFT adapter:", INIT_ADAPTER)
print(dbx.ensure_model(model["model"], model["revision"]))

# Read the adapter from local disk; the run records its hash either way.
local_adapter = dbx.local_root() / "init-adapters" / f"{INIT_ADAPTER.parent.name}"
if not local_adapter.is_dir():
    shutil.copytree(INIT_ADAPTER, local_adapter, copy_function=shutil.copyfile)

# COMMAND ----------

keys = ("completions", "invalid_completions", "mean_seg_reward", "mean_pq", "mean_recall", "mean_count_error")
for objective in OBJECTIVES:
    name = f"{objective}-{MODEL_KEY}-from-sft-{SFT_ARM}" + ("-canary" if CANARY else "")
    local, volume = dbx.new_run_dirs(name)
    dbx.run_cli(dbx.grpo_argv(MODEL_KEY, objective, local, local_adapter, SETTINGS, CANARY, config), local, volume)
    summary = json.loads((local / "summary.json").read_text())
    print(f"== {name}: {summary['status']}, {summary['optimizer_steps']} steps, "
          f"{summary['groups_with_signal']}/{summary['groups_seen']} groups with reward signal")
    for phase in ("before", "after"):
        values = summary.get(phase) or {}
        print(f"{phase:6}", {k: values.get(k) for k in keys},
              "micro_box_recall:", values.get("localization", {}).get("micro_box_recall"))
    print("Volume:", volume)
