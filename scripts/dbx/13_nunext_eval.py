# Databricks notebook source
# MAGIC %md
# MAGIC # 13 · NuNext reproduction: re-evaluate a saved model
# MAGIC
# MAGIC Greedy decoding on PanNuke fold 3 with an existing SFT or GRPO run (`<run>/model` on the Volume), sharded over
# MAGIC every visible GPU. Reports the strict detection F1 (malformed or truncated output = no predictions, as in
# MAGIC training) and the diagnostic `salvaged_*` F1, where such outputs are read up to their first raster-order break.
# MAGIC No training. About 10 minutes on 8×H100.

# COMMAND ----------

# MAGIC %pip install -q setuptools==75.8.0 wheel==0.45.1 torch==2.8.0 torchvision==0.23.0

# COMMAND ----------

import os
os.environ["SAM2_BUILD_CUDA"] = "0"

# COMMAND ----------

# MAGIC %pip install -q --no-build-isolation -r ../../requirements-databricks.txt scipy==1.15.3

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, sys, json
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
import nunext_common as nx
import torch

RUN = dbx.remote("runs", "nunext-sft-nsft-20261110-060748")   # Volume run folder that contains model/
MAX_NEW_TOKENS = 1024
LIMIT = None                                                   # e.g. 512 for a quick check

GPUS = torch.cuda.device_count()
nx.ensure_dataset()
model = nx.ensure_run_model(RUN)
local, remote = dbx.new_run_dirs("nunext-eval-" + RUN.rstrip("/").split("/")[-1])
launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []
argv = [*launcher, "-m", "nucleus_rl.nunext.evaluate", "--model", model, "--output", local,
        *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--manifest", m)],
        "--bins", str(nx.SFT_SETTINGS["bins"]), "--image-size", str(nx.SFT_SETTINGS["image_size"]),
        "--radius", str(nx.SFT_SETTINGS["radius"]), "--max-new-tokens", str(MAX_NEW_TOKENS)]
if LIMIT:
    argv += ["--limit", str(LIMIT)]
dbx.run_cli(argv, local, remote)

# COMMAND ----------

summary = json.loads((local / "summary.json").read_text())
for key in ("images", "micro_f1", "micro_precision", "micro_recall", "invalid", "errors",
            "salvaged_micro_f1", "salvaged_micro_precision", "salvaged_micro_recall", "seconds"):
    print(f"{key:26} {summary[key]}")
