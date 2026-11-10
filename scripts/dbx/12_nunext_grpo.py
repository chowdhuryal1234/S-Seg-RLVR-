# Databricks notebook source
# MAGIC %md
# MAGIC # 12 · NuNext reproduction: GRPO
# MAGIC
# MAGIC Serverless GPU **1×H100 (80 GB)**. Starts from a finished NuNext SFT run (notebook 11) and trains with the
# MAGIC distribution-matching F1 reward (`nucleus_rl.nunext.grpo`). Arms:
# MAGIC - `grpo`: plain GRPO. Paper: NSFT+GRPO 0.830; SASS+CoVT+GRPO 0.835.
# MAGIC - `grpo-lvgf`: + low-variance group filtering. Paper: 0.837 (with SASS+CoVT).
# MAGIC - `grpo-lvgf-fgas`: + fine-grained advantage shaping. Paper: 0.840 (with SASS+CoVT).
# MAGIC
# MAGIC The paper's arms all include CoVT, which this reproduction does not implement, so compare the GRPO gain over
# MAGIC its own SFT start rather than the absolute numbers. Watch `filtered_groups` in the log to tune `delta`.

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

SFT_ARM = "nsft"                 # Key in runs/nunext-sft-latest.json written by notebook 11
SFT_RUN = None                   # Or an explicit Volume run folder (string)
STAGES = ["canary", "main"]
ARMS = ["grpo", "grpo-lvgf-fgas"]
SETTINGS = {}                    # Overrides of nx.GRPO_SETTINGS

nx.ensure_dataset()
sft_run = SFT_RUN or dbx.read_json_remote(dbx.remote("runs", "nunext-sft-latest.json"))[SFT_ARM]
init = nx.ensure_run_model(sft_run)
print("init model:", sft_run)


def argv(arm, output, canary=False):
    settings = {**nx.GRPO_SETTINGS, **SETTINGS, **nx.GRPO_ARMS[arm], **(nx.GRPO_CANARY if canary else {})}
    return ["-m", "nucleus_rl.nunext.grpo", "--init-model", init, "--output", output,
            *[a for m in nx.manifests(nx.TRAIN_FOLDS) for a in ("--train-manifest", m)],
            *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--eval-manifest", m)],
            *nx.flags(settings)]


def show(local):
    summary = json.loads((local / "summary.json").read_text())
    keys = ("micro_f1", "micro_precision", "micro_recall", "invalid", "images")
    for phase in ("before", "after"):
        if phase in summary:
            print(f"{phase:6}", {k: summary[phase][k] for k in keys})
    last = summary["steps"][-10:]
    if last:
        print("last steps: reward", round(sum(s["reward"] for s in last) / len(last), 4),
              "filtered groups", sum(s["filtered_groups"] for s in last), "invalid", sum(s["invalid"] for s in last))
    return summary

# COMMAND ----------

if "canary" in STAGES:
    local, remote = dbx.new_run_dirs("nunext-grpo-canary")
    dbx.run_cli(argv("grpo-lvgf-fgas", local, canary=True), local, remote)
    summary = show(local)
    assert summary["completed_steps"] == nx.GRPO_CANARY["max_steps"], "canary did not finish its steps"
    print("canary passed:", remote)

# COMMAND ----------

if "main" in STAGES:
    for arm in ARMS:
        local, remote = dbx.new_run_dirs(f"nunext-{arm}-from-{SFT_ARM}")
        dbx.run_cli(argv(arm, local), local, remote)
        show(local)
        print("Volume:", remote)
