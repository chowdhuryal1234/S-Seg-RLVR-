# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · NuNext reproduction: supervised fine-tuning
# MAGIC
# MAGIC Serverless GPU **1×H100 (80 GB)**, environment Standard v6. Qwen2.5-VL-3B with 256 added coordinate tokens is
# MAGIC fully fine-tuned (language model, embeddings, output head; vision tower frozen) to emit raster-ordered nucleus
# MAGIC centroids on PanNuke folds 1+2, and evaluated by detection F1 on fold 3.
# MAGIC
# MAGIC Arms (`nucleus_rl.nunext.sft`):
# MAGIC - `nsft`: naive SFT, one-hot next-token loss. Paper target: **F1 0.812** (Table 5, NSFT row).
# MAGIC - `sass`: + spatial-aware soft supervision (σ = 1 bin). Paper: 0.819.
# MAGIC
# MAGIC Run the canary first, then the arms as a notebook **job**. Outputs go to `<artifacts>/runs/nunext-sft-<arm>-<time>/`.
# MAGIC An A10 (24 GB) does not fit full fine-tuning.

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

STAGES = ["canary", "main"]      # Drop "canary" once it has passed here.
ARMS = ["nsft", "sass"]          # Run one after another; "nsft" alone answers the first question.
SETTINGS = {}                    # Overrides of nx.SFT_SETTINGS, e.g. {"epochs": 1}

nx.ensure_dataset()
base = nx.ensure_base_model()
print("model:", base)


def argv(arm, output, canary=False):
    settings = {**nx.SFT_SETTINGS, **SETTINGS, **nx.SFT_ARMS[arm], **(nx.SFT_CANARY if canary else {})}
    return ["-m", "nucleus_rl.nunext.sft", "--model", base, "--revision", "main", "--output", output,
            *[a for m in nx.manifests(nx.TRAIN_FOLDS) for a in ("--train-manifest", m)],
            *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--eval-manifest", m)],
            *nx.flags(settings)]


def show(local):
    summary = json.loads((local / "summary.json").read_text())
    keys = ("micro_f1", "micro_precision", "micro_recall", "invalid", "images")
    for phase in ("before", "after"):
        if phase in summary:
            print(f"{phase:6}", {k: summary[phase][k] for k in keys})
    for entry in summary.get("evaluations", []):
        print("  step", entry["step"], {k: entry[k] for k in keys})
    return summary

# COMMAND ----------

if "canary" in STAGES:
    local, remote = dbx.new_run_dirs("nunext-sft-canary")
    dbx.run_cli(argv("nsft", local, canary=True), local, remote)
    summary = show(local)
    assert summary["completed_steps"] == nx.SFT_CANARY["max_steps"], "canary did not finish its steps"
    assert all(entry["loss"] == entry["loss"] for entry in summary["losses"]), "non-finite loss"
    print("canary passed:", remote)

# COMMAND ----------

runs = {}
if "main" in STAGES:
    for arm in ARMS:
        local, remote = dbx.new_run_dirs(f"nunext-sft-{arm}")
        dbx.run_cli(argv(arm, local), local, remote)
        show(local)
        runs[arm] = remote
    index = {**dbx.read_json_remote(dbx.remote("runs", "nunext-sft-latest.json")), **runs}
    dbx.volume().write_text(dbx.remote("runs", "nunext-sft-latest.json"), json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
