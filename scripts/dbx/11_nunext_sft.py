# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 11 · NuNext reproduction: supervised fine-tuning
# MAGIC
# MAGIC Serverless GPU **1×H100 or 8×H100**, environment Standard v6. Qwen2.5-VL-3B with 256 added coordinate tokens is
# MAGIC fully fine-tuned (language model, embeddings, output head; vision tower frozen) to emit raster-ordered nucleus
# MAGIC centroids on PanNuke folds 1+2, and evaluated by detection F1 on fold 3.
# MAGIC
# MAGIC Arms (`nucleus_rl.nunext.sft`):
# MAGIC - `nsft`: naive SFT, one-hot next-token loss. Paper target: **F1 0.812** (Table 5, NSFT row).
# MAGIC - `sass`: + spatial-aware soft supervision (σ = 1 bin). Paper: 0.819.
# MAGIC
# MAGIC Run the canary first, then the arms as a notebook **job**. Outputs go to `<artifacts>/runs/nunext-sft-<arm>-<time>/`.
# MAGIC An A10 (24 GB) does not fit full fine-tuning.
# MAGIC
# MAGIC **Multi-GPU:** with several visible GPUs the runner is started by `nucleus_rl.nunext.launch` (one process per GPU, gradients
# MAGIC averaged, validation sharded). The effective batch stays 16: per-GPU accumulation = 16 / GPUs. A CPU test showed
# MAGIC a 2-process run reproduces the single-process losses exactly. If the notebook sees only one GPU it runs as before.

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

STAGES = ["main"]      # Drop "canary" once it has passed here.
ARMS = ["nsft"]          # Run one after another; "nsft" alone answers the first question.
SETTINGS = {"image_size": 896, "epochs": 3, "eval_every": 500}                # Overrides of nx.SFT_SETTINGS, e.g. {"epochs": 1}

EFFECTIVE_BATCH = 16           # Images per optimizer step across all GPUs

import torch
GPUS = torch.cuda.device_count()
print("visible GPUs:", GPUS, [torch.cuda.get_device_name(i) for i in range(GPUS)])
nx.ensure_dataset()
base = nx.ensure_base_model()
print("model:", base)


def argv(arm, output, canary=False):
    settings = {**nx.SFT_SETTINGS, **SETTINGS, **nx.SFT_ARMS[arm]}
    gpus = GPUS if GPUS > 1 else 1
    settings["gradient_accumulation"] = max(1, EFFECTIVE_BATCH // gpus)
    if canary:
        settings.update(nx.SFT_CANARY, gradient_accumulation=max(1, nx.SFT_CANARY["gradient_accumulation"] // gpus))
    # Own launcher: torchrun --standalone's TCP rendezvous times out on Serverless GPU (file rendezvous instead).
    launcher = (["-m", "nucleus_rl.nunext.launch", "--nproc", str(gpus), "--"] if gpus > 1 else [])
    return [*launcher, "-m", "nucleus_rl.nunext.sft", "--model", base, "--revision", "main", "--output", output,
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

# COMMAND ----------

import os, sys, json
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
import nunext_common as nx
import collections
rows = [json.loads(l) for l in dbx.volume().read_text(
    dbx.remote("runs", "nunext-sft-nsft-20261110-060748", "eval-final", "predictions.jsonl")).splitlines()]
print(collections.Counter(r["error"] for r in rows))
import statistics
bad = [r["fn"] for r in rows if not r["valid"]]; good = [r["tp"] + r["fn"] for r in rows if r["valid"]]
print("nuclei per invalid image:", statistics.mean(bad), " per valid image:", statistics.mean(good))

# COMMAND ----------

