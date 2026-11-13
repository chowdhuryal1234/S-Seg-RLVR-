# Databricks notebook source
# MAGIC %md
# MAGIC # 14 · NuNext: panoptic quality with frozen SAM2
# MAGIC
# MAGIC The detector outputs nucleus centres; PQ needs a mask per nucleus. Each point becomes a positive point prompt
# MAGIC for a frozen SAM2, and the masks are scored against the PanNuke fold 3 instance maps with the project's PQ
# MAGIC (`nucleus_rl.nunext.pq_eval`, a match is IoU > 0.5). Three prompt sources, each with SAM2.1 tiny and large:
# MAGIC - `oracle-point`: ground-truth centroids, the ceiling for a perfect point detector with this SAM
# MAGIC - `oracle-box`: ground-truth tight box + centroid, the prompt style of the project's earlier branches
# MAGIC - `pred-point`: the detector's points, read from an SFT/GRPO run's `eval-final/predictions.jsonl` (no Qwen here)
# MAGIC
# MAGIC Needs dataset v2 (instance masks): run `10_nunext_data` once first. Serverless GPU, 8×H100 or 1×H100.
# MAGIC SAM2 here is the general model, not fine-tuned on nuclei, so these PQs are not comparable with pipelines
# MAGIC that use a nucleus-trained SAM.

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

import os, sys, json, urllib.request
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
import nunext_common as nx
import torch

RUN = dbx.remote("runs", "nunext-sft-nsft-20261112-060031")   # whose eval-final/predictions.jsonl to segment
MODES = ["oracle-point", "oracle-box", "pred-point"]
SAM_SIZES = ["tiny", "large"]
LIMIT = None                                                   # e.g. 256 for a quick check

GPUS = max(torch.cuda.device_count(), 1)
print("visible GPUs:", GPUS)
nx.ensure_dataset()
sams = {"tiny": (str(dbx.ensure_sam()), "configs/sam2.1/sam2.1_hiera_t.yaml")}
if "large" in SAM_SIZES:
    large = dbx.local_root() / "checkpoints" / "sam2.1_hiera_large.pt"
    if not large.is_file():
        large.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(dbx.SAM_URL.replace("hiera_tiny", "hiera_large"), large)
    sams["large"] = (str(large), "configs/sam2.1/sam2.1_hiera_l.yaml")
predictions = dbx.local_root() / "pq" / RUN.rstrip("/").split("/")[-1] / "predictions.jsonl"
if "pred-point" in MODES and not predictions.is_file():
    dbx.volume().download(RUN.rstrip("/") + "/eval-final/predictions.jsonl", predictions)
print("predictions:", predictions)

# COMMAND ----------

results = {}
tag = RUN.rstrip("/").split("/")[-1].replace("nunext-sft-", "").replace("nunext-", "")
for size in SAM_SIZES:
    checkpoint, config = sams[size]
    for mode in MODES:
        local, remote = dbx.new_run_dirs(f"nunext-pq-{size}-{mode}" + (f"-{tag}" if mode == "pred-point" else ""))
        launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []
        argv = [*launcher, "-m", "nucleus_rl.nunext.pq_eval", "--mode", mode, "--output", local,
                "--sam-checkpoint", checkpoint, "--sam-config", config,
                *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--manifest", m)]]
        if mode == "pred-point":
            argv += ["--predictions", str(predictions)]
        if LIMIT:
            argv += ["--limit", str(LIMIT)]
        dbx.run_cli(argv, local, remote)
        results[(size, mode)] = json.loads((local / "summary.json").read_text())

# COMMAND ----------

print(f"{'SAM':6} {'prompts':13} {'PQ':>6} {'DQ':>6} {'SQ':>6} {'mean img PQ':>11} {'TP':>7} {'FP':>7} {'FN':>7}")
for (size, mode), s in results.items():
    print(f"{size:6} {mode:13} {s['pq']:6.3f} {s['dq']:6.3f} {s['sq']:6.3f} {s['mean_image_pq']:11.3f} "
          f"{s['tp']:7d} {s['fp']:7d} {s['fn']:7d}")
