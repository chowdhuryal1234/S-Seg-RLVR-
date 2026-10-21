# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Install dependencies and stage model assets
# MAGIC
# MAGIC Serverless GPU, 1×H100, environment **Standard v6**. Installs the pinned SFT/GRPO stack, downloads the SAM2.1 tiny
# MAGIC checkpoint (hash-checked against the Harvard runs), and mirrors the pinned Qwen2.5-VL-7B snapshot to the Volume so later
# MAGIC sessions can start without downloading. Safe to rerun: finished steps are skipped.

# COMMAND ----------

# MAGIC %pip install -q setuptools==75.8.0 wheel==0.45.1 torch==2.8.0 torchvision==0.23.0

# COMMAND ----------

import os
os.environ["SAM2_BUILD_CUDA"] = "0"  # The optional SAM2 CUDA extension is not used.

# COMMAND ----------

# MAGIC %pip install -q --no-build-isolation -r ../../requirements-databricks.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, shutil, sys, urllib.request
from pathlib import Path
sys.path.insert(0, os.getcwd())
import dbx_common as dbx

dbx.configure_environment()
import torch, transformers, peft, sam2
print({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "peft": peft.__version__})
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Attach a BF16-capable GPU (H100 or A10)"
print(torch.cuda.get_device_name(), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")

# COMMAND ----------

# SAM2.1 tiny checkpoint, verified against the hash recorded for the Harvard runs.
checkpoint = dbx.sam_checkpoint()
if not checkpoint.is_file():
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    local = dbx.local_root() / checkpoint.name
    urllib.request.urlretrieve(dbx.SAM_URL, local)
    shutil.copyfile(local, checkpoint)
    local.unlink()
digest = dbx.file_sha256(checkpoint)
assert digest == dbx.SAM_SHA256, f"SAM checkpoint hash {digest} differs from the frozen baseline"
print("SAM checkpoint OK:", checkpoint)

# COMMAND ----------

# Pinned Qwen snapshot: local cache <- Volume mirror <- Hugging Face (then mirrored).
model = dbx.sft_config()["models"]["7b"]
print(dbx.ensure_model(model["model"], model["revision"]))

# COMMAND ----------

# Smoke test: SAM2 builds on CPU and the Qwen processor loads from the pinned revision.
from nucleus_rl.segmenter import FrozenSAM2
from transformers import AutoProcessor
segmenter = FrozenSAM2(str(checkpoint), device="cpu")
processor = AutoProcessor.from_pretrained(model["model"], revision=model["revision"])
print("SAM2 and Qwen processor loaded.")
