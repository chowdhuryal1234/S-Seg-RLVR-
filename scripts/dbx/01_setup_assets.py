# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Install dependencies and stage model assets
# MAGIC
# MAGIC Serverless GPU, 1×H100, environment **Standard v6**. Installs the pinned SFT/GRPO stack, downloads the SAM2.1 tiny
# MAGIC checkpoint (hash-checked against the Harvard runs, and kept on the Volume), and downloads the pinned Qwen2.5-VL-7B
# MAGIC snapshot to local disk. Safe to rerun: finished steps are skipped. Notebooks 03 and 04 repeat these steps as needed
# MAGIC on their own compute, so this notebook is a quick check that the stack works.

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

import os, sys
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
checkpoint = dbx.ensure_sam()
print("SAM checkpoint OK:", checkpoint, "| Volume copy:", dbx.remote("checkpoints", dbx.SAM_NAME))

# COMMAND ----------

# Pinned Qwen snapshot in the local Hugging Face cache (about 16 GB).
model = dbx.sft_config()["models"]["7b"]
print(dbx.ensure_model(model["model"], model["revision"]))

# COMMAND ----------

# Smoke test: SAM2 builds on CPU and the Qwen processor loads from the pinned revision.
from nucleus_rl.segmenter import FrozenSAM2
from transformers import AutoProcessor
segmenter = FrozenSAM2(str(checkpoint), device="cpu")
processor = AutoProcessor.from_pretrained(model["model"], revision=model["revision"])
print("SAM2 and Qwen processor loaded.")