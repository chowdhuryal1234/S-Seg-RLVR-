# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 10 · NuNext reproduction: PanNuke data
# MAGIC
# MAGIC Downloads the three PanNuke folds from Hugging Face (`RationAI/PanNuke`, pinned revision, about 1.7 GB of
# MAGIC parquet in total), converts them with `scripts/prepare_pannuke.py` into PNG images and centroid manifests, and
# MAGIC stores the result on the Volume as small zips plus `index.json` under `<artifacts>/data/pannuke_nunext_v1/`.
# MAGIC Folds 1 and 2 are training data and fold 3 is internal validation, as in the paper.
# MAGIC
# MAGIC The Hugging Face copy was checked against the official Warwick fold 3: identical images, nucleus counts,
# MAGIC centroids and classes. `SOURCE = "warwick"` uses the official zips instead (about 30 GB of local disk).
# MAGIC Any compute works (no GPU needed). Downloads and Volume uploads retry after a dropped connection.

# COMMAND ----------

# MAGIC %pip install -q numpy==2.2.6 pillow==11.3.0 pyarrow==19.0.1 huggingface-hub==1.5.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, sys, json
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
import nunext_common as nx

SOURCE = "huggingface"   # or "warwick"
REBUILD = False          # True rebuilds the dataset even if it already exists on the Volume
STORE_RAW = False        # Warwick only: also keep the 700 MB official zips on the Volume

report = nx.prepare_dataset(rebuild=REBUILD, store_raw=STORE_RAW, source=SOURCE)
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Expected images: fold 1 = 2,656, fold 2 = 2,523, fold 3 = 2,722 (fold 3 has 66,654 nuclei).

# COMMAND ----------

folds = nx.ensure_dataset()["folds"]
print({f["fold"]: (f["images"], f["nuclei"]) for f in folds}, "total nuclei:", sum(f["nuclei"] for f in folds))