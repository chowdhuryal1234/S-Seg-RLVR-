# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · NuNext reproduction: PanNuke data
# MAGIC
# MAGIC Downloads the three official PanNuke folds (about 700 MB each, from Warwick TIA) unless they are already under
# MAGIC `<artifacts>/raw/pannuke/fold_{1,2,3}.zip`, converts them with `scripts/prepare_pannuke.py` into PNG images
# MAGIC and centroid manifests, and stores the result on the Volume as `<artifacts>/data/pannuke_nunext_v1.zip`.
# MAGIC Folds 1 and 2 are training data and fold 3 is internal validation, as in the paper.
# MAGIC
# MAGIC Any compute works (no GPU needed). `masks.npy` is memory-mapped, but extraction needs about 30 GB of local disk.
# MAGIC If warwick.ac.uk is blocked, upload the three zips to `<artifacts>/raw/pannuke/` by hand.

# COMMAND ----------

# MAGIC %pip install -q numpy==2.2.6 pillow==11.3.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, sys, json
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()
import nunext_common as nx

REBUILD = False   # True rebuilds the dataset zip even if it already exists on the Volume

report = nx.prepare_dataset(rebuild=REBUILD)
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Expected: fold 1 = 2,656 images, fold 2 = 2,523, fold 3 = 2,722; 189,744 nuclei in total.

# COMMAND ----------

folds = nx.ensure_dataset()["folds"]
print({f["fold"]: (f["images"], f["nuclei"]) for f in folds}, "total nuclei:", sum(f["nuclei"] for f in folds))
