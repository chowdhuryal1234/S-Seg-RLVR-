# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Prepare the official30 SFT/GRPO dataset
# MAGIC
# MAGIC CPU only (any serverless compute works). Recreates `data/monuseg_official30_grpo_v1` with the unchanged
# MAGIC `scripts/prepare_official30_grpo.py`: 256 training and 64 validation 128×128 crops, Iris's patient assignments,
# MAGIC 1–8 visible nuclei per crop, plus the fixed 8-crop validation benchmark.
# MAGIC
# MAGIC The preparation script reads a zip archive, so the extracted training folder is downloaded from the Volume (Files API)
# MAGIC and zipped on local disk first. Member names and bytes are unchanged; only the archive container differs from Harvard's.

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements-cpu.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path
sys.path.insert(0, os.getcwd())
import dbx_common as dbx
dbx.configure_environment()

# Optional: a local folder holding a copy of Harvard's data/monuseg_official30_grpo_v1/*.jsonl, for a row-by-row
# comparison. To use it, upload the files to the Volume and download them, e.g.
#   REFERENCE_DIR = dbx.local_root() / "reference"; dbx.volume().download_tree(dbx.remote("reference"), REFERENCE_DIR)
REFERENCE_DIR = None

store, train_dir, target = dbx.volume(), dbx.PATHS["train_dir"], dbx.remote("data", dbx.DATASET_NAME)
print("source folder:", train_dir, "| target:", target)
assert store.is_dir(train_dir), "Set PATHS['train_dir'] in dbx_common.py"
assert not store.is_dir(target), f"{target} already exists; the dataset is frozen. Delete it deliberately to rebuild."

# COMMAND ----------

# Download the extracted folder through the Files API, then zip it in sorted member order (stored, so bytes are
# unchanged). The archive path is fixed so that rebuilt manifests record the same source path.
source = dbx.local_root() / "source" / "monuseg_train"
if source.exists():
    shutil.rmtree(source)
print(f"downloaded {store.download_tree(train_dir, source)} files")
archive = dbx.local_root() / "source" / "MoNuSeg_2018_Training_data.zip"
files = sorted(p for p in source.rglob("*") if p.is_file())
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
    for path in files:
        z.write(path, path.relative_to(source).as_posix())
print(f"{len(files)} files -> {archive} ({archive.stat().st_size / 2**20:.0f} MiB)")
print("tissue images:", sum(p.suffix.lower() in (".tif", ".tiff") for p in files),
      "| annotations:", sum(p.suffix.lower() == ".xml" for p in files))

# COMMAND ----------

work = dbx.local_root() / "data-prep" / time.strftime("%Y%m%d-%H%M%S")
work.mkdir(parents=True)
local_out = work / dbx.DATASET_NAME
command = [sys.executable, "scripts/prepare_official30_grpo.py", "--train-archive", str(archive),
           "--output", str(local_out), "--report", str(work / "report"), "--protocol", str(work / "protocol.json")]
print("$", " ".join(command))
result = subprocess.run(command, cwd=dbx.REPO, capture_output=True, text=True)
print(result.stdout[-6000:], result.stderr[-6000:])
result.check_returncode()

# COMMAND ----------

# Upload the frozen dataset and its audit to the Volume, and keep the local copy for 03/04 on this compute.
print(f"uploaded {store.upload_tree(local_out, target)} files to {target}")
print(f"uploaded {store.upload_tree(work / 'report', target + '_report')} report files")
if dbx.dataset_dir().exists():
    shutil.rmtree(dbx.dataset_dir())
shutil.copytree(local_out, dbx.dataset_dir())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare with the Harvard dataset
# MAGIC
# MAGIC Manifest rows embed the absolute archive path, so whole-file hashes are expected to differ from Harvard's.
# MAGIC The path-independent checks are: same crop IDs, same pixel hashes. Use `REFERENCE_DIR` for the full row comparison.

# COMMAND ----------

protocol = json.loads((local_out / "frozen_protocol.json").read_text())
for name, expected in dbx.HARVARD_MANIFEST_SHA256.items():
    actual = protocol["manifest_sha256"][name]
    print(f"{name:22} {'same file hash as Harvard' if actual == expected else 'file hash differs (expected: archive path)'}")
for split in ("train", "validation", "validation_benchmark8"):
    rows = dbx.read_jsonl(local_out / f"{split}.jsonl")
    print(f"{split:22} {len(rows)} crops, {len({r['patient_id'] for r in rows})} patients, "
          f"{sum(r['instance_count'] for r in rows)} visible nuclei")
if REFERENCE_DIR:
    print(json.dumps(dbx.compare_manifests(local_out, REFERENCE_DIR), indent=2))