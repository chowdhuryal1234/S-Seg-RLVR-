# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Prepare the official30 SFT/GRPO dataset
# MAGIC
# MAGIC CPU only (any serverless compute works). Recreates `data/monuseg_official30_grpo_v1` with the unchanged
# MAGIC `scripts/prepare_official30_grpo.py`: 256 training and 64 validation 128×128 crops, Iris's patient assignments,
# MAGIC 1–8 visible nuclei per crop, plus the fixed 8-crop validation benchmark.
# MAGIC
# MAGIC The preparation script reads a zip archive, so the extracted training folder on the Volume is zipped to local disk
# MAGIC first. Member names and bytes are unchanged; only the archive container differs from Harvard's.

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

# Optional: a copy of Harvard's data/monuseg_official30_grpo_v1 (just the *.jsonl files) uploaded to the Volume,
# for a row-by-row comparison. Leave as None if you do not have it.
REFERENCE_DIR = None  # e.g. dbx.artifact("reference", "monuseg_official30_grpo_v1")

train_dir, target = dbx.PATHS["train_dir"], dbx.dataset_dir()
print("source folder:", train_dir, "| target:", target)
assert train_dir.is_dir(), "Set PATHS['train_dir'] in dbx_common.py"
assert not target.exists(), f"{target} already exists; the dataset is frozen. Delete it deliberately to rebuild."

# COMMAND ----------

# Zip the extracted folder in sorted member order (no compression, so bytes are stored as-is).
work = dbx.local_root() / "data-prep" / time.strftime("%Y%m%d-%H%M%S")
work.mkdir(parents=True)
archive = work / "MoNuSeg_2018_Training_data.zip"
files = sorted(p for p in train_dir.rglob("*") if p.is_file())
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
    for path in files:
        z.write(path, path.relative_to(train_dir).as_posix())
print(f"{len(files)} files -> {archive} ({archive.stat().st_size / 2**20:.0f} MiB)")
print("tissue images:", sum(p.suffix.lower() in (".tif", ".tiff") for p in files),
      "| annotations:", sum(p.suffix.lower() == ".xml" for p in files))

# COMMAND ----------

local_out = work / dbx.DATASET_NAME
command = [sys.executable, "scripts/prepare_official30_grpo.py", "--train-archive", str(archive),
           "--output", str(local_out), "--report", str(work / "report"), "--protocol", str(work / "protocol.json")]
print("$", " ".join(command))
result = subprocess.run(command, cwd=dbx.REPO, capture_output=True, text=True)
print(result.stdout[-6000:], result.stderr[-6000:])
result.check_returncode()

# COMMAND ----------

# Copy the frozen dataset and its audit to the Volume.
shutil.copytree(local_out, target, copy_function=shutil.copyfile)
shutil.copytree(work / "report", target.parent / f"{dbx.DATASET_NAME}_report", copy_function=shutil.copyfile,
                dirs_exist_ok=True)
print("written:", target)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare with the Harvard dataset
# MAGIC
# MAGIC Manifest rows embed the absolute archive path, so whole-file hashes are expected to differ from Harvard's.
# MAGIC The path-independent checks are: same crop IDs, same pixel hashes. Use `REFERENCE_DIR` for the full row comparison.

# COMMAND ----------

protocol = json.loads((target / "frozen_protocol.json").read_text())
for name, expected in dbx.HARVARD_MANIFEST_SHA256.items():
    actual = protocol["manifest_sha256"][name]
    print(f"{name:22} {'same file hash as Harvard' if actual == expected else 'file hash differs (expected: archive path)'}")
for split in ("train", "validation", "validation_benchmark8"):
    rows = dbx.read_jsonl(target / f"{split}.jsonl")
    print(f"{split:22} {len(rows)} crops, {len({r['patient_id'] for r in rows})} patients, "
          f"{sum(r['instance_count'] for r in rows)} visible nuclei")
if REFERENCE_DIR:
    print(json.dumps(dbx.compare_manifests(target, REFERENCE_DIR), indent=2))
