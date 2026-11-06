# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · SAM oracle on MoNuSeg vs CoNSeP: do the labels cap PQ?
# MAGIC
# MAGIC Question: is the PQ ceiling low because of MoNuSeg's annotations? Every annotated nucleus is prompted the way a
# MAGIC perfect policy would (tight box + interior point), frozen SAM2 segments it, and the result is scored against the same
# MAGIC annotation. No Qwen and no training, so this isolates **annotation vs SAM agreement**: a dataset whose labels SAM can
# MAGIC reproduce well has a high oracle PQ; noisy or unusual boundaries, merged or split nuclei lower it.
# MAGIC
# MAGIC Both datasets go through the same 128 px grid and the same sparse (1–8) / dense (9–32) split. MoNuSeg uses the
# MAGIC project's own rasteriser, so its labels are exactly the ones the models were scored on. All MoNuSeg official-30
# MAGIC images and both CoNSeP splits are used (nothing is trained). FASRC's oracle on the 64 MoNuSeg validation crops was
# MAGIC 0.773 with SAM2.1 tiny; that set is repeated here as a check.
# MAGIC
# MAGIC Any Serverless GPU (A10 is enough). Roughly 15–45 minutes, mostly SAM.

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
from scripts import monuseg_layout, prepare_consep

# ---------- Settings ----------
CONSEP_DIR = f"{dbx.VOLUME_ROOT}/raw/consep"
CONSEP_SPLITS = ("Train", "Test")      # Nothing is trained, so every CoNSeP image can be used
MAX_PER_SUBSET = None                   # e.g. 300 for a quicker, image-balanced run
USE_SAM_LARGE = True                    # Also run SAM2.1 large: is the tiny model itself the limit?
CROPS_NAME = "oracle_crops_v1"          # Volume folder for the crops; a new name rebuilds them
# ------------------------------

store = dbx.volume()
sam_tiny = dbx.ensure_sam()
sams = [f"tiny={sam_tiny}=configs/sam2.1/sam2.1_hiera_t.yaml"]
if USE_SAM_LARGE:
    large = dbx.local_root() / "checkpoints" / "sam2.1_hiera_large.pt"
    if not large.is_file():
        urllib.request.urlretrieve(dbx.SAM_URL.replace("hiera_tiny", "hiera_large"), large)
    sams.append(f"large={large}=configs/sam2.1/sam2.1_hiera_l.yaml")
print(sams)

# COMMAND ----------

# Crops for both datasets, built once and kept on the Volume.
crops = {}
for dataset in ("monuseg", "consep"):
    local, remote = dbx.local_root() / "data" / CROPS_NAME / dataset, dbx.remote("data", CROPS_NAME, dataset)
    if store.is_dir(remote):
        if not (local / "report.json").is_file():
            store.download_tree(remote, local)
    else:
        if dataset == "monuseg":
            source = dbx.local_root() / "source" / "monuseg_train"
            if not source.is_dir():
                store.download_tree(dbx.PATHS["train_dir"], source)
            layout = dbx.local_root() / "source" / "monuseg_layout"
            if not (layout / "layout_report.json").is_file():
                monuseg_layout.build(source, layout)
            prepare_consep.prepare(layout, local, ("All",), max_per_subset=MAX_PER_SUBSET, dataset="monuseg")
            (local / "layout_report.json").write_text((layout / "layout_report.json").read_text())
        else:
            source = dbx.local_root() / "source" / "consep"
            if not source.is_dir():
                store.download_tree(CONSEP_DIR, source)
            prepare_consep.prepare(source, local, CONSEP_SPLITS, max_per_subset=MAX_PER_SUBSET, dataset="consep")
        print(f"uploaded {store.upload_tree(local, remote)} files to {remote}")
    crops[dataset] = local
    report = json.loads((local / "report.json").read_text())
    print(dataset, json.dumps({k: report[k] for k in ("splits", "subsets", "skipped_crops")}))

layout_report = json.loads((crops["monuseg"] / "layout_report.json").read_text())
print("MoNuSeg images excluded (NegativeROA / degenerate polygons):", sorted(layout_report["excluded"]))

# COMMAND ----------

manifests = {"monuseg_val64": dbx.ensure_dataset() / "validation.jsonl",
             "monuseg_sparse": crops["monuseg"] / "sparse.jsonl", "monuseg_dense": crops["monuseg"] / "dense.jsonl",
             "consep_sparse": crops["consep"] / "sparse.jsonl", "consep_dense": crops["consep"] / "dense.jsonl"}
local, remote = dbx.new_run_dirs("sam-oracle")
argv = ["-m", "scripts.sam_oracle", "--output", local, "--max-objects", "32", "--device", "cuda"]
for name, path in manifests.items():
    argv += ["--manifest", f"{name}={path}"]
for spec in sams:
    argv += ["--sam", spec]
dbx.run_cli(argv, local, remote)
summary = json.loads((local / "summary.json").read_text())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC - **PQ** with perfect prompts is the ceiling for any policy using this SAM and these labels.
# MAGIC - **Nucleus IoU** compares each prompt's SAM mask with the nucleus it came from: median, share above 0.5 (the PQ
# MAGIC   match threshold), and whether SAM draws larger or smaller nuclei than the annotators.

# COMMAND ----------

print(f"{'SAM|set':24} {'crops':>5} {'PQ':>6} {'prec':>6} {'recall':>6} {'IoU med':>7} {'IoU>0.5':>7} {'SAM larger':>10}")
for key, r in summary["results"].items():
    print(f"{key:24} {r['crops']:>5} {r['mean_pq']:>6.3f} {r['mean_precision']:>6.3f} {r['mean_recall']:>6.3f} "
          f"{r['nucleus_iou']['median_iou']:>7.3f} {r['nucleus_iou']['iou_over_0.5']:>7.3f} "
          f"{r['sam_mask_larger_than_annotation']:>10.2f}")
print("\nFASRC reference: tiny | monuseg_val64 PQ 0.773")

print("\nNucleus IoU > 0.5 by nucleus size (tiny):")
for key, r in summary["results"].items():
    if key.startswith("tiny|"):
        print(f"  {key:24}", {size: round(v["iou_over_0.5"], 3) if v["iou_over_0.5"] is not None else None
                              for size, v in r["nucleus_iou_by_size"].items()})

print("\nLowest-agreement source images (tiny, sparse + dense):")
for dataset in ("monuseg", "consep"):
    images = {}
    for subset in ("sparse", "dense"):
        for source, v in summary["results"][f"tiny|{dataset}_{subset}"]["by_source_image"].items():
            images.setdefault(source, []).append(v["mean_pq"])
    worst = sorted((sum(v) / len(v), source) for source, v in images.items())[:5]
    print(f"  {dataset}:", [(source, round(pq, 3)) for pq, source in worst])
print("\nMoNuSeg annotation geometry per image (overlapping polygons, fragmented regions):")
for item in sorted(layout_report["kept"], key=lambda i: -i["regions_with_overlap"])[:5]:
    print("  ", item)
