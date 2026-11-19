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
# MAGIC - `pred-box`: a box-token detector's (arm `nsft-box` in 11) predicted boxes + centres; RUN must be such a run
# MAGIC
# MAGIC
# MAGIC `VARIANTS` are inference-only fixes for point prompts: `negatives` sends the nearest other points (within 32 px)
# MAGIC as negative prompts; `mask_choice: smallest` keeps SAM's smallest candidate mask instead of its top-scored one.
# MAGIC `pseudo_box` adds a square box around each point, sized from the distance to the nearest other point.
# MAGIC Box prompts always run as the baseline. First result (2026-11-13, no variants): oracle-point PQ 0.51,
# MAGIC oracle-box 0.82, pred-point 0.50–0.52, tiny and large within 0.01.
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

RUN = dbx.remote("runs", "nunext-sft-nsft-box-20261113-085309")
MODES = ["pred-box", "pred-point"]
SAM_SIZES = ["tiny"]
VARIANTS = {"neg4-smallest": {"negatives": 4, "mask_choice": "smallest"}}   # 只给 pred-point 用
BOX_VARIANTS = {"baseline": {}, "neg4": {"negatives": 4}}                    # 给 pred-box 用
LIMIT = None

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
if any(m.startswith("pred-") for m in MODES) and not predictions.is_file():
    dbx.volume().download(RUN.rstrip("/") + "/eval-final/predictions.jsonl", predictions)
print("predictions:", predictions)

# COMMAND ----------

results = {}
tag = RUN.rstrip("/").split("/")[-1].replace("nunext-sft-", "").replace("nunext-", "")
for size in SAM_SIZES:
    checkpoint, config = sams[size]
    for mode in MODES:
        for variant, options in (BOX_VARIANTS if mode.endswith("-box") else VARIANTS).items():
            name = f"nunext-pq-{size}-{mode}-{variant}" + (f"-{tag}" if mode.startswith("pred-") else "")
            local, remote = dbx.new_run_dirs(name)
            launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []
            argv = [*launcher, "-m", "nucleus_rl.nunext.pq_eval", "--mode", mode, "--output", local,
                    "--sam-checkpoint", checkpoint, "--sam-config", config,
                    *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--manifest", m)], *nx.flags(options)]
            if mode.startswith("pred-"):
                argv += ["--predictions", str(predictions)]
            if LIMIT:
                argv += ["--limit", str(LIMIT)]
            dbx.run_cli(argv, local, remote)
            results[(size, mode, variant)] = json.loads((local / "summary.json").read_text())

# COMMAND ----------

print(f"{'SAM':6} {'prompts':13} {'variant':14} {'PQ':>6} {'DQ':>6} {'SQ':>6} {'mean img PQ':>11} {'TP':>7} {'FP':>7} {'FN':>7}")
for (size, mode, variant), s in results.items():
    print(f"{size:6} {mode:13} {variant:14} {s['pq']:6.3f} {s['dq']:6.3f} {s['sq']:6.3f} {s['mean_image_pq']:11.3f} "
          f"{s['tp']:7d} {s['fp']:7d} {s['fn']:7d}")

# COMMAND ----------

import json, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nucleus_rl.nunext.pq_eval import SAMSegmenter, oracle_boxes
from nucleus_rl.rewards import instance_metrics, masks_to_instances

EXAMPLE_IDS = ["fold3_0035", "fold3_0012", "fold3_0103"]   # 6 / 23 / 72 nucleus
rows = {r["id"]: r for m in nx.manifests(nx.VALIDATION_FOLDS) for r in map(json.loads, open(m))}
preds = {r["id"]: r for r in map(json.loads, open(predictions))}
segment = SAMSegmenter(*sams["tiny"], "cuda", 64, "score")

def outline(ax, labels, color):
    edge = np.zeros(labels.shape, bool)
    for axis in (0, 1):
        for shift in (1, -1):
            edge |= labels != np.roll(labels, shift, axis)
    edge &= labels > 0
    overlay = np.zeros((*labels.shape, 4)); overlay[edge] = (*color, 1)
    ax.imshow(overlay)

fig, axes = plt.subplots(len(EXAMPLE_IDS), 4, figsize=(16, 4.1 * len(EXAMPLE_IDS)))
for row_axes, image_id in zip(np.atleast_2d(axes), EXAMPLE_IDS):
    row, pred = rows[image_id], preds[image_id]
    image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    gt = np.asarray(Image.open(row["mask_path"])).astype(np.int64)
    points, boxes = [tuple(p) for p in pred["pred"]], [tuple(b) for b in pred["pred_boxes"]]
    if points:
        masks, scores = segment(image, points, boxes, None)
        instances = masks_to_instances(np.asarray(masks, np.float32), scores=np.asarray(scores))
    else:
        instances = np.zeros_like(gt, dtype=np.int32)
    metrics = instance_metrics(instances, gt)
    print(f"===== {image_id}: GT {len(row['centroids'])} , Output {len(points)} "
          f"（{pred['tokens']} token，valid={pred['valid']}）")
    print(f"  Detection TP/FP/FN = {pred['tp']}/{pred['fp']}/{pred['fn']}；"
          f"Segmentation PQ = {metrics['pq']:.3f}  (TP/FP/FN = {metrics['tp']}/{metrics['fp']}/{metrics['fn']})")
    for (x, y), (x0, y0, x1, y1) in list(zip(points, boxes))[:5]:
        print(f"  Center ({x:5.1f},{y:5.1f})  box [{x0:5.1f},{y0:5.1f},{x1:5.1f},{y1:5.1f}] w x h {x1-x0:4.1f}×{y1-y0:4.1f}")
    a0, a1, a2, a3 = row_axes
    a0.imshow(image); a0.set_title(f"{image_id} input")
    a1.imshow(image)
    for x0, y0, x1, y1 in oracle_boxes(gt, len(row["centroids"])):
        a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=0.8))
    for (x0, y0, x1, y1), (x, y) in zip(boxes, points):
        a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1))
        a1.plot(x, y, "o", ms=2, c="yellow")
    a1.set_title("model: green box / yellow centre; GT box: red")
    a2.imshow(image); outline(a2, instances, (0, 1, 0)); a2.set_title(f"SAM masks from predicted boxes, PQ {metrics['pq']:.2f}")
    a3.imshow(image); outline(a3, gt, (1, 0, 0)); a3.set_title("ground-truth instances")
    for a in row_axes: a.axis("off")
plt.tight_layout(); plt.show()


# COMMAND ----------

import json, numpy as np
from scipy.optimize import linear_sum_assignment
rows = {r["id"]: r for m in nx.manifests(nx.VALIDATION_FOLDS) for r in map(json.loads, open(m))}
tot = {"paper": [0, 0, 0], "capped": [0, 0, 0]}
for p in map(json.loads, open(predictions)):
    pred = np.asarray(p["pred"], float).reshape(-1, 2)
    gt = np.asarray(rows[p["id"]]["centroids"], float).reshape(-1, 2)
    for name in tot:
        tp = 0
        if len(pred) and len(gt):
            d = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
            cost = np.where(d <= 12, d, 1e6) if name == "capped" else d
            i, j = linear_sum_assignment(cost)
            tp = int((d[i, j] <= 12).sum())
        tot[name][0] += tp; tot[name][1] += len(pred) - tp; tot[name][2] += len(gt) - tp
for name, (tp, fp, fn) in tot.items():
    print(f"{name:7} TP {tp} FP {fp} FN {fn}  F1 {2*tp/(2*tp+fp+fn):.4f}")

# COMMAND ----------

