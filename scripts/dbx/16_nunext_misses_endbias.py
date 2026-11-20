# Databricks notebook source
# MAGIC %md
# MAGIC # 16 · NuNext: where are the misses, and does stopping later help?
# MAGIC
# MAGIC D4 TTA showed that 92% of fold 3 nuclei are found in at least one view, yet a single view finds 82% (capped
# MAGIC matching). One explanation is that the model writes the end token too early.
# MAGIC
# MAGIC **Part A (no GPU, minutes).** For the box model's single-view and TTA predictions: recall by raster position and
# MAGIC by y, and how many misses lie after the last predicted point compared with what random misses would give.
# MAGIC Early stopping shows up as recall falling in the last bands and many more tail misses than `if_random`.
# MAGIC
# MAGIC **Part B (8×H100, about 4 minutes per bias).** Re-decodes fold 3 with a negative bias on the end token's logit
# MAGIC (greedy decoding then stops only when the end token wins by more than |bias|). Too strong a bias makes the model
# MAGIC run to the token limit, which shows up as `invalid` (no end token) and a falling F1.
# MAGIC
# MAGIC **Part C (8×H100, a few minutes).** PQ (predicted box + centre + 4 negative points) at the bias with the best
# MAGIC capped F1. Single-view reference: 0.647; TTA (≥ 4 votes): 0.668.

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
import torch

BOX_RUN = "nunext-sft-nsft-box-20261113-085309"
TTA_RUN = "nunext-eval-tta-nsft-box-20261113-085309-20261119-232938"   # from notebook 15
BIASES = [-0.5, -1.0, -2.0, -4.0]   # 0 is the existing single-view result
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 16
RUN_PQ = True
LIMIT = None                         # e.g. 256 for a quick check of parts B and C

GPUS = max(torch.cuda.device_count(), 1)
print("visible GPUs:", GPUS)
nx.ensure_dataset()
fold3 = nx.manifests(nx.VALIDATION_FOLDS)
manifest_args = [a for m in fold3 for a in ("--manifest", m)]
launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []

from nucleus_rl.nunext.data import read_manifests
from nucleus_rl.nunext.rescore import miss_positions, rescore
rows = read_manifests(fold3)


def load_records(run, sub="eval-final"):
    """predictions.jsonl of a training run (in eval-final/) or of an evaluate run (sub=None)."""
    local = dbx.local_root() / "misses" / run / "predictions.jsonl"
    if not local.is_file():
        parts = ("runs", run, sub, "predictions.jsonl") if sub else ("runs", run, "predictions.jsonl")
        dbx.volume().download(dbx.remote(*parts), local)
    return [json.loads(line) for line in local.read_text().splitlines() if line.strip()]


def show_positions(label, records):
    m = miss_positions(records, rows)
    print(f"--- {label}")
    print("  recall by raster position (first -> last eighth):", m["by_rank"])
    print("  recall by y (top -> bottom eighth):               ", m["by_y"])
    t = m["tail"]
    print(f"  misses after the last prediction: {t['after_last_prediction']} (random misses would give "
          f"{t['after_last_if_random']}); misses before it: {t['before_it']}")
    print("  predictions / ground truth by nuclei per image:",
          {k: v["pred_over_gt"] for k, v in sorted(m["count_ratio"].items(), key=lambda kv: float(kv[0].split('-')[0].rstrip('+')))})
    return m

# COMMAND ----------

# MAGIC %md ## A · Where the box model misses nuclei

# COMMAND ----------

positions = {"single view": show_positions("single view", load_records(BOX_RUN)),
             "TTA (>= 4 votes)": show_positions("TTA (>= 4 votes)", load_records(TTA_RUN, sub=None))}

# COMMAND ----------

# MAGIC %md ## B · End-token bias sweep (single view)

# COMMAND ----------

model = nx.ensure_run_model(dbx.remote("runs", BOX_RUN))
bias_results = {}
for bias in BIASES:
    local, remote = dbx.new_run_dirs(f"nunext-eval-endbias{bias:+g}-{BOX_RUN.replace('nunext-sft-', '')}")
    argv = [*launcher, "-m", "nucleus_rl.nunext.evaluate", "--model", model, "--output", local, *manifest_args,
            "--bins", str(nx.SFT_SETTINGS["bins"]), "--image-size", str(nx.SFT_SETTINGS["image_size"]),
            "--radius", str(nx.SFT_SETTINGS["radius"]), "--max-new-tokens", str(MAX_NEW_TOKENS),
            "--batch-size", str(BATCH_SIZE), "--box-tokens", f"--end-bias={bias}"]
    if LIMIT:
        argv += ["--limit", str(LIMIT)]
    dbx.run_cli(argv, local, remote)
    records = [json.loads(line) for line in (local / "predictions.jsonl").read_text().splitlines() if line.strip()]
    bias_results[bias] = {"local": local, "summary": json.loads((local / "summary.json").read_text()),
                          "positions": miss_positions(records, rows)}

reference = rescore(load_records(BOX_RUN), rows)
print(f"{'bias':>5} {'paper F1':>8} {'capped F1':>9} {'P capped':>8} {'R capped':>8} {'invalid':>7} {'tokens':>6} "
      f"{'tail misses':>11} {'last-eighth recall':>18}")
p, c = reference["paper"], reference["capped"]
print(f"{0:>5} {p['micro_f1']:8.4f} {c['micro_f1']:9.4f} {c['micro_precision']:8.4f} {c['micro_recall']:8.4f} "
      f"{'0':>7} {'':>6} {positions['single view']['tail']['after_last_prediction']:>11} "
      f"{positions['single view']['by_rank'][-1]:>18}")
for bias, r in bias_results.items():
    s, m = r["summary"], r["positions"]
    print(f"{bias:>5} {s['micro_f1']:8.4f} {s['capped_micro_f1']:9.4f} {s['capped_micro_precision']:8.4f} "
          f"{s['capped_micro_recall']:8.4f} {s['invalid']:>7} {s['mean_tokens']:>6} "
          f"{m['tail']['after_last_prediction']:>11} {m['by_rank'][-1]:>18}")

# COMMAND ----------

# MAGIC %md ## C · PQ at the best bias

# COMMAND ----------

if RUN_PQ and bias_results:
    best = max(bias_results, key=lambda b: bias_results[b]["summary"]["capped_micro_f1"])
    print("best bias by capped F1:", best)
    local, remote = dbx.new_run_dirs(f"nunext-pq-tiny-pred-box-endbias{best:+g}-neg4-{BOX_RUN.replace('nunext-sft-', '')}")
    argv = [*launcher, "-m", "nucleus_rl.nunext.pq_eval", "--mode", "pred-box", "--output", local,
            "--sam-checkpoint", str(dbx.ensure_sam()), "--sam-config", "configs/sam2.1/sam2.1_hiera_t.yaml",
            *manifest_args, "--predictions", str(bias_results[best]["local"] / "predictions.jsonl"), "--negatives", "4"]
    if LIMIT:
        argv += ["--limit", str(LIMIT)]
    dbx.run_cli(argv, local, remote)
    s = json.loads((local / "summary.json").read_text())
    print(f"end bias {best:+g}: PQ {s['pq']:.3f}  DQ {s['dq']:.3f}  SQ {s['sq']:.3f}  TP {s['tp']}  FP {s['fp']}  FN {s['fn']}")
    print("references: single view 0.647, TTA >= 4 votes 0.668")