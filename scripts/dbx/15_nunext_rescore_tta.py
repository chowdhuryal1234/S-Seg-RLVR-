# Databricks notebook source
# MAGIC %md
# MAGIC # 15 · NuNext: capped matching and D4 test-time augmentation
# MAGIC
# MAGIC **Part A (no GPU, minutes).** Re-scores the saved `eval-final/predictions.jsonl` of finished runs with both matching
# MAGIC protocols. `paper` solves the assignment on raw distances and then drops pairs beyond 12 px (the F1 reported so far);
# MAGIC on dense images that loses correct points a few pixels off. `capped` makes pairs beyond 12 px unassignable first.
# MAGIC
# MAGIC **Part B (8×H100, about 30 minutes).** Decodes every fold 3 image in all 8 D4 views with the box model, maps the
# MAGIC nuclei back, merges them, and counts in how many views each nucleus was found (`votes`). Prints F1 for every vote
# MAGIC threshold 1–8, both protocols. Goal: recall above the single-view 0.785 without losing much precision.
# MAGIC
# MAGIC **Part C (8×H100, a few minutes per threshold).** PQ with frozen SAM2.1-tiny for the TTA predictions at the
# MAGIC thresholds in `PQ_VOTES` (predicted box + centre, with and without 4 negative points). Single-view baseline: 0.644 / 0.647.
# MAGIC
# MAGIC The vote threshold is picked on fold 3, the same images it is reported on; treat the gain as optimistic until it is
# MAGIC confirmed on another fold.

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
RESCORE_RUNS = ["nunext-sft-nsft-20261112-060031", BOX_RUN,
                "nunext-grpo-s42-from-nsft-20261112-060031-20261112-162200",
                "nunext-grpo-s43-from-nsft-20261112-060031-20261112-162200",
                "nunext-grpo-lvgf-s42-from-nsft-20261112-060031-20261112-162201",
                "nunext-grpo-lvgf-s43-from-nsft-20261112-060031-20261112-162201",
                "nunext-grpo-lvgf-fgas-s42-from-nsft-20261112-060031-20261112-162201",
                "nunext-grpo-lvgf-fgas-s43-from-nsft-20261112-060031-20261112-162201"]
MERGE_RADIUS = 8.0          # px; 0.48% of fold 3 nuclei have a neighbour closer than 8 px
BATCH_SIZE = 32             # generated sequences per batch = 4 images x 8 views
MAX_NEW_TOKENS = 1536       # box model
PQ_VOTES = "best"           # "best" = the 3 thresholds around the best capped F1, or a list such as [3, 4, 5]
LIMIT = None                # e.g. 256 for a quick check of parts B and C

GPUS = max(torch.cuda.device_count(), 1)
print("visible GPUs:", GPUS)
nx.ensure_dataset()
fold3 = nx.manifests(nx.VALIDATION_FOLDS)
manifest_args = [a for m in fold3 for a in ("--manifest", m)]

# COMMAND ----------

# MAGIC %md ## A · Re-score finished runs with both matching protocols

# COMMAND ----------

from nucleus_rl.nunext.data import read_manifests
from nucleus_rl.nunext.rescore import rescore

rows = read_manifests(fold3)
rescored = {}
for name in RESCORE_RUNS:
    local = dbx.local_root() / "rescore" / name / "predictions.jsonl"
    if not local.is_file():
        try:
            dbx.volume().download(dbx.remote("runs", name, "eval-final", "predictions.jsonl"), local)
        except Exception as error:   # a run that was renamed or never finished
            print(f"skip {name}: {error}")
            continue
    records = [json.loads(line) for line in local.read_text().splitlines() if line.strip()]
    rescored[name] = rescore(records, rows)

print(f"{'run':64} {'paper F1':>8} {'capped F1':>9} {'P paper':>8} {'P capped':>8} {'R paper':>8} {'R capped':>8}")
for name, r in rescored.items():
    p, c = r["paper"], r["capped"]
    print(f"{name:64} {p['micro_f1']:8.4f} {c['micro_f1']:9.4f} {p['micro_precision']:8.4f} {c['micro_precision']:8.4f} "
          f"{p['micro_recall']:8.4f} {c['micro_recall']:8.4f}")
local, remote = dbx.new_run_dirs("nunext-rescore")
local.mkdir(parents=True, exist_ok=True)
(local / "rescore.json").write_text(json.dumps(rescored, indent=2) + "\n")
dbx.volume().upload_tree(local, remote)
print("saved", remote)

# COMMAND ----------

# MAGIC %md ## B · D4 test-time augmentation on the box model

# COMMAND ----------

model = nx.ensure_run_model(dbx.remote("runs", BOX_RUN))
tta_local, tta_remote = dbx.new_run_dirs(f"nunext-eval-tta-{BOX_RUN.replace('nunext-sft-', '')}")
launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []
argv = [*launcher, "-m", "nucleus_rl.nunext.evaluate", "--model", model, "--output", tta_local, *manifest_args,
        "--bins", str(nx.SFT_SETTINGS["bins"]), "--image-size", str(nx.SFT_SETTINGS["image_size"]),
        "--radius", str(nx.SFT_SETTINGS["radius"]), "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--batch-size", str(BATCH_SIZE), "--box-tokens", "--tta", "d4", "--merge-radius", str(MERGE_RADIUS)]
if LIMIT:
    argv += ["--limit", str(LIMIT)]
dbx.run_cli(argv, tta_local, tta_remote)
tta_summary = json.loads((tta_local / "summary.json").read_text())

single = rescored.get(BOX_RUN)
if single:
    print(f"single view      paper F1 {single['paper']['micro_f1']:.4f} (P {single['paper']['micro_precision']:.3f} "
          f"R {single['paper']['micro_recall']:.3f})   capped F1 {single['capped']['micro_f1']:.4f} "
          f"(P {single['capped']['micro_precision']:.3f} R {single['capped']['micro_recall']:.3f})")
for votes, s in tta_summary["tta_vote_sweep"].items():
    p, c = s["paper"], s["capped"]
    print(f"TTA votes >= {votes}  paper F1 {p['micro_f1']:.4f} (P {p['micro_precision']:.3f} R {p['micro_recall']:.3f})   "
          f"capped F1 {c['micro_f1']:.4f} (P {c['micro_precision']:.3f} R {c['micro_recall']:.3f})")
print("images with no valid view:", tta_summary["invalid"], "· seconds:", tta_summary["seconds"])

# COMMAND ----------

# MAGIC %md ## C · PQ for the TTA predictions

# COMMAND ----------

sweep = tta_summary["tta_vote_sweep"]
if PQ_VOTES == "best":
    best = max(sweep, key=lambda k: sweep[k]["capped"]["micro_f1"])
    votes_to_run = [v for v in (int(best) - 1, int(best), int(best) + 1) if 1 <= v <= 8]
else:
    votes_to_run = list(PQ_VOTES)
print("PQ at vote thresholds:", votes_to_run)

checkpoint, config = str(dbx.ensure_sam()), "configs/sam2.1/sam2.1_hiera_t.yaml"
pq_results = {}
for votes in votes_to_run:
    for variant, options in {"baseline": {}, "neg4": {"negatives": 4}}.items():
        name = f"nunext-pq-tiny-pred-box-tta{votes}-{variant}-{BOX_RUN.replace('nunext-sft-', '')}"
        local, remote = dbx.new_run_dirs(name)
        argv = [*launcher, "-m", "nucleus_rl.nunext.pq_eval", "--mode", "pred-box", "--output", local,
                "--sam-checkpoint", checkpoint, "--sam-config", config, *manifest_args,
                "--predictions", str(tta_local / "predictions.jsonl"), "--min-votes", str(votes), *nx.flags(options)]
        if LIMIT:
            argv += ["--limit", str(LIMIT)]
        dbx.run_cli(argv, local, remote)
        pq_results[(votes, variant)] = json.loads((local / "summary.json").read_text())

print(f"{'votes':>5} {'variant':9} {'PQ':>6} {'DQ':>6} {'SQ':>6} {'TP':>7} {'FP':>7} {'FN':>7}")
print(f"{'1 view':>5} {'baseline':9} {0.644:6.3f} {0.802:6.3f} {0.803:6.3f}   (single-view reference)")
for (votes, variant), s in pq_results.items():
    print(f"{votes:>5} {variant:9} {s['pq']:6.3f} {s['dq']:6.3f} {s['sq']:6.3f} {s['tp']:7d} {s['fp']:7d} {s['fn']:7d}")