# Databricks notebook source
# MAGIC %md
# MAGIC # 12 · NuNext reproduction: GRPO
# MAGIC
# MAGIC Serverless GPU **1×H100 or 8×H100**. Starts from a finished NuNext SFT run (notebook 11) and trains with the
# MAGIC distribution-matching F1 reward (`nucleus_rl.nunext.grpo`). Arms:
# MAGIC - `grpo`: plain GRPO. Paper: NSFT+GRPO 0.830; SASS+CoVT+GRPO 0.835.
# MAGIC - `grpo-lvgf`: + low-variance group filtering. Paper: 0.837 (with SASS+CoVT).
# MAGIC - `grpo-lvgf-fgas`: + fine-grained advantage shaping. Paper: 0.840 (with SASS+CoVT).
# MAGIC
# MAGIC The paper's arms all include CoVT, which this reproduction does not implement, so compare the GRPO gain over
# MAGIC its own SFT start rather than the absolute numbers. Watch `filtered_groups` in the log to tune `delta`.
# MAGIC
# MAGIC Box-model arms (`box-*` in `nunext_common.GRPO_ARMS`) start from `nsft-box`. Rounds 2–3 compare rewards: paper vs
# MAGIC capped matching, F1 vs recall-weighted F2, a box-IoU term, and PQ from frozen SAM2.1-tiny (`pq_weight`). The `pq`
# MAGIC stage scores every finished run with SAM (predicted box + centre + 4 negative points). Check the canary's
# MAGIC `minutes` per step: the PQ reward adds SAM inference for every rollout.
# MAGIC
# MAGIC Each GRPO run uses one GPU. On 8×H100 the arm × seed runs in `ARMS` and `SEEDS` run side by side, one per GPU
# MAGIC (more than 8 run in waves); their output is interleaved here and each keeps its own log on the Volume.

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

SFT_ARM = "nsft"                 # Key in runs/nunext-sft-latest.json written by notebook 11
# Round 3 (box model): PQ through frozen SAM as the reward.
# Round 2 (box model, 2026-11-20) used ARMS ["box-f1-paper", "box-f1-capped", "box-f2-capped", "box-f2-capped-iou"]:
# capped F1 up to +0.009 (F2, about what end-bias -2 gives without training), PQ unchanged (0.644-0.648 vs 0.647).
# Round 1 (point model) used SFT_RUN ...nunext-sft-nsft-20261112-060031, ARMS ["grpo", "grpo-lvgf",
# "grpo-lvgf-fgas"] and SETTINGS {}.
SFT_RUN = "/Volumes/aistudio_dev_catalog/aios_segrlvr_dev/processed/segrlvr/runs/nunext-sft-nsft-box-20261113-085309"
STAGES = ["canary", "main", "pq"]   # canary: 5 steps on the first arm; pq: SAM PQ of every finished main run
ARMS = ["box-pq", "box-pq-f2", "box-pq-endbias"]
SEEDS = [42, 43]                    # 3 arms x 2 seeds = 6 runs, one per GPU on 8xH100
# Overrides of nx.GRPO_SETTINGS ({} for a point model). The SAM paths are only used by arms with pq_weight.
SETTINGS = {**nx.GRPO_BOX_SETTINGS, "sam_checkpoint": str(dbx.ensure_sam()),
            "sam_config": "configs/sam2.1/sam2.1_hiera_t.yaml"}

import torch
GPUS = max(torch.cuda.device_count(), 1)
print("visible GPUs:", GPUS)

nx.ensure_dataset()
sft_run = SFT_RUN or dbx.read_json_remote(dbx.remote("runs", "nunext-sft-latest.json"))[SFT_ARM]
init = nx.ensure_run_model(sft_run)
print("init model:", sft_run)


def argv(arm, output, canary=False, seed=None):
    settings = {**nx.GRPO_SETTINGS, **SETTINGS, **nx.GRPO_ARMS[arm], **(nx.GRPO_CANARY if canary else {})}
    if seed is not None:
        settings["seed"] = seed
    return ["-m", "nucleus_rl.nunext.grpo", "--init-model", init, "--output", output,
            *[a for m in nx.manifests(nx.TRAIN_FOLDS) for a in ("--train-manifest", m)],
            *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--eval-manifest", m)],
            *nx.flags(settings)]


def show(local):
    summary = json.loads((local / "summary.json").read_text())
    keys = ("micro_f1", "capped_micro_f1", "capped_micro_precision", "capped_micro_recall", "invalid", "images")
    for phase in ("before", "after"):
        if phase in summary:
            print(f"{phase:6}", {k: summary[phase].get(k) for k in keys})
    last = summary["steps"][-10:]
    if last:
        print("last steps: reward", round(sum(s["reward"] for s in last) / len(last), 4),
              "filtered groups", sum(s["filtered_groups"] for s in last), "invalid", sum(s["invalid"] for s in last))
    return summary

# COMMAND ----------

if "canary" in STAGES:
    local, remote = dbx.new_run_dirs("nunext-grpo-canary")
    dbx.run_cli(argv(ARMS[0], local, canary=True), local, remote)
    summary = show(local)
    assert summary["completed_steps"] == nx.GRPO_CANARY["max_steps"], "canary did not finish its steps"
    print("canary passed:", remote)

# COMMAND ----------

if "main" in STAGES:
    tag = sft_run.rstrip("/").split("/")[-1].replace("nunext-sft-", "")
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            local, remote = dbx.new_run_dirs(f"nunext-{arm}-s{seed}-from-{tag}")
            jobs.append((argv(arm, local, seed=seed), local, remote))
    print(f"{len(jobs)} GRPO runs on {GPUS} GPU(s)")
    if GPUS > 1 and len(jobs) > 1:
        nx.run_parallel(jobs, GPUS)
    else:
        for job_argv, local, remote in jobs:
            dbx.run_cli(job_argv, local, remote)
    for _, local, remote in jobs:
        print("==", remote)
        show(local)

# COMMAND ----------

# MAGIC %md PQ of every finished main run: predicted box + centre + 4 negative points, frozen SAM2.1-tiny, all of fold 3
# MAGIC (box model before GRPO: 0.647). About 4 minutes per run on 8 GPUs.

# COMMAND ----------

if "pq" in STAGES and SETTINGS.get("box_tokens"):
    pq = {}
    launcher = ["-m", "nucleus_rl.nunext.launch", "--nproc", str(GPUS), "--"] if GPUS > 1 else []
    for _, run_local, run_remote in jobs:
        name = run_remote.rstrip("/").split("/")[-1]
        local, remote = dbx.new_run_dirs(f"nunext-pq-tiny-pred-box-neg4-{name.replace('nunext-', '')}")
        pq_argv = [*launcher, "-m", "nucleus_rl.nunext.pq_eval", "--mode", "pred-box", "--output", local,
                   "--sam-checkpoint", str(dbx.ensure_sam()), "--sam-config", "configs/sam2.1/sam2.1_hiera_t.yaml",
                   *[a for m in nx.manifests(nx.VALIDATION_FOLDS) for a in ("--manifest", m)],
                   "--predictions", str(run_local / "eval-final" / "predictions.jsonl"), "--negatives", "4"]
        dbx.run_cli(pq_argv, local, remote)
        pq[name] = (json.loads((run_local / "summary.json").read_text())["after"],
                    json.loads((local / "summary.json").read_text()))
    print(f"{'run':70} {'paper F1':>8} {'capped F1':>9} {'P':>6} {'R':>6} {'PQ':>6} {'DQ':>6} {'SQ':>6}")
    for name, (det, seg) in pq.items():
        print(f"{name:70} {det['micro_f1']:8.4f} {det['capped_micro_f1']:9.4f} {det['capped_micro_precision']:6.3f} "
              f"{det['capped_micro_recall']:6.3f} {seg['pq']:6.3f} {seg['dq']:6.3f} {seg['sq']:6.3f}")
    print("box model before GRPO: paper F1 0.8151, capped F1 0.8501 (P 0.884 R 0.819), PQ 0.647")

