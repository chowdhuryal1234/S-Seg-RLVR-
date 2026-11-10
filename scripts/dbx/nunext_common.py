"""Shared settings and inputs for the NuNext reproduction notebooks (10-12).

Builds on dbx_common (Volume access through the Files API, local disk, run_cli). The prepared
PanNuke dataset is stored on the Volume as one zip, because uploading ~8k PNGs one by one
through the Files API is slow.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

import dbx_common as dbx

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"   # Hugging Face main on 2025-04-06
PANNUKE_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{}.zip"
DATASET_NAME = "pannuke_nunext_v1"
TRAIN_FOLDS = (1, 2)          # Paper: folds 1 and 2 train, fold 3 internal validation
VALIDATION_FOLDS = (3,)

# Defaults for the reproduction; see nucleus_rl/nunext/README.md for which are assumptions.
SFT_SETTINGS = {"bins": 256, "image_size": 448, "augmentation": "d4", "epochs": 3, "gradient_accumulation": 16,
                "learning_rate": 1e-5, "warmup_ratio": 0.03, "radius": 12, "eval_every": 250,
                "eval_limit": 512, "eval_batch_size": 16, "max_new_tokens": 1024, "seed": 42}
SFT_ARMS = {"nsft": {"sigma": 0.0}, "sass": {"sigma": 1.0}}
SFT_CANARY = {"max_steps": 20, "gradient_accumulation": 2, "train_limit": 64, "final_eval_limit": 32,
              "eval_every": 0, "skip_before_eval": True}

GRPO_SETTINGS = {"bins": 256, "image_size": 448, "augmentation": "d4", "radius": 12, "group_size": 8,
                 "prompts_per_step": 4, "rollout_batch": 4, "temperature": 1.0, "learning_rate": 1e-6,
                 "warmup_steps": 10, "beta": 0.04, "max_steps": 500, "eval_every": 100, "eval_limit": 512,
                 "eval_batch_size": 16, "max_new_tokens": 1024, "seed": 42}
GRPO_ARMS = {"grpo": {"delta": 0.0, "fgas_decay": 1.0},
             "grpo-lvgf": {"delta": 0.02, "fgas_decay": 1.0},
             "grpo-lvgf-fgas": {"delta": 0.02, "fgas_decay": 0.5}}
GRPO_CANARY = {"max_steps": 5, "prompts_per_step": 2, "train_limit": 16, "final_eval_limit": 16,
               "eval_every": 0, "skip_before_eval": True}


def flags(settings: dict) -> list[str]:
    items = []
    for key, value in settings.items():
        option = "--" + key.replace("_", "-")
        if value is True:
            items.append(option)
        elif value is not False and value is not None:
            items += [option, str(value)]
    return items


def dataset_dir() -> Path:
    return dbx.local_root() / "data" / DATASET_NAME


def manifests(folds) -> list[str]:
    return [str(dataset_dir() / f"fold{k}.jsonl") for k in folds]


def ensure_pannuke_zip(fold: int) -> Path:
    """Official fold zip on local disk: from <artifacts>/raw/pannuke/ if present, else downloaded (and stored)."""
    local = dbx.local_root() / "source" / "pannuke" / f"fold_{fold}.zip"
    stored = dbx.remote("raw", "pannuke", f"fold_{fold}.zip")
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    if dbx.volume().is_file(stored):
        dbx.volume().download(stored, local)
    else:
        started = time.time()
        urllib.request.urlretrieve(PANNUKE_URL.format(fold), local)
        print(f"downloaded fold {fold} ({local.stat().st_size / 1e6:.0f} MB) in {time.time() - started:.0f}s")
        dbx.volume().upload(local, stored)
    return local


def prepare_dataset(rebuild: bool = False) -> dict:
    """Extract the three folds and convert them with scripts/prepare_pannuke.py; store one zip on the Volume."""
    stored = dbx.remote("data", DATASET_NAME + ".zip")
    if dbx.volume().is_file(stored) and not rebuild:
        return ensure_dataset()
    from scripts.prepare_pannuke import convert_fold

    out = dataset_dir()
    if out.exists():
        shutil.rmtree(out)
    report = []
    for fold in (*TRAIN_FOLDS, *VALIDATION_FOLDS):
        extracted = dbx.local_root() / "source" / "pannuke" / f"fold_{fold}"
        if not extracted.is_dir():
            with zipfile.ZipFile(ensure_pannuke_zip(fold)) as archive:
                archive.extractall(extracted)
        report.append(convert_fold(extracted, fold, out))
        print(json.dumps(report[-1]), flush=True)
    (out / "report.json").write_text(json.dumps({"folds": report}, indent=2) + "\n")
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    dbx.volume().upload(Path(archive), stored)
    return {"folds": report, "volume": stored}


def ensure_dataset() -> dict:
    """Local dataset; manifests are rewritten so image paths point at this machine's copy."""
    out = dataset_dir()
    if not (out / "report.json").is_file():
        archive = dbx.local_root() / "data" / (DATASET_NAME + ".zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        stored = dbx.remote("data", DATASET_NAME + ".zip")
        if not dbx.volume().is_file(stored):
            raise FileNotFoundError(f"{stored} not found; run 10_nunext_data first")
        dbx.volume().download(stored, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(archive.parent)
    for manifest in out.glob("fold*.jsonl"):
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        for row in rows:
            row["image_path"] = str(out / "images" / Path(row["image_path"]).name)
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return json.loads((out / "report.json").read_text())


def ensure_base_model() -> str:
    """Local path of the pinned Qwen2.5-VL-3B snapshot."""
    info = dbx.ensure_model(MODEL, REVISION)
    return info["path"]


def ensure_run_model(remote_run: str) -> Path:
    """Download <run>/model from the Volume (an SFT or GRPO output) to local disk."""
    name = remote_run.rstrip("/").split("/")[-1]
    local = dbx.local_root() / "init" / name / "model"
    if not (local / "config.json").is_file():
        dbx.volume().download_tree(remote_run.rstrip("/") + "/model", local)
    return local
