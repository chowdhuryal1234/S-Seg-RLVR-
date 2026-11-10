"""Shared settings and inputs for the NuNext reproduction notebooks (10-12).

Builds on dbx_common (Volume access through the Files API, local disk, run_cli). The prepared
PanNuke dataset is stored on the Volume as ~20 small zips plus index.json under
<artifacts>/data/pannuke_nunext_v1/: uploading ~8k PNGs one by one is slow, and one large
PUT can be cut off ("RemoteDisconnected"). Downloads and uploads retry with backoff.
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
# Hugging Face copy of the official folds (parquet). Checked against the Warwick fold 3: identical images,
# nucleus counts, centroids and classes after conversion.
HF_DATASET = "RationAI/PanNuke"
HF_REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
HF_FILE = "data/fold{}-00000-of-00001.parquet"
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


PART_IMAGES = 400            # Images per dataset part zip (~50-80 MB), small enough for one Files API PUT


def retry(action, what: str, attempts: int = 5):
    """Run action(); on a dropped connection or other transient error, wait and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:  # Connection resets surface as several exception types.
            if attempt == attempts:
                raise
            wait = 10 * 2 ** (attempt - 1)
            print(f"[{what}: {type(error).__name__}: {error}; retry {attempt}/{attempts - 1} in {wait}s]", flush=True)
            time.sleep(wait)


def http_download(url: str, destination: Path, chunk: int = 1 << 20) -> None:
    """Download with a browser-like User-Agent, resuming a partial file after a dropped connection."""
    partial = destination.with_name(destination.name + ".partial")

    def attempt():
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Accept": "*/*"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            if offset and response.status != 206:   # Server ignored Range: start over.
                offset = 0
            with partial.open("ab" if offset else "wb") as output:
                shutil.copyfileobj(response, output, chunk)

    retry(attempt, f"download {url}", attempts=8)
    partial.replace(destination)


def ensure_pannuke_zip(fold: int, store_raw: bool = False) -> Path:
    """Official fold zip on local disk: from <artifacts>/raw/pannuke/ if present, else downloaded.

    The 700 MB zips are copied to the Volume only with store_raw=True (a single large PUT can be cut off).
    """
    local = dbx.local_root() / "source" / "pannuke" / f"fold_{fold}.zip"
    stored = dbx.remote("raw", "pannuke", f"fold_{fold}.zip")
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    if dbx.volume().is_file(stored):
        retry(lambda: dbx.volume().download(stored, local), f"Volume download {stored}")
    else:
        started = time.time()
        http_download(PANNUKE_URL.format(fold), local)
        print(f"downloaded fold {fold} ({local.stat().st_size / 1e6:.0f} MB) in {time.time() - started:.0f}s", flush=True)
        if store_raw:
            try:
                retry(lambda: dbx.volume().upload(local, stored), f"Volume upload {stored}", attempts=3)
            except Exception as error:
                print(f"[raw zip not stored on the Volume: {type(error).__name__}: {error}]", flush=True)
    return local


def _upload_parts(out: Path) -> dict:
    """Store the dataset as several small zips plus index.json under <artifacts>/data/<name>/."""
    remote_dir = dbx.remote("data", DATASET_NAME)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    metadata = [p for p in files if p.parent == out]                      # manifests, report
    images = [p for p in files if p.parent != out]
    groups = [metadata] + [images[i:i + PART_IMAGES] for i in range(0, len(images), PART_IMAGES)]
    parts = []
    staging = out.parent / (DATASET_NAME + "-parts")
    staging.mkdir(exist_ok=True)
    for index, group in enumerate(groups):
        name = f"part-{index:03d}.zip"
        path = staging / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:  # PNGs are already compressed
            for file in group:
                archive.write(file, file.relative_to(out.parent).as_posix())
        retry(lambda: dbx.volume().upload(path, f"{remote_dir}/{name}"), f"Volume upload {name}")
        parts.append({"name": name, "files": len(group), "bytes": path.stat().st_size})
        print(f"uploaded {name}: {len(group)} files, {path.stat().st_size / 1e6:.0f} MB", flush=True)
    index = {"dataset": DATASET_NAME, "parts": parts}
    # index.json is written last, so a half-uploaded dataset is never treated as complete.
    retry(lambda: dbx.volume().write_text(f"{remote_dir}/index.json", json.dumps(index, indent=2)), "index.json")
    return index


def ensure_pannuke_parquet(fold: int) -> Path:
    """One fold's parquet from the pinned Hugging Face dataset revision, on local disk."""
    from huggingface_hub import hf_hub_download

    cache = dbx.local_root() / "source" / "pannuke_hf"
    return Path(retry(lambda: hf_hub_download(HF_DATASET, HF_FILE.format(fold), repo_type="dataset",
                                              revision=HF_REVISION, cache_dir=str(cache)),
                      f"Hugging Face download fold {fold}"))


def prepare_dataset(rebuild: bool = False, store_raw: bool = False, source: str = "huggingface") -> dict:
    """Convert the three folds with scripts/prepare_pannuke.py and store small zips on the Volume.

    source="huggingface" reads RationAI/PanNuke parquet files; source="warwick" uses the official zips.
    """
    if source not in ("huggingface", "warwick"):
        raise ValueError("source must be 'huggingface' or 'warwick'")
    stored = dbx.remote("data", DATASET_NAME, "index.json")
    if dbx.volume().is_file(stored) and not rebuild:
        return ensure_dataset()
    from scripts.prepare_pannuke import convert_fold, convert_parquet

    out = dataset_dir()
    if out.exists():
        shutil.rmtree(out)
    report = []
    for fold in (*TRAIN_FOLDS, *VALIDATION_FOLDS):
        if source == "huggingface":
            report.append(convert_parquet(ensure_pannuke_parquet(fold), fold, out))
        else:
            extracted = dbx.local_root() / "source" / "pannuke" / f"fold_{fold}"
            if not (extracted / ".extracted").is_file():
                with zipfile.ZipFile(ensure_pannuke_zip(fold, store_raw)) as archive:
                    archive.extractall(extracted)
                (extracted / ".extracted").write_text("ok\n")
            report.append(convert_fold(extracted, fold, out))
        print(json.dumps(report[-1]), flush=True)
    (out / "report.json").write_text(json.dumps({"folds": report}, indent=2) + "\n")
    index = _upload_parts(out)
    if not _localize_and_check(out):
        (out / COMPLETE_MARKER).write_text("ok\n")
    return {"folds": report, "volume": dbx.remote("data", DATASET_NAME), "parts": len(index["parts"])}


COMPLETE_MARKER = ".nunext-complete"


def _localize_and_check(out: Path) -> int:
    """Point manifest image paths at this machine's copy; return how many images are missing."""
    missing = 0
    for manifest in out.glob("fold*.jsonl"):
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        for row in rows:
            row["image_path"] = str(out / "images" / Path(row["image_path"]).name)
            missing += not Path(row["image_path"]).is_file()
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return missing


def _download_parts(out: Path) -> None:
    remote_dir = dbx.remote("data", DATASET_NAME)
    if not dbx.volume().is_file(f"{remote_dir}/index.json"):
        raise FileNotFoundError(f"{remote_dir}/index.json not found; run 10_nunext_data first")
    index = json.loads(dbx.volume().read_text(f"{remote_dir}/index.json"))
    staging = dbx.local_root() / "data" / (DATASET_NAME + "-download")
    staging.mkdir(parents=True, exist_ok=True)
    for part in index["parts"]:
        local = staging / part["name"]
        retry(lambda: dbx.volume().download(f"{remote_dir}/{part['name']}", local), f"Volume download {part['name']}")
        with zipfile.ZipFile(local) as handle:
            handle.extractall(out.parent)
        local.unlink()   # keep local disk use to one copy
    print(f"extracted {len(index['parts'])} dataset parts into {out}", flush=True)


def ensure_dataset() -> dict:
    """Complete local dataset with every manifest image present; re-downloads a partial copy.

    Completeness is a marker written after all parts are extracted and every image is checked:
    report.json alone is not enough, because it arrives in the first part.
    """
    out = dataset_dir()
    for attempt in range(2):
        if not (out / COMPLETE_MARKER).is_file():
            if out.exists():
                shutil.rmtree(out)
            _download_parts(out)
        missing = _localize_and_check(out)
        if not missing:
            (out / COMPLETE_MARKER).write_text("ok\n")
            return json.loads((out / "report.json").read_text())
        print(f"[{missing} manifest images missing in {out}; downloading again]", flush=True)
        (out / COMPLETE_MARKER).unlink(missing_ok=True)
    free = shutil.disk_usage(out.parent).free / 1e9
    raise RuntimeError(f"{missing} images still missing after a fresh download ({free:.1f} GB free on {out.parent})")


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
