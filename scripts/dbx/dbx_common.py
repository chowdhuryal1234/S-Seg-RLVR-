"""Shared helpers for the Databricks Serverless GPU notebooks in this folder.

Notebooks import this module after adding this folder to sys.path. It holds
the workspace paths in one place, keeps the Hugging Face cache on fast local
disk (with an optional Unity Catalog Volume mirror), runs the repository CLIs
in a subprocess, and mirrors run directories to the Volume while they run.
Edit the PATHS block once for your workspace; nothing else is hard-coded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[2]

# ---------- Edit for your workspace ----------
CATALOG_ROOT = Path("/Volumes/aistudio_dev_catalog/aios_segrlvr_dev")
PATHS = {
    # Extracted MoNuSeg 2018 training data (30 official images), as already uploaded.
    "train_dir": CATALOG_ROOT / "raw/monuseg/train",
    # Persistent outputs: data, checkpoints, model cache mirror, runs.
    "artifacts": CATALOG_ROOT / "processed/segrlvr",
}
# ---------------------------------------------

DATASET_NAME = "monuseg_official30_grpo_v1"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
# Manifest digests from the Harvard run. Manifests embed the absolute source
# archive path, so these match only if that path is identical; compare_manifests
# below checks the path-independent content instead.
HARVARD_MANIFEST_SHA256 = {
    "train": "a49d484b4202ddd21f1ce34eda3859641fc38bbbd72b665e27fcf63190300dcc",
    "validation": "1969cd648b83fa0d9f1bfcaa73d248a879e3f3156e2863e46c540cf7cce853f8",
    "validation_benchmark8": "15520402033fafe5560b7b3f14dbc619686cec7b208d35990ee558e3185fd60c",
}


def local_root():
    """Fast ephemeral disk for the model cache and in-progress runs."""
    base = Path("/local_disk0") if Path("/local_disk0").is_dir() and os.access("/local_disk0", os.W_OK) \
        else Path(tempfile.gettempdir())
    path = base / "segrlvr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact(*parts):
    return PATHS["artifacts"].joinpath(*parts)


def dataset_dir():
    return artifact("data", DATASET_NAME)


def sam_checkpoint():
    return artifact("checkpoints", "sam2.1_hiera_tiny.pt")


def sft_config():
    return json.loads((REPO / "configs/sft_experiments.json").read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_environment():
    """Environment variables shared by every notebook; returns the local HF cache."""
    hf_home = local_root() / "huggingface"
    os.environ.update(HF_HOME=str(hf_home), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                      SAM2_BUILD_CUDA="0")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return hf_home


# ---------- Command lines for the repository runners ----------

GRPO_SETTINGS = {"reward": "format=1,pq=1", "group_size": 8, "prompts_per_step": 2, "temperature": 1.0,
                 "beta": 0.04, "learning_rate": 2e-5, "warmup_steps": 10, "lora_rank": 16, "lora_alpha": 32,
                 "max_steps": 600, "eval_every": 150, "save_steps": 50, "augmentation": "d4", "seed": 42}
GRPO_CANARY = {"max_steps": 10, "warmup_steps": 1, "eval_every": 0, "save_steps": 10,
               "train_limit": 4, "eval_limit": 8}


def _flags(settings):
    return [item for key, value in settings.items() for item in ("--" + key.replace("_", "-"), str(value))]


def sft_argv(model_key, stage, augmentation, output, config=None):
    """Same arguments as scripts/run_sft_experiments.command(), without the Slurm deadline."""
    config = config or sft_config()
    model, data = config["models"][model_key], dataset_dir()
    settings = {**config["shared"], **{k: config[stage][k] for k in ("max_steps", "warmup_steps", "eval_every", "save_steps")}}
    if stage == "canary":
        settings.update({k: config["canary"][k] for k in ("train_limit", "eval_limit")})
    return ["-m", "nucleus_rl.sft", "--manifest", str(data / "train.jsonl"),
            "--eval-manifest", str(data / "validation.jsonl"), "--output", str(output),
            "--model", model["model"], "--revision", model["revision"],
            "--sam-checkpoint", str(sam_checkpoint()), "--sam-device", "cpu",
            "--cohort-protocol", str(REPO / "protocols/monuseg_official30.json"),
            "--coordinate-frame", model["coordinate_frame"], "--augmentation", augmentation, *_flags(settings)]


def grpo_argv(model_key, objective, output, init_adapter, settings=None, canary=False, config=None):
    """GRPO (or continued-SFT control) from a finished SFT adapter, with the SFT input contract."""
    config = config or sft_config()
    model, shared, data = config["models"][model_key], config["shared"], dataset_dir()
    settings = {**GRPO_SETTINGS, **(settings or {}), **(GRPO_CANARY if canary else {})}
    argv = ["-m", "nucleus_rl.grpo", "--objective", objective, "--manifest", str(data / "train.jsonl"),
            "--eval-manifest", str(data / "validation.jsonl"), "--output", str(output),
            "--model", model["model"], "--revision", model["revision"], "--init-adapter", str(init_adapter),
            "--sam-checkpoint", str(sam_checkpoint()), "--cohort-protocol", str(REPO / "protocols/monuseg_official30.json"),
            "--coordinate-frame", model["coordinate_frame"],
            "--min-pixels", str(shared["min_pixels"]), "--max-pixels", str(shared["max_pixels"]),
            "--max-sequence-length", str(shared["max_sequence_length"]), "--max-objects", str(shared["max_objects"]),
            "--eval-max-completion-length", str(shared["max_completion_length"]), *_flags(settings)]
    return argv + (["--skip-before-eval"] if canary else [])


# ---------- Hugging Face cache with a Volume mirror ----------

def _hub_dir(cache, model):
    return Path(cache) / "hub" / ("models--" + model.replace("/", "--"))


def _snapshot(cache, model, revision):
    return _hub_dir(cache, model) / "snapshots" / revision


COMPLETE = ".segrlvr-complete"  # Written last, so a partial copy is never mistaken for a snapshot.


def _copy_snapshot(source_hub, target_hub, revision):
    """Copy refs plus one snapshot as plain files (Volumes keep no symlinks), then mark it complete."""
    if (source_hub / "refs").is_dir():
        shutil.copytree(source_hub / "refs", target_hub / "refs", dirs_exist_ok=True, copy_function=shutil.copyfile)
    final = target_hub / "snapshots" / revision
    (final / COMPLETE).unlink(missing_ok=True)
    shutil.copytree(source_hub / "snapshots" / revision, final, symlinks=False, dirs_exist_ok=True,
                    copy_function=shutil.copyfile, ignore=shutil.ignore_patterns(COMPLETE))
    (final / COMPLETE).write_text(revision + "\n")


def _snapshot_complete(path):
    """Marker from _copy_snapshot, or a manually uploaded snapshot with every indexed weight shard."""
    if (path / COMPLETE).is_file():
        return True
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").is_file() or not index.is_file():
        return False
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return all((path / shard).is_file() for shard in shards)


def ensure_model(model, revision, mirror=True):
    """Make one pinned model snapshot available in the local HF cache.

    Order: local cache, then the Volume mirror (offline-capable), then a
    Hugging Face download, which is mirrored to the Volume when `mirror`.
    Sets HF_HUB_OFFLINE=1 when the model came from the mirror.
    """
    hf_home = configure_environment()
    local = _snapshot(hf_home, model, revision)
    volume_cache = artifact("cache", "huggingface")
    mirrored = _snapshot(volume_cache, model, revision)
    if (local / COMPLETE).is_file():
        return {"source": "local_cache", "path": str(local)}
    started = time.time()
    if _snapshot_complete(mirrored):
        _copy_snapshot(_hub_dir(volume_cache, model), _hub_dir(hf_home, model), revision)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return {"source": "volume_mirror", "path": str(local), "seconds": round(time.time() - started, 1)}
    from huggingface_hub import snapshot_download
    snapshot_download(model, revision=revision, cache_dir=str(hf_home / "hub"))  # Resumes partial downloads.
    (local / COMPLETE).write_text(revision + "\n")
    result = {"source": "huggingface_download", "path": str(local), "seconds": round(time.time() - started, 1)}
    if mirror:
        _copy_snapshot(_hub_dir(hf_home, model), _hub_dir(volume_cache, model), revision)
        result["mirrored_to"] = str(mirrored)
    return result


# ---------- Running the repository CLIs ----------

def mirror_tree(source, target):
    """Incrementally copy new or changed files; never deletes on the Volume."""
    source, target = Path(source), Path(target)
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        # Hidden names are in-progress writes (e.g. ".writing-checkpoint-128/").
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        destination = target / relative
        try:
            stat = path.stat()
            if destination.is_file() and destination.stat().st_size == stat.st_size \
                    and destination.stat().st_mtime >= stat.st_mtime:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            copied += 1
        except FileNotFoundError:
            continue  # The runner replaced a file mid-copy; the next pass catches up.
    return copied


def run_cli(argv, local_output=None, volume_output=None, log_name="run.log", mirror_every=600):
    """Run `python -m ...` from the repository root, streaming output to the notebook.

    If both output paths are given, the local run directory is mirrored to the
    Volume every `mirror_every` seconds and once more at the end, success or not.
    """
    configure_environment()
    command = [sys.executable, *map(str, argv)]
    print("$", " ".join(command), flush=True)
    # The log sits beside the run directory, so it survives even if the runner
    # refuses to start because its output directory is not empty.
    anchor = Path(volume_output or local_output or local_root() / "logs" / "run")
    log_path = anchor.with_name(f"{anchor.name}.{log_name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def mirror_loop():
        while not stop.wait(mirror_every):
            if Path(local_output).is_dir():
                mirror_tree(local_output, volume_output)

    worker = None
    if local_output and volume_output:
        worker = threading.Thread(target=mirror_loop, daemon=True)
        worker.start()
    started = time.time()
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=os.environ.copy())
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        finally:
            stop.set()
            if worker:
                worker.join()
            if local_output and volume_output and Path(local_output).is_dir():
                print(f"mirrored {mirror_tree(local_output, volume_output)} files to {volume_output}", flush=True)
    print(f"exit code {code} after {time.time() - started:.0f}s; log: {log_path}", flush=True)
    if code:
        raise RuntimeError(f"Command failed with exit code {code}; see {log_path}")
    return code


def new_run_dirs(name):
    """Fresh local and Volume directories for one run; refuses to reuse a name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    local, volume = local_root() / "runs" / f"{name}-{stamp}", artifact("runs", f"{name}-{stamp}")
    if volume.exists():
        raise FileExistsError(volume)
    return local, volume


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_manifests(produced_dir, reference_dir):
    """Path-independent comparison of crop rows against a reference dataset copy."""
    report = {}
    for name in ("train", "validation", "validation_benchmark8"):
        mine, theirs = Path(produced_dir) / f"{name}.jsonl", Path(reference_dir) / f"{name}.jsonl"
        if not theirs.is_file():
            report[name] = "reference missing"
            continue
        strip = lambda rows: [{k: v for k, v in row.items() if k != "source_archive"} for row in rows]
        a, b = strip(read_jsonl(mine)), strip(read_jsonl(theirs))
        if a == b:
            report[name] = f"identical ({len(a)} rows, ignoring source_archive)"
        else:
            ids_a, ids_b = [r["id"] for r in a], [r["id"] for r in b]
            pixels = {r["id"]: r.get("image_pixels_sha256") for r in b}
            report[name] = {"rows": [len(a), len(b)], "same_ids_in_order": ids_a == ids_b,
                            "only_here": sorted(set(ids_a) - set(ids_b)), "only_reference": sorted(set(ids_b) - set(ids_a)),
                            "pixel_mismatches": sorted(r["id"] for r in a if r["id"] in pixels
                                                       and r.get("image_pixels_sha256") != pixels[r["id"]])}
    return report
