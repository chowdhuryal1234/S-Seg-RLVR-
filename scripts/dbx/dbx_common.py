"""Shared helpers for the Databricks Serverless GPU notebooks in this folder.

Notebooks import this module after adding this folder to sys.path. It keeps
the workspace paths in one place, runs the repository CLIs in a subprocess,
and moves data between fast local disk and a Unity Catalog Volume.

All Volume access goes through the Databricks Files API (`Volume` below), not
the /Volumes FUSE mount: on Serverless GPU the mount can fail to reach the
storage account while the Files API still works. Inputs are downloaded to
local disk, every runner reads and writes local paths only, and run
directories are uploaded to the Volume while they run and when they finish.
Edit the PATHS block once for your workspace; nothing else is hard-coded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]

# ---------- Edit for your workspace ----------
VOLUME_ROOT = "/Volumes/aistudio_dev_catalog/aios_segrlvr_dev"
PATHS = {
    # Extracted MoNuSeg 2018 training data (30 official images), as already uploaded.
    "train_dir": f"{VOLUME_ROOT}/raw/monuseg/train",
    # Persistent outputs: dataset, SAM checkpoint, runs (and an optional offline model copy).
    "artifacts": f"{VOLUME_ROOT}/processed/segrlvr",
}
# ---------------------------------------------

DATASET_NAME = "monuseg_official30_grpo_v1"
SAM_NAME = "sam2.1_hiera_tiny.pt"
SAM_URL = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{SAM_NAME}"
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
# Manifest digests from the Harvard run. Manifests embed the absolute source
# archive path, so these match only if that path is identical; compare_manifests
# below checks the path-independent content instead.
HARVARD_MANIFEST_SHA256 = {
    "train": "a49d484b4202ddd21f1ce34eda3859641fc38bbbd72b665e27fcf63190300dcc",
    "validation": "1969cd648b83fa0d9f1bfcaa73d248a879e3f3156e2863e46c540cf7cce853f8",
    "validation_benchmark8": "15520402033fafe5560b7b3f14dbc619686cec7b208d35990ee558e3185fd60c",
}


# ---------- Paths ----------

def local_root():
    """Fast ephemeral disk for inputs, the model cache and in-progress runs."""
    base = Path("/local_disk0") if Path("/local_disk0").is_dir() and os.access("/local_disk0", os.W_OK) \
        else Path(tempfile.gettempdir())
    path = base / "segrlvr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remote(*parts):
    """A path under the Volume artifacts folder (a Files API path, not a mounted file)."""
    return str(PurePosixPath(PATHS["artifacts"], *parts))


def dataset_dir():
    return local_root() / "data" / DATASET_NAME


def sam_checkpoint():
    return local_root() / "checkpoints" / SAM_NAME


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


# ---------- Unity Catalog Volume through the Files API ----------

def _is_hidden(relative):
    # Hidden names are in-progress writes (e.g. ".writing-checkpoint-128/").
    return any(part.startswith(".") for part in PurePosixPath(relative).parts)


class Volume:
    """Minimal Volume client. `files` is a WorkspaceClient().files (or a test double)."""

    def __init__(self, files=None):
        self._files = files
        self._made = set()

    @property
    def files(self):
        if self._files is None:
            # Newer SDKs upload and download through presigned cloud-storage URLs, which Serverless GPU
            # cannot reach here ("Exceeded max retry attempts"). This switch keeps every transfer on the
            # plain Files API endpoints (5 GB per file), which do work.
            os.environ.setdefault("DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT", "true")
            from databricks.sdk import WorkspaceClient
            self._files = WorkspaceClient().files
        return self._files

    @staticmethod
    def _missing(error):
        return isinstance(error, FileNotFoundError) or type(error).__name__ in ("NotFound", "ResourceDoesNotExist")

    def is_file(self, path):
        try:
            self.files.get_metadata(path)
            return True
        except Exception as error:
            if self._missing(error):
                return False
            raise

    def is_dir(self, path):
        try:
            self.files.get_directory_metadata(path)
            return True
        except Exception as error:
            if self._missing(error):
                return False
            raise

    def mkdir(self, path):
        if path not in self._made:
            self.files.create_directory(path)  # Creates parents as needed.
            self._made.add(path)

    def walk(self, path):
        """Yield (relative path, size) for every file below `path`."""
        root = PurePosixPath(path)
        for entry in self.files.list_directory_contents(str(root)):
            if entry.is_directory:
                for relative, size in self.walk(entry.path):
                    yield str(PurePosixPath(entry.path).relative_to(root) / relative), size
            else:
                yield str(PurePosixPath(entry.path).relative_to(root)), entry.file_size

    def download(self, path, local_path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        partial = local_path.with_name(f".{local_path.name}.partial")
        stream = self.files.download(path).contents
        try:
            with partial.open("wb") as output:
                shutil.copyfileobj(stream, output, 1 << 20)
        finally:
            stream.close()
        partial.replace(local_path)

    def download_tree(self, path, local_dir):
        count = 0
        for relative, _ in self.walk(path):
            self.download(f"{path}/{relative}", Path(local_dir) / relative)
            count += 1
        return count

    def upload(self, local_path, path):
        self.mkdir(str(PurePosixPath(path).parent))
        with Path(local_path).open("rb") as stream:
            self.files.upload(path, stream, overwrite=True)

    def upload_tree(self, local_dir, path, state=None):
        """Upload new or changed files; `state` remembers what was sent. Never deletes."""
        state = {} if state is None else state
        local_dir, sent = Path(local_dir), 0
        for file in sorted(local_dir.rglob("*")):
            relative = file.relative_to(local_dir).as_posix()
            if not file.is_file() or _is_hidden(relative):
                continue
            try:
                stat = file.stat()
                key = (stat.st_size, stat.st_mtime_ns)
                if state.get(relative) == key:
                    continue
                self.upload(file, f"{path}/{relative}")
            except FileNotFoundError:
                continue  # The runner replaced a file mid-upload; the next pass catches up.
            state[relative] = key
            sent += 1
        return sent

    def read_text(self, path):
        stream = self.files.download(path).contents
        try:
            return stream.read().decode()
        finally:
            stream.close()

    def write_text(self, path, text):
        import io
        self.mkdir(str(PurePosixPath(path).parent))
        self.files.upload(path, io.BytesIO(text.encode()), overwrite=True)


_VOLUME = None


def volume():
    global _VOLUME
    if _VOLUME is None:
        _VOLUME = Volume()
    return _VOLUME


# ---------- Inputs on local disk ----------

def ensure_dataset():
    """Local copy of the frozen dataset written by 02_prepare_data."""
    local = dataset_dir()
    if not (local / "frozen_protocol.json").is_file():
        source = remote("data", DATASET_NAME)
        if not volume().is_dir(source):
            raise FileNotFoundError(f"{source} not found; run 02_prepare_data first")
        print(f"downloaded {volume().download_tree(source, local)} dataset files from {source}")
    return local


def ensure_sam():
    """SAM2.1 tiny checkpoint on local disk, hash-checked; also kept on the Volume."""
    local, stored = sam_checkpoint(), remote("checkpoints", SAM_NAME)
    if not local.is_file():
        if volume().is_file(stored):
            volume().download(stored, local)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SAM_URL, local)
    digest = file_sha256(local)
    if digest != SAM_SHA256:
        local.unlink()
        raise ValueError(f"SAM checkpoint hash {digest} differs from the frozen baseline")
    if not volume().is_file(stored):
        volume().upload(local, stored)
    return local


def _snapshot(cache, model, revision):
    return Path(cache) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots" / revision


def ensure_model(model, revision):
    """Make one pinned model snapshot available in the local HF cache.

    Downloads from Hugging Face. Without internet access, upload the snapshot
    files to <artifacts>/cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/
    and they are copied from the Volume instead (HF_HUB_OFFLINE=1 is then set).
    """
    hf_home = configure_environment()
    local = _snapshot(hf_home, model, revision)
    marker = local / ".segrlvr-complete"  # Written last, so a partial copy is never trusted.
    if marker.is_file():
        return {"source": "local_cache", "path": str(local)}
    started = time.time()
    mirrored = remote(*_snapshot("cache/huggingface", model, revision).parts)
    if volume().is_dir(mirrored):
        volume().download_tree(mirrored, local)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        source = "volume_copy"
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(model, revision=revision, cache_dir=str(hf_home / "hub"))  # Resumes partial downloads.
        source = "huggingface_download"
    marker.write_text(revision + "\n")
    return {"source": source, "path": str(local), "seconds": round(time.time() - started, 1)}


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


# ---------- Running the repository CLIs ----------

def new_run_dirs(name):
    """Fresh local run directory and its Volume destination; refuses to reuse a name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    local, destination = local_root() / "runs" / f"{name}-{stamp}", remote("runs", f"{name}-{stamp}")
    if volume().is_dir(destination):
        raise FileExistsError(destination)
    return local, destination


def run_cli(argv, local_output, remote_output, upload_every=600):
    """Run `python -m ...` from the repository root, streaming output to the notebook.

    The local run directory is uploaded to `remote_output` every `upload_every`
    seconds and once more at the end, success or not. The log is kept beside
    the run directory (local and Volume), so it survives a refused start.
    """
    configure_environment()
    command = [sys.executable, *map(str, argv)]
    print("$", " ".join(command), flush=True)
    local_output = Path(local_output)
    log_path = local_output.with_name(local_output.name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sent, lock, stop = {}, threading.Lock(), threading.Event()

    def upload():
        with lock:
            count = volume().upload_tree(local_output, remote_output, sent) if local_output.is_dir() else 0
            volume().upload(log_path, remote_output + ".log")
            return count

    def upload_loop():
        while not stop.wait(upload_every):
            try:
                upload()
            except Exception as error:  # A transient upload failure must not stop training.
                print(f"[upload deferred: {type(error).__name__}: {error}]", flush=True)

    worker = threading.Thread(target=upload_loop, daemon=True)
    worker.start()
    started = time.time()
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=os.environ.copy())
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
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
            worker.join()
    print(f"uploaded {upload()} files to {remote_output}", flush=True)
    print(f"exit code {code} after {time.time() - started:.0f}s; log: {remote_output}.log", flush=True)
    if code:
        raise RuntimeError(f"Command failed with exit code {code}; see {remote_output}.log")
    return code


def read_json_remote(path):
    return json.loads(volume().read_text(path)) if volume().is_file(path) else {}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_manifests(produced_dir, reference_dir):
    """Path-independent comparison of crop rows against a local reference dataset copy."""
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
