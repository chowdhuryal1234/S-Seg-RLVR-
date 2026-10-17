"""Verify four completed SFT canaries and submit one idempotent H200 array.

The default is a read-only scheduler dry run. --submit is required to allocate
resources. A durable receipt is written before sbatch; any uncertain outcome
blocks automatic resubmission, including after a crash or a second invocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


RECEIPT_NAME = "h200_submission_receipt.json"
LOCK_NAME = ".h200_submission.lock"
TASK_WINDOW_SECONDS = 23 * 3600
EXPECTED_DIRECTIVES = {
    "--job-name": "sseg-sft", "--account": "mooney_lab",
    "--partition": "gpu_h200,seas_gpu", "--gres": "gpu:nvidia_h200:1",
    "--nodes": "1", "--ntasks": "1", "--cpus-per-task": "8",
    "--mem": "128G", "--time": "1-00:00:00", "--array": "0-3%4",
}


class SubmissionBlocked(RuntimeError):
    """An existing receipt or scheduler job requires inspection before action."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable_json(path, value):
    """Atomically replace JSON only after flushing its contents to storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def submission_lock(root):
    with (Path(root) / LOCK_NAME).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SubmissionBlocked("Another submission helper holds the workspace lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _suite_api(code):
    code = Path(code).resolve()
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("_sft_submission_suite", code / "scripts/run_sft_experiments.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the staged SFT suite")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_batch_script(script):
    directives = {}
    for line in Path(script).read_text().splitlines():
        if line.startswith("#SBATCH"):
            words = shlex.split(line[len("#SBATCH"):].strip())
            if len(words) == 1 and "=" in words[0]:
                key, value = words[0].split("=", 1)
            elif len(words) == 2:
                key, value = words
            else:
                raise ValueError("Unexpected or combined SBATCH directives")
            if key in directives:
                raise ValueError(f"Repeated SBATCH directive: {key}")
            directives[key] = value
    if directives != EXPECTED_DIRECTIVES:
        raise ValueError("SBATCH allocation differs from four one-H200, 8-CPU, 128G, 24-hour tasks")
    checked = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=30, check=False)
    if checked.returncode:
        raise ValueError(f"SBATCH shell syntax failed: {checked.stderr}")
    return directives


def job_name_for_root(root):
    name = Path(root).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Workspace basename must be a plain, unique Slurm job name")
    return name


def submission_command(root):
    root = Path(root).resolve()
    if any(character in str(root) for character in (",", "\n", "\r", "\x00")):
        raise ValueError("Workspace path cannot contain Slurm export separators")
    code = root / "code"
    return ["sbatch", "--parsable", f"--export=ALL,SFT_ROOT={root}",
            f"--output={root / 'logs/h200-%A_%a.out'}", f"--chdir={code}",
            f"--job-name={job_name_for_root(root)}", str(code / "scripts/run_sft_h200.sh")]


def build_plan(root):
    """CPU-only preflight against the already frozen code, data, and canaries."""
    root = Path(root).resolve()
    code = root / "code"
    api = _suite_api(code)
    config_path = code / "configs/sft_experiments.json"
    config = api.load_config(config_path)
    pins = api.pin_inputs(code, root / "checkpoints/sam2.1_hiera_tiny.pt", config_path)
    api.verify_pins(pins)
    script = code / "scripts/run_sft_h200.sh"
    directives = validate_batch_script(script)
    forecasts, canaries = {}, {}
    reserve = config["budget"]["allocation_reserve_seconds"]
    available = TASK_WINDOW_SECONDS - reserve
    if not math.isfinite(available) or available <= 0:
        raise ValueError("Invalid allocation reserve")
    for key in api.MODEL_KEYS:
        record, result = api.canary_record(root / "runs/a100-canaries", key, config, pins)
        timing = api.timing_from_canary(record, result, config)
        required = timing["paired_hard_seconds"]
        if not isinstance(required, (int, float)) or not math.isfinite(required) or not 0 < required <= available:
            raise ValueError(f"Model {key} paired schedule cannot fit 23 hours minus its reserve: {required} > {available}")
        forecasts[key] = timing
        canaries[key] = {"status": record["status"], "files": dict(record["files"]),
                         "input_identity_sha256": record["input_identity_sha256"],
                         "elapsed_seconds": record["elapsed_seconds"]}
    if tuple(api.MODEL_KEYS) != ("7b", "32b", "9b", "27b"):
        raise ValueError("Array model mapping changed")
    api.verify_pins(pins)
    return {"schema_version": 1, "status": "ready", "root": str(root), "job_name": job_name_for_root(root),
            "command": submission_command(root), "input_identity_sha256": pins["input_identity_sha256"],
            "canaries": canaries, "forecasts": forecasts, "sbatch_directives": directives,
            "task_forecast_window_seconds": TASK_WINDOW_SECONDS, "reserve_seconds": reserve,
            "files": {"batch_script": {"path": str(script), "sha256": file_sha256(script)},
                      "submission_helper": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
                      "suite_config": {"path": str(config_path), "sha256": file_sha256(config_path)}},
            "created_utc": datetime.now(timezone.utc).isoformat()}


def active_matching_jobs(job_name):
    checked = subprocess.run(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%T"],
                             capture_output=True, text=True, timeout=30, check=False)
    if checked.returncode:
        raise RuntimeError(f"Cannot check existing Slurm jobs: {checked.stderr}")
    matching = []
    for line in checked.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 3:
            raise ValueError("Unexpected squeue output; cannot safely exclude duplicate jobs")
        if fields[1] == job_name:
            matching.append({"job_id": fields[0], "job_name": fields[1], "state": fields[2]})
    return matching


def parse_submission_id(stdout):
    match = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?\s*", stdout)
    if not match:
        raise ValueError("sbatch did not return one unambiguous parsable job ID")
    return match.group(1), match.group(2)


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run(root, *, submit=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    receipt_path = root / RECEIPT_NAME
    with submission_lock(root):
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text())
            except (ValueError, OSError) as error:
                raise SubmissionBlocked("An existing unreadable submission receipt blocks resubmission") from error
            if receipt.get("status") == "SUBMITTED":
                return {"status": "already_submitted", "receipt_path": str(receipt_path), "receipt": receipt}
            raise SubmissionBlocked(f"Existing receipt status {receipt.get('status')!r} blocks resubmission; inspect {receipt_path}")
        audit_path = root / "logs/submission-checks" / f"{time.time_ns()}-{os.getpid()}.json"
        audit = {"status": "checking", "submit_requested": submit, "root": str(root)}
        try:
            plan = build_plan(root)
            matching = active_matching_jobs(plan["job_name"])
            if matching:
                audit["matching_jobs"] = matching
                raise SubmissionBlocked("An existing job with this exact workspace job name blocks duplicate submission")
            for item in plan.get("files", {}).values():
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError(f"Submission source changed during preflight: {item['path']}")
            audit.update(status="ready", plan=plan)
            durable_json(audit_path, audit)
            if not submit:
                return {"status": "dry_run_ready", "plan": plan, "audit_path": str(audit_path)}
            (root / "logs").mkdir(exist_ok=True)
            receipt = {"schema_version": 1, "status": "SUBMITTING", "plan": plan,
                       "audit_path": str(audit_path), "started_utc": datetime.now(timezone.utc).isoformat()}
            durable_json(receipt_path, receipt)
            # Once this receipt is durable, every uncertain branch leaves it in
            # place. Never infer a successful allocation from a hypothetical ID.
            try:
                # Inherited SBATCH_* settings otherwise override directives in
                # the reviewed script, including GPU counts or wall time.
                environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
                # --export=ALL gives inherited values precedence over explicit
                # values, so the inherited workspace must match our audit too.
                environment["SFT_ROOT"] = str(root)
                response = subprocess.run(plan["command"], capture_output=True, text=True,
                                          timeout=120, check=False, env=environment)
                receipt["response"] = {"returncode": response.returncode, "stdout": response.stdout, "stderr": response.stderr}
                if response.returncode:
                    raise RuntimeError(f"sbatch returned {response.returncode}; its outcome requires inspection")
                job_id, cluster = parse_submission_id(response.stdout)
                receipt.update(status="SUBMITTED", job_id=job_id, cluster=cluster,
                               completed_utc=datetime.now(timezone.utc).isoformat())
                durable_json(receipt_path, receipt)
                return {"status": "submitted", "receipt_path": str(receipt_path), "receipt": receipt}
            except BaseException as error:
                receipt.update(status="AMBIGUOUS", error_type=type(error).__name__, error=str(error),
                               completed_utc=datetime.now(timezone.utc).isoformat())
                if isinstance(error, subprocess.TimeoutExpired):
                    receipt["timeout_output"] = {"stdout": _text(error.stdout), "stderr": _text(error.stderr)}
                durable_json(receipt_path, receipt)
                raise SubmissionBlocked("Submission outcome is uncertain; the durable receipt blocks automatic retries") from error
        except BaseException as error:
            audit.update(status="failed", error_type=type(error).__name__, error=str(error))
            durable_json(audit_path, audit)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true", help="Submit exactly one four-task H200 array after every check passes")
    args = parser.parse_args(argv)
    try:
        result = run(args.root, submit=args.submit)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__, "error": str(error)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
