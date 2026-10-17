"""Run one model's paired SFT experiments inside an existing GPU allocation.

This script never requests GPUs. Four verified A100 canaries gate the main H200
suite. Every arm starts from the same immutable base checkpoint and a fresh
adapter. Existing run directories and failed attempts are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nucleus_rl.training_budget import contract_digest, file_digest, write_json
from scripts.run_h200_experiments import DATASET, pin_inputs as pin_frozen_dataset, verify_pins


MODEL_KEYS = ("7b", "32b", "9b", "27b")
STOP_REQUESTED = False


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("dataset") != DATASET:
        raise ValueError("Unexpected SFT suite schema or frozen dataset")
    if set(config.get("models", {})) != set(MODEL_KEYS):
        raise ValueError("Expected exactly the four approved model candidates")
    for item in config["models"].values():
        if not isinstance(item.get("model"), str) or not re.fullmatch(r"[a-f0-9]{40}", item.get("revision", "")):
            raise ValueError("Every base model requires an immutable 40-character revision")
        if item.get("coordinate_frame") not in ("processed", "normalized1000"):
            raise ValueError("Each model needs an explicit supported coordinate frame")
    main, shared, canary, budget = (config[name] for name in ("main", "shared", "canary", "budget"))
    if (main["augmentations"] != ["none", "d4"] or main["max_steps"] != 512
            or main["train_examples"] != 256 or main["validation_examples"] != 64
            or shared["gradient_accumulation"] != 4
            or main["max_steps"] * shared["gradient_accumulation"] != 8 * main["train_examples"]):
        raise ValueError("Paired arms must each receive 2048 exposures across the fixed 256 training crops")
    if (canary["max_steps"] != 20 or canary["train_limit"] != 4 or canary["eval_limit"] != 8
            or canary["augmentation"] != "d4"):
        raise ValueError("Canaries must use the approved 20-step, four-train, eight-validation D4 check")
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive budget setting: {key}")
    return config


def pin_inputs(code, sam, config_path):
    pins = pin_frozen_dataset(code, sam)
    code = Path(code).resolve()
    for path in (code / "scripts/run_sft_experiments.py", Path(config_path), code / "requirements-sft.txt"):
        pins["files"][str(path.resolve())] = file_digest(path)
    pins["source_policy"] = "Fresh language-LoRA SFT base per arm; training-mask-derived prompt targets; fixed official30 validation; no official test"
    # This identity remains stable if an identical frozen code/data tree is
    # copied to another node or mount. Absolute paths remain recorded as well.
    identity = {}
    for filename, digest in pins["files"].items():
        path = Path(filename)
        try:
            name = path.relative_to(code).as_posix()
        except ValueError:
            name = "external/" + path.name
        if name in identity and identity[name] != digest:
            raise ValueError("Ambiguous external input basename in pinned inputs")
        identity[name] = digest
    pins["input_identity_sha256"] = contract_digest(identity)
    return pins


def canary_errors(result, metadata, model, revision):
    errors = []
    checks = {
        "verified_sft_run": result.get("status") == "verified_sft_run",
        "canary_scope": result.get("scope") == "canary",
        "model_revision": result.get("model") == model and result.get("resolved_revision") == revision,
        "all_20_steps": result.get("optimizer_steps") == 20 and result.get("completed_requested_steps") is True,
        "four_train_eight_validation": result.get("train_examples") == 4 and result.get("validation_examples") == 8,
        "adapter_changed": result.get("adapter_changed") is True,
        "base_frozen": result.get("frozen_base_versions_unchanged") is True,
        "vision_frozen": bool(result.get("vision_sha256_before")) and result.get("vision_sha256_before") == result.get("vision_sha256_after"),
        "sam_frozen": result.get("sam_freeze_verification", {}).get("unchanged") is True,
        "gpu_recorded": isinstance(metadata.get("gpu"), str) and bool(metadata["gpu"]),
    }
    audit = result.get("d4_collator_audit", {})
    checks["all_d4_collations"] = audit.get("passed") is True and audit.get("transforms") == list(range(8))
    for phase in ("before", "after"):
        checks[phase + "_evaluation8"] = result.get(phase, {}).get("completions") == 8
    before = result.get("fixed_training_loss_before", {}).get("mean_loss")
    after = result.get("fixed_training_loss_after", {}).get("mean_loss")
    checks["finite_decreasing_fixed_training_loss"] = (
        all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (before, after))
        and after < before
    )
    for name in ("mean_optimizer_step_seconds", "before_evaluation_seconds", "after_evaluation_seconds", "peak_gpu_reserved_bytes"):
        value = result.get(name)
        checks[name] = isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    errors.extend(name for name, passed in checks.items() if not passed)
    return errors


def canary_record(root, key, config, pins):
    directory = Path(root) / key
    record = json.loads((directory / "canary_result.json").read_text())
    if record.get("status") != "passed" or record.get("input_identity_sha256") != pins["input_identity_sha256"]:
        raise ValueError(f"Canary did not pass against these exact pinned inputs: {key}")
    payloads = {}
    for name in ("summary.json", "metadata.json"):
        path = directory / "d4" / name
        if file_digest(path) != record["files"][name]:
            raise ValueError(f"Canary result changed after verification: {key}/{name}")
        payloads[name] = json.loads(path.read_text())
    model = config["models"][key]
    errors = canary_errors(payloads["summary.json"], payloads["metadata.json"], model["model"], model["revision"])
    if errors:
        raise ValueError(f"Canary verification failed for {key}: {errors}")
    return record, payloads["summary.json"]


def timing_from_canary(record, result, config):
    budget, main = config["budget"], config["main"]
    step = result["mean_optimizer_step_seconds"]
    eval8 = max(result["before_evaluation_seconds"], result["after_evaluation_seconds"])
    elapsed = record["elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Missing positive measured canary wall time")
    # Residual includes cold loading, D4 audit, loss probes, checkpoint writes,
    # and other overhead. Retaining all of it is deliberately conservative.
    residual = max(0.0, elapsed - result["training_elapsed_seconds"]
                   - result["before_evaluation_seconds"] - result["after_evaluation_seconds"])
    # Successful SFT may emit many more objects than the untrained model.
    # Short initial completions therefore cannot set a tiny evaluation timeout.
    evaluation = max(budget["minimum_evaluation64_seconds"],
                     eval8 * main["validation_examples"] / 8 * budget["evaluation_safety_factor"])
    periodic = (main["max_steps"] - 1) // main["eval_every"] if main["eval_every"] else 0
    training = step * main["max_steps"] * budget["step_safety_factor"]
    checkpoint = math.ceil(main["max_steps"] / main["save_steps"]) * budget["checkpoint_seconds"]
    load = max(120.0, residual * budget["load_safety_factor"])
    soft = training + periodic * evaluation + checkpoint
    hard = soft + 2 * evaluation + load + budget["final_save_seconds"] + budget["termination_slack_seconds"]
    return {"measured_seconds_per_step": step, "training_seconds": training,
            "periodic_evaluations": periodic, "seconds_per_evaluation64": evaluation,
            "before_after_evaluation_seconds": 2 * evaluation,
            "periodic_evaluation_seconds": periodic * evaluation,
            "checkpoint_seconds": checkpoint, "cold_load_and_audit_seconds": load,
            "soft_training_seconds": soft, "hard_seconds_per_arm": math.ceil(hard),
            "paired_hard_seconds": 2 * math.ceil(hard),
            "method": "A100 canary measurements with 2x step and evaluation allowance; no assumed H200 speedup"}


def command(args, config, augmentation, output, soft_seconds):
    model, shared = config["models"][args.model_key], config["shared"]
    stage = config[args.stage]
    argv = [str(args.python), "-m", "nucleus_rl.sft", "--manifest", str(args.code / DATASET / "train.jsonl"),
            "--eval-manifest", str(args.code / DATASET / "validation.jsonl"), "--output", str(output),
            "--model", model["model"], "--revision", model["revision"],
            "--sam-checkpoint", str(args.sam_checkpoint), "--sam-device", "cpu",
            "--cohort-protocol", str(args.code / "protocols/monuseg_official30.json"),
            "--coordinate-frame", model["coordinate_frame"], "--augmentation", augmentation,
            "--wall-time-minutes", str(soft_seconds / 60)]
    settings = {**shared, **{key: stage[key] for key in ("max_steps", "warmup_steps", "eval_every", "save_steps")}}
    if args.stage == "canary":
        settings.update({key: stage[key] for key in ("train_limit", "eval_limit")})
    for key, value in settings.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    return argv


def resolve_deadline(explicit=None, environment=None):
    environment = os.environ if environment is None else environment
    value = explicit if explicit is not None else environment.get("SLURM_JOB_END_TIME")
    try:
        deadline = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Provide --deadline-epoch or SLURM_JOB_END_TIME from the existing allocation") from error
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("Allocation deadline must be a finite future Unix timestamp")
    return deadline


def gpu_identity(stage):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU must be visible")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if stage == "main" and "H200" not in name.upper():
        raise RuntimeError(f"Main suite requires the authorized H200 allocation, found {name}")
    if stage == "canary" and ("A100" not in name.upper() or memory < 70_000_000_000):
        raise RuntimeError(f"Canary requires the requested full A100 80GB, found {name} with {memory} bytes")
    return {"gpu": name, "total_memory_bytes": memory, "visible_gpus": 1,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def execute(argv, cwd, log, timeout):
    started = time.monotonic()
    with Path(log).open("xb") as output:
        process = subprocess.Popen(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        stop_reason = None
        while process.poll() is None:
            if STOP_REQUESTED or time.monotonic() - started >= timeout:
                stop_reason = "supervisor_signal" if STOP_REQUESTED else "hard_wall_time_budget"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    return {"exit_code": process.returncode, "elapsed_seconds": time.monotonic() - started,
            "stop_reason": stop_reason}


def main_run_errors(result, config, model_key=None):
    main = config["main"]
    checks = (result.get("status") == "verified_sft_run", result.get("scope") == "main",
              result.get("completed_requested_steps") is True,
              result.get("optimizer_steps") == main["max_steps"],
              result.get("training_exposures_this_run") == 2048,
              result.get("train_examples") == 256, result.get("validation_examples") == 64,
              result.get("adapter_changed") is True, result.get("frozen_base_versions_unchanged") is True,
              bool(result.get("vision_sha256_before")) and result.get("vision_sha256_before") == result.get("vision_sha256_after"),
              result.get("sam_freeze_verification", {}).get("unchanged") is True,
              result.get("before", {}).get("completions") == 64,
              result.get("after", {}).get("completions") == 64)
    errors = [] if all(checks) else ["main run did not complete its declared fixed-data, fixed-step training/freeze contract"]
    if model_key:
        model = config["models"][model_key]
        if result.get("model") != model["model"] or result.get("resolved_revision") != model["revision"]:
            errors.append("Main run base model or immutable revision differs")
    return errors


def run(args):
    config = load_config(args.config)
    directory = args.root / args.model_key
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Never adopt or overwrite a previous attempt.
    state = {"stage": args.stage, "model_key": args.model_key, "status": "preparing", "runs": [],
             "created_utc_epoch": time.time(), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        pins = pin_inputs(args.code, args.sam_checkpoint, args.config)
        verify_pins(pins)
        state["input_identity_sha256"] = pins["input_identity_sha256"]
        write_json(directory / "pinned_inputs.json", pins)
        write_json(directory / "suite_status.json", state)
        if args.stage == "main":
            if not args.canary_root:
                raise ValueError("Main runs require --canary-root with all four verified canaries")
            verified = {key: canary_record(args.canary_root, key, config, pins) for key in MODEL_KEYS}
            timing = timing_from_canary(*verified[args.model_key], config)
            state["timing_forecast"] = timing
            augmentations = config["main"]["augmentations"]
            hard_seconds = timing["hard_seconds_per_arm"]
            soft_seconds = timing["soft_training_seconds"]
        else:
            augmentations = [config["canary"]["augmentation"]]
            hard_seconds = config["budget"]["canary_max_seconds"]
            soft_seconds = max(60, hard_seconds - 600)
        reserve = config["budget"]["allocation_reserve_seconds"]
        if args.dry_run:
            state["status"] = "dry_run_verified_without_gpu_loading_or_training"
            state["commands"] = [command(args, config, arm, directory / arm, soft_seconds) for arm in augmentations]
            return state
        deadline = resolve_deadline(args.deadline_epoch)
        state["deadline_epoch"] = deadline
        state["gpu"] = gpu_identity(args.stage)
        required = hard_seconds * len(augmentations)
        if required > deadline - time.time() - reserve:
            raise RuntimeError(f"Declared paired schedule needs {required:.0f}s plus {reserve}s reserve; allocation cannot fit it")
        for arm in augmentations:
            if STOP_REQUESTED:
                raise RuntimeError("Supervisor received a stop signal")
            verify_pins(pins)
            available = deadline - time.time() - reserve
            if available < hard_seconds:
                state["runs"].append({"augmentation": arm, "status": "not_started_insufficient_deadline_budget"})
                continue
            output = directory / arm
            argv = command(args, config, arm, output, soft_seconds)
            state["status"] = "running"
            state["current_augmentation"] = arm
            write_json(directory / "suite_status.json", state)
            write_json(directory / f"{arm}_command.json", argv)
            record = {"augmentation": arm, **execute(argv, args.code, directory / f"{arm}.log", min(hard_seconds, available))}
            summary_path, metadata_path = output / "summary.json", output / "metadata.json"
            if record["exit_code"] == 0 and summary_path.is_file() and metadata_path.is_file():
                try:
                    result = json.loads(summary_path.read_text())
                    metadata = json.loads(metadata_path.read_text())
                    model = config["models"][args.model_key]
                    errors = (canary_errors(result, metadata, model["model"], model["revision"])
                              if args.stage == "canary" else main_run_errors(result, config, args.model_key))
                    if metadata.get("before_is_resume_checkpoint") is not False:
                        errors.append("Fresh-base adapter initialization was not verified")
                except (ValueError, TypeError, KeyError) as error:
                    errors = [f"Invalid result payload: {type(error).__name__}: {error}"]
                record.update(status="passed" if not errors else "verification_failed", errors=errors,
                              files={"summary.json": file_digest(summary_path), "metadata.json": file_digest(metadata_path)})
                if args.stage == "canary":
                    record.update(input_identity_sha256=pins["input_identity_sha256"], model_key=args.model_key)
                    write_json(directory / "canary_result.json", record)
            else:
                record["status"] = "process_failed_or_incomplete_results"
            state["runs"].append(record)
            write_json(directory / "suite_status.json", state)
            # Continue a distinct second arm after a concrete first-arm failure
            # only if its original fixed budget still fits. Never retry an arm.
        verify_pins(pins)
        state["status"] = "complete" if len(state["runs"]) == len(augmentations) and all(r.get("status") == "passed" for r in state["runs"]) else "incomplete_or_failed"
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error_type"], state["error"] = type(error).__name__, str(error)
        raise
    finally:
        state["updated_utc_epoch"] = time.time()
        write_json(directory / "suite_status.json", state)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("canary", "main"), required=True)
    result.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    result.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--config", type=Path)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--canary-root", type=Path)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--sam-checkpoint", type=Path, required=True)
    result.add_argument("--deadline-epoch", type=float)
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("code", "root", "sam_checkpoint", "python", "canary_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.config = (args.config or args.code / "configs/sft_experiments.json").resolve()
    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    state = run(args)
    print(json.dumps({"stage": state["stage"], "model_key": state["model_key"], "status": state["status"]}))
    return 0 if state["status"] in ("complete", "dry_run_verified_without_gpu_loading_or_training") else 1


if __name__ == "__main__":
    raise SystemExit(main())
