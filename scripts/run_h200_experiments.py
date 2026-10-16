"""Sequential, deadline-bounded experiments inside an existing GPU allocation.

This supervisor never allocates resources, resumes adapters, retries failed
models, or changes rewards. --dry-run validates inputs/commands without torch.
The root must be new: interrupted suites remain evidence and are not adopted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


MODELS = {
    "7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "32b": ("Qwen/Qwen2.5-VL-32B-Instruct", "7cfb30d71a1f4f49a57592323337a4a4727301da"),
}
DATASET = "data/monuseg_official30_grpo_v1"
MANIFEST_HASHES = {
    "train": "a49d484b4202ddd21f1ce34eda3859641fc38bbbd72b665e27fcf63190300dcc",
    "validation": "1969cd648b83fa0d9f1bfcaa73d248a879e3f3156e2863e46c540cf7cce853f8",
    "validation_benchmark8": "15520402033fafe5560b7b3f14dbc619686cec7b208d35990ee558e3185fd60c",
}
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
FINAL_RESERVE = 3600
STOP_REQUESTED = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def pin_inputs(code, sam):
    """Verify frozen cohort, manifests, and every selected image/mask on CPU."""
    code = Path(code)
    data = code / DATASET
    files = {str(path.resolve()): digest(path) for path in sorted((code / "nucleus_rl").glob("*.py"))}
    for path in (code / "scripts/run_h200_experiments.py", code / "protocols/monuseg_official30.json",
                 data / "frozen_protocol.json", Path(sam)):
        files[str(path.resolve())] = digest(path)
    if files[str(Path(sam).resolve())] != SAM_SHA256:
        raise ValueError("SAM tiny checkpoint hash differs from the frozen baseline")
    protocol = read_json(data / "frozen_protocol.json")
    cohort_path = code / "protocols/monuseg_official30.json"
    if digest(cohort_path) != protocol["official30_protocol_sha256"]:
        raise ValueError("Official30 cohort protocol changed")
    canonical = set(read_json(cohort_path)["image_ids"])
    datasets = {}
    for name, count in (("train", 256), ("validation", 64), ("validation_benchmark8", 8)):
        path = data / f"{name}.jsonl"
        actual = digest(path)
        if actual != MANIFEST_HASHES[name] or actual != protocol["manifest_sha256"][name]:
            raise ValueError(f"Frozen manifest changed: {name}")
        files[str(path.resolve())] = actual
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise ValueError(f"Unexpected row count or repeated IDs: {name}")
        expected_split = "train" if name == "train" else "validation"
        for row in rows:
            if row["split"] != expected_split or row["source_image_id"] not in canonical:
                raise ValueError(f"Noncanonical or wrong-split row: {row['id']}")
            for kind in ("image", "mask"):
                asset = (path.parent / row[f"{kind}_path"]).resolve()
                actual = files.setdefault(str(asset), digest(asset))
                if actual != row[f"{kind}_sha256"]:
                    raise ValueError(f"Changed crop: {asset}")
        datasets[name] = rows
    if {r["patient_id"] for r in datasets["train"]} & {r["patient_id"] for r in datasets["validation"]}:
        raise ValueError("Training and validation patient overlap")
    full_validation = {r["id"]: r for r in datasets["validation"]}
    if any(full_validation.get(r["id"]) != r for r in datasets["validation_benchmark8"]):
        raise ValueError("Benchmark8 is not an exact subset of validation64")
    return {"files": files, "manifest_sha256": MANIFEST_HASHES,
            "counts": {name: len(rows) for name, rows in datasets.items()},
            "source_policy": "Fresh base for every run; fixed full-mask rewards; no official test."}


def verify_pins(pins):
    changed = [name for name, expected in pins["files"].items()
               if not Path(name).is_file() or digest(name) != expected]
    if changed:
        raise ValueError(f"Pinned source/data/checkpoint changed: {changed}")


def useful(record, complete=False):
    result = record.get("result") or {}
    return (record.get("exit_code") == 0 and result.get("status") == "verified_training_pilot"
            and result.get("adapter_changed") is True and result.get("nonzero_gradient_steps", 0) > 0
            and result.get("sam", {}).get("unchanged") is True
            and (not complete or result.get("completed_requested_steps") is True))


def timing_from_benchmark(record):
    if not useful(record, complete=True):
        return None
    r = record["result"]
    if any(r.get(phase, {}).get("completions") != 8 for phase in ("before", "after")):
        return None
    n = r["optimizer_steps_this_run"]
    training = r["training_elapsed_seconds"]
    if n != 20 or not math.isfinite(training) or training <= 0:
        return None
    before, after = r.get("before_evaluation_seconds"), r.get("after_evaluation_seconds")
    residual = max(0.0, r["elapsed_seconds"] - training)
    if before is not None and after is not None:
        eval8 = max(before, after)
        cold = max(0.0, residual - before - after)
        method = "Measured slower before/after evaluation8; residual includes load and final save"
    else:
        # Conservative fallback: attribute ALL residual to EACH evaluation8
        # and to cold loading. This double-counts rather than hides uncertainty.
        eval8, cold = residual, residual
        method = "Conservative residual fallback; load/evaluation/save deliberately double-counted"
    return {"seconds_per_step": training / n,
            "evaluation64_seconds": max(60.0, eval8 * 8 * 1.5),
            "cold_load_seconds": max(60.0, cold * 1.25),
            "step_safety_factor": 1.25, "checkpoint_seconds_each": 15.0,
            "final_save_seconds": 60.0, "termination_slack_seconds": 60.0,
            "method": method, "benchmark_total_seconds": r["elapsed_seconds"]}


def forecast(timing, steps):
    """Explicit conservative allowances; estimates are not runtime guarantees."""
    periodic = (steps - 1) // 100
    training = timing["seconds_per_step"] * steps * timing["step_safety_factor"]
    checkpoints = math.ceil(steps / 50) * timing["checkpoint_seconds_each"]
    soft = training + periodic * timing["evaluation64_seconds"] + checkpoints
    total = (soft + 2 * timing["evaluation64_seconds"] + timing["cold_load_seconds"]
             + timing["final_save_seconds"] + timing["termination_slack_seconds"])
    return {"steps": steps, "training_seconds": training, "periodic_evaluations": periodic,
            "periodic_evaluation_seconds": periodic * timing["evaluation64_seconds"],
            "before_after_evaluation_seconds": 2 * timing["evaluation64_seconds"],
            "cold_load_seconds": timing["cold_load_seconds"], "checkpoint_seconds": checkpoints,
            "final_save_seconds": timing["final_save_seconds"],
            "termination_slack_seconds": timing["termination_slack_seconds"],
            "soft_training_seconds": soft, "hard_seconds": math.ceil(total)}


def affordable_steps(timing, budget, minimum=100):
    if timing is None:
        return 0
    return max((steps for steps in range(minimum, 1001, 100)
                if forecast(timing, steps)["hard_seconds"] <= budget), default=0)


def plan_main(timings, available):
    """Give 32B at most two hours, and give both 7B LR arms equal steps."""
    large_steps = affordable_steps(timings.get("32b"), min(7200, max(0, available)), minimum=500)
    large_cost = forecast(timings["32b"], large_steps)["hard_seconds"] if large_steps else 0
    pair_steps = affordable_steps(timings.get("7b"), max(0, available - large_cost) / 2)
    if timings.get("7b") and not pair_steps and large_steps:
        large_steps, large_cost = 0, 0
        pair_steps = affordable_steps(timings["7b"], max(0, available) / 2)
    return {"available_seconds_excluding_final_reserve": max(0, available),
            "7b_equal_steps": pair_steps, "32b_steps": large_steps,
            "32b_reserved_seconds": large_cost, "intended_7b_steps": 1000,
            "policy": "7B equal-step LR comparison has priority; 32B requires >=500 steps within two hours; no automatic failed-model retry."}


def command(args, model, name, steps, learning_rate, seed, soft_seconds, benchmark=False):
    code, root = Path(args.code), Path(args.root)
    model_name, revision = MODELS[model]
    return [str(args.python), "-m", "nucleus_rl.train", "--mode", "train",
            "--manifest", str(code / DATASET / "train.jsonl"),
            "--eval-manifest", str(code / DATASET / ("validation_benchmark8.jsonl" if benchmark else "validation.jsonl")),
            "--output", str(root / name), "--sam-checkpoint", str(args.sam_checkpoint),
            "--sam-config", "configs/sam2.1/sam2.1_hiera_t.yaml",
            "--cohort-protocol", str(code / "protocols/monuseg_official30.json"),
            "--model", model_name, "--revision", revision,
            "--max-steps", str(steps), "--seed", str(seed), "--learning-rate", str(learning_rate),
            "--lora-rank", "8", "--group-size", "2", "--max-completion-length", "512",
            "--max-objects", "8", "--beta", "0.04", "--coordinate-frame", "processed",
            "--prompt-style", "example", "--min-pixels", "50176", "--max-pixels", "200704",
            "--save-steps", "10" if benchmark else "50", "--eval-every", "0" if benchmark else "100",
            "--wall-time-minutes", str(soft_seconds / 60), "--direct-gpu-load"]


def validation_key(record):
    if not useful(record, complete=True):
        return None
    summary = record["result"].get("after", {})
    pq, recall = summary.get("mean_pq"), summary.get("mean_recall")
    if summary.get("completions") != 64 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (pq, recall)):
        return None
    return pq, recall


def run_child(cmd, cwd, log_path, hard_seconds, popen=subprocess.Popen):
    """A new process group permits bounded termination without killing this shell."""
    started = time.monotonic()
    timed_out = interrupted = False
    environment = dict(os.environ, OMP_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    with Path(log_path).open("w") as log:
        process = popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                        env=environment, start_new_session=True)
        while process.poll() is None:
            if STOP_REQUESTED or time.monotonic() - started >= hard_seconds:
                interrupted, timed_out = STOP_REQUESTED, not STOP_REQUESTED
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                break
            try:
                process.wait(timeout=min(1.0, max(0.01, hard_seconds - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                pass
    return {"exit_code": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
            "supervisor_elapsed_seconds": time.monotonic() - started}


def collect_outputs(directory):
    directory = Path(directory)
    return {"result": read_json(directory / "result.json"), "failure": read_json(directory / "failure.json"),
            "progress": read_json(directory / "progress.json"),
            "checkpoint_markers": [str(p.resolve()) for p in sorted(directory.glob("checkpoints/checkpoint-*/resume_contract.json"))]}


class Suite:
    def __init__(self, args, pins):
        self.args, self.pins = args, pins
        self.root = Path(args.root)
        self.summary = {"schema_version": 1, "status": "running", "started_epoch": time.time(),
                        "deadline_epoch": args.deadline_epoch, "final_reserve_seconds": FINAL_RESERVE,
                        "dry_run": args.dry_run, "runs": [], "decisions": [],
                        "selection": "Final validation64 macro PQ, then macro recall; no training-reward selection.",
                        "limitations": "Full-mask engineering validation; no held-out test or promised improvement. Planned equal budgets are not proof of equal completed steps."}

    def save(self):
        write_json(self.root / "experiment_summary.json", self.summary)

    def available(self):
        return max(0, self.args.deadline_epoch - FINAL_RESERVE - time.time() - 40)

    def decide(self, decision):
        self.summary["decisions"].append({"epoch": time.time(), **decision})
        self.save()

    def launch(self, model, name, steps, lr, seed, estimate, benchmark=False):
        cmd = command(self.args, model, name, steps, lr, seed, estimate["soft_training_seconds"], benchmark)
        record = {"name": name, "model_size": model, "planned_steps": steps, "learning_rate": lr,
                  "seed": seed, "benchmark": benchmark, "command": cmd, "forecast": estimate,
                  "output": str(self.root / name), "log": str(self.root / "logs" / f"{name}.log"),
                  "fresh_base": True, "status": "planned", "planned_epoch": time.time()}
        if (self.root / name).exists():
            raise ValueError(f"Run output already exists: {name}")
        verify_pins(self.pins)
        # Validate argparse contract without importing torch or starting a child.
        from nucleus_rl.train import build_parser as train_parser, validate_args
        parsed = train_parser().parse_args(cmd[3:])
        validate_args(parsed)
        self.summary["runs"].append(record)
        write_json(self.root / "plans" / f"{name}.json", record)
        self.save()
        if self.args.dry_run:
            record["status"] = "dry_run_not_launched"
        elif STOP_REQUESTED or self.available() < estimate["hard_seconds"]:
            record.update(status="skipped_insufficient_time_or_signal", exit_code=None)
        else:
            Path(record["log"]).parent.mkdir(parents=True, exist_ok=True)
            record.update(status="running", started_epoch=time.time())
            self.save()
            try:
                record.update(run_child(cmd, self.args.code, record["log"], estimate["hard_seconds"]))
                record.update(collect_outputs(record["output"]))
                record["status"] = (("completed" if record["result"].get("completed_requested_steps")
                                     else "partial_budget_stop") if useful(record) else
                                    "timed_out" if record["timed_out"] else "failed_or_no_useful_update")
            except (OSError, ValueError) as error:
                record.update(status="launch_or_artifact_failure", exit_code=record.get("exit_code"),
                              supervisor_error={"type": type(error).__name__, "message": str(error)})
            record["finished_epoch"] = time.time()
            self.save()
            verify_pins(self.pins)
        self.save()
        return record

    def execute(self):
        timings = {}
        for model in ("7b", "32b"):
            # A finite initial cap, replaced by measured forecasts only after success.
            cap = 1800 if model == "7b" else 2700
            estimate = {"soft_training_seconds": cap - 900, "hard_seconds": cap,
                        "unmeasured_allowance_seconds": 900,
                        "note": "Initial cap reserves 15 minutes for cold load, before/after eval8 and saving."}
            record = self.launch(model, f"benchmark_{model}_20", 20, 1e-5, 42, estimate, benchmark=True)
            timings[model] = None if self.args.dry_run else timing_from_benchmark(record)
        if self.args.dry_run:
            self.decide({"status": "main_stages_deferred_until_measured_benchmarks", "intended_7b_pair": [1000, 1000],
                         "32b_steps_range": [500, 1000], "synthetic_timing_used": False})
            # Validate intended full commands too, clearly marked as unbudgeted templates.
            for model, name, lr in (("7b", "main_7b_lr1e-5_seed42", 1e-5),
                                    ("7b", "main_7b_lr5e-5_seed42", 5e-5),
                                    ("32b", "main_32b_lr1e-5_seed42", 1e-5)):
                self.launch(model, name, 1000, lr, 42,
                            {"soft_training_seconds": 3600, "hard_seconds": 7200, "template_only": True})
            return
        plan = plan_main(timings, self.available())
        self.decide({"stage": "main_budget", "timings": timings, **plan})
        pair = []
        if plan["7b_equal_steps"]:
            n = plan["7b_equal_steps"]
            estimate = forecast(timings["7b"], n)
            for name, lr in (("main_7b_lr1e-5_seed42", 1e-5), ("main_7b_lr5e-5_seed42", 5e-5)):
                pair.append(self.launch("7b", name, n, lr, 42, estimate))
        if plan["32b_steps"]:
            n = affordable_steps(timings["32b"], min(7200, self.available()), minimum=500)
            self.decide({"stage": "32b_recheck", "steps": n, "reason": "Recheck remaining wall time after both 7B arms"})
            if n:
                self.launch("32b", "main_32b_lr1e-5_seed42", n, 1e-5, 42, forecast(timings["32b"], n))
        valid_pair = len(pair) == 2 and all(validation_key(r) is not None for r in pair)
        self.summary["7b_equal_budget_comparison_complete"] = valid_pair
        if valid_pair:
            best = max(pair, key=validation_key)
            n = best["planned_steps"]
            estimate = forecast(timings["7b"], n)
            self.decide({"stage": "seed43_repeat", "selected_configuration": best["name"],
                         "validation_key": validation_key(best), "steps": n,
                         "can_fit": self.available() >= estimate["hard_seconds"],
                         "criterion": "Final validation64 macro PQ, then macro recall; ties choose baseline LR"})
            if self.available() >= estimate["hard_seconds"]:
                self.launch("7b", "repeat_7b_selected_seed43", n, best["learning_rate"], 43, estimate)
        else:
            self.decide({"stage": "seed43_repeat", "skipped": "Both equal-budget 7B arms must finish with valid full64 evaluation before selecting a configuration"})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--root", required=True, help="New experiment-suite output directory")
    p.add_argument("--python", required=True, help="Absolute Python interpreter in the isolated GPU environment")
    p.add_argument("--code", required=True, help="Repository directory containing nucleus_rl and data")
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--dry-run", "--plan-only", dest="dry_run", action="store_true")
    return p


def main(argv=None):
    global STOP_REQUESTED
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time():
        raise ValueError("Deadline must be a future finite Unix timestamp")
    args.deadline_epoch = min(args.deadline_epoch, time.time() + 8 * 3600)
    # Resolving a venv's Python symlink can escape the venv and lose its packages.
    args.python = os.path.abspath(os.path.expanduser(args.python))
    for name in ("root", "code", "sam_checkpoint"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if not Path(args.python).is_file():
        raise ValueError("Python interpreter does not exist")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, args.code)
    suite = Suite(args, {})
    def stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        suite.pins = pin_inputs(args.code, args.sam_checkpoint)
        write_json(root / "input_fingerprints.json", suite.pins)
        suite.execute()
        incomplete = any(r["status"] != "completed" for r in suite.summary["runs"])
        suite.summary["status"] = ("dry_run_complete" if args.dry_run else "interrupted" if STOP_REQUESTED
                                   else "finished_with_failures_or_skips" if incomplete else "finished")
        return 0
    except Exception as error:
        suite.summary.update(status="supervisor_failed", error={"type": type(error).__name__, "message": str(error)})
        return 1
    finally:
        suite.summary["finished_epoch"] = time.time()
        suite.summary["remaining_seconds_to_deadline"] = args.deadline_epoch - time.time()
        suite.save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        print(json.dumps({"status": suite.summary["status"], "summary": str(root / "experiment_summary.json")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
