"""CPU-only preflight for the trustmix lanes (login node, no model loading, no GPU).

For every slurm/*.sbatch it evaluates the "# >>> lane" block with env.sh in bash, takes the exact
argument arrays (ARGS, ARGS_A1, ...) the job will pass, and then, with the trainers' own code:
  - builds the trainer's argparse and runs validate_args (reward spec, wpq weights, limits);
  - reads both manifests (read_manifest resolves ../ paths) and runs validate_sft_splits exactly as main();
  - checks --cohort-protocol is absent, outputs do not exist, SAM checkpoint and the HF snapshot exist;
  - GRPO: runs check_init_adapter; an init that does not exist yet (S2/S4 checkpoint-256) is replaced
    by an existing H2 checkpoint to exercise the code path; eval adapters are checked against the
    init_adapter_sha256 the trainer will require;
  - simulates the eval/checkpoint schedule and diffs the recipe against H2 / G7 arguments.json;
  - checks #SBATCH resources (<=12 h; 2 CPU/32 GB, or 4 CPU/64 GB for two co-located processes).
On success it writes receipts/preflight-<time>.json and receipts/preflight.sha256 (checked by submit.sh).

    $PY tools/preflight.py            (from $W; env.sh is sourced by bash)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

LANE_DUMP = r'''
set -euo pipefail
source "$1"
eval "$(sed -n '/^# >>> lane/,/^# <<< lane/p' "$2")"
for name in $(compgen -A arrayvar | grep -E '^ARGS(_[A-Z][0-9]+)?$' | sort); do
  eval "printf '%s\\0' \"@@\$name\" \"\${MODULE:-}\" \"\${INIT:-}\" \"\${${name}[@]}\""
done
'''
H2_KEYS_IGNORED = {"output", "manifest", "eval_manifest", "cohort_protocol", "sam_checkpoint", "resume_from_checkpoint"}
G7_KEYS_IGNORED = H2_KEYS_IGNORED | {"init_adapter", "reward_weights"}


def env_from(env_file):
    out = subprocess.run(["bash", "-c", 'set -a; source "$1"; env -0', "_", env_file],
                         check=True, capture_output=True).stdout.decode()
    return dict(item.split("=", 1) for item in out.split("\0") if "=" in item)


def lane_invocations(env_file, script):
    raw = subprocess.run(["bash", "-c", LANE_DUMP, "_", env_file, str(script)], check=True,
                         capture_output=True).stdout.decode().split("\0")[:-1]
    calls, current = [], None
    for item in raw:
        if item.startswith("@@"):
            current = {"array": item[2:], "argv": []}
            calls.append(current)
            current["_header"] = 2
        elif current["_header"] == 2:
            current["module"], current["_header"] = item, 1
        elif current["_header"] == 1:
            current["init_var"], current["_header"] = item, 0
        else:
            current["argv"].append(item)
    for call in calls:
        call.pop("_header")
    return calls


def sbatch_directives(script):
    values = {}
    for line in Path(script).read_text().splitlines():
        match = re.match(r"#SBATCH\s+--([a-z-]+)=(\S+)", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def hours(limit):
    parts = [int(p) for p in limit.replace("-", ":").split(":")]
    if "-" in limit:
        days, h, m, s = parts
        return days * 24 + h + m / 60 + s / 3600
    h, m, s = parts
    return h + m / 60 + s / 3600


def schedule(args, module):
    """Steps that get an in-run evaluation and a checkpoint, following the trainers' loop conditions."""
    evals, saves = [], []
    for step in range(1, args.max_steps + 1):
        due = bool(args.eval_every and step % args.eval_every == 0 and step < args.max_steps)
        if step % args.save_steps == 0 or due or step == args.max_steps:
            saves.append(step)
        if due:
            evals.append(step)
    before = not args.skip_before_eval
    return {"before_eval": before, "periodic_evals": evals, "final_eval": args.max_steps,
            "checkpoints_written": saves,
            "checkpoints_kept": saves if module == "nucleus_rl.sft" else sorted(set(evals) | {saves[-1]})}


def recipe_diff(args, reference, ignored):
    mine = vars(args)
    changed = {k: {"ref": reference[k], "new": mine[k]} for k in sorted(reference)
               if k in mine and k not in ignored and reference[k] != mine[k]}
    new_keys = {k: mine[k] for k in sorted(mine) if k not in reference and k not in ignored}
    return {"changed": changed, "keys_not_in_reference": new_keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--env-file")
    parser.add_argument("--scripts", nargs="*", help="sbatch stems to check (default: all)")
    parser.add_argument("--no-receipt", action="store_true")
    options = parser.parse_args()
    workspace = Path(options.workspace).resolve()
    env_file = Path(options.env_file or workspace / "env.sh").resolve()
    env = env_from(env_file)
    os.environ.update({k: env[k] for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if k in env})
    code = Path(env["W"]) / "code"
    sys.path.insert(0, str(code))
    os.chdir(code)
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.sft import MAX_OBJECTS, verify_sft_checkpoint
    from nucleus_rl.sft_data import validate_sft_splits
    from nucleus_rl.training_budget import file_digest

    old = Path(env["OLD"])
    h2_args = json.loads((old / "runs/h2-27b-d4-sft448/arguments.json").read_text())
    g7_args = json.loads((old / "runs/g7-27b448c128-pq/arguments.json").read_text())
    stand_ins = [old / "runs/h2-27b-d4-sft448/checkpoints" / f"checkpoint-{s}" for s in (256, 128, 384, 512)]
    stand_in = next((p for p in stand_ins if (p / "adapter_model.safetensors").is_file()), None)
    snapshot = Path(env["HF_HOME"]) / "hub" / "models--Qwen--Qwen3.6-27B" / "snapshots" / env["REV_27B"]

    scripts = sorted((workspace / "slurm").glob("*.sbatch"))
    if options.scripts:
        scripts = [s for s in scripts if s.stem in options.scripts]
    report = {"workspace": str(workspace), "code_commit": (code / "COMMIT").read_text().strip(),
              "python": sys.version.split()[0], "hf_snapshot": str(snapshot), "hf_snapshot_present": snapshot.is_dir(),
              "sam_checkpoint_present": Path(env["SAM_CKPT"]).is_file(), "stand_in_init": str(stand_in), "lanes": {}}
    failures = []
    if not report["hf_snapshot_present"]:
        failures.append(f"HF snapshot missing: {snapshot}")
    if not report["sam_checkpoint_present"]:
        failures.append(f"SAM checkpoint missing: {env['SAM_CKPT']}")
    split_cache = {}
    outputs_seen = {}
    for script in scripts:
        lane = {"sbatch": sbatch_directives(script), "calls": []}
        report["lanes"][script.stem] = lane
        directives = lane["sbatch"]
        calls = lane_invocations(str(env_file), script)
        co_located = len(calls) > 1
        want = ("4", "64G") if co_located else ("2", "32G")
        if (directives.get("cpus-per-task"), directives.get("mem")) != want:
            failures.append(f"{script.stem}: resources {directives.get('cpus-per-task')}/{directives.get('mem')}, want {want}")
        if hours(directives.get("time", "99:00:00")) > 12:
            failures.append(f"{script.stem}: time limit {directives.get('time')} > 12 h")
        if directives.get("job-name") != f"sseg-tm-{script.stem}":
            failures.append(f"{script.stem}: job-name {directives.get('job-name')} must be sseg-tm-{script.stem}")
        if directives.get("gres") != "gpu:nvidia_h200:1" or directives.get("partition") != "gpu_h200,seas_gpu":
            failures.append(f"{script.stem}: gres/partition {directives.get('gres')} {directives.get('partition')}")
        for call in calls:
            entry = {"array": call["array"], "module": call["module"], "argv": call["argv"], "checks": []}
            lane["calls"].append(entry)
            try:
                argv = call["argv"]
                if "--cohort-protocol" in argv:
                    raise ValueError("--cohort-protocol must not be passed on trustmix data")
                module = importlib.import_module(call["module"])
                args = module.build_parser().parse_args(argv)
                module.validate_args(args)
                entry["checks"].append("argparse+validate_args")
                key = (args.manifest, args.eval_manifest, args.max_objects)
                if key not in split_cache:
                    started = time.monotonic()
                    rows, validation = read_manifest(args.manifest), read_manifest(args.eval_manifest)
                    audit = validate_sft_splits(rows, validation, verify_files=True, max_objects=args.max_objects,
                                                validation_max_objects=MAX_OBJECTS)
                    train_counts, val_counts = {}, {}
                    for r in rows:
                        train_counts[r.get("dataset", "monuseg")] = train_counts.get(r.get("dataset", "monuseg"), 0) + 1
                    for r in validation:
                        val_counts[r.get("dataset", "monuseg")] = val_counts.get(r.get("dataset", "monuseg"), 0) + 1
                    split_cache[key] = {"train_rows": len(rows), "validation_rows": len(validation),
                                        "train_datasets": train_counts, "validation_datasets": val_counts,
                                        "max_validation_instances": max(r.get("instance_count", 0) for r in validation),
                                        "audit_status": audit["status"], "files_hashed": len(audit["all_files_sha256"]),
                                        "seconds": round(time.monotonic() - started, 1)}
                entry["splits"] = split_cache[key]
                entry["checks"].append("read_manifest+validate_sft_splits")
                output = Path(args.output)
                if output.exists():
                    raise ValueError(f"output already exists: {output}")
                if str(output) in outputs_seen:
                    raise ValueError(f"output also used by {outputs_seen[str(output)]}")
                outputs_seen[str(output)] = f"{script.stem}:{call['array']}"
                if not output.parent.is_dir() or not os.access(output.parent, os.W_OK):
                    raise ValueError(f"output parent not writable: {output.parent}")
                if args.revision != env["REV_27B"] or args.model != "Qwen/Qwen3.6-27B":
                    raise ValueError("model/revision differ from the H2/G7 base")
                if call["module"] == "nucleus_rl.sft":
                    entry["schedule"] = schedule(args, call["module"])
                    if 256 not in entry["schedule"]["checkpoints_written"]:
                        raise ValueError("SFT lane does not write checkpoint-256 (the GRPO gate)")
                    entry["recipe_diff_vs_H2"] = recipe_diff(args, h2_args, H2_KEYS_IGNORED)
                else:
                    from nucleus_rl.grpo import check_init_adapter
                    init = Path(args.init_adapter)
                    if (init / "adapter_model.safetensors").is_file():
                        entry["init_adapter"] = check_init_adapter(args)
                        entry["init_mode"] = "real"
                        if (init / "sft_checkpoint.json").is_file():
                            try:
                                verify_sft_checkpoint(init)
                                entry["checks"].append("verify_sft_checkpoint(init)")
                            except Exception as error:
                                # eval-only needs only the adapter files that check_init_adapter read;
                                # a training init must pass the full checkpoint verification.
                                if not args.eval_only:
                                    raise
                                entry["warning"] = f"init checkpoint marker does not verify ({error}); adapter files are present"
                    else:
                        if not args.eval_only and not re.search(r"/runs/s[24]-[^/]+/checkpoints/checkpoint-256$", str(init)):
                            raise ValueError(f"init adapter missing and not a pending S2/S4 checkpoint-256: {init}")
                        if stand_in is None:
                            raise ValueError("no H2 checkpoint available to exercise check_init_adapter")
                        probe = argparse.Namespace(**{**vars(args), "init_adapter": str(stand_in)})
                        entry["init_adapter"] = check_init_adapter(probe)
                        entry["init_mode"] = f"pending; code path exercised with stand-in {stand_in}"
                    entry["checks"].append("check_init_adapter")
                    if args.eval_adapter:
                        adapter = Path(args.eval_adapter)
                        if (adapter / "adapter_model.safetensors").is_file():
                            coordinates = json.loads((adapter / "coordinate_config.json").read_text())
                            if entry["init_mode"] != "real":
                                raise ValueError("eval adapter exists but its init does not")
                            if coordinates.get("init_adapter_sha256") != entry["init_adapter"]["adapter_sha256"]:
                                raise ValueError(f"--eval-adapter {adapter} was trained from a different init than {init}")
                            entry["eval_adapter"] = {"path": str(adapter), "init_sha256_matches": True}
                            entry["checks"].append("eval_adapter init sha256 match")
                        else:
                            entry["eval_adapter"] = {"path": str(adapter), "pending": "not trained yet"}
                    if args.eval_only:
                        entry["eval"] = {"manifest": args.eval_manifest, "max_objects": args.max_objects,
                                         "eval_max_completion_length": args.eval_max_completion_length,
                                         "eval_sam_device": args.eval_sam_device}
                    else:
                        entry["schedule"] = schedule(args, call["module"])
                        entry["reward_weights"] = args.reward_weights
                        entry["wpq_weights"] = [args.wpq_fp_weight, args.wpq_fn_weight]
                        entry["recipe_diff_vs_G7"] = recipe_diff(args, g7_args, G7_KEYS_IGNORED)
                        if call["init_var"] != args.init_adapter:
                            raise ValueError("INIT variable and --init-adapter differ")
                entry["ok"] = True
            except Exception as error:  # report every lane, then fail
                entry["ok"] = False
                entry["error"] = f"{type(error).__name__}: {error}"
                failures.append(f"{script.stem}:{call['array']}: {entry['error']}")
    report["failures"] = failures
    stamp = time.strftime("%Y%m%d-%H%M%S")
    text = json.dumps(report, indent=1, default=str)
    for stem, lane in report["lanes"].items():
        for call in lane["calls"]:
            status = "ok " if call.get("ok") else "FAIL"
            extra = call.get("error") or call.get("init_mode") or ""
            print(f"{status} {stem:8s} {call['array']:8s} {call['module']:16s} {extra}")
            if "schedule" in call:
                print(f"       schedule: {call['schedule']}")
            for key in ("recipe_diff_vs_H2", "recipe_diff_vs_G7"):
                if key in call:
                    print(f"       {key}: {json.dumps(call[key]['changed'])}")
            if "eval_adapter" in call:
                print(f"       eval_adapter: {call['eval_adapter']}")
    for key, value in split_cache.items():
        print(f"splits {Path(key[0]).parent.name}/{Path(key[0]).name} vs {Path(key[1]).parent.name}/{Path(key[1]).name} cap {key[2]}: {value}")
    print(f"hf_snapshot_present={report['hf_snapshot_present']} sam_checkpoint_present={report['sam_checkpoint_present']}")
    if not options.no_receipt:
        receipts = Path(env["W"]) / "receipts"
        (receipts / f"preflight-{stamp}.json").write_text(text)
        if not failures:
            tracked = [env_file, *sorted((workspace / "slurm").iterdir()), code / "COMMIT"]
            lines = [f"{file_digest(p)}  {os.path.relpath(p, env['W'])}" for p in tracked if p.is_file()]
            (receipts / "preflight.sha256").write_text("\n".join(lines) + "\n")
        print(f"report: {receipts / f'preflight-{stamp}.json'}")
    if failures:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
