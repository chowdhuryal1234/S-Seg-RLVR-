"""Vision-LoRA A/B study: completeness gate, pre-registered rule, paired tile bootstrap, support eval.

Read-only, stdlib only. Run after ./pull.sh:
    python3 analyze_ab.py [--results results] [--json results/ab_result.json]

The rule (PLAN.md, written before any job ran) is applied only when all six arms have a completed training run
(completed_requested_steps), a complete 96-tile greedy eval, and the full tile set. Otherwise every number is printed
under a PROVISIONAL banner and no verdict is given.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = Path("/Users/conny/Desktop/RLVR/S-Seg-RLVR-/data/trustmix256_v1/validation.jsonl")
RUNS = {
    ("A", 42): "t256s1-27b896-trustmix256-sft-seed42",
    ("A", 43): "vla-27b896-trustmix256-sft-visionnone-seed43",
    ("A", 44): "vla-27b896-trustmix256-sft-visionnone-seed44",
    ("B", 42): "vlb-27b896-trustmix256-sft-visionall-seed42",
    ("B", 43): "vlb-27b896-trustmix256-sft-visionall-seed43",
    ("B", 44): "vlb-27b896-trustmix256-sft-visionall-seed44",
}
SEEDS = (42, 43, 44)
EXPECTED_SCOPE = {"A": "none", "B": "all"}


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else None


def check_training(summary, arm):
    if summary is None:
        return ["training summary.json missing (still running, failed, or not pulled)"]
    problems = []
    if summary.get("status") != "verified_sft_run":
        problems.append(f"training status {summary.get('status')!r}")
    if summary.get("completed_requested_steps") is not True:
        problems.append(f"training stopped early: {summary.get('optimizer_steps')}/{summary.get('requested_optimizer_steps')}"
                        f" steps, stop_reason {summary.get('stop_reason')!r}")
    scope = summary.get("vision_lora", "none")  # t256s1 predates the flag: language-only
    if scope != EXPECTED_SCOPE[arm]:
        problems.append(f"vision_lora is {scope!r}, expected {EXPECTED_SCOPE[arm]!r}")
    return problems


def check_tiles(ids, manifest_ids, what):
    problems = []
    if len(ids) != len(set(ids)):
        problems.append(f"{what}: duplicate tile ids")
    if set(ids) != manifest_ids:
        problems.append(f"{what}: {len(set(ids) & manifest_ids)}/{len(manifest_ids)} manifest tiles present"
                        f"{', plus unknown ids' if set(ids) - manifest_ids else ''}")
    return problems


def greedy_metrics(records, dataset_of):
    groups = {"ALL": records}
    for record in records:
        groups.setdefault(dataset_of[record["id"]], []).append(record)
    out = {}
    for name, rows in sorted(groups.items()):
        tp, fp, fn = (sum(int(r[key]) for r in rows) for key in ("tp", "fp", "fn"))
        matched_iou = sum(float(r["sq"]) * int(r["tp"]) for r in rows)
        denominator = tp + 0.5 * fp + 0.5 * fn
        out[name] = {"tiles": len(rows), "gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "macro_pq": sum(float(r["pq"]) for r in rows) / len(rows),
                     "pooled_pq": matched_iou / denominator if denominator else None,
                     "invalid": sum(not r["format_valid"] for r in rows),
                     "at_answer_cap": sum(int(r.get("pred_count", 0)) >= int(r.get("answer_cap", 10 ** 9)) for r in rows)}
    return out


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(a, b, ids, statistic, reps, seed):
    """Point estimate and 95% percentile CI of statistic(b_rows) - statistic(a_rows) over tiles resampled together."""
    rng = random.Random(seed)
    point = statistic([b[i] for i in ids]) - statistic([a[i] for i in ids])
    draws = []
    for _ in range(reps):
        sample = rng.choices(ids, k=len(ids))
        draws.append(statistic([b[i] for i in sample]) - statistic([a[i] for i in sample]))
    return {"delta": point, "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
            "share_of_draws_above_0": sum(d > 0 for d in draws) / reps}


def total(key):
    return lambda rows: sum(int(r[key]) for r in rows)


def macro_pq(rows):
    return sum(float(r["pq"]) for r in rows) / len(rows)


def never_found_no_sliver(rows):
    gt = sum(r["gt_no_sliver"] for r in rows)
    return sum(r["never_no_sliver"] for r in rows) / gt if gt else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--json", default=None, help="also write every number to this JSON file")
    parser.add_argument("--reps", type=int, default=10000)
    args = parser.parse_args()
    results = Path(args.results)
    manifest = load_jsonl(Path(args.manifest))
    if manifest is None or any(row["split"] != "validation" for row in manifest):
        raise SystemExit("validation manifest missing or not all validation rows")
    manifest_ids = {row["id"] for row in manifest}
    dataset_of = {row["id"]: row["dataset"] for row in manifest}
    gt_of = {row["id"]: int(row["instance_count"]) for row in manifest}

    report = {"runs": {}, "problems": {}}
    tiles = {}
    support_tiles = {}
    for (arm, seed), name in RUNS.items():
        key = f"{arm}{seed}"
        problems = check_training(load_json(results / name / "summary.json"), arm)
        train = load_json(results / name / "summary.json") or {}
        entry = {"run": name, "training": {k: train.get(k) for k in (
            "optimizer_steps", "requested_optimizer_steps", "completed_requested_steps", "stop_reason",
            "mean_optimizer_step_seconds", "training_elapsed_seconds", "vision_lora")}}
        if train.get("peak_gpu_allocated_bytes"):
            entry["training"]["peak_gpu_allocated_gb"] = round(train["peak_gpu_allocated_bytes"] / 1e9, 1)
            entry["training"]["peak_gpu_reserved_gb"] = round(train["peak_gpu_reserved_bytes"] / 1e9, 1)
        quick = (train.get("after") or {})
        entry["quick48_mean_pq"] = quick.get("mean_pq")
        log = load_jsonl(results / name / "training_log.jsonl") or []
        norms = [r["gradient_norm"] for r in log if r.get("gradient_norm") is not None]
        if norms:
            # sft.py clips the joint norm of all trainable tensors at 1.0. Under B the vision LoRA shares that budget,
            # so a high clipped share means B's language LoRA moved less per step than A's.
            vision = [r["vision_gradient_norm"] for r in log if r.get("vision_gradient_norm") is not None]
            language = [max(0.0, n * n - v * v) ** 0.5 for n, v in
                        ((r["gradient_norm"], r["vision_gradient_norm"]) for r in log
                         if r.get("gradient_norm") is not None and r.get("vision_gradient_norm") is not None)]
            entry["gradients"] = {"steps": len(norms), "median_total_norm": percentile(norms, 0.5),
                                  "share_clipped_at_1": sum(n > 1.0 for n in norms) / len(norms),
                                  "median_vision_norm": percentile(vision, 0.5) if vision else None,
                                  "median_language_norm": percentile(language, 0.5) if language else None}

        evaluation = load_json(results / f"{name}-eval96" / "summary.json")
        records = load_jsonl(results / f"{name}-eval96" / "after" / "completions.jsonl")
        if evaluation is None or evaluation.get("status") != "evaluation_only":
            problems.append("eval96 summary missing or not evaluation_only")
        elif evaluation.get("validation_examples") != len(manifest_ids):
            problems.append(f"eval96 validation_examples {evaluation.get('validation_examples')}")
        if records is None:
            problems.append("eval96 completions.jsonl missing")
        else:
            problems += check_tiles([r["id"] for r in records], manifest_ids, "eval96")
            wrong_gt = [r["id"] for r in records if r["id"] in gt_of and int(r["gt_count"]) != gt_of[r["id"]]]
            if wrong_gt:
                problems.append(f"eval96 gt_count differs from manifest instance_count on {len(wrong_gt)} tiles")
            entry["greedy96"] = greedy_metrics([r for r in records if r["id"] in manifest_ids], dataset_of)
            tiles[key] = {r["id"]: r for r in records}

        support = load_json(results / f"{name}-support96" / "summary.json")
        rows = load_jsonl(results / f"{name}-support96" / "tiles.jsonl")
        if support is None:
            problems.append("support96 summary missing (optional for the rule)")
        else:
            if support.get("status") != "complete":
                problems.append(f"support96 status {support.get('status')!r} ({support.get('tiles_done')} tiles)")
            entry["support96"] = support.get("support")
        if rows is not None:
            problems += check_tiles([r["id"] for r in rows], manifest_ids, "support96")
            support_tiles[key] = {r["id"]: r for r in rows}
        report["runs"][key] = entry
        report["problems"][key] = problems

    blocking = {key: [p for p in found if "optional" not in p and not p.startswith("support96")]
                for key, found in report["problems"].items()}
    final = all(not found for found in blocking.values())
    report["final"] = final
    ids = sorted(manifest_ids)

    pairs = {}
    for seed in SEEDS:
        a, b = tiles.get(f"A{seed}"), tiles.get(f"B{seed}")
        pair = {}
        if a and b and set(a) >= manifest_ids and set(b) >= manifest_ids:
            for label, statistic in (("tp", total("tp")), ("fp", total("fp")), ("macro_pq", macro_pq)):
                pair[label] = paired_bootstrap(a, b, ids, statistic, args.reps, seed)
        sa, sb = support_tiles.get(f"A{seed}"), support_tiles.get(f"B{seed}")
        if sa and sb and set(sa) >= manifest_ids and set(sb) >= manifest_ids:
            pair["never_found_no_sliver"] = paired_bootstrap(sa, sb, ids, never_found_no_sliver, args.reps, seed)
        pairs[seed] = pair
    report["pairs"] = pairs

    tp = {key: report["runs"][key].get("greedy96", {}).get("ALL", {}).get("tp") for key in report["runs"]}
    rule = None
    if all(tp[f"{arm}{seed}"] is not None for arm in "AB" for seed in SEEDS):
        diffs = [tp[f"B{seed}"] - tp[f"A{seed}"] for seed in SEEDS]
        a_values = [tp[f"A{seed}"] for seed in SEEDS]
        a_range = max(a_values) - min(a_values)
        mean_diff = sum(diffs) / len(diffs)
        passed = all(d > 0 for d in diffs) and mean_diff > a_range
        worse = all(d < 0 for d in diffs) and -mean_diff > a_range
        rule = {"tp_diffs_B_minus_A": dict(zip(SEEDS, diffs)), "mean_diff": mean_diff, "A_range": a_range,
                "all_three_positive": all(d > 0 for d in diffs), "passes": passed,
                "outcome": "B passes" if passed else ("B is worse" if worse else "B ties A")}
    report["rule"] = rule

    banner = "FINAL (all six arms complete)" if final else "PROVISIONAL (not a result: some arms incomplete)"
    print(f"# Vision-LoRA A/B on 96 validation tiles: {banner}\n")
    for key, found in report["problems"].items():
        if found:
            print(f"- {key} ({RUNS[(key[0], int(key[1:]))]}): " + "; ".join(found))
    print("\n| arm | steps | s/step | peak GB | TP | FP | FN | recall | precision | macro PQ | pooled PQ | invalid |"
          " never-found (no sliver) | per-sample recall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, entry in report["runs"].items():
        g = entry.get("greedy96", {}).get("ALL", {})
        s = (entry.get("support96") or {}).get("ALL", {})
        t = entry["training"]
        fmt = lambda v, d=3: "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
        print(f"| {key} | {fmt(t.get('optimizer_steps'))} | {fmt(t.get('mean_optimizer_step_seconds'), 1)} |"
              f" {fmt(t.get('peak_gpu_allocated_gb'), 1)} | {fmt(g.get('tp'))} | {fmt(g.get('fp'))} | {fmt(g.get('fn'))} |"
              f" {fmt(g.get('recall'))} | {fmt(g.get('precision'))} | {fmt(g.get('macro_pq'))} | {fmt(g.get('pooled_pq'))} |"
              f" {fmt(g.get('invalid'))} | {fmt(s.get('never_found_no_sliver'))} | {fmt(s.get('per_sample_recall'))} |")
    print("\nGradients (pre-clip; clip = joint norm 1.0):")
    for key, entry in report["runs"].items():
        g = entry.get("gradients")
        if g:
            extra = (f", vision {g['median_vision_norm']:.3f} / language {g['median_language_norm']:.3f}"
                     if g["median_vision_norm"] is not None else "")
            print(f"- {key}: median total {g['median_total_norm']:.3f}, clipped {g['share_clipped_at_1']:.1%}"
                  f" of {g['steps']} steps{extra}")
    print("\nPer dataset (TP / FP / FN):")
    for key, entry in report["runs"].items():
        parts = [f"{name} {m['tp']}/{m['fp']}/{m['fn']}" for name, m in entry.get("greedy96", {}).items() if name != "ALL"]
        if parts:
            print(f"- {key}: " + ", ".join(parts))
    print("\nPaired tile bootstrap, B - A, 95% CI:")
    for seed, pair in pairs.items():
        if pair:
            print(f"- seed {seed}: " + "; ".join(
                f"{label} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for label, v in pair.items()))
    if rule:
        print(f"\nPre-registered rule: TP diffs {rule['tp_diffs_B_minus_A']}, mean {rule['mean_diff']:+.1f},"
              f" A seed range {rule['A_range']} -> {rule['outcome'] if final else 'no verdict until FINAL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
