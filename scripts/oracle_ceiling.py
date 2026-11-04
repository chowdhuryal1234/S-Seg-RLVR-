"""Ground-truth-prompt SAM oracle ceiling for any prepared tile manifest.

Every visible GT instance becomes one tight half-open box plus the foreground
pixel nearest its centroid (the exact SFT target construction in
``nucleus_rl.sft_data.instance_prompt_objects``, sorted top-to-bottom). The
prompts go through the frozen SAM2 call used in training and evaluation
(``FrozenSAM2.predict``), and the resulting instance map is scored against the
full GT map with ``nucleus_rl.rewards.instance_metrics``.

``--max-objects 8`` keeps only the first eight prompts in target order while
still scoring against every GT instance, i.e. the ceiling of a policy that
emits at most eight objects. ``all`` prompts every GT instance.

The numbers use privileged GT information: they bound the frozen segmenter
under this prompt format and are not learned-policy performance.

Example (MoNuSeg sparse validation, all objects plus cap 8):
    python scripts/oracle_ceiling.py \
        --manifest monuseg_sparse=data/monuseg_official30_grpo_v1/validation.jsonl \
        --max-objects all 8 --out ../local-notes/next-phase-20261104/oracle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

SCOPE = "ORACLE GT-derived prompts -> frozen SAM2; privileged information; not learned-policy performance"
DEFAULT_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
ALLOWED_SPLITS = ("train", "validation")


def parse_cap(value: str) -> int | None:
    """``all``/``none`` -> None (every GT instance); otherwise a positive integer."""
    if value.lower() in ("all", "none"):
        return None
    try:
        cap = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--max-objects must be a positive integer or 'all': {value}") from error
    if cap <= 0:
        raise argparse.ArgumentTypeError("--max-objects must be positive")
    return cap


def cap_label(cap: int | None) -> str:
    return "all" if cap is None else f"cap{cap}"


def parse_manifest_spec(spec: str) -> tuple[str, Path]:
    """``name=path`` or bare ``path`` (name = '<parent dir>/<stem>')."""
    name, sep, path = spec.partition("=")
    if not sep:
        path = spec
        name = f"{Path(spec).parent.name}/{Path(spec).stem}"
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Bad --manifest spec: {spec}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, action="append", type=parse_manifest_spec,
                        help="JSONL manifest, optionally NAME=PATH; repeat for several sets")
    parser.add_argument("--out", required=True, help="Output directory (outside git); summary merges by set")
    parser.add_argument("--max-objects", nargs="+", type=parse_cap, default=[None],
                        help="One or more caps: positive integers or 'all' (default: all GT instances)")
    parser.add_argument("--sam-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8, help="CPU intra-op threads")
    parser.add_argument("--limit", type=int, default=None, help="First N rows per manifest (smoke tests)")
    return parser


def check_rows(rows: list[dict[str, Any]]) -> None:
    """Refuse anything that is not a prepared train/validation tile."""
    bad = sorted({str(row.get("split")) for row in rows if row.get("split") not in ALLOWED_SPLITS})
    if bad:
        raise ValueError(f"Oracle ceiling accepts only {ALLOWED_SPLITS} rows; found split(s) {bad}")


def oracle_objects(mask: Any, cap: int | None = None) -> list[dict[str, list[int]]]:
    """SFT target objects for every visible instance, then the first ``cap``."""
    import numpy as np
    from nucleus_rl.sft_data import instance_prompt_objects

    labels = np.asarray(mask)
    count = len(np.unique(labels[labels > 0]))
    if count == 0:
        return []
    objects = instance_prompt_objects(labels, max_objects=count)
    return objects if cap is None else objects[:cap]


def load_mask(row: dict[str, Any]) -> Any:
    """PNG or .npy instance map, as in sft_data; refuse a count that disagrees with the manifest."""
    import numpy as np
    from PIL import Image

    path = Path(row["mask_path"])
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as handle:
            mask = np.asarray(handle)
    mask = mask.astype(np.int64)
    count = int(np.count_nonzero(np.unique(mask)))
    if "instance_count" in row and row["instance_count"] != count:
        raise ValueError(f"{row['id']}: mask has {count} instances, manifest says {row['instance_count']}")
    return mask


def score_row(segmenter: Any, image: Any, mask: Any, objects: list[dict], image_key: str) -> dict[str, Any]:
    from nucleus_rl.rewards import ObjectPrompt, instance_metrics

    prompts = [ObjectPrompt(box=tuple(obj["box"]), point=tuple(obj["point"])) for obj in objects]
    instances = segmenter.predict(image, prompts, image_key=image_key)
    metrics = instance_metrics(instances, mask)
    return {**metrics, "matched_iou_sum": metrics["sq"] * metrics["tp"], "prompted": len(prompts)}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro means over tiles (the repo's mean_pq convention) plus pooled counts."""
    n = len(records)
    tp, fp, fn = (sum(r[key] for r in records) for key in ("tp", "fp", "fn"))
    matched = sum(r["matched_iou_sum"] for r in records)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "tiles": n,
        "gt": sum(r["gt_count"] for r in records),
        "prompted": sum(r["prompted"] for r in records),
        "pq": sum(r["pq"] for r in records) / n,
        "sq": sum(r["sq"] for r in records) / n,
        "dq": sum(r["dq"] for r in records) / n,
        "tp": tp, "fp": fp, "fn": fn,
        "pooled_pq": matched / denominator if denominator else 1.0,
        "pooled_sq": matched / tp if tp else 0.0,
        "pooled_dq": tp / denominator if denominator else 1.0,
        "count_mae": sum(r["count_error"] for r in records) / n,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
    }


def run_manifest(name: str, rows: list[dict], segmenter: Any, caps: list[int | None],
                 out: Path) -> dict[str, dict[str, Any]]:
    """Score every row under every cap; one SAM image encoding per row."""
    import numpy as np
    from PIL import Image

    records: dict[str, list[dict]] = {cap_label(cap): [] for cap in caps}
    started = time.monotonic()
    for row in rows:
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB")
        mask = load_mask(row)
        for cap in caps:
            objects = oracle_objects(mask, cap)
            result = score_row(segmenter, image, mask, objects, row["image_path"])
            records[cap_label(cap)].append({"id": row["id"], **result})
    seconds = time.monotonic() - started
    results = {}
    slug = name.replace("/", "__")
    for label, rows_scored in records.items():
        path = out / f"{slug}__{label}.jsonl"
        path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows_scored))
        results[f"{name}|{label}"] = {"set": name, "cap": label, **aggregate(rows_scored),
                                      "records_path": str(path), "wall_seconds_all_caps": seconds}
    return results


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    header = ("| set | prompts | tiles | GT | PQ | SQ | DQ | TP | FP | FN | recall | count MAE | pooled PQ |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [
        f"| {r['set']} | {r['cap']} | {r['tiles']} | {r['gt']} | {r['pq']:.3f} | {r['sq']:.3f} | "
        f"{r['dq']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['recall']:.3f} | "
        f"{r['count_mae']:.2f} | {r['pooled_pq']:.3f} |"
        for _, r in sorted(results.items())
    ]
    note = ("\nPQ/SQ/DQ are macro means over tiles (repo `mean_pq` convention); pooled PQ sums "
            "matched IoU and TP/FP/FN over all tiles. `capN` prompts only the first N GT objects in "
            "SFT target order (top-to-bottom) but scores against all GT.\n")
    return f"# GT-prompt SAM oracle ceilings\n\n{SCOPE}\n\n" + header + "\n".join(lines) + "\n" + note


def write_summary(out: Path, new_results: dict[str, dict[str, Any]], run_info: dict[str, Any]) -> dict:
    """Merge into an existing summary.json by set|cap key, then re-render the table."""
    path = out / "summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else {"scope": SCOPE, "results": {}, "runs": []}
    summary["results"].update(new_results)
    summary["runs"].append(run_info)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "summary.md").write_text(markdown_table(summary["results"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    caps = list(dict.fromkeys(args.max_objects))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.segmenter import FrozenSAM2

    sets = []
    for name, path in args.manifest:
        rows = read_manifest(path)[:args.limit] if args.limit else read_manifest(path)
        if not rows:
            parser.error(f"Empty manifest: {path}")
        check_rows(rows)
        sets.append((name, path, rows))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    load_started = time.monotonic()
    segmenter = FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.device)
    load_seconds = time.monotonic() - load_started
    results = {}
    for name, path, rows in sets:
        part = run_manifest(name, rows, segmenter, caps, out)
        for key, value in part.items():
            value["manifest"] = str(Path(path).resolve())
            print(json.dumps({k: v for k, v in value.items() if k != "records_path"}), flush=True)
        results.update(part)
    run_info = {
        "argv": sys.argv[1:] if argv is None else argv, "device": args.device,
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "sam_load_seconds": load_seconds, "sam_freeze_audit": segmenter.verify_unchanged(),
        "keys": sorted(results),
    }
    write_summary(out, results, run_info)
    print(markdown_table(results))


if __name__ == "__main__":
    main()
