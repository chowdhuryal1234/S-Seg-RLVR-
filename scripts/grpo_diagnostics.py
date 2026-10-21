"""Two diagnostics for the GRPO-from-SFT experiments. Run from the repository root.

oracle : correct GT boxes + interior points -> frozen SAM2 on the 64 validation crops,
         i.e. the PQ ceiling of the frozen segmenter under this prompt format.
merge  : evaluate the SFT adapter unmerged vs merged into bf16 weights on the first
         --eval-limit validation crops, with the exact SFT validation code.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def oracle(args):
    import numpy as np
    from PIL import Image
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.rewards import ObjectPrompt, instance_metrics
    from nucleus_rl.segmenter import FrozenSAM2
    from scripts.prompt_diagnostic import oracle_prompt_objects
    rows = read_manifest(args.eval_manifest)
    results = {}
    for name, config, checkpoint in (("tiny", "configs/sam2.1/sam2.1_hiera_t.yaml", args.sam_tiny),
                                     ("large", "configs/sam2.1/sam2.1_hiera_l.yaml", args.sam_large)):
        if not checkpoint or not Path(checkpoint).is_file():
            continue
        sam = FrozenSAM2(checkpoint, config, device=args.device)
        records = []
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            reference = np.asarray(Image.open(row["mask_path"]))
            objects = [ObjectPrompt(box=tuple(item["box"]), point=tuple(item["point"]))
                       for item in oracle_prompt_objects(reference, padding=0, max_objects=8)]
            instances = sam.predict(image, objects, image_key=row["image_path"])
            records.append({"id": row["id"], **instance_metrics(instances, reference)})
        results[name] = {"checkpoint": checkpoint, "crops": len(records),
                         "macro_pq": sum(r["pq"] for r in records) / len(records),
                         "tp": sum(r["tp"] for r in records), "gt": sum(r["gt_count"] for r in records),
                         "count_mae": sum(r["count_error"] for r in records) / len(records), "records": records}
        print(name, {k: v for k, v in results[name].items() if k != "records"}, flush=True)
    return results


def merge(args):
    from peft import PeftModel
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.segmenter import FrozenSAM2
    from nucleus_rl.sft import evaluate_policy, load_model
    rows = read_manifest(args.eval_manifest)[:args.eval_limit]
    settings = SimpleNamespace(coordinate_frame="normalized1000", max_objects=8, max_completion_length=1024)
    load_args = SimpleNamespace(model=args.model, revision=args.revision, min_pixels=50176, max_pixels=200704, seed=42)
    sam = FrozenSAM2(args.sam_tiny, "configs/sam2.1/sam2.1_hiera_t.yaml", device="cpu")
    model, processor = load_model(load_args)
    unmerged = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    output = Path(args.output)
    first = evaluate_policy(unmerged, processor, rows, sam, output / "unmerged", settings)
    merged = unmerged.merge_and_unload()
    second = evaluate_policy(merged, processor, rows, sam, output / "merged", settings)
    texts = [[json.loads(line)["text"] for line in (output / arm / "completions.jsonl").read_text().splitlines()]
             for arm in ("unmerged", "merged")]
    result = {"unmerged_pq": first["mean_pq"], "merged_pq": second["mean_pq"],
              "unmerged_tp": first["instance_diagnostics"]["total_tp"],
              "merged_tp": second["instance_diagnostics"]["total_tp"],
              "identical_texts": sum(a == b for a, b in zip(*texts)), "crops": len(rows)}
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("oracle", "merge"))
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam-tiny", required=True)
    parser.add_argument("--sam-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--eval-limit", type=int, default=8)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    result = oracle(args) if args.mode == "oracle" else merge(args)
    (Path(args.output) / f"{args.mode}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
