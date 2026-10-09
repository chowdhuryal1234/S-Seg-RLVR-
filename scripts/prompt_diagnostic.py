"""Oracle GT-derived prompt diagnostic on at most four TRAIN crops.

Compares tight and five-pixel-padded instance boxes with identical interior
ground-truth points. This loads only a frozen SAM2, not a VLM, and performs no
optimization. The results are not automated or held-out model performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


PADDING_PIXELS = 5
SCOPE = "ORACLE GT-derived prompts; training crops only; no VLM; no optimization; not held-out performance"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training JSONL manifest only")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output", required=True, help="A new empty directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-images", type=int, default=4, help="First 1–4 rows, in manifest order")
    parser.add_argument("--threads", type=int, default=8, help="CPU intra-op threads, at most eight")
    parser.add_argument("--include-full-image-boxes", action="store_true",
                        help="Also use one full-image box per GT point, in separate SAM calls")
    return parser


def validate_args(args):
    if not 1 <= args.max_images <= 4:
        raise ValueError("--max-images must be between one and four")
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads must be between one and eight")


def select_training_rows(rows, max_images=4):
    """Reject mixed manifests, then use the first rows without cherry-picking."""
    if not 1 <= max_images <= 4:
        raise ValueError("Select one to four training crops")
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Oracle diagnostic requires a nonempty, entirely training-only manifest")
    return rows[:max_images]


def oracle_prompt_objects(instance_map, *, padding=0, full_image_boxes=False, max_objects=8):
    """One box and one actual foreground pixel per unique positive GT ID.

    Choose the visible-instance pixel nearest its arithmetic centroid. Ties use
    row-major order. A centroid outside a concave object is never used directly.
    Box coordinates are crop-local half-open XYXY. Truncated visible instances
    remain included, matching this engineering dataset's annotation convention.
    """
    import numpy as np

    labels = np.asarray(instance_map)
    if labels.ndim != 2 or min(labels.shape) == 0 or labels.dtype.kind not in "biuf":
        raise ValueError("Expected a nonempty 2D numeric instance map")
    if not np.all(np.isfinite(labels)) or np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("Ground-truth IDs must be finite nonnegative integers")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a nonnegative integer")
    instance_ids = np.unique(labels[labels > 0])
    if not 1 <= len(instance_ids) <= max_objects:
        raise ValueError(f"Diagnostic requires one to {max_objects} visible instances")
    height, width = labels.shape
    objects = []
    for instance_id in instance_ids:
        pixels_yx = np.argwhere(labels == instance_id)
        center_yx = pixels_yx.mean(axis=0)
        squared_distances = ((pixels_yx - center_yx) ** 2).sum(axis=1)
        point_y, point_x = pixels_yx[int(np.argmin(squared_distances))]
        lo_y, lo_x = pixels_yx.min(axis=0)
        hi_y, hi_x = pixels_yx.max(axis=0) + 1
        box = [0, 0, width, height] if full_image_boxes else [
            max(0, int(lo_x) - padding), max(0, int(lo_y) - padding),
            min(width, int(hi_x) + padding), min(height, int(hi_y) + padding),
        ]
        objects.append({"box": box, "point": [int(point_x), int(point_y)]})
    return objects


class RawRecordingSegmenter:
    """Delegate to FrozenSAM2 while recording its actual predictor outputs.

    The temporary predictor hook is restored even on error. Only use this
    wrapper sequentially: it is intentionally not a concurrent inference API.
    No second segmentation pass is required to obtain raw masks/logits.
    """

    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.last_raw = None
        self.last_predict_seconds = None

    def predict(self, image, objects, image_key=None):
        import numpy as np

        original_predict = self.segmenter.predictor.predict
        captured = []

        def record(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            captured.append(tuple(np.asarray(part).copy() for part in result))
            return result

        self.segmenter.predictor.predict = record
        self.last_raw = None
        started = time.monotonic()
        try:
            instances = self.segmenter.predict(image, objects, image_key=image_key)
        finally:
            self.last_predict_seconds = time.monotonic() - started
            self.segmenter.predictor.predict = original_predict
        if len(captured) != len(objects):
            raise RuntimeError("Expected one raw SAM predictor result per oracle object")
        width, height = image.size
        self.last_raw = {
            "masks": np.concatenate([parts[0] for parts in captured], axis=0)
            if captured else np.zeros((0, height, width), dtype=bool),
            "scores": np.concatenate([parts[1].reshape(-1) for parts in captured])
            if captured else np.zeros((0,), dtype=float),
            "low_resolution_logits": np.concatenate([parts[2] for parts in captured], axis=0)
            if captured else np.zeros((0, 0, 0), dtype=float),
        }
        return instances


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _gt_overlay(row):
    import numpy as np
    from PIL import Image

    with Image.open(row["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(row["mask_path"]) as image:
        labels = np.asarray(image)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rgb[boundary] = (255, 60, 60)
    return Image.fromarray(rgb)


def save_contact_sheet(rows, records, variants, destination):
    """Label oracle provenance and show every selected crop, without cherry-picking."""
    from PIL import Image, ImageDraw

    cell_width, image_size, row_height, header = 300, 256, 338, 106
    sheet = Image.new("RGB", ((len(variants) + 1) * cell_width, header + len(rows) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, font, small = _font(22), _font(16), _font(13)
    draw.text((14, 10), "ORACLE PROMPT DIAGNOSTIC - TRAIN ONLY", fill="black", font=title_font)
    draw.text((14, 42), "GT-derived boxes/points; frozen SAM2; no VLM or training. Red=GT, green=prediction.", fill="black", font=font)
    names = {"tight": "Tight GT box + GT point", "padded_5px": "+5px box; same GT point",
             "full_image": "Full-image box per GT point"}
    headings = ["Reference annotation"] + [names[name] for name in variants]
    for column, heading in enumerate(headings):
        draw.text((column * cell_width + 14, 76), heading, fill="black", font=font)
    lookup = {(record["id"], record["variant"]): record for record in records}
    for row_index, row in enumerate(rows):
        top = header + row_index * row_height
        draw.text((14, top), row["id"], fill="black", font=small)
        images = [_gt_overlay(row)]
        for variant in variants:
            with Image.open(lookup[(row["id"], variant)]["overlay_path"]) as overlay:
                images.append(overlay.copy())
        for column, image in enumerate(images):
            resized = image.resize((image_size, image_size), Image.Resampling.NEAREST)
            sheet.paste(resized, (column * cell_width + 14, top + 22))
            if column:
                record = lookup[(row["id"], variants[column - 1])]
                label = f"IoU {record['seg_reward']:.3f} | PQ {record['pq']:.3f}"
                detail = f"Pred {record['pred_count']} / GT {record['gt_count']} objects"
            else:
                label = f"Visible GT IDs: {row['diagnostic_gt_count']}"
                detail = f"Border-truncated IDs: {len(row.get('truncated_instance_ids', []))}"
            draw.text((column * cell_width + 14, top + 282), label, fill="black", font=font)
            draw.text((column * cell_width + 14, top + 305), detail, fill="black", font=small)
    sheet.save(destination)


def run_diagnostic(rows, segmenter, output, *, include_full_image_boxes=False, run_info=None):
    """Run at most four already-selected train rows; return the persisted summary."""
    import numpy as np
    from PIL import Image
    from nucleus_rl.evaluate import ArtifactScorer, write_json
    from nucleus_rl.rewards import parse_completion

    if not rows or len(rows) > 4 or any(row.get("split") != "train" for row in rows):
        raise ValueError("Only one to four training rows are permitted")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    variants = ["tight", "padded_5px"] + (["full_image"] if include_full_image_boxes else [])
    recorder = RawRecordingSegmenter(segmenter)
    scorers = {name: ArtifactScorer(recorder, output / name, max_objects=8) for name in variants}
    records, display_rows = [], []
    started = time.monotonic()
    write_json(output / "diagnostic_scope.json", {
        "scope": SCOPE, "oracle_prompts": True, "ground_truth_used_for_prompting": True,
        "split": "train", "selected_ids": [row["id"] for row in rows],
        "selection": "First requested rows of the supplied training manifest; no score-based selection",
        "partial_instance_policy": "Include every visible positive GT ID, including border-truncated instances",
        "point_rule": "Visible foreground pixel nearest its instance centroid; row-major tie break",
        "full_image_rule": "One full-image box per GT point, with one foreground point in each separate SAM call",
        "variants": variants, "run_info": run_info or {}, "sam_before": segmenter.metadata,
    })
    for row in rows:
        with Image.open(row["mask_path"]) as image:
            gt = np.asarray(image)
        height, width = gt.shape
        gt_ids = [int(value) for value in np.unique(gt[gt > 0])]
        display_rows.append({**row, "diagnostic_gt_count": len(gt_ids)})
        for variant in variants:
            objects = oracle_prompt_objects(gt, padding=PADDING_PIXELS if variant == "padded_5px" else 0,
                                             full_image_boxes=variant == "full_image")
            text = "<answer>" + json.dumps({"objects": objects}, separators=(",", ":")) + "</answer>"
            parsed = parse_completion(text, width=width, height=height)
            if not parsed.valid:
                raise RuntimeError(f"Derived oracle prompt violates output contract: {parsed.error}")
            score_started = time.monotonic()
            result = scorers[variant].score(text, row)
            raw_path = Path(result["prediction_path"]).with_suffix(".raw.npz")
            np.savez_compressed(raw_path, **recorder.last_raw)
            record = {
                **result, "variant": variant, "scope": SCOPE, "oracle_prompts": True,
                "ground_truth_used_for_prompting": True, "split": "train",
                "gt_instance_ids_in_prompt_order": gt_ids,
                "truncated_instance_ids": row.get("truncated_instance_ids", []),
                "source_clipped_instance_ids": row.get("source_clipped_instance_ids", []),
                "raw_sam_outputs_path": str(raw_path),
                "sam_predict_seconds": recorder.last_predict_seconds,
                "score_and_artifact_seconds": time.monotonic() - score_started,
                "timing_note": "First variant for each image includes SAM image encoding; subsequent variants reuse cached features",
            }
            records.append(record)
            with (output / "oracle_comparisons.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    unchanged = segmenter.verify_unchanged()
    contact_sheet = output / "oracle_contact_sheet.png"
    save_contact_sheet(display_rows, records, variants, contact_sheet)
    variant_summary = {}
    for variant in variants:
        variant_records = [record for record in records if record["variant"] == variant]
        # Keep oracle provenance attached even when a variant folder is read alone.
        (scorers[variant].output / "completions.jsonl").write_text("".join(
            json.dumps(record, allow_nan=False) + "\n" for record in variant_records
        ))
        variant_summary[variant] = {
            **scorers[variant].summary(), "scope": SCOPE, "oracle_prompts": True,
            "ground_truth_used_for_prompting": True, "split": "train", "variant": variant,
            "mean_sam_predict_seconds": statistics.mean(
                record["sam_predict_seconds"] for record in variant_records
            ),
        }
        write_json(scorers[variant].output / "summary.json", variant_summary[variant])
    summary = {
        "status": "completed_oracle_prompt_diagnostic", "scope": SCOPE,
        "oracle_prompts": True, "ground_truth_used_for_prompting": True,
        "optimizer_steps": 0, "vlm_loaded": False, "split": "train",
        "images": len(rows), "selected_ids": [row["id"] for row in rows],
        "variants": variant_summary, "sam_freeze_audit": unchanged,
        "wall_seconds": time.monotonic() - started,
        "contact_sheet": str(contact_sheet), "run_info": run_info or {},
        "timing_note": "Includes inference and artifact IO, excludes SAM loading; image features are reused across variants",
        "interpretation": "Prompt sensitivity with privileged GT information on a small training subset; not a learned-policy comparison or segmentation quality ceiling",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    # Direct script invocation works from any cwd; heavy imports occur after --help.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from nucleus_rl.evaluate import read_manifest
    from nucleus_rl.segmenter import FrozenSAM2

    rows = select_training_rows(read_manifest(args.manifest), args.max_images)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty --output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no diagnostic has run")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("FrozenSAM2 CUDA path uses BF16; use compatible hardware or --device cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    load_started = time.monotonic()
    segmenter = FrozenSAM2(args.sam_checkpoint, args.sam_config, device=args.device)
    run_info = {
        "arguments": vars(args), "torch_version": torch.__version__, "device": args.device,
        "cpu_intraop_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "sam_load_seconds": time.monotonic() - load_started,
    }
    summary = run_diagnostic(rows, segmenter, output,
                             include_full_image_boxes=args.include_full_image_boxes, run_info=run_info)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
