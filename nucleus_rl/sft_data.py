"""Training-only mask-to-prompt supervision and joint D4 image augmentation.

Targets contain one tight half-open box and one real foreground pixel per
visible instance. These helpers never turn validation masks into model inputs.
The fixed validation manifest is used only for split audits and evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nucleus_rl.cohort import load_official30_cohort, validate_training_rows
from nucleus_rl.evaluate import processed_image_size, prompt_for_image
from nucleus_rl.rewards import parse_completion


D4_TRANSFORMS = (
    "identity", "rot90", "rot180", "rot270",
    "mirror", "mirror_rot90", "mirror_rot180", "mirror_rot270",
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _instance_map(mask: Any) -> np.ndarray:
    labels = np.asarray(mask)
    if labels.ndim != 2 or min(labels.shape) == 0 or labels.dtype.kind not in "biuf":
        raise ValueError("Expected a nonempty two-dimensional numeric instance mask")
    if (not np.all(np.isfinite(labels)) or np.any(labels < 0)
            or np.any(labels != np.floor(labels))):
        raise ValueError("Instance mask IDs must be finite nonnegative integers")
    return labels


def _size(size: Any, name: str) -> tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"{name} must contain width and height")
    return _positive_int(size[0], name), _positive_int(size[1], name)


def coordinate_size(original_size, *, coordinate_mode="original", prompt_size=None):
    """Return an explicit frame; never infer processed pixels or model conventions."""
    original_size = _size(original_size, "original_size")
    if coordinate_mode == "normalized1000":
        expected = (1000, 1000)
    elif coordinate_mode == "original":
        expected = original_size
    elif coordinate_mode == "processed":
        if prompt_size is None:
            raise ValueError("Processed targets require a measured prompt_size")
        return _size(prompt_size, "prompt_size")
    else:
        raise ValueError("coordinate_mode must be original, processed, or normalized1000")
    if prompt_size is not None and _size(prompt_size, "prompt_size") != expected:
        raise ValueError("prompt_size differs from the explicitly declared coordinate frame")
    return expected


def apply_d4(image, mask, transform=0):
    """Jointly apply one of eight D4 transforms; return RGB PIL and a copied mask.

    Codes 0..3 rotate counterclockwise by 0, 90, 180, 270 degrees. Codes 4..7
    horizontally mirror first, then apply the same four rotations. Repeated
    appearances of a symmetric image are still only eight transform operators.
    """
    if (isinstance(transform, bool) or not isinstance(transform, (int, np.integer))
            or not 0 <= transform < 8):
        raise ValueError("transform must be an integer in 0..7")
    labels = _instance_map(mask)
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Image must be an RGB uint8 image")
    if rgb.shape[:2] != labels.shape:
        raise ValueError("Image and mask spatial dimensions differ")
    if transform >= 4:
        rgb, labels = np.fliplr(rgb), np.fliplr(labels)
    rgb, labels = np.rot90(rgb, int(transform) % 4), np.rot90(labels, int(transform) % 4)
    return Image.fromarray(np.ascontiguousarray(rgb)), np.array(labels, copy=True, order="C")


def instance_prompt_objects(mask, *, max_objects=8, prompt_size=None, coordinate_mode="original"):
    """Derive visible-instance prompts after augmentation, sorted top-to-bottom.

    Centroids can lie outside concave nuclei. The point is instead the actual
    foreground pixel closest to the visible centroid, with row-major tie breaks.
    Partial objects at crop edges remain included, matching the existing data.
    Fractional coordinates preserve geometry when changing coordinate frames.
    """
    labels = _instance_map(mask)
    max_objects = _positive_int(max_objects, "max_objects")
    instance_ids = np.unique(labels[labels > 0])
    if not 1 <= len(instance_ids) <= max_objects:
        raise ValueError(f"Targets require one to {max_objects} visible instances")
    height, width = labels.shape
    target_width, target_height = coordinate_size(
        (width, height), coordinate_mode=coordinate_mode, prompt_size=prompt_size,
    )
    scale_x, scale_y = target_width / width, target_height / height
    ordered = []
    for instance_id in instance_ids:
        pixels_yx = np.argwhere(labels == instance_id)
        centroid = pixels_yx.mean(axis=0)
        point_y, point_x = pixels_yx[np.argmin(((pixels_yx - centroid) ** 2).sum(axis=1))]
        lo_y, lo_x = pixels_yx.min(axis=0)
        hi_y, hi_x = pixels_yx.max(axis=0) + 1
        obj = {"box": [int(lo_x) * scale_x, int(lo_y) * scale_y,
                       int(hi_x) * scale_x, int(hi_y) * scale_y],
               "point": [int(point_x) * scale_x, int(point_y) * scale_y]}
        # Preserve exact integer coordinates when possible, without rounding
        # fractional box edges or moving a point outside a tiny foreground area.
        obj = {key: [int(v) if v.is_integer() else v for v in values]
               for key, values in obj.items()}
        ordered.append(((int(lo_y), int(lo_x), int(point_y), int(point_x), int(instance_id)), obj))
    return [obj for _, obj in sorted(ordered)]


def target_completion(mask, *, max_objects=8, prompt_size=None, coordinate_mode="original"):
    objects = instance_prompt_objects(mask, max_objects=max_objects, prompt_size=prompt_size,
                                      coordinate_mode=coordinate_mode)
    text = "<answer>" + json.dumps({"objects": objects}, separators=(",", ":"), allow_nan=False) + "</answer>"
    height, width = np.asarray(mask).shape
    width, height = coordinate_size((width, height), coordinate_mode=coordinate_mode, prompt_size=prompt_size)
    parsed = parse_completion(text, width=width, height=height, max_objects=max_objects)
    if not parsed.valid:
        raise ValueError(f"Generated an invalid supervised target: {parsed.error}")
    return text


def sft_prompt_messages(width, height, *, coordinate_mode="original", max_objects=8):
    if coordinate_mode != "normalized1000":
        return prompt_for_image(width, height, max_objects=max_objects, coordinate_frame=coordinate_mode)
    if _size((width, height), "prompt_size") != (1000, 1000):
        raise ValueError("Normalized prompts require a 1000 by 1000 coordinate frame")
    return [
        {"role": "system", "content": "You locate individual cell nuclei in microscopy images. Return only the requested answer."},
        {"role": "user", "content": (
            f"Locate every visible nucleus, at most {max_objects} objects. "
            "For each nucleus give a tight box [x0,y0,x1,y1] and an interior foreground point [x,y]. "
            "Use normalized coordinates from 0 to 1000 independently for image width and height: "
            "0<=x0<x1<=1000, 0<=y0<y1<=1000; "
            "the point must be inside its box and inside the image. Respond exactly as "
            '<answer>{"objects":[{"box":[x0,y0,x1,y1],"point":[x,y]}]}</answer>. '
            "Use one object per nucleus, with no prose or markdown."
        )},
    ]


def _load_training_pair(row):
    if row.get("split") != "train":
        raise ValueError("Supervised target creation accepts training rows only")
    validate_training_rows([row])
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    mask_path = Path(row["mask_path"])
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path, allow_pickle=False)
    else:
        with Image.open(mask_path) as source:
            mask = np.array(source)
    mask = _instance_map(mask)
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("Image and mask spatial dimensions differ")
    count = int(np.count_nonzero(np.unique(mask)))
    if "instance_count" in row and row["instance_count"] != count:
        raise ValueError("Saved mask instance count differs from its manifest")
    return image, mask


def build_sft_example(row, *, transform=0, prompt_size=None, coordinate_mode="original", max_objects=8):
    image, mask = _load_training_pair(row)
    image, mask = apply_d4(image, mask, transform)
    prompt_size = coordinate_size(image.size, coordinate_mode=coordinate_mode, prompt_size=prompt_size)
    objects = instance_prompt_objects(mask, max_objects=max_objects, prompt_size=prompt_size,
                                      coordinate_mode=coordinate_mode)
    completion = target_completion(mask, max_objects=max_objects, prompt_size=prompt_size,
                                   coordinate_mode=coordinate_mode)
    return {"id": row["id"], "image": image, "mask": mask, "objects": objects,
            "completion": completion, "prompt_size": prompt_size, "coordinate_mode": coordinate_mode,
            "augmentation_id": int(transform), "augmentation": D4_TRANSFORMS[int(transform)]}


def prepare_example(row, processor, augmentation_id=0, coordinate_frame="processed", max_objects=8):
    """Create one train-only teacher-forcing example in the model's measured frame."""
    image, mask = _load_training_pair(row)
    image, mask = apply_d4(image, mask, augmentation_id)
    processed = processed_image_size(processor, processor.image_processor(images=image, return_tensors="pt"))
    prompt_size = coordinate_size(image.size, coordinate_mode=coordinate_frame,
                                  prompt_size=processed if coordinate_frame == "processed" else None)
    completion = target_completion(mask, max_objects=max_objects, prompt_size=prompt_size,
                                   coordinate_mode=coordinate_frame)
    geometry = {"original_width": image.width, "original_height": image.height,
                "processed_width": processed[0], "processed_height": processed[1],
                "prompt_width": prompt_size[0], "prompt_height": prompt_size[1],
                "coordinate_frame": coordinate_frame}
    return {"id": row["id"], "image": image,
            "messages": sft_prompt_messages(*prompt_size, coordinate_mode=coordinate_frame, max_objects=max_objects),
            "completion": completion, "answer": completion, "prompt_size": prompt_size,
            "geometry": geometry, "augmentation_id": int(augmentation_id)}


def validate_sft_splits(train_rows, validation_rows, *, verify_files=True):
    """Audit the complete fixed split before any canary truncation or model loading.

    Exact image-byte duplicates across splits are rejected in addition to source
    and patient overlap. Identical mask bytes alone do not imply image leakage.
    Relative paths must already have been resolved by evaluate.read_manifest.
    """
    cohort = load_official30_cohort()
    train_audit = validate_training_rows(train_rows, cohort)
    if not validation_rows or any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("Expected a nonempty, entirely validation-only manifest")
    validate_training_rows([{**row, "split": "train"} for row in validation_rows], cohort)
    for key in ("id", "patient_id", "source_image_id"):
        overlap = {row[key] for row in train_rows} & {row[key] for row in validation_rows}
        if overlap:
            raise ValueError(f"Train/validation overlap in {key}: {sorted(overlap)}")
    for key in ("image_path", "mask_path"):
        paths = [{str(Path(row[key]).resolve()) for row in rows} for rows in (train_rows, validation_rows)]
        if paths[0] & paths[1]:
            raise ValueError(f"Train/validation overlap in {key}")
    files = []
    if verify_files:
        image_hashes = []
        for rows in (train_rows, validation_rows):
            split_hashes = set()
            for row in rows:
                record = {"id": row["id"], "split": row["split"]}
                for key in ("image_path", "mask_path"):
                    path = Path(row[key])
                    if not path.is_absolute():
                        raise ValueError("Resolve manifest file paths before auditing SFT splits")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hash_key = key.replace("_path", "_sha256")
                    if hash_key in row and row[hash_key] != digest:
                        raise ValueError(f"Manifest {hash_key} differs from saved file: {row['id']}")
                    record[hash_key] = digest
                    if key == "image_path":
                        split_hashes.add(digest)
                files.append(record)
            image_hashes.append(split_hashes)
        if image_hashes[0] & image_hashes[1]:
            raise ValueError("Train/validation images contain exact byte duplicates")
    return {"status": "official30_sft_split_verified", "cohort_id": cohort["cohort_id"],
            "train_crops": len(train_rows), "validation_crops": len(validation_rows),
            "train_patients": len({r["patient_id"] for r in train_rows}),
            "validation_patients": len({r["patient_id"] for r in validation_rows}),
            "train_source_image_ids": train_audit["selected_source_image_ids"],
            "validation_source_image_ids": sorted({r["source_image_id"] for r in validation_rows}),
            "all_files_sha256": files, "verify_files": verify_files,
            "augmentation_scope": "training only; labels rederived after joint image/mask transform"}


def assistant_token_labels(full_ids, prompt_ids, *, attention_mask=None, eos_token_id=None):
    """Mask an exact prompt-token prefix, preserving the assistant EOS suffix.

    Do not mask by token value: Qwen padding and EOS can share an ID. Padding is
    removed only with its attention mask. Any tokenizer boundary mismatch fails
    rather than silently putting the user/image tokens into the training loss.
    """
    full, prompt = np.asarray(full_ids), np.asarray(prompt_ids)
    if full.ndim != 1 or prompt.ndim != 1 or full.dtype.kind not in "iu" or prompt.dtype.kind not in "iu":
        raise ValueError("Token IDs must be one-dimensional integer arrays")
    if len(prompt) == 0 or len(full) <= len(prompt) or not np.array_equal(full[:len(prompt)], prompt):
        raise ValueError("Full conversation must contain the exact prompt token prefix and an assistant suffix")
    labels = full.astype(np.int64, copy=True)
    labels[:len(prompt)] = -100
    if attention_mask is not None:
        attention = np.asarray(attention_mask)
        if attention.shape != full.shape or not np.all((attention == 0) | (attention == 1)):
            raise ValueError("attention_mask must match token IDs and contain only zero/one")
        labels[attention == 0] = -100
    supervised = labels[labels != -100]
    if not len(supervised):
        raise ValueError("Assistant sequence contains no supervised tokens")
    if eos_token_id is not None and supervised[-1] != eos_token_id:
        raise ValueError("Assistant suffix must retain its EOS token")
    return labels.tolist()
