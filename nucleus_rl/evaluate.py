"""Shared input prompts, reference-mask scoring, and reviewable rollout artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import statistics


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "image_path", "mask_path", "patient_id", "split"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate record id: {row['id']}")
        seen.add(row["id"])
        for key in ("image_path", "mask_path"):
            item = Path(row[key]).expanduser()
            row[key] = str((item if item.is_absolute() else path.parent / item).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def processed_image_size(processor, inputs):
    """Recover the actual processed W,H from Qwen's unmerged patch grid."""
    grid = inputs["image_grid_thw"]
    grid = grid.tolist() if hasattr(grid, "tolist") else grid
    if len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("Expected exactly one image grid for each crop")
    patch_size = processor.image_processor.patch_size
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("Expected an integer Qwen image patch_size")
    _, grid_height, grid_width = grid[0]
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("Image patch grid dimensions must be positive")
    return int(grid_width * patch_size), int(grid_height * patch_size)


def prepare_coordinate_rows(rows, processor, coordinate_frame):
    """Measure preprocessing geometry without looking at reference masks."""
    from PIL import Image

    if coordinate_frame not in ("processed", "original"):
        raise ValueError("coordinate_frame must be processed or original")
    prepared = []
    for row in rows:
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        original_width, original_height = image.size
        image_inputs = processor.image_processor(images=image, return_tensors="pt")
        processed_width, processed_height = processed_image_size(processor, image_inputs)
        prompt_width, prompt_height = (
            (processed_width, processed_height) if coordinate_frame == "processed" else image.size
        )
        prepared.append({
            **row, "coordinate_frame": coordinate_frame,
            "original_width": original_width, "original_height": original_height,
            "processed_width": processed_width, "processed_height": processed_height,
            "prompt_width": prompt_width, "prompt_height": prompt_height,
        })
    return prepared


def prompt_size_for_row(row, original_size):
    """Legacy oracle rows use original pixels; prepared policy rows are explicit."""
    width, height = original_size
    frame = row.get("coordinate_frame", "original")
    if frame not in ("processed", "original"):
        raise ValueError(f"Unsupported coordinate frame: {frame}")
    if (row.get("original_width", width), row.get("original_height", height)) != original_size:
        raise ValueError("Image dimensions differ from the recorded original coordinate frame")
    if frame == "processed" and not all(key in row for key in (
        "prompt_width", "prompt_height", "processed_width", "processed_height"
    )):
        raise ValueError("Processed coordinate rows require measured prompt and processed dimensions")
    prompt_size = row.get("prompt_width", width), row.get("prompt_height", height)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in prompt_size):
        raise ValueError("Prompt dimensions must be positive integers")
    expected = (row["processed_width"], row["processed_height"]) if frame == "processed" else original_size
    if prompt_size != expected:
        raise ValueError("Prompt dimensions do not match the declared coordinate frame")
    return prompt_size


def objects_to_original(objects, prompt_size, original_size):
    """One declared linear conversion; no clipping, rounding, or frame guessing."""
    from nucleus_rl.rewards import ObjectPrompt

    scale_x = original_size[0] / prompt_size[0]
    scale_y = original_size[1] / prompt_size[1]
    converted = tuple(ObjectPrompt(
        box=(obj.box[0] * scale_x, obj.box[1] * scale_y, obj.box[2] * scale_x, obj.box[3] * scale_y),
        point=(obj.point[0] * scale_x, obj.point[1] * scale_y),
    ) for obj in objects)
    return converted, (scale_x, scale_y)


def assert_processed_dimensions(processor, inputs, row):
    actual = processed_image_size(processor, inputs)
    if "processed_width" in row and actual != (row["processed_width"], row["processed_height"]):
        raise ValueError(f"Actual processed dimensions {actual} differ from the recorded image frame")
    if row.get("coordinate_frame") == "processed" and actual != (row["prompt_width"], row["prompt_height"]):
        raise ValueError("Actual processed dimensions differ from the dimensions requested in the prompt")


def prompt_for_image(width, height, max_objects=8, coordinate_frame="original", prompt_style="schema"):
    if coordinate_frame not in ("processed", "original"):
        raise ValueError("coordinate_frame must be processed or original")
    if prompt_style not in ("schema", "example"):
        raise ValueError("prompt_style must be schema or example")
    frame_description = "PROCESSED image shown to you" if coordinate_frame == "processed" else "ORIGINAL image before preprocessing"
    messages = [
        {"role": "system", "content": "You locate individual cell nuclei in microscopy images. Return only the requested answer."},
        {"role": "user", "content": (
            f"Locate every visible nucleus, at most {max_objects} objects. "
            "For each nucleus give a tight box [x0,y0,x1,y1] and an interior foreground point [x,y]. "
            f"Use pixel coordinates of the {frame_description}, which is {width} pixels wide and {height} pixels high: "
            f"0<=x0<x1<={width}, 0<=y0<y1<={height}; "
            "the point must be inside its box and inside the image. "
            "Do not use normalized coordinates. Respond exactly as "
            '<answer>{"objects":[{"box":[x0,y0,x1,y1],"point":[x,y]}]}</answer>. '
            "Use one object per nucleus, with no prose or markdown."
        )},
    ]
    if prompt_style == "example":
        # Geometry is invented solely from frame dimensions. No reference masks
        # or nucleus locations enter this formatting demonstration.
        x0, y0 = width // 4, height // 4
        x1, y1 = max(x0 + 1, width // 2), max(y0 + 1, height // 2)
        demonstration = {"objects": [{
            "box": [x0, y0, x1, y1],
            "point": [(x0 + x1) // 2, (y0 + y1) // 2],
        }]}
        example = "<answer>" + json.dumps(demonstration, separators=(",", ":")) + "</answer>"
        messages[-1]["content"] += (
            "\n\nFormatting example with invented coordinates, unrelated to the supplied image:\n"
            + example
            + "\n\nNow inspect the supplied image and return its actual nuclei using the same format. "
            "Choose the objects and coordinates from this image; do not copy the example. "
            "Return only one complete <answer>...</answer> block containing valid JSON, with numeric coordinates."
        )
    return messages


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(message.get("content", "") for message in completion)
    raise TypeError(f"Unexpected completion type: {type(completion)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ArtifactScorer:
    """Score one completion at a time; never use count/PQ as training rewards."""
    def __init__(self, segmenter, output, max_objects=8):
        self.segmenter = segmenter
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_objects = max_objects
        self.results = []
        self.groups = []
        self.counter = 0

    def score(self, text, row, step=None):
        import numpy as np
        from PIL import Image
        from nucleus_rl.rewards import parse_completion, evaluate_prediction

        image = Image.open(row["image_path"]).convert("RGB")
        reference = np.asarray(Image.open(row["mask_path"]))
        width, height = image.size
        if reference.shape != (height, width):
            raise ValueError(f"Image/reference shape mismatch for {row['id']}")
        prompt_size = prompt_size_for_row(row, image.size)
        parsed = parse_completion(text, width=prompt_size[0], height=prompt_size[1], max_objects=self.max_objects)
        converted_objects, scales = objects_to_original(parsed.objects, prompt_size, image.size)
        instances = (
            self.segmenter.predict(image, converted_objects, image_key=row["image_path"])
            if parsed.valid else np.zeros(reference.shape, dtype=np.int32)
        )
        metrics = evaluate_prediction(text, instances, reference, max_objects=self.max_objects, prompt_size=prompt_size)
        self.counter += 1
        stem = f"{self.counter:06d}"
        np.save(self.output / f"{stem}.npy", instances, allow_pickle=False)
        (self.output / f"{stem}.txt").write_text(text)
        # Green marks prediction boundaries; red marks reference boundaries.
        overlay = np.asarray(image).copy()
        for labels, color in ((reference, (255, 60, 60)), (instances, (30, 255, 90))):
            boundary = np.zeros(labels.shape, dtype=bool)
            boundary[1:] |= labels[1:] != labels[:-1]
            boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            overlay[boundary] = color
        Image.fromarray(overlay).save(self.output / f"{stem}.png")
        result = {
            "id": row["id"], "patient_id": row["patient_id"], "step": step,
            "coordinate_frame": row.get("coordinate_frame", "original"),
            "prompt_style": row.get("prompt_style", "schema"),
            "prompt_size": list(prompt_size), "original_size": list(image.size),
            "scale_to_original": list(scales),
            "raw_objects": [asdict(obj) for obj in parsed.objects],
            "converted_objects": [asdict(obj) for obj in converted_objects],
            "text": text, "prediction_path": str((self.output / f"{stem}.npy").resolve()),
            "overlay_path": str((self.output / f"{stem}.png").resolve()),
            **metrics,
        }
        with (self.output / "completions.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        self.results.append(result)
        return result

    def record_group(self, results, step=None):
        rewards = [float(result["total_reward"]) for result in results]
        group = {
            "step": step, "ids": [result["id"] for result in results],
            "rewards": rewards, "reward_std": statistics.pstdev(rewards),
            "all_zero": all(reward == 0 for reward in rewards),
        }
        self.groups.append(group)
        with (self.output / "groups.jsonl").open("a") as stream:
            stream.write(json.dumps(group) + "\n")

    def summary(self):
        metrics = ("seg_reward", "format_reward", "total_reward", "pq", "precision", "recall", "count_error")
        summary = {"completions": len(self.results), "groups": len(self.groups)}
        for metric in metrics:
            values = [float(row[metric]) for row in self.results if row.get(metric) is not None]
            summary[f"mean_{metric}"] = statistics.mean(values) if values else None
        summary["invalid_completions"] = sum(not row["format_valid"] for row in self.results)
        summary["groups_with_reward_variation"] = sum(row["reward_std"] > 0 for row in self.groups)
        write_json(self.output / "summary.json", summary)
        return summary


def run_evaluation(model, processor, rows, scorer, seed, max_tokens=512, group_size=1):
    import torch
    from PIL import Image
    from transformers import set_seed

    set_seed(seed)
    model.eval()
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        prompt_size = prompt_size_for_row(row, image.size)
        prompt = prompt_for_image(
            *prompt_size, max_objects=scorer.max_objects,
            coordinate_frame=row.get("coordinate_frame", "original"),
            prompt_style=row.get("prompt_style", "schema"),
        )
        prompt[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        assert_processed_dimensions(processor, inputs, row)
        inputs = inputs.to(model.device)
        group = []
        for _ in range(group_size):
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=group_size > 1,
                    **({"temperature": 1.0} if group_size > 1 else {}),
                    use_cache=True,
                )
            answer_ids = generated[:, inputs["input_ids"].shape[1]:]
            completion = processor.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            group.append(scorer.score(completion, row))
        scorer.record_group(group)
    return scorer.summary()
