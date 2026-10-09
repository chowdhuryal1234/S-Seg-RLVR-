"""Shared input prompts, reference-mask scoring, and reviewable rollout artifacts."""
from __future__ import annotations

import json
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


def prompt_for_image(width, height, max_objects=8):
    return [
        {"role": "system", "content": "You locate individual cell nuclei in microscopy images. Return only the requested answer."},
        {"role": "user", "content": (
            f"Locate every visible nucleus in this {width} by {height} pixel image, at most {max_objects} objects. "
            "For each nucleus give a tight box [x0,y0,x1,y1] and an interior foreground point [x,y]. "
            f"Use ORIGINAL image pixel coordinates: 0<=x0<x1<={width}, 0<=y0<y1<={height}; "
            "the point must be inside its box and inside the image. "
            "Do not use normalized coordinates. Respond exactly as "
            '<answer>{"objects":[{"box":[x0,y0,x1,y1],"point":[x,y]}]}</answer>. '
            "Use one object per nucleus, with no prose or markdown."
        )},
    ]


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
        parsed = parse_completion(text, width=width, height=height, max_objects=self.max_objects)
        instances = (
            self.segmenter.predict(image, parsed.objects, image_key=row["image_path"])
            if parsed.valid else np.zeros(reference.shape, dtype=np.int32)
        )
        metrics = evaluate_prediction(text, instances, reference, max_objects=self.max_objects)
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
        prompt = prompt_for_image(*image.size, max_objects=scorer.max_objects)
        prompt[-1]["content"] = [{"type": "image"}, {"type": "text", "text": prompt[-1]["content"]}]
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
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
