"""Small CPU-testable contracts for bounded, single-GPU LoRA training.

Checkpoint state uses JSON plus safetensors, never pickle loading. This avoids
disabling Transformers' torch.load safety gate on the pinned Torch 2.5 stack.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path


CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
    "safe_training_state.json", "safe_training_state.safetensors", "coordinate_config.json",
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dataset_identity(rows):
    """Ordered image/mask contents and input geometry, independent of mount paths."""
    cache = {}
    result = []
    keys = ("id", "patient_id", "source_image_id", "split", "bbox_xyxy", "coordinate_frame",
            "prompt_style", "original_width", "original_height", "processed_width", "processed_height",
            "prompt_width", "prompt_height")
    for row in rows:
        item = {key: row.get(key) for key in keys}
        for key in ("image_path", "mask_path"):
            path = str(Path(row[key]).resolve())
            if path not in cache:
                cache[path] = file_digest(path)
            item[key.replace("_path", "_sha256")] = cache[path]
        result.append(item)
    return result


def check_cohort(train_rows, validation_rows, cohort):
    canonical = set(cohort["image_ids"])
    seen = set()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        if not rows:
            raise ValueError(f"Nonempty {split} manifest required")
        for row in rows:
            source = row.get("source_image_id")
            if row.get("split") != split or source not in canonical:
                raise ValueError(f"{split} row is not from the official30 training-source cohort: {row.get('id')}")
            if row.get("patient_id") != "-".join(source.split("-")[:3]):
                raise ValueError("Patient ID does not match source image")
            if not row.get("id") or row["id"] in seen:
                raise ValueError("Training and validation IDs must be distinct")
            seen.add(row["id"])
    if {row["patient_id"] for row in train_rows} & {row["patient_id"] for row in validation_rows}:
        raise ValueError("Train/validation patient overlap")


def seal_checkpoint(checkpoint, contract, progress):
    """Write the completion marker LAST, after all checkpoint payloads exist."""
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = state["global_step"]
    if type(step) is not int or step < 1 or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint name and Trainer step disagree")
    marker = {"schema_version": 1, "contract": contract, "contract_sha256": contract_digest(contract),
              "global_step": step, "progress": progress,
              "files": {name: file_digest(checkpoint / name) for name in CHECKPOINT_FILES}}
    write_json(checkpoint / "resume_contract.json", marker)
    return marker


def verify_checkpoint(checkpoint, expected_contract=None):
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / "resume_contract.json").read_text())
    if (marker.get("schema_version") != 1 or set(marker.get("files", {})) != set(CHECKPOINT_FILES)
            or marker.get("contract_sha256") != contract_digest(marker["contract"])):
        raise ValueError("Missing or invalid complete-checkpoint contract")
    for name in CHECKPOINT_FILES:
        if file_digest(checkpoint / name) != marker["files"][name]:
            raise ValueError(f"Checkpoint content changed: {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = state.get("global_step")
    if type(step) is not int or step < 1 or step != marker.get("global_step") or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint name, marker and Trainer step disagree")
    if expected_contract is not None and contract_digest(expected_contract) != marker["contract_sha256"]:
        changed = sorted(key for key in set(expected_contract) | set(marker["contract"])
                         if expected_contract.get(key) != marker["contract"].get(key))
        raise ValueError("Resume contract mismatch: " + ", ".join(changed))
    return marker


def pack_tree(value, tensors, is_tensor):
    """Encode optimizer trees with explicit container types and integer keys."""
    if is_tensor(value):
        name = f"tensor_{len(tensors):06d}"
        tensors[name] = value
        return {"kind": "tensor", "name": name}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[pack_tree(k, tensors, is_tensor), pack_tree(v, tensors, is_tensor)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [pack_tree(v, tensors, is_tensor) for v in value]}
    if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
        return {"kind": "scalar", "value": value}
    raise ValueError(f"Unsupported safe-checkpoint value: {type(value).__name__}")


def unpack_tree(node, tensors):
    kind = node.get("kind")
    if kind == "tensor" and set(node) == {"kind", "name"}:
        return tensors[node["name"]]
    if kind == "scalar" and set(node) == {"kind", "value"}:
        value = node["value"]
        if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
            return value
    if kind in ("dict", "list", "tuple") and set(node) == {"kind", "items"}:
        if kind == "dict":
            result = {}
            for key, value in node["items"]:
                key = unpack_tree(key, tensors)
                if key in result:
                    raise ValueError("Duplicate checkpoint dictionary key")
                result[key] = unpack_tree(value, tensors)
            return result
        values = [unpack_tree(v, tensors) for v in node["items"]]
        return tuple(values) if kind == "tuple" else values
    raise ValueError("Invalid safe-checkpoint tree")


def capture_rng():
    import random
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(checkpoint, optimizer, scheduler):
    import torch
    from safetensors.torch import save_file
    tensors = {}
    tree = pack_tree({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng()},
                     tensors, torch.is_tensor)
    path = Path(checkpoint)
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()},
              str(path / "safe_training_state.safetensors"))
    write_json(path / "safe_training_state.json", tree)


def load_training_state(checkpoint):
    from safetensors.torch import load_file
    path = Path(checkpoint)
    tensors = load_file(str(path / "safe_training_state.safetensors"), device="cpu")
    return unpack_tree(json.loads((path / "safe_training_state.json").read_text()), tensors)


@contextmanager
def preserve_evaluation_state(model, rng_getter=capture_rng, rng_setter=restore_rng):
    """Evaluation may seed RNGs/change mode; restore even if scoring raises."""
    modes = [(module, module.training) for module in model.modules()]
    rng = rng_getter()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        rng_setter(rng)


def wall_budget_reached(elapsed_seconds, minutes):
    return minutes is not None and elapsed_seconds >= minutes * 60
