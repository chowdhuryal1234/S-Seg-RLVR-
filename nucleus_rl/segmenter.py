"""Frozen, prompted SAM2 inference. Heavy dependencies load only at construction."""
from __future__ import annotations

import hashlib
from pathlib import Path

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_sha256(model, trainable_only=False):
    """Hash actual parameter bytes, including BF16, without dtype conversion."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenSAM2:
    def __init__(self, checkpoint, config=SAM2_CONFIG, device="cuda"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.model = build_sam2(config, str(checkpoint), device=device, mode="eval")
        self.model.requires_grad_(False)
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.current_image = None
        self.metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "config": config,
            "source_commit": SAM2_COMMIT,
            "parameters_before": parameter_sha256(self.model),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

    def predict(self, image, objects, image_key=None):
        import numpy as np
        from nucleus_rl.rewards import masks_to_instances

        width, height = image.size
        if not objects:
            return np.zeros((height, width), dtype=np.int32)
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16, enabled=self.device.startswith("cuda")
        ):
            if image_key is None or image_key != self.current_image:
                self.predictor.set_image(np.asarray(image.convert("RGB")))
                self.current_image = image_key
            masks, scores = [], []
            for obj in objects:
                predicted, quality, _ = self.predictor.predict(
                    point_coords=np.asarray([obj.point], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    box=np.asarray(obj.box, dtype=np.float32),
                    multimask_output=False,
                    normalize_coords=True,
                )
                masks.append(predicted[0])
                scores.append(float(quality[0]))
        return masks_to_instances(np.stack(masks), scores=np.asarray(scores))

    def verify_unchanged(self):
        after = parameter_sha256(self.model)
        if after != self.metadata["parameters_before"]:
            raise RuntimeError("Frozen SAM2 parameter hash changed")
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Frozen SAM2 unexpectedly has trainable parameters or gradients")
        return {**self.metadata, "parameters_after": after, "unchanged": True}
