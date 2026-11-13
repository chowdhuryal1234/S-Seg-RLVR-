"""Root-sum-square of ||(alpha/r) B A||_F over LoRA modules, split into language and vision, for each adapter given."""
import sys
import torch
from safetensors.torch import load_file

torch.set_num_threads(2)
SCALE = 32 / 16  # lora_alpha / rank in every arm
for path in sys.argv[1:]:
    tensors = load_file(path)
    total = {"language": 0.0, "vision": 0.0}
    count = {"language": 0, "vision": 0}
    for key, a in tensors.items():
        if "lora_A" not in key:
            continue
        b = tensors[key.replace("lora_A", "lora_B")].float()
        a = a.float()
        group = "vision" if any(w in key for w in ("visual", "vision", "merger")) else "language"
        total[group] += SCALE ** 2 * torch.trace((a @ a.T) @ (b.T @ b)).item()  # ||B A||_F^2 without forming B A
        count[group] += 1
    run = path.split("/checkpoints")[0].rsplit("/runs/", 1)[-1]
    print(f"{run[:44]:44s} language {count['language']} modules |dW| {total['language'] ** 0.5:.3f}"
          + (f" | vision {count['vision']} modules |dW| {total['vision'] ** 0.5:.3f}" if count["vision"] else ""))
