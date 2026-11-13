"""Login-node check (no GPU, no weights): Qwen3.6-27B vision module names, what --vision-lora selects, and
whether the vision code has train-mode-only behaviour besides gradient checkpointing. Run with the SFT venv."""
import inspect
import os
import re
import sys

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText

sys.path.insert(0, sys.argv[1])  # code dir
from nucleus_rl.sft import language_lora_targets, vision_encoder, vision_lora_targets  # noqa: E402

revision = os.environ["REV_27B"]
config = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B", revision=revision)
print("config:", type(config).__name__)
vision_config = getattr(config, "vision_config", None)
print("vision_config:", {k: v for k, v in vars(vision_config).items() if not k.startswith("_")} if vision_config else None)
with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config, torch_dtype=torch.bfloat16)
root, encoder = vision_encoder(model)
print("vision root:", root, type(encoder).__name__,
      "params (M):", round(sum(p.numel() for p in encoder.parameters()) / 1e6, 1))
for scope in ("merger", "all"):
    targets = vision_lora_targets(model, scope)
    kinds = sorted({re.sub(r"\.\d+\.", ".N.", name[len(root) + 1:]) for name in targets})
    print(f"--vision-lora {scope}: {len(targets)} Linear modules; kinds: {kinds}")
language = language_lora_targets(model)
print("language targets:", len(language), "overlap with vision 'all':", len(set(language) & set(vision_lora_targets(model, "all"))))
other = [n for n, m in encoder.named_modules() if n and not isinstance(m, torch.nn.Linear) and not list(m.children())]
print("vision leaf modules that are not Linear:", sorted({type(m).__name__ for n, m in encoder.named_modules()
                                                          if n in other}))
source = inspect.getsource(sys.modules[type(encoder).__module__])
for number, line in enumerate(source.splitlines(), 1):
    if re.search(r"self\.training|dropout|drop_path", line):
        print(f"modeling L{number}: {line.strip()}")
print("gradient checkpointing support:", getattr(encoder, "supports_gradient_checkpointing", None),
      "| blocks:", type(getattr(encoder, "blocks", [None])[0]).__mro__[1].__name__ if hasattr(encoder, "blocks") else None)
