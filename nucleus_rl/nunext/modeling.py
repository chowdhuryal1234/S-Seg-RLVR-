"""Qwen2.5-VL policy with coordinate tokens: loading, losses, generation and log-probs.

Torch and Transformers are imported lazily so CPU tests and --help need neither.
"""
from __future__ import annotations

import math
from typing import Sequence

from nucleus_rl.nunext.coords import token_strings
from nucleus_rl.nunext.data import INSTRUCTION


# ---------------------------------------------------------------- tokens and model

def add_coordinate_tokens(tokenizer, bins: int) -> int:
    """Add K coordinate tokens; return the id of bin 0. Ids must be contiguous."""
    tokens = token_strings(bins)
    tokenizer.add_tokens(tokens, special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if ids != list(range(ids[0], ids[0] + bins)):
        raise RuntimeError("Coordinate token ids are not contiguous; refusing an ambiguous vocabulary")
    return ids[0]


def coordinate_first_id(tokenizer, bins: int) -> int:
    """Id of bin 0 in a tokenizer that already contains the coordinate tokens."""
    tokens = token_strings(bins)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unknown = tokenizer.unk_token_id
    if any(token_id is None or token_id == unknown for token_id in ids) or ids != list(range(ids[0], ids[0] + bins)):
        raise ValueError("Tokenizer lacks contiguous coordinate tokens; was it saved by the NuNext SFT runner?")
    return ids[0]


COORD_INITS = ("digits", "mean")


def digit_weights(index: int, bins: int) -> list[tuple[str, float]]:
    """Bin index -> its 3-digit value on a 0-999 scale, as (digit token, weight) pairs, leading digit heaviest."""
    value = round(index / (bins - 1) * 999)
    return list(zip(f"{value:03d}", (0.7, 0.25, 0.05)))


def initialize_new_rows(model, first_id: int, bins: int, original_rows: int, tokenizer=None,
                        mode: str = "digits") -> None:
    """Initialise the coordinate-token embedding (and untied output) rows.

    "mean": the mean of the old vocabulary plus 2% noise. All 256 rows start almost identical, so the model
    cannot tell which coordinate it just emitted; an 8-GPU PanNuke run plateaued near the uniform loss
    (ln 256) with this. "digits" (default): a place-value-weighted mix of the embeddings of the digit tokens
    of the bin's 0-999 value plus 10% noise, so rows are distinct and ordered from the start.
    New ids may land in Qwen's unused padding rows, which are overwritten either way.
    """
    import torch

    if mode not in COORD_INITS:
        raise ValueError(f"coord init must be one of {COORD_INITS}")
    with torch.no_grad():
        for layer in {id(m): m for m in (model.get_input_embeddings(), model.get_output_embeddings())}.values():
            weight = layer.weight
            old = weight[:original_rows].float()
            scale = old.std()
            noise = torch.randn(bins, weight.shape[1], device=weight.device)
            if mode == "mean":
                rows = old.mean(dim=0) + noise * scale * 0.02
            else:
                if tokenizer is None:
                    raise ValueError("digits init needs the tokenizer")
                digit_ids = {d: tokenizer.convert_tokens_to_ids(d) for d in "0123456789"}
                if any(i is None or i == tokenizer.unk_token_id or i >= original_rows for i in digit_ids.values()):
                    raise RuntimeError("Tokenizer lacks single-digit tokens; use --coord-init mean")
                rows = torch.stack([sum(w * old[digit_ids[d]] for d, w in digit_weights(k, bins))
                                    for k in range(bins)]) + noise * scale * 0.1
            weight[first_id:first_id + bins] = rows.to(weight.dtype)


def load_policy(model_name: str, revision: str, *, bins: int | None = None, dtype: str = "float32",
                device: str | None = None, freeze_vision: bool = True, train_merger: bool = False,
                gradient_checkpointing: bool = True, trainable: bool = True, coord_init: str = "digits"):
    """Load Qwen2.5-VL and its processor.

    With ``bins`` set, coordinate tokens are added (base model). Without it, the tokenizer must
    already contain them (a NuNext SFT checkpoint). Returns (model, processor, first coordinate id).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, revision=revision, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa",
        device_map={"": device}, low_cpu_mem_usage=True)
    if bins is not None:
        original_tokens = len(processor.tokenizer)
        first_id = add_coordinate_tokens(processor.tokenizer, bins)
        rows = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > rows:
            model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64, mean_resizing=False)
        initialize_new_rows(model, first_id, bins, original_tokens, processor.tokenizer, coord_init)
    else:
        first_id = None
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        in_vision = "visual" in lowered or "vision" in lowered
        parameter.requires_grad = trainable and not (freeze_vision and in_vision
                                                    and not (train_merger and "merger" in lowered))
    if trainable and gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if not trainable:
        model.eval()
    return model, processor, first_id


# ---------------------------------------------------------------- prompts

def prompt_text(processor) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def end_token(processor) -> tuple[str, int]:
    tokenizer = processor.tokenizer
    token = "<|im_end|>"
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        token, token_id = tokenizer.eos_token, tokenizer.eos_token_id
    return token, int(token_id)


def end_ids(processor, model=None) -> set[int]:
    ids = {end_token(processor)[1]}
    if processor.tokenizer.eos_token_id is not None:
        ids.add(int(processor.tokenizer.eos_token_id))
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if configured is not None:
        ids.update(configured if isinstance(configured, (list, tuple)) else [configured])
    return ids


def encode_training_example(processor, image, target: str, max_length: int):
    """Teacher-forcing inputs and labels; the prompt part is masked with -100."""
    import torch

    prefix = prompt_text(processor)
    token, token_id = end_token(processor)
    prompt_inputs = processor(text=[prefix], images=[image], return_tensors="pt")
    full = processor(text=[prefix + target + token], images=[image], return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].tolist()
    full_ids = full["input_ids"][0].tolist()
    if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[-1] != token_id:
        raise ValueError("Target tokens do not extend the exact inference prompt")
    if len(full_ids) > max_length:
        raise ValueError(f"Sequence of {len(full_ids)} tokens exceeds --max-sequence-length; truncation is forbidden")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    full["labels"] = torch.tensor([labels], dtype=torch.long)
    return full, len(full_ids) - len(prompt_ids)


# ---------------------------------------------------------------- losses

def gaussian_targets(bins: int, sigma: float, device=None):
    """Row t is the soft label s(k | t) over the K bins (paper Eq. 7). sigma=0 gives one-hot rows."""
    import torch

    index = torch.arange(bins, dtype=torch.float32, device=device)
    if sigma <= 0:
        return torch.eye(bins, device=device)
    weights = torch.exp(-((index[None, :] - index[:, None]) ** 2) / (2 * sigma ** 2))
    return weights / weights.sum(dim=1, keepdim=True)


def soft_label_loss(logits, labels, first_id: int, bins: int, kernel):
    """Mean cross-entropy over completion tokens.

    ``logits``: (T, V) predictions for the T target tokens ``labels`` (T,). Coordinate targets use
    the Gaussian soft distribution over the coordinate bins (all other vocabulary entries get zero
    target mass); other targets (the end token) are one-hot. With a one-hot kernel this equals
    standard next-token cross-entropy (naive SFT).
    """
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    is_coord = (labels >= first_id) & (labels < first_id + bins)
    losses = torch.empty(labels.shape[0], device=logits.device, dtype=torch.float32)
    if is_coord.any():
        soft = kernel[labels[is_coord] - first_id]                   # (n, K)
        coord_log_probs = log_probs[is_coord, first_id:first_id + bins]
        losses[is_coord] = -(soft * coord_log_probs).sum(dim=-1)
    if (~is_coord).any():
        losses[~is_coord] = -log_probs[~is_coord].gather(1, labels[~is_coord, None]).squeeze(1)
    return losses.mean()


def completion_logits(model, inputs, completion_length: int):
    """Logits predicting the last ``completion_length`` tokens of ``inputs['input_ids']``."""
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, logits_to_keep=completion_length + 1,
                    use_cache=False)
    return outputs.logits[:, :-1]


# ---------------------------------------------------------------- generation

def generation_kwargs(processor, model, *, do_sample: bool, temperature: float, max_new_tokens: int,
                      num_return_sequences: int = 1) -> dict:
    """Plain sampling or greedy decoding: no repetition penalty, top-k or top-p.

    Qwen's shipped generation config sets a repetition penalty, which would discourage the
    repeated coordinate bins that neighbouring nuclei legitimately share.
    """
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "repetition_penalty": 1.0, "use_cache": True,
              "num_return_sequences": num_return_sequences, "eos_token_id": sorted(end_ids(processor, model)),
              "pad_token_id": processor.tokenizer.pad_token_id}
    if do_sample:
        kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
    else:
        kwargs.update(temperature=None, top_k=None, top_p=None)
    return kwargs


def trim_at_end(ids: Sequence[int], stop_ids) -> list[int]:
    """Ids up to and including the first end token; unchanged if none (a truncated sample).

    Qwen's pad token is also one of its end tokens, so trailing padding cannot simply be stripped.
    """
    stop_ids = set(stop_ids)
    for index, token in enumerate(ids):
        if token in stop_ids:
            return list(ids[:index + 1])
    return list(ids)


def generate(model, processor, images: Sequence, **kwargs) -> list[list[int]]:
    """Generated token ids (prompt removed, cut after the first end token) per image x num_return_sequences."""
    import torch

    prefix = prompt_text(processor)
    inputs = processor(text=[prefix] * len(images), images=list(images), padding=True,
                       return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.autocast(model.device.type, dtype=torch.bfloat16,
                                         enabled=model.device.type == "cuda"):
        output = model.generate(**inputs, **generation_kwargs(processor, model, **kwargs))
    prompt_length = inputs["input_ids"].shape[1]
    stop_ids = end_ids(processor, model)
    return [trim_at_end(row, stop_ids) for row in output[:, prompt_length:].tolist()]


# ---------------------------------------------------------------- GRPO log-probs

def rollout_batch(processor, image, completions: Sequence[Sequence[int]], device):
    """Stack one prompt with several completions (right-padded). All rows share the prompt."""
    import torch

    prefix = prompt_text(processor)
    prompt = processor(text=[prefix], images=[image], return_tensors="pt")
    count = len(completions)
    longest = max(len(c) for c in completions)
    pad = processor.tokenizer.pad_token_id
    ids = torch.full((count, longest), pad, dtype=torch.long)
    mask = torch.zeros((count, longest), dtype=torch.long)
    for row, completion in enumerate(completions):
        ids[row, :len(completion)] = torch.tensor(completion, dtype=torch.long)
        mask[row, :len(completion)] = 1
    prompt_ids = prompt["input_ids"].repeat(count, 1)
    batch = {
        "input_ids": torch.cat([prompt_ids, ids], dim=1),
        "attention_mask": torch.cat([prompt["attention_mask"].repeat(count, 1), mask], dim=1),
        "pixel_values": prompt["pixel_values"].repeat(count, 1),
        "image_grid_thw": prompt["image_grid_thw"].repeat(count, 1),
    }
    return {k: v.to(device) for k, v in batch.items()}, ids.to(device), mask.to(device)


def token_logprobs(model, batch, completion_ids, temperature: float = 1.0):
    """Per-token log pi(token) for the completion part, scored at the sampling temperature."""
    import torch

    logits = completion_logits(model, batch, completion_ids.shape[1]).float()
    if temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits, dim=-1).gather(2, completion_ids[..., None]).squeeze(-1)


def finite_or_raise(value, name: str):
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"{name} became non-finite")
    return value
