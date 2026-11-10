"""Minimal single-node data parallelism for the NuNext runners (launched with torchrun).

Gradients are averaged by hand after each accumulation cycle instead of wrapping the model in
DistributedDataParallel: no extra gradient buckets (memory is tight with fp32 full fine-tuning
of a 3B model), and no interaction with gradient checkpointing or the frozen vision tower.
Without torchrun (WORLD_SIZE unset or 1) every helper is a no-op.
"""
from __future__ import annotations

import os


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def enabled() -> bool:
    return world_size() > 1


def _device():
    import torch
    return f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"


def setup() -> str | None:
    """Initialise NCCL (gloo on CPU, for tests) and bind this process to its GPU; return its device."""
    if not enabled():
        return None
    import torch
    import torch.distributed as torch_dist

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    torch_dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return _device()


def teardown() -> None:
    if enabled():
        import torch.distributed as torch_dist
        if torch_dist.is_initialized():
            torch_dist.barrier()
            torch_dist.destroy_process_group()


def barrier() -> None:
    if enabled():
        import torch.distributed as torch_dist
        torch_dist.barrier()


def broadcast_parameters(model) -> None:
    """Make every rank start from rank 0's weights (new-token rows are randomly initialised)."""
    if enabled():
        import torch.distributed as torch_dist
        for parameter in model.parameters():
            torch_dist.broadcast(parameter.data, src=0)


def average_gradients(parameters) -> None:
    if enabled():
        import torch.distributed as torch_dist
        for parameter in parameters:
            if parameter.grad is not None:
                torch_dist.all_reduce(parameter.grad, op=torch_dist.ReduceOp.SUM)
                parameter.grad.div_(world_size())   # SUM then divide: gloo has no AVG


def any_true(flag: bool) -> bool:
    """True on every rank if it is true on any rank (keeps early stopping in lockstep)."""
    if not enabled():
        return flag
    import torch
    import torch.distributed as torch_dist
    value = torch.tensor([1.0 if flag else 0.0], device=_device())
    torch_dist.all_reduce(value, op=torch_dist.ReduceOp.MAX)
    return bool(value.item() > 0)


def mean(value: float) -> float:
    if not enabled():
        return value
    import torch
    import torch.distributed as torch_dist
    tensor = torch.tensor([float(value)], device=_device())
    torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.SUM)
    return float(tensor.item()) / world_size()


def gather_objects(items: list) -> list:
    """Concatenate per-rank Python lists on every rank."""
    if not enabled():
        return items
    import torch.distributed as torch_dist
    gathered = [None] * world_size()
    torch_dist.all_gather_object(gathered, items)
    return [item for part in gathered for item in part]
