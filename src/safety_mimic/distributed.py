"""Explicit gradient averaging after complete QP/BPTT backward passes.

The QP builds action Jacobians with autograd.grad. We synchronize completed
parameter gradients, avoiding reducer hooks inside those higher-order graphs.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class Distribution:
    def __init__(self, device="cpu", *, initialize=True, timeout_seconds=300):
        self.device = torch.device(device)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if initialize and self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo",
                timeout=timedelta(seconds=timeout_seconds),
            )

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def agree(self, error=None):
        """Raise on every rank before any rank applies an optimizer update."""
        failed = torch.tensor(int(error is not None), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            messages = [str(error) if error is not None else None]
            if self.world_size > 1:
                gathered = [None] * self.world_size
                dist.all_gather_object(gathered, messages[0])
                messages = gathered
            raise RuntimeError(f"Distributed operation failed: {messages}") from error

    def any(self, value):
        flag = torch.tensor(int(value), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @torch.no_grad()
    def broadcast_module(self, module):
        if self.world_size > 1:
            for value in module.state_dict().values():
                dist.broadcast(value, 0)

    @torch.no_grad()
    def average_gradients(self, parameters):
        parameters = list(parameters)
        error = None
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            error = RuntimeError("Nonfinite local gradient; optimizer step rejected")
        self.agree(error)
        if self.world_size == 1:
            return
        # Missing local gradients contribute zero. One flat collective keeps
        # ordering identical even when different QP active sets use different paths.
        flat = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters
        ])
        dist.all_reduce(flat)
        flat.div_(self.world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad = flat[offset:offset + count].view_as(parameter).clone()
            offset += count

    def gather_tensor(self, value):
        if self.world_size == 1:
            return value
        gathered = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(gathered, value.contiguous())
        return torch.cat(gathered, 0)

    def mean_metrics(self, metrics):
        keys = sorted(metrics)
        values = torch.tensor([metrics[key] for key in keys], device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(values)
            values.div_(self.world_size)
        return dict(zip(keys, values.cpu().tolist()))

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def max_metrics(self, metrics):
        keys = sorted(metrics)
        values = torch.tensor([metrics[key] for key in keys], device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return dict(zip(keys, values.cpu().tolist()))

    def assert_identical(self, value):
        if self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, value)
            if any(item != gathered[0] for item in gathered[1:]):
                raise RuntimeError("Model states diverged between training ranks")
