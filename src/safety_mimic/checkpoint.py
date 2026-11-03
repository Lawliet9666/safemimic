"""Atomic, complete training snapshots, with one replay/RNG shard per rank."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from .ball import BallState

FORMAT = "safetymimic.training.v1"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, BallState):
        return {"__ball__": True, **{
            key: cpu_tree(getattr(value, key)) for key in ("position", "velocity", "radius")
        }}
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        if value.get("__ball__") is True:
            return BallState(*(device_tree(value[key], device) for key in ("position", "velocity", "radius")))
        return {key: device_tree(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device_tree(item, device) for item in value)
    return value


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng(generator, device):
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == "cuda" else None,
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "exploration": generator.get_state().cpu(),
    }


def restore_rng(record, generator, device):
    torch.set_rng_state(record["torch"].cpu())
    if record["cuda"] is not None:
        torch.cuda.set_rng_state(record["cuda"].cpu(), device)
    random.setstate(record["python"])
    state = record["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    generator.set_state(record["exploration"].cpu())


def save_snapshot(output, update, common, local, distribution):
    output = Path(output)
    sequence = common.get("checkpoint_sequence", 0)
    directory = output / "checkpoints" / f"update_{update:06d}_save_{sequence:06d}"
    error = None
    try:
        if (directory / "manifest.pt").exists():
            raise FileExistsError("Training checkpoint generations are immutable")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(cpu_tree(local), directory / f"rank_{distribution.rank:05d}.pt")
    except Exception as caught:
        error = caught
    distribution.agree(error)
    if distribution.primary:
        try:
            atomic_torch_save({
                **cpu_tree(common), "format": FORMAT, "world_size": distribution.world_size,
                "update": update, "simulator_resume": "reset_episodes",
            }, directory / "manifest.pt")
            # Publish the pointer last; an interrupted shard write never becomes
            # the checkpoint that --resume latest.json selects.
            pointer = output / "checkpoints/latest.json"
            temporary = pointer.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"checkpoint": directory.name + "/manifest.pt"}) + "\n")
            temporary.replace(pointer)
        except Exception as caught:
            error = caught
    distribution.agree(error)
    distribution.barrier()
    return directory / "manifest.pt"


def load_snapshot(path, *, rank=0, world_size=1):
    path = Path(path).resolve()
    if path.suffix == ".json":
        target = json.loads(path.read_text())["checkpoint"]
        path = (path.parent / target).resolve()
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("format") != FORMAT:
        raise ValueError("Resume requires a complete training snapshot, not an actor-only checkpoint")
    if record["world_size"] != world_size:
        raise ValueError("Resume requires the same world size to restore rank replay/RNG state")
    for index in range(world_size):
        if not (path.parent / f"rank_{index:05d}.pt").exists():
            raise ValueError(f"Incomplete checkpoint: missing rank {index}")
    local = torch.load(path.parent / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
    return record, local, path
