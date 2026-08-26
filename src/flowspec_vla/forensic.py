from __future__ import annotations

import hashlib
import pickle
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(tensor: Tensor, *, statistics: bool = True) -> dict[str, Any]:
    detached = tensor.detach()
    record: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "sha256": sha256_bytes(tensor_bytes(detached)),
    }
    if statistics and detached.numel():
        values = detached.float()
        record.update(
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "l2": float(torch.linalg.vector_norm(values)),
            }
        )
    return record


def named_tensor_digest(named_tensors: Iterable[tuple[str, Tensor | None]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    missing = 0
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8") + b"\0")
        if tensor is None:
            digest.update(b"NONE\0")
            missing += 1
            continue
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor_bytes(tensor))
        count += 1
    return {"sha256": digest.hexdigest(), "tensor_count": count, "missing_count": missing}


def tensor_mapping_records(mapping: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, Tensor):
            records[key] = tensor_record(value)
    return records


def rng_records(*, flow_generator: torch.Generator, loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": sha256_bytes(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": sha256_bytes(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": tensor_record(torch.get_rng_state(), statistics=False),
        "torch_cuda_all": [tensor_record(state, statistics=False) for state in torch.cuda.get_rng_state_all()],
        "flow_generator": tensor_record(flow_generator.get_state(), statistics=False),
        "loader_generator": tensor_record(loader_generator.get_state(), statistics=False),
    }


def representative_parameter_names(policy) -> list[str]:
    names = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    action = [name for name in names if name.endswith("action_out_proj.weight")]
    if not action:
        raise RuntimeError("Could not find action_out_proj.weight")
    return [names[0], action[0], names[-1]]


def selected_parameter_records(policy, names: list[str], *, gradients: bool = False) -> dict[str, Any]:
    parameters = dict(policy.named_parameters())
    output = {}
    for name in names:
        tensor = parameters[name].grad if gradients else parameters[name]
        output[name] = None if tensor is None else tensor_record(tensor)
    return output


def optimizer_records(optimizer: torch.optim.Optimizer, policy, names: list[str]) -> dict[str, Any]:
    parameters = dict(policy.named_parameters())
    output: dict[str, Any] = {}
    for name in names:
        state = optimizer.state.get(parameters[name], {})
        output[name] = {
            key: tensor_record(value) if isinstance(value, Tensor) else value
            for key, value in sorted(state.items())
        }
    output["parameter_groups"] = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    return output


def compare_tensors(left: Tensor, right: Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "compatible": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    difference = left.float() - right.float()
    return {
        "compatible": True,
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.abs().mean()) if difference.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(difference)) if difference.numel() else 0.0,
    }
