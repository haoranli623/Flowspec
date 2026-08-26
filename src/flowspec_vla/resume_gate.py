from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from flowspec_vla.forensic import named_tensor_digest, rng_records, tensor_record
from flowspec_vla.phase1 import cosine_lr


SEED = 570001
ROOT = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/resume_gate")
PROTOCOL_FREEZE = "a2a37dd9a5e6a8549fea54d165727b9864650acf"


def configure_determinism(seed: int = SEED) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Resume gate requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def optimizer_for(policy, config):
    return torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )


def scheduler_record(update: int, optimizer, config) -> dict[str, Any]:
    expected = cosine_lr(
        update,
        peak_lr=config["training"]["learning_rate"],
        final_lr=config["training"]["final_learning_rate"],
        warmup_updates=config["training"]["warmup_updates"],
        decay_updates=config["training"]["decay_updates"],
    )
    return {
        "implementation": "manual_cosine_lr_pure_function",
        "last_completed_update": update,
        "optimizer_step_count": update,
        "current_lr": [group["lr"] for group in optimizer.param_groups],
        "expected_lr": expected,
        "warmup_updates": config["training"]["warmup_updates"],
        "decay_updates": config["training"]["decay_updates"],
        "peak_lr": config["training"]["learning_rate"],
        "final_lr": config["training"]["final_learning_rate"],
    }


def capture_rng_state(flow_generator: torch.Generator, loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "flow_generator": flow_generator.get_state().clone(),
        "loader_generator": loader_generator.get_state().clone(),
    }


def restore_rng_state(
    state: dict[str, Any], flow_generator: torch.Generator, loader_generator: torch.Generator
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    flow_generator.set_state(state["flow_generator"])
    loader_generator.set_state(state["loader_generator"])


def rng_record(flow_generator: torch.Generator, loader_generator: torch.Generator) -> dict[str, Any]:
    return rng_records(flow_generator=flow_generator, loader_generator=loader_generator)


def recursive_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_cpu(item) for item in value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_named_tensors(optimizer, policy) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        for key, value in sorted(optimizer.state.get(parameter, {}).items()):
            if isinstance(value, torch.Tensor):
                output[f"{name}::{key}"] = value
    return output


def optimizer_digest(optimizer, policy) -> dict[str, Any]:
    return named_tensor_digest(optimizer_named_tensors(optimizer, policy).items())


def optimizer_groups(optimizer, policy) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in policy.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: [names[id(parameter)] for parameter in value]
                if key == "params"
                else value
                for key, value in group.items()
            }
        )
    return groups


def tensor_mapping_hashes(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: tensor_record(value, statistics=False)
        for key, value in sorted(mapping.items())
        if isinstance(value, torch.Tensor)
    }


def save_complete_checkpoint(
    *,
    path: Path,
    policy,
    optimizer,
    config: dict[str, Any],
    flow_generator: torch.Generator,
    loader_generator: torch.Generator,
    training_state: dict[str, Any],
    substrate: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing resume-gate checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    before = rng_record(flow_generator, loader_generator)
    raw_rng = capture_rng_state(flow_generator, loader_generator)
    model_digest = named_tensor_digest(policy.state_dict().items())
    optim_digest = optimizer_digest(optimizer, policy)
    groups = optimizer_groups(optimizer, policy)
    payload = {
        "format": "flowspec_vla.complete_training_state",
        "version": 1,
        "model": recursive_cpu(policy.state_dict()),
        "model_training": bool(policy.training),
        "optimizer": recursive_cpu(optimizer.state_dict()),
        "optimizer_groups_named": groups,
        "scheduler": scheduler_record(training_state["global_update"], optimizer, config),
        "amp": {
            "precision": "bfloat16_autocast",
            "grad_scaler_applicable": False,
            "grad_scaler_state": None,
        },
        "training": training_state,
        "rng": recursive_cpu(raw_rng),
        "boundary_records": {
            "model": model_digest,
            "optimizer": optim_digest,
            "optimizer_groups_named": groups,
            "rng": before,
        },
        "substrate": substrate,
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.rename(path)
    torch.cuda.synchronize()
    after = rng_record(flow_generator, loader_generator)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rng_before_serialization": before,
        "rng_after_serialization": after,
        "serialization_rng_exact": before == after,
        "model_digest": model_digest,
        "optimizer_digest": optim_digest,
        "optimizer_groups_named": groups,
        "scheduler": payload["scheduler"],
        "amp": payload["amp"],
        "training": training_state,
        "inventory": {
            "model_state_dict": True,
            "model_mode": True,
            "optimizer_state_dict": True,
            "adam_first_second_moments_and_steps": True,
            "parameter_groups": True,
            "manual_scheduler_update_and_lr": True,
            "grad_scaler": "not applicable",
            "global_update_optimizer_steps_examples_seen": True,
            "gradient_accumulation_position": True,
            "epoch_sampler_batch_and_example_offsets": True,
            "python_numpy_torch_cpu_cuda_rng": True,
            "flow_generator": True,
            "loader_generator": True,
            "frozen_substrate_and_deterministic_settings": True,
        },
    }


def load_complete_checkpoint(
    *, path: Path, policy, optimizer, flow_generator: torch.Generator, loader_generator: torch.Generator
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "flowspec_vla.complete_training_state" or payload.get("version") != 1:
        raise RuntimeError("Unsupported complete checkpoint format")
    policy.load_state_dict(payload["model"], strict=True)
    policy.train(payload["model_training"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"], flow_generator, loader_generator)
    restored = {
        "model": named_tensor_digest(policy.state_dict().items()),
        "optimizer": optimizer_digest(optimizer, policy),
        "optimizer_groups_named": optimizer_groups(optimizer, policy),
        "scheduler": payload["scheduler"],
        "amp": payload["amp"],
        "training": payload["training"],
        "rng": rng_record(flow_generator, loader_generator),
    }
    expected = payload["boundary_records"]
    audit = {
        "model_exact": restored["model"] == expected["model"],
        "optimizer_exact": restored["optimizer"] == expected["optimizer"],
        "optimizer_groups_exact": restored["optimizer_groups_named"] == expected["optimizer_groups_named"],
        "rng_exact": restored["rng"] == expected["rng"],
        "scheduler": restored["scheduler"],
        "amp": restored["amp"],
        "training": restored["training"],
        "restored_records": restored,
        "expected_records": expected,
    }
    audit["all_exact"] = all(
        audit[key] for key in ("model_exact", "optimizer_exact", "optimizer_groups_exact", "rng_exact")
    )
    return payload, audit


def export_final_state(
    *,
    directory: Path,
    policy,
    optimizer,
    scheduler: dict[str, Any],
    amp: dict[str, Any],
    training: dict[str, Any],
    flow_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing final-state directory: {directory}")
    directory.mkdir(parents=True)
    torch.cuda.synchronize()
    policy.save_pretrained(directory / "model")
    optimizer_tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in optimizer_named_tensors(optimizer, policy).items()
    }
    save_file(optimizer_tensors, directory / "optimizer.safetensors")
    state = {
        "model_parameters": named_tensor_digest(policy.named_parameters()),
        "model_buffers": named_tensor_digest(policy.named_buffers()),
        "optimizer": named_tensor_digest(optimizer_tensors.items()),
        "optimizer_groups_named": optimizer_groups(optimizer, policy),
        "scheduler": scheduler,
        "amp": amp,
        "training": training,
        "rng": rng_record(flow_generator, loader_generator),
        "files": {
            "model": {
                "path": str(directory / "model/model.safetensors"),
                "sha256": sha256_file(directory / "model/model.safetensors"),
            },
            "optimizer": {
                "path": str(directory / "optimizer.safetensors"),
                "sha256": sha256_file(directory / "optimizer.safetensors"),
            },
        },
    }
    (directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    return state
