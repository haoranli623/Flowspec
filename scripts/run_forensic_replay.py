#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flowspec_vla.data import load_dataset
from flowspec_vla.forensic import (
    compare_tensors,
    named_tensor_digest,
    representative_parameter_names,
    rng_records,
    selected_parameter_records,
    tensor_mapping_records,
    tensor_record,
)
from flowspec_vla.phase1 import (
    configure_libero_policy,
    cosine_lr,
    load_phase1_config,
    load_split_manifest,
    make_phase1_processors,
    seeded_flow_randomness,
)
from run_phase1_training import FixedSampler, training_order, worker_seed


SEED = 560001


def reduced(losses: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    if action_is_pad is None:
        return losses.mean()
    keep = (~action_is_pad).unsqueeze(-1)
    return (losses * keep).sum() / (keep.sum() * losses.shape[-1]).clamp_min(1)


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_for(policy, config):
    return torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )


def aggregate_comparison(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict:
    names = sorted(set(left) | set(right))
    exact = True
    max_abs = 0.0
    squared_l2 = 0.0
    differing = 0
    first_differing = None
    for name in names:
        if name not in left or name not in right:
            exact = False
            differing += 1
            first_differing = first_differing or name
            continue
        comparison = compare_tensors(left[name], right[name])
        if not comparison.get("exact", False):
            exact = False
            differing += 1
            first_differing = first_differing or name
        max_abs = max(max_abs, comparison.get("max_abs", float("inf")))
        squared_l2 += comparison.get("l2", 0.0) ** 2
    return {
        "exact": exact,
        "tensor_count": len(names),
        "differing_tensor_count": differing,
        "first_differing_tensor": first_differing,
        "max_abs": max_abs,
        "l2": squared_l2**0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["default", "deterministic"])
    args = parser.parse_args()
    deterministic = args.mode == "deterministic"
    if deterministic and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("deterministic mode requires CUBLAS_WORKSPACE_CONFIG in the process environment")

    root = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic")
    root.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic)

    started = time.perf_counter()
    config = load_phase1_config()
    policy = configure_libero_policy(config)
    split = load_split_manifest(config)
    preprocessor, _ = make_phase1_processors(policy, split)
    dataset = load_dataset(with_action_chunk=True)
    microbatch = config["training"]["microbatch_size"]
    accumulation = config["training"]["gradient_accumulation"]
    order = training_order(split["train_indices"], microbatch * accumulation, SEED)
    loader_generator = torch.Generator().manual_seed(SEED + 102)
    flow_generator = torch.Generator().manual_seed(SEED + 103)
    loader = DataLoader(
        dataset,
        batch_size=microbatch,
        sampler=FixedSampler(order),
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_seed,
        generator=loader_generator,
    )
    iterator = iter(loader)
    cached = []
    for _ in range(accumulation):
        raw = next(iterator)
        batch = preprocessor(raw.copy())
        noise, timestep = seeded_flow_randomness(flow_generator, microbatch, 50, 32)
        cached.append(
            {
                "raw_identifiers": tensor_mapping_records(
                    {key: raw[key] for key in ("index", "episode_index", "frame_index", "task_index") if key in raw}
                ),
                "processed": {key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in batch.items()},
                "processed_records": tensor_mapping_records(batch),
                "noise": noise.to("cuda"),
                "timestep": timestep.to("cuda"),
                "noise_record": tensor_record(noise),
                "timestep_record": tensor_record(timestep),
            }
        )

    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items()}
    replay_rng = capture_rng()
    representatives = representative_parameter_names(policy)
    dummy_flow_generator = torch.Generator().manual_seed(1)
    dummy_loader_generator = torch.Generator().manual_seed(2)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["action_projection"] = output.detach().clone()

    handle = policy.model.action_out_proj.register_forward_hook(hook)
    pass_tensors = []
    pass_records = []
    torch.cuda.reset_peak_memory_stats()
    for pass_index in range(2):
        policy.load_state_dict(initial_state, strict=True)
        policy.train()
        restore_rng(replay_rng)
        optimizer = optimizer_for(policy, config)
        lr = cosine_lr(
            1,
            peak_lr=config["training"]["learning_rate"],
            final_lr=config["training"]["final_learning_rate"],
            warmup_updates=config["training"]["warmup_updates"],
            decay_updates=config["training"]["decay_updates"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        record = {
            "pass": pass_index + 1,
            "model_before": named_tensor_digest(policy.named_parameters()),
            "rng_before": rng_records(
                flow_generator=dummy_flow_generator, loader_generator=dummy_loader_generator
            ),
            "microbatches": [],
        }
        outputs = []
        losses = []
        valid_dim = policy.config.action_feature.shape[0]
        for accumulation_index, item in enumerate(cached):
            batch = item["processed"]
            actions = policy.prepare_action(batch)
            images, image_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_tensor = policy.model.forward(
                    images,
                    image_masks,
                    batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"],
                    state,
                    actions,
                    item["noise"],
                    item["timestep"],
                )[..., :valid_dim]
                loss = reduced(loss_tensor, batch.get("action_is_pad"))
            outputs.append(captured["action_projection"])
            losses.append(loss_tensor.detach().clone())
            record["microbatches"].append(
                {
                    "accumulation_index": accumulation_index,
                    "action_projection": tensor_record(captured["action_projection"]),
                    "loss_tensor": tensor_record(loss_tensor),
                    "loss": float(loss.detach()),
                }
            )
            (loss / accumulation).backward()
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        record["gradients"] = named_tensor_digest(gradients.items())
        record["selected_gradients"] = selected_parameter_records(policy, representatives, gradients=True)
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        record["gradient_norm_pre_clip"] = float(
            torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        torch.cuda.synchronize()
        record["model_after"] = named_tensor_digest(policy.named_parameters())
        record["selected_parameters_after"] = selected_parameter_records(policy, representatives)
        record["rng_after"] = rng_records(
            flow_generator=dummy_flow_generator, loader_generator=dummy_loader_generator
        )
        pass_records.append(record)
        pass_tensors.append({"outputs": outputs, "losses": losses, "gradients": gradients})

    handle.remove()
    comparisons = {
        "model_before_exact": pass_records[0]["model_before"] == pass_records[1]["model_before"],
        "rng_before_exact": pass_records[0]["rng_before"] == pass_records[1]["rng_before"],
        "forward": [compare_tensors(a, b) for a, b in zip(pass_tensors[0]["outputs"], pass_tensors[1]["outputs"], strict=True)],
        "loss": [compare_tensors(a, b) for a, b in zip(pass_tensors[0]["losses"], pass_tensors[1]["losses"], strict=True)],
        "gradients": aggregate_comparison(pass_tensors[0]["gradients"], pass_tensors[1]["gradients"]),
        "model_after_exact": pass_records[0]["model_after"] == pass_records[1]["model_after"],
    }
    first_divergence = None
    if not comparisons["model_before_exact"] or not comparisons["rng_before_exact"]:
        first_divergence = "restoration"
    elif not all(item["exact"] for item in comparisons["forward"]):
        first_divergence = "forward"
    elif not all(item["exact"] for item in comparisons["loss"]):
        first_divergence = "loss"
    elif not comparisons["gradients"]["exact"]:
        first_divergence = "backward"
    elif not comparisons["model_after_exact"]:
        first_divergence = "optimizer_or_parameters"

    torch.cuda.synchronize()
    result = {
        "status": "EXACT" if first_divergence is None else "DIVERGED",
        "mode": args.mode,
        "seed": SEED,
        "same_process": True,
        "replays": 2,
        "updates_per_replay": 1,
        "training_order": order,
        "cached_inputs": [
            {
                "raw_identifiers": item["raw_identifiers"],
                "processed": item["processed_records"],
                "flow_noise": item["noise_record"],
                "flow_timestep": item["timestep_record"],
            }
            for item in cached
        ],
        "passes": pass_records,
        "comparisons": comparisons,
        "first_divergence": first_divergence,
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "gpu_seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    path = root / f"one_step_replay_{args.mode}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(path), "status": result["status"], "first_divergence": first_divergence, "gradient_comparison": comparisons["gradients"], "gpu_seconds": result["gpu_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
