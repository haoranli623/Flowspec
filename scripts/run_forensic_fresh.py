#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flowspec_vla.data import load_dataset
from flowspec_vla.forensic import (
    named_tensor_digest,
    optimizer_records,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.updates <= 20:
        raise ValueError("Forensic fresh run must use 1-20 updates")
    if args.deterministic and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("deterministic mode requires CUBLAS_WORKSPACE_CONFIG in the process environment")

    config = load_phase1_config()
    root = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic") / args.label
    root.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(args.deterministic)

    started = time.perf_counter()
    policy = configure_libero_policy(config)
    split = load_split_manifest(config)
    preprocessor, _ = make_phase1_processors(policy, split)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )
    dataset = load_dataset(with_action_chunk=True)
    microbatch = config["training"]["microbatch_size"]
    accumulation = config["training"]["gradient_accumulation"]
    order = training_order(split["train_indices"], args.updates * microbatch * accumulation, SEED)
    loader_generator = torch.Generator().manual_seed(SEED + 102)
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
    flow_generator = torch.Generator().manual_seed(SEED + 103)
    representatives = representative_parameter_names(policy)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["action_projection"] = output.detach()

    handle = policy.model.action_out_proj.register_forward_hook(hook)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for update in range(1, args.updates + 1):
        policy.train()
        lr = cosine_lr(
            update,
            peak_lr=config["training"]["learning_rate"],
            final_lr=config["training"]["final_learning_rate"],
            warmup_updates=config["training"]["warmup_updates"],
            decay_updates=config["training"]["decay_updates"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        row = {
            "update": update,
            "lr": lr,
            "sample_offset_before": (update - 1) * microbatch * accumulation,
            "model_before": named_tensor_digest(policy.named_parameters()),
            "buffers_before": named_tensor_digest(policy.named_buffers()),
            "selected_parameters_before": selected_parameter_records(policy, representatives),
            "rng_before_update": rng_records(flow_generator=flow_generator, loader_generator=loader_generator),
            "microbatches": [],
        }
        losses = []
        for accumulation_index in range(accumulation):
            raw = next(iterator)
            batch = preprocessor(raw.copy())
            noise, timestep = seeded_flow_randomness(flow_generator, microbatch, 50, 32)
            micro = {
                "accumulation_index": accumulation_index,
                "raw_identifiers": tensor_mapping_records(
                    {key: raw[key] for key in ("index", "episode_index", "frame_index", "task_index") if key in raw}
                ),
                "raw_action": tensor_record(raw["action"]),
                "processed": tensor_mapping_records(batch),
                "flow_noise": tensor_record(noise),
                "flow_timestep": tensor_record(timestep),
                "rng_before_forward": rng_records(flow_generator=flow_generator, loader_generator=loader_generator),
            }
            noise_cuda = noise.to("cuda", non_blocking=True)
            timestep_cuda = timestep.to("cuda", non_blocking=True)
            valid_dim = policy.config.action_feature.shape[0]
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
                    noise_cuda,
                    timestep_cuda,
                )[..., :valid_dim]
                loss = reduced(loss_tensor, batch.get("action_is_pad"))
            micro.update(
                {
                    "action_projection": tensor_record(captured["action_projection"]),
                    "loss_tensor": tensor_record(loss_tensor),
                    "loss": float(loss.detach()),
                    "rng_after_forward": rng_records(flow_generator=flow_generator, loader_generator=loader_generator),
                }
            )
            (loss / accumulation).backward()
            micro["rng_after_backward"] = rng_records(
                flow_generator=flow_generator, loader_generator=loader_generator
            )
            row["microbatches"].append(micro)
            losses.append(float(loss.detach()))
        row["gradients"] = named_tensor_digest(
            (name, parameter.grad) for name, parameter in policy.named_parameters() if parameter.requires_grad
        )
        row["selected_gradients"] = selected_parameter_records(policy, representatives, gradients=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip_norm"])
        row["gradient_norm_pre_clip"] = float(grad_norm)
        row["gradients_after_clip"] = named_tensor_digest(
            (name, parameter.grad) for name, parameter in policy.named_parameters() if parameter.requires_grad
        )
        optimizer.step()
        torch.cuda.synchronize()
        row.update(
            {
                "loss_mean": float(np.mean(losses)),
                "model_after": named_tensor_digest(policy.named_parameters()),
                "selected_parameters_after": selected_parameter_records(policy, representatives),
                "optimizer_after": optimizer_records(optimizer, policy, representatives),
                "rng_after_update": rng_records(flow_generator=flow_generator, loader_generator=loader_generator),
                "sample_offset_after": update * microbatch * accumulation,
            }
        )
        rows.append(row)
        (root / "capture.json").write_text(json.dumps({"updates": rows}, indent=2) + "\n")
        print(json.dumps({"label": args.label, "update": update, "loss": row["loss_mean"]}), flush=True)

    handle.remove()
    checkpoint = root / "final_model"
    policy.save_pretrained(checkpoint)
    torch.cuda.synchronize()
    metadata = {
        "status": "COMPLETE",
        "label": args.label,
        "method": "m0",
        "seed": SEED,
        "updates": args.updates,
        "representative_parameters": representatives,
        "training_order": order,
        "data": {
            "num_workers": config["training"]["num_workers"],
            "persistent_workers": True,
            "prefetch_factor": 2,
            "microbatch": microbatch,
            "accumulation": accumulation,
        },
        "precision": "bfloat16_autocast",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {"torch": torch.__version__, "python": platform.python_version()},
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
