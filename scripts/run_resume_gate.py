#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
    selected_parameter_records,
    tensor_record,
)
from flowspec_vla.phase1 import (
    configure_libero_policy,
    cosine_lr,
    finite_or_raise,
    load_phase1_config,
    load_split_manifest,
    make_phase1_processors,
    seeded_flow_randomness,
)
from flowspec_vla.resume_gate import (
    PROTOCOL_FREEZE,
    ROOT,
    SEED,
    configure_determinism,
    export_final_state,
    load_complete_checkpoint,
    optimizer_digest,
    optimizer_for,
    rng_record,
    save_complete_checkpoint,
    scheduler_record,
    tensor_mapping_hashes,
)
from run_phase1_training import FixedSampler, sha256_ints, training_order, worker_seed


PROFILES = {
    "primary": {"total_updates": 100, "checkpoint_update": 50, "num_workers": 0},
    "multiworker": {"total_updates": 10, "checkpoint_update": 5, "num_workers": 4},
}


def reduced(losses: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    if action_is_pad is None:
        return losses.mean()
    keep = (~action_is_pad).unsqueeze(-1)
    return (losses * keep).sum() / (keep.sum() * losses.shape[-1]).clamp_min(1)


def loader_for(dataset, indices, batch_size, generator, workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": FixedSampler(indices),
        "num_workers": workers,
        "pin_memory": True,
        "generator": generator,
    }
    if workers:
        kwargs.update(
            {
                "prefetch_factor": 2,
                "persistent_workers": True,
                "worker_init_fn": worker_seed,
            }
        )
    return DataLoader(**kwargs)


def raw_identifiers(raw: dict) -> dict:
    return {
        key: tensor_record(raw[key], statistics=False)
        for key in ("index", "episode_index", "frame_index", "task_index")
        if key in raw
    }


def row_difference(reference: dict, candidate: dict) -> dict | None:
    checks = [
        ("checkpoint_loaded_parameters", "model_before"),
        ("checkpoint_loaded_parameters", "selected_parameters_before"),
        ("optimizer_state", "optimizer_before"),
        ("optimizer_state", "selected_optimizer_before"),
        ("scheduler_amp", "scheduler_before"),
        ("rng_state", "rng_before"),
        ("sampler_position", "sample_offset_before"),
        ("data_identity", "raw_identifiers"),
        ("data_identity", "raw_actions"),
        ("processed_batch", "processed"),
        ("flow_timestep", "flow_timestep"),
        ("gaussian_flow_noise", "flow_noise"),
        ("forward", "action_projection"),
        ("loss", "loss_tensors"),
        ("loss", "losses"),
        ("gradient", "gradients"),
        ("gradient", "selected_gradients"),
        ("gradient", "gradient_norm_pre_clip"),
        ("optimizer_update", "optimizer_after"),
        ("optimizer_update", "selected_optimizer_after"),
        ("parameters", "model_after"),
        ("parameters", "selected_parameters_after"),
        ("post_update_rng", "rng_after"),
    ]
    for stage, key in checks:
        reference_value = json.dumps(reference.get(key), sort_keys=True, allow_nan=True)
        candidate_value = json.dumps(candidate.get(key), sort_keys=True, allow_nan=True)
        if reference_value != candidate_value:
            return {"update": candidate["update"], "stage": stage, "field": key}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["reference", "resumed"])
    parser.add_argument("--profile", default="primary", choices=sorted(PROFILES))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    total_updates = profile["total_updates"]
    checkpoint_update = profile["checkpoint_update"]
    workers = profile["num_workers"]
    role_root = ROOT / args.profile / ("run_a" if args.role == "reference" else "run_b")
    if role_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {role_root}")
    role_root.mkdir(parents=True)

    started = time.perf_counter()
    configure_determinism()
    config = load_phase1_config()
    policy = configure_libero_policy(config)
    split = load_split_manifest(config)
    preprocessor, _ = make_phase1_processors(policy, split)
    optimizer = optimizer_for(policy, config)
    dataset = load_dataset(with_action_chunk=True)
    microbatch = config["training"]["microbatch_size"]
    accumulation = config["training"]["gradient_accumulation"]
    effective_batch = microbatch * accumulation
    full_order = training_order(split["train_indices"], total_updates * effective_batch, SEED)
    order_hash = sha256_ints(full_order)
    checkpoint_offset = checkpoint_update * effective_batch
    start_update = 1 if args.role == "reference" else checkpoint_update + 1
    sampler_order = full_order if args.role == "reference" else full_order[checkpoint_offset:]
    loader_generator = torch.Generator().manual_seed(SEED + 102)
    flow_generator = torch.Generator().manual_seed(SEED + 103)
    loader = loader_for(dataset, sampler_order, microbatch, loader_generator, workers)
    iterator = iter(loader)

    checkpoint_path = ROOT / args.profile / f"checkpoint_update{checkpoint_update:05d}.pt"
    restore_audit = None
    if args.role == "resumed":
        payload, restore_audit = load_complete_checkpoint(
            path=checkpoint_path,
            policy=policy,
            optimizer=optimizer,
            flow_generator=flow_generator,
            loader_generator=loader_generator,
        )
        training = payload["training"]
        required = {
            "global_update": checkpoint_update,
            "optimizer_step_count": checkpoint_update,
            "examples_seen": checkpoint_offset,
            "sample_offset": checkpoint_offset,
            "batch_position": checkpoint_update * accumulation,
            "gradient_accumulation_position": 0,
            "full_order_sha256": order_hash,
            "total_order_examples": len(full_order),
            "num_workers": workers,
        }
        mismatches = {key: {"expected": value, "loaded": training.get(key)} for key, value in required.items() if training.get(key) != value}
        restore_audit["position_mismatches"] = mismatches
        restore_audit["all_exact"] = restore_audit["all_exact"] and not mismatches
        (role_root / "restore_audit.json").write_text(json.dumps(restore_audit, indent=2) + "\n")
        if not restore_audit["all_exact"]:
            raise RuntimeError(f"Checkpoint restore audit failed: {mismatches}")

    representatives = representative_parameter_names(policy)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["action_projection"] = output.detach()

    handle = policy.model.action_out_proj.register_forward_hook(hook)
    reference_rows = None
    if args.role == "resumed":
        reference_rows = {
            row["update"]: row
            for row in json.loads((ROOT / args.profile / "run_a/trace.json").read_text())["updates"]
        }

    detailed_updates = (
        {checkpoint_update + 1, 60, 75, 100}
        if args.profile == "primary"
        else {checkpoint_update + 1, total_updates}
    )
    trace_rows = []
    checkpoint_inventory = None
    first_divergence = None
    torch.cuda.reset_peak_memory_stats()
    for update in range(start_update, total_updates + 1):
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
        capture = update > checkpoint_update
        detailed = update in detailed_updates
        row = None
        if capture:
            row = {
                "update": update,
                "lr": lr,
                "sample_offset_before": (update - 1) * effective_batch,
                "model_before": named_tensor_digest(policy.state_dict().items()) if detailed else None,
                "selected_parameters_before": selected_parameter_records(policy, representatives),
                "optimizer_before": optimizer_digest(optimizer, policy) if detailed else None,
                "selected_optimizer_before": optimizer_records(optimizer, policy, representatives),
                "scheduler_before": scheduler_record(update, optimizer, config),
                "rng_before": rng_record(flow_generator, loader_generator),
                "raw_identifiers": [],
                "raw_actions": [],
                "processed": [],
                "flow_noise": [],
                "flow_timestep": [],
                "action_projection": [],
                "loss_tensors": [],
                "losses": [],
            }
        for accumulation_index in range(accumulation):
            raw = next(iterator)
            batch = preprocessor(raw.copy())
            noise, timestep = seeded_flow_randomness(
                flow_generator,
                microbatch,
                config["dimensions"]["chunk"],
                config["dimensions"]["internal"],
            )
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
            finite_or_raise("resume_gate_loss", loss)
            if capture:
                row["raw_identifiers"].append(raw_identifiers(raw))
                row["raw_actions"].append(tensor_record(raw["action"], statistics=False))
                row["processed"].append(tensor_mapping_hashes(batch))
                row["flow_noise"].append(tensor_record(noise, statistics=False))
                row["flow_timestep"].append(tensor_record(timestep, statistics=False))
                row["action_projection"].append(tensor_record(captured["action_projection"], statistics=False))
                row["loss_tensors"].append(tensor_record(loss_tensor, statistics=False))
                row["losses"].append(float(loss.detach()))
            (loss / accumulation).backward()

        if capture:
            row["gradients"] = (
                named_tensor_digest(
                    (name, parameter.grad)
                    for name, parameter in policy.named_parameters()
                    if parameter.requires_grad
                )
                if detailed
                else None
            )
            row["selected_gradients"] = selected_parameter_records(
                policy, representatives, gradients=True
            )
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip_norm"])
        finite_or_raise("resume_gate_gradient_norm", grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        if capture:
            row.update(
                {
                    "gradient_norm_pre_clip": float(grad_norm),
                    "model_after": named_tensor_digest(policy.state_dict().items()) if detailed else None,
                    "selected_parameters_after": selected_parameter_records(policy, representatives),
                    "optimizer_after": optimizer_digest(optimizer, policy) if detailed else None,
                    "selected_optimizer_after": optimizer_records(optimizer, policy, representatives),
                    "rng_after": rng_record(flow_generator, loader_generator),
                    "sample_offset_after": update * effective_batch,
                }
            )
            trace_rows.append(row)
            (role_root / "trace.json").write_text(json.dumps({"updates": trace_rows}, indent=2) + "\n")
            if reference_rows is not None:
                first_divergence = row_difference(reference_rows[update], row)
                if first_divergence is not None:
                    (role_root / "first_divergence.json").write_text(
                        json.dumps(first_divergence, indent=2) + "\n"
                    )
                    raise RuntimeError(f"Resume gate diverged: {first_divergence}")

        if args.role == "reference" and update == checkpoint_update:
            training_state = {
                "global_update": update,
                "optimizer_step_count": update,
                "examples_seen": update * effective_batch,
                "epoch": 0,
                "sample_offset": update * effective_batch,
                "batch_position": update * accumulation,
                "gradient_accumulation_position": 0,
                "next_update": update + 1,
                "next_sample_offset": update * effective_batch,
                "full_order_sha256": order_hash,
                "total_order_examples": len(full_order),
                "num_workers": workers,
                "persistent_workers": bool(workers),
                "prefetch_factor": 2 if workers else None,
            }
            substrate = {
                "protocol_freeze_commit": PROTOCOL_FREEZE,
                "method": "m0",
                "seed": SEED,
                "base_revision": config["revisions"]["base_checkpoint_revision"],
                "dataset_revision": config["revisions"]["dataset_revision"],
                "lerobot_revision": config["revisions"]["lerobot_source_commit"],
                "determinism": {
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                },
                "precision": "bfloat16_autocast_no_grad_scaler",
                "microbatch": microbatch,
                "accumulation": accumulation,
                "effective_batch": effective_batch,
            }
            checkpoint_inventory = save_complete_checkpoint(
                path=checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                config=config,
                flow_generator=flow_generator,
                loader_generator=loader_generator,
                training_state=training_state,
                substrate=substrate,
            )
            inventory_path = ROOT / args.profile / "checkpoint_state_inventory.json"
            inventory_path.write_text(json.dumps(checkpoint_inventory, indent=2) + "\n")
            if not checkpoint_inventory["serialization_rng_exact"]:
                raise RuntimeError("Checkpoint serialization changed RNG state")

        if update == start_update or update % 10 == 0 or update == checkpoint_update:
            print(
                json.dumps(
                    {
                        "profile": args.profile,
                        "role": args.role,
                        "update": update,
                        "loss": float(np.mean(row["losses"])) if capture else None,
                        "lr": lr,
                    }
                ),
                flush=True,
            )

    handle.remove()
    final_training = {
        "global_update": total_updates,
        "optimizer_step_count": total_updates,
        "examples_seen": total_updates * effective_batch,
        "epoch": 0,
        "sample_offset": total_updates * effective_batch,
        "batch_position": total_updates * accumulation,
        "gradient_accumulation_position": 0,
        "next_update": total_updates + 1,
        "next_sample_offset": total_updates * effective_batch,
        "full_order_sha256": order_hash,
        "total_order_examples": len(full_order),
        "num_workers": workers,
    }
    final_amp = {
        "precision": "bfloat16_autocast",
        "grad_scaler_applicable": False,
        "grad_scaler_state": None,
    }
    final = export_final_state(
        directory=role_root / "final_state",
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler_record(total_updates, optimizer, config),
        amp=final_amp,
        training=final_training,
        flow_generator=flow_generator,
        loader_generator=loader_generator,
    )
    torch.cuda.synchronize()
    metadata = {
        "status": "COMPLETE",
        "profile": args.profile,
        "role": args.role,
        "method": "m0",
        "seed": SEED,
        "start_update": start_update,
        "checkpoint_update": checkpoint_update,
        "total_updates": total_updates,
        "continuation_updates": total_updates - checkpoint_update,
        "num_workers": workers,
        "order_sha256": order_hash,
        "restore_audit": restore_audit,
        "first_divergence": first_divergence,
        "final_state": final,
        "gpu_seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {"torch": torch.__version__, "python": platform.python_version()},
    }
    (role_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "profile": args.profile,
                "role": args.role,
                "gpu_seconds": metadata["gpu_seconds"],
                "peak_memory_gib": metadata["peak_memory_gib"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
