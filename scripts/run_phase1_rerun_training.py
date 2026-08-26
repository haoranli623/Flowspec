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
    phase1_source,
    reduced_flow_loss,
    seeded_flow_randomness,
    validate_method,
)
from flowspec_vla.resume_gate import (
    load_complete_checkpoint,
    optimizer_digest,
    optimizer_for,
    rng_record,
    save_complete_checkpoint,
    scheduler_record,
    tensor_mapping_hashes,
    configure_determinism,
)
from run_phase1_training import (
    FixedSampler,
    cache_validation_batches,
    evaluate,
    save_checkpoint,
    sha256_ints,
    training_order,
    worker_seed,
)


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
CONFIG = PROJECT / "config/phase1_rerun.yaml"
PROTOCOL_FREEZE = "82dfe670e6073bc30397d699ee612c0386932a00"


def sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def loader_for(dataset, indices, batch_size, generator, config):
    workers = int(config["training"]["num_workers"])
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
                "prefetch_factor": int(config["training"]["prefetch_factor"]),
                "persistent_workers": bool(config["training"]["persistent_workers"]),
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


def action_projection_valid_record(policy) -> dict:
    name, parameter = next(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if name.endswith("action_out_proj.weight")
    )
    valid_dim = policy.config.action_feature.shape[0]
    return {"name": name, "valid_rows": tensor_record(parameter[:valid_dim])}


def complete_training_state(update, effective_batch, accumulation, order_hash, order, config):
    return {
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
        "total_order_examples": len(order),
        "num_workers": int(config["training"]["num_workers"]),
        "persistent_workers": bool(config["training"]["persistent_workers"]),
        "prefetch_factor": int(config["training"]["prefetch_factor"]),
    }


def deterministic_record(seed: int) -> dict:
    return {
        "seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "precision": "bfloat16_autocast",
        "grad_scaler": None,
        "process_count": 1,
        "visible_device_count": torch.cuda.device_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["m0", "m1", "m2"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=["smoke", "primary"])
    parser.add_argument("--role", default="reference", choices=["reference", "resumed"])
    args = parser.parse_args()

    config = load_phase1_config(CONFIG)
    method = validate_method(args.method)
    expected_seeds = (
        [config["smoke_gate"]["seed"]]
        if args.mode == "smoke"
        else config["training"]["seeds"]
    )
    if args.seed not in expected_seeds:
        raise ValueError(f"Seed {args.seed} is not frozen for {args.mode}: {expected_seeds}")
    if args.role == "resumed" and args.mode == "primary" and args.seed not in config["full_run_spotcheck"]["seeds"]:
        raise ValueError("Primary resume spot checks are frozen to seed 520001")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each rerun process must see exactly one GPU through CUDA_VISIBLE_DEVICES")

    configure_determinism(args.seed)
    deterministic = deterministic_record(args.seed)
    expected_deterministic = {
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "visible_device_count": 1,
    }
    mismatches = {
        key: {"expected": expected, "actual": deterministic[key]}
        for key, expected in expected_deterministic.items()
        if deterministic[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"Deterministic substrate mismatch: {mismatches}")

    smoke = args.mode == "smoke"
    total_updates = (
        int(config["smoke_gate"]["updates"])
        if smoke
        else int(config["training"]["updates"])
    )
    checkpoint_updates = (
        [int(config["smoke_gate"]["checkpoint_update"])]
        if smoke
        else [int(value) for value in config["training"]["checkpoint_updates"]]
    )
    comparison_checkpoint = (
        int(config["smoke_gate"]["checkpoint_update"])
        if smoke
        else int(config["full_run_spotcheck"]["checkpoint_update"])
    )
    trace_updates = set(
        int(value)
        for value in (
            config["smoke_gate"]["trace_updates"]
            if smoke
            else config["full_run_spotcheck"]["trace_updates"]
        )
    )
    resumed_end = (
        int(config["smoke_gate"]["resume_end_update"])
        if smoke
        else int(config["full_run_spotcheck"]["resume_end_update"])
    )
    run_name = f"{method}_seed{args.seed}"
    artifact_root = Path(config["paths"]["artifacts"])
    model_root = Path(config["paths"]["trained_models"])
    if args.role == "reference":
        run_root = artifact_root / ("smoke" if smoke else "runs") / run_name
    else:
        run_root = artifact_root / ("smoke_resume" if smoke else "spotchecks") / run_name
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite rerun output: {run_root}")
    run_root.mkdir(parents=True)

    started = time.perf_counter()
    split = load_split_manifest(config)
    policy = configure_libero_policy(config)
    preprocessor, postprocessor = make_phase1_processors(policy, split, config)
    optimizer = optimizer_for(policy, config)
    initial_base_digest = named_tensor_digest(policy.state_dict().items())
    initial_valid_record = action_projection_valid_record(policy)
    dataset = load_dataset(with_action_chunk=True)
    microbatch = int(config["training"]["microbatch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    effective_batch = microbatch * accumulation
    full_order = training_order(split["train_indices"], total_updates * effective_batch, args.seed)
    order_hash = sha256_ints(full_order)
    checkpoint_offset = comparison_checkpoint * effective_batch
    start_update = 1 if args.role == "reference" else comparison_checkpoint + 1
    end_update = total_updates if args.role == "reference" else resumed_end
    sampler_order = full_order if args.role == "reference" else full_order[checkpoint_offset:]
    loader_generator = torch.Generator().manual_seed(args.seed + 102)
    flow_generator = torch.Generator().manual_seed(args.seed + 103)
    train_loader = loader_for(dataset, sampler_order, microbatch, loader_generator, config)
    train_iterator = iter(train_loader)

    checkpoint_path = (
        model_root
        / ("smoke" if smoke else "runs")
        / run_name
        / f"update_{comparison_checkpoint:05d}"
        / "complete_state.pt"
    )
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
        required = complete_training_state(
            comparison_checkpoint, effective_batch, accumulation, order_hash, full_order, config
        )
        position_mismatches = {
            key: {"expected": value, "loaded": training.get(key)}
            for key, value in required.items()
            if training.get(key) != value
        }
        substrate_mismatches = {
            key: {"expected": value, "loaded": payload["substrate"].get(key)}
            for key, value in {
                "protocol_freeze_commit": PROTOCOL_FREEZE,
                "method": method,
                "seed": args.seed,
            }.items()
            if payload["substrate"].get(key) != value
        }
        restore_audit["position_mismatches"] = position_mismatches
        restore_audit["substrate_mismatches"] = substrate_mismatches
        restore_audit["all_exact"] = bool(
            restore_audit["all_exact"] and not position_mismatches and not substrate_mismatches
        )
        (run_root / "restore_audit.json").write_text(json.dumps(restore_audit, indent=2) + "\n")
        if not restore_audit["all_exact"]:
            raise RuntimeError("Complete checkpoint restoration audit failed")

    validation_batches = None
    validation_indices: list[int] = []
    generation_noise = validation_flow_noise = validation_flow_time = None
    validation_updates: set[int] = set()
    validation_log: list[dict] = []
    if args.role == "reference":
        validation_indices = list(split["validation_indices"])
        if smoke:
            validation_indices = validation_indices[: int(config["smoke_gate"]["validation_states"])]
            validation_updates = set(int(x) for x in config["smoke_gate"]["validation_updates"])
        else:
            validation_updates = set(int(x) for x in config["training"]["validation_updates"])
        validation_batches = cache_validation_batches(dataset, validation_indices, microbatch)
        n_validation = len(validation_indices)
        validation_generator = torch.Generator().manual_seed(config["validation"]["generation_noise_seed"])
        generation_noise = torch.randn(
            (n_validation, config["dimensions"]["chunk"], config["dimensions"]["internal"]),
            generator=validation_generator,
        )
        validation_flow_generator = torch.Generator().manual_seed(config["validation"]["flow_noise_seed"])
        validation_flow_noise = torch.randn(
            (n_validation, config["dimensions"]["chunk"], config["dimensions"]["internal"]),
            generator=validation_flow_generator,
        )
        validation_time_generator = torch.Generator().manual_seed(config["validation"]["flow_time_seed"])
        validation_flow_time = (
            torch.rand((n_validation,), generator=validation_time_generator).pow(2.0 / 3.0) * 0.999
            + 0.001
        )
        metrics, arrays = evaluate(
            policy,
            preprocessor,
            validation_batches,
            method,
            split,
            generation_noise,
            validation_flow_noise,
            validation_flow_time,
        )
        metrics["update"] = 0
        validation_log.append(metrics)
        np.savez_compressed(run_root / "validation_update00000.npz", **arrays)
        print(json.dumps({"run": run_name, "validation": metrics}), flush=True)

    representatives = representative_parameter_names(policy)
    captured: dict[str, torch.Tensor] = {}

    def capture_projection(_module, _inputs, output):
        captured["action_projection"] = output.detach()

    hook = policy.model.action_out_proj.register_forward_hook(capture_projection)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    train_log: list[dict] = []
    trace_rows: list[dict] = []
    checkpoint_inventories: dict[str, dict] = {}
    first_update_probe = None
    torch.cuda.reset_peak_memory_stats()
    active_started = time.perf_counter()

    for update in range(start_update, end_update + 1):
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
        update_started = time.perf_counter()
        capture = update in trace_updates
        row = None
        if capture:
            row = {
                "update": update,
                "lr": lr,
                "sample_offset_before": (update - 1) * effective_batch,
                "model_before": named_tensor_digest(policy.state_dict().items()),
                "selected_parameters_before": selected_parameter_records(policy, representatives),
                "optimizer_before": optimizer_digest(optimizer, policy),
                "selected_optimizer_before": optimizer_records(optimizer, policy, representatives),
                "scheduler_before": scheduler_record(update, optimizer, config),
                "rng_before": rng_record(flow_generator, loader_generator),
                "raw_identifiers": [],
                "raw_actions": [],
                "processed": [],
                "flow_noise": [],
                "effective_source": [],
                "valid_source": [],
                "padded_source_max_abs": [],
                "flow_timestep": [],
                "action_projection": [],
                "losses": [],
            }
        micro_losses: list[float] = []
        valid_velocity_rms: list[float] = []
        padded_velocity_rms: list[float] = []
        sample_indices: list[int] = []
        probe_microbatches = []
        for accumulation_index in range(accumulation):
            raw = next(train_iterator)
            sample_indices.extend(int(value) for value in raw["index"])
            batch = preprocessor(raw.copy())
            noise, timestep = seeded_flow_randomness(
                flow_generator,
                microbatch,
                config["dimensions"]["chunk"],
                config["dimensions"]["internal"],
            )
            effective_source = phase1_source(noise, method, config["dimensions"]["executable"])
            noise_cuda = noise.to("cuda", non_blocking=True)
            timestep_cuda = timestep.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = reduced_flow_loss(policy, batch, method, noise_cuda, timestep_cuda)
            finite_or_raise("training_loss", loss)
            (loss / accumulation).backward()
            velocity = captured["action_projection"].float()
            valid_velocity_rms.append(float(torch.sqrt(velocity[..., :7].square().mean())))
            padded_velocity_rms.append(float(torch.sqrt(velocity[..., 7:].square().mean())))
            micro_losses.append(float(loss.detach()))
            source_probe = {
                "raw_identifiers": raw_identifiers(raw),
                "flow_noise": tensor_record(noise, statistics=False),
                "effective_source": tensor_record(effective_source, statistics=False),
                "valid_source": tensor_record(effective_source[..., :7], statistics=False),
                "padded_source_max_abs": float(effective_source[..., 7:].abs().max()),
                "flow_timestep": tensor_record(timestep, statistics=False),
            }
            if update == 1:
                probe_microbatches.append(source_probe)
            if capture:
                row["raw_identifiers"].append(source_probe["raw_identifiers"])
                row["raw_actions"].append(tensor_record(raw["action"], statistics=False))
                row["processed"].append(tensor_mapping_hashes(batch))
                row["flow_noise"].append(source_probe["flow_noise"])
                row["effective_source"].append(source_probe["effective_source"])
                row["valid_source"].append(source_probe["valid_source"])
                row["padded_source_max_abs"].append(source_probe["padded_source_max_abs"])
                row["flow_timestep"].append(source_probe["flow_timestep"])
                row["action_projection"].append(
                    tensor_record(captured["action_projection"], statistics=False)
                )
                row["losses"].append(float(loss.detach()))

        if update == 1:
            first_update_probe = {"update": 1, "microbatches": probe_microbatches}
        if capture:
            row["gradients"] = named_tensor_digest(
                (name, parameter.grad)
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            )
            row["selected_gradients"] = selected_parameter_records(
                policy, representatives, gradients=True
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config["training"]["gradient_clip_norm"]
        )
        finite_or_raise("gradient_norm", grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        if capture:
            row.update(
                {
                    "gradient_norm_pre_clip": float(grad_norm),
                    "model_after": named_tensor_digest(policy.state_dict().items()),
                    "selected_parameters_after": selected_parameter_records(policy, representatives),
                    "optimizer_after": optimizer_digest(optimizer, policy),
                    "selected_optimizer_after": optimizer_records(optimizer, policy, representatives),
                    "rng_after": rng_record(flow_generator, loader_generator),
                    "sample_offset_after": update * effective_batch,
                }
            )
            trace_rows.append(row)
            (run_root / "trace.json").write_text(json.dumps({"updates": trace_rows}, indent=2) + "\n")

        record = {
            "update": update,
            "loss": float(np.mean(micro_losses)),
            "gradient_norm_pre_clip": float(grad_norm),
            "valid_velocity_rms": float(np.mean(valid_velocity_rms)),
            "padded_velocity_rms": float(np.mean(padded_velocity_rms)),
            "lr": lr,
            "examples_seen": update * effective_batch,
            "sample_index_first": sample_indices[0],
            "sample_index_last": sample_indices[-1],
            "sample_indices_sha256": sha256_ints(sample_indices),
            "update_seconds": time.perf_counter() - update_started,
        }
        train_log.append(record)
        if update == start_update or update % 25 == 0 or update == end_update:
            print(json.dumps({"run": run_name, "role": args.role, "train": record}), flush=True)

        if args.role == "reference" and update in validation_updates:
            metrics, arrays = evaluate(
                policy,
                preprocessor,
                validation_batches,
                method,
                split,
                generation_noise,
                validation_flow_noise,
                validation_flow_time,
            )
            metrics["update"] = update
            validation_log.append(metrics)
            np.savez_compressed(run_root / f"validation_update{update:05d}.npz", **arrays)
            print(json.dumps({"run": run_name, "validation": metrics}), flush=True)

        if args.role == "reference" and update in checkpoint_updates:
            training_state = complete_training_state(
                update, effective_batch, accumulation, order_hash, full_order, config
            )
            substrate = {
                "protocol_freeze_commit": PROTOCOL_FREEZE,
                "execution_commit": os.environ.get("FLOWSPEC_EXECUTION_COMMIT"),
                "mode": args.mode,
                "method": method,
                "seed": args.seed,
                "base_revision": config["revisions"]["base_checkpoint_revision"],
                "dataset_revision": config["revisions"]["dataset_revision"],
                "lerobot_revision": config["revisions"]["lerobot_source_commit"],
                "determinism": deterministic,
                "microbatch": microbatch,
                "accumulation": accumulation,
                "effective_batch": effective_batch,
            }
            directory = (
                model_root
                / ("smoke" if smoke else "runs")
                / run_name
                / f"update_{update:05d}"
            )
            inventory = save_complete_checkpoint(
                path=directory / "complete_state.pt",
                policy=policy,
                optimizer=optimizer,
                config=config,
                flow_generator=flow_generator,
                loader_generator=loader_generator,
                training_state=training_state,
                substrate=substrate,
            )
            before_export = rng_record(flow_generator, loader_generator)
            save_checkpoint(policy, preprocessor, postprocessor, directory)
            after_export = rng_record(flow_generator, loader_generator)
            inventory["model_export_rng_exact"] = before_export == after_export
            if not inventory["serialization_rng_exact"] or not inventory["model_export_rng_exact"]:
                raise RuntimeError("Checkpoint or model export changed RNG state")
            checkpoint_inventories[str(update)] = inventory
            (run_root / f"checkpoint_update{update:05d}_inventory.json").write_text(
                json.dumps(inventory, indent=2) + "\n"
            )

    hook.remove()
    torch.cuda.synchronize()
    active_seconds = time.perf_counter() - active_started
    final_model_digest = named_tensor_digest(policy.state_dict().items())
    final_optimizer_digest = optimizer_digest(optimizer, policy)
    result = {
        "status": "COMPLETE",
        "run_name": run_name,
        "mode": args.mode,
        "role": args.role,
        "method": method,
        "seed": args.seed,
        "protocol_freeze_commit": PROTOCOL_FREEZE,
        "execution_commit": os.environ.get("FLOWSPEC_EXECUTION_COMMIT"),
        "start_update": start_update,
        "end_update": end_update,
        "full_reference_updates": total_updates,
        "microbatch_size": microbatch,
        "gradient_accumulation": accumulation,
        "effective_batch_size": effective_batch,
        "examples_seen_at_end": end_update * effective_batch,
        "training_order_sha256": order_hash,
        "training_order_examples": len(full_order),
        "validation_indices_sha256": sha256_ints(validation_indices) if validation_indices else None,
        "initial_base_model_digest": initial_base_digest,
        "initial_valid_parameter_record": initial_valid_record,
        "first_update_probe": first_update_probe,
        "restore_audit": restore_audit,
        "trace_updates": sorted(trace_updates),
        "trace_sha256": sha256_json(trace_rows),
        "final_model_digest": final_model_digest,
        "final_optimizer_digest": final_optimizer_digest,
        "checkpoint_inventories": checkpoint_inventories,
        "optimizer": {
            "name": "AdamW",
            "peak_lr": config["training"]["learning_rate"],
            "betas": config["training"]["betas"],
            "eps": config["training"]["eps"],
            "weight_decay": config["training"]["weight_decay"],
            "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        },
        "scheduler": scheduler_record(end_update, optimizer, config),
        "determinism": deterministic,
        "train_log": train_log,
        "validation_log": validation_log,
        "timing": {
            "gpu_active_wall_seconds": active_seconds,
            "process_wall_seconds": time.perf_counter() - started,
            "gpu_hours": active_seconds / 3600.0,
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "environment": {"torch": torch.__version__, "python": platform.python_version()},
        "model_root": str(model_root / ("smoke" if smoke else "runs") / run_name),
    }
    (run_root / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "run": run_name,
                "mode": args.mode,
                "role": args.role,
                "end_update": end_update,
                "gpu_hours": result["timing"]["gpu_hours"],
                "peak_memory_gib": result["timing"]["peak_memory_gib"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
