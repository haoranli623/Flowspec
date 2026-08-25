#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from flowspec_vla.data import load_dataset
from flowspec_vla.phase1 import (
    configure_libero_policy,
    cosine_lr,
    finite_or_raise,
    flow_loss_tensor,
    load_phase1_config,
    load_split_manifest,
    make_phase1_processors,
    normalized_physical_squared_error,
    reduced_flow_loss,
    sample_actions_with_trace,
    seeded_flow_randomness,
    unnormalize_actions,
    validate_method,
)


class FixedSampler(Sampler[int]):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def training_order(indices: list[int], count: int, seed: int) -> list[int]:
    generator = np.random.default_rng(seed + 101)
    order: list[int] = []
    base = np.asarray(indices, dtype=np.int64)
    while len(order) < count:
        order.extend(generator.permutation(base).tolist())
    return order[:count]


def sha256_ints(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def cache_validation_batches(dataset, indices: list[int], batch_size: int) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=FixedSampler(indices),
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    return list(loader)


@torch.inference_mode()
def evaluate(
    policy,
    preprocessor,
    raw_batches: list[dict],
    method,
    split_manifest: dict,
    generation_noise: torch.Tensor,
    flow_noise: torch.Tensor,
    flow_time: torch.Tensor,
) -> tuple[dict, dict[str, np.ndarray]]:
    was_training = policy.training
    policy.eval()
    valid_dim = policy.config.action_feature.shape[0]
    offset = 0
    standardized_sse = 0.0
    physical_sse = np.zeros(valid_dim, dtype=np.float64)
    coordinate_count = np.zeros(valid_dim, dtype=np.int64)
    chunk_sse = np.zeros(policy.config.chunk_size, dtype=np.float64)
    chunk_count = np.zeros(policy.config.chunk_size, dtype=np.int64)
    flow_sse = 0.0
    flow_count = 0
    predictions = []
    targets = []
    masks = []

    for raw in raw_batches:
        batch_size = raw["action"].shape[0]
        batch = preprocessor(raw.copy())
        selected = slice(offset, offset + batch_size)
        noise = generation_noise[selected].to("cuda", non_blocking=True)
        fixed_flow_noise = flow_noise[selected].to("cuda", non_blocking=True)
        fixed_flow_time = flow_time[selected].to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized, _ = sample_actions_with_trace(
                policy, batch, method, noise, collect_trace=False
            )
            loss_tensor = flow_loss_tensor(
                policy, batch, method, fixed_flow_noise, fixed_flow_time
            )
        normalized = normalized[..., :valid_dim].float()
        prediction = unnormalize_actions(normalized, split_manifest)
        target = raw["action"].to("cuda", non_blocking=True).float()
        keep = (~raw["action_is_pad"].to("cuda", non_blocking=True)).unsqueeze(-1)
        standardized = normalized_physical_squared_error(prediction, target, split_manifest)
        standardized_sse += float((standardized * keep).sum())
        flow_sse += float((loss_tensor.float() * keep).sum())
        flow_count += int(keep.sum()) * valid_dim
        squared = (prediction - target).square()
        physical_sse += (squared * keep).sum(dim=(0, 1)).cpu().numpy()
        coordinate_count += keep.sum(dim=(0, 1)).expand(valid_dim).cpu().numpy()
        chunk_sse += (standardized * keep).sum(dim=(0, 2)).cpu().numpy()
        chunk_count += (keep.sum(dim=0).squeeze(-1) * valid_dim).cpu().numpy()
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(target.cpu().numpy().astype(np.float32))
        masks.append(keep.squeeze(-1).cpu().numpy())
        offset += batch_size

    if was_training:
        policy.train()
    per_dim_rmse = np.sqrt(physical_sse / coordinate_count)
    metrics = {
        "states": offset,
        "whole_action_normalized_physical_rmse": float(
            np.sqrt(standardized_sse / int(coordinate_count.sum()))
        ),
        "fixed_flow_mse": float(flow_sse / flow_count),
        "translation_rmse": float(np.sqrt(physical_sse[:3].sum() / coordinate_count[:3].sum())),
        "rotation_rmse": float(np.sqrt(physical_sse[3:6].sum() / coordinate_count[3:6].sum())),
        "gripper_rmse": float(per_dim_rmse[6]),
        "per_dimension_physical_rmse": per_dim_rmse.tolist(),
        "chunk_normalized_physical_rmse": np.sqrt(chunk_sse / chunk_count).tolist(),
        "valid_action_values": int(coordinate_count.sum()),
        "all_outputs_finite": bool(np.isfinite(np.concatenate(predictions)).all()),
    }
    arrays = {
        "prediction_physical": np.concatenate(predictions),
        "target_physical": np.concatenate(targets),
        "valid_chunk_mask": np.concatenate(masks),
    }
    return metrics, arrays


def save_checkpoint(policy, preprocessor, postprocessor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="m0", choices=["m0", "m1", "m2"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-reproduction", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config()
    method = validate_method(args.method)
    if args.baseline_reproduction:
        if method != "m0":
            raise ValueError("The reproduction gate is defined only for official M0")
        run_config = config["baseline_reproduction"]
        seed = run_config["seed"] if args.seed is None else args.seed
        updates = run_config["updates"]
        validation_updates = run_config["validation_updates"]
        validation_limit = run_config["validation_states"]
        run_name = f"baseline_m0_seed{seed}"
        checkpoint_updates = [updates]
    else:
        if args.seed is None or args.seed not in config["training"]["seeds"]:
            raise ValueError(f"Primary seed must be one of {config['training']['seeds']}")
        seed = args.seed
        updates = config["training"]["updates"]
        validation_updates = config["training"]["validation_updates"]
        validation_limit = None
        run_name = f"{method}_seed{seed}"
        checkpoint_updates = config["training"]["checkpoint_updates"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    split_manifest = load_split_manifest(config)
    artifact_root = Path(config["paths"]["artifacts"])
    run_root = artifact_root / "runs" / run_name
    model_root = Path(config["paths"]["trained_models"]) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    policy = configure_libero_policy(config)
    preprocessor, postprocessor = make_phase1_processors(policy, split_manifest)
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
    total_examples = updates * microbatch * accumulation
    order = training_order(split_manifest["train_indices"], total_examples, seed)
    loader_generator = torch.Generator().manual_seed(seed + 102)
    train_loader = DataLoader(
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
    train_iterator = iter(train_loader)
    validation_indices = split_manifest["validation_indices"]
    if validation_limit is not None:
        validation_indices = validation_indices[:validation_limit]
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
    flow_generator = torch.Generator().manual_seed(seed + 103)

    captured_velocity: dict[str, torch.Tensor] = {}

    def capture_velocity(_module, _inputs, output):
        captured_velocity["value"] = output.detach()

    hook = policy.model.action_out_proj.register_forward_hook(capture_velocity)
    validation_log = []
    train_log = []
    process_started = time.perf_counter()
    active_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    initial_metrics, initial_arrays = evaluate(
        policy,
        preprocessor,
        validation_batches,
        method,
        split_manifest,
        generation_noise,
        validation_flow_noise,
        validation_flow_time,
    )
    initial_metrics["update"] = 0
    validation_log.append(initial_metrics)
    np.savez_compressed(run_root / "validation_update00000.npz", **initial_arrays)
    print(json.dumps({"run": run_name, "validation": initial_metrics}), flush=True)

    for update in range(1, updates + 1):
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
        micro_losses = []
        valid_velocity_rms = []
        padded_velocity_rms = []
        sample_indices = []
        for _ in range(accumulation):
            raw = next(train_iterator)
            sample_indices.extend(int(value) for value in raw["index"])
            batch = preprocessor(raw)
            noise, timestep = seeded_flow_randomness(
                flow_generator,
                microbatch,
                config["dimensions"]["chunk"],
                config["dimensions"]["internal"],
            )
            noise = noise.to("cuda", non_blocking=True)
            timestep = timestep.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = reduced_flow_loss(policy, batch, method, noise, timestep)
            finite_or_raise("training_loss", loss)
            (loss / accumulation).backward()
            velocity = captured_velocity["value"].float()
            valid_velocity_rms.append(float(torch.sqrt(velocity[..., :7].square().mean())))
            padded_velocity_rms.append(float(torch.sqrt(velocity[..., 7:].square().mean())))
            micro_losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config["training"]["gradient_clip_norm"]
        )
        finite_or_raise("gradient_norm", grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        record = {
            "update": update,
            "loss": float(np.mean(micro_losses)),
            "gradient_norm_pre_clip": float(grad_norm),
            "valid_velocity_rms": float(np.mean(valid_velocity_rms)),
            "padded_velocity_rms": float(np.mean(padded_velocity_rms)),
            "lr": lr,
            "examples_seen": update * microbatch * accumulation,
            "sample_index_first": sample_indices[0],
            "sample_index_last": sample_indices[-1],
            "update_seconds": time.perf_counter() - update_started,
        }
        train_log.append(record)
        if update == 1 or update % 25 == 0:
            print(json.dumps({"run": run_name, "train": record}), flush=True)

        if update in validation_updates:
            metrics, arrays = evaluate(
                policy,
                preprocessor,
                validation_batches,
                method,
                split_manifest,
                generation_noise,
                validation_flow_noise,
                validation_flow_time,
            )
            metrics["update"] = update
            validation_log.append(metrics)
            np.savez_compressed(run_root / f"validation_update{update:05d}.npz", **arrays)
            print(json.dumps({"run": run_name, "validation": metrics}), flush=True)
        if update in checkpoint_updates:
            save_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                model_root / f"update_{update:05d}",
            )

    hook.remove()
    torch.cuda.synchronize()
    active_seconds = time.perf_counter() - active_started
    result = {
        "run_name": run_name,
        "method": method,
        "seed": seed,
        "baseline_reproduction": args.baseline_reproduction,
        "updates": updates,
        "microbatch_size": microbatch,
        "gradient_accumulation": accumulation,
        "effective_batch_size": microbatch * accumulation,
        "examples_seen": total_examples,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "training_order_sha256": sha256_ints(order),
        "validation_indices_sha256": sha256_ints(validation_indices),
        "optimizer": {
            "name": "AdamW",
            "peak_lr": config["training"]["learning_rate"],
            "betas": config["training"]["betas"],
            "eps": config["training"]["eps"],
            "weight_decay": config["training"]["weight_decay"],
            "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        },
        "precision": config["training"]["precision"],
        "train_log": train_log,
        "validation_log": validation_log,
        "timing": {
            "gpu_active_wall_seconds": active_seconds,
            "process_wall_seconds": time.perf_counter() - process_started,
            "gpu_hours": active_seconds / 3600,
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "environment": {"torch": torch.__version__, "python": platform.python_version()},
        "model_root": str(model_root),
    }
    run_json = run_root / "run.json"
    run_json.write_text(json.dumps(result, indent=2) + "\n")

    if args.baseline_reproduction:
        thresholds = config["thresholds"]
        first_loss = float(np.mean([row["loss"] for row in train_log[:50]]))
        last_loss = float(np.mean([row["loss"] for row in train_log[-50:]]))
        initial = validation_log[0]
        final = validation_log[-1]
        finite = bool(
            np.isfinite([row["loss"] for row in train_log]).all()
            and np.isfinite([row["gradient_norm_pre_clip"] for row in train_log]).all()
            and all(row["all_outputs_finite"] for row in validation_log)
        )
        plausible_validation = (
            final["whole_action_normalized_physical_rmse"]
            / initial["whole_action_normalized_physical_rmse"]
            <= thresholds["baseline_final_to_initial_nrmse_max"]
            or final["fixed_flow_mse"] / initial["fixed_flow_mse"]
            <= thresholds["baseline_final_to_initial_flow_mse_max"]
        )
        passed = bool(
            finite
            and last_loss / first_loss <= thresholds["baseline_last_to_first_train_loss_max"]
            and plausible_validation
            and result["timing"]["peak_memory_gib"] <= thresholds["baseline_peak_memory_gib_max"]
        )
        summary = {
            "status": "PASS" if passed else "FAIL",
            "phase1_stop_triggered": not passed,
            "run": str(run_json),
            "seed": seed,
            "updates": updates,
            "first_50_train_loss_mean": first_loss,
            "last_50_train_loss_mean": last_loss,
            "last_to_first_train_loss_ratio": last_loss / first_loss,
            "initial_validation": initial,
            "final_validation": final,
            "finite_losses_gradients_outputs": finite,
            "plausible_validation_direction": plausible_validation,
            "peak_memory_gib": result["timing"]["peak_memory_gib"],
            "thresholds": {
                key: value for key, value in thresholds.items() if key.startswith("baseline_")
            },
            "gpu_hours": result["timing"]["gpu_hours"],
        }
        (artifact_root / "baseline_reproduction_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
