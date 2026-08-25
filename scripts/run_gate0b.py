#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


CHECKPOINT = "/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7"
ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
EVAL_UPDATES = [0, 16, 32, 48, 64]
TRANSFORMS = ["official", "local", "group_scaled"]


def observation_input(sample: dict) -> dict:
    return {
        "observation.images.image": sample["observation.images.image"],
        "observation.images.image2": sample["observation.images.image2"],
        "observation.state": sample["observation.state"],
        "task": sample["task"],
    }


def transform_action(action: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (action - mean) / scale


def inverse_action(action: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return action * scale + mean


def prepare_batch(preprocessor, sample: dict, mean: torch.Tensor, scale: torch.Tensor) -> dict:
    batch = preprocessor(observation_input(sample))
    batch["action"] = transform_action(sample["action"].to("cuda"), mean, scale).unsqueeze(0)
    batch["action_is_pad"] = sample["action_is_pad"].to("cuda").unsqueeze(0)
    return batch


def full_losses(policy, batch: dict, noise: torch.Tensor, timestep: torch.Tensor, physical_scale: torch.Tensor):
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    losses = policy.model.forward(
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state,
        actions,
        noise,
        timestep,
    )[:, :, :7]
    raw = losses.mean()
    physicalized = (losses * physical_scale.square()).mean()
    return raw, physicalized


@torch.inference_mode()
def validation_losses(policy, preprocessor, validation, randomness, mean, scale) -> tuple[float, float]:
    raw_values = []
    physical_values = []
    policy.eval()
    for index, sample in enumerate(validation):
        batch = prepare_batch(preprocessor, sample, mean, scale)
        noise = randomness["validation_noise"][index : index + 1].to("cuda")
        timestep = randomness["validation_time"][index : index + 1].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw, physical = full_losses(policy, batch, noise, timestep, scale)
        raw_values.append(float(raw))
        physical_values.append(float(physical))
    return float(np.mean(raw_values)), float(np.mean(physical_values))


@torch.inference_mode()
def generate_physical(policy, preprocessor, validation, randomness, mean, scale) -> np.ndarray:
    predictions = []
    policy.eval()
    for index, sample in enumerate(validation):
        batch = preprocessor(observation_input(sample))
        noise = randomness["generation_noise"][index : index + 1].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.predict_action_chunk(batch, noise=noise)
        predictions.append(inverse_action(action[0].float(), mean, scale).cpu().numpy())
    return np.stack(predictions).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", required=True, choices=TRANSFORMS)
    args = parser.parse_args()

    torch.manual_seed(430001)
    torch.cuda.manual_seed_all(430001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    frozen = torch.load(ARTIFACT_DIR / "gate0b_frozen_data.pt", map_location="cpu", weights_only=False)
    randomness = torch.load(ARTIFACT_DIR / "gate0b_randomness.pt", map_location="cpu", weights_only=False)
    transform_data = json.loads((ARTIFACT_DIR / "gate0b_transforms.json").read_text())[args.transform]
    mean = torch.tensor(transform_data["mean"], dtype=torch.float32, device="cuda")
    scale = torch.tensor(transform_data["scale_physical_per_coordinate_unit"], dtype=torch.float32, device="cuda")

    process_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT, local_files_only=True)
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )

    active_started = time.perf_counter()
    eval_log = []
    train_log = []
    baseline_predictions = generate_physical(
        policy, preprocessor, frozen["validation"], randomness, mean, scale
    )
    raw_val, physical_val = validation_losses(
        policy, preprocessor, frozen["validation"], randomness, mean, scale
    )
    eval_log.append({"update": 0, "raw_flow_mse": raw_val, "physicalized_velocity_mse": physical_val})

    for update in range(1, 65):
        policy.train()
        sample_index = int(randomness["order"][update - 1])
        sample = frozen["train"][sample_index]
        batch = prepare_batch(preprocessor, sample, mean, scale)
        noise = randomness["train_noise"][update - 1 : update].to("cuda")
        timestep = randomness["train_time"][update - 1 : update].to("cuda")
        optimizer.zero_grad(set_to_none=True)
        update_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_loss, physical_loss = full_losses(policy, batch, noise, timestep, scale)
        raw_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        torch.cuda.synchronize()
        train_log.append(
            {
                "update": update,
                "sample_index": sample_index,
                "raw_flow_mse": float(raw_loss.detach()),
                "physicalized_velocity_mse": float(physical_loss.detach()),
                "gradient_norm_pre_clip": float(grad_norm),
                "lr": 1e-4,
                "update_seconds": time.perf_counter() - update_started,
            }
        )
        print(
            f"transform={args.transform} update={update}/64 loss={float(raw_loss):.6f} "
            f"grad={float(grad_norm):.4f}",
            flush=True,
        )
        if update in EVAL_UPDATES[1:]:
            raw_val, physical_val = validation_losses(
                policy, preprocessor, frozen["validation"], randomness, mean, scale
            )
            eval_log.append(
                {"update": update, "raw_flow_mse": raw_val, "physicalized_velocity_mse": physical_val}
            )

    final_predictions = generate_physical(
        policy, preprocessor, frozen["validation"], randomness, mean, scale
    )
    torch.cuda.synchronize()
    active_seconds = time.perf_counter() - active_started
    process_seconds = time.perf_counter() - process_started
    targets = torch.stack([sample["action"] for sample in frozen["validation"]]).numpy()

    npz_path = ARTIFACT_DIR / f"gate0b_{args.transform}.npz"
    np.savez_compressed(
        npz_path,
        baseline_physical=baseline_predictions,
        final_physical=final_predictions,
        target_physical=targets.astype(np.float32),
    )
    result = {
        "protocol_commit": "337c95305698e527ef09db6f53040fb377a9d6a5",
        "transform": args.transform,
        "updates": 64,
        "batch_size": 1,
        "seed": 430001,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
        },
        "precision": "bfloat16 autocast, float32 elementwise loss",
        "train_log": train_log,
        "validation_log": eval_log,
        "timing": {
            "gpu_active_wall_seconds": active_seconds,
            "process_wall_seconds": process_seconds,
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "environment": {"torch": torch.__version__, "python": platform.python_version()},
        "arrays": str(npz_path),
    }
    json_path = ARTIFACT_DIR / f"gate0b_{args.transform}_run.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"json": str(json_path), "npz": str(npz_path), "active_seconds": active_seconds}))


if __name__ == "__main__":
    main()
