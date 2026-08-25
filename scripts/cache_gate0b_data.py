#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flowspec_vla.data import load_dataset
from lerobot.policies.common.flow_matching import sample_time_beta


ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")


def cache_sample(dataset, row: dict) -> dict:
    sample = dataset[row["dataset_index"]]
    if int(sample["index"]) != row["dataset_index"] or sample["task"] != row["task"]:
        raise RuntimeError("Frozen manifest/data mismatch")
    if bool(sample["action_is_pad"].any()):
        raise RuntimeError("Frozen Gate 0B anchor unexpectedly has padded action targets")
    return {
        "observation.images.image": sample["observation.images.image"].cpu().clone(),
        "observation.images.image2": sample["observation.images.image2"].cpu().clone(),
        "observation.state": sample["observation.state"].cpu().clone(),
        "action": sample["action"].cpu().clone(),
        "action_is_pad": sample["action_is_pad"].cpu().clone(),
        "task": sample["task"],
        "task_index": int(sample["task_index"]),
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "dataset_index": int(sample["index"]),
    }


def main() -> None:
    manifest = json.loads((ARTIFACT_DIR / "state_manifest.json").read_text())
    dataset = load_dataset(with_action_chunk=True)
    train = [cache_sample(dataset, row) for row in manifest["gate0b_train"]]
    validation = [cache_sample(dataset, row) for row in manifest["gate0b_validation"]]
    cache_path = ARTIFACT_DIR / "gate0b_frozen_data.pt"
    torch.save({"train": train, "validation": validation}, cache_path)

    all_train_actions = torch.stack([sample["action"] for sample in train]).float()
    local_mean = all_train_actions.mean(dim=(0, 1))
    local_std = all_train_actions.std(dim=(0, 1), correction=0)
    if bool((local_std <= 0).any()):
        raise RuntimeError("Local action transform is singular")

    from safetensors.torch import load_file

    stats = load_file(
        "/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7/"
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    official_mean = stats["action.mean"].float()
    official_std = stats["action.std"].float()
    diagonal = torch.tensor([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0])
    transforms = {
        "official": {"mean": official_mean, "scale": official_std},
        "local": {"mean": local_mean, "scale": local_std},
        "group_scaled": {"mean": official_mean, "scale": official_std / diagonal},
    }

    held_out = torch.stack([sample["action"] for sample in validation]).float()
    transform_report = {}
    for name, transform in transforms.items():
        mean, scale = transform["mean"], transform["scale"]
        encoded = (held_out - mean) / scale
        recovered = encoded * scale + mean
        max_error = float((recovered - held_out).abs().max())
        if max_error > 1e-6:
            raise RuntimeError(f"Transform {name} inverse error {max_error}")
        transform_report[name] = {
            "mean": mean.tolist(),
            "scale_physical_per_coordinate_unit": scale.tolist(),
            "inverse_max_abs_error": max_error,
        }
    (ARTIFACT_DIR / "gate0b_transforms.json").write_text(json.dumps(transform_report, indent=2) + "\n")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(430001)
    noises = torch.randn((64, 50, 32), generator=generator, dtype=torch.float32)
    torch.manual_seed(430001)
    times = sample_time_beta(64, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    order = np.random.default_rng(430001).permutation(128)[:64].astype(np.int64)

    val_generator = torch.Generator(device="cpu")
    val_generator.manual_seed(430002)
    validation_noises = torch.randn((32, 50, 32), generator=val_generator, dtype=torch.float32)
    torch.manual_seed(430002)
    validation_times = sample_time_beta(32, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    generation_noises = []
    for index in range(32):
        state_generator = torch.Generator(device="cpu")
        state_generator.manual_seed(450000 + index)
        generation_noises.append(torch.randn((50, 32), generator=state_generator, dtype=torch.float32))
    randomness_path = ARTIFACT_DIR / "gate0b_randomness.pt"
    torch.save(
        {
            "order": torch.from_numpy(order),
            "train_noise": noises,
            "train_time": times,
            "validation_noise": validation_noises,
            "validation_time": validation_times,
            "generation_noise": torch.stack(generation_noises),
        },
        randomness_path,
    )
    print(
        json.dumps(
            {
                "train": len(train),
                "validation": len(validation),
                "cache": str(cache_path),
                "randomness": str(randomness_path),
                "transforms": transform_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
