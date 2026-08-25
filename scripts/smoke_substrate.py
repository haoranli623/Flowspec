#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch

from flowspec_vla.data import load_dataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


CHECKPOINT = "/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7"
MANIFEST = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0/state_manifest.json")


def model_input(sample: dict) -> dict:
    return {key: value for key, value in sample.items() if key.startswith("observation.") or key == "task"}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    dataset = load_dataset(with_action_chunk=False)
    selected = manifest["gate0a"][0]
    sample = dataset[selected["dataset_index"]]
    if int(sample["index"]) != selected["dataset_index"]:
        raise RuntimeError("Dataset index mismatch")

    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT, local_files_only=True)
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    batch = preprocessor(model_input(sample))
    report = {
        "checkpoint": CHECKPOINT,
        "policy_device": str(next(policy.parameters()).device),
        "policy_dtype": str(next(policy.parameters()).dtype),
        "task": sample["task"],
        "raw_state_shape": list(sample["observation.state"].shape),
        "batch": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
    }
    out = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0/substrate_smoke.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
