#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from flowspec_vla.data import load_dataset
from flowspec_vla.phase1 import (
    configure_libero_policy,
    load_phase1_config,
    load_split_manifest,
    make_phase1_processors,
    project_executable_subspace,
    validate_method,
)
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


K = 16
VALID = 7


def observation(sample: dict) -> dict:
    return {key: value for key, value in sample.items() if key.startswith("observation.") or key == "task"}


@torch.inference_mode()
def prefix_cache(policy: SmolVLAPolicy, batch: dict):
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    model = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state=state,
    )
    masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=masks,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
    )
    cache.batch_repeat_interleave(K)
    return prefix_pad_masks.repeat_interleave(K, dim=0), cache


@torch.inference_mode()
def integrate(policy, prefix_masks, cache, source: torch.Tensor, method: str):
    working_cache = copy.deepcopy(cache)
    x_t = source.to("cuda")
    if method == "m2":
        x_t = project_executable_subspace(x_t, VALID)
    valid_trace = [x_t[..., :VALID].cpu().numpy()]
    padded_state_rms = [torch.sqrt(x_t[..., VALID:].square().mean(dim=(1, 2))).cpu().numpy()]
    dt = -1.0 / policy.config.num_steps
    for step in range(policy.config.num_steps):
        timestep = torch.full((K,), 1.0 + step * dt, dtype=torch.float32, device="cuda")
        velocity = policy.model.denoise_step(prefix_masks, working_cache, x_t, timestep)
        x_t = x_t + dt * velocity
        if method == "m2":
            x_t = project_executable_subspace(x_t, VALID)
        valid_trace.append(x_t[..., :VALID].cpu().numpy())
        padded_state_rms.append(torch.sqrt(x_t[..., VALID:].square().mean(dim=(1, 2))).cpu().numpy())
    return np.stack(valid_trace).astype(np.float32), np.stack(padded_state_rms).astype(np.float32)


def noise_for_state(ordinal: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(410000 + ordinal)
    return torch.randn((K, 50, 32), generator=generator, dtype=torch.float32)


def interventions(source: torch.Tensor) -> dict[str, torch.Tensor]:
    padded = source.clone()
    padded[..., :VALID] = source[0:1, ..., :VALID]
    valid = source.clone()
    valid[..., VALID:] = source[0:1, ..., VALID:]
    zero = source[0:1].repeat(K, 1, 1)
    zero[..., VALID:] = 0.0
    return {"pad": padded, "valid": valid, "native": source, "zero": zero}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["m0", "m1", "m2"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-base", action="store_true")
    args = parser.parse_args()
    method = validate_method(args.method)
    config = load_phase1_config()
    artifact_root = Path(config["paths"]["artifacts"])
    gate0_root = artifact_root.parent / "gate0"
    manifest = json.loads((gate0_root / "state_manifest.json").read_text())
    if args.initial_base:
        if args.seed is not None:
            raise ValueError("Initial-base audit does not take a training seed")
        policy = configure_libero_policy(config)
        preprocessor, _ = make_phase1_processors(policy, load_split_manifest(config))
        suffix = f"initial_{method}"
    else:
        if args.seed not in config["training"]["seeds"]:
            raise ValueError("A frozen primary training seed is required")
        checkpoint = (
            Path(config["paths"]["trained_models"])
            / f"{method}_seed{args.seed}"
            / "update_05000"
        )
        policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
        processor_path = str(checkpoint)
        suffix = f"{method}_seed{args.seed}"
    policy.eval()
    if not args.initial_base:
        preprocessor, _ = make_pre_post_processors(
            policy.config,
            processor_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
    dataset = load_dataset(with_action_chunk=False)
    arrays = {name: [] for name in ["pad_trace", "valid_trace", "native_trace", "repeat_trace", "zero_trace"]}
    operational_padded_rms = []
    started = time.perf_counter()
    for ordinal, row in enumerate(manifest["gate0a"]):
        sample = dataset[row["dataset_index"]]
        batch = preprocessor(observation(sample))
        prefix_masks, cache = prefix_cache(policy, batch)
        source = noise_for_state(ordinal)
        variants = interventions(source)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for name, variant in variants.items():
                trace, _ = integrate(policy, prefix_masks, cache, variant, method)
                arrays[f"{name}_trace"].append(trace)
            repeat, _ = integrate(policy, prefix_masks, cache, variants["native"], method)
            arrays["repeat_trace"].append(repeat)
            operational = source.clone()
            if method in ("m1", "m2"):
                operational[..., VALID:] = 0.0
            _, padded_rms = integrate(policy, prefix_masks, cache, operational, method)
            operational_padded_rms.append(padded_rms)
        if (ordinal + 1) % 8 == 0:
            print(f"audit={suffix} state={ordinal + 1}/64", flush=True)
    torch.cuda.synchronize()
    output = artifact_root / "leakage"
    output.mkdir(parents=True, exist_ok=True)
    npz = output / f"{suffix}.npz"
    packed = {name: np.stack(value) for name, value in arrays.items()}
    packed["operational_padded_state_rms"] = np.stack(operational_padded_rms)
    packed["state_ordinals"] = np.arange(64, dtype=np.int64)
    np.savez_compressed(npz, **packed)
    metadata = {
        "method": method,
        "seed": args.seed,
        "initial_base": args.initial_base,
        "states": 64,
        "padded_interventions": 16,
        "valid_interventions": 16,
        "flow_steps": 10,
        "array": str(npz),
        "gpu_seconds": time.perf_counter() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / f"{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
