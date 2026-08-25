#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from flowspec_vla.data import load_dataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


CHECKPOINT = "/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7"
ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
MANIFEST = ARTIFACT_DIR / "state_manifest.json"
K = 16
CHUNK = 50
D = 32
VALID = 7


def model_input(sample: dict) -> dict:
    return {key: value for key, value in sample.items() if key.startswith("observation.") or key == "task"}


@torch.inference_mode()
def prefix_cache(policy: SmolVLAPolicy, batch: dict, repeats: int):
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
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
    )
    if cache is None:
        raise RuntimeError("Official inference did not return a prefix cache")
    cache.batch_repeat_interleave(repeats)
    return prefix_pad_masks.repeat_interleave(repeats, dim=0), cache


@torch.inference_mode()
def integrate_with_trace(policy: SmolVLAPolicy, prefix_pad_masks, cache, source: torch.Tensor) -> torch.Tensor:
    if source.shape != (K, CHUNK, D):
        raise ValueError(f"Unexpected source shape {tuple(source.shape)}")
    working_cache = copy.deepcopy(cache)
    x_t = source.to(device="cuda", dtype=torch.float32)
    traces = [x_t[:, :, :VALID].cpu().numpy()]
    dt = -1.0 / policy.config.num_steps
    for step in range(policy.config.num_steps):
        timestep = torch.full((K,), 1.0 + step * dt, dtype=torch.float32, device="cuda")
        velocity = policy.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=working_cache,
            x_t=x_t,
            timestep=timestep,
        )
        x_t = x_t + dt * velocity
        traces.append(x_t[:, :, :VALID].cpu().numpy())
    return np.stack(traces, axis=0).astype(np.float32, copy=False)


def native_noise(state_ordinal: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(410000 + state_ordinal)
    return torch.randn((K, CHUNK, D), generator=generator, dtype=torch.float32)


def interventions(source: torch.Tensor) -> dict[str, torch.Tensor]:
    pad = source.clone()
    pad[:, :, :VALID] = source[0:1, :, :VALID]
    valid = source.clone()
    valid[:, :, VALID:] = source[0:1, :, VALID:]
    zero = source[0:1].clone().repeat(K, 1, 1)
    zero[:, :, VALID:] = 0.0
    return {"pad": pad, "valid": valid, "native": source, "zero": zero}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("Invalid shard")

    torch.manual_seed(400001)
    torch.cuda.manual_seed_all(400001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    manifest = json.loads(MANIFEST.read_text())
    selected = [
        (ordinal, row)
        for ordinal, row in enumerate(manifest["gate0a"])
        if ordinal % args.num_shards == args.shard
    ]
    dataset = load_dataset(with_action_chunk=False)
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT, local_files_only=True)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    records = []
    arrays = {key: [] for key in ["pad_trace", "valid_trace", "native_trace", "repeat_trace", "zero_trace"]}
    started = time.perf_counter()
    gpu_seconds = 0.0

    for local_index, (ordinal, row) in enumerate(selected):
        sample = dataset[row["dataset_index"]]
        if int(sample["index"]) != row["dataset_index"] or sample["task"] != row["task"]:
            raise RuntimeError("Frozen state manifest does not match decoded dataset item")
        batch = preprocessor(model_input(sample))
        torch.cuda.synchronize()
        state_start = time.perf_counter()
        prefix_pad_masks, cache = prefix_cache(policy, batch, K)
        sources = interventions(native_noise(ordinal))
        traces = {}
        for name in ["pad", "valid", "native", "zero"]:
            traces[name] = integrate_with_trace(policy, prefix_pad_masks, cache, sources[name])
        traces["repeat"] = integrate_with_trace(policy, prefix_pad_masks, cache, sources["native"])
        torch.cuda.synchronize()
        state_seconds = time.perf_counter() - state_start
        gpu_seconds += state_seconds

        arrays["pad_trace"].append(traces["pad"])
        arrays["valid_trace"].append(traces["valid"])
        arrays["native_trace"].append(traces["native"])
        arrays["repeat_trace"].append(traces["repeat"])
        arrays["zero_trace"].append(traces["zero"])
        records.append({**row, "state_ordinal": ordinal, "noise_seed": 410000 + ordinal, "gpu_seconds": state_seconds})
        print(
            f"shard={args.shard} state={local_index + 1}/{len(selected)} ordinal={ordinal} "
            f"task={row['task_index']} seconds={state_seconds:.3f}",
            flush=True,
        )

    order = np.argsort([record["state_ordinal"] for record in records])
    packed = {
        name: np.stack(values, axis=0)[order]
        for name, values in arrays.items()
    }
    ordered_records = [records[index] for index in order]
    packed["state_ordinals"] = np.asarray([record["state_ordinal"] for record in ordered_records], dtype=np.int64)
    packed["task_indices"] = np.asarray([record["task_index"] for record in ordered_records], dtype=np.int64)
    packed["dataset_indices"] = np.asarray([record["dataset_index"] for record in ordered_records], dtype=np.int64)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = ARTIFACT_DIR / f"gate0a_raw_shard{args.shard}.npz"
    np.savez_compressed(npz_path, **packed)
    elapsed = time.perf_counter() - started
    metadata = {
        "protocol_commit": manifest["protocol_commit"],
        "shard": args.shard,
        "num_shards": args.num_shards,
        "states": ordered_records,
        "wall_seconds": elapsed,
        "gpu_seconds_approx": gpu_seconds,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": CHECKPOINT,
        "array_file": str(npz_path),
    }
    metadata_path = ARTIFACT_DIR / f"gate0a_run_shard{args.shard}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"npz": str(npz_path), "metadata": str(metadata_path), "wall_seconds": elapsed}))


if __name__ == "__main__":
    main()
