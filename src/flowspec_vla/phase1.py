from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies import make_pre_post_processors


Method = Literal["m0", "m1", "m2"]
METHODS: tuple[Method, ...] = ("m0", "m1", "m2")
PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
DEFAULT_CONFIG = PROJECT / "config/phase1.yaml"


def load_phase1_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def validate_method(method: str) -> Method:
    if method not in METHODS:
        raise ValueError(f"Unknown Phase-1 method {method!r}; expected one of {METHODS}")
    return method  # type: ignore[return-value]


def phase1_source(noise: Tensor, method: Method, valid_dim: int = 7) -> Tensor:
    """Apply the sole training/source intervention without modifying valid coordinates."""
    validate_method(method)
    if method == "m0":
        return noise
    source = noise.clone()
    source[..., valid_dim:] = 0.0
    return source


def project_executable_subspace(state: Tensor, valid_dim: int = 7) -> Tensor:
    """Project only padded coordinates to zero while preserving the autograd graph."""
    valid = state[..., :valid_dim]
    padded = torch.zeros_like(state[..., valid_dim:])
    return torch.cat([valid, padded], dim=-1)


def configure_libero_policy(config: dict[str, Any]) -> SmolVLAPolicy:
    """Load base weights with the released official LIBERO fine-tuning configuration."""
    base = config["paths"]["base_checkpoint"]
    recipe = config["paths"]["libero_recipe_checkpoint"]
    policy_config = SmolVLAPolicy.config_class.from_pretrained(recipe, local_files_only=True)
    policy_config.pretrained_path = base
    policy_config.repo_id = None
    policy_config.push_to_hub = False
    policy_config.device = "cuda"
    policy_config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(config["dimensions"]["executable"],))
    }
    return SmolVLAPolicy.from_pretrained(
        base,
        config=policy_config,
        local_files_only=True,
    )


def tensor_stats(stats: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    return {
        key: {
            statistic: torch.as_tensor(value, dtype=torch.float32)
            for statistic, value in values.items()
        }
        for key, values in stats.items()
    }


def make_phase1_processors(policy: SmolVLAPolicy, split_manifest: dict[str, Any]):
    stats = tensor_stats(split_manifest["normalization_stats_renamed"])
    config = load_phase1_config()
    rename_map = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    return make_pre_post_processors(
        policy.config,
        config["paths"]["libero_recipe_checkpoint"],
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": rename_map},
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": stats,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": stats,
            },
            "device_processor": {"device": "cpu"},
        },
    )


def flow_loss_tensor(
    policy: SmolVLAPolicy,
    batch: dict[str, Tensor],
    method: Method,
    noise: Tensor,
    timestep: Tensor,
) -> Tensor:
    """Return official per-coordinate FM MSE with only the source intervention applied."""
    valid_dim = policy.config.action_feature.shape[0]
    actions = policy.prepare_action(batch)
    source = phase1_source(noise, method, valid_dim)
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    losses = policy.model.forward(
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state,
        actions,
        source,
        timestep,
    )
    return losses[..., :valid_dim]


def reduced_flow_loss(
    policy: SmolVLAPolicy,
    batch: dict[str, Tensor],
    method: Method,
    noise: Tensor,
    timestep: Tensor,
) -> Tensor:
    losses = flow_loss_tensor(policy, batch, method, noise, timestep)
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is None:
        return losses.mean()
    keep = (~action_is_pad).unsqueeze(-1)
    denominator = (keep.sum() * losses.shape[-1]).clamp_min(1)
    return (losses * keep).sum() / denominator


@torch.inference_mode()
def sample_actions_with_trace(
    policy: SmolVLAPolicy,
    batch: dict[str, Tensor],
    method: Method,
    noise: Tensor,
    *,
    collect_trace: bool = True,
) -> tuple[Tensor, dict[str, list[float]]]:
    """Run the official Euler path, adding only source masking and M2 projection."""
    valid_dim = policy.config.action_feature.shape[0]
    source = phase1_source(noise, method, valid_dim)
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

    x_t = project_executable_subspace(source, valid_dim) if method == "m2" else source
    trace: dict[str, list[float]] = {
        "padded_state_rms": [],
        "valid_state_rms": [],
        "padded_velocity_rms": [],
        "valid_velocity_rms": [],
    }
    if collect_trace:
        trace["padded_state_rms"].append(
            float(torch.sqrt(torch.mean(x_t[..., valid_dim:].square())))
        )
        trace["valid_state_rms"].append(
            float(torch.sqrt(torch.mean(x_t[..., :valid_dim].square())))
        )
    dt = -1.0 / model.config.num_steps
    for step in range(model.config.num_steps):
        timestep = torch.full(
            (x_t.shape[0],), 1.0 + step * dt, dtype=torch.float32, device=x_t.device
        )
        velocity = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=cache,
            x_t=x_t,
            timestep=timestep,
        )
        if collect_trace:
            trace["padded_velocity_rms"].append(
                float(torch.sqrt(torch.mean(velocity[..., valid_dim:].square())))
            )
            trace["valid_velocity_rms"].append(
                float(torch.sqrt(torch.mean(velocity[..., :valid_dim].square())))
            )
        x_t = x_t + dt * velocity
        if method == "m2":
            x_t = project_executable_subspace(x_t, valid_dim)
        if collect_trace:
            trace["padded_state_rms"].append(
                float(torch.sqrt(torch.mean(x_t[..., valid_dim:].square())))
            )
            trace["valid_state_rms"].append(
                float(torch.sqrt(torch.mean(x_t[..., :valid_dim].square())))
            )
    return x_t, trace


def cosine_lr(
    update: int,
    *,
    peak_lr: float,
    final_lr: float,
    warmup_updates: int,
    decay_updates: int,
) -> float:
    if update <= 0:
        return 0.0
    if update <= warmup_updates:
        return peak_lr * update / warmup_updates
    progress = min(1.0, (update - warmup_updates) / (decay_updates - warmup_updates))
    return final_lr + 0.5 * (peak_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


def seeded_flow_randomness(
    generator: torch.Generator,
    batch_size: int,
    chunk: int,
    internal_dim: int,
) -> tuple[Tensor, Tensor]:
    noise = torch.randn((batch_size, chunk, internal_dim), generator=generator, dtype=torch.float32)
    # Beta(1.5, 1.0) has inverse CDF u**(2/3); preserve the official scale and offset.
    timestep = torch.rand((batch_size,), generator=generator, dtype=torch.float32).pow(2.0 / 3.0)
    timestep = timestep * 0.999 + 0.001
    return noise, timestep


def unnormalize_actions(actions: Tensor, split_manifest: dict[str, Any]) -> Tensor:
    stats = split_manifest["normalization_stats_raw"]["action"]
    mean = torch.as_tensor(stats["mean"], device=actions.device, dtype=actions.dtype)
    std = torch.as_tensor(stats["std"], device=actions.device, dtype=actions.dtype)
    return actions * std + mean


def normalized_physical_squared_error(
    prediction_physical: Tensor,
    target_physical: Tensor,
    split_manifest: dict[str, Any],
) -> Tensor:
    std = torch.as_tensor(
        split_manifest["normalization_stats_raw"]["action"]["std"],
        device=prediction_physical.device,
        dtype=prediction_physical.dtype,
    )
    return ((prediction_physical - target_physical) / std).square()


def load_split_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["paths"]["artifacts"]) / "split_manifest.json"
    return json.loads(path.read_text())


def rms(tensor: Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float().square())))


def finite_or_raise(name: str, value: Tensor | float) -> None:
    finite = bool(torch.isfinite(value).all()) if isinstance(value, Tensor) else bool(np.isfinite(value))
    if not finite:
        raise FloatingPointError(f"Non-finite Phase-1 value: {name}")
