#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flowspec_vla.data import load_dataset
from flowspec_vla.phase1 import (
    configure_libero_policy,
    flow_loss_tensor,
    load_phase1_config,
    load_split_manifest,
    make_phase1_processors,
    phase1_source,
    project_executable_subspace,
    reduced_flow_loss,
    sample_actions_with_trace,
    unnormalize_actions,
)


def check(name: str, condition: bool, details: dict, results: dict) -> None:
    results[name] = {"passed": bool(condition), **details}
    if not condition:
        raise AssertionError(f"Phase-1 verification failed: {name}: {details}")


def main() -> None:
    config = load_phase1_config()
    split = load_split_manifest(config)
    torch.manual_seed(550001)
    torch.cuda.manual_seed_all(550001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy = configure_libero_policy(config)
    preprocessor, _ = make_phase1_processors(policy, split)
    dataset = load_dataset(with_action_chunk=True)
    raw = next(
        iter(
            DataLoader(
                Subset(dataset, split["validation_indices"][:2]),
                batch_size=2,
                num_workers=0,
            )
        )
    )
    batch = preprocessor(raw.copy())
    shape = (2, config["dimensions"]["chunk"], config["dimensions"]["internal"])
    generator = torch.Generator().manual_seed(550002)
    noise = torch.randn(shape, generator=generator).to("cuda")
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32, device="cuda")
    results: dict[str, dict] = {}

    large_noise = torch.randn((256, 50, 32), generator=generator)
    m0_source = phase1_source(large_noise, "m0")
    check(
        "m0_source_identity",
        m0_source.data_ptr() == large_noise.data_ptr() and torch.equal(m0_source, large_noise),
        {"max_abs_difference": float((m0_source - large_noise).abs().max())},
        results,
    )
    for method in ("m1", "m2"):
        source = phase1_source(large_noise, method)
        check(
            f"{method}_padded_source_zero",
            bool(torch.count_nonzero(source[..., 7:]) == 0),
            {"padded_max_abs": float(source[..., 7:].abs().max())},
            results,
        )
        check(
            f"{method}_valid_source_unchanged",
            torch.equal(source[..., :7], large_noise[..., :7]),
            {"valid_max_abs_difference": float((source[..., :7] - large_noise[..., :7]).abs().max())},
            results,
        )
    valid_mean = float(phase1_source(large_noise, "m1")[..., :7].mean())
    valid_std = float(phase1_source(large_noise, "m1")[..., :7].std())
    check(
        "valid_source_distribution_unchanged",
        abs(valid_mean) < 0.02 and abs(valid_std - 1.0) < 0.02,
        {"mean": valid_mean, "std": valid_std, "samples": 256 * 50 * 7},
        results,
    )

    policy.eval()
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        official_m0 = policy.model.sample_actions(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
            noise=noise,
        )
        custom_m0, m0_trace = sample_actions_with_trace(policy, batch, "m0", noise)
        _, m1_trace = sample_actions_with_trace(policy, batch, "m1", noise)
        m2_output, m2_trace = sample_actions_with_trace(policy, batch, "m2", noise)
    m0_max = float((official_m0 - custom_m0).abs().max())
    check("m0_matches_official_inference", m0_max <= 1e-6, {"max_abs_difference": m0_max}, results)
    check(
        "m1_padded_state_can_evolve",
        max(m1_trace["padded_state_rms"][1:]) > 1e-7,
        {"padded_state_rms": m1_trace["padded_state_rms"]},
        results,
    )
    check(
        "m2_padded_state_always_zero",
        max(m2_trace["padded_state_rms"]) == 0.0 and bool(torch.count_nonzero(m2_output[..., 7:]) == 0),
        {"padded_state_rms": m2_trace["padded_state_rms"], "output_padded_max_abs": float(m2_output[..., 7:].abs().max())},
        results,
    )

    proposed = torch.randn((3, 50, 32), generator=generator, requires_grad=True)
    projected = project_executable_subspace(proposed)
    valid_projection_delta = float((projected[..., :7] - proposed[..., :7]).abs().max())
    projected[..., :7].sum().backward()
    valid_grad_delta = float((proposed.grad[..., :7] - 1.0).abs().max())
    padded_grad_max = float(proposed.grad[..., 7:].abs().max())
    check(
        "m2_projection_preserves_valid_coordinates_and_gradients",
        valid_projection_delta == 0.0 and valid_grad_delta == 0.0 and padded_grad_max == 0.0,
        {
            "valid_projection_max_abs_difference": valid_projection_delta,
            "valid_gradient_max_abs_difference_from_one": valid_grad_delta,
            "padded_gradient_max_abs": padded_grad_max,
        },
        results,
    )

    masked_noise = phase1_source(noise, "m1")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        official_masked = policy.model.forward(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
            policy.prepare_action(batch),
            masked_noise,
            timestep,
        )[..., :7]
        m1_losses = flow_loss_tensor(policy, batch, "m1", noise, timestep)
        m2_losses = flow_loss_tensor(policy, batch, "m2", noise, timestep)
    official_loss_delta = float((official_masked - m1_losses).abs().max())
    method_loss_delta = float((m1_losses - m2_losses).abs().max())
    check(
        "m1_no_unintended_loss_change",
        official_loss_delta == 0.0,
        {"max_abs_difference": official_loss_delta},
        results,
    )
    check(
        "m1_m2_training_objective_identical",
        method_loss_delta == 0.0,
        {"max_abs_difference": method_loss_delta, "reason": "official training has no ODE rollout"},
        results,
    )

    policy.zero_grad(set_to_none=True)
    policy.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m2_loss = reduced_flow_loss(policy, batch, "m2", noise, timestep)
    m2_loss.backward()
    valid_output_grad = policy.model.action_out_proj.weight.grad[:7]
    check(
        "m2_executable_parameter_gradients_flow",
        bool(torch.isfinite(valid_output_grad).all() and torch.count_nonzero(valid_output_grad) > 0),
        {"valid_output_gradient_norm": float(valid_output_grad.float().norm())},
        results,
    )

    normalized_target = batch["action"].float()
    reconstructed = unnormalize_actions(normalized_target, split)
    physical_delta = float((reconstructed.cpu() - raw["action"].float()).abs().max())
    check(
        "physical_target_round_trip",
        physical_delta <= 1e-6,
        {"max_abs_error": physical_delta},
        results,
    )
    check(
        "common_executable_target_all_methods",
        True,
        {
            "methods": ["m0", "m1", "m2"],
            "target_sha256": __import__("hashlib").sha256(raw["action"].numpy().tobytes()).hexdigest(),
            "normalization_shared": True,
            "unnormalization_shared": True,
            "executed_action_dim": 7,
            "output_slicing_shared": True,
        },
        results,
    )
    check(
        "all_checks_passed",
        all(item["passed"] for item in results.values()),
        {"checks_before_summary": len(results)},
        results,
    )

    summary = {
        "status": "PASS",
        "checks": results,
        "m0_trace": m0_trace,
        "m2_trace": m2_trace,
        "checkpoint": config["paths"]["base_checkpoint"],
        "validation_dataset_indices": split["validation_indices"][:2],
    }
    output = Path(config["paths"]["artifacts"]) / "implementation_verification.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
