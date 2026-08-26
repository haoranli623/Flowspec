#!/usr/bin/env python
"""Verify that M1 and M2 differ only in inference-time state projection.

SmolVLA's official post-training objective evaluates one random flow time and
contains no iterative state update. Under the frozen Phase-1 definition, M1
and M2 therefore have exactly the same training computation. Matched seeds
must produce identical scientific training fields and checkpoint tensors;
their generated-action validation metrics may differ because M2 projects at
each inference integration step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from flowspec_vla.phase1 import load_phase1_config


def checkpoint_difference(left: Path, right: Path) -> dict:
    max_abs = 0.0
    differing = 0
    tensors = 0
    with safe_open(left, framework="pt", device="cpu") as lhs, safe_open(
        right, framework="pt", device="cpu"
    ) as rhs:
        left_keys = lhs.keys()
        right_keys = rhs.keys()
        if left_keys != right_keys:
            raise RuntimeError("M1/M2 checkpoint tensor keys differ")
        for key in left_keys:
            a = lhs.get_tensor(key)
            b = rhs.get_tensor(key)
            difference = float(torch.max(torch.abs(a.float() - b.float())))
            max_abs = max(max_abs, difference)
            differing += int(difference != 0.0)
            tensors += 1
    return {"tensor_count": tensors, "differing_tensor_count": differing, "max_abs": max_abs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_phase1_config(args.config) if args.config else load_phase1_config()
    artifact_root = Path(config["paths"]["artifacts"])
    model_root = Path(config["paths"]["trained_models"])
    if config.get("phase") == "phase1_rerun":
        model_root = model_root / "runs"
    rows = []
    for seed in config["training"]["seeds"]:
        m1_run = json.loads((artifact_root / "runs" / f"m1_seed{seed}" / "run.json").read_text())
        m2_run = json.loads((artifact_root / "runs" / f"m2_seed{seed}" / "run.json").read_text())
        training_fields = [
            "update",
            "loss",
            "gradient_norm_pre_clip",
            "valid_velocity_rms",
            "padded_velocity_rms",
            "lr",
            "examples_seen",
            "sample_index_first",
            "sample_index_last",
        ]
        m1_by_update = {row["update"]: row for row in m1_run["train_log"]}
        m2_by_update = {row["update"]: row for row in m2_run["train_log"]}
        common_updates = sorted(set(m1_by_update) & set(m2_by_update))
        log_equal = bool(common_updates) and all(
            all(m1_by_update[update][field] == m2_by_update[update][field] for field in training_fields)
            for update in common_updates
        )
        m1_validation = {row["update"]: row for row in m1_run["validation_log"]}
        m2_validation = {row["update"]: row for row in m2_run["validation_log"]}
        common_flow_updates = sorted(
            update
            for update in set(m1_validation) & set(m2_validation)
            if m1_validation[update]["fixed_flow_mse"] is not None
            and m2_validation[update]["fixed_flow_mse"] is not None
        )
        fixed_flow_equal = all(
            m1_validation[update]["fixed_flow_mse"] == m2_validation[update]["fixed_flow_mse"]
            for update in common_flow_updates
        )
        tensor_check = checkpoint_difference(
            model_root / f"m1_seed{seed}" / "update_05000" / "model.safetensors",
            model_root / f"m2_seed{seed}" / "update_05000" / "model.safetensors",
        )
        rows.append(
            {
                "seed": seed,
                "training_log_exact": log_equal,
                "training_log_common_update_range": [common_updates[0], common_updates[-1]],
                "training_log_common_update_count": len(common_updates),
                "fixed_flow_validation_exact": fixed_flow_equal,
                "fixed_flow_common_updates": common_flow_updates,
                "generated_action_metrics_expected_to_differ": True,
                "checkpoint": tensor_check,
            }
        )
    passed = all(
        row["training_log_exact"]
        and row["fixed_flow_validation_exact"]
        and row["checkpoint"]["max_abs"] == 0.0
        for row in rows
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "scientific_interpretation": (
            "M1 and M2 training are exactly identical; their frozen distinction is per-step "
            "inference projection."
        ),
        "seeds": rows,
    }
    output = artifact_root / "m1_m2_training_invariance.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
