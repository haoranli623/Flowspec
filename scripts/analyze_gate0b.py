#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
TRANSFORMS = ["official", "local", "group_scaled"]
BOOTSTRAPS = 10_000


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    squared = (prediction - target) ** 2
    per_state_mse = squared.mean(axis=(1, 2))
    return {
        "rmse": float(np.sqrt(squared.mean())),
        "per_state_mse": per_state_mse,
        "translation_0_3_rmse": float(np.sqrt(squared[:, :, :3].mean())),
        "rotation_3_6_rmse": float(np.sqrt(squared[:, :, 3:6].mean())),
        "gripper_6_rmse": float(np.sqrt(squared[:, :, 6].mean())),
        "chunk_time_rmse": np.sqrt(squared.mean(axis=(0, 2))),
    }


def paired_relative_ci(variant_mse: np.ndarray, official_mse: np.ndarray, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    n = len(official_mse)
    indices = generator.integers(0, n, size=(BOOTSTRAPS, n))
    official_rmse = np.sqrt(official_mse[indices].mean(axis=1))
    variant_rmse = np.sqrt(variant_mse[indices].mean(axis=1))
    relative = (variant_rmse - official_rmse) / official_rmse
    return [float(x) for x in np.quantile(relative, [0.025, 0.975])]


def main() -> None:
    arrays = {name: np.load(ARTIFACT_DIR / f"gate0b_{name}.npz") for name in TRANSFORMS}
    runs = {name: json.loads((ARTIFACT_DIR / f"gate0b_{name}_run.json").read_text()) for name in TRANSFORMS}
    target = arrays["official"]["target_physical"]
    for name in TRANSFORMS[1:]:
        if not np.array_equal(target, arrays[name]["target_physical"]):
            raise RuntimeError("Coordinate runs do not share exact physical validation targets")

    baseline = {name: metrics(arrays[name]["baseline_physical"], target) for name in TRANSFORMS}
    final = {name: metrics(arrays[name]["final_physical"], target) for name in TRANSFORMS}
    comparisons = {}
    b1 = False
    b2 = False
    official_rmse = final["official"]["rmse"]
    official_curve = {x["update"]: x["physicalized_velocity_mse"] for x in runs["official"]["validation_log"]}

    for offset, name in enumerate(TRANSFORMS[1:]):
        delta = (final[name]["rmse"] - official_rmse) / official_rmse
        ci = paired_relative_ci(final[name]["per_state_mse"], final["official"]["per_state_mse"], 440001 + offset)
        curve = {x["update"]: x["physicalized_velocity_mse"] for x in runs[name]["validation_log"]}
        direction = np.sign(delta)
        direction_matches = sum(
            np.sign(curve[update] - official_curve[update]) == direction for update in [16, 32, 48, 64]
        )
        excludes_neutral = ci[0] > 0.02 or ci[1] < -0.02
        strong = abs(delta) >= 0.10 and excludes_neutral and direction_matches >= 3
        weak = abs(delta) >= 0.02 or ci[0] > 0.0 or ci[1] < 0.0
        b1 = b1 or strong
        b2 = b2 or weak
        comparisons[name] = {
            "relative_final_physical_rmse": float(delta),
            "paired_bootstrap_95ci": ci,
            "post_baseline_direction_matches": int(direction_matches),
            "strong_condition": bool(strong),
            "weak_condition": bool(weak),
        }

    primary_verdict = "B1_STRONG_SIGNAL" if b1 else "B2_WEAK_SIGNAL" if b2 else "B3_NO_MATERIAL_SIGNAL"

    replicate_arrays = {
        name: np.load(ARTIFACT_DIR / f"gate0b_{name}_seed430002.npz") for name in TRANSFORMS
    }
    replicate_runs = {
        name: json.loads((ARTIFACT_DIR / f"gate0b_{name}_seed430002_run.json").read_text())
        for name in TRANSFORMS
    }
    replicate_final = {
        name: metrics(replicate_arrays[name]["final_physical"], target) for name in TRANSFORMS
    }
    replicate_official_rmse = replicate_final["official"]["rmse"]
    replicate_official_curve = {
        x["update"]: x["physicalized_velocity_mse"]
        for x in replicate_runs["official"]["validation_log"]
    }
    replicate_comparisons = {}
    robust_strong = False
    any_signal = b2
    for offset, name in enumerate(TRANSFORMS[1:]):
        delta = (replicate_final[name]["rmse"] - replicate_official_rmse) / replicate_official_rmse
        ci = paired_relative_ci(
            replicate_final[name]["per_state_mse"],
            replicate_final["official"]["per_state_mse"],
            440101 + offset,
        )
        curve = {
            x["update"]: x["physicalized_velocity_mse"]
            for x in replicate_runs[name]["validation_log"]
        }
        direction = np.sign(delta)
        direction_matches = sum(
            np.sign(curve[update] - replicate_official_curve[update]) == direction
            for update in [16, 32, 48, 64]
        )
        strong = (
            abs(delta) >= 0.10
            and (ci[0] > 0.02 or ci[1] < -0.02)
            and direction_matches >= 3
        )
        weak = abs(delta) >= 0.02 or ci[0] > 0.0 or ci[1] < 0.0
        same_direction = np.sign(delta) == np.sign(comparisons[name]["relative_final_physical_rmse"])
        robust_strong = robust_strong or (
            comparisons[name]["strong_condition"] and strong and same_direction
        )
        any_signal = any_signal or weak
        replicate_comparisons[name] = {
            "relative_final_physical_rmse": float(delta),
            "paired_bootstrap_95ci": ci,
            "post_baseline_direction_matches": int(direction_matches),
            "strong_condition": bool(strong),
            "weak_condition": bool(weak),
            "same_direction_as_primary_seed": bool(same_direction),
        }

    # B1 requires a clear reproducible signal. Once the budget-permitted replicate
    # exists, a primary-only large effect is conservatively classified as the
    # protocol's explicitly seed-sensitive B2 case.
    verdict = (
        "B1_STRONG_SIGNAL"
        if robust_strong
        else "B2_WEAK_SIGNAL"
        if any_signal
        else "B3_NO_MATERIAL_SIGNAL"
    )
    serializable_baseline = {
        name: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items() if key != "per_state_mse"}
        for name, result in baseline.items()
    }
    serializable_final = {
        name: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items() if key != "per_state_mse"}
        for name, result in final.items()
    }
    serializable_replicate_final = {
        name: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
            if key != "per_state_mse"
        }
        for name, result in replicate_final.items()
    }
    summary = {
        "protocol_commit": "337c95305698e527ef09db6f53040fb377a9d6a5",
        "verdict": verdict,
        "counts": {
            "transforms": 3,
            "training_anchors": 128,
            "validation_states": 32,
            "updates_per_transform": 64,
            "seeds": 2,
            "total_optimizer_updates": 384,
        },
        "baseline_physical_metrics": serializable_baseline,
        "final_physical_metrics": serializable_final,
        "comparisons_to_official": comparisons,
        "primary_seed_verdict": primary_verdict,
        "seed_replication": {
            "seed": 430002,
            "final_physical_metrics": serializable_replicate_final,
            "comparisons_to_official": replicate_comparisons,
            "robust_strong_condition": robust_strong,
        },
        "validation_trajectories": {
            name: runs[name]["validation_log"] for name in TRANSFORMS
        },
        "training": {
            name: {
                "first_raw_flow_mse": runs[name]["train_log"][0]["raw_flow_mse"],
                "last_raw_flow_mse": runs[name]["train_log"][-1]["raw_flow_mse"],
                "median_gradient_norm": float(np.median([x["gradient_norm_pre_clip"] for x in runs[name]["train_log"]])),
                "mean_update_seconds": float(
                    np.mean([x["update_seconds"] for x in runs[name]["train_log"]])
                ),
            }
            for name in TRANSFORMS
        },
        "timing": {
            "aggregate_gpu_seconds": float(
                sum(runs[name]["timing"]["gpu_active_wall_seconds"] for name in TRANSFORMS)
                + sum(
                    replicate_runs[name]["timing"]["gpu_active_wall_seconds"]
                    for name in TRANSFORMS
                )
            ),
            "aggregate_gpu_hours": float(
                (
                    sum(runs[name]["timing"]["gpu_active_wall_seconds"] for name in TRANSFORMS)
                    + sum(
                        replicate_runs[name]["timing"]["gpu_active_wall_seconds"]
                        for name in TRANSFORMS
                    )
                )
                / 3600
            ),
            "gpu_names": [runs[name]["timing"]["gpu_name"] for name in TRANSFORMS]
            + [replicate_runs[name]["timing"]["gpu_name"] for name in TRANSFORMS],
        },
        "thresholds": {"strong_relative_rmse": 0.10, "neutral_band": 0.02, "weak_relative_rmse": 0.02},
    }
    (ARTIFACT_DIR / "gate0b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in TRANSFORMS:
        log = runs[name]["validation_log"]
        axes[0].plot([x["update"] for x in log], [x["physicalized_velocity_mse"] for x in log], marker="o", label=name)
        axes[1].plot(range(50), final[name]["chunk_time_rmse"], label=name)
    axes[0].set_xlabel("optimizer update")
    axes[0].set_ylabel("physicalized validation velocity MSE")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("chunk step")
    axes[1].set_ylabel("final physical action RMSE")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "gate0b_dynamics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
