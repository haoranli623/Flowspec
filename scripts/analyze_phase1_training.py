#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowspec_vla.phase1 import METHODS, load_phase1_config, load_split_manifest


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def paired_summary(candidate: list[float], baseline: list[float]) -> dict:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    difference = candidate_array - baseline_array
    relative_reduction = (baseline_array - candidate_array) / baseline_array
    sd = difference.std(ddof=1)
    return {
        "paired_difference_candidate_minus_baseline": difference.tolist(),
        "paired_relative_reduction": relative_reduction.tolist(),
        "mean_difference": float(difference.mean()),
        "mean_relative_reduction": float((baseline_array.mean() - candidate_array.mean()) / baseline_array.mean()),
        "direction_favors_candidate_count": int(np.sum(difference < 0)),
        "paired_effect_size_dz": float(difference.mean() / sd) if sd > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_phase1_config(args.config) if args.config else load_phase1_config()
    split = load_split_manifest(config)
    root = Path(config["paths"]["artifacts"])
    seeds = config["training"]["seeds"]
    runs = {}
    for method in METHODS:
        for seed in seeds:
            path = root / "runs" / f"{method}_seed{seed}" / "run.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing primary run: {path}")
            runs[(method, seed)] = json.loads(path.read_text())

    expected_updates = config["training"]["updates"]
    expected_examples = expected_updates * config["training"]["effective_batch_size"]
    for (method, seed), run in runs.items():
        run_updates = run.get("updates", run.get("full_reference_updates"))
        run_examples = run.get("examples_seen", run.get("examples_seen_at_end"))
        if run_updates != expected_updates or run_examples != expected_examples:
            raise RuntimeError(f"Budget mismatch in {method}/{seed}")
        if "trainable_parameters" in run and run["trainable_parameters"] != 392_904_096:
            raise RuntimeError(f"Trainable-parameter mismatch in {method}/{seed}")
        if len(run["validation_log"]) != len(config["training"]["validation_updates"]):
            raise RuntimeError(f"Validation schedule mismatch in {method}/{seed}")
    for seed in seeds:
        hashes = {runs[(method, seed)]["training_order_sha256"] for method in METHODS}
        if len(hashes) != 1:
            raise RuntimeError(f"Training order differs across methods for seed {seed}")

    updates = config["training"]["validation_updates"]
    metrics_by_method = {}
    for method in METHODS:
        seed_curves = []
        seed_flow_curves = []
        final = []
        aulc = []
        component_final = {"translation": [], "rotation": [], "gripper": []}
        chunk_final = []
        for seed in seeds:
            log = runs[(method, seed)]["validation_log"]
            if [row["update"] for row in log] != updates:
                raise RuntimeError(f"Unexpected validation updates for {method}/{seed}")
            curve = [row["whole_action_normalized_physical_rmse"] for row in log]
            flow_curve = [row["fixed_flow_mse"] for row in log]
            seed_curves.append(curve)
            seed_flow_curves.append(flow_curve)
            final.append(curve[-1])
            aulc.append(float(np.trapezoid(curve, updates) / updates[-1]))
            component_final["translation"].append(log[-1]["translation_rmse"])
            component_final["rotation"].append(log[-1]["rotation_rmse"])
            component_final["gripper"].append(log[-1]["gripper_rmse"])
            chunk_final.append(log[-1]["chunk_normalized_physical_rmse"])
        chunk_final_array = np.asarray(chunk_final, dtype=np.float64)
        metrics_by_method[method] = {
            "seeds": seeds,
            "validation_updates": updates,
            "nrmse_curves": seed_curves,
            "flow_mse_curves": seed_flow_curves,
            "final_nrmse": mean_std(final),
            "normalized_aulc": mean_std(aulc),
            "final_translation_rmse": mean_std(component_final["translation"]),
            "final_rotation_rmse": mean_std(component_final["rotation"]),
            "final_gripper_rmse": mean_std(component_final["gripper"]),
            "final_chunk_normalized_physical_rmse_by_seed": chunk_final_array.tolist(),
            "final_chunk_normalized_physical_rmse_mean": chunk_final_array.mean(axis=0).tolist(),
            "final_chunk_normalized_physical_rmse_std": chunk_final_array.std(axis=0, ddof=1).tolist(),
        }

    comparisons = {}
    m0_final = metrics_by_method["m0"]["final_nrmse"]["values"]
    m0_aulc = metrics_by_method["m0"]["normalized_aulc"]["values"]
    for method in ("m1", "m2"):
        comparisons[f"{method}_vs_m0"] = {
            "final_nrmse": paired_summary(metrics_by_method[method]["final_nrmse"]["values"], m0_final),
            "normalized_aulc": paired_summary(metrics_by_method[method]["normalized_aulc"]["values"], m0_aulc),
        }
    comparisons["m2_vs_m1"] = {
        "final_nrmse": paired_summary(
            metrics_by_method["m2"]["final_nrmse"]["values"],
            metrics_by_method["m1"]["final_nrmse"]["values"],
        ),
        "normalized_aulc": paired_summary(
            metrics_by_method["m2"]["normalized_aulc"]["values"],
            metrics_by_method["m1"]["normalized_aulc"]["values"],
        ),
    }

    validation_summary = {
        "status": "COMPLETE",
        "protocol_freeze_commit": next(iter(runs.values()))["protocol_freeze_commit"],
        "counts": {
            "methods": 3,
            "seeds_per_method": 3,
            "runs": 9,
            "updates_per_run": expected_updates,
            "examples_per_run": expected_examples,
            "training_frames": split["training_frame_count"],
            "validation_states": split["validation_state_count"],
        },
        "methods": metrics_by_method,
        "paired_comparisons": comparisons,
        "thresholds": {
            "practical_final_relative_reduction": config["thresholds"]["practical_final_relative_reduction"],
            "practical_aulc_relative_reduction": config["thresholds"]["practical_aulc_relative_reduction"],
            "seed_direction_count": config["thresholds"]["seed_direction_count"],
            "equivalence_relative_band": config["thresholds"]["equivalence_relative_band"],
        },
    }
    (root / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2) + "\n")

    training_methods = {}
    for method in METHODS:
        training_methods[method] = {}
        for seed in seeds:
            run = runs[(method, seed)]
            losses = np.asarray([row["loss"] for row in run["train_log"]])
            gradients = np.asarray([row["gradient_norm_pre_clip"] for row in run["train_log"]])
            full_log = bool(run["train_log"] and run["train_log"][0]["update"] == 1)
            training_methods[method][str(seed)] = {
                "first_100_loss_mean": float(losses[:100].mean()) if full_log else None,
                "first_100_loss_status": "AVAILABLE" if full_log else "LOST_WITH_INTERRUPTED_PROCESS",
                "last_100_loss_mean": float(losses[-100:].mean()),
                "median_gradient_norm": float(np.median(gradients)),
                "max_gradient_norm": float(gradients.max()),
                "nonfinite_events": int((~np.isfinite(losses)).sum() + (~np.isfinite(gradients)).sum()),
                "mean_update_seconds": float(np.mean([row["update_seconds"] for row in run["train_log"]])),
                "gpu_hours": run["timing"]["gpu_hours"],
                "peak_memory_gib": run["timing"]["peak_memory_gib"],
                "final_valid_velocity_rms": run["train_log"][-1]["valid_velocity_rms"],
                "final_padded_velocity_rms": run["train_log"][-1]["padded_velocity_rms"],
                "training_order_sha256": run["training_order_sha256"],
                "train_log_update_range": [
                    run["train_log"][0]["update"],
                    run["train_log"][-1]["update"],
                ],
                "interruption_recovery": run.get("interruption_recovery"),
            }
    training_summary = {
        "status": "COMPLETE",
        "protocol_freeze_commit": next(iter(runs.values()))["protocol_freeze_commit"],
        "runs": 9,
        "total_optimizer_updates": 9 * expected_updates,
        "total_examples_seen": 9 * expected_examples,
        "methods": training_methods,
        "aggregate_gpu_hours": float(sum(run["timing"]["gpu_hours"] for run in runs.values())),
        "aggregate_gpu_hours_status": (
            "LOWER_BOUND_INTERRUPTED_INITIAL_SEGMENT_NOT_RECORDED"
            if any(run.get("interruption_recovery") for run in runs.values())
            else "COMPLETE"
        ),
    }
    (root / "training_summary.json").write_text(json.dumps(training_summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"m0": "#4C78A8", "m1": "#F58518", "m2": "#54A24B"}
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in METHODS:
        curves = np.asarray(metrics_by_method[method]["nrmse_curves"])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1)
        axis.plot(updates, mean, marker="o", color=colors[method], label=method.upper())
        axis.fill_between(updates, mean - std, mean + std, color=colors[method], alpha=0.18)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Validation physical-action NRMSE")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "validation_error_vs_updates.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    x = np.arange(len(METHODS))
    for method_index, method in enumerate(METHODS):
        values = metrics_by_method[method]["final_nrmse"]["values"]
        offsets = np.linspace(-0.08, 0.08, len(values))
        axis.scatter(np.full(len(values), x[method_index]) + offsets, values, s=45, color=colors[method])
        axis.hlines(np.mean(values), x[method_index] - 0.22, x[method_index] + 0.22, color="black", linewidth=2)
    axis.set_xticks(x, [method.upper() for method in METHODS])
    axis.set_ylabel("Final physical-action NRMSE")
    fig.tight_layout()
    fig.savefig(root / "per_seed_final_metric.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    labels = ["M1 - M0", "M2 - M0", "M2 - M1"]
    keys = ["m1_vs_m0", "m2_vs_m0", "m2_vs_m1"]
    for index, key in enumerate(keys):
        values = comparisons[key]["final_nrmse"]["paired_difference_candidate_minus_baseline"]
        axis.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, s=45)
        axis.hlines(np.mean(values), index - 0.22, index + 0.22, color="black", linewidth=2)
    axis.axhline(0.0, color="#777777", linewidth=1, linestyle="--")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("Paired final NRMSE difference")
    fig.tight_layout()
    fig.savefig(root / "paired_seed_differences.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"training_summary": str(root / "training_summary.json"), "validation_summary": str(root / "validation_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
