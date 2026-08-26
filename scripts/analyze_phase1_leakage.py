#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowspec_vla.phase1 import METHODS, load_phase1_config


def bootstrap_median(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(values), size=(resamples, len(values)))
    medians = np.median(values[index], axis=1)
    return np.quantile(medians, [0.025, 0.975]).astype(float).tolist()


def metrics(path: Path, bootstrap_seed: int, resamples: int) -> dict:
    with np.load(path) as data:
        pad = data["pad_trace"][:, -1]
        valid = data["valid_trace"][:, -1]
        native = data["native_trace"][:, -1]
        repeat = data["repeat_trace"][:, -1]
        pad_variance = np.var(pad, axis=1, ddof=1).mean(axis=(1, 2))
        valid_variance = np.var(valid, axis=1, ddof=1).mean(axis=(1, 2))
        pad_rms = np.sqrt(pad_variance)
        valid_rms = np.sqrt(valid_variance)
        operational = data["operational_padded_state_rms"]
        return {
            "median_padded_rms": float(np.median(pad_rms)),
            "padded_rms_bootstrap_95ci": bootstrap_median(pad_rms, bootstrap_seed, resamples),
            "median_valid_rms": float(np.median(valid_rms)),
            "valid_rms_bootstrap_95ci": bootstrap_median(valid_rms, bootstrap_seed + 1, resamples),
            "r_leak": float(np.median(pad_variance) / np.median(valid_variance)),
            "repeat_rms": float(np.sqrt(np.mean((native - repeat) ** 2))),
            "per_state_padded_rms": pad_rms.astype(float).tolist(),
            "per_state_valid_rms": valid_rms.astype(float).tolist(),
            "operational_padded_state_rms_by_step": np.median(operational, axis=(0, 2)).astype(float).tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_phase1_config(args.config) if args.config else load_phase1_config()
    root = Path(config["paths"]["artifacts"])
    leakage_root = root / "leakage"
    seeds = config["training"]["seeds"]
    resamples = config["validation"]["bootstrap_resamples"]
    bootstrap_seed = config["validation"]["bootstrap_seed"]
    initial = {
        method: metrics(leakage_root / f"initial_{method}.npz", bootstrap_seed + index * 10, resamples)
        for index, method in enumerate(METHODS)
    }
    final = {}
    for method_index, method in enumerate(METHODS):
        final[method] = {}
        for seed_index, seed in enumerate(seeds):
            final[method][str(seed)] = metrics(
                leakage_root / f"{method}_seed{seed}.npz",
                bootstrap_seed + 100 + method_index * 10 + seed_index * 2,
                resamples,
            )
    gate0 = json.loads((root.parent / "gate0" / "gate0a_summary.json").read_text())
    thresholds = config["thresholds"]
    method_summary = {}
    for method in METHODS:
        padded = [final[method][str(seed)]["median_padded_rms"] for seed in seeds]
        ratios = [final[method][str(seed)]["r_leak"] for seed in seeds]
        method_summary[method] = {
            "median_padded_rms_by_seed": padded,
            "r_leak_by_seed": ratios,
            "mean_median_padded_rms": float(np.mean(padded)),
            "mean_r_leak": float(np.mean(ratios)),
            "strong_suppression_seed_count": int(
                sum(
                    rms <= thresholds["mechanism_rms_floor"]
                    and ratio <= thresholds["mechanism_ratio_floor"]
                    for rms, ratio in zip(padded, ratios, strict=True)
                )
            ),
        }
    summary = {
        "status": "COMPLETE",
        "protocol_freeze_commit": "beb292024fcb37d321338474249d03598cfa5e90",
        "counts": {"states": 64, "padded_interventions": 16, "valid_interventions": 16, "seeds": 3},
        "gate0_task_checkpoint_reference": {
            "median_padded_rms": gate0["primary"]["median_rms_pad_standardized"],
            "median_valid_rms": gate0["primary"]["median_rms_valid_standardized"],
            "r_leak": gate0["primary"]["r_leak_ratio_of_median_variances"],
        },
        "base_initial": initial,
        "trained": final,
        "method_summary": method_summary,
        "thresholds": {
            "mechanism_rms_floor": thresholds["mechanism_rms_floor"],
            "mechanism_ratio_floor": thresholds["mechanism_ratio_floor"],
        },
    }
    (root / "leakage_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"m0": "#4C78A8", "m1": "#F58518", "m2": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(METHODS))
    for idx, method in enumerate(METHODS):
        initial_value = initial[method]["median_padded_rms"]
        final_values = [final[method][str(seed)]["median_padded_rms"] for seed in seeds]
        axes[0].scatter(idx - 0.16, initial_value, marker="D", color=colors[method], alpha=0.7)
        axes[0].scatter(np.full(3, idx + 0.08), final_values, color=colors[method])
        axes[0].hlines(np.mean(final_values), idx - 0.02, idx + 0.20, color="black", linewidth=1.5)
        trace = np.asarray([final[method][str(seed)]["operational_padded_state_rms_by_step"] for seed in seeds])
        axes[1].plot(range(11), trace.mean(axis=0), marker="o", color=colors[method], label=method.upper())
        axes[1].fill_between(range(11), trace.mean(axis=0) - trace.std(axis=0, ddof=1), trace.mean(axis=0) + trace.std(axis=0, ddof=1), color=colors[method], alpha=0.16)
    axes[0].set_xticks(x, [method.upper() for method in METHODS])
    axes[0].set_yscale("symlog", linthresh=1e-8)
    axes[0].set_ylabel("Padded-intervention executable RMS")
    axes[0].set_title("Diamond: base; circles: trained seeds")
    axes[1].set_xlabel("Euler state (0=source, 10=output)")
    axes[1].set_ylabel("Operational padded-state RMS")
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "leakage_and_padded_state.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"summary": str(root / "leakage_audit_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
