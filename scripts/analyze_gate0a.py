#!/usr/bin/env python
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.torch import load_file


ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
CHECKPOINT = Path("/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 420001


def bootstrap_median(values: np.ndarray, generator: np.random.Generator) -> list[float]:
    n = len(values)
    indices = generator.integers(0, n, size=(BOOTSTRAPS, n))
    medians = np.median(values[indices], axis=1)
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def maximum_pairwise_rms(samples: np.ndarray) -> float:
    return float(
        max(
            np.sqrt(np.mean((samples[i] - samples[j]) ** 2))
            for i, j in itertools.combinations(range(samples.shape[0]), 2)
        )
    )


def main() -> None:
    shard_data = [np.load(ARTIFACT_DIR / f"gate0a_raw_shard{shard}.npz") for shard in range(2)]
    keys = shard_data[0].files
    merged = {key: np.concatenate([data[key] for data in shard_data], axis=0) for key in keys}
    order = np.argsort(merged["state_ordinals"])
    merged = {key: value[order] for key, value in merged.items()}
    if merged["state_ordinals"].tolist() != list(range(64)):
        raise RuntimeError("Gate 0A shards do not cover the frozen 64 states exactly")

    pad = merged["pad_trace"][:, -1]
    valid = merged["valid_trace"][:, -1]
    native = merged["native_trace"][:, -1]
    repeat = merged["repeat_trace"][:, -1]
    zero = merged["zero_trace"][:, -1, 0]

    stats = load_file(str(CHECKPOINT / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    mean = stats["action.mean"].numpy()
    std = stats["action.std"].numpy()
    physical = lambda x: x * std + mean

    var_pad = np.var(pad, axis=1, ddof=1).mean(axis=(1, 2))
    var_valid = np.var(valid, axis=1, ddof=1).mean(axis=(1, 2))
    rms_pad = np.sqrt(var_pad)
    rms_valid = np.sqrt(var_valid)
    repeat_delta = native - repeat
    repeat_rms = float(np.sqrt(np.mean(repeat_delta**2)))
    repeat_max_abs = float(np.max(np.abs(repeat_delta)))
    zero_delta = physical(zero) - physical(native[:, 0])
    zero_state_rms = np.sqrt(np.mean((zero_delta / std) ** 2, axis=(1, 2)))
    ratio = float(np.median(var_pad) / np.median(var_valid))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pad_ci = bootstrap_median(rms_pad, rng)
    valid_ci = bootstrap_median(rms_valid, rng)
    floor = max(10.0 * repeat_rms, 1e-4)
    implementation_valid = float(np.median(rms_valid)) >= 1e-4 and repeat_max_abs <= 1e-5
    if not implementation_valid:
        verdict = "A_INVALID"
    elif float(np.median(rms_pad)) >= 0.01 and ratio >= 0.01 and pad_ci[0] > floor:
        verdict = "A1_STRONG_SIGNAL"
    elif float(np.median(rms_pad)) > floor and (ratio >= 1e-4 or pad_ci[0] > floor):
        verdict = "A2_WEAK_SIGNAL"
    else:
        verdict = "A3_NO_MATERIAL_SIGNAL"

    per_dim_pad = np.sqrt(np.var(pad, axis=1, ddof=1).mean(axis=(0, 1)))
    per_dim_valid = np.sqrt(np.var(valid, axis=1, ddof=1).mean(axis=(0, 1)))
    pairwise = np.asarray([maximum_pairwise_rms(x) for x in pad])
    raw_physical_rms = np.sqrt(np.var(physical(pad), axis=1, ddof=1).mean(axis=(1, 2)))

    step_pad = np.sqrt(np.var(merged["pad_trace"], axis=2, ddof=1).mean(axis=(0, 2, 3)))
    step_valid = np.sqrt(np.var(merged["valid_trace"], axis=2, ddof=1).mean(axis=(0, 2, 3)))
    task_summary = {}
    rows = []
    for i in range(64):
        task = int(merged["task_indices"][i])
        rows.append(
            {
                "state_ordinal": i,
                "task_index": task,
                "dataset_index": int(merged["dataset_indices"][i]),
                "rms_pad_standardized": float(rms_pad[i]),
                "rms_valid_standardized": float(rms_valid[i]),
                "variance_ratio": float(var_pad[i] / var_valid[i]),
                "max_pairwise_rms_standardized": float(pairwise[i]),
                "rms_pad_raw_physical": float(raw_physical_rms[i]),
                "zero_pad_rms_standardized": float(zero_state_rms[i]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(ARTIFACT_DIR / "gate0a_per_state.csv", index=False)
    for task, group in frame.groupby("task_index"):
        task_summary[str(task)] = {
            "states": len(group),
            "median_rms_pad_standardized": float(group.rms_pad_standardized.median()),
            "median_rms_valid_standardized": float(group.rms_valid_standardized.median()),
            "median_state_variance_ratio": float(group.variance_ratio.median()),
        }

    run_meta = [json.loads((ARTIFACT_DIR / f"gate0a_run_shard{s}.json").read_text()) for s in range(2)]
    summary = {
        "protocol_commit": "337c95305698e527ef09db6f53040fb377a9d6a5",
        "verdict": verdict,
        "implementation_valid": implementation_valid,
        "counts": {"states": 64, "tasks": 4, "k_pad": 16, "k_valid": 16, "flow_steps": 10},
        "primary": {
            "median_rms_pad_standardized": float(np.median(rms_pad)),
            "median_rms_pad_bootstrap_95ci": pad_ci,
            "median_rms_valid_standardized": float(np.median(rms_valid)),
            "median_rms_valid_bootstrap_95ci": valid_ci,
            "r_leak_ratio_of_median_variances": ratio,
            "median_rms_pad_raw_physical_coordinates": float(np.median(raw_physical_rms)),
            "median_max_pairwise_rms_standardized": float(np.median(pairwise)),
        },
        "controls": {
            "repeat_rms_standardized": repeat_rms,
            "repeat_max_abs_standardized": repeat_max_abs,
            "decision_floor_standardized": floor,
            "median_zero_pad_rms_standardized": float(np.median(zero_state_rms)),
        },
        "per_dimension": {
            "pad_rms_standardized": per_dim_pad.tolist(),
            "valid_rms_standardized": per_dim_valid.tolist(),
        },
        "by_task": task_summary,
        "by_euler_step": {
            "step": list(range(11)),
            "pad_rms_standardized": step_pad.tolist(),
            "valid_rms_standardized": step_valid.tolist(),
        },
        "timing": {
            "aggregate_gpu_seconds_approx": float(sum(x["gpu_seconds_approx"] for x in run_meta)),
            "aggregate_gpu_hours_approx": float(sum(x["gpu_seconds_approx"] for x in run_meta) / 3600),
            "parallel_wall_seconds_max": float(max(x["wall_seconds"] for x in run_meta)),
            "gpus": [x["gpu_name"] for x in run_meta],
        },
        "thresholds": {
            "strong_rms": 0.01,
            "strong_ratio": 0.01,
            "weak_ratio": 1e-4,
            "absolute_floor": 1e-4,
            "repeat_multiple": 10.0,
        },
    }
    (ARTIFACT_DIR / "gate0a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(rms_valid, rms_pad, c=merged["task_indices"], cmap="tab10", alpha=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("valid-noise RMS (standardized)")
    axes[0].set_ylabel("padded-noise RMS (standardized)")
    axes[0].set_title("Frozen states")
    axes[1].plot(range(11), step_pad, marker="o", label="pad varied")
    axes[1].plot(range(11), step_valid, marker="o", label="valid varied")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Euler state (0=source, 10=output)")
    axes[1].set_ylabel("RMS across native noise draws")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "gate0a_effects.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
