#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowspec_vla.phase1 import METHODS, load_phase1_config


def wilson(successes: int, episodes: int, z: float = 1.96) -> list[float]:
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / episodes + z * z / (4 * episodes**2)) / denominator
    return [float(center - half), float(center + half)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_phase1_config(args.config) if args.config else load_phase1_config()
    root = Path(config["paths"]["artifacts"])
    rollout_root = root / "rollouts"
    seeds = config["training"]["seeds"]
    runs = {}
    for method in METHODS:
        runs[method] = {}
        for seed in seeds:
            path = rollout_root / f"{method}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            run = json.loads(path.read_text())
            if run["episode_count"] != config["rollout"]["episodes_per_checkpoint"]:
                raise RuntimeError(f"Rollout episode count mismatch: {method}/{seed}")
            runs[method][str(seed)] = run

    methods = {}
    for method in METHODS:
        rates = [runs[method][str(seed)]["success_rate"] for seed in seeds]
        suites = {}
        for suite in config["rollout"]["suites"]:
            suite_seed_rates = []
            for seed in seeds:
                selected = [row for row in runs[method][str(seed)]["tasks"] if row["suite"] == suite]
                suite_seed_rates.append(sum(row["successes"] for row in selected) / sum(len(row["episodes"]) for row in selected))
            suites[suite] = {
                "seed_success_rates": suite_seed_rates,
                "mean": float(np.mean(suite_seed_rates)),
                "std": float(np.std(suite_seed_rates, ddof=1)),
            }
        successes = sum(runs[method][str(seed)]["successes"] for seed in seeds)
        episodes = sum(runs[method][str(seed)]["episode_count"] for seed in seeds)
        methods[method] = {
            "seed_success_rates": rates,
            "mean": float(np.mean(rates)),
            "std": float(np.std(rates, ddof=1)),
            "pooled_successes_descriptive_only": successes,
            "pooled_episodes_descriptive_only": episodes,
            "pooled_wilson_95ci_descriptive_only": wilson(successes, episodes),
            "suites": suites,
        }
    comparisons = {}
    for candidate, baseline in [("m1", "m0"), ("m2", "m0"), ("m2", "m1")]:
        differences = np.asarray(methods[candidate]["seed_success_rates"]) - np.asarray(methods[baseline]["seed_success_rates"])
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "paired_success_rate_difference": differences.astype(float).tolist(),
            "mean_percentage_point_difference": float(differences.mean() * 100),
            "favors_candidate_seed_count": int(np.sum(differences > 0)),
        }
    protocol_freeze = (
        "82dfe670e6073bc30397d699ee612c0386932a00"
        if config.get("phase") == "phase1_rerun"
        else "beb292024fcb37d321338474249d03598cfa5e90"
    )
    summary = {
        "status": "COMPLETE",
        "protocol_freeze_commit": protocol_freeze,
        "counts": {
            "runs": 9,
            "episodes_per_run": config["rollout"]["episodes_per_checkpoint"],
            "total_episodes": 9 * config["rollout"]["episodes_per_checkpoint"],
            "tasks_per_run": 40,
        },
        "methods": methods,
        "paired_comparisons": comparisons,
        "aggregate_gpu_hours": float(
            sum(runs[method][str(seed)]["gpu_active_wall_seconds"] for method in METHODS for seed in seeds) / 3600
        ),
    }
    (root / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"m0": "#4C78A8", "m1": "#F58518", "m2": "#54A24B"}
    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    x = np.arange(3)
    for index, method in enumerate(METHODS):
        rates = np.asarray(methods[method]["seed_success_rates"]) * 100
        axis.bar(index, rates.mean(), color=colors[method], alpha=0.75, width=0.65)
        axis.errorbar(index, rates.mean(), yerr=rates.std(ddof=1), color="black", capsize=4)
        axis.scatter(np.full(3, index) + np.linspace(-0.08, 0.08, 3), rates, color="black", s=25)
    axis.set_xticks(x, [method.upper() for method in METHODS])
    axis.set_ylabel("LIBERO success (%)")
    axis.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(root / "rollout_success.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"summary": str(root / "rollout_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
