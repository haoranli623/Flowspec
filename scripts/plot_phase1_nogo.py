#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowspec_vla.phase1 import METHODS, load_phase1_config


def invalid_title(figure) -> None:
    figure.suptitle("DESCRIPTIVE ONLY — P6 INVALID REPRODUCIBILITY", color="#B22222", fontsize=10)


def main() -> None:
    config = load_phase1_config()
    root = Path(config["paths"]["artifacts"])
    validation = json.loads((root / "validation_summary.json").read_text())
    audit = json.loads((root / "reproducibility_audit_summary.json").read_text())
    colors = {"m0": "#4C78A8", "m1": "#F58518", "m2": "#54A24B"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for method in METHODS:
        values = np.asarray(validation["methods"][method]["nrmse_curves"])
        updates = validation["methods"][method]["validation_updates"]
        axis.plot(updates, values.mean(0), marker="o", color=colors[method], label=method.upper())
        axis.fill_between(updates, values.mean(0) - values.std(0, ddof=1), values.mean(0) + values.std(0, ddof=1), color=colors[method], alpha=0.17)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Validation physical-action NRMSE")
    axis.legend(frameon=False)
    invalid_title(fig)
    fig.tight_layout()
    fig.savefig(root / "validation_error_vs_updates.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.4))
    for index, method in enumerate(METHODS):
        values = np.asarray(validation["methods"][method]["final_nrmse"]["values"])
        axis.scatter(np.full(3, index) + np.linspace(-0.08, 0.08, 3), values, s=45, color=colors[method])
        axis.hlines(values.mean(), index - 0.22, index + 0.22, color="black", linewidth=2)
    axis.set_xticks(range(3), [method.upper() for method in METHODS])
    axis.set_ylabel("Final physical-action NRMSE")
    invalid_title(fig)
    fig.tight_layout()
    fig.savefig(root / "per_seed_final_metric.png", dpi=220)
    plt.close(fig)

    final = [row["checkpoint"]["max_abs"] for row in audit["cross_device_final_check"]["seeds"]]
    fig, axis = plt.subplots(figsize=(6.0, 4.4))
    axis.bar(["final 520001", "final 520002", "final 520003", "same-device\n1000"], final + [audit["same_device_recovery"]["seed_520001"]["max_abs_parameter_difference"]], color=["#777777"] * 3 + ["#B22222"])
    axis.axhline(audit["protocol_threshold_max_abs"], color="black", linestyle="--", label="Frozen 1e-6 tolerance")
    axis.set_yscale("log")
    axis.set_ylabel("M1/M2 max |parameter difference|")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "reproducibility_failure.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
