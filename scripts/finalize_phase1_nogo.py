#!/usr/bin/env python
"""Materialize the preregistered P6 stop artifacts after reproducibility failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flowspec_vla.phase1 import METHODS, load_phase1_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = load_phase1_config()
    root = Path(config["paths"]["artifacts"])
    archived = root / "reproducibility_audit" / "invalid_cross_device_analysis"
    threshold = 1e-6
    first_m1 = (
        Path(config["paths"]["trained_models"]).parent
        / "phase1_reproducibility_audit/same_device_m1/m1_seed520001/update_01000/model.safetensors"
    )
    first_m2 = Path(config["paths"]["trained_models"]) / "m2_seed520001/update_01000/model.safetensors"

    failed_final = json.loads((archived / "m1_m2_training_invariance.json").read_text())
    audit = {
        "status": "FAIL_UNRESOLVED",
        "protocol_threshold_max_abs": threshold,
        "trigger": "Matched M1/M2 final checkpoints differed above the frozen tolerance for all seeds.",
        "cross_device_final_check": failed_final,
        "same_device_recovery": {
            "policy": "Rerun all M1 seeds on the physical GPU used by matching M0/M2; continue only if the first update-1000 hard check passes.",
            "seed_520001": {
                "device": "0",
                "checkpoint_update": 1000,
                "m1_sha256": sha256(first_m1),
                "m2_sha256": sha256(first_m2),
                "max_abs_parameter_difference": 0.0203399658203125,
                "differing_tensors": 419,
                "tensor_count": 500,
                "passed": False,
            },
            "seed_520002": {
                "device": "1",
                "stopped_after_completed_update": 900,
                "reason": "Symmetric audit run interrupted once seed 520001 established failure.",
            },
            "seed_520003": {"status": "NOT_RUN_AFTER_HARD_FAILURE"},
        },
        "implementation_code_path": {
            "m1_m2_training_objective_same_branch": True,
            "shared_input_loss_test_max_abs": 0.0,
            "interpretation": "The method implementation is isolated, but the CUDA training substrate did not reproduce identical objectives to the preregistered checkpoint tolerance across independent executions, including same-device rerun.",
        },
        "decision": "P6. NO-GO — TRAINING / SUBSTRATE INVALID",
        "downstream_comparison_valid": False,
    }
    (root / "reproducibility_audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")

    training = json.loads((archived / "training_summary.json").read_text())
    training.update(
        {
            "status": "INVALID_REPRODUCIBILITY",
            "descriptive_only": True,
            "scientific_run_count": 9,
            "completed_reproducibility_audit_runs": 0,
            "interrupted_reproducibility_audit_runs": 2,
            "stop_decision": audit["decision"],
            "reproducibility_audit": str(root / "reproducibility_audit_summary.json"),
        }
    )
    (root / "training_summary.json").write_text(json.dumps(training, indent=2) + "\n")

    validation = json.loads((archived / "validation_summary.json").read_text())
    validation.update(
        {
            "status": "INVALID_REPRODUCIBILITY",
            "descriptive_only": True,
            "stop_decision": audit["decision"],
            "warning": "Values are retained for transparency but cannot support causal M0/M1/M2 claims.",
        }
    )
    (root / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")

    gate0 = json.loads((root.parent / "gate0" / "gate0a_summary.json").read_text())
    leakage = {
        "status": "NOT_RUN_PREREGISTERED_STOP",
        "reason": audit["decision"],
        "trained_checkpoint_audits": 0,
        "before_training_gate0_reference": {
            "padded_rms": gate0["primary"]["median_rms_pad_standardized"],
            "valid_rms": gate0["primary"]["median_rms_valid_standardized"],
            "r_leak": gate0["primary"]["r_leak_ratio_of_median_variances"],
        },
        "after_training": None,
    }
    (root / "leakage_audit_summary.json").write_text(json.dumps(leakage, indent=2) + "\n")
    rollout = {
        "status": "NOT_RUN_PREREGISTERED_STOP",
        "reason": audit["decision"],
        "checkpoint_runs": 0,
        "episode_count": 0,
        "methods": {method: None for method in METHODS},
    }
    (root / "rollout_summary.json").write_text(json.dumps(rollout, indent=2) + "\n")
    print(json.dumps({"decision": audit["decision"], "artifacts_written": 5}, indent=2))


if __name__ == "__main__":
    main()
