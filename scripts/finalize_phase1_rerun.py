#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
ROOT = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun")
PROTOCOL_FREEZE = "82dfe670e6073bc30397d699ee612c0386932a00"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(name: str, value: dict) -> None:
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def run_gpu_hours(pattern: str) -> float:
    return sum(read(path)["timing"]["gpu_hours"] for path in ROOT.glob(pattern))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    smoke = read(ROOT / "smoke_gate_summary.json")
    validation = read(ROOT / "validation_summary.json")
    rollout = read(ROOT / "rollout_summary.json")
    invariance = read(ROOT / "m1_m2_training_invariance.json")
    leakage = read(ROOT / "leakage_reaudit_summary.json")
    # Preserve the original required artifact name while retaining the more explicit rerun name.
    write("leakage_audit_summary.json", leakage)

    baseline = smoke["methods"]["m0"]
    write(
        "baseline_reproduction_summary.json",
        {
            "status": "PASS" if baseline["status"] == "PASS" else "FAIL",
            "protocol_freeze_commit": PROTOCOL_FREEZE,
            "source": "deterministic clean-rerun M0 smoke/reproduction gate",
            "seed": smoke["seed"],
            "updates": baseline["updates"],
            "examples_seen": baseline["examples_seen"],
            "initial_physical_nrmse": baseline["initial_validation"][
                "whole_action_normalized_physical_rmse"
            ],
            "final_physical_nrmse": baseline["final_validation"][
                "whole_action_normalized_physical_rmse"
            ],
            "initial_fixed_flow_mse": baseline["initial_validation"]["fixed_flow_mse"],
            "final_fixed_flow_mse": baseline["final_validation"]["fixed_flow_mse"],
            "first_50_loss_mean": baseline["first_50_train_loss_mean"],
            "last_50_loss_mean": baseline["last_50_train_loss_mean"],
            "peak_memory_gib": baseline["peak_memory_gib"],
            "finite": baseline["finite_losses_gradients_outputs"],
            "resume_exact": baseline["resume_comparison"]["exact"],
        },
    )

    write(
        "seed_pairing_summary.json",
        {
            "status": "COMPLETE",
            "protocol_freeze_commit": PROTOCOL_FREEZE,
            "independent_replicate": "training seed",
            "seeds": [520001, 520002, 520003],
            "validation": validation["paired_comparisons"],
            "rollout": rollout["paired_comparisons"],
        },
    )

    spotchecks = {}
    for method in ("m0", "m1", "m2"):
        reference = read(ROOT / "runs" / f"{method}_seed520001" / "trace.json")
        resumed = read(ROOT / "spotchecks" / f"{method}_seed520001" / "trace.json")
        run = read(ROOT / "spotchecks" / f"{method}_seed520001" / "run.json")
        spotchecks[method] = {
            "updates": [1001, 1002, 1003, 1004, 1005],
            "trace_exact": reference == resumed,
            "restore_exact": run["restore_audit"]["all_exact"],
        }
    recovery = {}
    for method in ("m0", "m1"):
        reference = read(ROOT / "runs" / f"{method}_seed520003" / "trace.json")
        directory = ROOT / "recoveries" / f"{method}_seed520003" / "from_update_01000"
        resumed = read(directory / "trace.json")
        audit = read(directory / "restore_audit.json")
        recovery[method] = {
            "checkpoint_update": 1000,
            "updates": [1001, 1002, 1003, 1004, 1005],
            "trace_exact": reference == resumed,
            "restore_exact": audit["all_exact"],
        }
    reproducibility_pass = bool(
        smoke["status"] == "PASS"
        and invariance["status"] == "PASS"
        and all(row["trace_exact"] and row["restore_exact"] for row in spotchecks.values())
        and all(row["trace_exact"] and row["restore_exact"] for row in recovery.values())
    )
    write(
        "reproducibility_spotcheck.json",
        {
            "status": "PASS" if reproducibility_pass else "FAIL",
            "protocol_freeze_commit": PROTOCOL_FREEZE,
            "smoke_resume_gate": smoke["status"],
            "primary_update1000_spotchecks": spotchecks,
            "interruption_recovery_spotchecks": recovery,
            "m1_m2_training_invariance": invariance["status"],
        },
    )

    smoke_hours = run_gpu_hours("smoke/*/run.json")
    smoke_resume_hours = run_gpu_hours("smoke_resume/*/run.json")
    spotcheck_hours = run_gpu_hours("spotchecks/*/run.json")
    primary_reported_hours = run_gpu_hours("runs/*/run.json")
    interrupted_durable_hours = {}
    for method in ("m0", "m1"):
        directory = ROOT / "runs" / f"{method}_seed520003"
        start = (directory / "validation_update00000.npz").stat().st_mtime
        last_durable = (directory / "trace.json").stat().st_mtime
        interrupted_durable_hours[method] = (last_durable - start) / 3600.0
    leakage_hours = sum(
        read(path)["gpu_seconds"] / 3600.0
        for path in (ROOT / "leakage").glob("*.json")
        if "gpu_seconds" in read(path)
    )
    components = {
        "smoke_reference": smoke_hours,
        "smoke_resume_checks": smoke_resume_hours,
        "primary_reported": primary_reported_hours,
        "interrupted_primary_durable_lower_bound": sum(interrupted_durable_hours.values()),
        "primary_spotchecks": spotcheck_hours,
        "leakage_reaudit": leakage_hours,
        "rollout_evaluation": rollout["aggregate_gpu_hours"],
    }
    lower_bound = sum(components.values())
    write(
        "gpu_usage.json",
        {
            "status": "ACCOUNTED_LOWER_BOUND",
            "protocol_freeze_commit": PROTOCOL_FREEZE,
            "unit": "device-hours",
            "components": components,
            "interrupted_primary_durable_hours_by_method": interrupted_durable_hours,
            "total_accounted_device_hours_lower_bound": lower_bound,
            "ceiling_device_hours": 72.0,
            "below_ceiling_by_at_least_device_hours": 72.0 - lower_bound,
            "limitation": (
                "M0/M1 seed-520003 performed a short unmetered tail after the last durable "
                "trace and before interactive-session termination; method-integrity setup time "
                "was also not device-timed. Therefore an exact all-inclusive total is unavailable."
            ),
        },
    )

    # The clean scientific result is negative for practical benefit but positive for M2 mechanism
    # suppression. The preregistered verdict is therefore P2, and conditional later phases stop.
    decision = {
        "status": "COMPLETE",
        "protocol_freeze_commit": PROTOCOL_FREEZE,
        "verdict": "P2. MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK",
        "m1_gt_m0_reproduced": False,
        "m2_gt_m1_reproduced_materially": False,
        "m2_mechanism_strong_suppression_seed_count": leakage["method_summary"]["m2"][
            "strong_suppression_seed_count"
        ],
        "phase2_authorized_by_conditional_rule": False,
        "stop_after_phase1": True,
    }
    write("phase1_rerun_decision.json", decision)

    manifest_paths = [
        PROJECT / "PHASE1_RERUN_PROTOCOL.md",
        PROJECT / "PHASE1_RERUN_REPORT.md",
        PROJECT / "EXPERIMENT_STATE.md",
        PROJECT / "README.md",
        PROJECT / "config/phase1_rerun.yaml",
    ]
    manifest_paths.extend(sorted(ROOT.glob("*.json")))
    manifest_paths.extend(sorted((ROOT / "rollouts").glob("m*_seed*.json")))
    entries = []
    for path in manifest_paths:
        if path.exists() and path.name != "artifact_manifest.json":
            entries.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write(
        "artifact_manifest.json",
        {
            "status": "COMPLETE",
            "protocol_freeze_commit": PROTOCOL_FREEZE,
            "entries": entries,
        },
    )
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "accounted_device_hours_lower_bound": lower_bound,
                "manifest_entries": len(entries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
