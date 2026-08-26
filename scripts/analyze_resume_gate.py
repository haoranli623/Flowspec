#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from flowspec_vla.forensic import compare_tensors
from flowspec_vla.resume_gate import ROOT


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensor_difference(left: Path, right: Path) -> dict[str, Any]:
    maximum = 0.0
    sum_abs = 0.0
    sum_square = 0.0
    elements = 0
    differing = []
    with safe_open(left, framework="pt", device="cpu") as lhs, safe_open(
        right, framework="pt", device="cpu"
    ) as rhs:
        left_keys = list(lhs.keys())
        right_keys = list(rhs.keys())
        if left_keys != right_keys:
            return {
                "compatible": False,
                "left_only": sorted(set(left_keys) - set(right_keys)),
                "right_only": sorted(set(right_keys) - set(left_keys)),
            }
        for key in left_keys:
            left_tensor = lhs.get_tensor(key)
            right_tensor = rhs.get_tensor(key)
            comparison = compare_tensors(left_tensor, right_tensor)
            count = left_tensor.numel()
            maximum = max(maximum, comparison["max_abs"])
            sum_abs += comparison["mean_abs"] * count
            sum_square += comparison["l2"] ** 2
            elements += count
            if not comparison["exact"]:
                differing.append({"name": key, **comparison})
    differing.sort(key=lambda row: row["max_abs"], reverse=True)
    return {
        "compatible": True,
        "exact": not differing,
        "tensor_count": len(left_keys),
        "differing_tensor_count": len(differing),
        "element_count": elements,
        "max_abs": maximum,
        "mean_abs": sum_abs / elements if elements else 0.0,
        "l2": sum_square**0.5,
        "largest_differences": differing[:20],
    }


STAGES = [
    ("checkpoint_loaded_parameters", "model_before"),
    ("checkpoint_loaded_parameters", "selected_parameters_before"),
    ("optimizer_state", "optimizer_before"),
    ("optimizer_state", "selected_optimizer_before"),
    ("scheduler_amp_state", "scheduler_before"),
    ("rng_state", "rng_before"),
    ("sampler_data_position", "sample_offset_before"),
    ("data_identity", "raw_identifiers"),
    ("data_identity", "raw_actions"),
    ("processed_batch", "processed"),
    ("flow_timestep", "flow_timestep"),
    ("gaussian_flow_noise", "flow_noise"),
    ("forward", "action_projection"),
    ("loss", "loss_tensors"),
    ("loss", "losses"),
    ("gradient", "gradients"),
    ("gradient", "selected_gradients"),
    ("gradient", "gradient_norm_pre_clip"),
    ("optimizer_update", "optimizer_after"),
    ("optimizer_update", "selected_optimizer_after"),
    ("parameters", "model_after"),
    ("parameters", "selected_parameters_after"),
    ("post_update_rng", "rng_after"),
    ("post_update_position", "sample_offset_after"),
]


def analyze_profile(profile: str) -> dict[str, Any] | None:
    base = ROOT / profile
    if not (base / "run_a/metadata.json").exists() or not (base / "run_b/metadata.json").exists():
        return None
    metadata_a = read(base / "run_a/metadata.json")
    metadata_b = read(base / "run_b/metadata.json")
    trace_a = read(base / "run_a/trace.json")["updates"]
    trace_b = read(base / "run_b/trace.json")["updates"]
    if [row["update"] for row in trace_a] != [row["update"] for row in trace_b]:
        raise RuntimeError(f"{profile}: continuation update labels differ")
    comparisons = []
    first = None
    for left, right in zip(trace_a, trace_b, strict=True):
        if left["lr"] != right["lr"] and first is None:
            first = {"update": right["update"], "stage": "scheduler_amp_state", "field": "lr"}
        comparisons.append({"update": right["update"], "stage": "lr", "exact": left["lr"] == right["lr"]})
        for stage, field in STAGES:
            exact = left.get(field) == right.get(field)
            comparisons.append({"update": right["update"], "stage": stage, "field": field, "exact": exact})
            if not exact and first is None:
                first = {"update": right["update"], "stage": stage, "field": field}

    final_a = base / "run_a/final_state"
    final_b = base / "run_b/final_state"
    model = safetensor_difference(
        final_a / "model/model.safetensors", final_b / "model/model.safetensors"
    )
    optimizer = safetensor_difference(
        final_a / "optimizer.safetensors", final_b / "optimizer.safetensors"
    )
    state_a = read(final_a / "state.json")
    state_b = read(final_b / "state.json")
    groups_exact = state_a["optimizer_groups_named"] == state_b["optimizer_groups_named"]
    scheduler_exact = state_a["scheduler"] == state_b["scheduler"]
    amp_exact = state_a["amp"] == state_b["amp"]
    rng_exact = state_a["rng"] == state_b["rng"]
    training_position_exact = state_a["training"] == state_b["training"]
    restore = metadata_b["restore_audit"]
    passed = bool(
        first is None
        and restore["all_exact"]
        and model.get("max_abs", float("inf")) < 1e-6
        and optimizer.get("exact", False)
        and groups_exact
        and scheduler_exact
        and amp_exact
        and rng_exact
        and training_position_exact
    )
    return {
        "profile": profile,
        "status": "PASS" if passed else "FAIL",
        "checkpoint_update": metadata_a["checkpoint_update"],
        "continuation_endpoint": metadata_a["total_updates"],
        "continuation_updates": len(trace_a),
        "first_divergence": first,
        "restore_audit": restore,
        "trace_comparisons": comparisons,
        "logical_identity": {
            "batch_ids": all(row["exact"] for row in comparisons if row.get("field") == "raw_identifiers"),
            "raw_actions": all(row["exact"] for row in comparisons if row.get("field") == "raw_actions"),
            "processed_tensors": all(row["exact"] for row in comparisons if row.get("field") == "processed"),
            "flow_timesteps": all(row["exact"] for row in comparisons if row.get("field") == "flow_timestep"),
            "gaussian_noise": all(row["exact"] for row in comparisons if row.get("field") == "flow_noise"),
            "forward_outputs": all(row["exact"] for row in comparisons if row.get("field") == "action_projection"),
            "losses": all(row["exact"] for row in comparisons if row.get("field") in {"loss_tensors", "losses"}),
            "gradients": all(row["exact"] for row in comparisons if row["stage"] == "gradient"),
        },
        "final_model_difference": model,
        "final_optimizer_difference": optimizer,
        "optimizer_parameter_groups_exact": groups_exact,
        "scheduler_exact": scheduler_exact,
        "amp_exact": amp_exact,
        "rng_exact": rng_exact,
        "training_position_exact": training_position_exact,
        "gpu_seconds": metadata_a["gpu_seconds"] + metadata_b["gpu_seconds"],
        "run_gpu_seconds": {"reference": metadata_a["gpu_seconds"], "resumed": metadata_b["gpu_seconds"]},
        "num_workers": metadata_a["num_workers"],
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    primary = analyze_profile("primary")
    if primary is None:
        raise RuntimeError("Primary resume-gate runs are incomplete")
    optional = analyze_profile("multiworker")
    if primary["status"] != "PASS":
        restore = primary["restore_audit"]
        verdict = (
            "R-C — RESUME STATE BUG IDENTIFIED BUT NOT FULLY FIXED"
            if not restore["all_exact"]
            else "R-D — RESUME GATE FAILED"
        )
    elif optional is not None and optional["status"] == "PASS":
        verdict = "R-A — RESUME GATE PASSED"
    else:
        verdict = "R-B — RESUME GATE PASSED WITH RESTRICTED LOADER"

    shutil.copyfile(ROOT / "primary/checkpoint_state_inventory.json", ROOT / "checkpoint_state_inventory.json")
    shutil.copyfile(ROOT / "primary/run_a/trace.json", ROOT / "uninterrupted_trace.json")
    shutil.copyfile(ROOT / "primary/run_b/trace.json", ROOT / "resumed_trace.json")
    comparison = {
        "verdict": verdict,
        "primary": primary,
        "optional_multiworker": optional,
        "phase1_technically_eligible": verdict in {
            "R-A — RESUME GATE PASSED",
            "R-B — RESUME GATE PASSED WITH RESTRICTED LOADER",
        },
        "phase1_rerun_authorized": False,
    }
    write(ROOT / "resume_comparison.json", comparison)
    write(
        ROOT / "optimizer_state_comparison.json",
        {
            "status": "PASS" if primary["final_optimizer_difference"].get("exact") and primary["optimizer_parameter_groups_exact"] else "FAIL",
            "primary_tensor_difference": primary["final_optimizer_difference"],
            "parameter_groups_exact": primary["optimizer_parameter_groups_exact"],
            "representative_and_periodic_trace_exact": all(
                row["exact"]
                for row in primary["trace_comparisons"]
                if row["stage"] in {"optimizer_state", "optimizer_update"}
            ),
            "optional_multiworker": None if optional is None else {
                "tensor_difference": optional["final_optimizer_difference"],
                "parameter_groups_exact": optional["optimizer_parameter_groups_exact"],
            },
        },
    )
    write(
        ROOT / "rng_state_comparison.json",
        {
            "status": "PASS" if primary["restore_audit"]["rng_exact"] and primary["rng_exact"] else "FAIL",
            "checkpoint_restore_exact": primary["restore_audit"]["rng_exact"],
            "per_update_exact": all(
                row["exact"] for row in primary["trace_comparisons"] if row["stage"] in {"rng_state", "post_update_rng"}
            ),
            "final_exact": primary["rng_exact"],
            "streams": ["python", "numpy", "torch_cpu", "torch_cuda_all", "flow_generator", "loader_generator"],
        },
    )
    write(
        ROOT / "sampler_state_comparison.json",
        {
            "status": "PASS" if primary["training_position_exact"] and primary["logical_identity"]["batch_ids"] else "FAIL",
            "implementation": "materialized FixedSampler order plus explicit absolute example offset",
            "checkpoint_offset": primary["checkpoint_update"] * 32,
            "batch_ids_exact": primary["logical_identity"]["batch_ids"],
            "processed_tensors_exact": primary["logical_identity"]["processed_tensors"],
            "final_training_position_exact": primary["training_position_exact"],
            "primary_num_workers": primary["num_workers"],
            "optional_multiworker": None if optional is None else {
                "status": optional["status"],
                "batch_ids_exact": optional["logical_identity"]["batch_ids"],
                "processed_tensors_exact": optional["logical_identity"]["processed_tensors"],
            },
        },
    )
    successful_seconds = primary["gpu_seconds"] + (0.0 if optional is None else optional["gpu_seconds"])
    failed_comparator_attempt_seconds = 52.5
    total_seconds = successful_seconds + failed_comparator_attempt_seconds
    write(
        ROOT / "gpu_usage.json",
        {
            "budget_device_hours": 0.5,
            "primary": primary["run_gpu_seconds"],
            "primary_seconds": primary["gpu_seconds"],
            "optional_multiworker_seconds": 0.0 if optional is None else optional["gpu_seconds"],
            "successful_seconds": successful_seconds,
            "failed_online_comparator_attempt_seconds_accounted": failed_comparator_attempt_seconds,
            "failed_attempt_note": "Conservative process-wall accounting. Restore was exact; an online tuple-versus-JSON-list representation comparison stopped before update 51 training and was fixed without changing checkpoint or training semantics.",
            "total_seconds": total_seconds,
            "total_device_hours": total_seconds / 3600,
            "budget_exceeded": total_seconds / 3600 > 0.5,
        },
    )

    important = [
        ROOT / name
        for name in [
            "checkpoint_state_inventory.json",
            "uninterrupted_trace.json",
            "resumed_trace.json",
            "resume_comparison.json",
            "optimizer_state_comparison.json",
            "rng_state_comparison.json",
            "sampler_state_comparison.json",
            "gpu_usage.json",
        ]
    ] + [
        PROJECT / "RESUME_GATE_PROTOCOL.md",
        PROJECT / "RESUME_GATE_REPORT.md",
        PROJECT / "EXPERIMENT_STATE.md",
        PROJECT / "config/phase1.yaml",
        PROJECT / "scripts/run_resume_gate.py",
        PROJECT / "scripts/analyze_resume_gate.py",
        PROJECT / "src/flowspec_vla/resume_gate.py",
    ]
    large_outputs = []
    for profile_name in ("primary", "multiworker"):
        profile_root = ROOT / profile_name
        inventory_path = profile_root / "checkpoint_state_inventory.json"
        if not inventory_path.exists():
            continue
        inventory = read(inventory_path)
        large_outputs.append(
            {
                "path": inventory["path"],
                "bytes": inventory["bytes"],
                "sha256": inventory["sha256"],
            }
        )
        for role in ("run_a", "run_b"):
            state = read(profile_root / role / "final_state/state.json")
            for file_record in state["files"].values():
                path = Path(file_record["path"])
                large_outputs.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": file_record["sha256"],
                    }
                )
    manifest = {
        "status": "COMPLETE",
        "verdict": verdict,
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in important
            if path.exists()
        ] + large_outputs,
    }
    write(ROOT / "artifact_manifest.json", manifest)
    print(json.dumps({"verdict": verdict, "primary": primary["status"], "optional": None if optional is None else optional["status"], "gpu_hours": total_seconds / 3600}, indent=2))


if __name__ == "__main__":
    main()
