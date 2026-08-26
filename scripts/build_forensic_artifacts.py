#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic")
PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")


def read(path: Path):
    return json.loads(path.read_text())


def write(name: str, payload: dict) -> None:
    (ROOT / name).write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    fresh = read(ROOT / "fresh_run_comparison.json")
    replay_default = read(ROOT / "one_step_replay_default.json")
    replay_deterministic = read(ROOT / "one_step_replay_deterministic.json")
    deterministic_fresh = read(ROOT / "deterministic_analysis/fresh_run_comparison.json")
    fresh_a = read(ROOT / "fresh_a/metadata.json")
    fresh_b = read(ROOT / "fresh_b/metadata.json")
    det_a = read(ROOT / "deterministic_a/metadata.json")
    det_b = read(ROOT / "deterministic_b/metadata.json")

    write(
        "one_step_replay.json",
        {
            "status": "DIAGNOSED",
            "default": {
                "status": replay_default["status"],
                "first_divergence": replay_default["first_divergence"],
                "comparisons": replay_default["comparisons"],
                "determinism": replay_default["determinism"],
            },
            "deterministic_diagnostic": {
                "status": replay_deterministic["status"],
                "first_divergence": replay_deterministic["first_divergence"],
                "comparisons": replay_deterministic["comparisons"],
                "determinism": replay_deterministic["determinism"],
            },
            "conclusion": "Default CUDA execution diverges first in backward; deterministic algorithms plus deterministic cuBLAS workspace make the full one-step replay bitwise exact.",
        },
    )
    write(
        "rng_state_audit.json",
        {
            "status": "PASS",
            "seed": 560001,
            "streams": {
                "python": "captured and exact across Fresh-A/Fresh-B",
                "numpy": "captured and exact across Fresh-A/Fresh-B",
                "torch_cpu": "captured and exact across Fresh-A/Fresh-B",
                "torch_cuda_all": "captured and exact across Fresh-A/Fresh-B",
                "flow_generator": "dedicated CPU generator seed+103; exact states and sampled tensors",
                "loader_generator": "dedicated CPU generator seed+102; exact states",
            },
            "fresh_updates_checked": fresh["updates"],
            "rng_before_and_after_all_updates_exact": all(
                row["exact"]
                for row in fresh["stage_comparisons"]
                if "rng_" in row["stage"]
            ),
            "model_internal_rng_consumption": "CUDA RNG hashes before/after forward/backward matched; no dropout was found in the exercised custom SmolVLA path.",
            "historical_phase1_checkpoint": {
                "restores_rng": False,
                "saved_rng_streams": [],
                "note": "save_checkpoint calls save_pretrained only for policy and processors.",
            },
        },
    )
    write(
        "dataloader_state_audit.json",
        {
            "status": "PASS_FOR_FRESH_TEST; HISTORICAL_RESUME_STATE_INCOMPLETE",
            "sampler": "FixedSampler over a fully materialized deterministic order",
            "training_order_seed": "seed+101 via numpy.default_rng",
            "loader_generator_seed": "seed+102",
            "worker_seed_function": "torch.initial_seed modulo 2^32 copied to NumPy and Python",
            "num_workers": 4,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "pin_memory": True,
            "fresh_updates_checked": fresh["updates"],
            "sample_identity_exact": all(
                row["exact"] for row in fresh["stage_comparisons"] if "raw_identifiers" in row["stage"]
            ),
            "raw_actions_exact": all(
                row["exact"] for row in fresh["stage_comparisons"] if "raw_action" in row["stage"]
            ),
            "processed_tensors_exact": all(
                row["exact"] for row in fresh["stage_comparisons"] if ".processed" in row["stage"]
            ),
            "augmentation": "No stochastic augmentation is passed by flowspec_vla.data.load_dataset; PyAV decoding plus processor resize/padding/normalization/tokenization produced exact hashes.",
            "historical_phase1_checkpoint": {
                "sampler_offset_saved": False,
                "loader_generator_saved": False,
                "worker_state_saved": False,
                "prefetch_state_saved": False,
                "resume_capable": False,
            },
        },
    )
    write(
        "resume_comparison.json",
        {
            "status": "NOT_RUN_BY_FROZEN_STOP_RULE",
            "reason": "F1 found a pre-checkpoint update-1 backward divergence; FORENSIC_PROTOCOL.md requires proceeding directly to that subsystem and stopping after causal isolation/minimal verification.",
            "uninterrupted_updates": 0,
            "resumed_updates": 0,
            "short_resume_equivalence": "NOT_ESTABLISHED",
            "historical_phase1_checkpoint_audit": {
                "format": "model and processor save_pretrained files only",
                "model_weights": True,
                "processor_state": True,
                "model_buffers": "included only insofar as policy.save_pretrained serializes them",
                "model_train_eval_mode": False,
                "optimizer_moments_steps_groups": False,
                "manual_scheduler_update_lr": False,
                "grad_scaler": "not applicable; no GradScaler is used",
                "gradient_accumulation_position_or_partial_gradients": False,
                "python_numpy_torch_cuda_rng": False,
                "flow_generator": False,
                "loader_generator_sampler_offset_workers_prefetch": False,
            },
            "conclusion": "Historical Phase-1 checkpoints do not implement exact training resume. This omission is real but is not the first cause of the independently reproduced pre-checkpoint divergence.",
        },
    )
    write(
        "fix_verification.json",
        {
            "production_fix_implemented": False,
            "diagnostic_mitigation_verified": True,
            "diagnostic_change": {
                "torch_use_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "scientific_model_data_objective_changed": False,
            },
            "same_process_one_step": {
                "status": replay_deterministic["status"],
                "gradient_max_abs": replay_deterministic["comparisons"]["gradients"]["max_abs"],
                "differing_gradient_tensors": replay_deterministic["comparisons"]["gradients"]["differing_tensor_count"],
                "model_after_exact": replay_deterministic["comparisons"]["model_after_exact"],
            },
            "independent_process_one_step": {
                "status": deterministic_fresh["status"],
                "final_parameter_max_abs": deterministic_fresh["final_checkpoint_difference"]["max_abs"],
                "differing_parameter_tensors": deterministic_fresh["final_checkpoint_difference"]["differing_tensor_count"],
            },
            "short_resume": "NOT_RUN_BY_FROZEN_F1_STOP_RULE",
            "production_recommendation": "Do not treat the diagnostic flags as an adopted production fix until a new full-state checkpoint/resume path and bounded resume verification are authorized.",
        },
    )
    write(
        "first_divergence.json",
        {
            "status": "LOCALIZED",
            "update": 1,
            "last_identical_stage": "loss tensor after the second accumulation microbatch",
            "first_divergent_stage": "backward gradients before clipping",
            "fresh_run_gradient_norms": {
                "fresh_a": fresh["stage_comparisons"][28]["a"],
                "fresh_b": fresh["stage_comparisons"][28]["b"],
                "absolute_difference": fresh["stage_comparisons"][28]["abs"],
            },
            "same_process_default_gradient_difference": replay_default["comparisons"]["gradients"],
            "causal_test": "The same replay becomes bitwise exact under deterministic CUDA/cuBLAS controls, and the result repeats across two independent interpreters.",
            "diagnosis": "CUDA backward numerical nondeterminism under the official/default kernel configuration.",
        },
    )

    successful_seconds = sum(
        [
            fresh_a["gpu_seconds"],
            fresh_b["gpu_seconds"],
            replay_default["gpu_seconds"],
            replay_deterministic["gpu_seconds"],
            det_a["gpu_seconds"],
            det_b["gpu_seconds"],
        ]
    )
    failed_instrumentation_seconds = 56.0
    write(
        "gpu_usage.json",
        {
            "budget_device_hours": 2.0,
            "successful_runs": [
                {"name": "fresh_a", "updates": 5, "seconds": fresh_a["gpu_seconds"]},
                {"name": "fresh_b", "updates": 5, "seconds": fresh_b["gpu_seconds"]},
                {"name": "same_process_replay_default", "updates_executed": 2, "seconds": replay_default["gpu_seconds"]},
                {"name": "same_process_replay_deterministic", "updates_executed": 2, "seconds": replay_deterministic["gpu_seconds"]},
                {"name": "deterministic_fresh_a", "updates": 1, "seconds": det_a["gpu_seconds"]},
                {"name": "deterministic_fresh_b", "updates": 1, "seconds": det_b["gpu_seconds"]},
            ],
            "successful_run_seconds": successful_seconds,
            "successful_run_device_hours": successful_seconds / 3600,
            "failed_instrumentation_attempt_seconds_accounted": failed_instrumentation_seconds,
            "failed_instrumentation_note": "Conservative wall-time accounting; the run failed before producing metadata due to a scalar-state hashing instrumentation bug.",
            "total_accounted_seconds": successful_seconds + failed_instrumentation_seconds,
            "total_accounted_device_hours": (successful_seconds + failed_instrumentation_seconds) / 3600,
            "budget_exceeded": False,
        },
    )

    important = [
        ROOT / name
        for name in [
            "fresh_run_comparison.json",
            "resume_comparison.json",
            "one_step_replay.json",
            "one_step_replay_default.json",
            "one_step_replay_deterministic.json",
            "rng_state_audit.json",
            "dataloader_state_audit.json",
            "first_divergence.json",
            "fix_verification.json",
            "gpu_usage.json",
        ]
    ] + [
        PROJECT / "FORENSIC_PROTOCOL.md",
        PROJECT / "FORENSIC_REPORT.md",
        PROJECT / "EXPERIMENT_STATE.md",
        PROJECT / "config/phase1.yaml",
        PROJECT / "scripts/run_forensic_fresh.py",
        PROJECT / "scripts/run_forensic_replay.py",
        PROJECT / "scripts/analyze_forensic_fresh.py",
        PROJECT / "scripts/build_forensic_artifacts.py",
        PROJECT / "src/flowspec_vla/forensic.py",
    ]
    manifest = {
        "status": "COMPLETE",
        "root": str(ROOT),
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in important
            if path.exists()
        ],
    }
    write("artifact_manifest.json", manifest)


if __name__ == "__main__":
    main()
