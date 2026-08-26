#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flowspec_vla.phase1 import load_phase1_config


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
CONFIG = PROJECT / "config/phase1_rerun.yaml"
METHODS = ("m0", "m1", "m2")


def exact_json(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=True) == json.dumps(
        right, sort_keys=True, allow_nan=True
    )


def trace_comparison(reference: dict, resumed: dict) -> dict:
    reference_rows = {row["update"]: row for row in reference}
    resumed_rows = {row["update"]: row for row in resumed}
    fields = [
        "lr",
        "sample_offset_before",
        "model_before",
        "selected_parameters_before",
        "optimizer_before",
        "selected_optimizer_before",
        "scheduler_before",
        "rng_before",
        "raw_identifiers",
        "raw_actions",
        "processed",
        "flow_noise",
        "effective_source",
        "valid_source",
        "padded_source_max_abs",
        "flow_timestep",
        "action_projection",
        "losses",
        "gradients",
        "selected_gradients",
        "gradient_norm_pre_clip",
        "model_after",
        "selected_parameters_after",
        "optimizer_after",
        "selected_optimizer_after",
        "rng_after",
        "sample_offset_after",
    ]
    comparisons = []
    first_divergence = None
    for update in sorted(reference_rows):
        if update not in resumed_rows:
            first_divergence = {"update": update, "field": "missing_update"}
            break
        for field in fields:
            exact = exact_json(reference_rows[update].get(field), resumed_rows[update].get(field))
            comparisons.append({"update": update, "field": field, "exact": exact})
            if not exact and first_divergence is None:
                first_divergence = {"update": update, "field": field}
                break
        if first_divergence:
            break
    exact = first_divergence is None and set(reference_rows) == set(resumed_rows)
    return {
        "exact": exact,
        "max_parameter_difference": 0.0 if exact else None,
        "first_divergence": first_divergence,
        "trace_updates": sorted(reference_rows),
        "comparisons": comparisons,
    }


def main() -> None:
    config = load_phase1_config(CONFIG)
    root = Path(config["paths"]["artifacts"])
    seed = int(config["smoke_gate"]["seed"])
    threshold = float(config["thresholds"]["resume_max_parameter_difference_exclusive"])
    references = {}
    resumes = {}
    traces = {}
    method_results = {}
    for method in METHODS:
        name = f"{method}_seed{seed}"
        reference_root = root / "smoke" / name
        resumed_root = root / "smoke_resume" / name
        references[method] = json.loads((reference_root / "run.json").read_text())
        resumes[method] = json.loads((resumed_root / "run.json").read_text())
        reference_trace = json.loads((reference_root / "trace.json").read_text())["updates"]
        resumed_trace = json.loads((resumed_root / "trace.json").read_text())["updates"]
        comparison = trace_comparison(reference_trace, resumed_trace)
        traces[method] = comparison
        run = references[method]
        losses = np.asarray([row["loss"] for row in run["train_log"]], dtype=np.float64)
        gradients = np.asarray(
            [row["gradient_norm_pre_clip"] for row in run["train_log"]], dtype=np.float64
        )
        first_loss = float(losses[:50].mean())
        last_loss = float(losses[-50:].mean())
        initial = run["validation_log"][0]
        final = run["validation_log"][-1]
        thresholds = config["thresholds"]
        finite = bool(
            np.isfinite(losses).all()
            and np.isfinite(gradients).all()
            and all(row["all_outputs_finite"] for row in run["validation_log"])
        )
        plausible_validation = bool(
            final["whole_action_normalized_physical_rmse"]
            / initial["whole_action_normalized_physical_rmse"]
            <= thresholds["baseline_final_to_initial_nrmse_max"]
            or final["fixed_flow_mse"] / initial["fixed_flow_mse"]
            <= thresholds["baseline_final_to_initial_flow_mse_max"]
        )
        checkpoint = run["checkpoint_inventories"][str(config["smoke_gate"]["checkpoint_update"])]
        source_probe = run["first_update_probe"]["microbatches"]
        padded = [row["padded_source_max_abs"] for row in source_probe]
        method_passed = bool(
            run["status"] == "COMPLETE"
            and run["end_update"] == config["smoke_gate"]["updates"]
            and run["examples_seen_at_end"] == config["smoke_gate"]["examples_seen"]
            and finite
            and last_loss / first_loss <= thresholds["baseline_last_to_first_train_loss_max"]
            and plausible_validation
            and run["timing"]["peak_memory_gib"] <= thresholds["baseline_peak_memory_gib_max"]
            and checkpoint["serialization_rng_exact"]
            and checkpoint["model_export_rng_exact"]
            and resumes[method]["restore_audit"]["all_exact"]
            and comparison["exact"]
            and comparison["max_parameter_difference"] < threshold
            and ((method == "m0" and max(padded) > 0.0) or (method != "m0" and max(padded) == 0.0))
        )
        method_results[method] = {
            "status": "PASS" if method_passed else "FAIL",
            "finite_losses_gradients_outputs": finite,
            "first_50_train_loss_mean": first_loss,
            "last_50_train_loss_mean": last_loss,
            "last_to_first_train_loss_ratio": last_loss / first_loss,
            "initial_validation": initial,
            "final_validation": final,
            "plausible_validation_direction": plausible_validation,
            "updates": run["end_update"],
            "examples_seen": run["examples_seen_at_end"],
            "peak_memory_gib": run["timing"]["peak_memory_gib"],
            "padded_source_max_abs": max(padded),
            "restore_audit_exact": resumes[method]["restore_audit"]["all_exact"],
            "resume_comparison": comparison,
            "checkpoint_serialization_rng_exact": checkpoint["serialization_rng_exact"],
            "model_export_rng_exact": checkpoint["model_export_rng_exact"],
            "gpu_hours_reference": run["timing"]["gpu_hours"],
            "gpu_hours_resumed": resumes[method]["timing"]["gpu_hours"],
        }

    initial_model_exact = len(
        {references[m]["initial_base_model_digest"]["sha256"] for m in METHODS}
    ) == 1
    initial_valid_exact = len(
        {json.dumps(references[m]["initial_valid_parameter_record"], sort_keys=True) for m in METHODS}
    ) == 1
    order_exact = len({references[m]["training_order_sha256"] for m in METHODS}) == 1
    batch_sequence_exact = len(
        {
            tuple(row["sample_indices_sha256"] for row in references[m]["train_log"])
            for m in METHODS
        }
    ) == 1
    probes = {m: references[m]["first_update_probe"]["microbatches"] for m in METHODS}
    raw_noise_exact = all(
        exact_json(probes["m0"][i]["flow_noise"], probes[m][i]["flow_noise"])
        for m in ("m1", "m2")
        for i in range(len(probes["m0"]))
    )
    valid_source_exact = all(
        exact_json(probes["m0"][i]["valid_source"], probes[m][i]["valid_source"])
        for m in ("m1", "m2")
        for i in range(len(probes["m0"]))
    )
    timestep_exact = all(
        exact_json(probes["m0"][i]["flow_timestep"], probes[m][i]["flow_timestep"])
        for m in ("m1", "m2")
        for i in range(len(probes["m0"]))
    )
    m1_m2_train_exact = exact_json(references["m1"]["train_log"], references["m2"]["train_log"])
    # Runtime fields differ slightly even under identical numerical training.
    numerical_fields = (
        "update",
        "loss",
        "gradient_norm_pre_clip",
        "valid_velocity_rms",
        "padded_velocity_rms",
        "lr",
        "examples_seen",
        "sample_index_first",
        "sample_index_last",
        "sample_indices_sha256",
    )
    m1_m2_numerical_exact = all(
        all(left[field] == right[field] for field in numerical_fields)
        for left, right in zip(references["m1"]["train_log"], references["m2"]["train_log"], strict=True)
    )
    m1_m2_final_model_exact = (
        references["m1"]["final_model_digest"] == references["m2"]["final_model_digest"]
    )
    overall_passed = bool(
        all(value["status"] == "PASS" for value in method_results.values())
        and initial_model_exact
        and initial_valid_exact
        and order_exact
        and batch_sequence_exact
        and raw_noise_exact
        and valid_source_exact
        and timestep_exact
        and m1_m2_numerical_exact
        and m1_m2_final_model_exact
    )
    summary = {
        "status": "PASS" if overall_passed else "FAIL",
        "verdict": (
            "SMOKE GATE PASSED"
            if overall_passed
            else "PHASE1-RERUN NO-GO — METHOD-SPECIFIC REPRODUCIBILITY FAILURE"
        ),
        "scientific_comparison_permitted": overall_passed,
        "seed": seed,
        "methods": method_results,
        "matched_checks": {
            "initial_model_exact": initial_model_exact,
            "initial_valid_parameters_exact": initial_valid_exact,
            "training_order_exact": order_exact,
            "all_update_batch_sequences_exact": batch_sequence_exact,
            "first_update_raw_noise_exact": raw_noise_exact,
            "first_update_valid_source_exact": valid_source_exact,
            "first_update_timestep_exact": timestep_exact,
            "m1_m2_full_train_log_exact_including_runtime": m1_m2_train_exact,
            "m1_m2_numerical_training_exact": m1_m2_numerical_exact,
            "m1_m2_final_model_exact": m1_m2_final_model_exact,
        },
        "thresholds": {
            "max_parameter_difference_exclusive": threshold,
            **{key: value for key, value in config["thresholds"].items() if key.startswith("baseline_")},
        },
        "note": "The 250-update runs are an engineering gate and were not interpreted as a method comparison.",
    }
    output = root / "smoke_gate_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not overall_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
