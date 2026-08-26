#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from flowspec_vla.forensic import compare_tensors


ROOT = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic")


def all_tensor_hashes(value) -> list[str]:
    hashes = []
    if isinstance(value, dict):
        if "sha256" in value and "shape" in value:
            hashes.append(value["sha256"])
        else:
            for key in sorted(value):
                hashes.extend(all_tensor_hashes(value[key]))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(all_tensor_hashes(item))
    return hashes


def hash_equal(left, right) -> bool:
    return all_tensor_hashes(left) == all_tensor_hashes(right)


def model_difference(left: Path, right: Path) -> dict:
    max_abs = 0.0
    sum_abs = 0.0
    sum_square = 0.0
    elements = 0
    differing = []
    tensor_count = 0
    with safe_open(left, framework="pt", device="cpu") as lhs, safe_open(
        right, framework="pt", device="cpu"
    ) as rhs:
        if lhs.keys() != rhs.keys():
            raise RuntimeError("Fresh checkpoint keys differ")
        tensor_count = len(lhs.keys())
        for key in lhs.keys():
            a, b = lhs.get_tensor(key), rhs.get_tensor(key)
            result = compare_tensors(a, b)
            count = a.numel()
            max_abs = max(max_abs, result["max_abs"])
            sum_abs += result["mean_abs"] * count
            sum_square += result["l2"] ** 2
            elements += count
            if not result["exact"]:
                differing.append({"name": key, **result})
    differing.sort(key=lambda row: row["max_abs"], reverse=True)
    return {
        "exact": not differing,
        "tensor_count": tensor_count,
        "differing_tensor_count": len(differing),
        "element_count": elements,
        "max_abs": max_abs,
        "mean_abs": sum_abs / elements,
        "l2": sum_square**0.5,
        "largest_differences": differing[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, default=ROOT / "fresh_a")
    parser.add_argument("--right", type=Path, default=ROOT / "fresh_b")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = json.loads((args.left / "capture.json").read_text())
    b = json.loads((args.right / "capture.json").read_text())
    if len(a["updates"]) != len(b["updates"]):
        raise RuntimeError("Fresh run update counts differ")
    stages = []
    first = None
    for left, right in zip(a["updates"], b["updates"], strict=True):
        update = left["update"]
        checks = [
            ("lr", left["lr"] == right["lr"], {"a": left["lr"], "b": right["lr"]}),
            ("model_before", left["model_before"]["sha256"] == right["model_before"]["sha256"], {}),
            ("buffers_before", left["buffers_before"]["sha256"] == right["buffers_before"]["sha256"], {}),
            ("rng_before_update", left["rng_before_update"] == right["rng_before_update"], {}),
        ]
        for micro_a, micro_b in zip(left["microbatches"], right["microbatches"], strict=True):
            index = micro_a["accumulation_index"]
            checks.extend(
                [
                    (f"micro{index}.raw_identifiers", hash_equal(micro_a["raw_identifiers"], micro_b["raw_identifiers"]), {}),
                    (f"micro{index}.raw_action", micro_a["raw_action"]["sha256"] == micro_b["raw_action"]["sha256"], {}),
                    (f"micro{index}.processed", hash_equal(micro_a["processed"], micro_b["processed"]), {}),
                    (f"micro{index}.flow_noise", micro_a["flow_noise"]["sha256"] == micro_b["flow_noise"]["sha256"], {}),
                    (f"micro{index}.flow_timestep", micro_a["flow_timestep"]["sha256"] == micro_b["flow_timestep"]["sha256"], {}),
                    (f"micro{index}.rng_before_forward", micro_a["rng_before_forward"] == micro_b["rng_before_forward"], {}),
                    (f"micro{index}.action_projection", micro_a["action_projection"]["sha256"] == micro_b["action_projection"]["sha256"], {}),
                    (f"micro{index}.loss_tensor", micro_a["loss_tensor"]["sha256"] == micro_b["loss_tensor"]["sha256"], {}),
                    (f"micro{index}.loss", micro_a["loss"] == micro_b["loss"], {"a": micro_a["loss"], "b": micro_b["loss"], "abs": abs(micro_a["loss"] - micro_b["loss"])}),
                    (f"micro{index}.rng_after_forward", micro_a["rng_after_forward"] == micro_b["rng_after_forward"], {}),
                    (f"micro{index}.rng_after_backward", micro_a["rng_after_backward"] == micro_b["rng_after_backward"], {}),
                ]
            )
        checks.extend(
            [
                ("gradients", left["gradients"]["sha256"] == right["gradients"]["sha256"], {}),
                ("selected_gradients", hash_equal(left["selected_gradients"], right["selected_gradients"]), {}),
                ("gradient_norm_pre_clip", left["gradient_norm_pre_clip"] == right["gradient_norm_pre_clip"], {"a": left["gradient_norm_pre_clip"], "b": right["gradient_norm_pre_clip"], "abs": abs(left["gradient_norm_pre_clip"] - right["gradient_norm_pre_clip"])}),
                ("gradients_after_clip", left["gradients_after_clip"]["sha256"] == right["gradients_after_clip"]["sha256"], {}),
                ("model_after", left["model_after"]["sha256"] == right["model_after"]["sha256"], {}),
                ("optimizer_after", hash_equal(left["optimizer_after"], right["optimizer_after"]), {}),
                ("rng_after_update", left["rng_after_update"] == right["rng_after_update"], {}),
            ]
        )
        for stage, equal, detail in checks:
            record = {"update": update, "stage": stage, "exact": equal, **detail}
            stages.append(record)
            if not equal and first is None:
                first = record
    final = model_difference(
        args.left / "final_model/model.safetensors",
        args.right / "final_model/model.safetensors",
    )
    result = {
        "status": "IDENTICAL" if first is None and final["exact"] else "DIVERGED",
        "runs": [args.left.name, args.right.name],
        "updates": len(a["updates"]),
        "first_divergence": first,
        "stage_comparisons": stages,
        "final_checkpoint_difference": final,
    }
    (args.output_dir / "fresh_run_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "first_divergence.json").write_text(
        json.dumps({"status": result["status"], "first_divergence": first}, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
