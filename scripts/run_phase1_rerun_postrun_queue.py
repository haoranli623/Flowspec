#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
CONFIG = PROJECT / "config/phase1_rerun.yaml"
ARTIFACTS = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun")
METHODS = ("m0", "m1", "m2")
SEEDS = (520001, 520002, 520003)


def leakage_tasks() -> list[tuple[str, int | None]]:
    return [(method, None) for method in METHODS] + [
        (method, seed) for seed in SEEDS for method in METHODS
    ]


def rollout_tasks() -> list[tuple[str, int]]:
    return [(method, seed) for seed in SEEDS for method in METHODS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["leakage", "rollout"])
    parser.add_argument("--shard", required=True, type=int, choices=[0, 1])
    args = parser.parse_args()

    if args.kind == "leakage":
        tasks = leakage_tasks()
    else:
        tasks = rollout_tasks()

    selected = [task for index, task in enumerate(tasks) if index % 2 == args.shard]
    for method, seed in selected:
        if args.kind == "leakage":
            suffix = f"initial_{method}" if seed is None else f"{method}_seed{seed}"
            output = ARTIFACTS / "leakage" / f"{suffix}.json"
            command = [
                sys.executable,
                str(PROJECT / "scripts/run_phase1_leakage.py"),
                "--config",
                str(CONFIG),
                "--method",
                method,
            ]
            if seed is None:
                command.append("--initial-base")
            else:
                command.extend(["--seed", str(seed)])
        else:
            output = ARTIFACTS / "rollouts" / f"{method}_seed{seed}.json"
            command = [
                sys.executable,
                str(PROJECT / "scripts/run_phase1_rollout.py"),
                "--config",
                str(CONFIG),
                "--method",
                method,
                "--seed",
                str(seed),
            ]

        if output.exists():
            existing = json.loads(output.read_text())
            if args.kind == "leakage" or existing.get("status") == "COMPLETE":
                print(json.dumps({"status": "SKIP_COMPLETE", "output": str(output)}), flush=True)
                continue
            raise RuntimeError(f"Existing output is not complete: {output}")
        print(json.dumps({"status": "START", "kind": args.kind, "task": [method, seed]}), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)
        print(json.dumps({"status": "COMPLETE", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
