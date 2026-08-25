#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

from flowspec_vla.phase1 import METHODS, load_phase1_config


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
PROTOCOL_COMMIT = "beb292024fcb37d321338474249d03598cfa5e90"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={PROJECT}", *args], cwd=PROJECT, text=True
    ).strip()


def record(scope: str, path: Path) -> dict:
    return {"scope": scope, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    config = load_phase1_config()
    root = Path(config["paths"]["artifacts"])
    seeds = config["training"]["seeds"]
    baseline = json.loads((root / "baseline_reproduction_summary.json").read_text())
    method_hours = {}
    for method in METHODS:
        method_hours[method] = float(
            sum(
                json.loads((root / "runs" / f"{method}_seed{seed}" / "run.json").read_text())[
                    "timing"
                ]["gpu_hours"]
                for seed in seeds
            )
        )
    leakage_hours = float(
        sum(json.loads(path.read_text())["gpu_seconds"] for path in (root / "leakage").glob("*.json"))
        / 3600
    ) if (root / "leakage").exists() else 0.0
    rollouts = json.loads((root / "rollout_summary.json").read_text())
    diagnostic_path = root / "diagnostics_gpu_usage.json"
    diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {
        "gpu_hours": None,
        "status": "NOT_INSTRUMENTED",
    }
    rollout_hours = float(rollouts.get("aggregate_gpu_hours", 0.0))
    audit_estimate = 0.833
    known = baseline["gpu_hours"] + sum(method_hours.values()) + leakage_hours + rollout_hours
    exact_logged = known + (diagnostics["gpu_hours"] or 0.0)
    total = exact_logged + audit_estimate
    gpu_usage = {
        "status": "COMPLETE_WITH_ESTIMATED_INTERRUPTED_AUDIT",
        "units": "device-hours",
        "baseline_reproduction": baseline["gpu_hours"],
        "m0": method_hours["m0"],
        "m1": method_hours["m1"],
        "m2": method_hours["m2"],
        "leakage_audits": leakage_hours,
        "rollout_evaluation": rollout_hours,
        "implementation_diagnostics": diagnostics,
        "reproducibility_audit_estimated": audit_estimate,
        "failed_duplicate_start_not_instrumented": True,
        "exact_logged_gpu_hours": exact_logged,
        "total_accounted_gpu_hours": total,
        "note": "The two interrupted reproducibility-audit processes did not write final timing records; 0.833 device-hours is reconstructed from artifact/process timestamps. The failed pre-update duplicate start is uninstrumented and excluded.",
    }
    (root / "gpu_usage.json").write_text(json.dumps(gpu_usage, indent=2) + "\n")

    packages = sorted(
        ({"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions()),
        key=lambda item: (item["name"] or "").lower(),
    )
    environment = {
        "project_git_commit": git("rev-parse", "HEAD"),
        "protocol_git_commit": PROTOCOL_COMMIT,
        "source_git_commit": config["revisions"]["lerobot_source_commit"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "local_environment_patch": {
            "path": "/mnt/NAS/data/hl5757/conda_envs/flowspec-vla/lib/python3.10/site-packages/robosuite/utils/log_utils.py",
            "change": "FileHandler honors ROBOSUITE_LOG_PATH to avoid an unwritable shared /tmp/robosuite.log; no simulator or policy semantics changed.",
        },
    }
    (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "robosuite.log"}:
            entries.append(record("artifact", path))
    for relative in git("ls-files").splitlines():
        path = PROJECT / relative
        if path.is_file():
            entries.append(record("project", path))
    model_root = Path(config["paths"]["trained_models"])
    for method in METHODS:
        for seed in seeds:
            path = model_root / f"{method}_seed{seed}" / "update_05000" / "model.safetensors"
            entries.append(record("final_checkpoint", path))
    manifest = {
        "project": "FlowSpec-VLA",
        "phase": 1,
        "generated_at": "2026-08-25",
        "project_git_commit": environment["project_git_commit"],
        "protocol_git_commit": PROTOCOL_COMMIT,
        "entry_count": len(entries),
        "entries": entries,
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"gpu_usage": gpu_usage, "manifest_entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
