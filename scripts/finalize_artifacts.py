#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch


PROJECT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
ARTIFACTS = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
CHECKPOINT = Path("/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_libero-31d453f7")
DATASET = Path("/mnt/NAS/data/hl5757/datasets/flowspec-vla/lerobot-libero-a1aaacb7")


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
    return {
        "scope": scope,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    packages = sorted(
        ({"name": distribution.metadata["Name"], "version": distribution.version}
         for distribution in importlib.metadata.distributions()),
        key=lambda item: (item["name"] or "").lower(),
    )
    environment = {
        "project_git_commit": git("rev-parse", "HEAD"),
        "protocol_git_commit": "337c95305698e527ef09db6f53040fb377a9d6a5",
        "source_git_commit": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT / 'upstream/lerobot'}",
                "-C",
                str(PROJECT / "upstream/lerobot"),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
    }
    environment_path = ARTIFACTS / "environment.json"
    environment_path.write_text(json.dumps(environment, indent=2) + "\n")

    entries = []
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append(record("artifact", path))

    tracked = [PROJECT / relative for relative in git("ls-files").splitlines()]
    for path in tracked:
        if path.is_file():
            entries.append(record("project", path))

    for path in sorted(CHECKPOINT.iterdir()):
        if path.is_file():
            entries.append(record("checkpoint_input", path))

    for relative in ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet"]:
        path = DATASET / relative
        entries.append(record("dataset_metadata_input", path))

    manifest = {
        "project": "FlowSpec-VLA",
        "generated_at": "2026-08-25",
        "project_git_commit": environment["project_git_commit"],
        "protocol_git_commit": environment["protocol_git_commit"],
        "source_git_commit": environment["source_git_commit"],
        "checkpoint_revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
        "dataset_revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest_path = ARTIFACTS / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
