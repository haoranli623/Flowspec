#!/usr/bin/env python3
"""Build the local, Git-sized FlowSpec-VLA recovery snapshot.

This script only copies small immutable evidence and inventories external large
assets.  It never writes to the source artifact, model, or dataset trees.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(
    os.environ.get(
        "FLOWSPEC_ARTIFACT_ROOT",
        REPO.parent / "generated_artifacts" / "flowspec-vla",
    )
)
MODELS = Path(
    os.environ.get(
        "FLOWSPEC_MODEL_ROOT",
        REPO.parent / "models" / "flowspec-vla",
    )
)
FROZEN = REPO / "results" / "frozen"


def is_cache_artifact(path: Path) -> bool:
    """Return true for interpreter caches that must never enter recovery metadata."""
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or sha256(source) != sha256(destination):
        raise RuntimeError(f"byte-identity check failed: {source} -> {destination}")


def copy_glob(source_root: Path, destination_root: Path, patterns: list[str]) -> None:
    selected: set[Path] = set()
    for pattern in patterns:
        selected.update(path for path in source_root.glob(pattern) if path.is_file())
    for source in sorted(selected):
        copy_exact(source, destination_root / source.relative_to(source_root))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def archive_small_evidence() -> None:
    gate0 = ARTIFACTS / "gate0"
    copy_glob(
        gate0,
        FROZEN / "gate0",
        [
            "*.json",
            "*.csv",
            "*.png",
            "gate0b_*.npz",
        ],
    )

    clean = ARTIFACTS / "phase1_rerun"
    copy_glob(clean, FROZEN / "phase1_clean", ["*.json", "*.png"])
    copy_glob(clean / "runs", FROZEN / "phase1_clean" / "runs", ["*/*.json"])
    copy_glob(clean / "rollouts", FROZEN / "phase1_clean" / "rollouts", ["*.json"])
    copy_glob(clean / "leakage", FROZEN / "phase1_clean" / "leakage", ["*.json"])
    copy_glob(clean / "recoveries", FROZEN / "phase1_clean" / "recoveries", ["*/*/*.json"])
    copy_glob(clean / "spotchecks", FROZEN / "phase1_clean" / "spotchecks", ["*/*.json"])

    # The clean rerun deliberately reused the originally frozen deterministic split.
    copy_exact(
        ARTIFACTS / "phase1" / "split_manifest.json",
        FROZEN / "phase1_clean" / "split_manifest.json",
    )

    copy_glob(ARTIFACTS / "forensic", FROZEN / "forensic", ["*.json"])
    copy_glob(ARTIFACTS / "resume_gate", FROZEN / "resume_gate", ["*.json"])
    copy_glob(
        ARTIFACTS / "phase1",
        FROZEN / "historical_invalid_phase1",
        ["*.json", "*.png"],
    )


def build_checkpoint_manifests() -> None:
    inventory_root = ARTIFACTS / "phase1_rerun" / "runs"
    model_root = MODELS / "phase1_rerun" / "runs"
    methods = ("m0", "m1", "m2")
    seeds = (520001, 520002, 520003)
    updates = (1000, 2500, 5000)
    final_file_hashes: dict[tuple[str, int], str] = {}

    # Hash all logical final checkpoints and prove the three M2 files are exact
    # byte duplicates of same-seed M1 before reducing the recovery asset set.
    for method in methods:
        for seed in seeds:
            model_file = model_root / f"{method}_seed{seed}" / "update_05000" / "model.safetensors"
            final_file_hashes[(method, seed)] = sha256(model_file)
    for seed in seeds:
        if final_file_hashes[("m2", seed)] != final_file_hashes[("m1", seed)]:
            raise RuntimeError(f"M2/M1 final weight byte identity failed for seed {seed}")

    checkpoints: list[dict[str, object]] = []
    intermediates: list[dict[str, object]] = []
    for method in methods:
        for seed in seeds:
            run_name = f"{method}_seed{seed}"
            run_inventories: list[dict[str, object]] = []
            for update in updates:
                inventory_path = inventory_root / run_name / f"checkpoint_update{update:05d}_inventory.json"
                inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
                record = {
                    "method": method,
                    "seed": seed,
                    "update": update,
                    "inventory_archived_path": str(
                        Path("results/frozen/phase1_clean/runs") / run_name / inventory_path.name
                    ),
                    "complete_state_original_path": inventory["path"],
                    "complete_state_size": inventory["bytes"],
                    "complete_state_sha256": inventory["sha256"],
                    "model_tensor_digest": inventory["model_digest"]["sha256"],
                    "optimizer_tensor_digest": inventory["optimizer_digest"]["sha256"],
                    "serialization_rng_exact": inventory["serialization_rng_exact"],
                }
                run_inventories.append(record)
                if update != 5000:
                    intermediates.append(record)

            final_dir = model_root / run_name / "update_05000"
            final_model = final_dir / "model.safetensors"
            final_inventory = run_inventories[-1]
            checkpoints.append(
                {
                    **final_inventory,
                    "checkpoint_original_path": str(final_dir),
                    "model_safetensors_original_path": str(final_model),
                    "model_safetensors_size": final_model.stat().st_size,
                    "model_safetensors_sha256": final_file_hashes[(method, seed)],
                    "model_file_hash_provenance": "BYTE_HASHED_AT_THIS_PATH",
                    "rollout_usage": "FROZEN_200_EPISODE_EVALUATION",
                    "logical_checkpoint_required_for_inference": True,
                    "unique_weight_asset_required_for_inference": method != "m2",
                    "inference_weight_source": (
                        f"m1_seed{seed}/update_05000/model.safetensors"
                        if method == "m2"
                        else f"{run_name}/update_05000/model.safetensors"
                    ),
                    "required_for_full_training_resume": True,
                    "inference_projection": method == "m2",
                    "minimum_unique_weight_counterpart": f"m1_seed{seed}" if method == "m2" else run_name,
                }
            )

    checkpoint_map = {
        "schema_version": 1,
        "scientific_commit": "2064fb43073c5f2ef5254550ce107028cc29d492",
        "protocol_freeze_commit": "82dfe670e6073bc30397d699ee612c0386932a00",
        "final_rollout_checkpoints": checkpoints,
        "intermediate_checkpoint_inventories": intermediates,
        "m2_weight_identity": {
            "statement": "M2 final model tensors are identical to same-seed M1; behavior differs through inference projection.",
            "same_seed_model_safetensors_byte_identity_verified": True,
            "evidence": "results/frozen/phase1_clean/m1_m2_training_invariance.json",
        },
        "minimum_unique_final_inference_weight_set": [
            f"{method}_seed{seed}/update_05000/model.safetensors"
            for method in ("m0", "m1")
            for seed in seeds
        ],
    }
    write_json(REPO / "checkpoints" / "checkpoint_map.json", checkpoint_map)

    rows: list[dict[str, object]] = []
    for method in ("m0", "m1"):
        for seed in seeds:
            source = model_root / f"{method}_seed{seed}" / "update_05000" / "model.safetensors"
            rows.append(
                {
                    "original_path": source,
                    "filename": source.name,
                    "size_bytes": source.stat().st_size,
                    "sha256": final_file_hashes[(method, seed)],
                    "method": method,
                    "seed": seed,
                    "update": 5000,
                    "expected_recovery_destination": f"checkpoints/assets/{method}_seed{seed}/update_05000/model.safetensors",
                }
            )
    write_tsv(
        REPO / "checkpoints" / "LFS_CANDIDATES.tsv",
        rows,
        [
            "original_path",
            "filename",
            "size_bytes",
            "sha256",
            "method",
            "seed",
            "update",
            "expected_recovery_destination",
        ],
    )


def export_current_package_state() -> None:
    """Export versions only; this is evidence, not a portable lock file."""
    python = Path(os.environ.get("FLOWSPEC_PYTHON", sys.executable))
    result = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
    )
    header = (
        "# Point-in-time PASS 2A package inventory from the historical local venv.\n"
        "# This is not a portable lock and does not establish wheel hashes.\n"
    )
    (REPO / "recovery" / "current_installed_packages.txt").write_text(
        header + result.stdout,
        encoding="utf-8",
    )


def build_raw_binary_manifest() -> None:
    candidates: list[tuple[Path, str, str, str]] = []
    gate0 = ARTIFACTS / "gate0"
    for path in sorted(gate0.glob("*.npz")):
        if path.name.startswith("gate0b_") and path.stat().st_size < 1_000_000:
            continue  # These compact source arrays are already Git archived.
        candidates.append((path, "Gate-0 raw numerical audit", "YES", "future_lfs_candidate"))
    for path in sorted(gate0.glob("*.pt")):
        candidates.append((path, "Gate-0 frozen/raw tensor evidence", "YES", "future_lfs_candidate"))

    clean = ARTIFACTS / "phase1_rerun"
    for path in sorted(clean.glob("runs/*/validation_*.npz")):
        candidates.append((path, "Clean rerun physical validation arrays", "YES", "future_lfs_candidate"))
    for path in sorted(clean.glob("leakage/*.npz")):
        candidates.append((path, "Clean rerun leakage mechanism arrays", "YES", "future_lfs_candidate"))

    rows = []
    for path, role, aggregate_without, category in candidates:
        rows.append(
            {
                "original_path": path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "scientific_role": role,
                "aggregate_analysis_without_binary": aggregate_without,
                "archive_class": category,
            }
        )
    write_tsv(
        REPO / "results" / "raw_binary_manifest.tsv",
        rows,
        [
            "original_path",
            "size_bytes",
            "sha256",
            "scientific_role",
            "aggregate_analysis_without_binary",
            "archive_class",
        ],
    )


def build_integrity_files() -> None:
    rows: list[dict[str, object]] = []
    source_roots = {
        "results/frozen/gate0": ARTIFACTS / "gate0",
        "results/frozen/phase1_clean": ARTIFACTS / "phase1_rerun",
        "results/frozen/forensic": ARTIFACTS / "forensic",
        "results/frozen/resume_gate": ARTIFACTS / "resume_gate",
        "results/frozen/historical_invalid_phase1": ARTIFACTS / "phase1",
    }
    for path in sorted(FROZEN.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO).as_posix()
        original: Path | None = None
        if relative == "results/frozen/phase1_clean/split_manifest.json":
            original = ARTIFACTS / "phase1" / "split_manifest.json"
        else:
            for prefix, source_root in source_roots.items():
                if relative == prefix or relative.startswith(prefix + "/"):
                    original = source_root / Path(relative).relative_to(prefix)
                    break
        if original is None or not original.is_file():
            raise RuntimeError(f"unable to map archived source: {path}")
        stored_hash = sha256(path)
        original_hash = sha256(original)
        if stored_hash != original_hash:
            raise RuntimeError(f"archived file not byte-identical: {path}")
        archive_class = "historical_invalid" if "historical_invalid_phase1" in relative else "stored_in_git"
        rows.append(
            {
                "archive_class": archive_class,
                "stored_path": relative,
                "original_path": original,
                "size_bytes": path.stat().st_size,
                "original_sha256": original_hash,
                "stored_or_portable_sha256": stored_hash,
                "path_substitutions": "NONE_BYTE_IDENTICAL_COPY",
                "scientific_role": "frozen structured/figure evidence",
            }
        )

    # Add authored recovery metadata and scripts present at generation time.
    for base in (REPO / "recovery", REPO / "checkpoints"):
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or path.name in {"SHA256SUMS", "artifact_manifest.tsv"}
                or is_cache_artifact(path)
            ):
                continue
            relative = path.relative_to(REPO).as_posix()
            rows.append(
                {
                    "archive_class": "stored_in_git",
                    "stored_path": relative,
                    "original_path": relative,
                    "size_bytes": path.stat().st_size,
                    "original_sha256": sha256(path),
                    "stored_or_portable_sha256": sha256(path),
                    "path_substitutions": "NOT_APPLICABLE_AUTHORED_RECOVERY_FILE",
                    "scientific_role": "recovery metadata/tooling",
                }
            )

    raw_manifest = REPO / "results" / "raw_binary_manifest.tsv"
    if raw_manifest.is_file():
        rows.append(
            {
                "archive_class": "stored_in_git",
                "stored_path": "results/raw_binary_manifest.tsv",
                "original_path": "results/raw_binary_manifest.tsv",
                "size_bytes": raw_manifest.stat().st_size,
                "original_sha256": sha256(raw_manifest),
                "stored_or_portable_sha256": sha256(raw_manifest),
                "path_substitutions": "NOT_APPLICABLE_AUTHORED_RECOVERY_FILE",
                "scientific_role": "external raw-binary inventory",
            }
        )

    with (REPO / "checkpoints" / "LFS_CANDIDATES.tsv").open(encoding="utf-8") as handle:
        for candidate in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "archive_class": "future_lfs_candidate",
                    "stored_path": candidate["expected_recovery_destination"],
                    "original_path": candidate["original_path"],
                    "size_bytes": candidate["size_bytes"],
                    "original_sha256": candidate["sha256"],
                    "stored_or_portable_sha256": "NOT_STORED_IN_PASS_2A",
                    "path_substitutions": "RECOVERY_DESTINATION_ONLY",
                    "scientific_role": "minimum unique final inference weights",
                }
            )

    with raw_manifest.open(encoding="utf-8") as handle:
        for candidate in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "archive_class": "future_lfs_candidate",
                    "stored_path": "NOT_STORED_IN_PASS_2A",
                    "original_path": candidate["original_path"],
                    "size_bytes": candidate["size_bytes"],
                    "original_sha256": candidate["sha256"],
                    "stored_or_portable_sha256": "NOT_STORED_IN_PASS_2A",
                    "path_substitutions": "NONE",
                    "scientific_role": candidate["scientific_role"],
                }
            )

    omissions = [
        ("regenerable_upstream", "upstream/lerobot", "LeRobot exact pinned checkout"),
        ("regenerable_upstream", "datasets/lerobot/libero", "pinned Hugging Face LIBERO dataset"),
        ("regenerable_upstream", "models/huggingface", "pinned upstream model snapshots"),
        ("intentionally_omitted", "checkpoints/**/complete_state.pt", "large full training-resume states"),
        ("intentionally_omitted", "generated_artifacts/**/logs", "large/transient runtime logs"),
        ("environment_only", "conda_envs/flowspec-vla", "nonportable historical venv"),
    ]
    for archive_class, original, role in omissions:
        rows.append(
            {
                "archive_class": archive_class,
                "stored_path": "NOT_STORED_IN_PASS_2A",
                "original_path": original,
                "size_bytes": "UNKNOWN_OR_MULTIPLE",
                "original_sha256": "NOT_APPLICABLE_OR_INVENTORIED_ELSEWHERE",
                "stored_or_portable_sha256": "NOT_STORED_IN_PASS_2A",
                "path_substitutions": "NONE",
                "scientific_role": role,
            }
        )

    columns = [
        "archive_class",
        "stored_path",
        "original_path",
        "size_bytes",
        "original_sha256",
        "stored_or_portable_sha256",
        "path_substitutions",
        "scientific_role",
    ]
    write_tsv(REPO / "recovery" / "artifact_manifest.tsv", rows, columns)

    checksum_paths: set[Path] = set()
    for base in (FROZEN, REPO / "recovery", REPO / "checkpoints"):
        checksum_paths.update(
            path for path in base.rglob("*")
            if path.is_file() and not is_cache_artifact(path)
        )
    checksum_paths.add(raw_manifest)
    checksum_paths.discard(REPO / "recovery" / "SHA256SUMS")
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(REPO).as_posix()}"
        for path in sorted(checksum_paths)
    ]
    (REPO / "recovery" / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main() -> None:
    archive_small_evidence()
    build_checkpoint_manifests()
    build_raw_binary_manifest()
    export_current_package_state()
    # Run again after all authored documentation is present to refresh manifests.
    build_integrity_files()
    print("FlowSpec-VLA local recovery evidence snapshot built successfully")


if __name__ == "__main__":
    main()
