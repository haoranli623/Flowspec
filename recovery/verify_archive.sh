#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"

required=(
  recovery/RECOVERY.md
  recovery/ENVIRONMENT.md
  recovery/UPSTREAMS.yaml
  recovery/artifact_manifest.tsv
  recovery/SHA256SUMS
  recovery/reproduce_key_results.sh
  recovery/download_upstreams.sh
  recovery/patches/robosuite-1.4.0-log-path.patch
  checkpoints/checkpoint_map.json
  checkpoints/LFS_CANDIDATES.tsv
  results/raw_binary_manifest.tsv
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "MISSING: $path" >&2; exit 1; }
done

sha256sum --check --quiet recovery/SHA256SUMS

python3 - "$repo_root" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint_map = json.loads((root / "checkpoints/checkpoint_map.json").read_text())
if len(checkpoint_map["final_rollout_checkpoints"]) != 9:
    raise SystemExit("checkpoint map must contain 9 final checkpoints")
if len(checkpoint_map["intermediate_checkpoint_inventories"]) != 18:
    raise SystemExit("checkpoint map must contain 18 update-1000/2500 inventories")
with (root / "checkpoints/LFS_CANDIDATES.tsv").open(newline="") as handle:
    lfs = list(csv.DictReader(handle, delimiter="\t"))
if len(lfs) != 6 or len({row["sha256"] for row in lfs}) != 6:
    raise SystemExit("LFS candidate manifest must contain 6 uniquely hashed inference weights")
with (root / "results/raw_binary_manifest.tsv").open(newline="") as handle:
    raw = list(csv.DictReader(handle, delimiter="\t"))
if not raw or any(len(row["sha256"]) != 64 for row in raw):
    raise SystemExit("raw binary manifest hash coverage is incomplete")
runs = list((root / "results/frozen/phase1_clean/runs").glob("*/run.json"))
traces = list((root / "results/frozen/phase1_clean/runs").glob("*/trace.json"))
inventories = list((root / "results/frozen/phase1_clean/runs").glob("*/checkpoint_*_inventory.json"))
rollouts = list((root / "results/frozen/phase1_clean/rollouts").glob("*.json"))
if (len(runs), len(traces), len(inventories), len(rollouts)) != (9, 9, 27, 9):
    raise SystemExit(
        f"structured evidence count mismatch: runs={len(runs)}, traces={len(traces)}, "
        f"inventories={len(inventories)}, rollouts={len(rollouts)}"
    )
print(f"PASS: structured counts (9 runs, 9 traces, 27 inventories, 9 rollouts)")
print(f"PASS: 6 unique inference candidates; {len(raw)} raw-binary candidates fully hashed")
PY

recovery/reproduce_key_results.sh
echo "PASS: archive checksums and analysis replay"
