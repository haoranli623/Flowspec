#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

python3 - "$repo_root" <<'PY'
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])


def load(relative):
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"MISSING REQUIRED FROZEN ARTIFACT: {relative}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual, expected, *, atol=1e-12, label="value"):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise SystemExit(f"MISMATCH {label}: got {actual!r}, expected {expected!r}")


# Gate 0: independently rebuild the median-variance ratio from per-state CSV.
# The CSV stores RMS values, so squaring its decimal serialization introduces a
# few parts in 1e9 of round-trip error relative to the original float arrays.
csv_path = root / "results/frozen/gate0/gate0a_per_state.csv"
if not csv_path.is_file():
    raise SystemExit("MISSING REQUIRED FROZEN ARTIFACT: results/frozen/gate0/gate0a_per_state.csv")
with csv_path.open(encoding="utf-8", newline="") as handle:
    gate_rows = list(csv.DictReader(handle))
if len(gate_rows) != 64:
    raise SystemExit(f"Gate-0 state count mismatch: {len(gate_rows)}")
pad_variance_median = statistics.median(float(row["rms_pad_standardized"]) ** 2 for row in gate_rows)
valid_variance_median = statistics.median(float(row["rms_valid_standardized"]) ** 2 for row in gate_rows)
gate_ratio = pad_variance_median / valid_variance_median
gate_summary = load("results/frozen/gate0/gate0a_summary.json")
gate_frozen = float(gate_summary["primary"]["r_leak_ratio_of_median_variances"])
close(gate_ratio, gate_frozen, atol=5e-9, label="Gate-0 R_leak CSV replay")
close(gate_frozen, 0.05729682371020317, label="Gate-0 frozen R_leak")

# Validation: recompute means and the paired M2/M1 result from seed values.
validation = load("results/frozen/phase1_clean/validation_summary.json")
expected_nrmse = {"m0": 0.7137868102588092, "m1": 0.7169568478235702, "m2": 0.7143867055090901}
nrmse_values = {}
for method, expected in expected_nrmse.items():
    block = validation["methods"][method]["final_nrmse"]
    values = [float(value) for value in block["values"]]
    if len(values) != 3:
        raise SystemExit(f"{method} NRMSE seed count mismatch: {len(values)}")
    replay_mean = statistics.fmean(values)
    close(replay_mean, float(block["mean"]), label=f"{method} NRMSE stored mean")
    close(replay_mean, expected, label=f"{method} frozen NRMSE")
    nrmse_values[method] = values
m2_better = sum(candidate < baseline for candidate, baseline in zip(nrmse_values["m2"], nrmse_values["m1"]))
if m2_better != 3:
    raise SystemExit(f"M2-vs-M1 NRMSE directional count mismatch: {m2_better}/3")
relative = (expected_nrmse["m1"] - expected_nrmse["m2"]) / expected_nrmse["m1"]
close(relative, 0.0035847935929227853, label="M2-vs-M1 relative NRMSE reduction")

# Leakage: reconstruct method means from the three seed-level ratios.
leakage = load("results/frozen/phase1_clean/leakage_reaudit_summary.json")
expected_leakage = {"m0": 0.015760391329725582, "m1": 0.10302889595429103, "m2": 1.3989087485332766e-13}
for method, expected in expected_leakage.items():
    block = leakage["method_summary"][method]
    replay_mean = statistics.fmean(float(value) for value in block["r_leak_by_seed"])
    close(replay_mean, float(block["mean_r_leak"]), atol=1e-18, label=f"{method} stored leakage mean")
    close(replay_mean, expected, atol=1e-18, label=f"{method} frozen leakage")

# Rollout: independently aggregate the nine episode-level JSON records.
rollout_dir = root / "results/frozen/phase1_clean/rollouts"
files = sorted(rollout_dir.glob("*.json"))
if len(files) != 9:
    raise SystemExit(f"rollout checkpoint file count mismatch: {len(files)}")
expected_runs = {(method, seed) for method in ("m0", "m1", "m2") for seed in (520001, 520002, 520003)}
seen_runs = set()
successes = defaultdict(int)
episodes = defaultdict(int)
suites = set()
task_keys = set()
run_rates = defaultdict(dict)
for path in files:
    record = load(path.relative_to(root))
    if record.get("status") != "COMPLETE" or int(record.get("checkpoint_update", -1)) != 5000:
        raise SystemExit(f"invalid rollout status/checkpoint: {path.name}")
    key = (record["method"], int(record["seed"]))
    if key in seen_runs:
        raise SystemExit(f"duplicate rollout run: {key}")
    seen_runs.add(key)
    run_successes = 0
    run_episodes = 0
    local_tasks = set()
    for task in record["tasks"]:
        task_key = (task["suite"], int(task["task_id"]))
        if task_key in local_tasks:
            raise SystemExit(f"duplicate task in {path.name}: {task_key}")
        local_tasks.add(task_key)
        task_keys.add(task_key)
        suites.add(task["suite"])
        eps = task["episodes"]
        if len(eps) != 5 or sorted(int(ep["episode_ix"]) for ep in eps) != list(range(5)):
            raise SystemExit(f"invalid episode coverage in {path.name}: {task_key}")
        ep_seeds = [int(ep["seed"]) for ep in eps]
        if len(ep_seeds) != len(set(ep_seeds)):
            raise SystemExit(f"duplicate episode seed in {path.name}: {task_key}")
        task_successes = sum(bool(ep["success"]) for ep in eps)
        if task_successes != int(task["successes"]):
            raise SystemExit(f"task success mismatch in {path.name}: {task_key}")
        run_successes += task_successes
        run_episodes += len(eps)
    if len(local_tasks) != 40 or run_episodes != 200:
        raise SystemExit(f"run coverage mismatch in {path.name}: tasks={len(local_tasks)}, episodes={run_episodes}")
    if run_successes != int(record["successes"]):
        raise SystemExit(f"run success mismatch in {path.name}")
    successes[key[0]] += run_successes
    episodes[key[0]] += run_episodes
    run_rates[key[0]][key[1]] = run_successes / run_episodes

if seen_runs != expected_runs:
    raise SystemExit(f"rollout run matrix mismatch: missing={expected_runs-seen_runs}, extra={seen_runs-expected_runs}")
if len(suites) != 4 or len(task_keys) != 40:
    raise SystemExit(f"rollout suite/task mismatch: suites={len(suites)}, tasks={len(task_keys)}")
expected_success = {"m0": 290, "m1": 269, "m2": 284}
if dict(successes) != expected_success or dict(episodes) != {"m0": 600, "m1": 600, "m2": 600}:
    raise SystemExit(f"rollout aggregate mismatch: successes={dict(successes)}, episodes={dict(episodes)}")
if sum(successes.values()) != 843 or sum(episodes.values()) != 1800:
    raise SystemExit("rollout grand-total mismatch")
rollout_summary = load("results/frozen/phase1_clean/rollout_summary.json")
for method in expected_success:
    close(
        successes[method] / episodes[method],
        float(rollout_summary["methods"][method]["mean"]),
        label=f"{method} rollout summary",
    )

m1_better_validation = sum(a < b for a, b in zip(nrmse_values["m1"], nrmse_values["m0"]))
m2_better_rollout = sum(run_rates["m2"][seed] > run_rates["m1"][seed] for seed in (520001, 520002, 520003))
m2_tie_rollout = sum(run_rates["m2"][seed] == run_rates["m1"][seed] for seed in (520001, 520002, 520003))

print("PASS: analysis-only FlowSpec-VLA replay")
print(f"Gate-0 R_leak: {gate_frozen:.11f} (CSV replay {gate_ratio:.11f})")
print("Final physical NRMSE: " + ", ".join(f"{m.upper()}={expected_nrmse[m]:.8f}" for m in ("m0", "m1", "m2")))
print(f"M2 vs M1 relative NRMSE reduction: {relative * 100:.6f}% (directionally better 3/3 seeds)")
print(f"M1 vs M0 NRMSE directionally better: {m1_better_validation}/3 seeds")
print("Trained R_leak: " + ", ".join(f"{m.upper()}={expected_leakage[m]:.11g}" for m in ("m0", "m1", "m2")))
print(f"Rollout: M0={successes['m0']}/600, M1={successes['m1']}/600, M2={successes['m2']}/600")
print(f"Rollout matrix: 9 checkpoints, {len(suites)} suites, {len(task_keys)} tasks, 1800 episodes, 843 successes")
print("Rollout video dependency: none (episode-level replay used JSON only)")
print(f"M2 vs M1 rollout: strictly better {m2_better_rollout}/3 seeds, tied {m2_tie_rollout}/3")
PY
