# FlowSpec-VLA disaster recovery

Assumption: **The original Shannon/NAS machine no longer exists.** This guide reconstructs what can be recovered from this repository plus separately retained private assets. It does not assume `/mnt/NAS`, the original username, the old venv, or any old cache survives.

## 1. Repository structure

- Root Markdown files are the immutable scientific protocols, audit reports, final rerun report, state record, and review bundle.
- `src/`, `scripts/`, and `config/` contain the scientific implementation and frozen configurations.
- `results/frozen/gate0/` contains byte-identical compact Gate-0 evidence, figures, and small Gate-0B source arrays.
- `results/frozen/phase1_clean/` contains byte-identical clean-rerun summaries, all 9 final run/trace records, all 27 checkpoint inventories, all 9 rollout records, mechanism summaries, and figures.
- `results/frozen/forensic/` and `results/frozen/resume_gate/` establish the CUDA nondeterminism investigation and resume validity.
- `results/frozen/historical_invalid_phase1/` preserves the earlier invalid Phase-1 aggregate record without promoting it to valid evidence.
- `checkpoints/` inventories final and intermediate checkpoints; no large weight file is stored in ordinary Git.
- `recovery/` contains upstream/environment provenance, the robosuite patch, verification/replay scripts, checksums, and the complete asset manifest.

All archived scientific files are byte-identical to their source artifacts. Absolute `/mnt/NAS/...` strings inside them are historical provenance, not live dependencies. No transformed result copy was needed: recovery scripts use repository-relative archived paths. `artifact_manifest.tsv` records `NONE_BYTE_IDENTICAL_COPY` for these records; future installations use `path_map.example.env`.

## 2. Final scientific state

The frozen pre-packaging scientific commit is `2064fb43073c5f2ef5254550ce107028cc29d492`, annotated by local tag `flowspec-vla-scientific-final-2026-08-27`. Gate 0 found a real padded-action pathway (`R_leak = 0.05729682371020317`). The clean Phase-1 rerun found final physical NRMSE means M0 `0.7137868102588092`, M1 `0.7169568478235702`, and M2 `0.7143867055090901`. M2 was directionally better than M1 in 3/3 paired seeds, but by only `0.35847935929227853%`, inside the frozen practical-equivalence band.

Trained mean `R_leak` was M0 `0.015760391329725582`, M1 `0.10302889595429103`, and M2 `1.3989087485332766e-13`. Thus M2 removed the measured nuisance pathway essentially to numerical zero. Rollout success was M0 `290/600` (48.33%), M1 `269/600` (44.83%), and M2 `284/600` (47.33%), across 9 checkpoints, 40 tasks, 4 suites, and 1,800 episodes.

The scientific conclusion is negative for practical benefit: padded-action leakage is real and M2 removes it, but the frozen SmolVLA/LIBERO protocol does not show improved downstream control. No Phase 2 was authorized.

## 3. Upstream dependencies and exact revisions

`UPSTREAMS.yaml` is authoritative. Exactly pinned items are LeRobot commit `8b256a6c0d4769c3cc3e7e98f04940126398a391`, SmolVLA base revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, SmolVLA LIBERO recipe revision `31d453f7edd78c839a8bbc39744a292686daf0de`, SmolVLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, and LeRobot LIBERO dataset revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. Package versions without wheel hashes are marked `PARTIALLY_PINNED`; missing system versions are `UNKNOWN` rather than guessed.

`download_upstreams.sh` prints the pinned reconstruction plan by default and performs downloads only with an explicit `--execute` argument.

## 4. Environment reconstruction

Follow `ENVIRONMENT.md`. Never copy or archive the old venv as a solution: its interpreter metadata and editable-package paths are machine-specific. Retain new wheel hashes, OS package versions, driver version, and container digest during reconstruction to improve future provenance.

## 5. OSMesa, Mujoco, and LIBERO

The valid rollout substrate used Mujoco 3.8.1, LIBERO through `hf_libero` 0.1.4, robosuite 1.4.0, relative control, two 256x256 cameras, the standard 8D LIBERO state, and `MUJOCO_GL=osmesa`. Install a distribution-appropriate Mesa/OSMesa implementation and verify a headless smoke test before any scientific rollout. Early non-OSMesa launch attempts produced no counted episodes; the archived nine final rollout JSON files contain the complete counted evaluation. No rollout video is required for recovery or aggregation.

## 6. Robosuite patch application

The sole installed dependency modification is `robosuite/utils/log_utils.py`. Against the clean robosuite 1.4.0 wheel (`aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994`), the clean file hash is `f89259ab1809772f10353bc2c4ce45d29cd4017f27ffde225e5e70c0c60a3199`. Apply:

```bash
cd /path/to/site-packages
patch -p1 < /path/to/flowspec-vla/recovery/patches/robosuite-1.4.0-log-path.patch
sha256sum robosuite/utils/log_utils.py
```

The expected patched hash is `02be2de43b701b02b7da7f5863c733c35fa00945cc91dbb2e54e3b2413235290`. The patch only imports `os` and makes `logging.FileHandler` read `ROBOSUITE_LOG_PATH`, falling back to `/tmp/robosuite.log`; it does not alter simulator or policy semantics.

## 7. SmolVLA and LeRobot reconstruction

Clone LeRobot at its exact commit, download the three pinned model/recipe snapshots, and use the existing FlowSpec source/configs. The official baseline M0 retains full-dimensional source and flow. M1 masks only padded source noise. M2 masks padded source noise and projects padded flow state after every update. Do not infer M2 solely from its weight bytes: M2 final tensors equal same-seed M1 tensors, while M2 behavior differs through inference-time projection in the FlowSpec implementation.

## 8. Dataset reconstruction

Download `lerobot/libero` at its exact revision. The archived `results/frozen/phase1_clean/split_manifest.json` is the frozen deterministic split and normalization evidence: 53,762 selected training frames and 400 validation states across all 40 tasks. Never regenerate a new split and call it equivalent. Verify IDs, episode membership, and training statistics against the archived manifest before full reproduction.

## 9. Checkpoint reconstruction

`checkpoints/checkpoint_map.json` inventories every update-1,000, update-2,500, and update-5,000 checkpoint state using archived inventory hashes. `checkpoints/LFS_CANDIDATES.tsv` lists the six unique final inference weight files (3 M0 and 3 M1). M2 same-seed tensors are identical to M1, but M2 must be invoked with projection. Copy independently retained weights to each listed `expected_recovery_destination`, verify SHA256, and never substitute a checkpoint selected after observing results.

Full training resume requires the matching `complete_state.pt`, including optimizer/RNG/sampler state. These files are intentionally omitted from PASS 2A; their complete-state hashes and sizes remain in the checkpoint map and archived inventories.

## 10. Analysis-only reproduction

Run:

```bash
recovery/verify_archive.sh
recovery/reproduce_key_results.sh
```

This verifies Git-stored checksums and independently aggregates the nine rollout JSON files, Gate-0 ratio, validation NRMSE, trained leakage, paired M2/M1 direction, suites/tasks, and total episodes. It requires only Python's standard library and the Git-tracked evidence.

## 11. Full GPU reproduction

After upstreams, environment, dataset, and base assets are reconstructed, follow the frozen protocol commits and existing scripts. Reproduce the baseline gate before the 9-run matrix; use the exact deterministic settings, seeds `520001`–`520003`, 5,000 updates/run, shared split, and fixed final checkpoint. Full rollout requires the 9 logical checkpoints (M2 uses same-seed M1 weights plus M2 projection), four suites, 40 tasks, five episodes/task, and OSMesa. This snapshot enables reconstruction but does not claim bitwise GPU identity because driver, system library, and most wheel hashes were not frozen.

## 12. Intentionally omitted large artifacts

Ordinary Git excludes all model weights, `complete_state.pt`, large Gate-0 raw arrays, clean validation/leakage NPZ files, dataset bytes, Hugging Face cache, LeRobot checkout, venv, simulator outputs, videos, and large logs. The six minimum unique final inference weights and deep-audit arrays are fully hashed in manifests for a later private/LFS asset pass. Upstreams are revision-pinned and regenerable. Aggregate analysis does not need omitted NPZ files or videos.

## 13. Historical invalid Phase-1 results

The first Phase-1 run was invalidated after resume/reproducibility investigation exposed CUDA execution nondeterminism and insufficient exact-resume guarantees. It is retained under `historical_invalid_phase1` as forensic history, not as a replicate and not as evidence for the final comparison. `FORENSIC_REPORT.md`, `RESUME_GATE_REPORT.md`, and the clean-rerun report document the correction. Never combine historical invalid metrics with the clean 9-run matrix.

## 14. Two known interpretation/documentation issues

First, final documents and the machine-readable decision record use **P2 — MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK**, while a literal application of the original frozen decision-tree wording appears to map the observed lack of downstream improvement to **P5 — NO-GO, LEAKAGE NOT PRACTICALLY HARMFUL**. This is a terminology/decision-tree consistency issue, not data corruption. Preserve both facts; do not silently relabel historical reports.

Second, the displayed NRMSE formula in the rerun protocol is not textually identical to the original protocol and actual implementation. Final reported values use the implemented original metric: divide each physical-coordinate error by that coordinate's frozen training-split standard deviation, square, average across valid 7D chunk coordinates, then take the square root. The numerical artifacts and final report consistently use that implementation. This is a formula-documentation discrepancy, not altered result data.
