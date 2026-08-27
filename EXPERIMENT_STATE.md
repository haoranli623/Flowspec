# FlowSpec-VLA Experiment State

Last updated: 2026-08-27

## Current state

The deterministic clean Phase-1 rerun is complete. No training, leakage, or
rollout process remains active. The project is stopped at the authorized
Phase-1 boundary.

## Frozen substrate

- protocol: `82dfe670e6073bc30397d699ee612c0386932a00`
- execution: `1a69e586cd0fad70a5fe71f27a33eddaaec38901`
- SmolVLA base: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`
- LIBERO: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- split: 320 train episodes / 53,762 frames; 200 held-out episodes /
  400 validation states

## Completed evidence

- method-integrity and deterministic resume gates: PASS
- primary matrix: 9/9 runs, 5,000 updates and 160,000 examples/run
- M1/M2 training invariance: exact in all seeds
- mechanism re-audit: 12/12 complete
- rollout: 9/9 checkpoints and 1,800/1,800 episodes; structural audit PASS

Final physical NRMSE means are M0 `0.71379`, M1 `0.71696`, and M2 `0.71439`.
Rollout means are M0 `48.33%`, M1 `44.83%`, and M2 `47.33%`. Post-training
mean `R_leak` is M0 `0.015760`, M1 `0.103029`, and M2 `1.399e-13`.

## Decision

**P2. MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK.**

M2 removes the measured nuisance pathway but does not improve matched-budget
validation or rollout over M0. M1 also provides no benefit. No VLA performance
claim is supported. P2 does not trigger the conditional Phase-2 authorization.
Do not begin further experiments without new authorization.

## Outputs

- report: `PHASE1_RERUN_REPORT.md`
- artifacts: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun/`
- manifest: `phase1_rerun/artifact_manifest.json`
- accounted compute: at least `62.9174` device-hours; exact total unavailable
  because of a short unmetered interrupted tail
