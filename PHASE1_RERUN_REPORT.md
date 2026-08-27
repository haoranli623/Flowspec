# FlowSpec-VLA Clean Phase-1 Rerun Report

## 1. Executive verdict

**P2. MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK.** M2 suppresses
padded-noise leakage to numerical zero in all three seeds, but neither M1 nor
M2 improves preregistered physical-action validation or frozen LIBERO rollout
success over M0. This is a negative practical result. Phase 1 stops, and the
conditional Phase-2 authorization is not triggered.

## 2. Research question

Does removing stochastic flow dynamics associated with SmolVLA's 25
non-executable padded action dimensions improve matched-budget LIBERO
post-training? The comparison separates initial padded source noise from
continued padded-state evolution.

## 3. Exact frozen protocol

- clean-rerun protocol: `82dfe670e6073bc30397d699ee612c0386932a00`
- execution implementation: `1a69e586cd0fad70a5fe71f27a33eddaaec38901`
- original Phase-1 protocol: `beb292024fcb37d321338474249d03598cfa5e90`
- deterministic resume gate: `67baa3ad689a158dd34fdd1b751387bf338671c2`

No method, seed, data fraction, checkpoint rule, metric, or threshold was
changed after comparative results were available.

## 4. Substrate and revisions

- LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`
- `lerobot/smolvla_base`: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- `lerobot/libero`: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- two RTX 3090 24 GiB GPUs; bfloat16 autocast
- deterministic algorithms, cuBLAS workspace `:4096:8`, TF32 disabled

## 5. M0/M1/M2 implementation

- **M0:** native 32D Gaussian source and native 32D flow.
- **M1:** padded source coordinates are zero; later padded evolution remains.
- **M2:** padded source is zero and padded state is projected to zero after
  every inference flow update.

Official training evaluates one flow time without iterative ODE updates, so
M1 and M2 have exactly the same training computation; M2 differs during
sampling/evaluation.

## 6. Verification tests

The method-integrity gate passed: official M0 equivalence, exact M1 masking,
exact M2 projection, valid-coordinate/gradient preservation, and shared
targets/normalization/output slicing. Smoke and primary resume checks passed.
M0/M1 seed 520003 resumed from complete update-1000 checkpoints and reproduced
their durable 1001--1005 traces exactly. Final M1/M2 checkpoints are bitwise
identical in all seeds (`max_abs=0`, 500 tensors each).

## 7. Baseline reproduction

The deterministic 250-update M0 gate passed: first/last 50-update loss
`2.18908/0.75096`; physical NRMSE `1.30475 -> 0.88755`; fixed-flow MSE
`2.37081 -> 0.67592`; peak memory `18.2632 GiB`; zero non-finite events; exact
checkpoint resume.

## 8. Dataset and split

All 40 LIBERO tasks; 320 training episodes; 53,762 training frames; fixed
160,000-example sequence/run; 200 held-out episodes and 400 validation states.
Split SHA-256: `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0`.
No training episode entered validation.

## 9. Optimization

Nine runs (3 methods x 3 seeds), seeds `520001/520002/520003`, 5,000 updates
and 160,000 examples/run. AdamW, LR `1e-4 -> 2.5e-6`, betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay `1e-10`, 200-update warmup, microbatch 16,
accumulation 2, effective batch 32, gradient clip 10. Checkpoints were at
1,000/2,500/5,000 and validation at 0/250/500/1,000/2,000/3,000/4,000/5,000.

## 10. Physical-space validation results

Final update-5,000 whole-action physical NRMSE (lower is better):

| Method | 520001 | 520002 | 520003 | Mean | SD |
|---|---:|---:|---:|---:|---:|
| M0 | 0.71300 | 0.71481 | 0.71356 | 0.71379 | 0.00093 |
| M1 | 0.71712 | 0.72172 | 0.71203 | 0.71696 | 0.00485 |
| M2 | 0.71431 | 0.71937 | 0.70948 | 0.71439 | 0.00495 |

M1 is 0.444% worse than M0 and favors M1 in 1/3 seeds. M2 is 0.084% worse
than M0 and favors M2 in 1/3. Neither meets the frozen 5% and 2/3 rule.

## 11. Learning/sample efficiency

Normalized NRMSE AULC is `0.78192 +/- 0.00057` (M0),
`0.78581 +/- 0.00263` (M1; 0.498% worse), and
`0.78370 +/- 0.00272` (M2; 0.227% worse). M2 is consistently 0.358% better
than M1 finally and 0.270% better in AULC, far below material thresholds; M1
and M2 are practically equivalent under the frozen 2% band.

## 12. Leakage/mechanism audit

Gate-0's task-checkpoint reference was padded RMS `0.01047` and
`R_leak=0.05730`. The distinct Phase-1 starting base has M0 padded RMS
`0.02351` and `R_leak=0.00634`.

| Method | Post-train padded RMS | Post-train R_leak | Strong seeds |
|---|---:|---:|---:|
| M0 | 0.035265 | 0.015760 | 0/3 |
| M1 | 0.090083 | 0.103029 | 0/3 |
| M2 | 1.070e-7 | 1.399e-13 | 3/3 |

M2 exceeds both `<=1e-4` suppression requirements and keeps padded state at
numerical zero. M1 does not suppress diagnostic sensitivity; its operationally
zero padded source evolves away from zero.

## 13. LIBERO rollout success

All update-5,000 checkpoints were evaluated on four suites, 10 tasks/suite and
5 episodes/task: 200/checkpoint and 1,800 total. Full structural audit passed.

| Method | Seed rates | Mean | SD | vs M0 |
|---|---|---:|---:|---:|
| M0 | 49.0%, 46.5%, 49.5% | 48.33% | 1.61 pp | -- |
| M1 | 48.5%, 41.5%, 44.5% | 44.83% | 3.51 pp | -3.50 pp |
| M2 | 48.5%, 46.0%, 47.5% | 47.33% | 1.26 pp | -1.00 pp |

M1/M2 favor the candidate over M0 in 0/3 seeds. M2 exceeds M1 by 2.50 points
in 2/3 seeds, below the frozen 5-point supportive threshold.

## 14. Seed-level statistical analysis

Training seed is the independent replicate (`n=3`); frames/tasks/episodes are
not treated as training replicates. M1-minus-M0 final NRMSE is `+0.003170`
(`dz=0.738`), M2-minus-M0 is `+0.000600` (`dz=0.137`), and M2-minus-M1 is
`-0.002570` in all seeds. Conclusions use magnitude and consistency, not
p-values.

## 15. Failure cases

- M0/M1 seed 520003 lost their interactive processes but resumed exactly from
  complete update-1,000 state. Early physical arrays survived; early fixed-flow
  MSE was unavailable and is not fabricated.
- M2 seed 520003 was initially omitted, then run in full before valid analysis.
- Early rollout launches failed before recording episodes due to headless GL.
  Every counted episode used the same OSMesa substrate; both queues exited 0.
- Rollout records store success/reward but not a separate termination-vs-horizon
  field.

## 16. Interpretation

M1 improves neither validation, AULC, rollout, nor leakage. M2 removes the
nuisance pathway and is slightly better than M1, but does not materially beat
M0. Gate-0 leakage is measurable; removing it is not practically beneficial
under this frozen Phase-1 setting.

## 17. Limitations

One model family/base checkpoint and LIBERO only; three seeds; one data/budget
regime; five rollout episodes/task; M2 differs from M1 only at inference under
official single-time FM training; no explicit rollout termination causes; and
exact all-inclusive device time is unavailable after interruption.

## 18. Total GPU usage

Accounted usage is at least **62.9174 device-hours**: smoke `0.4604`, smoke
resume `0.0713`, reported primary `19.2767`, interrupted durable primary
`0.9799`, primary spotchecks `0.0747`, leakage `0.3460`, and rollout `41.7083`.
A short interrupted tail and method-integrity setup were unmetered, so an exact
total cannot be honestly reconstructed. The lower bound is 9.08 hours below
the 72-hour ceiling.

## 19. Scientific decision

> **P2. MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK**

No FlowSpec VLA-performance improvement claim is supported.

## 20. Scientifically justified next step

Stop after Phase 1. Conditional Phase 2 requires P1/P3/P4 and is not authorized
by P2. Any future explanation of the small M2-over-M1 difference is report-only.

## 21. Explicitly not tested

No M3/additional mask; auxiliary/Jacobian/representation/leakage loss;
normalization/adapter; RL/GRPO/preference/reward/critic/world model; architecture,
model-size, chunk, optimizer, or method-specific LR change; alternate fraction,
task subset, checkpoint, adaptive seed/episode extension; cross-model/dataset
study; Phase 2/3; or paper experiment.

## Artifacts

Machine-readable summaries and hashes are under
`/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun/`, including
`validation_summary.json`, `training_summary.json`,
`leakage_reaudit_summary.json`, `rollout_summary.json`,
`seed_pairing_summary.json`, `reproducibility_spotcheck.json`, `gpu_usage.json`,
and `artifact_manifest.json`.
