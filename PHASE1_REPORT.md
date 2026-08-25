# FlowSpec-VLA Phase-1 Report

Date: 2026-08-25  
Protocol-freeze commit: `beb292024fcb37d321338474249d03598cfa5e90`

## 1. Executive verdict

**P6. NO-GO — TRAINING / SUBSTRATE INVALID**

The official-baseline reproduction passed and all nine planned primary runs completed, but the
preregistered M1/M2 reproducibility invariant failed. M1 and M2 have the same training computation,
yet matched final checkpoints differed by maximum absolute parameter values of `0.02718`, `0.03040`,
and `0.02802`, versus the frozen tolerance of `1e-6`. Repeating M1 seed 520001 on the same physical GPU
as its paired M2 run still produced `0.02034` maximum parameter difference at update 1,000. The issue
was therefore unresolved. Per the frozen stop condition, the apparent method differences are
descriptive only; trained-checkpoint leakage audits and LIBERO rollouts were not run.

## 2. Research question

Does removing stochastic flow dynamics in SmolVLA's 25 non-executable padded action dimensions
improve matched-budget LIBERO post-training, and is any effect attributable to initial source noise
or continued padded-state evolution?

## 3. Exact frozen protocol

The exact preregistration is `PHASE1_PROTOCOL.md` at commit
`beb292024fcb37d321338474249d03598cfa5e90`. It fixes three methods, three seeds, one split, one
optimizer schedule, fixed update-5,000 selection, physical-space metrics, mechanism audit, rollout
protocol, decision thresholds, compute ceiling, and stop conditions. The relevant stop rule requires
a reproducibility audit when paired M1/M2 checkpoint parameters differ by more than `1e-6`, and halts
comparison if the discrepancy cannot be resolved.

## 4. Substrate and revisions

- Project implementation revision: `19cb03d7de4eb68a4cb10e2dbe82797765f9f560`.
- LeRobot source: `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Base checkpoint: `lerobot/smolvla_base` revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Official LIBERO recipe checkpoint/config: `lerobot/smolvla_libero` revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Dataset: `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- Hardware: two RTX 3090 24 GB GPUs; PyTorch `2.7.1+cu118`; bf16 CUDA autocast.
- Local environment-only patch: robosuite's log `FileHandler` honors `ROBOSUITE_LOG_PATH` because the
  shared `/tmp/robosuite.log` was unwritable. This did not change policy or simulator behavior.

## 5. M0/M1/M2 implementation

- **M0 — Standard FM:** unchanged 32-dimensional Gaussian source and official 10-step Euler sampler.
- **M1 — Source-Masked FM:** source coordinates 7–31 are exactly zero; all coordinates may then evolve.
- **M2 — Full-Subspace FM:** the M1 source mask plus projection of coordinates 7–31 to zero after every
  Euler update. Valid coordinates 0–6 are untouched.

Official SmolVLA training evaluates a single random flow time and does not perform iterative ODE
integration. Consequently M1 and M2 intentionally share the exact training branch; M2's additional
projection exists only in sampling/evaluation. No auxiliary loss, architecture change, normalization
variant, RL method, or fourth method was added.

## 6. Verification tests

Pre-training checks passed twice, including a timed final rerun:

- M0 output exactly matched the official sampler (`max_abs=0`).
- M1/M2 padded source was exactly zero and valid source was unchanged.
- M1 padded-state RMS became nonzero after integration; M2 remained exactly zero at all 11 states.
- M2 projection preserved valid coordinates and valid gradients exactly.
- M1/M2 shared-input training loss difference was exactly zero.
- The executable target, normalization, unnormalization, 7-D slicing, and parameter gradients matched.
- Physical action normalization round-trip maximum error was `2.98e-8`.

These tests validate intervention isolation for a shared execution. They do not repair the later
independent-run reproducibility failure.

## 7. Baseline reproduction

The frozen 250-update M0 reproduction gate **passed**. Mean loss fell from `2.1890` over the first 50
updates to `0.7607` over the last 50 (`0.3475` ratio). Validation physical NRMSE improved from
`1.30455` to `0.88620`, and fixed-noise flow MSE from `2.3713` to `0.6807`. Losses, gradients, and
outputs were finite; peak allocated memory was `18.21 GiB`. This authorized the primary runs.

## 8. Dataset and split

All 40 LIBERO tasks were used. Per task, SHA-256 ranking with seed 510001 selected five held-out
episodes and eight training episodes. The fixed training set contains 320 episodes and 53,762 frames
(`19.659554%` of 273,465 frames). The validation set contains two fixed anchors from each of 200
held-out episodes: 400 states. Training and validation episodes are disjoint. All methods used the
same indices and train-only normalization statistics. Split-manifest SHA-256:
`b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0`.

## 9. Optimization setup

Each primary run used 5,000 optimizer updates and 160,000 examples seen. Microbatch size was 16,
gradient accumulation 2, and effective batch size 32. AdamW used peak LR `1e-4`, betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay `1e-10`, gradient clipping at 10, 200 warmup updates, and cosine decay to
`2.5e-6`. All 392,904,096 trainable parameters were shared across methods. Checkpoints were saved at
1,000, 2,500, and 5,000; validation ran at 0, 250, 500, 1,000, 2,000, 3,000, 4,000, and 5,000.

## 10. Physical-space validation results

The following values are retained for transparency but are **not valid causal method comparisons**:

| Method | Final NRMSE by seed (520001/2/3) | Mean ± SD | Translation RMSE | Rotation RMSE | Gripper RMSE |
|---|---|---:|---:|---:|---:|
| M0 | 0.71485 / 0.71629 / 0.70770 | 0.71295 ± 0.00460 | 0.23832 | 0.05033 | 0.59824 |
| M1 | 0.72087 / 0.72313 / 0.72178 | 0.72192 ± 0.00114 | 0.24153 | 0.05058 | 0.62932 |
| M2 | 0.71463 / 0.72314 / 0.71181 | 0.71653 ± 0.00590 | 0.23945 | 0.05042 | 0.60814 |

Descriptively, M1 was `1.26%` worse than M0 and favored M1 in `0/3` seeds. M2 was `0.50%` worse than
M0 and favored M2 in `1/3` seeds. M2 was `0.75%` better than M1 and favored M2 in `2/3` seeds, but
this was below the frozen 2% equivalence band and is confounded by checkpoint irreproducibility.

## 11. Learning and sample efficiency

Descriptive normalized AULC means were M0 `0.78198`, M1 `0.78826`, and M2 `0.78495` (lower is better).
M1 versus M0 was `0.80%` worse and favored M1 in `0/3` seeds. M2 versus M0 was `0.38%` worse and
favored M2 in `1/3` seeds. M2 versus M1 was `0.42%` better and favored M2 in `3/3` seeds. None met the
frozen 3% practical AULC threshold, and none supports a claim because P6 invalidates the comparison.
The complete 50-position chunk-error arrays remain in `validation_summary.json`.

## 12. Leakage/mechanism audit after training

Not run because the P6 stop condition occurred before mechanism evaluation. No after-training
leakage value is reported. The pre-training Gate-0 reference remains padded RMS `0.0104702`, valid RMS
`0.0437406`, and `R_leak=0.0572968`; it must not be interpreted as task-level degradation.

## 13. Rollout success

Not run because the preregistered P6 stop condition requires halting the invalid comparison. Rollout
count is exactly 0 episodes; there is no M0/M1/M2 rollout-success comparison.

## 14. Seed-level results

Training seed was the intended independent replicate. All nine primary runs completed with zero
recorded non-finite events and matching data-order hashes within seed. The descriptive paired final
NRMSE differences (candidate minus baseline) were:

- M1−M0: `+0.00602`, `+0.00684`, `+0.01407`.
- M2−M0: `-0.00022`, `+0.00685`, `+0.00411`.
- M2−M1: `-0.00624`, `+0.00001`, `-0.00996`.

These values are not promoted to scientific effects after the reproducibility failure.

## 15. Statistical and effect-size analysis

Descriptive paired effect sizes `dz` for final NRMSE were M1−M0 `+2.03`, M2−M0 `+1.00`, and M2−M1
`-1.07` (positive candidate-minus-baseline is worse). With only three seeds these would already require
cautious magnitude/direction interpretation; under P6 they are invalid for inference. Validation
states were never treated as independent training replicates, and no p-value claim was made.

## 16. Failure cases and reproducibility audit

The first audit found final M1/M2 maximum parameter differences of `0.02718`, `0.03040`, and `0.02802`
for matched seeds, with 419 of 500 tensors differing. Those pairs had run on opposite physical GPUs.
Before examining any recovery result, the project fixed a symmetric plan to rerun all M1 seeds on the
same GPUs as M0/M2 and continue only if the first update-1,000 checkpoint passed.

The seed-520001 same-device checkpoint still differed from M2 by `0.02034` at update 1,000 (419/500
tensors), despite identical source masking, data order, random streams, hardware device, and shared
input loss. Seed 520002 was interrupted after update 900 once hard failure was established; seed
520003 was not rerun. The precise low-level cause is unresolved. It may involve execution-state or
CUDA numerical nondeterminism, but the evidence does not identify a unique cause. This is exactly the
kind of unexplained divergence covered by the frozen stop rule.

A separate duplicate M1/520002 start occurred earlier after its live process lost the interactive
session; the duplicate exited before update 1 because the original still occupied the GPU. The
original process completed all 5,000 updates, its early artifacts were restored, and the duplicate
contributed no scientific result.

## 17. Interpretation of M0 vs M1 vs M2

No valid Phase-1 method ordering can be assigned. The descriptive numbers do not meet the frozen
practical thresholds, but the correct verdict is P6 rather than P5 because substrate validity failed
first. In particular, neither `M1 > M0` nor `M2 > M1` reproduced as a valid causal result. Gate 0A
continues to show a measurable nuisance pathway, but Phase 1 cannot determine whether suppressing it
helps downstream post-training.

## 18. Limitations

- Independent bf16 CUDA training did not meet the preregistered deterministic tolerance.
- The same-device audit ruled out physical device assignment as a sufficient explanation, but did
  not isolate the low-level nondeterministic operation.
- Descriptive validation differences are small relative to the invalidity trigger.
- No trained-checkpoint mechanism linkage or rollout success is available due to the required stop.
- The study covers one SmolVLA base checkpoint, one 19.66% LIBERO split, and one optimization budget.

## 19. GPU usage

Exactly logged usage is `17.4612` GPU-hours: baseline `0.0937`, M0 `5.6719`, M1 `5.9523`, M2
`5.7343`, and timed implementation diagnostics `0.0090`. The interrupted same-device audit adds an
estimated `0.8330` device-hours reconstructed from process/artifact timestamps, for `18.2942`
accounted GPU-hours. The failed pre-update duplicate start was not instrumented and is excluded;
therefore an exact all-process total cannot honestly be claimed. Leakage and rollout used 0 GPU-hours.

## 20. Scientifically justified next step

Stop Phase 1. A future owner-authorized protocol should first establish an independent-run
reproducibility gate under a strictly deterministic CUDA configuration, then rerun all methods from
scratch symmetrically. No method claim or additional flow variant should be attempted before that
gate passes. This recommendation was not implemented.

## 21. Explicitly not tested

No auxiliary/Jacobian/representation loss, normalization trick, coordinate adapter, action weighting,
RL, GRPO, preference optimization, reward/critic/world model, architecture or model-size change,
learned/soft projection, separate action head, learned covariance, whitening, manifold flow, M3,
cross-model validation, large-scale training, or paper-writing experiment was tested. Trained-checkpoint
leakage, rollout success, and cross-dimensional sensitivity were also not run after the P6 stop.
