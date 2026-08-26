# FlowSpec-VLA Experiment State

Last updated: 2026-08-25

## PLANNED

- Phase 1 is closed. No further method, rollout, leakage audit, or training is
  authorized under the current protocol.
- The bounded M0 forensic audit is closed. Any production determinism/resume
  repair or Phase-1 rerun requires new authorization.

## MEASURED

- The new project and persistent NAS paths were created.
- Official LeRobot source is frozen at commit
  `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Official `lerobot/smolvla_libero` checkpoint is frozen at revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Official `lerobot/libero` payload is frozen at revision
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- Phase -1 source audit is complete. All Gate-0 results were run after protocol
  freeze, and the frozen scientific settings were not altered.
- Gate 0A: 64 states, four tasks, 16 pad draws and 16 valid draws per state;
  `A1_STRONG_SIGNAL` with median standardized padded RMS `0.0104702` and
  variance ratio `0.0572968`.
- Gate 0B: 128 training anchors, 32 validation anchors, three transforms, two
  seeds, 64 updates per run (384 total); `B2_WEAK_SIGNAL` after the strongest
  primary-seed effect failed to replicate.
- Aggregate measured experimental GPU usage: `0.0725613 GPU-hours`.
- Phase-1 implementation revision:
  `19cb03d7de4eb68a4cb10e2dbe82797765f9f560`.
- Phase-1 protocol-freeze commit:
  `beb292024fcb37d321338474249d03598cfa5e90`.
- Official `lerobot/smolvla_base` is frozen at revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- The Phase-1 split was frozen before training: all 40 tasks, 320 training
  episodes, 53,762 training frames (19.659554%), 200 held-out episodes, and
  400 validation states. Split-manifest SHA-256:
  `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0`.
- Pre-training M0/M1/M2 implementation and physical-equivalence checks passed.
- The frozen 250-update M0 reproduction gate passed: first/last 50-update mean
  loss `2.1890`/`0.7607`, validation physical NRMSE `1.3045` to `0.8862`,
  fixed-noise flow MSE `2.3713` to `0.6807`, no non-finite values, and peak
  allocated memory `18.21 GiB`.
- Nine primary runs completed: M0/M1/M2 × seeds 520001/520002/520003, each at
  5,000 updates and 160,000 examples seen.
- Descriptive final physical NRMSE means were M0 `0.71295`, M1 `0.72192`, and
  M2 `0.71653`; these values are invalid for causal comparison.
- Final matched M1/M2 checkpoints differed by maximum absolute parameter values
  `0.02718`, `0.03040`, and `0.02802`, above the frozen `1e-6` tolerance.
- Same-device recovery for seed 520001 still differed by `0.02034` at update
  1,000; reproducibility could not be restored.
- Trained-checkpoint leakage and rollout evaluation were not run due to the
  preregistered P6 stop condition.
- M0 forensic protocol-freeze commit:
  `e1cc6d1eb8011dffbf9038bc71b08f9160780570`.
- Two five-update fresh M0 runs matched exactly through inputs, preprocessing,
  all audited RNG states, explicit flow tensors, forward outputs, and loss, but
  first diverged in update-1 backward gradients. The final five-update
  checkpoint maximum parameter difference was `7.62939453125e-6`.
- Default same-process one-step rollback reproduced the backward boundary: 180
  of 489 gradient tensors differed, with maximum absolute difference
  `7.32421875e-4`.
- With deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the
  same-process replay and two independent-process one-update runs were bitwise
  exact; independent final parameter maximum difference was `0`.
- Historical Phase-1 checkpoints are model/processor-only and omit optimizer,
  update/LR, RNG, dedicated generators, sampler offset, model mode, and
  accumulation state; exact short resume remains untested.
- The forensic audit used `0.0824591` accounted device-hours, including a
  conservative 56-second failed-instrumentation allowance.

## INTERPRETATION

- Current official code has `d_valid=7`, `D=32`; padded coordinates enter the
  shared flow network during training and inference and are discarded only at
  the supervised-loss/output boundary. This supports testability, not a result.
- Current dataset task metadata lost language labels; labels are recoverable
  exactly from parent revision `1595a93b...` without changing payload data.
- Padded source coordinates create stable executable-coordinate variation under
  the frozen official checkpoint.
- Coordinate convention changes pretrained behavior and early optimization, but
  the largest short-pilot difference is seed-sensitive and the pilot degrades
  all conditions.
- Official SmolVLA training evaluates one flow time and has no ODE rollout.
  Consequently M1 and M2 have identical training objectives; their isolated
  additional difference is M2 projection during sampling/evaluation.
- The shared-input M1/M2 objective is exactly equal, but independent training
  execution is not reproducible to the frozen checkpoint tolerance. Physical
  GPU assignment alone does not explain the divergence.
- The first reproducible instability occurs before checkpointing: official
  default CUDA execution produces nondeterministic backward gradients despite
  exact data, RNG, forward outputs, and loss. Deterministic CUDA/cuBLAS controls
  eliminate it in bounded replay and independent-process tests.
- Checkpoint-state omission is a separate confirmed resume limitation, not the
  first cause of the fresh-run divergence.

## DECISION

- Gate-0 verdict remains `STRONG GO`, due to independent Gate 0A A1 evidence.
- Phase-1 verdict: **P6. NO-GO — TRAINING / SUBSTRATE INVALID**.
- Phase 1 stops here. Descriptive primary metrics cannot support an M0/M1/M2
  method ordering, and no further variants are authorized.
- Forensic verdict: **F-C — HARDWARE/NUMERICAL NONDETERMINISM IDENTIFIED**.
- Phase 1 is not technically safe to rerun until an opt-in production
  deterministic configuration and a full-state M0 resume gate are implemented
  and verified under new authorization.
