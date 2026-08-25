# FlowSpec-VLA Experiment State

Last updated: 2026-08-25

## PLANNED

- Phase 1 is active under the frozen M0/M1/M2 protocol. The next required gate
  is the 250-update official M0 baseline reproduction. M1/M2 primary training
  is prohibited unless that gate passes.

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

## DECISION

- Gate-0 verdict remains `STRONG GO`, due to independent Gate 0A A1 evidence.
- Phase-1 verdict: pending the frozen baseline gate and matched comparison.
- Phase-1 verdict: pending completion of the nine matched primary runs,
  mechanism audit, and rollouts. The baseline gate authorizes continuation.
