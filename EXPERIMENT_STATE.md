# FlowSpec-VLA Experiment State

Last updated: 2026-08-25

## PLANNED

- None within authorized scope. Stop after the completed Gate-0 report.

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

## DECISION

- Overall verdict: `STRONG GO`, due to independent Gate 0A A1 evidence.
- Stop. Do not implement a solution or full downstream post-training under this
  project authorization.
