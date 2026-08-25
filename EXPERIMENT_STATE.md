# FlowSpec-VLA Experiment State

Last updated: 2026-08-25

## PLANNED

- Commit the completed Phase -1 audit and frozen Gate-0 protocol.
- Materialize the deterministic state/anchor manifests and validate tensor paths.
- Run Gate 0A, then the matched three-condition Gate 0B pilot.

## MEASURED

- The new project and persistent NAS paths were created.
- Official LeRobot source is frozen at commit
  `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Official `lerobot/smolvla_libero` checkpoint is frozen at revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Official `lerobot/libero` payload is frozen at revision
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- Phase -1 source audit is complete. No Gate-0 mechanism result has been run or
  inspected.

## INTERPRETATION

- Current official code has `d_valid=7`, `D=32`; padded coordinates enter the
  shared flow network during training and inference and are discarded only at
  the supervised-loss/output boundary. This supports testability, not a result.
- Current dataset task metadata lost language labels; labels are recoverable
  exactly from parent revision `1595a93b...` without changing payload data.

## DECISION

- The substrate is valid for the preregistered tests.
- Do not inspect final Gate results until `GATE0_PROTOCOL.md` is committed and
  its hash is recorded.
