# FlowSpec-VLA

Action-interface consistent post-training for flow-matching VLAs.

The deterministic clean Phase-1 rerun is complete.

**Final verdict: P2 — mechanism confirmed, practical benefit weak.**

M2 suppresses padded-dimension leakage to numerical zero, but neither M1 nor
M2 improves matched-budget physical validation or frozen LIBERO rollout success
over M0. The project makes no VLA-performance improvement claim and stops at
the Phase-1 boundary.

Key documents:

- [Clean rerun protocol](PHASE1_RERUN_PROTOCOL.md)
- [Clean rerun report](PHASE1_RERUN_REPORT.md)
- [Experiment state](EXPERIMENT_STATE.md)
- [Gate-0 protocol](GATE0_PROTOCOL.md)
- [Gate-0 report](GATE0_REPORT.md)

Artifacts:

`/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun/`
