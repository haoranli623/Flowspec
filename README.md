# FlowSpec-VLA

Action-interface consistency audit and controlled post-training study for the
official Hugging Face LeRobot SmolVLA implementation on LIBERO.

This repository is an independent research project. It does not import
experimental conclusions, configurations, or results from ContactShift-VLA or
ForkTune-VLA. Persistent models, datasets, caches, environments, and artifacts
live under `/mnt/NAS/data/hl5757`.

Gate 0 is complete. The frozen overall verdict is **STRONG GO**, driven by a
stable A1 padded-source coupling result; coordinate sensitivity was B2 after a
second-seed replicate. See `GATE0_REPORT.md` for the bounded interpretation.

Phase 1 is active under `PHASE1_PROTOCOL.md`. It compares exactly M0 official
flow matching, M1 padded-source masking, and M2 padded-source masking plus
per-Euler-step subspace projection. The fixed study uses the official base
checkpoint, all 40 LIBERO tasks, 53,762 training frames, 400 held-out validation
states, 5,000 updates, and three matched seeds per method. No additional method
is authorized.

Key commits:

- initial skeleton: `382b43b`
- protocol freeze: `337c95305698e527ef09db6f53040fb377a9d6a5`
- Gate-0 final: `d17670f51169478a8dcd3b48ac7354baa6deac13`
- Phase-1 implementation: `19cb03d7de4eb68a4cb10e2dbe82797765f9f560`
- Phase-1 protocol freeze: `beb292024fcb37d321338474249d03598cfa5e90`
