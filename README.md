# FlowSpec-VLA

Action-interface consistency audit for the official Hugging Face LeRobot
SmolVLA implementation on LIBERO.

This repository is an independent research project. It does not import
experimental conclusions, configurations, or results from ContactShift-VLA or
ForkTune-VLA. Persistent models, datasets, caches, environments, and artifacts
live under `/mnt/NAS/data/hl5757`.

The authorized scope ends after Phase -1, Gate 0A, Gate 0B, and the Gate-0
report. Full method development and downstream post-training are out of scope.

Gate 0 is complete. The frozen overall verdict is **STRONG GO**, driven by a
stable A1 padded-source coupling result; coordinate sensitivity was B2 after a
second-seed replicate. See `GATE0_REPORT.md` for the bounded interpretation.

Key commits:

- initial skeleton: `382b43b`
- protocol freeze: `337c95305698e527ef09db6f53040fb377a9d6a5`
