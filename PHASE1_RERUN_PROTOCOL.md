# FlowSpec-VLA Phase-1 Clean Rerun Protocol

**Status:** FROZEN BEFORE RERUN RESULTS  
**Project:** FlowSpec-VLA — Action-Interface Consistent Post-Training for Flow-Matching VLAs  
**Audited starting repository revision:** `67baa3ad689a158dd34fdd1b751387bf338671c2`  
**Original Phase-1 protocol:** `PHASE1_PROTOCOL.md` at protocol-freeze commit `beb292024fcb37d321338474249d03598cfa5e90`  
**Resume-equivalence protocol:** `RESUME_GATE_PROTOCOL.md` at `a2a37dd9a5e6a8549fea54d165727b9864650acf`  
**Rerun protocol-freeze commit:** recorded in `EXPERIMENT_STATE.md` and `phase1_rerun/protocol_snapshot.json` immediately after this document is committed  

> This rerun preserves the original scientific Phase-1 protocol and changes only the execution/reproducibility infrastructure required by the R-A resume-equivalence gate.

## 1. Scientific status and quarantine

Gate 0 is valid and produced a strong mechanistic signal: the executable dimension is 7, the SmolVLA internal action dimension is 32, and changing only the 25 padded source coordinates changed executable predictions (`R_leak = 0.05729682371020317`; padded RMS `0.010470231994986534`; valid RMS `0.043740563094615936`; exact-repeat error 0).

The first Phase-1 comparison is scientifically invalid because its interrupted/resumed training was not numerically equivalent to uninterrupted training. Its M0/M1/M2 performance numbers are quarantined historical artifacts. They were not used to choose any field in this protocol and will not be used for stopping, checkpoint selection, interpretation, or claims.

The R-A gate subsequently demonstrated exact complete-state resume equivalence under deterministic CUDA execution, including exact batches, source randomness, losses, gradients, model parameters, all 1,467 optimizer-state tensors, scheduler state, and training position. Its short 4-worker check also passed exactly.

## 2. Frozen question and methods

Question: **Does removing stochastic flow dynamics from non-executable padded action dimensions improve flow-VLA post-training?**

Exactly three methods are permitted:

- **M0 — Standard FM:** native 32D Gaussian source; all dimensions evolve; official loss and output slicing.
- **M1 — Source-Masked FM:** set source coordinates 7:32 to zero; coordinates 0:7 retain the identical native draw; after initialization all coordinates may evolve; no projection or auxiliary loss.
- **M2 — Full-Subspace FM:** M1 source mask plus zero padded target/state and projection of coordinates 7:32 to zero after every inference Euler update; padded velocity cannot accumulate; valid coordinates, architecture, loss, and output slicing are unchanged.

No fourth method, auxiliary objective, normalization change, architecture change, RL method, data-fraction search, task search, seed search, or method-specific tuning is authorized.

## 3. Exact substrate and revisions

- Repository audited base: `67baa3ad689a158dd34fdd1b751387bf338671c2`.
- The exact execution commit for every run will be recorded before launch in `protocol_snapshot.json` and in each run record. No source edit is allowed after launch without invalidating and restarting affected runs from the same base state.
- LeRobot source commit: `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Base model: `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, local path `/mnt/NAS/data/hl5757/models/flowspec-vla/smolvla_base-c83c3163`.
- Dataset: `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`, local path `/mnt/NAS/data/hl5757/datasets/flowspec-vla/lerobot-libero-a1aaacb7`.
- Existing environment: `/mnt/NAS/data/hl5757/conda_envs/flowspec-vla`.
- New artifacts only: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun`.
- New trained models only: `/mnt/NAS/data/hl5757/models/flowspec-vla/phase1_rerun`.
- Old `phase1/` artifacts and models are read-only quarantine inputs and must not be overwritten.

## 4. Data and split freeze

The original deterministic task-wise split is reused byte-for-byte from `phase1/split_manifest.json` (SHA-256 `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0`).

- Selection seed: `510001`.
- Ranking: ascending SHA-256 of `"{seed}:{task_index}:{episode_index}"` within each task.
- All 40 LIBERO tasks.
- Five held-out episodes per task, 200 total.
- Eight training episodes per task, 320 total.
- Training frames: 53,762 (19.659554239116522% of all frames).
- Validation: two anchors (fractions 0.25 and 0.60) from each held-out episode, 400 states total.
- Normalization statistics are the frozen training-only statistics in the manifest.
- M0/M1/M2 use exactly the same target tensors, normalization, unnormalization, sample order, and validation states. No train episode may enter validation.

## 5. Model/action freeze

- Executable LIBERO action dimension: 7.
- Internal action dimension: 32.
- Action chunk length: 50.
- Euler flow steps: 10.
- Executable action semantics and component metrics are translation dimensions 0:3, rotation 3:6, and gripper dimension 6.
- All model parameters are trainable, identically for all methods.
- All runs start independently from the same frozen base checkpoint. No task-specific saturated checkpoint is used.

## 6. Optimization freeze

Each primary run uses:

- Seeds: `520001`, `520002`, `520003` for every method, paired by seed.
- Updates: 5,000 exactly.
- Microbatch: 16; gradient accumulation: 2; effective batch: 32.
- Examples seen: 160,000 per run.
- Optimizer: AdamW; LR `1e-4`; betas `(0.9, 0.95)`; epsilon `1e-8`; weight decay `1e-10`.
- Gradient clipping: global norm 10.0.
- Schedule: 200-update linear warmup, then cosine decay through update 5,000 to `2.5e-6`.
- Precision: bf16 autocast; no GradScaler.
- Loader: 4 workers, persistent workers enabled, prefetch factor 2, explicit deterministic order and sample offset.
- Checkpoints: updates 1,000, 2,500, 5,000.
- Validation: updates 0, 250, 500, 1,000, 2,000, 3,000, 4,000, 5,000.
- No early stopping, extension, checkpoint cherry-picking, or method-specific adjustment.

## 7. Execution/reproducibility changes from original Phase 1

These are the complete allowed differences. They do not alter the scientific intervention, data, optimization, or metrics.

1. Each run is one process bound to one explicit GPU. Two independent runs may execute concurrently, one per GPU.
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialization.
3. `torch.use_deterministic_algorithms(True)` is enabled.
4. cuDNN deterministic mode is enabled and benchmark mode disabled.
5. CUDA matmul and cuDNN TF32 are disabled.
6. Python, NumPy, torch CPU, all CUDA, dedicated flow, loader, and any dataset/sampler generator states are explicitly seeded and checkpointed.
7. Training uses a fixed, materialized sample order with an explicit sample offset; resumption begins at the saved next-example offset rather than reconstructing position indirectly.
8. The 4-worker/persistent/prefetch-2 loader profile that passed the R-A short gate replaces the old incomplete loader-resume semantics.
9. Checkpoints are complete training-state checkpoints containing model, optimizer, pure-function scheduler record, AMP applicability, global update, optimizer step, examples seen, epoch, microbatch/accumulation position, sampler order/hash/offset, every RNG/generator state, method, config, and provenance.
10. A resumed run performs an explicit restoration audit before its next update and records the first resumed batches/random draws/losses/parameters for comparison.

Everything not listed above remains as in the original frozen Phase-1 protocol.

## 8. Deterministic smoke gate

Before primary training, run M0, M1, and M2 with the same non-primary seed `519001` for exactly 250 updates. This is an engineering gate, not a scientific comparison; relative smoke performance must not be interpreted.

- Same frozen train split and matched sample order.
- Same optimization configuration and baseline validation schedule `[0, 50, 100, 250]` on the frozen 64-state pilot subset.
- Save a complete checkpoint at update 125.
- The uninterrupted reference records exact transition evidence for updates 126–130 and continues through update 250.
- In a fresh process, restore update 125 and rerun updates 126–130.
- Require identical batch IDs/actions, processed tensors, flow timesteps, valid source noise, loss, gradients, scheduler, optimizer, RNG state, and post-update training position.
- Require final model maximum absolute parameter difference `< 1e-6`; exact zero is preferred.
- Require finite loss/gradients, update count 250, 8,000 examples seen, aligned initial model/valid parameters, M0 native padded noise, and exact-zero padded source for M1/M2.
- Because official training has no ODE rollout, M1 and M2 training objectives must be exactly identical under matched seed/order; their differing inference projection remains verified separately.

Any method failure stops Phase 1 with **PHASE1-RERUN NO-GO — METHOD-SPECIFIC REPRODUCIBILITY FAILURE**. No primary run launches.

## 9. Method-integrity gate

The frozen implementation verification suite (historical artifact SHA-256 `1a610712d186da1debf0344f9f4c520ca0e1b89b94f76053dcd7b29cc109cd53`) is rerun under deterministic settings and written anew to `method_integrity_summary.json`.

It must show M0 exact official equivalence; M1 exact source masking with valid draws preserved and later padded evolution allowed; M2 exact initialization/projection and valid-coordinate/gradient preservation; shared executable targets, normalization, unnormalization, output slicing, and target round trip. Any failure stops before primary training.

## 10. Primary run matrix and resume spot checks

After both gates pass, run exactly nine primary trainings: three methods by the three frozen seeds, 5,000 updates each.

For seed `520001` of each method, use the natural update-1,000 checkpoint for a fresh-process reproducibility spot check over updates 1,001–1,005. The uninterrupted run records the corresponding reference transition evidence. The same logical and numerical criteria as the smoke resume check apply, including max parameter difference `< 1e-6`. This check does not duplicate or replace the full run.

If a production interruption occurs, resume only from the latest complete checkpoint, audit restoration, and compare the first available resumed transitions. A failed resume or spot check invalidates the comparison and triggers P6 without scientific interpretation.

## 11. Frozen primary metrics

Validation generation uses seeds `530001` (generation noise), `530002` (flow noise), `530003` (flow time), and bootstrap seed `530004` with 10,000 state-level resamples.

### P1 — Physical executable-action error

The primary metric is whole-action physical-space NRMSE on the frozen validation states:

`sqrt(mean((prediction_physical - target_physical)^2)) / sqrt(mean(target_physical^2))`,

aggregated across the executable 7D chunk coordinates exactly as in the original protocol. Translation, rotation, gripper, per-horizon, and chunk temporal errors are reported as preregistered secondary decompositions.

### P2 — Learning/sample efficiency

Report the final update-5,000 NRMSE and normalized trapezoidal AULC over the frozen validation schedule. Updates-to-threshold is not primary because no numeric threshold was preregistered in the original protocol.

### P3 — Mechanism suppression

Repeat Gate 0A on every final checkpoint with the same 64 frozen states, 16 padded and 16 valid interventions per state, 10 Euler steps, and frozen noise construction. Report padded RMS, valid RMS, and `R_leak`, bootstrapping over states where used. Record padded-state magnitude at every flow step.

Training seeds—not validation frames or minibatches—are the independent experimental replicates. Report every seed, mean, sample SD, matched-seed differences, relative effects, direction counts, and effect sizes. With n=3, magnitude and consistency take precedence over p-values.

## 12. Frozen quantitative rules

A method materially outperforms another only when all applicable original criteria hold:

- mean final NRMSE is at least 5% lower;
- the direction favors it in at least 2 of 3 paired seeds; and
- normalized AULC is at least 3% lower.

M1 and M2 are practically equivalent when the magnitude of their mean final relative difference is at most 2% and neither materially outperforms the other.

Mechanism suppression is strong when padded RMS `<= 1e-4` and `R_leak <= 1e-4`. It is partial when both are reduced by at least 50% relative to matched M0 in the same direction for at least 2 seeds.

Rollout evidence is supportive when absolute success improves by at least 5 percentage points in the same direction for at least 2 of 3 seeds. It cannot overturn the primary physical-space decision.

Baseline/smoke engineering sanity requires final/initial NRMSE `<= 1.05`, final/initial flow MSE `<= 0.95`, last/first train loss `<= 1.00`, finite numerics, and peak allocated GPU memory `<= 23 GiB`.

Verdict order is the original order: P6 substrate invalid; P4 continued-coupling effect; P3 source-only effect; P1 strong go; P2 mechanism confirmed/practical benefit weak; otherwise P5 leakage not practically harmful.

## 13. Frozen rollout evaluation

If all primary runs and reproducibility checks are valid, evaluate only update-5,000 checkpoints on all four frozen suites: `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`; 10 tasks per suite; 5 episodes per task; seed base `540001`. This is 200 episodes per checkpoint and 1,800 episodes total for nine checkpoints. No best-checkpoint selection, task change, or adaptive episode extension is allowed.

## 14. Compute and storage discipline

- Approximate all-inclusive ceiling remains 72 device-hours; stop and document before exceeding it.
- Track smoke, every primary method/seed, spot checks, leakage, and rollout separately.
- Prefer two concurrent isolated single-GPU runs; scheduling order cannot depend on observed outcomes.
- Large logs/checkpoints stay outside Git under the new artifact/model roots.
- Required summaries and analytically useful plots are produced only from the clean rerun.

## 15. Stop conditions

Stop without scientific interpretation if any smoke resume fails, any method differs beyond the intended intervention, source/target/split/order/initialization/optimizer matching fails, data contamination is found, complete checkpoint contents are missing, any primary run cannot be reliably resumed, any full-run spot check has max parameter difference `>= 1e-6`, or reproducibility breaks.

Ordinary implementation bugs may be fixed only before scientific results are inspected, followed by regenerating all affected smoke evidence. Any code change after primary launch invalidates and restarts every affected primary run.

## 16. Required outputs and final boundary

Repository outputs: `PHASE1_RERUN_REPORT.md`, updated `EXPERIMENT_STATE.md`, and compact README status. Machine-readable outputs under the new artifact root: `protocol_snapshot.json`, `smoke_gate_summary.json`, `method_integrity_summary.json`, `training_summary.json`, `validation_summary.json`, `seed_pairing_summary.json`, `reproducibility_spotcheck.json`, `leakage_reaudit_summary.json`, `rollout_summary.json` if reached, `gpu_usage.json`, and `artifact_manifest.json` with SHA-256 hashes.

Plots, if supported by reached results: validation metric versus updates with seed uncertainty; final per-seed metric; paired seed differences; pre/post leakage; padded-state trajectories; rollout success.

The project stops after the clean Phase-1 rerun verdict. No new method, cross-model study, benchmark expansion, scaling study, or paper experiment is authorized.
