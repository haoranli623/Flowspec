# FlowSpec-VLA Deterministic/Resume Forensic Protocol

Status: **FROZEN BEFORE NEW FORENSIC GPU RESULTS**  
Date: 2026-08-25

## 1. Bounded question

For official M0 SmolVLA post-training, find the first failed arrow in:

`saved state → next batch → explicit randomness → forward → loss → gradients → optimizer step → parameters`.

This audit does not revise the Phase-1 P6 verdict, rerun Phase 1, run M1/M2, evaluate rollouts, or
change the scientific model/data/objective. It stops as soon as a causal source is convincingly
isolated or at two additional device-hours, whichever occurs first.

## 2. Frozen substrate

| Item | Value |
|---|---|
| Code revision audited | `1784b512b036807bf85c684d35f079b2cf717e78` |
| Starting weights | official `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Starting weight SHA-256 | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| LeRobot source | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| Dataset | `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Split manifest | Phase-1 frozen manifest SHA-256 `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0` |
| Method | M0 only, unmodified full-dimensional flow source |
| GPU | physical GPU 0 for all matched executions |
| Topology | one Python process, one GPU, no DDP, no compilation |
| Precision | Phase-1 bf16 CUDA autocast; no GradScaler |
| TF32/cuDNN | TF32 disabled; cuDNN benchmark false and deterministic true, matching Phase 1 |
| Seed | `560001` for Python, NumPy, torch CPU/CUDA; dedicated streams derived below |
| Optimizer | Phase-1 AdamW and manual cosine LR, unchanged |
| Batch | microbatch 16, accumulation 2, effective batch 32 |

## 3. Data and stochastic configuration

Fresh and resume-process tests use the Phase-1 fixed training-order construction with seed 560001,
four workers, `persistent_workers=True`, prefetch factor 2, pinned memory, and a loader generator
seeded with `seed+102`. Flow noise/timestep use a dedicated CPU `torch.Generator` seeded `seed+103`.
No image augmentation is configured in `load_dataset`; resize/padding, normalization, camera rename,
and tokenization are deterministic candidates to be verified by tensor hashes.

The audit records Python, NumPy, torch CPU, every visible CUDA RNG, loader-generator, and flow-generator
states. The fixed sampler has no mutable permutation state; its resume state is the explicit offset
into the precomputed order. Worker seeds and prefetch behavior are recorded separately. Model-internal
stochasticity is audited through CUDA RNG states before/after forward/backward and module/config
inspection.

## 4. Stable capture and comparison

At each examined update, record deterministic CPU-byte SHA-256, shape, and dtype where applicable:

1. sample indices, episode/frame/task IDs, and raw actions;
2. processed images, state, language tokens/attention, normalized action, and padding mask;
3. explicit Gaussian flow noise and timestep for both accumulation microbatches;
4. Python/NumPy/torch/CUDA/dedicated-generator states;
5. representative early, action-output, and late parameter tensors before forward;
6. captured action projection output, per-coordinate flow loss, and reduced scalar loss;
7. representative gradients plus global gradient norm and per-tensor difference summaries;
8. representative parameters and Adam moments immediately after `optimizer.step()`;
9. assigned LR, update counter, accumulation boundary, and explicit next data offset.

Cross-run tensor comparisons report exact hash equality plus maximum absolute, mean absolute, and L2
difference. Full model/checkpoint hashes are recorded at boundaries. No string/object representation
is accepted as tensor evidence.

## 5. Test sequence

### F1 — Fresh determinism

Run Fresh-A and Fresh-B sequentially as separate interpreters on GPU 0, each for five M0 updates from
the official base. They use identical seeds, process topology, loader settings, explicit order, and
flow generator. Compare every captured stage at every update.

- If they diverge before checkpointing, classify F1-A and proceed directly to the earliest subsystem.
- If all five updates agree within `1e-6` parameters, classify F1-B and proceed to F2.

### F2 — Uninterrupted versus resumed

Only if F1 does not already isolate the cause: train M0 to update 10, save immediately after
`optimizer.step()` at a clean accumulation boundary, then continue uninterrupted through update 20.
Restore the update-10 state in a new process and execute updates 11–20. The forensic checkpoint
contains model/buffers/mode, optimizer moments and counters, LR/update, Python/NumPy/torch/CUDA RNG,
flow and loader generators, explicit sample offset, and accumulation position. Compare all ten
continuation updates. Separately document that historical Phase-1 model-only checkpoints omit these
states and cannot satisfy exact resume semantics.

### F3 — Same-process one-step rollback

At the earliest useful boundary, cache one exact raw batch and explicit flow tensors, capture full
model/optimizer/RNG/mode state, execute one update, restore in the same process, and replay the update.
Compare forward, loss, gradients, Adam state, and parameters stage by stage. This test removes process,
loader, and checkpoint-serialization differences.

### Deterministic-kernel diagnostic

Only when inputs, processed tensors, explicit randomness, RNG, and pre-forward model state match but
forward/backward diverges: run at most two additional one-update diagnostics. First enable
`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. If needed, force the
PyTorch math SDPA backend while disabling flash and memory-efficient SDPA. These are diagnostic paths,
not production changes. Unsupported operations, warnings, speed, and remaining differences are logged.

## 6. Checkpoint/state audit

The audit explicitly inventories model weights/buffers/mode; optimizer moments/steps/parameter groups;
manual scheduler update/LR; absence of GradScaler state; accumulation boundary and gradients; all RNG
states; fixed-order offset; loader generator/workers/prefetch; processor state; and dataset randomness.
Checkpoint labels are post-update labels: update `t` means the optimizer step for `t` is complete, LR
for `t` has been applied, gradients were cleared at the beginning of that update, and the next batch
is at offset `t × 32`.

## 7. Stop and success rules

- Hard maximum additional compute: 2.0 device-hours.
- No training execution may exceed 20 updates under this protocol (well below the authorized 500).
- Stop immediately once the first causal divergence is isolated and minimally verified.
- Do not implement speculative bundles. A fix requires evidence, one minimal change, and bounded V1
  replay plus V2 resume and V3 fresh verification only when relevant.
- Strict parameter reference is maximum absolute difference `<1e-6`; exact equality is reported when
  achieved.
- If localization remains incomplete, report the last identical and first divergent stages and stop.

Large raw captures and checkpoints live under
`/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic/` and remain outside git.
