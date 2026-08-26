# FlowSpec-VLA Deterministic M0 Resume-Gate Protocol

Status: **FROZEN BEFORE NEW RESUME-GATE GPU RESULTS**

Date: 2026-08-25

## 1. Bounded engineering question

Can a newly created, complete M0 training checkpoint reproduce an uninterrupted deterministic
continuation after a fresh-process restore?

The primary chain is:

`official M0 base → deterministic updates 0–50 → complete post-update checkpoint →`
`uninterrupted updates 51–100 versus fresh-process resumed updates 51–100`.

This is an infrastructure gate only. It does not rerun Phase 1, test M1/M2, evaluate rollouts,
change the flow objective, or reinterpret the prior P6/F-C results.

## 2. Frozen substrate

| Item | Value |
|---|---|
| Starting repository revision | `78553da2cf2799bfd84a8a354d025826f9053e2b` |
| Method | M0 only, official full-dimensional source |
| Base checkpoint | `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Base weight SHA-256 | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| LeRobot source | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| Dataset | `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Split manifest SHA-256 | `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0` |
| GPU | physical GPU 0, RTX 3090 |
| Topology | one process and one GPU per run; no DDP or compilation |
| Seed | `570001` |
| Precision | CUDA bf16 autocast; no GradScaler |
| Microbatch / accumulation | 16 / 2; effective batch 32 |
| Optimizer | Phase-1 AdamW, unchanged |
| LR schedule | Phase-1 manual warmup/cosine function, unchanged |
| Primary DataLoader | `num_workers=0`, pinned memory, fixed sampler, no prefetch workers |

## 3. Deterministic execution configuration

Every GPU process must be launched with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA
initialization and must apply:

- `torch.use_deterministic_algorithms(True)`;
- `torch.backends.cudnn.deterministic=True`;
- `torch.backends.cudnn.benchmark=False`;
- `torch.backends.cuda.matmul.allow_tf32=False`;
- `torch.backends.cudnn.allow_tf32=False`.

These are exactly the combined controls that produced zero parameter difference in the prior
forensic audit. No additional numerical or scientific change is permitted.

## 4. Data order and loader semantics

The complete 100-update order is materialized once from the Phase-1 training indices using NumPy
`default_rng(seed+101)`. Run A consumes this order from offset zero. Run B reconstructs the same full
order, verifies its SHA-256, and constructs an offset sampler starting at exactly
`50 × 16 × 2 = 1600` examples.

The primary loader uses `num_workers=0` to remove worker/prefetch ambiguity. A dedicated loader
generator starts at `seed+102`. PyTorch iterator construction consumes this generator even with zero
workers; therefore Run B constructs its iterator and then restores the saved loader-generator state
at the checkpoint boundary. Global and all other dedicated RNG states are likewise restored only
after model, optimizer, dataset, processor, loader, and iterator construction.

No stochastic dataset augmentation is configured. Sample IDs, raw actions, and all emitted
processed tensors are nevertheless hashed for direct verification.

## 5. Flow and seed policy

Python, NumPy, torch CPU, and all visible CUDA RNG streams start from seed 570001. The materialized
order uses `seed+101`, the loader generator `seed+102`, and flow Gaussian noise/timestep generator
`seed+103`. The flow generator state is stored and restored directly. CUDA RNG is recorded even
though the audited SmolVLA path has no active dropout.

## 6. Complete checkpoint inventory

The new versioned checkpoint must contain:

- complete model `state_dict`, including parameters and buffers;
- model train/eval mode;
- complete AdamW `state_dict`, including first/second moments, step tensors, and parameter groups;
- manual scheduler name/config, last completed update, current LR, and optimizer-step count;
- an explicit record that GradScaler/AMP scaler state is not applicable;
- global update, examples seen, epoch, batch/microbatch position, accumulation position, and explicit
  sampler/example offset;
- Python, NumPy, torch CPU, and every visible CUDA RNG state;
- loader and flow `torch.Generator` states;
- seed, full-order SHA-256, intended total order length, and loader configuration;
- frozen model/dataset/source/config revisions and deterministic execution settings.

Processor configuration and normalization are reproducibly reconstructed from the frozen base,
recipe, and split manifest; their hashes/revisions are checkpoint metadata. No partially accumulated
gradients exist because saving occurs at accumulation position zero.

## 7. Exact save boundary and restore semantics

Checkpoint `update=50` means:

1. both microbatches for update 50 have completed backward;
2. gradients have been clipped;
3. AdamW optimizer step 50 has completed;
4. the manual LR assigned for update 50 is present in the optimizer;
5. CUDA work is synchronized;
6. accumulation position is zero and no gradient is semantically live;
7. the next sample offset is 1600; and
8. the batch for update 51 has not been requested from the iterator.

This is a post-optimizer, post-schedule-assignment, pre-next-batch boundary. RNG and generator hashes
are captured immediately before serialization and again after serialization; saving must not mutate
them. Run B loads the checkpoint into a newly initialized policy/optimizer, reconstructs the offset
loader, restores every RNG/generator state, verifies all boundary hashes/state, and begins at update
51.

## 8. Primary executions

- **Run A — uninterrupted:** official base, updates 1–100, save complete state at update 50, retain
  forensic trace for updates 51–100 and final model/optimizer state.
- **Run B — resumed:** new interpreter, load the new update-50 complete checkpoint, run updates
  51–100, and retain an identically structured trace and final state.

No run exceeds 100 updates. Run B must use the same physical GPU and environment as Run A.

## 9. Captured comparison quantities

For every continuation update, record:

- update, LR, optimizer-step count, example offsets, and exact batch/sample IDs;
- raw-action, processed-tensor, Gaussian-noise, and flow-timestep hashes;
- both scalar losses and their tensor hashes;
- selected early/action/late parameter, gradient, and Adam-moment hashes;
- global gradient norm and boundary RNG/generator hashes.

At updates 51, 60, 75, and 100 additionally record whole-model, buffer, full-gradient, and full
optimizer tensor digests. At update 100 save complete model and name-keyed optimizer tensors for
streaming numerical comparison. Compare model parameters, buffers, Adam first/second moments, step
counters, parameter groups, LR/manual scheduler record, and the explicit non-applicability of
GradScaler.

## 10. Frozen equivalence and verdict rules

The primary gate passes only if:

- all batch IDs, processed tensor hashes, flow timesteps/noise, and loss tensors/scalars match
  exactly for updates 51–100;
- no earlier state comparison differs;
- final maximum absolute model/buffer difference is `<1e-6` (exact zero is preferred and reported);
- final optimizer tensor states and parameter groups match exactly;
- manual scheduler/update/LR state and AMP applicability match exactly; and
- final RNG, loader generator, flow generator, sampler offset, and accumulation position match.

Report final model maximum/mean/L2 difference and the first differing update/stage, if any. Any
logical state mismatch makes the gate fail even if immediate model parameters happen to agree.

Apply the first applicable verdict:

1. **R-D — RESUME GATE FAILED** for unexplained divergence under the controlled configuration.
2. **R-C — RESUME STATE BUG IDENTIFIED BUT NOT FULLY FIXED** when a concrete state bug remains.
3. **R-E — INFRASTRUCTURE BLOCKER** for a concrete external/library blocker requiring substantial
   engineering.
4. **R-A — RESUME GATE PASSED** only if the primary gate passes and an optional intended four-worker
   check also passes.
5. **R-B — RESUME GATE PASSED WITH RESTRICTED LOADER** when the primary zero-worker gate passes but
   the four-worker configuration is untested or fails.

## 11. Optional four-worker check

Only after a primary pass and if accounted usage remains comfortably below 0.5 device-hours, run a
fresh, symmetric M0 check using Phase-1 loader settings (`num_workers=4`, persistent workers,
prefetch factor 2) over updates `0→5→10`, with a new process resuming updates 6–10. Use the same
complete checkpoint format and deterministic CUDA controls. This test may establish R-A for the
current deterministic, augmentation-free dataset path. Worker RNG processes are not checkpointed;
if exactness fails, stop and return R-B without elaborate worker-state engineering.

## 12. Stop conditions and compute ceiling

- Maximum new usage: **0.5 device-hours**, summed over all resume-gate processes.
- Stop at the first unexplained primary divergence or after one evidence-driven minimal
  infrastructure fix and rerun.
- Skip the optional worker check if projected total usage is not comfortably below the cap.
- Stop after an R-A/R-B pass; do not begin Phase 1, M1/M2, rollouts, hyperparameter tests, or method
  development.
- A corrupted checkpoint, mismatched frozen revision/order, non-finite value, nonzero accumulation
  position, or inability to restore boundary state invalidates the relevant run.

Large checkpoints and traces live outside git under
`/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/resume_gate/`.
