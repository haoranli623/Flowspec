# FlowSpec-VLA M0 Deterministic/Resume Forensic Report

Date: 2026-08-25

Scope: bounded M0-only infrastructure audit

Verdict: **F-C — HARDWARE/NUMERICAL NONDETERMINISM IDENTIFIED**

## 1. Executive verdict

The first failed arrow is:

`identical loss → different backward gradients` at **update 1**.

Fresh-A and Fresh-B were identical through model/buffer state, sample identity, raw and processed
inputs, all audited RNG states, explicit flow noise/timestep, action-projection output, and the full
loss tensor for both accumulation microbatches. Their gradients then differed before clipping. The
default same-process rollback reproduced that boundary. Enabling
`torch.use_deterministic_algorithms(True)` together with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` made both the same-process replay and two independent-process
one-update runs bitwise exact. This causally localizes the observed fresh-run instability to CUDA
backward numerical execution under the official/default kernel configuration.

No production fix was adopted. The historical Phase-1 checkpoint is also model/processor-only and
is not an exact training-resume checkpoint. Therefore Phase 1 is **not** technically safe to rerun,
and its existing P6 verdict is unchanged.

## 2. Exact question investigated

For official M0 SmolVLA post-training, identify the first failure in:

`saved state → batch → stochastic tensors → forward → loss → gradients → optimizer → parameters`.

The audit did not run M1, M2, rollouts, a long training job, a method comparison, or any new
scientific variant.

## 3. Protocol freeze

- Protocol: `FORENSIC_PROTOCOL.md`
- Protocol-freeze commit: `e1cc6d1eb8011dffbf9038bc71b08f9160780570`
- Audited starting code: `1784b512b036807bf85c684d35f079b2cf717e78`
- Model: `lerobot/smolvla_base` revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`
- Dataset: `lerobot/libero` revision
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- Hardware: physical GPU 0, NVIDIA RTX 3090, single process, no DDP or compilation
- Precision: bf16 CUDA autocast, no GradScaler
- Seed: 560001 with dedicated loader and flow generators
- Frozen compute cap: 2.0 device-hours

## 4. Prior P6 context

Phase 1 remains **P6. NO-GO — TRAINING / SUBSTRATE INVALID**. Its descriptive final physical NRMSE
values (M0 `0.71295 ± 0.00460`, M1 `0.72192 ± 0.00114`, M2
`0.71653 ± 0.00590`) do not establish a method ordering. Prior matched-run parameter differences
were roughly `0.02718`, `0.03040`, and `0.02802`; a same-device recovery comparison still reached
`0.02034` at update 1000. The present audit diagnoses infrastructure only and does not reinterpret
those scientific outcomes.

## 5. Fresh-vs-fresh result

Fresh-A and Fresh-B each ran five official M0 updates with identical configurations on the same
physical GPU in separate interpreters.

| Chain stage at update 1 | Result |
|---|---:|
| Initial parameters and buffers | exact |
| Python/NumPy/torch CPU/all CUDA RNG | exact |
| Loader/flow generator states | exact |
| Sample IDs and raw actions | exact |
| All processed tensors | exact |
| Flow Gaussian noise and timestep | exact |
| Both action-projection outputs | exact |
| Both loss tensors/scalars | exact |
| Gradients before clipping | **different** |
| Gradient norm | `20.3197994` vs `20.3206997` |
| Optimizer result and parameters | different |

After five updates, 344 checkpoint tensors differed. The maximum parameter difference was
`7.62939453125e-6`, mean absolute difference `1.8547922e-9`, and global L2 difference
`0.0013746294`. Thus checkpointing is not necessary for the failure to appear.

## 6. Uninterrupted-vs-resumed result

This GPU test was **not run by the frozen F1 stop rule**. F1 had already found a pre-checkpoint
update-1 divergence, and the protocol required proceeding directly to that earliest subsystem and
stopping after minimal causal verification.

Static audit nevertheless establishes that the historical Phase-1 `save_checkpoint` calls
`save_pretrained` only for the policy, preprocessor, and postprocessor. It does not store optimizer,
manual LR/update, RNG, dedicated generators, sampler offset, worker/prefetch state, model mode, or
accumulation position. Consequently it cannot provide exact training resume. Short resume
equivalence remains **not established**.

## 7. Same-process replay result

One exact pair of processed microbatches and explicit flow tensors was cached. The full initial model
and RNG state was restored in place, and one accumulated update was executed twice.

- Default CUDA path: forward and loss exact; 180 of 489 gradient tensors differed; gradient maximum
  absolute difference `7.32421875e-4`; gradient L2 difference `0.045322`; updated model hash differed.
- Deterministic diagnostic: all 489 gradient tensors exact; gradient maximum difference `0`; updated
  model exact.

This removes dataset workers, new interpreter state, checkpoint serialization, and batch selection as
causes of the first failure.

## 8. RNG audit

Python, NumPy, torch CPU, every visible CUDA RNG, the `seed+102` loader generator, and the `seed+103`
flow generator were hashed before and after the relevant stages. Fresh-A/Fresh-B hashes matched at
all five updates, including before/after forward and backward. Explicit flow tensors also matched.
No exercised custom SmolVLA dropout path was found. Global or dedicated RNG mismatch is eliminated
as the first cause.

Historical Phase-1 checkpoints save none of these RNG states, which remains a separate resume defect.

## 9. Dataloader/sampler audit

Training uses `FixedSampler` over a fully materialized permutation generated by NumPy seed
`seed+101`; the DataLoader has four workers, pinned memory, prefetch factor 2, persistent workers,
and a generator at `seed+102`. Worker initialization copies `torch.initial_seed()` into NumPy and
Python.

Across five Fresh updates, sample IDs, episode/frame/task IDs, raw actions, and processed tensors were
exact. The loader is deterministic from process start for this test. Exact resume would still require
the explicit order offset, loader generator, and worker/prefetch semantics, none of which the
historical checkpoint records.

## 10. Stochastic flow-input audit

Gaussian source noise and flow timesteps were sampled by the dedicated CPU generator and captured
before transfer to CUDA. Both accumulation microbatches matched exactly at every Fresh update. Flow
randomness is not the first divergent quantity.

## 11. Forward/backward audit

The action projection, full valid-coordinate loss tensor, and scalar loss were exact in both Fresh
microbatches at update 1. Divergence first appeared only after `.backward()`.

SmolVLA's exercised attention implementation is its custom eager path using `torch.matmul`, softmax,
and related reductions rather than PyTorch SDPA. The combined deterministic-algorithm/cuBLAS
workspace diagnostic eliminates the divergence. The evidence therefore identifies the CUDA/cuBLAS
backward numerical-execution class; it does not uniquely attribute the effect to one individual
kernel without a lower-level profiler experiment, which was unnecessary after satisfying the frozen
causal stop rule.

## 12. Optimizer/scheduler/AMP audit

The two Fresh runs began with identical empty AdamW state, parameter groups, and assigned LR. Their
gradients differed before clipping or `optimizer.step`, so AdamW is downstream rather than the first
cause. The scheduler is a pure manual cosine calculation from the update integer; there is no
scheduler object. bf16 autocast is used without GradScaler, so there is no scale/growth/skipped-step
state. Checkpoints nevertheless omit Adam moments/steps and the update/LR needed for resume.

## 13. First divergence point

- Update: **1**
- Last identical quantity: the second accumulation microbatch's loss tensor
- First divergent quantity: accumulated model gradients before clipping
- First differing tensor in same-process replay:
  `model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias`
- Default replay: 180/489 differing gradient tensors, max abs `7.32421875e-4`

## 14. Causal diagnosis

**CUDA backward numerical nondeterminism under the official/default execution configuration.**

The diagnosis is causal at the configuration/class level because:

1. every upstream data, state, RNG, forward, and loss quantity matched;
2. the failure reproduced inside one interpreter after full rollback;
3. deterministic CUDA/cuBLAS controls changed the replay from divergent to bitwise exact; and
4. two new interpreters under those controls produced identical final checkpoints.

Checkpoint omission is independently confirmed but is not the earliest cause of Fresh divergence.

## 15. Fix, if any

No production fix was implemented. The model, objective, optimizer, dataset, and Phase-1 training
entry point remain unchanged. Deterministic algorithms plus the deterministic cuBLAS workspace were
used only as a diagnostic mitigation, as frozen in the protocol.

Implementing exact resume would additionally require a versioned full-state checkpoint format. That
work was not authorized after the F1 stop condition and is not bundled here.

## 16. Post-fix verification

There is no production fix. Diagnostic verification results were:

- Same-process one-step replay: exact forward, loss, all gradients, and updated model; max gradient
  difference `0`.
- Two independent-process one-update runs: identical final checkpoints; 0 differing tensors and max
  parameter difference `0`.
- Short uninterrupted/resumed continuation: not run by the frozen F1 stop rule.

## 17. Remaining limitations

- The combined deterministic settings were tested as one minimal diagnostic; their individual
  necessity was not split further.
- No individual CUDA kernel was profiled because the causal subsystem and effective control were
  already established.
- The deterministic setting's long-run performance and throughput were not evaluated.
- Exact full-state resume was not implemented or tested.
- The test covers one RTX 3090/software stack and one M0 seed.

## 18. Exact additional GPU-hours

Successful recorded executions consumed `0.0669035` device-hours. Including a conservatively
accounted 56-second failed instrumentation attempt, total additional usage was
**`0.0824591` device-hours**, below the 2.0-hour cap. No run exceeded five updates; the replay tests
executed two one-update passes each.

## 19. Whether Phase 1 is now technically safe to rerun

**No.** The diagnosis makes repair realistic, but the production path does not yet enforce the
verified deterministic configuration and has no full-state resume checkpoint. A new M0/M1/M2 run
would therefore remain outside the frozen reproducibility guarantee. No Phase-1 rerun is authorized.

## 20. Recommended next action — description only

Under new authorization, add an opt-in deterministic production launch configuration and a versioned
full training-state checkpoint containing model/mode, AdamW, update/LR, all RNG and dedicated
generators, explicit sampler offset, and accumulation boundary. Then run one bounded M0-only
uninterrupted-versus-resumed verification before considering any Phase-1 rerun. Do not run M1/M2
until that infrastructure gate passes.
