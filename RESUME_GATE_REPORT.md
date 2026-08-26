# FlowSpec-VLA Deterministic M0 Resume-Gate Report

Date: 2026-08-25

Verdict: **R-A — RESUME GATE PASSED**

## 1. Verdict

A new complete checkpoint at update 50 reproduced the uninterrupted M0 continuation through update
100 in a fresh process. All 50 continuation batches, processed tensors, flow timesteps, Gaussian
noise tensors, forward outputs, losses, gradients, optimizer updates, and RNG boundaries matched
exactly. At update 100, all 500 model tensors and all 1,467 name-keyed AdamW tensors were bitwise
identical; maximum, mean, and L2 differences were all zero.

The optional intended four-worker/prefetch check also passed from update 5 through update 10 with
zero model and optimizer difference. The infrastructure is therefore technically eligible for a
future clean Phase-1 rerun under the exact controlled configuration below. No Phase-1 rerun is
authorized or performed.

## 2. Motivation

The prior forensic result localized fresh-run divergence to default CUDA backward numerical
nondeterminism and showed that deterministic CUDA/cuBLAS controls remove it. Historical Phase-1
checkpoints were model/processor-only and could not establish exact resume. This gate tests the
missing engineering link: complete-state, deterministic, fresh-process resume for M0.

## 3. Exact deterministic configuration

- Protocol-freeze commit: `a2a37dd9a5e6a8549fea54d165727b9864650acf`.
- Starting code: `78553da2cf2799bfd84a8a354d025826f9053e2b`.
- M0 official base revision:
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- LeRobot source: `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Dataset revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- Physical GPU 0, one process, one RTX 3090, no DDP or compilation.
- Seed 570001; order/loader/flow streams use `seed+101/+102/+103`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization.
- `torch.use_deterministic_algorithms(True)`.
- cuDNN deterministic true, benchmark false; matmul and cuDNN TF32 false.
- bf16 CUDA autocast without GradScaler.
- AdamW, LR schedule, batch 16, accumulation 2, and all scientific training semantics unchanged.

## 4. Exact checkpoint semantics

Checkpoint `update=50` is post-backward, post-gradient-clipping, and post-AdamW step 50. LR for
update 50 is present in the optimizer. CUDA is synchronized, accumulation position is zero, examples
seen are 1,600, and the next absolute sample offset is 1,600. It is saved before the main process
requests the update-51 batch. Checkpoint serialization did not change any audited RNG state.

## 5. Complete state inventory

The versioned `flowspec_vla.complete_training_state` checkpoint stores:

- model parameters, buffers, and train/eval mode;
- full AdamW state dict, moments, step counters, and parameter groups;
- manual cosine scheduler configuration, completed update, optimizer-step count, and LR;
- explicit bf16/no-GradScaler AMP record;
- update, examples, epoch, batch/accumulation positions, sampler offset, and next update;
- Python, NumPy, torch CPU, every visible CUDA RNG, loader generator, and flow generator;
- full materialized-order hash and loader configuration;
- model/dataset/source revisions and deterministic execution settings.

The update-50 checkpoint is `2,492,028,710` bytes with SHA-256
`4b6366eab60f81055912020249b0543e302117d421d0dc6fb62f8d316829316e`.

## 6. Run A setup

Run A started from the official base and trained M0 from updates 1–100. The primary DataLoader used
`num_workers=0`, pinned memory, and a fixed materialized order. It saved complete state at update 50
and continued uninterrupted through update 100. Detailed whole-state hashes were retained at updates
51, 60, 75, and 100; compact chain evidence was retained at every continuation update.

## 7. Run B setup

Run B launched in a new interpreter on the same physical GPU. It initialized the frozen substrate,
constructed an offset sampler beginning at example 1,600, loaded model and AdamW state, and restored
all global and dedicated RNG states after iterator construction. Its pre-update-51 restore audit
found exact model, optimizer, parameter-group, RNG, scheduler, AMP, and training-position state.
It then ran updates 51–100 while comparing each update online against Run A.

## 8. First resumed-update comparison

At update 51, all of the following were exact:

- pre-forward model and selected parameters;
- complete and representative optimizer state;
- LR/manual scheduler state and AMP applicability;
- every RNG/generator state and sampler offset;
- 32 sample identities and raw actions across both microbatches;
- every emitted processed tensor;
- flow timestep and Gaussian-noise tensors;
- action-projection output and full loss tensor;
- scalar losses, accumulated gradients, gradient norm, Adam update, and resulting parameters.

No first divergence exists.

## 9. Final update comparison

Update 100 retained the same logical identity. The complete final model and optimizer exports were
compared tensor by tensor. Scheduler, AMP, RNG, and final training position were compared separately.
Every comparison was exact.

## 10. Model parameter equivalence

- Tensors compared: 500
- Elements compared: 450,046,176
- Differing tensors: 0
- Maximum absolute difference: `0`
- Mean absolute difference: `0`
- L2 difference: `0`
- Bitwise identity: **yes**

This is stricter than the frozen `<1e-6` threshold.

## 11. Optimizer equivalence

- Name-keyed AdamW tensors compared: 1,467
- Elements compared: 785,808,681
- Differing tensors: 0
- Maximum/mean/L2 difference: `0 / 0 / 0`
- First moments, second moments, and per-parameter step tensors: exact
- Parameter groups, including LR and hyperparameters: exact

## 12. Scheduler/AMP equivalence

The manual cosine record, completed update, optimizer-step count, expected/current LR, and schedule
constants matched exactly. There is no scheduler object. Both paths used bf16 autocast without a
GradScaler; the explicit non-applicability record matched exactly.

## 13. RNG equivalence

Checkpoint-loaded Python, NumPy, torch CPU, all visible CUDA, loader-generator, and flow-generator
states matched the saved boundary. Their per-update and final hashes also matched. Flow timesteps and
Gaussian noise matched directly, not merely by inferred seed equivalence.

## 14. Sampler/data equivalence

The full 3,200-example primary order SHA-256 was reconstructed and the resumed sampler began at
absolute offset 1,600. Batch IDs, raw actions, images/state/tokens/normalized actions, and all other
processed tensors matched for every continuation update. Final sample/batch/accumulation positions
were exact.

The optional intended loader used four workers, persistent workers, pinned memory, and prefetch
factor 2. Its fresh-process update-5 checkpoint continuation through update 10 also matched exactly,
including final model and Adam state. The current dataset/processor path has no stochastic worker
augmentation.

## 15. Discovered bug and fix

The historical infrastructure defects were addressed by two deliberate changes: the already proven
deterministic CUDA/cuBLAS launch configuration and a new complete-state checkpoint/restore format.
No scientific model or objective changed.

One first Run-B attempt stopped before update-51 training because the online comparator treated an
in-memory optimizer `tuple` as unequal to the identical JSON-decoded `list`. Full restore digests
were exact and the two serialized records were identical. The comparator alone was changed to
canonical JSON comparison, after which the unchanged checkpoint passed. This was instrumentation,
not a training-state divergence.

## 16. Remaining limitations

- The main 50-update continuation used `num_workers=0`; the intended four-worker configuration was
  verified over a shorter five-update continuation.
- Worker process RNG internals are not checkpointed. This is safe for the currently verified
  augmentation-free deterministic dataset path, not a claim about future stochastic transforms.
- The gate covers one M0 seed, one RTX 3090/software stack, and 100 total updates.
- Long-run throughput and scientific performance under deterministic kernels were not tested.
- Old Phase-1 checkpoints and results remain invalid; they were not reused.

## 17. Exact GPU usage

Successful runs used `1,082.2893` device-seconds (`0.3006359` device-hours): primary reference
`488.6384` s, primary resumed `289.3230` s, optional four-worker reference `185.2207` s, and optional
resumed `119.1072` s. Including a conservative `52.5` seconds for the comparator-only failed attempt,
total accounted usage was **`0.3152193` device-hours**, below the 0.5-hour cap. No run exceeded 100
updates.

## 18. Whether Phase 1 is technically safe to rerun

**Yes, technically eligible under this controlled infrastructure; not authorized in this gate.**
The prior P6 result and old scientific outputs remain invalid. A future Phase-1 comparison must start
from scratch and apply the same deterministic and complete-checkpoint configuration symmetrically to
every method and seed.

## 19. Exact configuration required for such a rerun

Use `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch algorithms, deterministic cuDNN with
benchmark and TF32 disabled, bf16 autocast without GradScaler, one process/GPU, the materialized
fixed sampler with explicit absolute offset, captured loader/flow generators and all global RNG
states, and the new versioned complete checkpoint at clean post-update boundaries. The intended
four-worker/persistent-worker/prefetch-factor-2 loader is supported only for the currently verified
augmentation-free dataset path. Any future stochastic worker transform requires a new loader-state
gate.
