# FlowSpec-VLA Phase-1 Frozen Protocol

**Action-Interface Consistent Post-Training for Flow-Matching VLAs**

Status: **FROZEN BEFORE PHASE-1 TRAINING OR COMPARATIVE RESULTS**

The protocol-freeze commit is recorded in `EXPERIMENT_STATE.md` immediately after this file is
committed. The implementation revision governed by this protocol is
`19cb03d7de4eb68a4cb10e2dbe82797765f9f560`.

## 1. Question and scope

The sole primary question is:

> Does removing stochastic flow dynamics associated with non-executable padded action dimensions
> improve matched-budget SmolVLA post-training on LIBERO?

Gate 0 established that varying only the 25 padded source-noise dimensions changes predictions in
the seven executable dimensions. It did not establish task-level harm. Phase 1 tests exactly three
methods and no others:

- **M0 — Standard FM:** official full 32-dimensional Gaussian source and unmodified Euler flow.
- **M1 — Source-Masked FM:** coordinates 7:32 of source noise are set exactly to zero. All 32
  coordinates subsequently evolve without projection.
- **M2 — Full-Subspace FM:** the same source mask as M1, followed by projection of coordinates
  7:32 to zero after every Euler update. Valid coordinates 0:7 are not altered by projection.

No auxiliary loss, architecture change, normalization variant, RL objective, preference method,
learned reward, action reweighting, or post-result tuning is permitted.

## 2. Implementation audit and exact intervention

Pinned LeRobot source `8b256a6c0d4769c3cc3e7e98f04940126398a391` implements training as one
flow-matching regression evaluation:

`x_t = t * noise + (1 - t) * action`, `u_t = noise - action`, followed by MSE to the predicted
velocity. LIBERO actions are padded from 7 to 32 before this call, while loss and output are sliced
to seven dimensions only at the boundary. Inference uses ten forward-Euler updates from `t=1` to
`t=0`.

The intervention is isolated in `src/flowspec_vla/phase1.py`:

- `phase1_source` returns official noise unchanged for M0 and zeros only `[..., 7:]` for M1/M2.
- Training otherwise calls the official model forward path, with the official valid-coordinate loss
  slice and temporal padding mask.
- `sample_actions_with_trace` reproduces the official prefix/cache/denoise path. M1 changes only the
  initial source. M2 additionally calls `project_executable_subspace` after every Euler update.
- Output slicing, valid targets, normalization, and unnormalization are shared.

The official training objective contains no ODE rollout. Since padded target and padded source are
both zero for M1/M2, their training inputs and objectives are necessarily identical. The separately
executed matched M1/M2 runs test this invariant; their additional scientific distinction is M2's
projection during sampling/evaluation. Adding a repeated training-time rollout or padded loss would
be a new intervention and is prohibited.

Pre-training verification in
`generated_artifacts/flowspec-vla/phase1/implementation_verification.json` passed all checks:

- M0 custom inference versus official inference: maximum absolute difference 0.
- M1/M2 padded source: exactly zero; valid source: exactly unchanged.
- M1 padded flow state becomes nonzero after the first network update.
- M2 padded state is exactly zero at source and after every update.
- M2 valid-coordinate projection and valid gradients are exactly unchanged.
- M1 and M2 training losses are exactly identical for shared inputs.
- Executable prediction gradients are finite and nonzero.
- Physical target normalization round-trip maximum error: `2.9802322387695312e-08`.

Failure of these invariants in a real run invalidates the comparison and triggers P6.

## 3. Frozen substrate and revisions

| Item | Frozen value |
|---|---|
| Phase-1 implementation | `19cb03d7de4eb68a4cb10e2dbe82797765f9f560` |
| Gate-0 protocol | `337c95305698e527ef09db6f53040fb377a9d6a5` |
| Gate-0 final | `d17670f51169478a8dcd3b48ac7354baa6deac13` |
| LeRobot source | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| Base checkpoint | `lerobot/smolvla_base` |
| Base revision | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Base weight SHA-256 | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| Base config SHA-256 | `650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d` |
| LIBERO recipe source | released `lerobot/smolvla_libero` training configuration |
| Recipe revision | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| Recipe config SHA-256 | `df25e6fce4cf2161b7a22c495230e332bad5f73f1ca07605b35fa77500917b9c` |
| Dataset | `lerobot/libero` |
| Dataset revision | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Split-manifest SHA-256 | `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0` |
| Internal / executable dimensions | 32 / 7 |
| Action chunk / execution dimensions | 50 / 7 |
| Euler steps | 10 |

The starting weights are the official general SmolVLA base, not the saturated task-specific LIBERO
checkpoint. The released LIBERO configuration supplies the official downstream feature convention
and trainability settings: vision is unfrozen, training is not expert-only, and state projection is
trainable. This gives 392,904,096 trainable parameters of 450,046,176 total. Two LIBERO cameras are
renamed to the base model's `camera1` and `camera2` keys; its unused third-camera declaration does
not inject an image because `empty_cameras=0`.

## 4. Frozen task suite, split, and normalization

All 40 tasks are used: the ten tasks in each of `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. No task selection will follow results.

Episodes are ranked independently within each task by ascending
`SHA256("510001:{task_index}:{episode_index}")`:

- first five episodes per task: held out entirely;
- next eight episodes per task: training;
- all other episodes: unused.

This produces exactly 320 training episodes and 53,762 training frames, or 19.659554% of all
273,465 official frames. It is one fixed low-data fraction. The held-out pool has 200 episodes. Two
validation states per held-out episode are fixed at integer offsets `floor(0.25 * length)` and
`floor(0.60 * length)`, producing exactly 400 validation states. Chunk padding is tracked and
excluded from every error denominator. No held-out episode or validation frame occurs in training.

State/action mean and population standard deviation are computed once from the 53,762 selected
training frames only, then shared by all methods and all seeds. Visual normalization remains the
official identity transform. This avoids validation-statistic leakage and is not a tested
normalization method. The exact indices and statistics are stored in `split_manifest.json`.

## 5. Baseline reproduction gate

Before any M1/M2 primary run, M0 is run from the base checkpoint for 250 updates with the exact
Phase-1 data loader, preprocessing, optimizer, precision, microbatch, accumulation, scheduler prefix,
and validation code. It uses seed `519001`, the first 64 frozen validation states, and evaluation at
updates 0, 50, 100, and 250.

The reproduction passes only if all conditions hold:

1. every loss, gradient norm, and generated action is finite;
2. mean loss over updates 201–250 is no greater than mean loss over updates 1–50;
3. either final physical NRMSE is at most 1.05 times its initial value or final fixed-noise flow MSE
   is at most 0.95 times its initial value;
4. peak allocated memory is no more than 23.0 GiB;
5. the pre-training M0 exact-equivalence and physical round-trip tests remain passed.

Failure after correction of a clearly identifiable engineering bug gives
**P6. NO-GO — TRAINING / SUBSTRATE INVALID**. M1/M2 will not run on a failed baseline.

## 6. Primary matched training

| Setting | Frozen value |
|---|---|
| Methods | M0, M1, M2 |
| Seeds | `520001`, `520002`, `520003` for every method |
| Runs | 9 total |
| Updates per run | 5,000 fixed; no early stopping |
| Examples per run | 160,000, approximately 2.98 passes over the subset |
| Microbatch | 16 on one GPU |
| Gradient accumulation | 2 |
| Effective batch | 32 |
| Optimizer | AdamW |
| Peak LR | `1e-4` |
| Betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Weight decay | `1e-10` |
| Gradient clipping | global norm 10.0 |
| Schedule | 200-update linear warmup, cosine decay through update 5,000 |
| Final LR | `2.5e-6` |
| Precision | bfloat16 CUDA autocast; float32 loss/projection/metrics |
| Augmentation | disabled, matching released LIBERO recipe |
| Trainable parameters | exactly 392,904,096 for every run |
| Checkpoints | fixed updates 1,000, 2,500, 5,000 |
| Validation | updates 0, 250, 500, 1,000, 2,000, 3,000, 4,000, 5,000 |
| Main checkpoint | fixed final update 5,000 |

Independent runs are assigned to separate 3090s; no DDP is used. Corresponding seeds share exact
training indices, valid source-noise values, time draws, initialization, and validation randomness.
Data-order randomness and flow randomness use separate streams. No method receives a different
hyperparameter or checkpoint-selection rule.

Because M1/M2 training objectives are identical, corresponding M1/M2 loss trajectories and weights
must agree to deterministic numerical tolerance. A maximum paired checkpoint-parameter difference
above `1e-6`, or unexplained metric divergence before sampling, triggers a reproducibility audit and
halts comparison if unresolved.

## 7. Frozen metrics

### P1 — Physical executable action error

Predicted valid actions are converted to raw LIBERO physical coordinates using the one shared
training-split transform. Targets are the original raw dataset actions. Temporal padding is excluded.

The primary scalar is whole-action normalized physical RMSE: each physical coordinate error is
divided by its frozen training-split physical standard deviation, then squared and averaged over
valid chunk positions and the seven coordinates. Lower is better. Also report raw-coordinate RMSE
for translation dimensions 0:3, rotation dimensions 3:6, gripper dimension 6, every individual
dimension, and each of the 50 chunk positions.

### P2 — Optimization/sample efficiency

The primary learning curve is whole-action normalized physical RMSE at the eight fixed validation
updates. Report final update-5,000 error and trapezoidal area under this error curve divided by 5,000
(AULC; lower is better). Examples seen equal `32 * update`. No threshold-crossing metric is primary,
because a defensible absolute threshold is not available before the reproduction gate.

### P3 — Mechanism suppression

At each final checkpoint, repeat the Gate-0A protocol on the exact same 64 frozen states, tasks,
state order, 16 padded interventions, 16 valid interventions, noise seeds `410000 + state_ordinal`,
and ten Euler steps. Report padded RMS, valid RMS, and
`R_leak = median(state padded variance) / median(state valid variance)`, with a 10,000-resample
state bootstrap.

For M1, the Gate-0 intervention deliberately injects padded source noise only for diagnosis and then
allows ordinary evolution; the operational M1 sampler still starts padded coordinates at zero. For
M2, source masking/projection is applied before the first denoise call and after every update, so the
same injected padded perturbation must be suppressed. Separately record operational padded-state and
valid/padded velocity RMS across integration steps on the frozen states.

### Secondary diagnostics

Report training loss, global gradient norm, NaN/overflow events, valid and padded predicted-velocity
RMS, update time, peak GPU memory, and M0/M1/M2 padded-state trajectories. Cross-dimensional
sensitivity may be estimated only if it does not delay the primary experiment.

## 8. Statistical unit and summaries

Training seed is the independent replicate. Report every seed, mean, standard deviation, paired
same-seed differences, relative differences, and paired standardized effect size
`d_z = mean(paired difference) / SD(paired difference)`. With three seeds, magnitude, direction, and
reproducibility take priority over p-values. State-level mechanism confidence intervals bootstrap
states, not individual chunk coordinates. Validation states are not represented as independent
training replicates.

## 9. Frozen practical thresholds and verdict decision tree

For method A to **materially outperform** B on primary post-training behavior, all must hold:

1. mean final whole-action normalized physical RMSE is at least 5% lower;
2. the paired final difference favors A in at least two of three matched seeds;
3. mean normalized physical AULC is at least 3% lower.

M1 and M2 are **practically equivalent** when their mean final relative difference has magnitude no
more than 2% and neither method materially outperforms the other. A rollout difference is supportive
when mean success differs by at least five percentage points in the same direction for at least two
of three seeds; rollout cannot overturn a failed primary physical/efficiency criterion.

Mechanism suppression is strong when padded-noise RMS is at most `1e-4` and `R_leak` is at most
`1e-4`. Partial suppression means at least a 50% reduction in both padded RMS and `R_leak` relative
to matched M0, with the same direction in at least two seeds.

Apply the first matching verdict in this order:

1. **P6. NO-GO — TRAINING / SUBSTRATE INVALID:** a stop condition invalidates the experiment.
2. **P4. CONTINUED-COUPLING EFFECT:** M2 materially outperforms both M1 and M0 and strongly
   suppresses the mechanism.
3. **P3. SOURCE-ONLY EFFECT:** M1 and M2 each materially outperform M0, are practically equivalent,
   and M2 strongly suppresses the mechanism.
4. **P1. STRONG GO:** M2 materially outperforms M0 with strong mechanism suppression and reproducible
   seed direction, but the pattern does not satisfy the more specific P3/P4 definitions.
5. **P2. MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK:** strong/partial mechanism suppression occurs
   and at least one intervention has a positive mean final or AULC reduction of at least 2%, but the
   material-outperformance rule fails or direction is inconsistent.
6. **P5. NO-GO — LEAKAGE NOT PRACTICALLY HARMFUL:** neither intervention materially outperforms M0,
   and improvements are absent, below 2%, or fail to favor the intervention in two matched seeds.

No new method will be added for any outcome.

## 10. Frozen rollout evaluation

Rollouts begin only after all nine primary final checkpoints and primary validation analysis exist.
Use the fixed update-5,000 checkpoint for every run. Evaluate the same four suites and all ten tasks
per suite, using the first five official deterministic initialization states per task: 200 episodes
per checkpoint and 1,800 episodes total. Environment horizons are the pinned LeRobot defaults:
280 (`spatial`), 280 (`object`), 300 (`goal`), and 520 (`libero_10`). Control mode is relative; two
256x256 cameras and the standard eight-dimensional LIBERO state are used. Report seed-level and
suite/task-level success with binomial intervals descriptively; training seed remains the replicate
for method comparison. Episodes will not be added conditionally.

If the rollout substrate proves invalid after normal engineering correction, record the failure in
`rollout_summary.json`; do not substitute tasks, states, or episode counts. A rollout-only substrate
failure does not erase a valid primary offline comparison but limits the final claim.

## 11. Compute budget and accounting

One feasibility update with batch 16 used 15.90 GiB on an RTX 3090. The approximate authorized
budget is:

- baseline reproduction: 0.5 GPU-hour;
- nine primary training runs including scheduled validation: 36 GPU-hours;
- leakage and diagnostics: 2 GPU-hours;
- 1,800 rollouts: up to 30 GPU-hours;
- total ceiling: approximately 72 GPU-hours.

Actual active wall time is recorded per GPU process and summed by category. Independent runs are
paired across the two GPUs when possible. No method-dependent OOM fallback is allowed. If identical
batching cannot fit for all methods, fix the shared implementation or stop.

## 12. Stop conditions

Stop the relevant branch rather than substitute an experiment if any occurs:

- baseline reproduction fails after reasonable correction of a legitimate implementation bug;
- non-finite loss, gradient, action, or unexplained overflow persists;
- M0 fails exact official-path equivalence;
- source masking or M2 per-step projection fails its invariant;
- executable targets, normalization, unnormalization, slicing, trainable parameters, data order, or
  update budget differ across methods;
- train/validation episode overlap or split corruption is detected;
- corresponding M1/M2 training objectives unexpectedly differ and reproducibility cannot be restored;
- a required checkpoint is missing/corrupt or evaluation cannot be applied symmetrically;
- projected total compute exceeds the 72 GPU-hour ceiling before all symmetric runs can finish.

Ordinary data-loader, logging, headless-rendering, cache, or scheduler bugs may be fixed if the frozen
scientific comparison is unchanged and the fix is documented.

## 13. Required outputs

Large checkpoints live under `/mnt/NAS/data/hl5757/models/flowspec-vla/phase1`; raw logs and arrays live
under `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1`. Final machine-readable files are:

- `baseline_reproduction_summary.json`
- `training_summary.json`
- `validation_summary.json`
- `leakage_audit_summary.json`
- `rollout_summary.json`
- `gpu_usage.json`
- `artifact_manifest.json`

The final report and plots will use only actual completed results. The final fixed report is
`PHASE1_REPORT.md`; `README.md` and `EXPERIMENT_STATE.md` will record the verdict and commits. The
artifact manifest will SHA-256 hash important configs, reports, code revisions, inputs, and outputs.

