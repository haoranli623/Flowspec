# FlowSpec-VLA Technical-Report Review Bundle

This is a factual handoff for an external technical-report writer. It is not a paper, does not add an experiment or method, and does not elevate quarantined evidence. Unless explicitly marked historical, Phase-1 performance claims below use only the deterministic clean rerun frozen at commit `82dfe670e6073bc30397d699ee612c0386932a00`.

## 1. Project identity

- **Project:** FlowSpec-VLA — *Action-Interface Consistent Post-Training for Flow-Matching VLAs*.
- **Research question:** Does removing stochastic flow dynamics associated with SmolVLA's non-executable padded action dimensions improve matched-budget LIBERO post-training?
- **Model / benchmark:** official SmolVLA base adapted on the official LeRobot LIBERO dataset family; offline validation and simulator rollouts cover all 40 tasks in `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
- **Project-declared final scientific verdict:** **P2 — MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK**.
- **Status:** **COMPLETE / STOPPED** at the Phase-1 boundary; no Phase 2 was authorized.
- **Final scientific-record commit before this documentation-only bundle:** `1acd919f03f55a93f821ac9a13872c8b6ce655ec`.
- **Protocol-freeze commits:** Gate 0 `337c95305698e527ef09db6f53040fb377a9d6a5`; original Phase 1 `beb292024fcb37d321338474249d03598cfa5e90`; forensic audit `e1cc6d1eb8011dffbf9038bc71b08f9160780570`; resume gate `a2a37dd9a5e6a8549fea54d165727b9864650acf`; clean Phase-1 rerun `82dfe670e6073bc30397d699ee612c0386932a00`.

The project-declared P2 label is repeated in `PHASE1_RERUN_REPORT.md`, `EXPERIMENT_STATE.md`, and `phase1_rerun_decision.json`. Section 21 records a conflict between that label and the literal original Phase-1 decision tree. A writer must preserve both facts rather than silently rewriting the outcome.

## 2. Motivation

### Verified implementation facts

LIBERO exposes a 7-dimensional executable action, while the audited SmolVLA configuration operates on a 32-dimensional internal action vector. The processor normalizes the 7 physical action coordinates, and the model path pads that normalized vector with 25 zeros. Standard flow matching nevertheless samples 32-dimensional Gaussian source noise, forms a 32-dimensional interpolated state, predicts 32-dimensional velocity, and integrates all 32 coordinates. The policy slices back to the executable coordinates only at the supervised-loss/output boundary.

### Research hypothesis

The 25 non-executable coordinates could create a nuisance pathway into the seven executable predictions. Phase 1 separated two possibilities: harm from initial Gaussian source noise in the padded coordinates, and additional harm from allowing padded state to evolve throughout the flow.

### Later empirical findings

Gate 0 showed measurable executable-output sensitivity to padded source noise. In the clean rerun, M2 reduced the diagnostic leakage ratio to approximately numerical zero. That suppression did **not** yield a material validation or rollout advantage over standard flow matching. M1 did not suppress the diagnostic pathway and did not improve downstream results.

## 3. Exact SmolVLA / LeRobot code-path audit

### Frozen substrate and revisions

| Component | Identifier / revision |
|---|---|
| LeRobot source | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| SmolVLA base | `lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| LIBERO recipe/config checkpoint | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` |
| SmolVLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Dataset | `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Dataset task-label parent | `1595a93b43aa055e55c127a4f0b4a99bb8035447` |

### Audited path

- **Dimensions:** executable `d_valid=7`; internal `D=32`; padded `25`; action chunk length `50`; inference Euler steps `10`.
- **Normalization:** checkpoint processors use `MEAN_STD`. For coordinate `j`, normalized action is `(a_j-mean_j)/(std_j+1e-8)` and unnormalization is `y_j*std_j+mean_j`. All methods share the same frozen training-split statistics.
- **Padding:** after normalization, `SmolVLAPolicy.prepare_action` calls `pad_vector(..., max_action_dim)` and pads the 7-vector to 32 dimensions with zeros.
- **Source noise:** official training/inference uses standard Gaussian noise with shape `[batch, 50, 32]`.
- **Flow interpolation:** training constructs `x_t = t*noise + (1-t)*action` with `t ~ Beta(1.5,1.0)*0.999+0.001`; target velocity is `u_t=noise-action`.
- **Velocity prediction:** the dense action projections consume 32-dimensional action states and emit 32-dimensional velocity. No separate valid/padded head exists.
- **Loss:** elementwise `MSE(u_t,v_t)` is produced for all 32 coordinates inside the model. The policy wrapper slices to the original executable action dimension and applies the temporal `action_is_pad` mask before reduction. The Phase-1 helper reproduces the same seven-coordinate reduction.
- **Inference:** official sampling begins with 32-dimensional Gaussian noise and performs ten Euler updates over all 32 coordinates.
- **Output:** executable action coordinates are sliced to the first seven only at the output boundary and then unnormalized.

Relevant upstream references are `upstream/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` lines 298–330 (loss slicing/masking), 415–418 (action padding), 689–724 (flow interpolation and velocity loss), and 726–773 (32D sampling and Euler integration), plus `upstream/lerobot/src/lerobot/policies/common/flow_matching.py` for noise/time sampling and Euler integration. Project intervention references are `src/flowspec_vla/phase1.py` lines 35–49, 125–154, and 158–231. Training/evaluation entry points are `scripts/run_phase1_rerun_training.py`, `scripts/run_phase1_training.py`, `scripts/run_phase1_leakage.py`, and `scripts/run_phase1_rollout.py`.

## 4. Gate 0 protocol and results

Gate 0A used 64 frozen states drawn evenly from four LIBERO tasks (task indices 0, 10, 20, and 30). Each state received `K_pad=16` padded-noise interventions and `K_valid=16` executable-noise interventions under the same ten-step sampler. Controls were native sampling, exact repeat, and zero-padded source. The implementation was valid only if repeat RMS was zero or no more than `1e-7`; a detected effect also had to exceed both the absolute `1e-4` floor and ten times repeat error. The frozen A1 strong-signal threshold was padded RMS at least `0.01` or leakage ratio at least `0.01`; A2 weak signal covered a reproducible effect above the floor but below the strong threshold.

| Gate-0A quantity | Exact value |
|---|---:|
| States / tasks | 64 / 4 |
| Padded interventions / state | 16 |
| Valid interventions / state | 16 |
| Padded-noise executable-action RMS | `0.010470231994986534` |
| State-bootstrap 95% CI | `[0.009601770900189877, 0.011180084198713303]` |
| Valid-noise executable-action RMS | `0.043740563094615936` |
| Leakage ratio, ratio of median variances | `0.05729682371020317` |
| Exact-repeat RMS / max absolute error | `0 / 0` |
| Padded-noise RMS in raw physical coordinates | `0.0030263536609709263` |
| Median maximum pairwise RMS | `0.017627698369324207` |
| Native vs zero-pad RMS | `0.018983714282512665` |
| Gate-0A verdict | `A1_STRONG_SIGNAL` |
| Overall Gate-0 verdict | **STRONG GO** |

Gate 0 established that, for the audited task checkpoint and frozen states, changing only Gaussian source noise in non-executable dimensions measurably changes executable action predictions. It established an implemented cross-dimensional coupling and exact repeatability of the audit. It did **not** show that the coupling harms training, validation, or task success; did not compare M0/M1/M2 post-training; and did not establish generality beyond the audited SmolVLA/LIBERO substrate.

## 5. Gate 0B normalization branch

Gate 0B was a bounded diagnostic with three coordinate systems—official, local, and group-scaled—two seeds (`430001`, `430002`), 64 updates per transform/seed, 128 frozen training anchors, and 32 validation states. Six runs therefore contributed 384 optimizer updates.

| Seed | Official RMSE | Local RMSE | Local vs official | Group-scaled RMSE | Group-scaled vs official |
|---|---:|---:|---:|---:|---:|
| 430001 | `0.441353` | `0.416337` | `-5.67%` | `0.510638` | `+15.70%` |
| 430002 | `0.445417` | `0.407181` | `-8.58%` | `0.437528` | `-1.77%` |

The frozen result was **B2 — WEAK SIGNAL**. Local coordinates improved the short diagnostic in both seeds, but the group-scaled result changed sign and was not robust. Because the evidence did not isolate normalization as a strong, reproducible cause, normalization was not promoted to Phase 1 and is not part of the final method claim.

## 6. Phase-1 methods

All three methods use the same base checkpoint, data, executable targets, normalization, model architecture, trainable parameters, optimizer, source draws in valid dimensions, chunk length, loss slicing, output slicing, and update budget.

### M0 — Standard FM

Uses the native 32-dimensional Gaussian source. Padded coordinates receive Gaussian noise and all 32 state coordinates evolve under the ordinary flow. There is no projection.

### M1 — Source-Masked FM

Clones the same source tensor and sets coordinates `7:32` to exactly zero; coordinates `0:7` are bitwise unchanged. It does not repeatedly project the state. The network can therefore produce padded velocity and padded coordinates can become nonzero after the first update. There is no auxiliary loss or architecture change.

### M2 — Full-Subspace FM

Applies the same initial source mask as M1, starts padded state at zero, and after every inference Euler update projects coordinates `7:32` back to zero while preserving valid coordinates and their autograd graph. Padded velocity may be predicted, but it cannot accumulate in padded state. There is no auxiliary loss or architecture change.

Official SmolVLA training evaluates a single sampled flow time and does not perform an iterative ODE rollout. Consequently M1 and M2 have exactly the same training computation; M2 differs from M1 during iterative sampling/evaluation. Clean-rerun final M1/M2 checkpoints were bitwise identical in all three seeds (`max_abs=0` over 500 model tensors per checkpoint).

## 7. Original invalid Phase 1

> **HISTORICAL INVALID EXPERIMENT — NOT FINAL SCIENTIFIC EVIDENCE**

The original Phase-1 attempt completed nine nominal runs (M0/M1/M2 × three seeds, 5,000 updates each), but its same-seed M1/M2 training-invariance requirement failed. The frozen reproducibility tolerance was maximum parameter difference `<1e-6`. Final matched M1/M2 checkpoint maxima were `0.02718`, `0.03040`, and `0.02802`; a same-device seed-520001 update-1,000 comparison still differed by `0.02034`. The experiment therefore received **P6 — NO-GO, TRAINING / SUBSTRATE INVALID**.

Those checkpoints lacked a complete resume state and were generated under a numerically nondeterministic training substrate. Their validation values are quarantined and must not enter final method tables, effect estimates, or claims. No leakage re-audit or valid rollout comparison followed that failed gate; the original rollout summary records zero episodes. The old experiment is relevant only as provenance for the forensic and resume-equivalence work.

## 8. Forensic audit

The forensic verdict was **F-C — hardware/numerical nondeterminism**.

- Fresh matched M1/M2 runs agreed exactly in initialization, materialized data order, processed batches, all recorded RNG states/generators, source noise, flow time, interpolated flow state, forward outputs, and scalar loss.
- The earliest observed divergence was the first update's accumulated backward gradients, before gradient clipping or optimizer stepping. Global norms were `20.3197994` and `20.3206997` on the two physical GPUs.
- After five updates, 344 tensors differed; maximum parameter difference was `7.62939453125e-6`, mean absolute difference `1.8547922e-9`, and L2 difference `0.0013746294`.
- In a same-process default backward diagnostic, 180 of 489 gradient tensors differed, with maximum absolute difference `7.32421875e-4` and L2 difference `0.045322`.
- Enabling deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8` made gradients and model updates exactly equal in the corresponding diagnostic and in independent short runs.

The evidence supports a class-level CUDA/cuBLAS backward-nondeterminism diagnosis, not attribution to one named kernel. The historical Phase-1 checkpoints remained non-resumable because they omitted optimizer/LR state, update position, complete RNG/generator state, sampler position, train/eval mode, and accumulation position. That historical omission could not be repaired retroactively.

## 9. Resume-equivalence gate

The resume gate verdict was **R-A — exact resume equivalence**. A reference run saved a complete checkpoint at update 50; an uninterrupted continuation and a new-interpreter resumed continuation were compared through update 100.

| Resume-equivalence item | Result |
|---|---|
| Batches, updates 51–100 | Exact for all 50 updates |
| Processed model inputs | Exact |
| Flow times | Exact |
| Gaussian source noise | Exact |
| Forward outputs / losses | Exact |
| Gradients and parameter updates | Exact |
| Final model | 500 tensors / 450,046,176 elements; max, mean, L2 difference all `0` |
| Final optimizer | 1,467 tensors / 785,808,681 elements; all differences `0` |
| Scheduler / AMP | Exact; bf16 autocast, no GradScaler |
| Checkpoint | 2,492,028,710 bytes; SHA-256 `4b6366eab60f81055912020249b0543e302117d421d0dc6fb62f8d316829316e` |
| Multi-worker check | 4 workers, persistent workers, prefetch 2; update 5→10 resume exact |

Required checkpoint state included model, complete AdamW state, scheduler/AMP state, current and next update, explicit next-example/sampler offset, accumulation position, Python/NumPy/Torch CPU/Torch CUDA RNG state, named generators for data order, flow noise/time, validation and loader seeding, configuration and revision identity, and mode metadata. Execution fixed deterministic PyTorch algorithms, cuDNN deterministic mode, cuDNN benchmark off, TF32 off, and cuBLAS workspace `:4096:8`, using one process and one GPU per run. This gate supplied the missing engineering precondition for a valid clean rerun.

## 10. Clean Phase-1 rerun protocol

| Item | Frozen value |
|---|---|
| Methods | M0, M1, M2 |
| Training seeds | `520001`, `520002`, `520003` |
| Scientific runs | 9 |
| Updates / examples per run | 5,000 / 160,000 |
| Total primary updates / examples | 45,000 / 1,440,000 |
| Tasks | 40 |
| Train split | 320 episodes (8/task), 53,762 frames |
| Held-out split | 200 episodes (5/task) |
| Validation | 400 states (2 anchors/held-out episode at fractions 0.25 and 0.60) |
| Split manifest SHA-256 | `b3150f3a4893f7f20167186a47f9f41a372eda0e95e1f91b6a6cb0851d32def0` |
| Microbatch / accumulation / effective batch | 16 / 2 / 32 |
| Optimizer | AdamW; betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `1e-10` |
| LR schedule | `1e-4` peak to `2.5e-6`, 200-update warmup, cosine decay through update 5,000 |
| Gradient clip | global norm `10.0` |
| Precision | CUDA bfloat16 autocast; no GradScaler |
| Trainable policy | all model parameters trainable; 392,904,096 trainable of 450,046,176 total |
| Workers | 4, persistent, prefetch factor 2 |
| Checkpoints | updates 1,000, 2,500, 5,000 |
| Validation | updates 0, 250, 500, 1,000, 2,000, 3,000, 4,000, 5,000 |
| Primary checkpoint rule | fixed final update 5,000; no cherry-picking |

Deterministic execution used one process/GPU, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch algorithms, deterministic cuDNN, cuDNN benchmark off, matmul/cuDNN TF32 off, explicit sampler offset, and complete RNG checkpoints. Corresponding seeds shared the same sample sequence and non-intervention randomness as closely as the protocol allowed.

Frozen practical rules were: material final NRMSE reduction at least 5%, paired direction in at least 2/3 seeds, and normalized AULC reduction at least 3%; M1/M2 equivalence band ±2% when neither materially outperforms the other; strong mechanism suppression padded RMS and `R_leak` each `<=1e-4`; partial suppression at least 50% for both relative to matched M0 in at least 2/3 seeds; supportive rollout improvement at least 5 percentage points in 2/3 seeds. Section 21 separately records the original decision-tree conflict.

The deterministic 250-update M0 reproduction gate passed: first/last 50-update train loss `2.18908/0.75096`, validation NRMSE `1.30475→0.88755`, fixed-flow MSE `2.37081→0.67592`, peak memory `18.2632 GiB`, finite numerics, and exact checkpoint resume.

## 11. Clean validation results

The operational metric used by code and artifacts is the original protocol's whole-action normalized physical RMSE: divide each physical-coordinate error by that coordinate's frozen training-split standard deviation, square, average across valid 7D chunk coordinates, then take the square root. Lower is better. Training seed is the independent replicate.

| Method | Seed 520001 | Seed 520002 | Seed 520003 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| M0 | `0.71299784` | `0.71480608` | `0.71355651` | `0.71378681` | `0.00092586` |
| M1 | `0.71711866` | `0.72171916` | `0.71203272` | `0.71695685` | `0.00484525` |
| M2 | `0.71430960` | `0.71937257` | `0.70947794` | `0.71438671` | `0.00494777` |

| Paired comparison | Candidate-minus-baseline by seed | Mean difference | Mean relative reduction | Candidate-favoring seeds | Paired `d_z` |
|---|---|---:|---:|---:|---:|
| M1 − M0 | `+0.00412083, +0.00691308, -0.00152379` | `+0.00317004` | `-0.444115%` | 1/3 | `+0.73755` |
| M2 − M0 | `+0.00131176, +0.00456649, -0.00407857` | `+0.00059990` | `-0.084044%` | 1/3 | `+0.13739` |
| M2 − M1 | `-0.00280906, -0.00234659, -0.00255477` | `-0.00257014` | `+0.358479%` | 3/3 | `-11.09634` |

M1 did not outperform M0 reproducibly: only seed 520003 favored M1 and its mean was worse. M2 was directionally better than M1 in all three seeds, but the relative improvement was only `0.358479%`, well inside the frozen 2% practical-equivalence band and below material criteria. M2 also did not improve over M0.

Normalized NRMSE AULC means (sample SD) were M0 `0.78191985 (0.00057370)`, M1 `0.78581313 (0.00262987)`, and M2 `0.78369518 (0.00272116)`. Relative to M0, M1 was `0.497913%` worse and M2 `0.227048%` worse. M2 was `0.269523%` better than M1 in AULC, again in all three seeds but far below the 3% material threshold.

Final physical-coordinate decompositions—secondary, in dataset physical units—were:

| Method | Translation RMSE | Rotation RMSE | Gripper RMSE |
|---|---:|---:|---:|
| M0 | `0.23928382 ± 0.00093454` | `0.05031408 ± 0.00012474` | `0.60135044 ± 0.00677120` |
| M1 | `0.23777698 ± 0.00297632` | `0.05050159 ± 0.00025168` | `0.62269269 ± 0.00792101` |
| M2 | `0.23765485 ± 0.00281485` | `0.05046880 ± 0.00025869` | `0.60543209 ± 0.00929694` |

## 12. Leakage / mechanism re-audit

The diagnostic repeats the same 64 Gate-0 states, 16 padded and 16 valid interventions per state, frozen source construction, and ten Euler steps. Gate-0's task-specific checkpoint and the Phase-1 base are different starting checkpoints, so their baseline ratios are contextual references rather than a training trajectory.

| Checkpoint / method | Padded RMS | Valid RMS | `R_leak` | Interpretation status |
|---|---:|---:|---:|---|
| Gate-0 task checkpoint | `0.01047023199` | `0.04374056309` | `0.05729682371` | Original mechanistic reference |
| Clean Phase-1 base, M0 diagnostic | `0.02350951359` | `0.29530790448` | `0.006337537430` | Clean base reference |
| Trained M0, mean over seeds | `0.03526504586` | `0.28202054898` | `0.01576039133` | 0/3 strong-suppression seeds |
| Trained M1, mean over seeds | `0.09008282423` | `0.28091188272` | `0.10302889595` | 0/3; no suppression |
| Trained M2, mean over seeds | `1.069772892e-7` | `0.28603221973` | `1.398908749e-13` | 3/3 strong-suppression seeds |

Per-seed `R_leak` was M0 `[0.01542064548, 0.01922211610, 0.01263841242]`, M1 `[0.10648344457, 0.09325581044, 0.10934743285]`, and M2 `[1.418707952e-13, 1.394648692e-13, 1.383369602e-13]`. Per-seed padded RMS was M0 `[0.03418526798, 0.03952090070, 0.03208896890]`, M1 `[0.09163200855, 0.08706348389, 0.09155298024]`, and M2 `[1.069230962e-7, 1.069999911e-7, 1.070087805e-7]`.

M2 therefore essentially eliminated measured padded-noise leakage and held operational padded-state RMS exactly zero at every recorded Euler state. M1 began with zero operational padded source but its padded state evolved away from zero; under the deliberately injected diagnostic perturbation, its leakage was larger than matched M0 rather than suppressed. The data support no claim about why M1 increased diagnostic sensitivity. Crucially, M2's strong mechanism suppression did not translate into a material validation gain.

## 13. Rollout evaluation

The frozen evaluation used every update-5,000 checkpoint: three methods × three training seeds. Each checkpoint ran all four suites, ten tasks/suite, and five episodes/task, giving 200 episodes/checkpoint and 1,800 total. The first five deterministic initialization states were used. The task seed formula was `540001 + suite_index*1000 + task_id*10`; episode seeds were task seed plus 0–4. Horizons were 280 (`spatial`), 280 (`object`), 300 (`goal`), and 520 (`libero_10`). The substrate used relative control, two 256×256 cameras, the standard 8D LIBERO state, and headless OSMesa rendering. No best-checkpoint selection or adaptive episode extension occurred. All nine rollout files passed structural integrity checks; early failed non-OSMesa launches recorded no counted episodes.

| Method | Successes / episodes | Seed success rates | Mean | Sample SD | Paired vs M0 |
|---|---:|---|---:|---:|---:|
| M0 | 290 / 600 | 49.0%, 46.5%, 49.5% | 48.33% | 1.61 pp | — |
| M1 | 269 / 600 | 48.5%, 41.5%, 44.5% | 44.83% | 3.51 pp | −3.50 pp; candidate favored 0/3 |
| M2 | 284 / 600 | 48.5%, 46.0%, 47.5% | 47.33% | 1.26 pp | −1.00 pp; candidate favored 0/3 |

M2 exceeded M1 by 2.50 percentage points on average and in 2/3 seeds (one tie), below the frozen 5-point supportive threshold. The pooled counts and Wilson intervals are descriptive only; the training seed, not the episode, is the training replicate.

| Suite | M0 mean | M1 mean | M2 mean |
|---|---:|---:|---:|
| `libero_spatial` | 39.33% | 41.33% | 45.33% |
| `libero_object` | 56.67% | 52.00% | 52.67% |
| `libero_goal` | 61.33% | 62.67% | 60.67% |
| `libero_10` | 36.00% | 23.33% | 30.67% |

Task-level pooled successes below aggregate the three training seeds, so every cell is successes out of 15 episodes. They are descriptive and were not used as independent replicates.

| Suite | Task ID | M0 | M1 | M2 |
|---|---:|---:|---:|---:|
| `libero_10` | 0 | 0/15 | 1/15 | 0/15 |
| `libero_10` | 1 | 3/15 | 3/15 | 3/15 |
| `libero_10` | 2 | 7/15 | 6/15 | 9/15 |
| `libero_10` | 3 | 11/15 | 7/15 | 9/15 |
| `libero_10` | 4 | 1/15 | 0/15 | 0/15 |
| `libero_10` | 5 | 11/15 | 6/15 | 9/15 |
| `libero_10` | 6 | 11/15 | 6/15 | 6/15 |
| `libero_10` | 7 | 1/15 | 1/15 | 1/15 |
| `libero_10` | 8 | 7/15 | 3/15 | 5/15 |
| `libero_10` | 9 | 2/15 | 2/15 | 4/15 |
| `libero_goal` | 0 | 7/15 | 6/15 | 7/15 |
| `libero_goal` | 1 | 13/15 | 13/15 | 13/15 |
| `libero_goal` | 2 | 12/15 | 11/15 | 10/15 |
| `libero_goal` | 3 | 9/15 | 6/15 | 7/15 |
| `libero_goal` | 4 | 9/15 | 8/15 | 7/15 |
| `libero_goal` | 5 | 7/15 | 8/15 | 9/15 |
| `libero_goal` | 6 | 3/15 | 6/15 | 3/15 |
| `libero_goal` | 7 | 15/15 | 14/15 | 15/15 |
| `libero_goal` | 8 | 11/15 | 11/15 | 10/15 |
| `libero_goal` | 9 | 6/15 | 11/15 | 10/15 |
| `libero_object` | 0 | 8/15 | 7/15 | 9/15 |
| `libero_object` | 1 | 7/15 | 2/15 | 4/15 |
| `libero_object` | 2 | 12/15 | 11/15 | 11/15 |
| `libero_object` | 3 | 7/15 | 8/15 | 4/15 |
| `libero_object` | 4 | 9/15 | 6/15 | 6/15 |
| `libero_object` | 5 | 9/15 | 6/15 | 6/15 |
| `libero_object` | 6 | 6/15 | 9/15 | 9/15 |
| `libero_object` | 7 | 11/15 | 14/15 | 12/15 |
| `libero_object` | 8 | 8/15 | 8/15 | 11/15 |
| `libero_object` | 9 | 8/15 | 7/15 | 7/15 |
| `libero_spatial` | 0 | 8/15 | 7/15 | 7/15 |
| `libero_spatial` | 1 | 5/15 | 6/15 | 7/15 |
| `libero_spatial` | 2 | 4/15 | 7/15 | 8/15 |
| `libero_spatial` | 3 | 10/15 | 7/15 | 9/15 |
| `libero_spatial` | 4 | 3/15 | 4/15 | 4/15 |
| `libero_spatial` | 5 | 3/15 | 2/15 | 2/15 |
| `libero_spatial` | 6 | 6/15 | 8/15 | 8/15 |
| `libero_spatial` | 7 | 9/15 | 10/15 | 11/15 |
| `libero_spatial` | 8 | 6/15 | 5/15 | 7/15 |
| `libero_spatial` | 9 | 5/15 | 6/15 | 5/15 |

No preregistered significance claim follows from these task cells. Rollout records contain success and reward fields but no separate termination-versus-horizon reason, an observability limitation noted in Section 17.

## 14. Final decision

The project's recorded evidence chain for **P2 — MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK** is:

1. Gate 0 established nonzero coupling from padded source noise to executable output.
2. The M2 intervention can suppress that measured pathway.
3. M2 reduced trained-checkpoint mean `R_leak` to `1.3989e-13` and passed strong suppression in 3/3 seeds.
4. Under matched deterministic post-training, M1 and M2 did not materially improve physical-action NRMSE or AULC over M0.
5. The frozen 1,800-episode rollout showed 48.33% M0, 44.83% M1, and 47.33% M2; neither intervention improved success over M0.
6. Thus the measured mechanism is real and controllable, but the tested intervention does not support a practical VLA-performance claim in this setting.

Not justified: padded dimensions materially degrade this training regime; M1 or M2 improves SmolVLA/LIBERO; Full-Subspace FM is superior to standard FM; rollout success was harmed causally by one specific pathway; or the mechanism/practical result generalizes across VLA families or benchmarks. Section 21 explains that the verbal evidence chain matches “mechanism suppression without practical gain,” while the exact P2 versus P5 label is internally inconsistent with the original decision tree.

## 15. Compute accounting

| Phase | Accounted device-hours | Status / source |
|---|---:|---|
| Gate 0A | `0.0175716` | Measured |
| Gate 0B | `0.0549897` | Measured, six runs |
| Gate 0 total | `0.0725613` | Measured |
| Original invalid Phase 1 | `17.4612380` | Exactly logged |
| Original invalid Phase 1, including reconstructed audit time | `18.2942380` | Accounted; not exact all-process total |
| Forensic audit | `0.0824591` | Accounted, including conservative failed-attempt estimate |
| Resume gate | `0.3152193` | Accounted |
| Clean rerun smoke reference | `0.4603645` | Clean-rerun component |
| Clean rerun smoke resume | `0.0713017` | Clean-rerun component |
| Clean rerun primary reported | `19.2767447` | Clean-rerun component |
| Clean rerun interrupted durable primary | `0.9799394` | Lower bound |
| Clean rerun primary spotchecks | `0.0747002` | Clean-rerun component |
| Clean rerun leakage re-audit | `0.3460363` | Clean-rerun component |
| Clean rerun rollout evaluation | `41.7082925` | Clean-rerun component |
| **Clean-rerun accounted total** | **`>=62.9173794`** | Defensible lower bound, not exact total |

The clean-rerun lower bound is at least `9.0826206` device-hours below the 72-hour ceiling. A short unmetered tail after the last durable trace for interrupted M0/M1 seed-520003 runs, plus method-integrity setup time, prevents an honest exact all-inclusive clean-rerun total. Phase totals above must not be summed and presented as an exact whole-project number because two phases themselves contain reconstructed/lower-bound components.

## 16. Reproducibility / engineering contributions

- Deterministic CUDA execution was pinned with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch/cuDNN settings, benchmark and TF32 disabled, one process/GPU, and bf16 autocast without loss scaling.
- Complete checkpoints preserved the full model and AdamW state, scheduler/AMP state, update and accumulation position, materialized sampler offset, every relevant RNG state and named generator, mode, configuration, and provenance.
- Resume equivalence was checked on batches, processed tensors, flow time, Gaussian noise, loss, gradients, parameters, optimizer tensors, scheduler, and AMP—not only on final loss.
- The clean run matrix used persistent tmux/session-based execution so jobs could survive chat/client availability; process interruptions were resumed only from complete checkpoints after exact trace checks.
- The final clean artifact manifest contains 28 SHA-256 entries, including all nine rollout files. A review-time hash recomputation found 28/28 matches.
- Scientific milestones and protocol freezes were committed in Git; large raw logs/checkpoints remained outside Git. The repository was clean before this documentation-only bundle was created.

These are meaningful reproducibility controls for interpreting the experiment, not a claim of algorithmic novelty.

## 17. Limitations

- One VLA family, one SmolVLA base checkpoint, and one internal/executable geometry (`32D→7D`) were tested.
- Evaluation is LIBERO-only; there is no cross-model, robot, action-interface, or real-world validation.
- Only three independent training seeds and one fixed data/update regime were used.
- The train split was 320 episodes/53,762 frames, and the conclusion is specific to 5,000-update downstream adaptation.
- Rollout used five episodes/task/checkpoint; training seed—not pooled episodes—is the independent model-training replicate.
- M2 differs from M1 only during iterative inference because official single-time FM training gives them identical training objectives and final checkpoints. The experiment does not isolate a distinct M2 training effect.
- The practical result is negative: strong diagnostic suppression did not improve validation or rollout over M0.
- Rollout artifacts record success/reward but not an explicit termination-reason field, so success-versus-horizon is visible but the exact non-success termination category is not.
- Exact all-inclusive clean device time is unavailable; `62.9173794` device-hours is a lower bound.
- The forensic audit isolated deterministic backward execution as the effective remedy but did not name one offending CUDA kernel.
- The clean base and Gate-0 task checkpoint are different checkpoints; their leakage ratios should not be presented as a simple before/after training pair.
- Two documentation-level inconsistencies concerning the metric formula and final verdict label are recorded in Section 21.

## 18. What is publishable / resume-safe to claim

### Safe claims

- In the audited SmolVLA/LeRobot path, 25 padded dimensions participate in a 32D flow although LIBERO executes seven action coordinates.
- On 64 frozen Gate-0 states, padded-source perturbations caused nonzero changes in executable predictions: RMS `0.0104702`, `R_leak=0.0572968`, with exact repeat error zero.
- M2's per-step projection reduced the trained-checkpoint diagnostic leakage ratio to mean `1.3989e-13` and kept padded state at zero.
- M1's source-only mask did not suppress the diagnostic in the same way; trained M1 mean `R_leak` was `0.103029`.
- In the deterministic clean rerun, neither M1 nor M2 materially improved physical NRMSE, AULC, or rollout success over M0.
- M2 was directionally better than M1 in final NRMSE in 3/3 seeds, but only by `0.358479%`, inside the frozen equivalence band.
- The final practical result is negative under the tested SmolVLA/LIBERO/data/budget setting.
- The project records its verdict as P2, subject to the decision-tree inconsistency disclosed in Section 21.

### Unsafe / unsupported claims

- “FlowSpec/SpecFlow improves SmolVLA,” “M2 improves LIBERO,” or “M1/M2 improve rollout success.”
- “Padded dimensions hurt rollout performance” or “Gate 0 proves task-level degradation.”
- “Continued padded flow is the dominant cause of downstream error.”
- “Source masking improves training” or “M1 suppresses leakage.”
- “The mechanism generalizes to all VLAs/action interfaces.”
- “Normalization is a strong causal factor.”
- Any final claim or effect size derived from the invalid original Phase-1 checkpoints.
- Any p-value or claim that treats frames, validation states, tasks, or rollout episodes as independent training replicates.
- Any claim about a new method, architecture change, learned mask, auxiliary regularizer, RL method, cross-model result, or unrun Phase 2.
- An exact all-inclusive device-hour total.

## 19. Final canonical numbers table

This table is the anti-mismatch reference. “Final-valid” means usable for the stated role; diagnostics are not automatically primary performance evidence.

| Metric | Canonical value | Phase | Primary / secondary / diagnostic | Source | Final-valid? |
|---|---:|---|---|---|---|
| Executable / internal / padded dimensions | `7 / 32 / 25` | Audit | Implementation fact | `GATE0_REPORT.md`, `config/phase1_rerun.yaml` | Yes |
| Chunk length / Euler steps | `50 / 10` | Audit | Implementation fact | `config/phase1_rerun.yaml` | Yes |
| Gate-0 states / tasks | `64 / 4` | Gate 0A | Diagnostic design | `gate0a_summary.json` | Yes |
| Gate-0 `K_pad / K_valid` | `16 / 16` | Gate 0A | Diagnostic design | `gate0a_summary.json` | Yes |
| Gate-0 padded RMS | `0.010470231994986534` | Gate 0A | Diagnostic | `gate0a_summary.json` | Yes |
| Gate-0 padded RMS 95% CI | `[0.009601770900189877, 0.011180084198713303]` | Gate 0A | Diagnostic | `gate0a_summary.json` | Yes |
| Gate-0 valid RMS | `0.043740563094615936` | Gate 0A | Diagnostic | `gate0a_summary.json` | Yes |
| Gate-0 `R_leak` | `0.05729682371020317` | Gate 0A | Diagnostic | `gate0a_summary.json` | Yes |
| Gate-0 repeat RMS / max | `0 / 0` | Gate 0A | Control | `gate0a_summary.json` | Yes |
| Gate-0 raw physical padded RMS | `0.0030263536609709263` | Gate 0A | Diagnostic | `gate0a_summary.json` | Yes |
| Gate-0 verdict | `STRONG GO` (`A1_STRONG_SIGNAL` in 0A) | Gate 0 | Decision | `GATE0_REPORT.md`, `gate0a_summary.json` | Yes |
| Gate-0B transforms / seeds / updates | `3 / 2 / 64 each` | Gate 0B | Diagnostic design | `gate0b_summary.json` | Yes |
| Gate-0B anchors / validation states | `128 / 32` | Gate 0B | Diagnostic design | `gate0b_summary.json` | Yes |
| Gate-0B total updates / verdict | `384 / B2_WEAK_SIGNAL` | Gate 0B | Diagnostic | `gate0b_summary.json` | Yes |
| Invalid M1/M2 final max differences | `0.02718, 0.03040, 0.02802` | Old Phase 1 | Invalidity evidence | `PHASE1_REPORT.md` | Historical only |
| Invalid same-device update-1000 max difference | `0.02034` | Old Phase 1 | Invalidity evidence | `PHASE1_REPORT.md` | Historical only |
| Frozen reproducibility tolerance | `<1e-6` | Phase 1 | Gate threshold | `PHASE1_PROTOCOL.md` | Yes |
| Forensic first gradient norms | `20.3197994 / 20.3206997` | Forensic | Diagnostic | `FORENSIC_REPORT.md` | Yes |
| Forensic 5-update max / mean / L2 | `7.6293945e-6 / 1.8547922e-9 / 0.0013746294` | Forensic | Diagnostic | `FORENSIC_REPORT.md` | Yes |
| Resume span | checkpoint 50 → update 100 | Resume gate | Engineering gate | `resume_comparison.json` | Yes |
| Resume final model differences | max/mean/L2 `0/0/0` | Resume gate | Engineering gate | `resume_comparison.json` | Yes |
| Resume model / optimizer tensors | `500 / 1,467` | Resume gate | Engineering gate | `resume_comparison.json` | Yes |
| Train / held-out episodes | `320 / 200` | Clean rerun | Data design | `config/phase1_rerun.yaml` | Yes |
| Train frames / validation states | `53,762 / 400` | Clean rerun | Data design | `validation_summary.json` | Yes |
| Methods × seeds / runs | `3 × 3 / 9` | Clean rerun | Design | `validation_summary.json` | Yes |
| Seed IDs | `520001, 520002, 520003` | Clean rerun | Design | `config/phase1_rerun.yaml` | Yes |
| Updates / examples per run | `5,000 / 160,000` | Clean rerun | Design | `training_summary.json` | Yes |
| M0 final NRMSE | `0.7137868103 ± 0.0009258602` | Clean rerun | **Primary** | `validation_summary.json` | Yes |
| M1 final NRMSE | `0.7169568478 ± 0.0048452483` | Clean rerun | **Primary** | `validation_summary.json` | Yes |
| M2 final NRMSE | `0.7143867055 ± 0.0049477651` | Clean rerun | **Primary** | `validation_summary.json` | Yes |
| M1 vs M0 final relative reduction | `-0.444115%`; favors 1/3 | Clean rerun | Primary comparison | `validation_summary.json` | Yes |
| M2 vs M0 final relative reduction | `-0.084044%`; favors 1/3 | Clean rerun | Primary comparison | `validation_summary.json` | Yes |
| M2 vs M1 final relative reduction | `+0.358479%`; favors 3/3 | Clean rerun | Primary comparison | `validation_summary.json` | Yes |
| M0 / M1 / M2 AULC | `0.78191985 / 0.78581313 / 0.78369518` | Clean rerun | Primary efficiency | `validation_summary.json` | Yes |
| M2 vs M1 AULC reduction | `+0.269523%`; favors 3/3 | Clean rerun | Primary efficiency | `validation_summary.json` | Yes |
| Material / AULC / direction thresholds | `5% / 3% / 2 of 3` | Clean rerun | Frozen criteria | `config/phase1_rerun.yaml` | Yes |
| Equivalence band | `±2%` | Clean rerun | Frozen criterion | `config/phase1_rerun.yaml` | Yes |
| Clean base padded / valid RMS | `0.02350951359 / 0.29530790448` | Leakage | Diagnostic | `leakage_reaudit_summary.json` | Yes |
| Clean base `R_leak` | `0.006337537430` | Leakage | Diagnostic | `leakage_reaudit_summary.json` | Yes |
| Trained M0 padded / valid RMS | `0.03526504586 / 0.28202054898` | Leakage | Diagnostic means | `leakage_reaudit_summary.json` | Yes |
| Trained M0 `R_leak` | `0.01576039133` | Leakage | Diagnostic | `leakage_reaudit_summary.json` | Yes |
| Trained M1 padded / valid RMS | `0.09008282423 / 0.28091188272` | Leakage | Diagnostic means | `leakage_reaudit_summary.json` | Yes |
| Trained M1 `R_leak` | `0.10302889595` | Leakage | Diagnostic | `leakage_reaudit_summary.json` | Yes |
| Trained M2 padded / valid RMS | `1.069772892e-7 / 0.28603221973` | Leakage | Diagnostic means | `leakage_reaudit_summary.json` | Yes |
| Trained M2 `R_leak` | `1.398908749e-13` | Leakage | Diagnostic | `leakage_reaudit_summary.json` | Yes |
| Strong mechanism floors | RMS `<=1e-4`; ratio `<=1e-4` | Leakage | Frozen criterion | `config/phase1_rerun.yaml` | Yes |
| Rollout checkpoints / episodes | `9 / 1,800` | Rollout | Secondary design | `rollout_summary.json` | Yes |
| Episodes/checkpoint / task | `200 / 5` | Rollout | Secondary design | `rollout_summary.json` | Yes |
| M0 rollout | `290/600 = 48.3333%`; seed mean SD `1.6073 pp` | Rollout | Secondary | `rollout_summary.json` | Yes |
| M1 rollout | `269/600 = 44.8333%`; seed mean SD `3.5119 pp` | Rollout | Secondary | `rollout_summary.json` | Yes |
| M2 rollout | `284/600 = 47.3333%`; seed mean SD `1.2583 pp` | Rollout | Secondary | `rollout_summary.json` | Yes |
| M1 vs M0 rollout | `−3.50 pp`; favors 0/3 | Rollout | Secondary comparison | `rollout_summary.json` | Yes |
| M2 vs M0 rollout | `−1.00 pp`; favors 0/3 | Rollout | Secondary comparison | `rollout_summary.json` | Yes |
| M2 vs M1 rollout | `+2.50 pp`; favors 2/3 | Rollout | Secondary comparison | `rollout_summary.json` | Yes |
| Frozen supportive rollout threshold | `+5 pp` in 2/3 seeds | Rollout | Frozen criterion | `config/phase1_rerun.yaml` | Yes |
| Clean-rerun compute | `>=62.9173794 device-hours` | Clean rerun | Accounting | `gpu_usage.json` | Yes, lower bound |
| Rollout compute | `41.7082925 device-hours` | Rollout | Accounting | `rollout_summary.json` | Yes |
| Project-declared verdict | `P2 — MECHANISM CONFIRMED, PRACTICAL BENEFIT WEAK` | Final | Decision | `phase1_rerun_decision.json` | Yes, label conflict noted |

## 20. Source index

### Authoritative repository documents

- `/mnt/NAS/data/hl5757/projects/flowspec-vla/GATE0_PROTOCOL.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/GATE0_REPORT.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/PHASE1_PROTOCOL.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/PHASE1_REPORT.md` — invalid historical experiment only
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/FORENSIC_PROTOCOL.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/FORENSIC_REPORT.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/RESUME_GATE_PROTOCOL.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/RESUME_GATE_REPORT.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/PHASE1_RERUN_PROTOCOL.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/PHASE1_RERUN_RECOVERY.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/PHASE1_RERUN_REPORT.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/EXPERIMENT_STATE.md`
- `/mnt/NAS/data/hl5757/projects/flowspec-vla/README.md`
- Git log through scientific-record commit `1acd919f03f55a93f821ac9a13872c8b6ce655ec`.

### Config and code

- `config/gate0a.yaml`, `config/gate0b.yaml`, `config/phase1.yaml`, and `config/phase1_rerun.yaml`.
- `src/flowspec_vla/phase1.py`, `src/flowspec_vla/data.py`, `src/flowspec_vla/forensic.py`, and `src/flowspec_vla/resume_gate.py`.
- Critical scripts: `scripts/run_gate0a.py`, `scripts/analyze_gate0a.py`, `scripts/run_gate0b.py`, `scripts/analyze_gate0b.py`, `scripts/verify_phase1_implementation.py`, `scripts/run_phase1_training.py`, `scripts/analyze_phase1_training.py`, `scripts/run_forensic_fresh.py`, `scripts/analyze_forensic_fresh.py`, `scripts/run_resume_gate.py`, `scripts/analyze_resume_gate.py`, `scripts/run_phase1_rerun_training.py`, `scripts/run_phase1_leakage.py`, `scripts/analyze_phase1_leakage.py`, `scripts/run_phase1_rollout.py`, `scripts/analyze_phase1_rollouts.py`, and `scripts/finalize_phase1_rerun.py`.
- Upstream: `upstream/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`, `configuration_smolvla.py`, and `upstream/lerobot/src/lerobot/policies/common/flow_matching.py`.

### Critical machine-readable evidence

- Gate 0: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0/gate0a_summary.json`, `gate0b_summary.json`, `state_manifest.json`, `environment.json`, and `artifact_manifest.json`.
- Invalid Phase 1: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1/reproducibility_audit_summary.json`, `training_summary.json`, `validation_summary.json`, `rollout_summary.json`, `gpu_usage.json`, and `artifact_manifest.json`—provenance/invalidity only.
- Forensic: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/forensic/first_divergence.json`, `fresh_run_comparison.json`, `one_step_replay_default.json`, `one_step_replay_deterministic.json`, `fix_verification.json`, `gpu_usage.json`, and `artifact_manifest.json`.
- Resume: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/resume_gate/resume_comparison.json`, `checkpoint_state_inventory.json`, `optimizer_state_comparison.json`, `rng_state_comparison.json`, `sampler_state_comparison.json`, `gpu_usage.json`, and `artifact_manifest.json`.
- Clean rerun: `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun/protocol_snapshot.json`, `method_integrity_summary.json`, `smoke_gate_summary.json`, `baseline_reproduction_summary.json`, `training_summary.json`, `validation_summary.json`, `seed_pairing_summary.json`, `reproducibility_spotcheck.json`, `m1_m2_training_invariance.json`, `leakage_reaudit_summary.json`, `rollout_summary.json`, `gpu_usage.json`, `phase1_rerun_decision.json`, and `artifact_manifest.json`.
- Seed-level rollout evidence: all nine JSON files under `/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/phase1_rerun/rollouts/`.

The review corpus comprised **129 source artifacts** under a reproducible counting rule: 55 tracked Markdown/config/project-code files, three pinned upstream source files, 62 top-level machine-readable JSON artifacts across the five project phases, and nine clean-rerun rollout JSONs. Large binary checkpoints/NPZ tensors and plots were checked through recorded manifests and summary provenance rather than counted as separately interpreted source documents.

## 21. Internal consistency audit

Repeated headline results were cross-checked across reports, configs, JSON summaries, all nine rollout files, and manifests. Final NRMSE seed values and means match `validation_summary.json`; leakage values match both `leakage_reaudit_summary.json` and its byte-identical `leakage_audit_summary.json` alias; rollout seed/suite/task totals sum to 1,800 and match `rollout_summary.json`; compute components sum to the recorded lower bound. The final clean manifest has 28 entries, and review-time SHA-256 recomputation produced 28/28 matches.

Two authoritative-looking documentation conflicts remain:

1. **Final verdict label conflict — unresolved.** The original frozen `PHASE1_PROTOCOL.md` lines 252–256 defines P2 as requiring strong/partial mechanism suppression **and** at least one intervention with a positive mean final-NRMSE or AULC reduction of at least 2%; otherwise, when improvements are absent/below 2% or direction fails, it defines P5. Clean results show M1 and M2 both worse than M0 in mean final NRMSE and AULC, and only 1/3 paired seeds favors either intervention. A literal first-match application therefore appears to select P5. Nevertheless, the clean report, experiment state, and machine-readable decision artifact all declare P2, and the user's completion record explicitly fixes P2. Provenance does not contain a documented amendment that resolves this mismatch. This bundle preserves the official project-declared **P2** label but requires any technical report to disclose that its literal preregistered label assignment is inconsistent; the evidence-level conclusion “mechanism suppressed, no practical benefit” is unaffected.

2. **NRMSE formula text conflict — resolved operationally, documentation error remains.** Original `PHASE1_PROTOCOL.md` lines 184–188 defines coordinate-standardized physical RMSE. `PHASE1_RERUN_PROTOCOL.md` lines 138–142 displays a different global ratio `sqrt(mean(error^2))/sqrt(mean(target^2))` while simultaneously saying aggregation is “exactly as in the original protocol.” The implementation in `src/flowspec_vla/phase1.py` lines 270–280 and `scripts/run_phase1_training.py` lines 121–148 divides each coordinate error by the frozen action standard deviation, matching the original protocol; durable reconstruction in `scripts/run_phase1_rerun_training.py` lines 153–166 does the same. All reported `0.71379/0.71696/0.71439` values use this operational/original definition. Therefore the displayed rerun formula is not the metric used and should not be reproduced as the canonical formula.

Other apparent differences are resolved by provenance:

- Gate-0 `R_leak=0.05730` and clean-base `R_leak=0.00634` refer to different checkpoints, not contradictory measurements.
- Exactly logged original Phase-1 compute (`17.4612`) and accounted original compute (`18.2942`) differ because the latter includes reconstructed interrupted-audit time.
- Clean-rerun `62.9174` is explicitly a lower bound, not an exact total.
- Early fixed-flow MSE entries for two interrupted runs are unavailable; physical validation was reconstructed from durable prediction/target arrays and no missing flow metric was fabricated.
- OSMesa was the common successful headless renderer; failed earlier launches contributed no counted episodes.

No quantitative final-result mismatch was found among the clean summary JSONs and their seed-level rollout/checkpoint evidence.

## 22. Bundle quality and evidence boundary

- Final method/performance evidence comes only from the deterministic clean rerun.
- The original Phase-1 attempt is quarantined and appears only as invalidity/forensic history.
- Preregistered primary quantities are separated from rollout, task pooling, mechanism diagnostics, and engineering checks.
- Training seeds are the independent experimental replicates; no sample-, state-, task-, or episode-level pseudoreplication is used for method conclusions.
- No new hypothesis, method, experiment, novelty claim, significance claim, or exact unrecorded value has been added.
- The appropriate report tone is a clean negative practical result with a verified mechanistic observation, plus transparent disclosure of the P2/P5 labeling inconsistency.
- Resume boundary: the project is complete and stopped. Further methods, larger training, cross-model validation, benchmark expansion, or Phase 2 require new authorization.
