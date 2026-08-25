# Frozen Gate-0 Protocol

Protocol freeze date: 2026-08-25. This file must be committed before any final
Gate-0 intervention output is inspected. Engineering smoke tests may establish
that loading, decoding, and tensor shapes work; they may not be used to alter
the selections, transformations, metrics, or thresholds below.

## Frozen substrate

- LeRobot source: `https://github.com/huggingface/lerobot.git` at
  `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Checkpoint: `lerobot/smolvla_libero` at
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- VLM backbone: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Demonstrations: `lerobot/libero` payload at
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; language mapping only is
  recovered from parent `1595a93b43aa055e55c127a4f0b4a99bb8035447`.
- `d_valid=7`, `D=32`, chunk length 50, 10 Euler flow steps, native unit
  Gaussian source, checkpoint MEAN_STD action coordinates.

## Gate 0A: frozen null-dimension intervention

No training is permitted. Use exactly four preregistered task indices, chosen
before model outputs: `0`, `10`, `20`, `30`. They provide one task from each of
LIBERO-10, goal, object, and spatial in the dataset's frozen mapping. For each
task use its lowest numbered episode. Within that episode select 16 indices
`floor(j * (L-1) / 17)` for `j=1,...,16`, giving 64 states without inspecting
model behavior.

For state index `s`, generate `E_s` of shape `[16, 50, 32]` from the policy's
native float32 N(0,1) distribution with deterministic seed `410000+s`. Define:

- pad intervention: valid coordinates `0:7` fixed from `E_s[0]`, padded
  coordinates `7:32` varied over all 16 draws;
- valid positive control: valid coordinates varied, padded coordinates fixed
  from `E_s[0]`;
- native/all diagnostic: all coordinates varied as `E_s`;
- repeat control: the identical full `E_s[0]` is evaluated twice;
- zero-pad diagnostic: compare `E_s[0]` with a source having the same valid
  coordinates and exactly zero coordinates `7:32`.

No source coordinate is rescaled. Observations, language, state, checkpoint,
precision, Euler schedule, and all model randomness are fixed. Use batches only
as a computational implementation detail. Save output after every Euler step
when practical; failure of tracing must not change the primary final-output
test. Jacobians are optional and will be omitted if they threaten the bounded
budget.

### Gate 0A metrics and unit of analysis

Convert valid predictions to physical dataset control coordinates with the
checkpoint inverse transform. Scale deviations dimensionwise by the checkpoint
action standard deviation for cross-coordinate aggregation while retaining raw
physical-coordinate results.

For each independent state, compute sample variance across K in every
`[time,valid-dimension]` cell, average those cell variances to obtain `V_pad(s)`
and `V_valid(s)`, and report `RMS_pad(s)=sqrt(V_pad(s))` and its valid-control
counterpart. Primary aggregate:

`R_leak = median_s V_pad(s) / median_s V_valid(s)`.

Also report median and bootstrap 95% state-level confidence intervals (10,000
bootstrap resamples, seed 420001), per-dimension variance/RMS, raw executable L2
variation, maximum pairwise difference, task distributions, stepwise effects,
zero-pad effect, and exact-repeat maximum absolute/RMS discrepancy. States, not
dimensions, chunks, or denoising steps, are the independent bootstrap units.

The implementation is valid only if median valid-control RMS is at least
`1e-4` standardized units and repeat-control maximum absolute discrepancy is at
most `1e-5` standardized units. Otherwise diagnose once; if the official path
is still degenerate, Mechanism A is substrate-invalid rather than positive.

Frozen decision thresholds:

- **A1 STRONG SIGNAL:** median standardized `RMS_pad >= 0.01`,
  `R_leak >= 0.01`, and the bootstrap lower bound for median RMS exceeds
  `max(10 * repeat_RMS, 1e-4)`.
- **A2 WEAK SIGNAL:** not A1, but median standardized `RMS_pad` exceeds
  `max(10 * repeat_RMS, 1e-4)` and either `R_leak >= 1e-4` or the bootstrap
  lower bound exceeds that floor.
- **A3 NO MATERIAL SIGNAL:** valid implementation and neither prior condition.

If A3, the padding branch terminates with no rescue tuning.

## Gate 0B: matched coordinate pilot

Run this regardless of Gate 0A unless the substrate is invalid. The physical
action `a` is represented in exactly three preregistered, invertible coordinate
systems:

1. `official`: `y=(a-mu_ckpt)/(sigma_ckpt+1e-8)`;
2. `local`: `y=(a-mu_local)/(sigma_local+1e-8)`, with population mean/std
   computed once from all physical target cells in the frozen training chunks;
3. `group_scaled`: `y=S*(a-mu_ckpt)/(sigma_ckpt+1e-8)`, where
   `diag(S)=[0.5,0.5,0.5,2,2,2,1]`.

All inverses must recover held-out physical actions with maximum absolute error
at most `1e-6` before training. Padded target coordinates remain exactly zero.
No further transform may be tried.

Use the same four task indices. For each task, use its first four episodes for
training and its fifth episode for validation. In each training episode select
eight anchors `floor(j*(L-50)/9)`, `j=1,...,8`; in each validation episode
select eight anchors by the same rule. This freezes 128 training anchors and 32
held-out validation anchors, each with its following 50-action target chunk.

Every run starts from identical checkpoint bytes. Use all and only parameters
marked trainable by the checkpoint/current official configuration; do not add an
adapter or alter the architecture. Use one seed (`430001`), the same fixed
permutation, batch size 1, exactly 64 optimizer updates, AdamW at constant
`1e-4` with betas `(0.9,0.95)`, eps `1e-8`, weight decay `1e-10`, gradient-norm
clip 10, and BF16 autocast with float32 loss. Precompute matched native flow
noise and Beta timesteps per update so all coordinates receive the same base
random numbers before their deterministic coordinate transformation. The
observation/state/language checkpoint preprocessing remains identical; only the
valid action target coordinate map and its inverse differ.

Log every update's raw flow MSE, physically transformed velocity MSE, pre-clip
gradient norm, learning rate, and elapsed/GPU time. Evaluate teacher-forced
validation flow loss on the same frozen noise/times at updates 0, 16, 32, 48,
and 64. Generate physical action chunks with one fixed native source per state
at updates 0 and 64 and report pooled, groupwise (`0:3`, `3:6`, `6`), and
chunk-time physical RMSE. Comparisons of raw normalized losses are descriptive;
the decision uses matched physical-coordinate error. Bootstrap paired validation
states 10,000 times with seed `440001`. A second training seed is omitted unless
all three primary runs leave enough of the total two-GPU-hour cap; a second seed,
if run, must cover all three transforms symmetrically and cannot change the
thresholds.

For each non-official convention, define final relative physical RMSE difference
`Delta=(RMSE_variant-RMSE_official)/RMSE_official`.

Frozen decision thresholds:

- **B1 STRONG SIGNAL:** at least one `abs(Delta) >= 0.10`, its paired bootstrap
  95% interval excludes the material-neutral band `[-0.02,0.02]`, and the
  physicalized validation-loss trajectory differs in the same qualitative
  direction at a majority of post-baseline checkpoints.
- **B2 WEAK SIGNAL:** not B1, but at least one `abs(Delta) >= 0.02` or a paired
  bootstrap interval excludes zero; this explicitly includes a large but
  heterogeneous or one-seed-sensitive difference.
- **B3 NO MATERIAL SIGNAL:** neither B1 nor B2.

If B3, the coordinate branch terminates. No transform or optimization setting
may be added after results.

## Overall decision and budget

- **STRONG GO** iff A1 or B1.
- **SIGNAL EXISTS BUT WEAK** iff no strong signal and A2 or B2.
- **NO-GO — ACTION INTERFACE EFFECT NOT MATERIAL** iff A3 and B3.
- **NO-GO — SUBSTRATE / IMPLEMENTATION INVALID** iff a clean test cannot be
  performed on the official substrate after one documented diagnosis.

Total Gate 0B training may not exceed two aggregate GPU-hours. Gate 0A inference
and compatibility smoke tests are recorded separately. Stop after the Gate-0
report; do not implement a method.
