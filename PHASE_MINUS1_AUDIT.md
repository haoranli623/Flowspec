# Phase -1 Substrate Audit

Audit date: 2026-08-25

This document records the source path before any Gate-0 mechanism result was
computed. The audited implementation is the current official Hugging Face
LeRobot repository at `https://github.com/huggingface/lerobot.git`, commit
`8b256a6c0d4769c3cc3e7e98f04940126398a391`. The frozen capable checkpoint is
the official Hub repository `lerobot/smolvla_libero`, revision
`31d453f7edd78c839a8bbc39744a292686daf0de`. It names `lerobot/smolvla_base` as
its base and `lerobot/libero` as its training dataset.

## End-to-end action path

The checkpoint declares a seven-dimensional LIBERO action, a 50-step action
chunk, 50 action steps, and a 32-dimensional internal action interface. The
dataset payload contains seven-dimensional actions and eight-dimensional
states. The checkpoint config declares a six-dimensional state and three image
features while the dataset has an eight-dimensional state and two images. The
current implementation is shape-tolerant here: it pads the actual state tensor
to 32, processes the present expected images, and does not synthesize missing
cameras when `empty_cameras=0`. This mismatch is recorded and is not repaired
by changing the observation.

For both STATE and ACTION, checkpoint processing uses `MEAN_STD`. The saved
normalizer implements `(x - mean) / (std + 1e-8)`, and the postprocessor uses
`x * std + mean`. Both processor state files contain checkpoint statistics from
273,465 frames. Thus the checkpoint statistics, rather than recomputed current
dataset statistics, are authoritative during normal evaluation. Replacing the
action statistics during fine-tuning changes the coordinates in which the
standard-normal flow source and velocity objective are defined.

After normalization, `SmolVLAPolicy.prepare_action` right-pads each physical
seven-vector with zeros to `max_action_dim=32`. Training samples float32
standard-normal noise with the same `[batch, 50, 32]` shape. It samples
`t = Beta(1.5, 1.0) * 0.999 + 0.001`, constructs
`x_t = t * noise + (1-t) * action`, and targets `u_t = noise - action`.
`action_in_proj` is a shared dense `Linear(32, expert_hidden_size)`; the action
tokens enter the shared flow expert, and `action_out_proj` emits 32 velocities.
The core computes elementwise MSE for all 32 coordinates, but the policy wrapper
slices the loss to the original seven action coordinates before applying the
temporal `action_is_pad` mask and reduction. Therefore invalid coordinates are
absent from the direct supervised loss but present in the noisy generative
input.

At inference, the same policy samples a float32 standard-normal
`[batch, 50, 32]` source. It caches the observation/language prefix and performs
10 forward-Euler updates from `t=1` to `t=0` with `dt=-0.1`. All 32 action
coordinates evolve through the shared dense input projection and action expert.
Only after integration does `_get_action_chunk` slice the output to seven
coordinates. Consequently, current official code permits mathematical influence
from padded source coordinates to valid predictions; whether that influence is
measurable is left to Gate 0A.

## Checkpoint normalization statistics

Checkpoint action mean:

`[0.062781565, 0.086840808, -0.090373062, 0.000540743,
  0.005643380, -0.005229099, -0.049640723]`

Checkpoint action standard deviation:

`[0.335523725, 0.378446996, 0.444728613, 0.039243542,
  0.063392967, 0.077970274, 0.998767138]`

LIBERO actions are seven relative-control values. This audit does not attach
unverified metric-unit semantics to the first six coordinates; reports use the
dataset's coordinate groups `0:3`, `3:6`, and gripper `6`, plus the exact raw
physical control coordinates after checkpoint unnormalization.

## Official data and evaluation path

The frozen dataset payload is `lerobot/libero` revision
`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`: 1,693 episodes, 273,465 frames,
40 tasks, 10 Hz, two 256x256 AV1 camera streams, eight state coordinates, and
seven action coordinates. The current official `tasks.parquet` has a metadata
regression: its task-language index is lost. Its parent revision
`1595a93b43aa055e55c127a4f0b4a99bb8035447` retains the 40 language strings.
Experiments use the current payload and recover only that label mapping from the
parent; pixels, states, actions, timestamps, and indices are unchanged.

The checkpoint training config specifies official `libero` environment support,
relative control, pixel observations, initialized states, 50 evaluation
episodes in one non-async batch, and `eval_freq=0` during its 25,000-step
training. Gate 0 is an offline mechanism test on official demonstrations, not a
claim about simulator success rate, so no environment or task is modified.

## Audit conclusion

The proposed interventions are valid on the current official substrate. Padding
exists (`d_valid=7`, `D=32`), invalid dimensions traverse shared generative
computation, and affine normalization changes the coordinate system carrying an
isotropic Gaussian source. This establishes testability only, not an empirical
effect.
