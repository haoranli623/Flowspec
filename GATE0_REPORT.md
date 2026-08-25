# FlowSpec-VLA Gate-0 Report

Date: 2026-08-25

## 1. Executive verdict

**STRONG GO.** Mechanism A met the frozen **A1 STRONG SIGNAL** criterion:
native stochastic variation confined to 25 non-executable padded coordinates
changed the seven executable predictions at about 1.05% of checkpoint action
standard deviation RMS, with a padded/valid variance ratio of 5.73% and zero
repeat error. Mechanism B is **B2 WEAK SIGNAL** after a budget-permitted second
seed: coordinate-dependent post-training differences were large in the primary
seed but not seed-robust. This is a mechanism result, not a policy improvement or
a novelty claim.

## 2. Exact substrate and revisions

- Official source: `https://github.com/huggingface/lerobot.git`, commit
  `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- Official checkpoint: `lerobot/smolvla_libero`, revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Backbone: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Official data payload: `lerobot/libero`, revision
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- Task strings recovered from the payload's parent revision
  `1595a93b43aa055e55c127a4f0b4a99bb8035447` because the current
  `tasks.parquet` accidentally lost its index. No pixels, states, actions, or
  indices were changed.
- Environment: Python 3.12, LeRobot 0.6.2 editable at the commit above,
  PyTorch 2.7.1+cu118, Transformers 5.5.4; two RTX 3090 24 GB GPUs.
- Protocol-freeze commit:
  `337c95305698e527ef09db6f53040fb377a9d6a5`.

## 3. Implementation audit

The official checkpoint uses a seven-dimensional LIBERO action
(`d_valid=7`), a 50-step chunk, a 32-dimensional fixed internal action
interface (`D=32`), and 10 Euler denoising steps. Its checkpoint processor uses
MEAN_STD action and state normalization and stores statistics for 273,465
frames. Action normalization is `(a-mean)/(std+1e-8)`; evaluation applies
`y*std+mean` after inference.

Training pads the normalized seven-vector with 25 zeros, samples float32
standard-normal noise for all 32 coordinates, samples
`t=Beta(1.5,1)*0.999+0.001`, forms `x_t=t*noise+(1-t)*action`, and targets
`noise-action`. A shared dense `Linear(32,h)` embeds the entire noisy action.
The flow expert emits 32 velocities and elementwise MSE, after which the policy
wrapper slices the supervised loss to the original seven dimensions and applies
the temporal padding mask.

Inference initializes all 32 coordinates from N(0,1), evolves all 32 through 10
Euler updates, and slices to seven only after sampling. Because the invalid
coordinates enter the same dense action projection and shared expert, they can
mathematically affect valid outputs. The code audit therefore validated the
premise but did not assume the magnitude.

The current data/checkpoint metadata has two further discrepancies: the payload
state has eight coordinates while checkpoint metadata declares six, and the
payload has two cameras while checkpoint metadata declares three. The official
model path accepts the actual eight-vector, pads it to 32, and processes the two
present cameras; the compatibility smoke test loaded 450,046,176 parameters and
used the unmodified observations.

The checkpoint training config names `lerobot/libero`, relative control, pixel
observations, 25,000 updates, and an official evaluation configuration of 50
episodes in one non-async batch, although evaluation was disabled during that
training run (`eval_freq=0`). Gate 0 deliberately used offline demonstrations,
not simulator success.

## 4. Frozen protocol

`GATE0_PROTOCOL.md` was committed before final mechanism output. It fixed the
source/checkpoint/data revisions; task indices `[0,10,20,30]`; deterministic
state and episode selection; 16 native source draws per intervention; exact
controls; physical and standardized metrics; 10,000 state bootstraps; three
coordinate systems; 64 updates; optimizer, precision, seeds, and all numerical
decision thresholds. Neither the protocol nor transforms were changed after
results.

The first Gate 0B seed left enough of the two-GPU-hour budget, so the protocol's
optional seed-variability clause was exercised symmetrically for all three
coordinate systems with the deterministic next seed, 430002. The primary gate
thresholds stayed fixed.

## 5. Gate 0A setup

The frozen set contained 64 states: 16 evenly spaced positions from the lowest
numbered episode of each of four tasks, one each from LIBERO-10, goal, object,
and spatial. For every state, a native `[16,50,32]` source tensor was sampled.
The pad intervention fixed coordinates `0:7` and varied `7:32`; the positive
control did the converse. Native, exact-repeat, and zero-pad diagnostics used the
same sources. The frozen checkpoint was never trained or modified.

The independent unit was the state. Variance was first computed over the 16
noise draws within each chunk cell, then averaged within state. Physical
differences were also divided by checkpoint action standard deviation to make
the seven control coordinates comparable.

## 6. Gate 0A controls

- Exact repeat RMS: `0.0`; maximum absolute error: `0.0`.
- Median valid-noise positive-control RMS: `0.0437406`, safely above the
  `1e-4` validity floor.
- Median native-versus-zero-pad RMS: `0.0189837` standardized units.
- No source noise was scaled or injected outside the official N(0,1) source.
- Step traces show valid-noise diversity contracting from `1.00093` at the
  source to `0.04410` after Euler step 10, while padded-only coupling grows from
  numerical zero to `0.01585` in the pooled step statistic.

## 7. Gate 0A quantitative results

| Quantity | Result |
|---|---:|
| States / tasks / draws | 64 / 4 / 16 |
| Median padded-noise RMS, standardized | 0.0104702 |
| Bootstrap 95% CI for median | [0.0096018, 0.0111801] |
| Median valid-noise RMS, standardized | 0.0437406 |
| Ratio of median variances, `R_leak` | 0.0572968 |
| Median padded-noise RMS, raw physical control coordinates | 0.00302635 |
| Median maximum pairwise RMS, standardized | 0.0176277 |

Taskwise median padded RMS values were `0.01082`, `0.01226`, `0.00862`, and
`0.01066` for task indices 0, 10, 20, and 30. Per-coordinate padded RMS ranged
from `0.00941` to `0.03059`; the gripper coordinate was largest. The effect was
not produced by one selected task.

## 8. Gate 0A verdict

**A1 STRONG SIGNAL.** The frozen rule required median padded RMS at least 0.01,
variance ratio at least 0.01, and a bootstrap lower bound above the repeatability
floor. Results were 0.01047, 0.0573, and 0.00960 versus a `1e-4` floor. This is a
stable generative coupling result. It does not by itself show a simulator success
effect.

## 9. Gate 0B setup

The three invertible representations were:

1. official checkpoint mean/std;
2. mean/std computed over all target cells of the frozen 128-anchor local
   training set;
3. fixed group scaling `diag([0.5,0.5,0.5,2,2,2,1])` applied to official
   normalized coordinates.

All transform/inverse checks had maximum absolute error
`5.96e-8 < 1e-6`. The data contained 128 training anchors from the first four
episodes of each task and 32 validation anchors from the fifth episodes. Every
run started from identical checkpoint bytes and used all 392,904,096 parameters
marked trainable, batch size 1, 64 AdamW updates at `1e-4`, identical order,
identical physical targets, identical native flow noise/times, BF16 autocast,
float32 loss, and gradient clipping at 10. Six runs covered three transforms and
two RNG seeds, for 384 optimizer updates total.

## 10. Gate 0B quantitative results

Physical action-chunk RMSE after 64 updates:

| Seed | Official | Local | Group-scaled | Local vs official | Group vs official |
|---|---:|---:|---:|---:|---:|
| 430001 | 0.44135 | 0.41634 | 0.51064 | -5.67% | +15.70% |
| 430002 | 0.44542 | 0.40718 | 0.43753 | -8.58% | -1.77% |

For seed 430001, the group-scaled paired state bootstrap CI was
`[+9.20%, +23.62%]` and the physicalized validation-loss direction matched at
three of four post-baseline checkpoints, satisfying the single-seed B1 rule.
For seed 430002, its CI was `[-6.97%, +4.09%]`; the large effect did not
replicate and changed sign. Local normalization was modestly lower-RMSE in both
seeds. Its primary CI crossed zero (`[-15.17%, +3.80%]`), while the replicate CI
was `[-14.12%, -3.40%]`.

The frozen pretrained diagnostic already showed large coordinate mismatch before
updates: RMSE was `0.05493` official, `0.13877` local, and `0.28027`
group-scaled. Thus Gate 0B cannot attribute all later separation purely to
optimization geometry; checkpoint-coordinate mismatch is substantial. The
pilot also degraded all conditions sharply from their own starting behavior.
This is not evidence that any convention improves the policy.

Primary-seed physicalized validation velocity MSE at updates 0→64 was
`0.00542→0.26968` official, `0.16945→0.21316` local, and
`0.38347→0.29880` group-scaled. Median pre-clip gradient norms were 16.60,
18.73, and 22.73 respectively, demonstrating different early optimization
scales under matched physical data.

## 11. Gate 0B verdict

**B2 WEAK SIGNAL.** The primary seed alone met B1, but the full symmetric seed
replicate showed that its strongest group-scaled difference was seed-sensitive.
The protocol explicitly assigns a large but seed-sensitive difference to B2.
Local normalization showed a smaller, directionally consistent difference. No
additional transforms or tuning were attempted.

## 12. Overall verdict

**STRONG GO**, because Mechanism A independently achieved A1. Mechanism B adds
evidence that coordinates matter but remains B2 under seed replication. This
verdict supports a next causal validation stage; it does not authorize or imply
a successful corrective method.

## 13. Limitations

- Gate 0A measures frozen-policy action changes offline. It does not establish a
  change in task success or closed-loop behavior.
- The 64 states cover four tasks but only the first episode of each; confidence
  intervals describe those frozen states, not all LIBERO tasks or episodes.
- Gate 0B is deliberately short and unstable: the official `1e-4` learning rate
  with full-model training sharply degraded validation generation in 64 updates.
- Only two training RNG seeds were run. The strongest coordinate effect was not
  seed-robust.
- Gate 0B begins from a checkpoint trained in official coordinates, so it mixes
  pretrained-coordinate mismatch, flow-source geometry, and optimization
  sensitivity. The experiment identifies action-interface dependence, not a
  complete decomposition.
- Official checkpoint/data feature declarations and current task metadata are
  imperfect, although the exact compatibility path and label-only recovery are
  recorded.

## 14. GPU usage

Measured experimental aggregate GPU active wall-time:

- Gate 0A: `63.2579 GPU-s = 0.0175716 GPU-h`.
- Gate 0B, six runs including evaluation/generation: `197.9628 GPU-s =
  0.0549897 GPU-h`.
- Total: **`261.2206 GPU-s = 0.0725613 GPU-h`**.

Two GPUs were used concurrently where symmetric runs allowed. Model-loading and
CPU video-decoding setup time are excluded from active experimental windows; the
one load/preprocess compatibility smoke test made no action forward call.

## 15. Next scientifically justified step

Before designing a solution, test behavioral relevance of Mechanism A with
paired official LIBERO rollouts from identical initial states and matched valid
source noise, comparing only native padded noise against a preregistered padded
source control. Use task success and trajectory divergence with sufficient
episode-level replication. This is a description only; no projected flow,
adapter, auxiliary loss, RL, or other method was implemented here.

Machine-readable results are in `gate0a_summary.json` and
`gate0b_summary.json`; the hashed inventory is `artifact_manifest.json`.
