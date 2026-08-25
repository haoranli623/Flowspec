#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch

from flowspec_vla.phase1 import load_phase1_config, phase1_source, project_executable_subspace, validate_method
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.scripts.lerobot_eval import eval_policy


def install_sampler(policy: SmolVLAPolicy, method: str) -> None:
    method = validate_method(method)

    @torch.inference_mode()
    def phase1_sample_actions(
        model,
        images,
        image_masks,
        language_tokens,
        language_masks,
        state,
        noise=None,
        **_kwargs,
    ):
        batch_size = state.shape[0]
        if noise is None:
            noise = model.sample_noise(
                (batch_size, model.config.chunk_size, model.config.max_action_dim), state.device
            )
        x_t = phase1_source(noise, method, 7)
        if method == "m2":
            x_t = project_executable_subspace(x_t, 7)
        prefix, prefix_masks, prefix_attention = model.embed_prefix(
            images, image_masks, language_tokens, language_masks, state=state
        )
        attention = make_att_2d_masks(prefix_masks, prefix_attention)
        positions = torch.cumsum(prefix_masks, dim=1) - 1
        _, cache = model.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=model.config.use_cache,
        )
        dt = -1.0 / model.config.num_steps
        for step in range(model.config.num_steps):
            timestep = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=state.device
            )
            velocity = model.denoise_step(prefix_masks, cache, x_t, timestep)
            x_t = x_t + dt * velocity
            if method == "m2":
                x_t = project_executable_subspace(x_t, 7)
        return x_t

    policy.model.sample_actions = types.MethodType(phase1_sample_actions, policy.model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["m0", "m1", "m2"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    method = validate_method(args.method)
    config = load_phase1_config()
    if args.seed not in config["training"]["seeds"]:
        raise ValueError("Unknown primary training seed")
    checkpoint = (
        Path(config["paths"]["trained_models"])
        / f"{method}_seed{args.seed}"
        / "update_05000"
    )
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.eval()
    install_sampler(policy, method)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    root = Path(config["paths"]["artifacts"]) / "rollouts"
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{method}_seed{args.seed}.json"
    partial = root / f"{method}_seed{args.seed}.partial.json"
    records = json.loads(partial.read_text())["tasks"] if partial.exists() else []
    completed = {(row["suite"], row["task_id"]) for row in records}
    started = time.perf_counter()
    episodes_per_task = config["rollout"]["episodes_per_task"]
    for suite_index, suite in enumerate(config["rollout"]["suites"]):
        for task_id in range(config["rollout"]["tasks_per_suite"]):
            if (suite, task_id) in completed:
                continue
            task_seed = config["rollout"]["episode_seed"] + suite_index * 1000 + task_id * 10
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            np.random.seed(task_seed)
            env_config = LiberoEnvConfig(
                task=suite,
                task_ids=[task_id],
                fps=20,
                observation_height=256,
                observation_width=256,
                obs_type="pixels_agent_pos",
                control_mode="relative",
                init_states=True,
                hard_reset=True,
            )
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, policy.config)
            mapping = make_env(env_config, n_envs=episodes_per_task, use_async_envs=False)
            env = mapping[suite][task_id]
            task_started = time.perf_counter()
            try:
                result = eval_policy(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=episodes_per_task,
                    max_episodes_rendered=0,
                    return_episode_data=False,
                    start_seed=task_seed,
                )
            finally:
                env.close()
            record = {
                "suite": suite,
                "task_id": task_id,
                "task_seed": task_seed,
                "episodes": result["per_episode"],
                "successes": int(sum(row["success"] for row in result["per_episode"])),
                "success_rate": float(np.mean([row["success"] for row in result["per_episode"]])),
                "seconds": time.perf_counter() - task_started,
            }
            records.append(record)
            partial.write_text(json.dumps({"method": method, "seed": args.seed, "tasks": records}, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    torch.cuda.synchronize()
    summary = {
        "status": "COMPLETE",
        "method": method,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_update": config["rollout"]["checkpoint_update"],
        "tasks": records,
        "task_count": len(records),
        "episode_count": sum(len(row["episodes"]) for row in records),
        "successes": sum(row["successes"] for row in records),
        "success_rate": float(
            sum(row["successes"] for row in records)
            / sum(len(row["episodes"]) for row in records)
        ),
        "gpu_active_wall_seconds": time.perf_counter() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
