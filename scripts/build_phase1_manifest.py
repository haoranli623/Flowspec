#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flowspec_vla.phase1 import load_phase1_config


QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def statistics(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [int(len(values))],
        **{name: np.quantile(values, q, axis=0).astype(float).tolist() for name, q in QUANTILES.items()},
    }


def score(seed: int, task: int, episode: int) -> str:
    return hashlib.sha256(f"{seed}:{task}:{episode}".encode()).hexdigest()


def main() -> None:
    config = load_phase1_config()
    dataset_root = Path(config["paths"]["dataset"])
    artifact_root = Path(config["paths"]["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "task_index",
        "observation.state",
        "action",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in sorted(dataset_root.glob("data/**/*.parquet"))],
        ignore_index=True,
    ).sort_values("index")
    if frame["index"].astype(int).tolist() != list(range(len(frame))):
        raise RuntimeError("LIBERO frame indices are not contiguous")

    selection_seed = config["split"]["selection_seed"]
    train_episodes_per_task = config["split"]["training_episodes_per_task"]
    held_out_per_task = config["split"]["held_out_episodes_per_task"]
    anchor_fractions = config["split"]["validation_anchor_fractions"]
    train_indices: list[int] = []
    validation_indices: list[int] = []
    task_records = []
    held_out_episodes: set[int] = set()
    training_episodes: set[int] = set()

    for task_index, task_frame in frame.groupby("task_index", sort=True):
        episodes = []
        for episode_index, episode_frame in task_frame.groupby("episode_index", sort=True):
            episode_frame = episode_frame.sort_values("frame_index").reset_index(drop=True)
            episodes.append(
                (score(selection_seed, int(task_index), int(episode_index)), int(episode_index), episode_frame)
            )
        episodes.sort(key=lambda row: row[0])
        held_out = episodes[:held_out_per_task]
        training = episodes[held_out_per_task : held_out_per_task + train_episodes_per_task]
        for _, episode_index, episode_frame in training:
            training_episodes.add(episode_index)
            train_indices.extend(episode_frame["index"].astype(int).tolist())
        for _, episode_index, episode_frame in held_out:
            held_out_episodes.add(episode_index)
            n = len(episode_frame)
            for fraction in anchor_fractions:
                offset = min(int(fraction * n), n - 1)
                validation_indices.append(int(episode_frame.iloc[offset]["index"]))
        task_records.append(
            {
                "task_index": int(task_index),
                "available_episodes": len(episodes),
                "training_episodes": [episode for _, episode, _ in training],
                "held_out_episodes": [episode for _, episode, _ in held_out],
                "training_frames": int(sum(len(rows) for _, _, rows in training)),
                "validation_states": len(held_out) * len(anchor_fractions),
            }
        )

    if training_episodes & held_out_episodes:
        raise RuntimeError("Training and held-out episodes overlap")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Training and validation frame indices overlap")
    if len(task_records) != config["split"]["tasks"]:
        raise RuntimeError("Unexpected number of tasks")

    train_frame = frame.set_index("index").loc[train_indices]
    states = np.stack(train_frame["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(train_frame["action"].to_numpy()).astype(np.float32)
    raw_stats = {
        "observation.state": statistics(states),
        "action": statistics(actions),
    }
    renamed_stats = {
        "observation.state": raw_stats["observation.state"],
        "action": raw_stats["action"],
    }
    manifest = {
        "selection_before_phase1_results": True,
        "selection_algorithm": "SHA256('{seed}:{task_index}:{episode_index}'), ascending within task",
        "selection_seed": selection_seed,
        "dataset_revision": config["revisions"]["dataset_revision"],
        "total_dataset_frames": len(frame),
        "task_count": len(task_records),
        "training_episode_count": len(training_episodes),
        "training_frame_count": len(train_indices),
        "training_fraction_of_all_frames": len(train_indices) / len(frame),
        "held_out_episode_count": len(held_out_episodes),
        "validation_state_count": len(validation_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "tasks": task_records,
        "normalization_stats_raw": raw_stats,
        "normalization_stats_renamed": renamed_stats,
    }
    output = artifact_root / "split_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "training_episodes": len(training_episodes),
                "training_frames": len(train_indices),
                "training_fraction": manifest["training_fraction_of_all_frames"],
                "held_out_episodes": len(held_out_episodes),
                "validation_states": len(validation_indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
