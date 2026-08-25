#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path

from flowspec_vla.data import PROJECT_ROOT, episode_rows, frame_table, recovered_tasks


ARTIFACT_DIR = Path("/mnt/NAS/data/hl5757/generated_artifacts/flowspec-vla/gate0")
TASK_INDICES = [0, 10, 20, 30]
CHUNK = 50


def row_record(row, task_index: int, task: str, role: str) -> dict:
    return {
        "role": role,
        "task_index": task_index,
        "task": task,
        "episode_index": int(row.episode_index),
        "frame_index": int(row.frame_index),
        "dataset_index": int(row["index"]),
    }


def main() -> None:
    table = frame_table()
    tasks = recovered_tasks()
    gate0a = []
    gate0b_train = []
    gate0b_validation = []

    for task_index in TASK_INDICES:
        task = str(tasks.iloc[task_index].name)
        episodes = episode_rows(table, task_index)
        if len(episodes) < 5:
            raise RuntimeError(f"Task {task_index} has only {len(episodes)} episodes")

        first = episodes[0]
        length = len(first)
        for j in range(1, 17):
            offset = math.floor(j * (length - 1) / 17)
            gate0a.append(row_record(first.iloc[offset], task_index, task, "gate0a_state"))

        for rank in range(4):
            episode = episodes[rank]
            max_anchor = len(episode) - CHUNK
            if max_anchor < 0:
                raise RuntimeError(f"Task {task_index} episode {rank} shorter than action chunk")
            for j in range(1, 9):
                offset = math.floor(j * max_anchor / 9)
                gate0b_train.append(
                    row_record(episode.iloc[offset], task_index, task, f"gate0b_train_episode_rank_{rank}")
                )

        validation = episodes[4]
        max_anchor = len(validation) - CHUNK
        for j in range(1, 9):
            offset = math.floor(j * max_anchor / 9)
            gate0b_validation.append(
                row_record(validation.iloc[offset], task_index, task, "gate0b_validation_episode_rank_4")
            )

    payload = {
        "protocol_commit": "337c95305698e527ef09db6f53040fb377a9d6a5",
        "selection_before_model_results": True,
        "dataset_frames": len(table),
        "gate0a": gate0a,
        "gate0b_train": gate0b_train,
        "gate0b_validation": gate0b_validation,
    }
    if len(gate0a) != 64 or len(gate0b_train) != 128 or len(gate0b_validation) != 32:
        raise RuntimeError("Unexpected frozen manifest counts")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "state_manifest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out)
    print(json.dumps({k: len(payload[k]) for k in ["gate0a", "gate0b_train", "gate0b_validation"]}))


if __name__ == "__main__":
    main()
