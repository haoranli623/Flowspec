from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path("/mnt/NAS/data/hl5757/projects/flowspec-vla")
DATASET_ROOT = Path("/mnt/NAS/data/hl5757/datasets/flowspec-vla/lerobot-libero-a1aaacb7")


def recovered_tasks() -> pd.DataFrame:
    payload = json.loads((PROJECT_ROOT / "config/task_map.json").read_text())
    ordered = [payload["tasks"][str(i)] for i in range(40)]
    return pd.DataFrame(
        {"task_index": list(range(40))},
        index=pd.Index(ordered, name="task"),
    )


def load_dataset(*, with_action_chunk: bool = True) -> LeRobotDataset:
    delta_timestamps = {"action": [i / 10.0 for i in range(50)]} if with_action_chunk else None
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    # Dataset payload revision a1aaacb7 accidentally dropped the parquet index.
    # DatasetReader keeps a reference to this metadata object, so restoring the
    # exact parent mapping here affects only the emitted task string.
    dataset.meta.tasks = recovered_tasks()
    return dataset


def frame_table() -> pd.DataFrame:
    columns = ["index", "episode_index", "frame_index", "task_index"]
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(DATASET_ROOT.glob("data/**/*.parquet"))]
    table = pd.concat(frames, ignore_index=True).sort_values("index").reset_index(drop=True)
    if not (table["index"].to_numpy() == range(len(table))).all():
        raise RuntimeError("Dataset absolute indices are not contiguous and ordered")
    return table


def episode_rows(table: pd.DataFrame, task_index: int) -> list[pd.DataFrame]:
    task_rows = table[table["task_index"] == task_index]
    return [group.sort_values("frame_index") for _, group in task_rows.groupby("episode_index", sort=True)]
