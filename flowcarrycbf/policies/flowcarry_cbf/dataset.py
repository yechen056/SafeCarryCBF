"""Zarr dataset and normalization for the Phase 3 RGB Flow policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from flowcarrycbf.policies.flowcarry_cbf.schema import (
    ACTION_DIMENSION,
    CAMERA_COUNT,
    HORIZON,
    PROPRIO_DIMENSION,
    TRAIN_CAMERA_HEIGHT,
    TRAIN_CAMERA_WIDTH,
    ZARR_FORMAT_VERSION,
)


def _zarr():
    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("Zarr support requires `pip install 'zarr<3'`") from error
    return zarr


def open_zarr(root: str | Path):
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(path)
    return _zarr().open_group(str(path), mode="r")


def validate_zarr_dataset(
    root: str | Path,
    *,
    expected_episodes: int | None = None,
    expected_camera_count: int | None = None,
    expected_robot: str | None = None,
) -> dict[str, Any]:
    group = open_zarr(root)
    required = {"rgb", "proprio", "actions", "episode_ends"}
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise ValueError(f"Zarr dataset is missing arrays: {missing}")
    rgb = group["rgb"]
    proprio = group["proprio"]
    actions = group["actions"]
    ends = np.asarray(group["episode_ends"][:], dtype=np.int64)
    if int(group.attrs.get("format_version", -1)) != ZARR_FORMAT_VERSION:
        raise ValueError("unsupported Phase 3 Zarr format")
    if expected_episodes is not None and len(ends) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, found {len(ends)}")
    camera_count = int(group.attrs.get("camera_count", CAMERA_COUNT))
    if expected_camera_count is not None and camera_count != int(expected_camera_count):
        raise ValueError(f"expected {expected_camera_count} cameras, found {camera_count}")
    if expected_robot is not None and str(group.attrs.get("robot", "tiago")) != str(expected_robot):
        raise ValueError(f"dataset robot must be {expected_robot!r}, got {group.attrs.get('robot')!r}")
    expected_rgb_tail = (camera_count, TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH, 3)
    if rgb.shape[1:] != expected_rgb_tail or rgb.dtype != np.dtype("u1"):
        raise ValueError(f"rgb must be uint8 (N,{expected_rgb_tail}), got {rgb.shape} {rgb.dtype}")
    if proprio.shape != (len(rgb), PROPRIO_DIMENSION) or proprio.dtype != np.dtype("f4"):
        raise ValueError(f"invalid proprio array: {proprio.shape} {proprio.dtype}")
    if actions.shape != (len(rgb), ACTION_DIMENSION) or actions.dtype != np.dtype("f4"):
        raise ValueError(f"invalid action array: {actions.shape} {actions.dtype}")
    if len(ends) == 0 or ends[-1] != len(rgb) or np.any(np.diff(ends) < HORIZON):
        raise ValueError("episode ends are inconsistent with RGB frames or horizon")
    if not np.all(np.isfinite(proprio[:])) or not np.all(np.isfinite(actions[:])):
        raise ValueError("proprio or action arrays contain non-finite values")
    return {
        "episodes": int(len(ends)),
        "total_frames": int(len(rgb)),
        "rgb_shape": list(rgb.shape),
        "proprio_shape": list(proprio.shape),
        "action_shape": list(actions.shape),
        "source": str(group.attrs.get("source", "")),
    }


class RGBFlowZarrDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        episode_indices: list[int],
        *,
        horizon: int = HORIZON,
    ) -> None:
        self.root = Path(root)
        self.episode_indices = [int(value) for value in episode_indices]
        self.horizon = int(horizon)
        group = open_zarr(self.root)
        self.episode_ends = np.asarray(group["episode_ends"][:], dtype=np.int64)
        self.episode_starts = np.concatenate((np.zeros(1, dtype=np.int64), self.episode_ends[:-1]))
        self.items: list[tuple[int, int]] = []
        for episode_index in self.episode_indices:
            if not 0 <= episode_index < len(self.episode_ends):
                raise IndexError(f"episode index {episode_index} is out of range")
            start = int(self.episode_starts[episode_index])
            end = int(self.episode_ends[episode_index])
            self.items.extend((frame, end) for frame in range(start, end))
        self._group = None

    def _arrays(self):
        if self._group is None:
            self._group = open_zarr(self.root)
        return self._group

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        frame, episode_end = self.items[item]
        arrays = self._arrays()
        rgb = np.asarray(arrays["rgb"][frame], dtype=np.uint8)
        proprio = np.asarray(arrays["proprio"][frame], dtype=np.float32)
        target_end = min(frame + self.horizon, episode_end)
        actions = np.asarray(arrays["actions"][frame:target_end], dtype=np.float32)
        if len(actions) < self.horizon:
            actions = np.concatenate(
                (actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)),
                axis=0,
            )
        return torch.from_numpy(rgb), torch.from_numpy(proprio), torch.from_numpy(actions)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_group"] = None
        return state


def split_episode_indices(count: int) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("at least two episodes are required")
    return list(range(count - 1)), [count - 1]


def fit_minmax_normalization(root: str | Path) -> dict[str, np.ndarray]:
    """Fit per-dimension [-1, 1] transforms over all collected episodes."""

    group = open_zarr(root)
    result: dict[str, np.ndarray] = {}
    for name in ("proprio", "actions"):
        values = np.asarray(group[name][:], dtype=np.float32)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        value_range = maximum - minimum
        if np.any(value_range < 1e-7):
            dimensions = np.flatnonzero(value_range < 1e-7).tolist()
            raise ValueError(f"{name} has constant dimensions: {dimensions}")
        result[f"{name}_scale"] = (2.0 / value_range).astype(np.float32)
        result[f"{name}_offset"] = (-1.0 - 2.0 * minimum / value_range).astype(np.float32)
        result[f"{name}_min"] = minimum.astype(np.float32)
        result[f"{name}_max"] = maximum.astype(np.float32)
    return result
