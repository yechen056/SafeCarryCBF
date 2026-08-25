"""Data contract for the RGB-conditioned carry model."""

from __future__ import annotations

import numpy as np

ACTION_DIMENSION = 17
PROPRIO_DIMENSION = 17
CAMERA_COUNT = 2
HORIZON = 16
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
TRAIN_CAMERA_HEIGHT = 240
TRAIN_CAMERA_WIDTH = 320
DATASET_FORMAT_VERSION = 1
ZARR_FORMAT_VERSION = 1

# State positions used by TiagoDualCarryTask: base pose, left arm, right arm.
CONTROLLED_STATE_INDICES = np.asarray(
    list(range(0, 3)) + list(range(8, 15)) + list(range(15, 22)), dtype=np.int64
)


def controlled_positions(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32).reshape(-1)
    if value.size != 60:
        raise ValueError(f"expected full carry state with 60 values, got {value.shape}")
    return value[CONTROLLED_STATE_INDICES].astype(np.float32, copy=True)


def validate_episode(rgb: np.ndarray, proprio: np.ndarray, actions: np.ndarray) -> None:
    if rgb.ndim != 5 or rgb.shape[1:] != (CAMERA_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise ValueError(f"rgb shape must be (T,2,480,640,3), got {rgb.shape}")
    if proprio.shape != (len(rgb), PROPRIO_DIMENSION):
        raise ValueError(f"proprio shape must be (T,17), got {proprio.shape}")
    if actions.shape != (len(rgb), ACTION_DIMENSION):
        raise ValueError(f"actions shape must be (T,17), got {actions.shape}")
    for name, value in (("rgb", rgb), ("proprio", proprio), ("actions", actions)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
