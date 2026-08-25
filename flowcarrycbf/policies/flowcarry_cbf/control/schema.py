"""Shared V2 tensor schema and visual instance palette."""

from __future__ import annotations

import numpy as np

STATE_DIMENSION = 60
MAX_OBSTACLES = 5
OBSTACLE_FEATURES = 15
ACTION_DIMENSION = 17
HORIZON = 32
CONDITION_DIMENSION = STATE_DIMENSION + MAX_OBSTACLES * OBSTACLE_FEATURES

POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
ACCELERATION = slice(6, 9)
HALF_EXTENTS = slice(9, 12)
TYPE_INDEX = 12
UNCERTAINTY_INDEX = 13
VALID_INDEX = 14

SPHERE_TYPE = 0.0
CUBE_TYPE = 1.0

# Saturated, well-separated colors permit instance tracking without semantic IDs.
INSTANCE_COLORS = np.asarray(
    [
        [0.90, 0.12, 0.12],
        # Deep blue remains separated from the cyan slot after renderer
        # lighting shifts hue toward green-blue.
        [0.04, 0.14, 0.95],
        [0.05, 0.90, 0.90],
        [0.10, 0.78, 0.35],
        [0.72, 0.18, 0.88],
    ],
    dtype=np.float32,
)


def flatten_condition(state: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(STATE_DIMENSION)
    obstacles = np.asarray(obstacles, dtype=np.float32).reshape(
        MAX_OBSTACLES, OBSTACLE_FEATURES
    )
    return np.concatenate((state, obstacles.reshape(-1))).astype(np.float32)


def validate_action_sequence(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (HORIZON, ACTION_DIMENSION):
        raise ValueError(
            f"expected action sequence {(HORIZON, ACTION_DIMENSION)}, got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("action sequence contains non-finite values")
    return value
