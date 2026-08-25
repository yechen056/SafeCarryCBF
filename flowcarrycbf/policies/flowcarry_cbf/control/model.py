"""Conditional Flow Matching model whose output is a whole-body action sequence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .flow_model import SafeFlowTrajectoryModel
from .schema import ACTION_DIMENSION, CONDITION_DIMENSION, HORIZON


class SafeFlowActionModel(SafeFlowTrajectoryModel):
    """SafeFlowMatcher Temporal U-Net with V2's direct-action contract."""

    def __init__(
        self,
        horizon: int = HORIZON,
        action_dimension: int = ACTION_DIMENSION,
        condition_dimension: int = CONDITION_DIMENSION,
        base_dimension: int = 64,
    ) -> None:
        if int(action_dimension) != ACTION_DIMENSION or int(condition_dimension) != CONDITION_DIMENSION:
            raise ValueError("V2 requires action_dimension=17 and condition_dimension=135")
        super().__init__(horizon=horizon, path_dimension=action_dimension, condition_dimension=condition_dimension, base_dimension=base_dimension)

    @property
    def action_dimension(self) -> int:
        return self.config.path_dimension

    def checkpoint(self, normalization: dict[str, np.ndarray], **metadata: Any) -> dict[str, Any]:
        return {
            "format_version": 2,
            "model_config": {
                "horizon": self.config.horizon,
                "action_dimension": self.config.path_dimension,
                "condition_dimension": self.config.condition_dimension,
                "base_dimension": self.config.base_dimension,
            },
            "model_state": self.state_dict(),
            "normalization": {key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()},
            "metadata": metadata,
        }

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if int(checkpoint.get("format_version", 0)) != 2:
            raise ValueError("checkpoint is not a Safe Carry V2 direct-action checkpoint")
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        return model, checkpoint
