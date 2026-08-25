"""Compact Conditional Flow Matching model adapted from SafeFlowMatcher."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=value.dtype) * -scale
        )
        embedding = value[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int = 5) -> None:
        super().__init__()
        groups = min(8, outputs)
        while outputs % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(inputs, outputs, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, outputs),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, embedding_dim: int) -> None:
        super().__init__()
        self.first = ConvBlock(inputs, outputs)
        self.second = ConvBlock(outputs, outputs)
        self.embedding = nn.Sequential(nn.Mish(), nn.Linear(embedding_dim, outputs))
        self.residual = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        result = self.first(value) + self.embedding(embedding).unsqueeze(-1)
        return self.second(result) + self.residual(value)


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 64
    path_dimension: int = 3
    condition_dimension: int = 84
    base_dimension: int = 32


class SafeFlowTrajectoryModel(nn.Module):
    """Temporal U-Net velocity field for 64-point base trajectories."""

    def __init__(
        self,
        horizon: int = 64,
        path_dimension: int = 3,
        condition_dimension: int = 84,
        base_dimension: int = 32,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            horizon=horizon,
            path_dimension=path_dimension,
            condition_dimension=condition_dimension,
            base_dimension=base_dimension,
        )
        embedding_dim = base_dimension * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_dimension),
            nn.Linear(base_dimension, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dimension, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        channels = [path_dimension, base_dimension, base_dimension * 2, base_dimension * 4]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channel_pairs = list(zip(channels[:-1], channels[1:]))
        for index, (inputs, outputs) in enumerate(channel_pairs):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(inputs, outputs, embedding_dim),
                        ResidualTemporalBlock(outputs, outputs, embedding_dim),
                    ]
                )
            )
            self.downsamples.append(
                nn.Identity()
                if index == len(channel_pairs) - 1
                else nn.Conv1d(outputs, outputs, 3, 2, 1)
            )

        middle = channels[-1]
        self.middle = nn.ModuleList(
            [
                ResidualTemporalBlock(middle, middle, embedding_dim),
                ResidualTemporalBlock(middle, middle, embedding_dim),
            ]
        )
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_pairs = channel_pairs[1:][::-1]
        current = middle
        for inputs, outputs in reversed_pairs:
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(current + outputs, inputs, embedding_dim),
                        ResidualTemporalBlock(inputs, inputs, embedding_dim),
                    ]
                )
            )
            self.upsamples.append(nn.ConvTranspose1d(inputs, inputs, 4, 2, 1))
            current = inputs
        self.output = nn.Sequential(
            ConvBlock(base_dimension, base_dimension),
            nn.Conv1d(base_dimension, path_dimension, 1),
        )

    def forward(
        self, path: torch.Tensor, condition: torch.Tensor, flow_time: torch.Tensor
    ) -> torch.Tensor:
        value = path.transpose(1, 2)
        embedding = self.time_embedding(flow_time) + self.condition_embedding(condition)
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            value = blocks[0](value, embedding)
            value = blocks[1](value, embedding)
            skips.append(value)
            value = downsample(value)
        value = self.middle[0](value, embedding)
        value = self.middle[1](value, embedding)
        for index, (blocks, upsample) in enumerate(zip(self.up_blocks, self.upsamples)):
            skip = skips.pop()
            if value.shape[-1] != skip.shape[-1]:
                value = functional.interpolate(value, size=skip.shape[-1], mode="nearest")
            value = torch.cat((value, skip), dim=1)
            value = blocks[0](value, embedding)
            value = blocks[1](value, embedding)
            value = upsample(value)
        return self.output(value).transpose(1, 2)

    def cfm_loss(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Simulation-free conditional flow matching objective."""

        source = torch.randn_like(target)
        flow_time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        interpolation = flow_time[:, None, None]
        current = (1.0 - interpolation) * source + interpolation * target
        target_velocity = target - source
        predicted_velocity = self(current, condition, flow_time)
        return functional.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        prediction_steps: int = 1,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        path = torch.randn(
            batch,
            self.config.horizon,
            self.config.path_dimension,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        steps = max(1, int(prediction_steps))
        for index in range(steps):
            flow_time = torch.full(
                (batch,), index / steps, device=condition.device, dtype=condition.dtype
            )
            path = path + self(path, condition, flow_time) / steps
        return path

    def checkpoint(self, normalization: dict[str, np.ndarray], **metadata: Any) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_config": asdict(self.config),
            "model_state": self.state_dict(),
            "normalization": {
                key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()
            },
            "metadata": metadata,
        }

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["SafeFlowTrajectoryModel", dict[str, Any]]:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        return model, checkpoint
