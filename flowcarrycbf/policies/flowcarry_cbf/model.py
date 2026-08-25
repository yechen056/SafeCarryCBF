"""Paper-aligned dual-camera Conditional Flow Matching policy network.

The ResNet18 visual encoder and FiLM ConditionalUnet1D follow the official
implementation of *Affordance-based Robot Manipulation with Flow Matching*
vendored in ``third_party/flow_matching``. The two cameras share one encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision.models import resnet18

from .schema import TRAIN_CAMERA_HEIGHT, TRAIN_CAMERA_WIDTH


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=value.dtype) * -scale
        )
        embedding = value[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Downsample1D(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dimension, dimension, 3, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1D(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dimension, dimension, 4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Conv1DBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inputs, outputs, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, outputs),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        condition_dimension: int,
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.outputs = outputs
        self.blocks = nn.ModuleList(
            (
                Conv1DBlock(inputs, outputs, kernel_size, groups),
                Conv1DBlock(outputs, outputs, kernel_size, groups),
            )
        )
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dimension, outputs * 2)
        )
        self.residual = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        result = self.blocks[0](value)
        scale, bias = self.condition(condition).reshape(-1, 2, self.outputs, 1).unbind(1)
        result = scale * result + bias
        return self.blocks[1](result) + self.residual(value)


class ConditionalUnet1D(nn.Module):
    """FiLM temporal U-Net from the official Flow Matching policy code."""

    def __init__(
        self,
        input_dimension: int,
        global_condition_dimension: int,
        time_embedding_dimension: int = 256,
        down_dimensions: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        super().__init__()
        down_dimensions = tuple(int(value) for value in down_dimensions)
        all_dimensions = (input_dimension,) + down_dimensions
        condition_dimension = time_embedding_dimension + global_condition_dimension
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dimension),
            nn.Linear(time_embedding_dimension, time_embedding_dimension * 4),
            nn.Mish(),
            nn.Linear(time_embedding_dimension * 4, time_embedding_dimension),
        )
        pairs = tuple(zip(all_dimensions[:-1], all_dimensions[1:]))
        self.down_modules = nn.ModuleList()
        for index, (inputs, outputs) in enumerate(pairs):
            self.down_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(inputs, outputs, condition_dimension, kernel_size, groups),
                        ConditionalResidualBlock1D(outputs, outputs, condition_dimension, kernel_size, groups),
                        Downsample1D(outputs) if index < len(pairs) - 1 else nn.Identity(),
                    )
                )
            )
        middle = all_dimensions[-1]
        self.middle_modules = nn.ModuleList(
            (
                ConditionalResidualBlock1D(middle, middle, condition_dimension, kernel_size, groups),
                ConditionalResidualBlock1D(middle, middle, condition_dimension, kernel_size, groups),
            )
        )
        self.up_modules = nn.ModuleList()
        for inputs, outputs in reversed(pairs[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(outputs * 2, inputs, condition_dimension, kernel_size, groups),
                        ConditionalResidualBlock1D(inputs, inputs, condition_dimension, kernel_size, groups),
                        Upsample1D(inputs),
                    )
                )
            )
        self.output = nn.Sequential(
            Conv1DBlock(down_dimensions[0], down_dimensions[0], kernel_size, groups),
            nn.Conv1d(down_dimensions[0], input_dimension, 1),
        )

    def forward(
        self, sample: torch.Tensor, flow_time: torch.Tensor, global_condition: torch.Tensor
    ) -> torch.Tensor:
        value = sample.moveaxis(-1, -2)
        if flow_time.ndim == 0:
            flow_time = flow_time[None]
        flow_time = flow_time.expand(value.shape[0])
        condition = torch.cat((self.time_embedding(flow_time), global_condition), dim=-1)
        skips = []
        for first, second, downsample in self.down_modules:
            value = first(value, condition)
            value = second(value, condition)
            skips.append(value)
            value = downsample(value)
        for middle in self.middle_modules:
            value = middle(value, condition)
        for first, second, upsample in self.up_modules:
            value = torch.cat((value, skips.pop()), dim=1)
            value = first(value, condition)
            value = second(value, condition)
            value = upsample(value)
        return self.output(value).moveaxis(-1, -2)


def _replace_batch_norm(module: nn.Module) -> nn.Module:
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _replace_batch_norm(child)
    return module


@dataclass(frozen=True)
class RGBModelConfig:
    horizon: int = 16
    action_dimension: int = 17
    proprio_dimension: int = 17
    camera_count: int = 2
    visual_feature_dimension: int = 512
    time_embedding_dimension: int = 256
    down_dimensions: tuple[int, ...] = (256, 512, 1024)
    image_height: int = TRAIN_CAMERA_HEIGHT
    image_width: int = TRAIN_CAMERA_WIDTH


class RGBFlowTrajectoryModel(nn.Module):
    """Flow Matching velocity field over 16 future absolute control targets."""

    def __init__(
        self,
        horizon: int = 16,
        action_dimension: int = 17,
        proprio_dimension: int = 17,
        camera_count: int = 2,
        visual_feature_dimension: int = 512,
        time_embedding_dimension: int = 256,
        down_dimensions: Sequence[int] = (256, 512, 1024),
        image_height: int = TRAIN_CAMERA_HEIGHT,
        image_width: int = TRAIN_CAMERA_WIDTH,
    ) -> None:
        super().__init__()
        self.config = RGBModelConfig(
            horizon=horizon,
            action_dimension=action_dimension,
            proprio_dimension=proprio_dimension,
            camera_count=camera_count,
            visual_feature_dimension=visual_feature_dimension,
            time_embedding_dimension=time_embedding_dimension,
            down_dimensions=tuple(down_dimensions),
            image_height=image_height,
            image_width=image_width,
        )
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        self.camera_encoder = _replace_batch_norm(encoder)
        condition_dimension = visual_feature_dimension * camera_count + proprio_dimension
        self.temporal = ConditionalUnet1D(
            input_dimension=action_dimension,
            global_condition_dimension=condition_dimension,
            time_embedding_dimension=time_embedding_dimension,
            down_dimensions=down_dimensions,
        )

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"expected five-dimensional RGB input, got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3)
        if images.shape[1:3] != (self.config.camera_count, 3):
            raise ValueError(f"expected {self.config.camera_count} RGB cameras, got {tuple(images.shape)}")
        divisor = 255.0 if images.dtype == torch.uint8 else 1.0
        images = images.float() / divisor
        batch = images.shape[0]
        images = images.flatten(0, 1)
        if images.shape[-2:] != (self.config.image_height, self.config.image_width):
            images = functional.interpolate(
                images,
                size=(self.config.image_height, self.config.image_width),
                mode="area",
            )
        return images.reshape(batch, self.config.camera_count, 3, self.config.image_height, self.config.image_width)

    def encode_condition(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        images = self._images(images)
        batch = images.shape[0]
        visual = self.camera_encoder(images.flatten(0, 1)).reshape(batch, -1)
        return torch.cat((visual, proprio.float()), dim=-1)

    def forward(
        self,
        path: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        return self.temporal(path, flow_time, self.encode_condition(images, proprio))

    def cfm_loss(self, target: torch.Tensor, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        source = torch.randn_like(target)
        flow_time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        interpolation = flow_time[:, None, None]
        current = (1.0 - interpolation) * source + interpolation * target
        return functional.mse_loss(self(current, images, proprio, flow_time), target - source)

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        prediction_steps: int = 8,
    ) -> torch.Tensor:
        condition = self.encode_condition(images, proprio)
        path = torch.randn(
            condition.shape[0],
            self.config.horizon,
            self.config.action_dimension,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        steps = max(1, int(prediction_steps))
        for index in range(steps):
            flow_time = torch.full(
                (condition.shape[0],), index / steps,
                device=condition.device, dtype=condition.dtype,
            )
            path = path + self.temporal(path, flow_time, condition) / steps
        return path

    def checkpoint(self, normalization: dict[str, np.ndarray], **metadata: Any) -> dict[str, Any]:
        return {
            "format_version": 2,
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
    ) -> tuple["RGBFlowTrajectoryModel", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if int(payload.get("format_version", 0)) != 2:
            raise ValueError("checkpoint is not a paper-aligned RGB Flow model")
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["model_state"])
        model.to(device).eval()
        return model, payload
