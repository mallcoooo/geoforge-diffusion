from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionLevel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WellConditionLevel(nn.Module):
    """Encode well values without normalizing away their absolute amplitude."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.value_conv = nn.Conv2d(
            1,
            hidden_channels,
            3,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.mask_network = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x[:, :1]
        mask = x[:, 1:2]
        return self.refine(self.value_conv(values) + self.mask_network(mask))


class SparseConditionEncoder(nn.Module):
    """Encode a sparse spatial condition into U-Net stage-aligned residuals."""

    def __init__(
        self,
        *,
        kind: str,
        in_channels: int,
        stage_channels: Sequence[int],
        encoder_channels: int = 32,
        max_encoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if kind not in {"well", "horizon"}:
            raise ValueError("Sparse condition kind must be 'well' or 'horizon'")
        if in_channels != 2:
            raise ValueError(
                f"{kind} condition currently requires exactly two [value, mask] channels"
            )
        if not stage_channels or any(channels <= 0 for channels in stage_channels):
            raise ValueError("stage_channels must contain positive values")
        if encoder_channels <= 0 or max_encoder_channels < encoder_channels:
            raise ValueError("Condition encoder channel counts are invalid")
        self.kind = kind
        self.in_channels = int(in_channels)
        hidden_channels = [
            min(int(encoder_channels * (2**level)), int(max_encoder_channels))
            for level in range(len(stage_channels))
        ]
        if kind == "well":
            self.levels = nn.ModuleList(WellConditionLevel(hidden) for hidden in hidden_channels)
        else:
            self.levels = nn.ModuleList(
                ConditionLevel(in_channels, hidden) for hidden in hidden_channels
            )
        self.projections = nn.ModuleList(
            nn.Conv2d(hidden, output, kernel_size=1)
            for hidden, output in zip(hidden_channels, stage_channels, strict=True)
        )
        self.bottleneck_projection = nn.Conv2d(
            hidden_channels[-1], int(stage_channels[-1]), kernel_size=1
        )
        for projection in [*self.projections, self.bottleneck_projection]:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        condition: torch.Tensor,
        *,
        presence: torch.Tensor | None,
        target_shapes: Sequence[tuple[int, int]],
    ) -> tuple[torch.Tensor, ...]:
        if condition.ndim != 4 or condition.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.kind} condition [batch, {self.in_channels}, height, width], "
                f"got {tuple(condition.shape)}"
            )
        if len(target_shapes) != len(self.levels):
            raise ValueError(f"Expected {len(self.levels)} target shapes")
        gate = self._presence_gate(condition, presence)
        features: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for target_shape, level, projection in zip(
            target_shapes, self.levels, self.projections, strict=True
        ):
            resized = self._resize_condition(condition, target_shape)
            feature = level(resized)
            features.append(feature)
            residuals.append(projection(feature) * gate)
        residuals.append(self.bottleneck_projection(features[-1]) * gate)
        return tuple(residuals)

    def _presence_gate(
        self,
        condition: torch.Tensor,
        presence: torch.Tensor | None,
    ) -> torch.Tensor:
        if presence is None:
            values = torch.ones(condition.shape[0], device=condition.device, dtype=condition.dtype)
        else:
            values = presence.reshape(-1).to(device=condition.device, dtype=condition.dtype)
            if values.numel() != condition.shape[0]:
                raise ValueError(f"{self.kind}_present must contain one value per sample")
            if not torch.isfinite(values).all() or torch.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"{self.kind}_present values must be finite and lie in [0, 1]")
        return values[:, None, None, None]

    def _resize_condition(
        self,
        condition: torch.Tensor,
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(condition.shape[-2:]) == tuple(target_shape):
            return condition
        if any(
            target > source
            for target, source in zip(target_shape, condition.shape[-2:], strict=True)
        ):
            raise ValueError("Sparse condition encoder does not upsample condition maps")
        values = condition[:, :1]
        mask = condition[:, 1:2].clamp(0.0, 1.0)
        if self.kind == "well":
            support = functional.adaptive_avg_pool2d(mask, target_shape)
            numerator = functional.adaptive_avg_pool2d(values * mask, target_shape)
            pooled_values = torch.where(
                support > 0.0,
                numerator / support.clamp_min(1.0e-8),
                torch.zeros_like(numerator),
            )
            pooled_mask = functional.adaptive_max_pool2d(mask, target_shape)
            return torch.cat([pooled_values, pooled_mask], dim=1)
        return torch.cat(
            [
                functional.adaptive_max_pool2d(values, target_shape),
                functional.adaptive_max_pool2d(mask, target_shape),
            ],
            dim=1,
        )
