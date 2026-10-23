from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from geoforge_diffusion.models.conditioning import SparseConditionEncoder


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        padding_mode="reflect",
    )


class FourierNoiseEmbedding(nn.Module):
    """Fourier features for EDM's continuous ``log(sigma) / 4`` label."""

    def __init__(self, channels: int, embedding_channels: int) -> None:
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("Noise embedding channels must be a positive even number")
        half = channels // 2
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(self, noise_labels: torch.Tensor) -> torch.Tensor:
        labels = noise_labels.reshape(-1, 1).to(dtype=self.frequencies.dtype)
        angles = labels * self.frequencies.reshape(1, -1) * (2.0 * math.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.projection(features)


class ScaleEmbedding(nn.Module):
    """Global embedding of relative time/horizontal sampling factors."""

    def __init__(self, embedding_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(self, log2_scale_factors: torch.Tensor) -> torch.Tensor:
        return self.projection(log2_scale_factors.to(dtype=self.projection[0].weight.dtype))


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, embedding_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm0 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv0 = _conv3x3(in_channels, out_channels)
        self.embedding = nn.Linear(embedding_channels, 2 * out_channels)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = _conv3x3(out_channels, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv0(functional.silu(self.norm0(x)))
        scale, shift = (
            self.embedding(functional.silu(embedding)).reshape(x.shape[0], -1, 1, 1).chunk(2, dim=1)
        )
        h = self.norm1(h) * (1.0 + scale) + shift
        h = self.conv1(self.dropout(functional.silu(h)))
        return residual + h


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0 or channels % num_heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        head_channels = channels // self.num_heads

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.num_heads, head_channels, height * width).transpose(
                2, 3
            )

        attended = functional.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), dropout_p=0.0
        )
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)
        return x + self.projection(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.avg_pool2d(x, kernel_size=2, stride=2))


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(current_channels, out_channels, embedding_channels, dropout)
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, embedding)
        return x


class MultiResolutionUNet(nn.Module):
    """Fully convolutional 2D U-Net for all configured impedance buckets."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        channel_mult: Sequence[int] = (1, 1, 2, 2, 4),
        num_blocks: int = 2,
        embedding_channels: int = 512,
        num_heads: int = 8,
        dropout: float = 0.05,
        use_scale_embedding: bool = True,
        conditioning: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or model_channels <= 0:
            raise ValueError("Channel counts must be positive")
        if not channel_mult or any(multiplier <= 0 for multiplier in channel_mult):
            raise ValueError("channel_mult must contain positive values")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if embedding_channels <= 0:
            raise ValueError("embedding_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_scale_embedding = bool(use_scale_embedding)
        self.downsample_factor = 2 ** (len(channel_mult) - 1)
        channels = [int(model_channels * multiplier) for multiplier in channel_mult]
        self.stage_channels = tuple(channels)

        condition_settings = dict(conditioning or {})
        self.conditioning_enabled = bool(condition_settings.get("enabled", False))
        if self.conditioning_enabled:
            condition_kwargs = {
                "stage_channels": self.stage_channels,
                "encoder_channels": int(condition_settings.get("encoder_channels", 32)),
                "max_encoder_channels": int(condition_settings.get("max_encoder_channels", 128)),
            }
            self.well_condition_encoder = SparseConditionEncoder(
                kind="well",
                in_channels=2,
                **condition_kwargs,
            )
            self.horizon_condition_encoder = SparseConditionEncoder(
                kind="horizon",
                in_channels=2,
                **condition_kwargs,
            )
        else:
            self.well_condition_encoder = None
            self.horizon_condition_encoder = None

        self.noise_embedding = FourierNoiseEmbedding(model_channels, embedding_channels)
        self.scale_embedding = (
            ScaleEmbedding(embedding_channels) if self.use_scale_embedding else None
        )
        self.input_conv = _conv3x3(in_channels, channels[0])

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels[0]
        for level, output_channels in enumerate(channels):
            self.encoder_stages.append(
                ResidualStage(
                    current_channels, output_channels, embedding_channels, num_blocks, dropout
                )
            )
            current_channels = output_channels
            if level < len(channels) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid_block0 = ResidualBlock(
            current_channels, current_channels, embedding_channels, dropout
        )
        self.mid_attention = SelfAttention2d(current_channels, num_heads)
        self.mid_block1 = ResidualBlock(
            current_channels, current_channels, embedding_channels, dropout
        )

        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            output_channels = channels[level]
            self.decoder_stages.append(
                ResidualStage(
                    current_channels + channels[level],
                    output_channels,
                    embedding_channels,
                    num_blocks,
                    dropout,
                )
            )
            current_channels = output_channels
            if level > 0:
                self.upsamples.append(Upsample(current_channels))

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_conv = _conv3x3(current_channels, out_channels)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        noise_labels: torch.Tensor,
        *,
        log2_scale_factors: torch.Tensor | None = None,
        control_residuals: Sequence[torch.Tensor] | None = None,
        well_condition: torch.Tensor | None = None,
        horizon_condition: torch.Tensor | None = None,
        well_present: torch.Tensor | None = None,
        horizon_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(x, noise_labels, log2_scale_factors, control_residuals)
        condition_residuals = self._condition_residuals(
            x,
            well_condition=well_condition,
            horizon_condition=horizon_condition,
            well_present=well_present,
            horizon_present=horizon_present,
        )
        control_residuals = self._combine_residuals(control_residuals, condition_residuals)
        embedding = self.noise_embedding(noise_labels).to(dtype=x.dtype)
        if self.scale_embedding is not None:
            assert log2_scale_factors is not None
            embedding = embedding + self.scale_embedding(log2_scale_factors).to(dtype=x.dtype)

        x = self.input_conv(x)
        skips: list[torch.Tensor] = []
        for level, stage in enumerate(self.encoder_stages):
            x = stage(x, embedding)
            if control_residuals is not None:
                residual = control_residuals[level]
                if residual.shape != x.shape:
                    raise ValueError(
                        f"Control residual {level} shape={tuple(residual.shape)} "
                        f"does not match {tuple(x.shape)}"
                    )
                x = x + residual
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)

        x = self.mid_block0(x, embedding)
        x = self.mid_attention(x)
        x = self.mid_block1(x, embedding)
        if control_residuals is not None:
            residual = control_residuals[-1]
            if residual.shape != x.shape:
                raise ValueError(
                    f"Bottleneck control residual shape={tuple(residual.shape)} "
                    f"does not match {tuple(x.shape)}"
                )
            x = x + residual

        for index, stage in enumerate(self.decoder_stages):
            x = torch.cat([x, skips.pop()], dim=1)
            x = stage(x, embedding)
            if index < len(self.upsamples):
                x = self.upsamples[index](x)
        return self.output_conv(functional.silu(self.output_norm(x)))

    def _condition_residuals(
        self,
        x: torch.Tensor,
        *,
        well_condition: torch.Tensor | None,
        horizon_condition: torch.Tensor | None,
        well_present: torch.Tensor | None,
        horizon_present: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...] | None:
        supplied = well_condition is not None or horizon_condition is not None
        if not self.conditioning_enabled:
            if supplied:
                raise ValueError(
                    "Condition maps were supplied but model.conditioning.enabled is false"
                )
            return None
        target_shapes = [
            (x.shape[-2] // (2**level), x.shape[-1] // (2**level))
            for level in range(len(self.stage_channels))
        ]
        groups: list[tuple[torch.Tensor, ...]] = []
        if well_condition is not None:
            self._validate_condition_shape("well", well_condition, x)
            assert self.well_condition_encoder is not None
            groups.append(
                self.well_condition_encoder(
                    well_condition,
                    presence=well_present,
                    target_shapes=target_shapes,
                )
            )
        elif well_present is not None and torch.any(well_present != 0.0):
            raise ValueError("well_present is nonzero but well_condition is absent")
        if horizon_condition is not None:
            self._validate_condition_shape("horizon", horizon_condition, x)
            assert self.horizon_condition_encoder is not None
            groups.append(
                self.horizon_condition_encoder(
                    horizon_condition,
                    presence=horizon_present,
                    target_shapes=target_shapes,
                )
            )
        elif horizon_present is not None and torch.any(horizon_present != 0.0):
            raise ValueError("horizon_present is nonzero but horizon_condition is absent")
        if not groups:
            return None
        return tuple(sum(parts) for parts in zip(*groups, strict=True))

    @staticmethod
    def _validate_condition_shape(name: str, condition: torch.Tensor, x: torch.Tensor) -> None:
        expected = (x.shape[0], 2, x.shape[-2], x.shape[-1])
        if tuple(condition.shape) != expected:
            raise ValueError(
                f"{name}_condition must have shape {expected}, got {tuple(condition.shape)}"
            )

    @staticmethod
    def _combine_residuals(
        first: Sequence[torch.Tensor] | None,
        second: Sequence[torch.Tensor] | None,
    ) -> Sequence[torch.Tensor] | None:
        if first is None:
            return second
        if second is None:
            return first
        if len(first) != len(second):
            raise ValueError("Control and sparse-condition residual counts differ")
        return tuple(left + right for left, right in zip(first, second, strict=True))

    def _validate_inputs(
        self,
        x: torch.Tensor,
        noise_labels: torch.Tensor,
        log2_scale_factors: torch.Tensor | None,
        control_residuals: Sequence[torch.Tensor] | None,
    ) -> None:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [batch, {self.in_channels}, height, width], got {tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError(
                f"Spatial dimensions must be divisible by {self.downsample_factor}, "
                f"got {(height, width)}"
            )
        if noise_labels.reshape(-1).numel() != x.shape[0]:
            raise ValueError("noise_labels must contain one value per sample")
        if self.use_scale_embedding:
            expected_shape = (x.shape[0], 2)
            if log2_scale_factors is None or tuple(log2_scale_factors.shape) != expected_shape:
                raise ValueError(f"log2_scale_factors must have shape {expected_shape}")
        if control_residuals is not None and len(control_residuals) != len(self.encoder_stages) + 1:
            raise ValueError(f"Expected {len(self.encoder_stages) + 1} control residuals")
