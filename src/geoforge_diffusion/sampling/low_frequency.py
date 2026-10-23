"""Differentiable low-frequency log-impedance constraint for DPS sampling."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional

from geoforge_diffusion.data.normalization import GlobalAIStats


class LowFrequencyLogImpedancePrior(torch.nn.Module):
    """Compare Gaussian-filtered log AI with the already smoothed TV background.

    Filtering uses the same separable Gaussian and nearest-edge policy as TV.
    The background is the target of the filter; it is not smoothed a second time.
    Network padding is cropped before filtering and never enters the prior loss.
    """

    def __init__(
        self,
        stats: GlobalAIStats,
        background_physical: np.ndarray,
        *,
        ai_scale: float,
        time_sigma_samples: float,
        trace_sigma: float,
        log_std: float,
        mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        background = np.asarray(background_physical, dtype=np.float64)
        if background.ndim != 2 or not np.isfinite(background).all() or np.any(background <= 0):
            raise ValueError("Low-frequency background must be a finite positive 2D array")
        for name, value in (
            ("ai_scale", ai_scale),
            ("time_sigma_samples", time_sigma_samples),
            ("log_std", log_std),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(trace_sigma) or trace_sigma < 0:
            raise ValueError("trace_sigma must be finite and non-negative")
        if stats.domain not in {"ai", "log_ai"}:
            raise ValueError(f"Unsupported impedance statistics domain: {stats.domain}")
        support = np.ones_like(background) if mask is None else np.asarray(mask, dtype=np.float64)
        if support.shape != background.shape or not np.isfinite(support).all():
            raise ValueError("Low-frequency mask must match the background and be finite")
        if np.any(support < 0) or not np.any(support > 0):
            raise ValueError("Low-frequency mask must be nonnegative with positive support")
        self.domain, self.mean, self.std = stats.domain, float(stats.mean), float(stats.std)
        self.log_std = float(log_std)
        self.height, self.width = background.shape
        self.register_buffer(
            "target_log", torch.from_numpy(np.log(background * ai_scale)).float()[None, None]
        )
        self.register_buffer("mask", torch.from_numpy(support).float()[None, None])
        self.register_buffer("time_kernel", self._kernel(time_sigma_samples).view(1, 1, -1, 1))
        self.register_buffer("trace_kernel", self._kernel(trace_sigma).view(1, 1, 1, -1))

    @staticmethod
    def _kernel(sigma: float) -> torch.Tensor:
        if sigma == 0:
            return torch.ones(1)
        radius = int(4.0 * sigma + 0.5)
        offset = torch.arange(-radius, radius + 1, dtype=torch.float64)
        weights = torch.exp(-0.5 * (offset / sigma).square())
        return (weights / weights.sum()).float()

    def forward(self, normalized_ai: torch.Tensor) -> torch.Tensor:
        if normalized_ai.ndim != 4 or normalized_ai.shape[1] != 1:
            raise ValueError("Low-frequency prior expects [batch, 1, time, trace] impedance")
        if normalized_ai.shape[2] < self.height or normalized_ai.shape[3] < self.width:
            raise ValueError("Low-frequency prediction is smaller than the background")
        with torch.autocast(device_type=normalized_ai.device.type, enabled=False):
            values = normalized_ai[..., : self.height, : self.width].float() * self.std + self.mean
            log_ai = values if self.domain == "log_ai" else torch.log(values.clamp_min(1e-6))
            time_radius = self.time_kernel.shape[2] // 2
            low = functional.conv2d(
                functional.pad(log_ai, (0, 0, time_radius, time_radius), mode="replicate"),
                self.time_kernel,
            )
            trace_radius = self.trace_kernel.shape[3] // 2
            if trace_radius:
                low = functional.conv2d(
                    functional.pad(low, (trace_radius, trace_radius, 0, 0), mode="replicate"),
                    self.trace_kernel,
                )
            squared_error = ((low - self.target_log) / self.log_std).square()
            return (squared_error * self.mask).sum(dim=(1, 2, 3)) / self.mask.sum()
