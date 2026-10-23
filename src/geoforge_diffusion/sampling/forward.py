"""Differentiable post-stack operator for seismic-guided sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from geoforge_diffusion.data.normalization import GlobalAIStats


class PoststackForward(torch.nn.Module):
    """Apply d = 0.5 W D log(Z) independently along each trace."""

    def __init__(self, stats: GlobalAIStats, wavelet: np.ndarray, seismic_scale: float):
        super().__init__()
        wavelet = np.asarray(wavelet, dtype=np.float32)
        if wavelet.ndim != 1 or wavelet.size < 3 or wavelet.size % 2 != 1:
            raise ValueError("Wavelet must be a 1D array with an odd length of at least three")
        if not np.isfinite(wavelet).all() or not np.isfinite(seismic_scale) or seismic_scale <= 0:
            raise ValueError("Wavelet must be finite and seismic scale must be positive")
        self.stats = stats
        self.seismic_scale = seismic_scale
        self.register_buffer("kernel", torch.from_numpy(wavelet[::-1].copy()).view(1, 1, -1, 1))

    def forward(self, normalized):
        with torch.autocast(normalized.device.type, enabled=False):
            values = normalized.float() * self.stats.std + self.stats.mean
            log_ai = values if self.stats.domain == "log_ai" else values.clamp_min(1e-6).log()
            derivative = functional.pad(log_ai[:, :, 1:] - log_ai[:, :, :-1], (0, 0, 0, 1))
            return (
                functional.conv2d(
                    0.5 * derivative, self.kernel, padding=(self.kernel.shape[2] // 2, 0)
                )
                / self.seismic_scale
            )
