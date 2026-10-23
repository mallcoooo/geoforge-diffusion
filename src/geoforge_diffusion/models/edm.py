from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _sigma_vector(
    sigma: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.as_tensor(sigma, device=device, dtype=dtype).reshape(-1)
    if values.numel() == 1:
        values = values.expand(batch_size)
    if values.numel() != batch_size or not torch.isfinite(values).all() or torch.any(values <= 0.0):
        raise ValueError("sigma must be finite, positive, and contain one value per sample")
    return values


class EDMPreconditioner(nn.Module):
    """EDM preconditioning that returns a direct denoised estimate."""

    def __init__(self, network: nn.Module, sigma_data: float = 1.0) -> None:
        super().__init__()
        if sigma_data <= 0.0:
            raise ValueError("sigma_data must be positive")
        self.network = network
        self.sigma_data = float(sigma_data)

    def network_inputs(
        self, x: torch.Tensor, sigma: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_vector = _sigma_vector(
            sigma,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        sigma_squared = sigma_vector.square()
        data_squared = self.sigma_data**2
        c_in = torch.rsqrt(sigma_squared + data_squared)
        c_noise = sigma_vector.log() / 4.0
        return x * c_in[:, None, None, None], c_noise

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor | float, **network_kwargs: object
    ) -> torch.Tensor:
        sigma_vector = _sigma_vector(
            sigma,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        sigma_squared = sigma_vector.square()
        data_squared = self.sigma_data**2
        c_skip = data_squared / (sigma_squared + data_squared)
        c_out = sigma_vector * self.sigma_data / torch.sqrt(sigma_squared + data_squared)
        network_input, noise_labels = self.network_inputs(x, sigma_vector)
        model_output = self.network(network_input, noise_labels, **network_kwargs)
        return c_skip[:, None, None, None] * x + c_out[:, None, None, None] * model_output


@dataclass(frozen=True)
class EDMLossConfig:
    sigma_data: float = 1.0
    p_mean: float = -0.5
    p_std: float = 1.2


class EDMLoss:
    """EDM weighted direct-denoising objective."""

    def __init__(self, config: EDMLossConfig) -> None:
        if config.sigma_data <= 0.0 or config.p_std <= 0.0:
            raise ValueError("sigma_data and p_std must be positive")
        self.config = config

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
        return torch.exp(normal * self.config.p_std + self.config.p_mean)

    def __call__(
        self,
        model: EDMPreconditioner,
        clean: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        model_kwargs: dict[str, object] | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        sigma = self.sample_sigma(
            clean.shape[0],
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        sigma_data = self.config.sigma_data
        weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
        noisy = clean + noise * sigma[:, None, None, None]
        denoised = model(noisy, sigma, **(model_kwargs or {}))
        per_sample = (weight[:, None, None, None] * (denoised - clean).square()).mean(dim=(1, 2, 3))
        return per_sample.mean() if reduction == "mean" else per_sample
