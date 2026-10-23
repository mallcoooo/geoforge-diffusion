from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from geoforge_diffusion.sampling.edm_sampler import EDMSamplerConfig, karras_noise_schedule


@dataclass(frozen=True)
class EDMDPSConfig:
    """Data-consistency correction used alongside an EDM probability-flow step.

    ``guidance_scale`` is expressed in normalized impedance units after the
    likelihood gradient has been normalized to unit RMS for each realization.
    The signal-ratio annealing keeps the correction weak while the state is
    mostly noise and progressively activates it near the denoising endpoint.
    A positive ``guidance_taper_sigma`` optionally reduces the late correction
    linearly below that noise level; zero keeps the original numerical path.
    """

    guidance_scale: float = 0.05
    guidance_anneal_power: float = 0.5
    gradient_epsilon: float = 1.0e-8
    guidance_taper_sigma: float = 0.0
    low_frequency_weight: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(self.guidance_anneal_power) or self.guidance_anneal_power < 0.0:
            raise ValueError("guidance_anneal_power must be finite and non-negative")
        if not math.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0.0:
            raise ValueError("gradient_epsilon must be finite and positive")
        if not math.isfinite(self.guidance_taper_sigma) or self.guidance_taper_sigma < 0.0:
            raise ValueError("guidance_taper_sigma must be finite and non-negative")
        if not math.isfinite(self.low_frequency_weight) or self.low_frequency_weight < 0.0:
            raise ValueError("low_frequency_weight must be finite and non-negative")


def edm_dps_sample(
    model: torch.nn.Module,
    latents: torch.Tensor,
    config: EDMSamplerConfig,
    *,
    forward_operator: Callable[[torch.Tensor], torch.Tensor],
    observation: torch.Tensor,
    dps_config: EDMDPSConfig,
    observation_mask: torch.Tensor | None = None,
    low_frequency_loss: Callable[[torch.Tensor], torch.Tensor] | None = None,
    model_kwargs: dict[str, object] | None = None,
    generator: torch.Generator | None = None,
    step_callback: Callable[[int, torch.Tensor, torch.Tensor, dict[str, float]], None]
    | None = None,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Sample an EDM prior while applying a differentiable data correction.

    ``forward_operator`` receives each current denoised estimate in the same
    normalized impedance representation as the diffusion model.  It must
    return a tensor compatible with ``observation``; an observation batch of
    one is broadcast to every requested realization.
    """

    if latents.ndim != 4:
        raise ValueError(
            f"Expected [batch, channel, height, width] latents, got {tuple(latents.shape)}"
        )
    if not torch.isfinite(latents).all():
        raise ValueError("Latents must be finite")
    if observation.ndim != 4:
        raise ValueError(f"Expected a 4D observation tensor, got {tuple(observation.shape)}")
    if observation.shape[0] not in {1, latents.shape[0]}:
        raise ValueError("Observation batch size must be one or match the latent batch size")
    if not torch.isfinite(observation).all():
        raise ValueError("Observation must be finite")

    if observation_mask is not None:
        if observation_mask.ndim != 4:
            raise ValueError(f"Expected a 4D observation mask, got {tuple(observation_mask.shape)}")
        if observation_mask.shape[0] not in {1, latents.shape[0]}:
            raise ValueError(
                "Observation-mask batch size must be one or match the latent batch size"
            )
        if observation_mask.shape[1:] != observation.shape[1:]:
            raise ValueError(
                "Observation mask must match observation channels and spatial shape; "
                f"got mask {tuple(observation_mask.shape)} and "
                f"observation {tuple(observation.shape)}"
            )
        if not torch.isfinite(observation_mask).all():
            raise ValueError("Observation mask must be finite")
        if bool((observation_mask < 0.0).any()):
            raise ValueError("Observation mask must be non-negative")
        if not bool((observation_mask > 0.0).any()):
            raise ValueError("Observation mask must contain at least one positive value")

    if dps_config.low_frequency_weight > 0 and low_frequency_loss is None:
        raise ValueError("Positive low_frequency_weight requires a low-frequency loss")
    kwargs = model_kwargs or {}
    schedule = karras_noise_schedule(config, device=latents.device, dtype=torch.float32)
    state = latents.to(dtype=torch.float32) * schedule[0]
    observation = observation.to(device=state.device, dtype=state.dtype)
    mask = (
        observation_mask.to(device=state.device, dtype=state.dtype)
        if observation_mask is not None
        else None
    )
    batch_size = state.shape[0]
    sigma_data = float(getattr(model, "sigma_data", 1.0))
    if not math.isfinite(sigma_data) or sigma_data <= 0.0:
        sigma_data = 1.0
    diagnostics: dict[str, list[float]] = {
        "data_mse": [],
        "gradient_rms": [],
        "guidance_weight": [],
    }
    if dps_config.low_frequency_weight > 0:
        diagnostics["low_frequency_loss"] = []

    for index, (sigma_current, sigma_next) in enumerate(
        zip(schedule[:-1], schedule[1:], strict=True)
    ):
        gamma = 0.0
        sigma_value = float(sigma_current)
        if config.s_min <= sigma_value <= config.s_max:
            gamma = min(config.s_churn / config.num_steps, math.sqrt(2.0) - 1.0)
        sigma_hat = sigma_current * (1.0 + gamma)
        state_hat = state
        if gamma > 0.0:
            epsilon = torch.randn(
                state.shape,
                device=state.device,
                dtype=state.dtype,
                generator=generator,
            )
            state_hat = (
                state
                + epsilon * (sigma_hat.square() - sigma_current.square()).sqrt() * config.s_noise
            )

        state_for_gradient = state_hat.detach().requires_grad_(True)
        sigma_batch = sigma_hat.expand(batch_size)
        denoised = model(state_for_gradient, sigma_batch, **kwargs).to(dtype=state.dtype)
        predicted = forward_operator(denoised)
        if predicted.ndim != 4:
            raise ValueError(
                f"Forward operator must return a 4D tensor, got {tuple(predicted.shape)}"
            )
        if predicted.shape[0] != batch_size or predicted.shape[1:] != observation.shape[1:]:
            raise ValueError(
                "Forward operator output must match observation channels and spatial shape; "
                f"got prediction {tuple(predicted.shape)} and "
                f"observation {tuple(observation.shape)}"
            )
        squared_residual = (predicted - observation).square()
        if mask is None:
            data_mse_per_sample = squared_residual.mean(dim=(1, 2, 3))
        else:
            expanded_mask = mask.expand(batch_size, -1, -1, -1)
            valid_weight = expanded_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            data_mse_per_sample = (squared_residual * expanded_mask).sum(
                dim=(1, 2, 3)
            ) / valid_weight
        guidance_loss = data_mse_per_sample
        if dps_config.low_frequency_weight > 0:
            prior_loss = low_frequency_loss(denoised)
            if prior_loss.shape != (batch_size,) or not torch.isfinite(prior_loss).all():
                raise ValueError("Low-frequency loss must be finite with one value per sample")
            guidance_loss = guidance_loss + dps_config.low_frequency_weight * prior_loss
            diagnostics["low_frequency_loss"].append(float(prior_loss.detach().mean().cpu()))
        data_gradient = torch.autograd.grad(guidance_loss.sum(), state_for_gradient)[0]
        gradient_rms = data_gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        normalized_gradient = data_gradient / gradient_rms.clamp_min(dps_config.gradient_epsilon)
        signal_ratio = sigma_data**2 / (sigma_hat.square() + sigma_data**2)
        guidance_weight = dps_config.guidance_scale * signal_ratio.pow(
            dps_config.guidance_anneal_power
        )
        if dps_config.guidance_taper_sigma > 0.0:
            guidance_weight = guidance_weight * (sigma_hat / dps_config.guidance_taper_sigma).clamp(
                max=1.0
            )

        diagnostics["data_mse"].append(float(data_mse_per_sample.detach().mean().cpu()))
        diagnostics["gradient_rms"].append(float(gradient_rms.detach().mean().cpu()))
        diagnostics["guidance_weight"].append(float(guidance_weight.detach().cpu()))

        with torch.no_grad():
            denoised_detached = denoised.detach()
            derivative = (state_hat - denoised_detached) / sigma_hat
            euler = state_hat + (sigma_next - sigma_hat) * derivative
            use_heun = config.solver == "heun" and index < config.num_steps - 1
            if use_heun:
                next_batch = sigma_next.expand(batch_size)
                denoised_next = model(euler, next_batch, **kwargs).to(dtype=state.dtype)
                derivative_next = (euler - denoised_next) / sigma_next
                prior_state = state_hat + (sigma_next - sigma_hat) * 0.5 * (
                    derivative + derivative_next
                )
            else:
                prior_state = euler

            if sigma_next > 0.0:
                state = prior_state - guidance_weight * normalized_gradient.detach()
            else:
                state = denoised_detached
        if step_callback is not None:
            step_callback(
                index + 1,
                state,
                sigma_next,
                {
                    "data_mse": diagnostics["data_mse"][-1],
                    "gradient_rms": diagnostics["gradient_rms"][-1],
                    "guidance_weight": diagnostics["guidance_weight"][-1],
                },
            )
    return state, diagnostics
