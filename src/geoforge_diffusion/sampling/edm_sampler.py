from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EDMSamplerConfig:
    num_steps: int = 32
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    solver: str = "heun"
    s_churn: float = 0.0
    s_min: float = 0.0
    s_max: float = math.inf
    s_noise: float = 1.0

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("EDM sampling requires at least two steps")
        if self.sigma_min <= 0.0 or self.sigma_max <= self.sigma_min:
            raise ValueError("Sampler sigmas must be positive with sigma_max > sigma_min")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if self.s_churn < 0.0 or self.s_noise <= 0.0 or self.s_max < self.s_min:
            raise ValueError("Invalid stochastic sampling configuration")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EDMSamplerConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


def karras_noise_schedule(
    config: EDMSamplerConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, config.num_steps, device=device, dtype=dtype)
    minimum = config.sigma_min ** (1.0 / config.rho)
    maximum = config.sigma_max ** (1.0 / config.rho)
    sigmas = (maximum + ramp * (minimum - maximum)).pow(config.rho)
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])


@torch.no_grad()
def edm_sample(
    model: torch.nn.Module,
    latents: torch.Tensor,
    config: EDMSamplerConfig,
    *,
    model_kwargs: dict[str, object] | None = None,
    generator: torch.Generator | None = None,
    step_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Solve the EDM probability-flow ODE with Euler or Heun updates.

    The optional callback observes (completed step, state, sigma), including the
    initial noisy state at step 0. Copy retained states; do not mutate them.
    """

    if latents.ndim != 4:
        raise ValueError(
            f"Expected [batch, channel, height, width] latents, got {tuple(latents.shape)}"
        )
    if not torch.isfinite(latents).all():
        raise ValueError("Latents must be finite")
    kwargs = model_kwargs or {}
    schedule = karras_noise_schedule(config, device=latents.device, dtype=torch.float32)
    state = latents.to(dtype=torch.float32) * schedule[0]
    batch_size = state.shape[0]
    if step_callback is not None:
        step_callback(0, state, schedule[0])

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

        sigma_batch = sigma_hat.expand(batch_size)
        denoised = model(state_hat, sigma_batch, **kwargs).to(dtype=state.dtype)
        derivative = (state_hat - denoised) / sigma_hat
        euler = state_hat + (sigma_next - sigma_hat) * derivative

        use_heun = config.solver == "heun" and index < config.num_steps - 1
        if use_heun:
            next_batch = sigma_next.expand(batch_size)
            denoised_next = model(euler, next_batch, **kwargs).to(dtype=state.dtype)
            derivative_next = (euler - denoised_next) / sigma_next
            state = state_hat + (sigma_next - sigma_hat) * 0.5 * (derivative + derivative_next)
        else:
            state = euler
        if step_callback is not None:
            step_callback(index + 1, state, sigma_next)
    return state
