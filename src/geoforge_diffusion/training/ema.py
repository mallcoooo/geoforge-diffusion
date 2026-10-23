from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    """A full-model EMA with an optional early-update warmup."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, warmup: bool = True) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie between zero and one")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.decay
        if self.warmup:
            decay = min(decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        ema_parameters = dict(self.module.named_parameters())
        model_parameters = dict(model.named_parameters())
        if ema_parameters.keys() != model_parameters.keys():
            raise ValueError("EMA and online model parameter structures differ")
        for name, ema_parameter in ema_parameters.items():
            source = model_parameters[name].detach()
            ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)

        ema_buffers = dict(self.module.named_buffers())
        model_buffers = dict(model.named_buffers())
        if ema_buffers.keys() != model_buffers.keys():
            raise ValueError("EMA and online model buffer structures differ")
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(model_buffers[name])

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "model": self.module.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if float(state["decay"]) != self.decay or bool(state["warmup"]) != self.warmup:
            raise ValueError("EMA configuration differs from checkpoint")
        self.module.load_state_dict(state["model"])
        self.num_updates = int(state["num_updates"])
