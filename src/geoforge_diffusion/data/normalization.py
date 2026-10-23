from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GlobalAIStats:
    """Training-set statistics for one reversible global value transform."""

    domain: str
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]
    num_values: int
    num_samples: int

    def __post_init__(self) -> None:
        if self.domain not in {"ai", "log_ai"}:
            raise ValueError(f"Unsupported value domain: {self.domain!r}")
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError(f"Standard deviation must be positive, got {self.std}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalAIStats:
        return cls(
            domain=str(data["domain"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
            minimum=float(data["minimum"]),
            maximum=float(data["maximum"]),
            quantiles={str(key): float(value) for key, value in data["quantiles"].items()},
            num_values=int(data["num_values"]),
            num_samples=int(data["num_samples"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "quantiles": self.quantiles,
            "num_values": self.num_values,
            "num_samples": self.num_samples,
            "normalization": "global_affine_without_clipping",
        }


def to_value_domain(ai: np.ndarray, domain: str) -> np.ndarray:
    values = np.asarray(ai, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Impedance values must be finite and strictly positive")
    if domain == "ai":
        return values
    if domain == "log_ai":
        return np.log(values, dtype=np.float32)
    raise ValueError(f"Unsupported value domain: {domain!r}")


def normalize_ai(ai: np.ndarray, stats: GlobalAIStats) -> np.ndarray:
    values = to_value_domain(ai, stats.domain)
    return ((values - stats.mean) / stats.std).astype(np.float32, copy=False)


def denormalize_ai(normalized: np.ndarray, stats: GlobalAIStats) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float32) * stats.std + stats.mean
    if stats.domain == "log_ai":
        values = np.exp(values, dtype=np.float32)
    return values.astype(np.float32, copy=False)
