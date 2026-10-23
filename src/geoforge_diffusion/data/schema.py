from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any


@dataclass(frozen=True)
class BucketSpec:
    """One fixed source field of view and its output tensor geometry."""

    name: str
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    probability: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Bucket name cannot be empty")
        if len(self.source_shape) != 2 or len(self.target_shape) != 2:
            raise ValueError("Source and target shapes must be [time, horizontal]")
        if any(value <= 0 for value in (*self.source_shape, *self.target_shape)):
            raise ValueError("Bucket dimensions must be positive")
        if any(
            target > source
            for source, target in zip(self.source_shape, self.target_shape, strict=True)
        ):
            raise ValueError(f"Upsampling is not supported in bucket {self.name!r}")
        scale_t, scale_x = self.scale_factors
        if not isclose(scale_t, round(scale_t)) or not isclose(scale_x, round(scale_x)):
            raise ValueError(f"Bucket {self.name!r} must use integer downsampling factors")
        if not isclose(scale_t, scale_x):
            raise ValueError(
                f"Bucket {self.name!r} would distort geometry: "
                f"time scale={scale_t:g}, horizontal scale={scale_x:g}"
            )
        if self.probability <= 0.0:
            raise ValueError("Bucket probability must be positive")

    @property
    def scale_factors(self) -> tuple[float, float]:
        return (
            self.source_shape[0] / self.target_shape[0],
            self.source_shape[1] / self.target_shape[1],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BucketSpec:
        return cls(
            name=str(data["name"]),
            source_shape=_shape(data["source_shape"], "source_shape"),
            target_shape=_shape(data["target_shape"], "target_shape"),
            probability=float(data.get("probability", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "scale_factors": list(self.scale_factors),
            "probability": self.probability,
        }


def parse_buckets(data: list[dict[str, Any]]) -> list[BucketSpec]:
    if not data:
        raise ValueError("At least one resolution bucket is required")
    buckets = [BucketSpec.from_dict(item) for item in data]
    names = [bucket.name for bucket in buckets]
    if len(names) != len(set(names)):
        raise ValueError("Bucket names must be unique")
    total = sum(bucket.probability for bucket in buckets)
    return [
        BucketSpec(
            name=bucket.name,
            source_shape=bucket.source_shape,
            target_shape=bucket.target_shape,
            probability=bucket.probability / total,
        )
        for bucket in buckets
    ]


def _shape(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [time, horizontal]")
    return int(value[0]), int(value[1])
