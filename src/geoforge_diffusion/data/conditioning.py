from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CONDITION_SUBSETS = ("unconditional", "well", "horizon", "both")


@dataclass(frozen=True)
class SparseConditionConfig:
    enabled: bool = False
    subset_probabilities: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    min_wells: int = 1
    max_wells: int = 4
    well_min_spacing_fraction: float = 0.08
    well_max_bottom_truncation_fraction: float = 0.5
    horizon_archive_name: str = "geology.npz"
    horizon_key: str = "horizon_surfaces_time_samples"
    min_horizons: int = 1
    max_horizons: int = 4
    horizon_min_spacing_fraction: float = 0.04
    horizon_vertical_bins: int = 3
    horizon_sigma_pixels: float = 1.5
    minimum_horizon_valid_fraction: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> SparseConditionConfig:
        if not values or not bool(values.get("enabled", False)):
            return cls()

        raw_probabilities = values.get("subset_probabilities", {})
        if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(
            CONDITION_SUBSETS
        ):
            raise ValueError(
                f"conditioning.subset_probabilities must exactly contain {list(CONDITION_SUBSETS)}"
            )
        probabilities = tuple(float(raw_probabilities[name]) for name in CONDITION_SUBSETS)
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("Condition subset probabilities must be finite and non-negative")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"Condition subset probabilities must sum to one, got {total}")

        well = values.get("well", {})
        horizon = values.get("horizon", {})
        if not isinstance(well, dict) or not isinstance(horizon, dict):
            raise ValueError("conditioning.well and conditioning.horizon must be mappings")
        result = cls(
            enabled=True,
            subset_probabilities=probabilities,
            min_wells=int(well.get("min_wells", 1)),
            max_wells=int(well.get("max_wells", 4)),
            well_min_spacing_fraction=float(well.get("min_spacing_fraction", 0.08)),
            well_max_bottom_truncation_fraction=float(
                well.get("max_bottom_truncation_fraction", 0.5)
            ),
            horizon_archive_name=str(horizon.get("archive_name", "geology.npz")),
            horizon_key=str(horizon.get("key", "horizon_surfaces_time_samples")),
            min_horizons=int(horizon.get("min_horizons", 1)),
            max_horizons=int(horizon.get("max_horizons", 4)),
            horizon_min_spacing_fraction=float(horizon.get("min_spacing_fraction", 0.04)),
            horizon_vertical_bins=int(horizon.get("vertical_bins", 3)),
            horizon_sigma_pixels=float(horizon.get("gaussian_sigma_pixels", 1.5)),
            minimum_horizon_valid_fraction=float(horizon.get("minimum_valid_fraction", 0.25)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.min_wells <= 0 or self.max_wells < self.min_wells:
            raise ValueError("well min_wells/max_wells must satisfy 0 < min_wells <= max_wells")
        if not 0.0 < self.well_min_spacing_fraction < 1.0:
            raise ValueError("well min_spacing_fraction must lie in (0, 1)")
        if not 0.0 <= self.well_max_bottom_truncation_fraction <= 0.5:
            raise ValueError("well max_bottom_truncation_fraction must lie in [0, 0.5]")
        if self.min_horizons <= 0 or self.max_horizons < self.min_horizons:
            raise ValueError(
                "horizon min_horizons/max_horizons must satisfy 0 < min_horizons <= max_horizons"
            )
        if not 0.0 < self.horizon_min_spacing_fraction < 1.0:
            raise ValueError("horizon min_spacing_fraction must lie in (0, 1)")
        if self.horizon_vertical_bins <= 0 or self.min_horizons < self.horizon_vertical_bins:
            raise ValueError(
                "horizon vertical_bins must be positive and no greater than min_horizons"
            )
        if not math.isfinite(self.horizon_sigma_pixels) or self.horizon_sigma_pixels <= 0.0:
            raise ValueError("horizon gaussian_sigma_pixels must be finite and positive")
        if not 0.0 < self.minimum_horizon_valid_fraction <= 1.0:
            raise ValueError("horizon minimum_valid_fraction must lie in (0, 1]")
        if not self.horizon_archive_name or not self.horizon_key:
            raise ValueError("horizon archive_name and key cannot be empty")


def sample_condition_subset(
    rng: np.random.Generator,
    probabilities: tuple[float, float, float, float],
) -> str:
    return CONDITION_SUBSETS[int(rng.choice(len(CONDITION_SUBSETS), p=probabilities))]


def make_well_condition(
    normalized_impedance: np.ndarray,
    *,
    rng: np.random.Generator,
    min_wells: int,
    max_wells: int,
    minimum_spacing_pixels: int,
    max_bottom_truncation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create ``[value, mask]`` maps from complete vertical synthetic wells."""

    if normalized_impedance.ndim != 2:
        raise ValueError(
            f"Expected a 2D normalized impedance field, got {normalized_impedance.shape}"
        )
    height, width = normalized_impedance.shape
    number = min(int(rng.integers(min_wells, max_wells + 1)), width)
    columns = _sample_spaced_indices(width, number, minimum_spacing_pixels, rng)
    values = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    maximum_cut = int(math.floor(height * max_bottom_truncation_fraction))
    for column in columns:
        cut = int(rng.integers(0, maximum_cut + 1)) if maximum_cut > 0 else 0
        valid_end = height - cut
        values[:valid_end, column] = normalized_impedance[:valid_end, column]
        mask[:valid_end, column] = 1.0
    return np.stack([values, mask]), columns.astype(np.int64, copy=False)


def make_horizon_condition(
    surfaces_time_samples: np.ndarray,
    *,
    crop_top: int,
    crop_left: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    flipped: bool,
    rng: np.random.Generator,
    min_horizons: int,
    max_horizons: int,
    minimum_spacing_pixels: int,
    vertical_bins: int,
    gaussian_sigma_pixels: float,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a random subset of true horizon surfaces as ``[heatmap, pick_mask]``."""

    if surfaces_time_samples.ndim != 2:
        raise ValueError(f"Expected [horizon, x] surfaces, got {surfaces_time_samples.shape}")
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if crop_left < 0 or crop_left + source_w > surfaces_time_samples.shape[1]:
        raise ValueError("Horizon crop lies outside the available horizontal samples")

    local_x = np.rint((np.arange(target_w, dtype=np.float64) + 0.5) * source_w / target_w - 0.5)
    local_x = np.clip(local_x.astype(np.int64), 0, source_w - 1)
    if flipped:
        local_x = source_w - 1 - local_x
    sampled = surfaces_time_samples[:, crop_left + local_x].astype(np.float64, copy=False)
    local_time = sampled - float(crop_top)
    valid = np.isfinite(local_time) & (local_time >= 0.0) & (local_time < source_h)
    valid_fractions = valid.mean(axis=1)
    candidates = np.flatnonzero(valid_fractions >= minimum_valid_fraction)
    if candidates.size == 0:
        raise ValueError(
            "No horizon intersects this crop with the configured minimum valid fraction; "
            "lower conditioning.horizon.minimum_valid_fraction"
        )

    if candidates.size < min_horizons:
        raise ValueError(
            f"Only {candidates.size} valid horizons intersect the crop; "
            f"at least {min_horizons} are required"
        )
    number = min(int(rng.integers(min_horizons, max_horizons + 1)), int(candidates.size))
    target_time = (local_time + 0.5) * target_h / source_h - 0.5
    positions = np.asarray(
        [np.median(target_time[index, valid[index]]) for index in candidates],
        dtype=np.float64,
    )
    selected = _select_stratified_horizons(
        candidates,
        positions,
        count=number,
        target_height=target_h,
        minimum_spacing=minimum_spacing_pixels,
        vertical_bins=vertical_bins,
        rng=rng,
    )
    heatmap = np.zeros((target_h, target_w), dtype=np.float32)
    pick_mask = np.zeros((target_h, target_w), dtype=np.float32)
    radius = max(1, int(math.ceil(3.0 * gaussian_sigma_pixels)))
    columns = np.arange(target_w, dtype=np.int64)

    for horizon_index in selected:
        horizon_valid = valid[horizon_index]
        horizon_time = (local_time[horizon_index] + 0.5) * target_h / source_h - 0.5
        center_rows = np.where(horizon_valid, np.rint(horizon_time), 0.0).astype(
            np.int64, copy=False
        )
        center_valid = horizon_valid & (center_rows >= 0) & (center_rows < target_h)
        valid_columns = columns[center_valid]
        valid_rows = center_rows[center_valid]
        pick_mask[valid_rows, valid_columns] = 1.0
        for offset in range(-radius, radius + 1):
            rows = valid_rows + offset
            inside = (rows >= 0) & (rows < target_h)
            weight = np.float32(math.exp(-0.5 * (offset / gaussian_sigma_pixels) ** 2))
            heatmap[rows[inside], valid_columns[inside]] = np.maximum(
                heatmap[rows[inside], valid_columns[inside]],
                weight,
            )

    return np.stack([heatmap, pick_mask]), selected.astype(np.int64, copy=False)


def _sample_spaced_indices(
    size: int,
    count: int,
    minimum_spacing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size <= 0 or count <= 0 or minimum_spacing <= 0:
        raise ValueError("Spaced sampling arguments must be positive")
    compressed_size = size - (count - 1) * (minimum_spacing - 1)
    if compressed_size < count:
        raise ValueError(
            f"Cannot place {count} conditions in {size} pixels with "
            f"minimum spacing {minimum_spacing}"
        )
    compressed = np.sort(rng.choice(compressed_size, size=count, replace=False))
    return compressed + np.arange(count, dtype=np.int64) * (minimum_spacing - 1)


def _select_stratified_horizons(
    candidates: np.ndarray,
    positions: np.ndarray,
    *,
    count: int,
    target_height: int,
    minimum_spacing: int,
    vertical_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if candidates.shape != positions.shape:
        raise ValueError("Horizon candidates and positions must have matching shapes")
    if candidates.size < vertical_bins:
        raise ValueError("Fewer valid horizons than configured vertical groups")
    bin_indexes = np.clip(
        (positions * vertical_bins / target_height).astype(np.int64),
        0,
        vertical_bins - 1,
    )
    if any(not np.any(bin_indexes == index) for index in range(vertical_bins)):
        bin_indexes = np.empty(candidates.size, dtype=np.int64)
        ordered = np.argsort(positions, kind="stable")
        for bin_index, group in enumerate(np.array_split(ordered, vertical_bins)):
            bin_indexes[group] = bin_index

    for _ in range(128):
        selected_local: list[int] = []
        for bin_index in rng.permutation(vertical_bins):
            choices = np.flatnonzero(bin_indexes == bin_index)
            choices = rng.permutation(choices)
            accepted = next(
                (
                    int(choice)
                    for choice in choices
                    if all(
                        abs(float(positions[choice] - positions[previous])) >= minimum_spacing
                        for previous in selected_local
                    )
                ),
                None,
            )
            if accepted is None:
                break
            selected_local.append(accepted)
        if len(selected_local) != vertical_bins:
            continue
        if len(selected_local) == count:
            local = np.asarray(selected_local, dtype=np.int64)
            selected = candidates[local]
            return selected[np.argsort(positions[local])]
        remaining = rng.permutation(
            np.asarray(
                [index for index in range(len(candidates)) if index not in selected_local],
                dtype=np.int64,
            )
        )
        for choice in remaining:
            if all(
                abs(float(positions[choice] - positions[previous])) >= minimum_spacing
                for previous in selected_local
            ):
                selected_local.append(int(choice))
                if len(selected_local) == count:
                    selected = candidates[np.asarray(selected_local, dtype=np.int64)]
                    order = np.argsort(positions[np.asarray(selected_local, dtype=np.int64)])
                    return selected[order]
    raise ValueError(
        f"Cannot select {count} horizons spanning {vertical_bins} vertical bins with "
        f"minimum spacing {minimum_spacing}"
    )
