from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geoforge_diffusion.data.normalization import GlobalAIStats, to_value_domain
from geoforge_diffusion.data.raw import (
    array_fingerprint,
    discover_sample_ids,
    load_impedance_time_x,
    validate_impedance,
)
from geoforge_diffusion.data.schema import BucketSpec, parse_buckets
from geoforge_diffusion.utils.io import write_ids, write_json


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        local_count = int(flat.size)
        local_mean = float(flat.mean())
        local_m2 = float(np.square(flat - local_mean).sum())
        if self.count == 0:
            self.count = local_count
            self.mean = local_mean
            self.m2 = local_m2
            return
        delta = local_mean - self.mean
        total = self.count + local_count
        self.m2 += local_m2 + delta * delta * self.count * local_count / total
        self.mean += delta * local_count / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            raise ValueError("At least two values are required to estimate standard deviation")
        return float(np.sqrt(self.m2 / self.count))


def split_sample_ids(
    sample_ids: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    if not sample_ids:
        raise ValueError("No sample ids were provided")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("Split fractions must lie between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than one")
    ordered = np.asarray(sorted(sample_ids), dtype=object)
    shuffled = ordered[np.random.default_rng(seed).permutation(len(ordered))].tolist()
    train_end = int(round(len(shuffled) * train_fraction))
    valid_end = train_end + int(round(len(shuffled) * validation_fraction))
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def prepare_metadata(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    raw_dir_override: str | Path | None = None,
    force: bool = False,
) -> Path:
    data_config = config["data"]
    project_config = config["project"]
    stats_config = data_config["statistics"]
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Metadata already exists: {manifest_path}; use --force to overwrite")

    raw_dir = Path(raw_dir_override or data_config["raw_dir"]).resolve()
    sample_ids = discover_sample_ids(raw_dir)
    buckets = parse_buckets(data_config["buckets"])
    split_config = data_config["split"]
    seed = int(project_config["seed"])
    splits = split_sample_ids(
        sample_ids,
        seed=seed,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
    )
    if any(not ids for ids in splits.values()):
        raise ValueError("Dataset is too small to form nonempty train, validation and test splits")
    selected_train_ids = _select_ids(
        splits["train"], int(stats_config["sample_count"]), np.random.default_rng(seed + 17)
    )
    stats, audit, sample_shapes = _audit_and_estimate(
        sample_ids=sample_ids,
        selected_train_ids=set(selected_train_ids),
        raw_dir=raw_dir,
        target_key=str(data_config.get("target_key", "ip")),
        domain=str(stats_config.get("domain", "ai")),
        pixels_per_sample=int(stats_config["pixels_per_sample"]),
        buckets=buckets,
        seed=seed + 31,
    )

    for split_name, ids in splits.items():
        write_ids(output / "splits" / f"{split_name}.txt", ids)
    write_json(output / "stats.json", stats.to_dict())
    write_json(output / "audit.json", audit)
    validation_crops = build_validation_crop_records(
        splits["validation"],
        buckets,
        sample_shapes=sample_shapes,
        seed=seed + 47,
    )
    write_json(
        output / "validation_crops.json",
        {
            "version": 1,
            "seed": seed + 47,
            "records": validation_crops,
        },
    )
    write_json(
        manifest_path,
        {
            "version": 1,
            "raw_dir": Path(os.path.relpath(raw_dir, output.resolve())).as_posix(),
            "archive_name": "data.npz",
            "target_key": str(data_config.get("target_key", "ip")),
            "raw_layout": "x_time",
            "training_layout": "time_x",
            "seed": seed,
            "split_counts": {name: len(ids) for name, ids in splits.items()},
            "buckets": [bucket.to_dict() for bucket in buckets],
            "statistics": {
                "domain": stats.domain,
                "selected_training_samples": len(selected_train_ids),
                "pixels_per_sample": int(stats_config["pixels_per_sample"]),
                "source": "raw training pixels before spatial resampling",
            },
            "spatial_preprocessing": {
                "order": "crop_then_downsample",
                "downsampling": "area_mean",
                "geometric_distortion": "rejected_by_bucket_validation",
            },
            "spatial_coordinates": "none",
            "files": {
                "audit": "audit.json",
                "stats": "stats.json",
                "splits": "splits",
                "validation_crops": "validation_crops.json",
            },
        },
    )
    return manifest_path


def _audit_and_estimate(
    *,
    sample_ids: list[str],
    selected_train_ids: set[str],
    raw_dir: Path,
    target_key: str,
    domain: str,
    pixels_per_sample: int,
    buckets: list[BucketSpec],
    seed: int,
) -> tuple[GlobalAIStats, dict[str, Any], dict[str, tuple[int, int]]]:
    if pixels_per_sample <= 0:
        raise ValueError("pixels_per_sample must be positive")
    rng = np.random.default_rng(seed)
    moments = RunningMoments()
    quantile_parts: list[np.ndarray] = []
    stats_minimum = np.inf
    stats_maximum = -np.inf
    dataset_minimum = np.inf
    dataset_maximum = -np.inf
    time_sizes: list[int] = []
    horizontal_sizes: set[int] = set()
    fingerprints: dict[str, list[str]] = {}
    sample_shapes: dict[str, tuple[int, int]] = {}
    required_time = max(bucket.source_shape[0] for bucket in buckets)
    required_horizontal = max(bucket.source_shape[1] for bucket in buckets)

    for index, sample_id in enumerate(sample_ids, start=1):
        impedance = load_impedance_time_x(raw_dir, sample_id, target_key)
        validate_impedance(impedance, sample_id)
        time_size, horizontal_size = impedance.shape
        sample_shapes[sample_id] = (int(time_size), int(horizontal_size))
        if time_size < required_time or horizontal_size < required_horizontal:
            raise ValueError(
                f"{sample_id} shape={impedance.shape} cannot supply largest source window "
                f"{(required_time, required_horizontal)}"
            )
        time_sizes.append(int(time_size))
        horizontal_sizes.add(int(horizontal_size))
        dataset_minimum = min(dataset_minimum, float(impedance.min()))
        dataset_maximum = max(dataset_maximum, float(impedance.max()))
        digest = array_fingerprint(impedance)
        fingerprints.setdefault(digest, []).append(sample_id)

        if sample_id in selected_train_ids:
            domain_values = to_value_domain(impedance, domain).reshape(-1)
            sampled = _sample_pixels(domain_values, pixels_per_sample, rng)
            moments.update(sampled)
            quantile_parts.append(sampled)
            stats_minimum = min(stats_minimum, float(domain_values.min()))
            stats_maximum = max(stats_maximum, float(domain_values.max()))
        if index % 100 == 0 or index == len(sample_ids):
            print(f"Audited {index}/{len(sample_ids)} raw samples")

    quantile_values = np.concatenate(quantile_parts)
    stats = GlobalAIStats(
        domain=domain,
        mean=moments.mean,
        std=moments.std,
        minimum=float(stats_minimum),
        maximum=float(stats_maximum),
        quantiles={
            "p001": float(np.quantile(quantile_values, 0.001)),
            "p01": float(np.quantile(quantile_values, 0.01)),
            "p05": float(np.quantile(quantile_values, 0.05)),
            "p50": float(np.quantile(quantile_values, 0.50)),
            "p95": float(np.quantile(quantile_values, 0.95)),
            "p99": float(np.quantile(quantile_values, 0.99)),
            "p999": float(np.quantile(quantile_values, 0.999)),
        },
        num_values=moments.count,
        num_samples=len(selected_train_ids),
    )
    duplicate_groups = [ids for ids in fingerprints.values() if len(ids) > 1]
    return (
        stats,
        {
            "validated_samples": len(sample_ids),
            "time_samples": {
                "minimum": min(time_sizes),
                "maximum": max(time_sizes),
            },
            "horizontal_samples": sorted(horizontal_sizes),
            "impedance": {
                "minimum": float(dataset_minimum),
                "maximum": float(dataset_maximum),
                "finite": True,
                "strictly_positive": True,
            },
            "duplicate_detection": {
                "algorithm": "blake2b-128 over float32 time_x impedance bytes",
                "duplicate_group_count": len(duplicate_groups),
                "groups": duplicate_groups,
            },
            "largest_required_source_shape": [required_time, required_horizontal],
        },
        sample_shapes,
    )


def build_validation_crop_records(
    sample_ids: list[str],
    buckets: list[BucketSpec],
    *,
    sample_shapes: dict[str, tuple[int, int]],
    seed: int,
) -> list[dict[str, Any]]:
    """Create repeatable validation crops shared by buckets with the same FOV."""

    records: list[dict[str, Any]] = []
    for bucket in buckets:
        source_h, source_w = bucket.source_shape
        for sample_id in sample_ids:
            if sample_id not in sample_shapes:
                raise KeyError(f"Missing audited shape for validation sample {sample_id}")
            sample_h, sample_w = sample_shapes[sample_id]
            max_top = sample_h - source_h
            max_left = sample_w - source_w
            if max_top < 0 or max_left < 0:
                raise ValueError(
                    f"{sample_id} shape={sample_shapes[sample_id]} is too small for {bucket.name}"
                )
            family_key = f"{sample_id}:{source_h}:{source_w}"
            family_seed = zlib.crc32(family_key.encode("utf-8"))
            rng = np.random.default_rng(np.random.SeedSequence([seed, family_seed]))
            top = int(rng.integers(0, max_top + 1))
            left = int(rng.integers(0, max_left + 1))
            records.append(
                {
                    "sample_id": sample_id,
                    "bucket": bucket.name,
                    "source_shape": list(bucket.source_shape),
                    "target_shape": list(bucket.target_shape),
                    "scale_factors": list(bucket.scale_factors),
                    "crop_top": top,
                    "crop_left": left,
                    "horizontal_flip": False,
                }
            )
    return records


def _sample_pixels(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= count:
        return values.astype(np.float32, copy=False)
    indexes = rng.choice(values.size, size=count, replace=False)
    return values[indexes].astype(np.float32, copy=False)


def _select_ids(sample_ids: list[str], count: int, rng: np.random.Generator) -> list[str]:
    if count <= 0:
        raise ValueError("statistics.sample_count must be positive")
    selected = min(count, len(sample_ids))
    indexes = rng.choice(len(sample_ids), size=selected, replace=False)
    return [sample_ids[int(index)] for index in indexes]
