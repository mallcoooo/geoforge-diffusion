from __future__ import annotations

import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from geoforge_diffusion.data.conditioning import (
    SparseConditionConfig,
    make_horizon_condition,
    make_well_condition,
    sample_condition_subset,
)
from geoforge_diffusion.data.normalization import GlobalAIStats, normalize_ai
from geoforge_diffusion.data.raw import (
    load_horizon_surfaces_time_samples,
    load_impedance_time_x,
    resize_area,
    validate_impedance,
)
from geoforge_diffusion.data.schema import BucketSpec
from geoforge_diffusion.utils.io import read_json


class ImpedanceBucketDataset(Dataset[dict[str, object]]):
    """Deterministic online crops for one fixed output-size bucket."""

    def __init__(
        self,
        *,
        sample_ids: list[str],
        raw_dir: str | Path,
        stats: GlobalAIStats | str | Path,
        bucket: BucketSpec,
        samples_per_epoch: int,
        seed: int,
        target_key: str = "ip",
        horizontal_flip: bool = True,
        cache_size: int = 2,
        conditioning: dict[str, object] | SparseConditionConfig | None = None,
    ) -> None:
        if not sample_ids:
            raise ValueError("At least one sample id is required")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        self.sample_ids = list(sample_ids)
        self.raw_dir = Path(raw_dir)
        self.stats = (
            stats if isinstance(stats, GlobalAIStats) else GlobalAIStats.from_dict(read_json(stats))
        )
        self.bucket = bucket
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.target_key = str(target_key)
        self.horizontal_flip = bool(horizontal_flip)
        self.cache_size = int(cache_size)
        self.conditioning = (
            conditioning
            if isinstance(conditioning, SparseConditionConfig)
            else SparseConditionConfig.from_dict(conditioning)
        )
        self.epoch = 0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._horizon_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bucket_seed = zlib.crc32(bucket.name.encode("utf-8"))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index), self._bucket_seed])
        )
        sample_id = self.sample_ids[int(rng.integers(0, len(self.sample_ids)))]
        impedance = self._load(sample_id)
        source_h, source_w = self.bucket.source_shape
        max_top = impedance.shape[0] - source_h
        max_left = impedance.shape[1] - source_w
        if max_top < 0 or max_left < 0:
            raise ValueError(
                f"{sample_id} shape={impedance.shape} is smaller than source window "
                f"{self.bucket.source_shape}"
            )
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
        source = np.ascontiguousarray(impedance[top : top + source_h, left : left + source_w])
        sample = resize_area(source, self.bucket.target_shape)
        flipped = self.horizontal_flip and bool(rng.integers(0, 2))
        if flipped:
            sample = np.ascontiguousarray(sample[:, ::-1])
        return self._format_item(
            sample,
            sample_id=sample_id,
            top=top,
            left=left,
            flipped=flipped,
            rng=rng,
        )

    def _format_item(
        self,
        sample: np.ndarray,
        *,
        sample_id: str,
        top: int,
        left: int,
        flipped: bool,
        rng: np.random.Generator,
    ) -> dict[str, object]:
        normalized = np.ascontiguousarray(normalize_ai(sample, self.stats))
        scale_t, scale_x = self.bucket.scale_factors
        item: dict[str, object] = {
            "sample": torch.from_numpy(normalized[None]),
            "sample_id": sample_id,
            "bucket": self.bucket.name,
            "source_shape": torch.tensor(self.bucket.source_shape, dtype=torch.int64),
            "target_shape": torch.tensor(self.bucket.target_shape, dtype=torch.int64),
            "scale_factors": torch.tensor([scale_t, scale_x], dtype=torch.float32),
            "log2_scale_factors": torch.tensor(np.log2([scale_t, scale_x]), dtype=torch.float32),
            "crop_top": top,
            "crop_left": left,
            "horizontal_flip": flipped,
        }
        if self.conditioning.enabled:
            subset = sample_condition_subset(rng, self.conditioning.subset_probabilities)
            well_present = subset in {"well", "both"}
            horizon_present = subset in {"horizon", "both"}
            if well_present:
                well_condition, _ = make_well_condition(
                    normalized,
                    rng=rng,
                    min_wells=self.conditioning.min_wells,
                    max_wells=self.conditioning.max_wells,
                    minimum_spacing_pixels=max(
                        1,
                        int(round(self.conditioning.well_min_spacing_fraction * sample.shape[1])),
                    ),
                    max_bottom_truncation_fraction=(
                        self.conditioning.well_max_bottom_truncation_fraction
                    ),
                )
            else:
                well_condition = np.zeros((2, *self.bucket.target_shape), dtype=np.float32)
            if horizon_present:
                horizon_condition, _ = make_horizon_condition(
                    self._load_horizons(sample_id),
                    crop_top=top,
                    crop_left=left,
                    source_shape=self.bucket.source_shape,
                    target_shape=self.bucket.target_shape,
                    flipped=flipped,
                    rng=rng,
                    min_horizons=self.conditioning.min_horizons,
                    max_horizons=self.conditioning.max_horizons,
                    minimum_spacing_pixels=max(
                        1,
                        int(
                            round(
                                self.conditioning.horizon_min_spacing_fraction
                                * self.bucket.target_shape[0]
                            )
                        ),
                    ),
                    vertical_bins=self.conditioning.horizon_vertical_bins,
                    gaussian_sigma_pixels=self.conditioning.horizon_sigma_pixels,
                    minimum_valid_fraction=self.conditioning.minimum_horizon_valid_fraction,
                )
            else:
                horizon_condition = np.zeros((2, *self.bucket.target_shape), dtype=np.float32)
            item.update(
                {
                    "well_condition": torch.from_numpy(well_condition),
                    "horizon_condition": torch.from_numpy(horizon_condition),
                    "well_present": torch.tensor(float(well_present), dtype=torch.float32),
                    "horizon_present": torch.tensor(float(horizon_present), dtype=torch.float32),
                    "condition_subset": subset,
                }
            )
        return item

    def _load(self, sample_id: str) -> np.ndarray:
        if sample_id in self._cache:
            self._cache.move_to_end(sample_id)
            return self._cache[sample_id]
        impedance = load_impedance_time_x(self.raw_dir, sample_id, self.target_key)
        validate_impedance(impedance, sample_id)
        if self.cache_size > 0:
            self._cache[sample_id] = impedance
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return impedance

    def _load_horizons(self, sample_id: str) -> np.ndarray:
        if sample_id in self._horizon_cache:
            self._horizon_cache.move_to_end(sample_id)
            return self._horizon_cache[sample_id]
        horizons = load_horizon_surfaces_time_samples(
            self.raw_dir,
            sample_id,
            key=self.conditioning.horizon_key,
            archive_name=self.conditioning.horizon_archive_name,
        )
        if self.cache_size > 0:
            self._horizon_cache[sample_id] = horizons
            while len(self._horizon_cache) > self.cache_size:
                self._horizon_cache.popitem(last=False)
        return horizons


class FixedValidationBucketDataset(ImpedanceBucketDataset):
    """A fixed, non-augmented validation view for one resolution bucket."""

    def __init__(
        self,
        *,
        records: list[dict[str, object]],
        raw_dir: str | Path,
        stats: GlobalAIStats | str | Path,
        bucket: BucketSpec,
        target_key: str = "ip",
        cache_size: int = 2,
        conditioning: dict[str, object] | SparseConditionConfig | None = None,
        condition_seed: int = 0,
    ) -> None:
        selected = [record for record in records if str(record["bucket"]) == bucket.name]
        if not selected:
            raise ValueError(f"No fixed validation records found for bucket {bucket.name!r}")
        for record in selected:
            if tuple(int(value) for value in record["source_shape"]) != bucket.source_shape:
                raise ValueError(f"Validation source shape does not match bucket {bucket.name!r}")
            if tuple(int(value) for value in record["target_shape"]) != bucket.target_shape:
                raise ValueError(f"Validation target shape does not match bucket {bucket.name!r}")
        self.records = selected
        super().__init__(
            sample_ids=sorted({str(record["sample_id"]) for record in selected}),
            raw_dir=raw_dir,
            stats=stats,
            bucket=bucket,
            samples_per_epoch=len(selected),
            seed=condition_seed,
            target_key=target_key,
            horizontal_flip=False,
            cache_size=cache_size,
            conditioning=conditioning,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        sample_id = str(record["sample_id"])
        top = int(record["crop_top"])
        left = int(record["crop_left"])
        source_h, source_w = self.bucket.source_shape
        impedance = self._load(sample_id)
        source = np.ascontiguousarray(impedance[top : top + source_h, left : left + source_w])
        if source.shape != self.bucket.source_shape:
            raise ValueError(f"Fixed crop for {sample_id} is out of bounds: shape={source.shape}")
        sample = resize_area(source, self.bucket.target_shape)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index), self._bucket_seed])
        )
        return self._format_item(
            sample,
            sample_id=sample_id,
            top=top,
            left=left,
            flipped=False,
            rng=rng,
        )
