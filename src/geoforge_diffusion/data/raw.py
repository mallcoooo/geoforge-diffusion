from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

SAMPLE_RE = re.compile(r"^sample_\d{6}$")


def discover_sample_ids(raw_dir: str | Path) -> list[str]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    sample_ids = [
        child.name
        for child in root.iterdir()
        if child.is_dir() and SAMPLE_RE.match(child.name) and (child / "data.npz").is_file()
    ]
    if not sample_ids:
        raise ValueError(f"No sample_XXXXXX/data.npz samples found under {root}")
    return sorted(sample_ids)


def load_raw_array(
    raw_dir: str | Path,
    sample_id: str,
    key: str = "ip",
    archive_name: str = "data.npz",
) -> np.ndarray:
    path = Path(raw_dir) / sample_id / archive_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive for {sample_id}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"Key {key!r} is absent from {path}; available={archive.files}")
        values = np.asarray(archive[key], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D {key!r} for {sample_id}, got shape={values.shape}")
    return np.ascontiguousarray(values)


def load_impedance_time_x(
    raw_dir: str | Path,
    sample_id: str,
    key: str = "ip",
) -> np.ndarray:
    """Load raw [x, time] impedance as contiguous [time, x]."""

    return np.ascontiguousarray(load_raw_array(raw_dir, sample_id, key).T)


def load_horizon_surfaces_time_samples(
    raw_dir: str | Path,
    sample_id: str,
    *,
    key: str = "horizon_surfaces_time_samples",
    archive_name: str = "geology.npz",
) -> np.ndarray:
    """Load horizon picks as ``[horizon, x]`` floating-point time-sample indices."""

    surfaces = load_raw_array(raw_dir, sample_id, key=key, archive_name=archive_name)
    if surfaces.shape[1] <= 0:
        raise ValueError(f"{sample_id} contains no horizontal samples in {key!r}")
    return surfaces


def validate_impedance(values: np.ndarray, sample_id: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{sample_id} contains non-finite impedance values")
    if np.any(values <= 0.0):
        raise ValueError(f"{sample_id} contains non-positive impedance values")


def array_fingerprint(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    buffer = memoryview(contiguous).cast("B")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(buffer)
    return digest.hexdigest()


def resize_area(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Area downsampling for a 2D field, preserving its mean."""

    if values.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape={values.shape}")
    if any(target > source for source, target in zip(values.shape, target_shape, strict=True)):
        raise ValueError(f"Upsampling is not supported: {values.shape} -> {target_shape}")
    if values.shape == target_shape:
        return np.ascontiguousarray(values.astype(np.float32, copy=False))

    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[None, None]
    resized = functional.interpolate(tensor, size=target_shape, mode="area")
    return np.ascontiguousarray(resized[0, 0].numpy().astype(np.float32, copy=False))
