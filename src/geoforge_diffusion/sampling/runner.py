"""Sample on the input grid using minimal network padding and output cropping."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from geoforge_diffusion.data.conditioning import (
    SparseConditionConfig,
    make_horizon_condition,
    make_well_condition,
)
from geoforge_diffusion.data.normalization import GlobalAIStats, denormalize_ai, normalize_ai
from geoforge_diffusion.data.raw import load_impedance_time_x, validate_impedance
from geoforge_diffusion.models import create_edm_model
from geoforge_diffusion.sampling.dps import EDMDPSConfig, edm_dps_sample
from geoforge_diffusion.sampling.edm_sampler import EDMSamplerConfig, edm_sample
from geoforge_diffusion.sampling.forward import PoststackForward
from geoforge_diffusion.sampling.low_frequency import LowFrequencyLogImpedancePrior
from geoforge_diffusion.training.checkpoint import load_checkpoint
from geoforge_diffusion.utils.io import read_json, write_json


def minimal_shape(shape, divisor):
    """Round up each dimension; reflection convolutions need at least two bottleneck pixels."""
    if len(shape) != 2 or min(shape) < 1 or divisor < 1:
        raise ValueError("Expected positive spatial dimensions and network divisor")
    return tuple(max(2 * divisor, ((value + divisor - 1) // divisor) * divisor) for value in shape)


def condition_maps(sample_dir, ai, stats, settings, subset, seed):
    config = SparseConditionConfig.from_dict(settings)
    if subset != "unconditional" and not config.enabled:
        raise ValueError("This checkpoint was trained without sparse conditions")
    height, width = ai.shape
    rng = np.random.default_rng(seed)
    well = np.zeros((2, height, width), dtype=np.float32)
    horizon = np.zeros_like(well)
    fixed = sample_dir / "conditions.npz"
    if fixed.is_file() and subset != "unconditional":
        with np.load(fixed, allow_pickle=False) as archive:
            if subset in {"well", "both"}:
                well = archive["well_condition"].copy()
                if (
                    stats.domain != "ai"
                    or stats.mean != float(archive["normalization_mean"])
                    or stats.std != float(archive["normalization_std"])
                ):
                    valid = well[1] > 0
                    well[0, valid] = normalize_ai(archive["well_impedance"][valid], stats)
            if subset in {"horizon", "both"}:
                horizon = archive["horizon_condition"]
        if any(
            values.shape != (2, height, width) or not np.isfinite(values).all()
            for values in (well, horizon)
        ):
            raise ValueError("Fixed condition maps must be finite and match the source grid")
        return well, horizon
    if subset in {"well", "both"}:
        well, _ = make_well_condition(
            normalize_ai(ai, stats),
            rng=rng,
            min_wells=config.min_wells,
            max_wells=config.max_wells,
            minimum_spacing_pixels=max(1, round(config.well_min_spacing_fraction * width)),
            max_bottom_truncation_fraction=config.well_max_bottom_truncation_fraction,
        )
    if subset in {"horizon", "both"}:
        with np.load(sample_dir / config.horizon_archive_name, allow_pickle=False) as archive:
            surfaces = archive[config.horizon_key]
        horizon, _ = make_horizon_condition(
            surfaces,
            crop_top=0,
            crop_left=0,
            source_shape=ai.shape,
            target_shape=ai.shape,
            flipped=False,
            rng=rng,
            min_horizons=config.min_horizons,
            max_horizons=config.max_horizons,
            minimum_spacing_pixels=max(1, round(config.horizon_min_spacing_fraction * height)),
            vertical_bins=config.horizon_vertical_bins,
            gaussian_sigma_pixels=config.horizon_sigma_pixels,
            minimum_valid_fraction=config.minimum_horizon_valid_fraction,
        )
    return well, horizon


def sample(args):
    sample_dir = Path(args.sample_dir)
    settings_path = sample_dir / "sampling.json"
    settings = read_json(settings_path) if settings_path.is_file() else {}
    defaults = {
        "condition_subset": "unconditional",
        "seeds": [1001],
        "precision": "fp32",
        "condition_seed": 42,
        "dps_scale": 0.05,
        "low_frequency_weight": 0.0,
    }
    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, settings.get(key, default))
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty, unique and nonnegative")
    saved = load_checkpoint(args.checkpoint)
    if saved.get("format") != "geoforge_public_edm_v1":
        raise ValueError("Use a checkpoint produced by the public training entry point")
    stats = GlobalAIStats.from_dict(saved["stats"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_edm_model(saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["ema"]["model"])
    model.requires_grad_(False)
    ai = load_impedance_time_x(sample_dir.parent, sample_dir.name)
    validate_impedance(ai, sample_dir.name)
    height, width = ai.shape
    network_shape = minimal_shape(ai.shape, model.network.downsample_factor)
    padding = (0, network_shape[1] - width, 0, network_shape[0] - height)
    well, horizon = condition_maps(
        sample_dir,
        ai,
        stats,
        saved["config"].get("conditioning"),
        args.condition_subset,
        args.condition_seed,
    )

    kwargs = {"log2_scale_factors": torch.zeros((1, 2), device=device)}
    if model.network.conditioning_enabled:
        kwargs.update(
            {
                "well_condition": functional.pad(torch.from_numpy(well)[None].to(device), padding),
                "horizon_condition": functional.pad(
                    torch.from_numpy(horizon)[None].to(device), padding
                ),
                "well_present": torch.tensor(
                    [float(args.condition_subset in {"well", "both"})], device=device
                ),
                "horizon_present": torch.tensor(
                    [float(args.condition_subset in {"horizon", "both"})], device=device
                ),
            }
        )
    sampler = EDMSamplerConfig.from_dict(
        {**saved["model_config"]["edm"]["sampler"], **settings.get("edm", {})}
    )
    if args.num_steps is None and "num_steps" in settings:
        sampler = replace(sampler, num_steps=settings["num_steps"])
    if args.num_steps is not None:
        sampler = replace(sampler, num_steps=args.num_steps)
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected device does not support BF16")
    diagnostics = []
    prior_loss = None
    if args.method == "dps":
        with np.load(sample_dir / "data.npz", allow_pickle=False) as archive:
            if "poststack" not in archive:
                raise ValueError("DPS requires observed seismic; use examples/paper_case")
            seismic = np.asarray(archive["poststack"].T, dtype=np.float32)
        if seismic.shape != ai.shape or not np.isfinite(seismic).all():
            raise ValueError("Synthetic seismic must be finite and match the input grid")
        wavelet_path = Path(args.wavelet) if args.wavelet else sample_dir / "wavelet.npy"
        wavelet = np.load(wavelet_path, allow_pickle=False)
        mask = np.ones_like(seismic, dtype=bool)
        if (sample_dir / "conditions.npz").is_file():
            with np.load(sample_dir / "conditions.npz", allow_pickle=False) as archive:
                mask = archive["seismic_mask"].astype(bool)
        if mask.shape != ai.shape or not mask.any():
            raise ValueError("Seismic support must match the grid and contain valid samples")
        scale = float(seismic[mask].std())
        operator = PoststackForward(stats, wavelet, scale).to(device)
        observed = torch.from_numpy(seismic.copy())[None, None].to(device) / scale
        observation_mask = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
        if args.low_frequency_weight > 0:
            with np.load(sample_dir / "data.npz", allow_pickle=False) as archive:
                dt = float(np.mean(np.diff(archive["time"])))
            prior_loss = LowFrequencyLogImpedancePrior(
                stats,
                np.load(sample_dir / "background.npy", allow_pickle=False),
                ai_scale=1e-6,
                time_sigma_samples=settings.get("background_time_sigma_ms", 60.0) * 1e-3 / dt,
                trace_sigma=settings.get("background_trace_sigma", 12.0),
                log_std=settings.get("low_frequency_log_std", 0.05),
                mask=mask,
            ).to(device)

        def forward(values):
            return operator(values[..., :height, :width])

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        if settings.get("cpu_latents", False):
            latents = torch.randn(
                (1, 1, *network_shape), generator=torch.Generator().manual_seed(seed)
            ).to(device)
        else:
            latents = torch.randn((1, 1, *network_shape), generator=generator, device=device)
        dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        with torch.autocast(
            device.type, dtype=dtype, enabled=device.type == "cuda" and args.precision != "fp32"
        ):
            if args.method == "prior":
                values = edm_sample(
                    model, latents, sampler, model_kwargs=kwargs, generator=generator
                )
            else:
                values, diagnostic = edm_dps_sample(
                    model,
                    latents,
                    sampler,
                    forward_operator=forward,
                    observation=observed,
                    dps_config=EDMDPSConfig(
                        guidance_scale=args.dps_scale,
                        guidance_anneal_power=settings.get("guidance_anneal_power", 0.5),
                        guidance_taper_sigma=settings.get("guidance_taper_sigma", 0.0),
                        low_frequency_weight=args.low_frequency_weight,
                    ),
                    observation_mask=observation_mask,
                    low_frequency_loss=prior_loss,
                    model_kwargs=kwargs,
                    generator=generator,
                )
                diagnostics.append(diagnostic)
        cropped = values[0, 0, :height, :width].detach().float().cpu().numpy()
        if not np.isfinite(cropped).all():
            raise FloatingPointError("Sample contains non-finite values")
        results.append(cropped)
        print(f"seed={seed}: {height}x{width}, network={network_shape}", flush=True)
    normalized = np.stack(results)
    with np.load(sample_dir / "data.npz", allow_pickle=False) as archive:
        coordinates = {"time": archive["time"]} if "time" in archive else {}
    np.savez_compressed(
        output / "samples.npz",
        normalized=normalized,
        ip=denormalize_ai(normalized, stats),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        well_condition=well,
        horizon_condition=horizon,
        **coordinates,
    )
    write_json(
        output / "metadata.json",
        {
            "method": args.method,
            "condition_subset": args.condition_subset,
            "condition_seed": args.condition_seed,
            "seeds": args.seeds,
            "original_shape": [height, width],
            "sampling_shape": list(network_shape),
            "padding_bottom_right": [padding[3], padding[1]],
            "log2_scale_factors": [0.0, 0.0],
            "array_layout": "sample_time_trace",
            "num_steps": sampler.num_steps,
            "condition_source": "fixed" if (sample_dir / "conditions.npz").is_file() else "sampled",
            "low_frequency_weight": args.low_frequency_weight if args.method == "dps" else 0.0,
            "diagnostics": diagnostics,
        },
    )
    return output
