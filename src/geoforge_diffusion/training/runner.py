"""Multi-resolution EDM training with deterministic crops and resumable state."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from geoforge_diffusion.data.dataset import FixedValidationBucketDataset, ImpedanceBucketDataset
from geoforge_diffusion.data.normalization import GlobalAIStats
from geoforge_diffusion.data.schema import parse_buckets
from geoforge_diffusion.models import create_edm_loss, create_edm_model
from geoforge_diffusion.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from geoforge_diffusion.training.ema import ExponentialMovingAverage
from geoforge_diffusion.training.loaders import CyclingDataLoader
from geoforge_diffusion.utils.io import read_ids, read_json, read_yaml, write_json


def _loaders(config, metadata, raw_dir, buckets, workers, seed):
    settings = config["data"]
    records = read_json(metadata / "validation_crops.json")["records"]
    common = {"num_workers": workers, "pin_memory": settings["loader"]["pin_memory"]}
    if workers:
        common["prefetch_factor"] = settings["loader"].get("prefetch_factor", 2)
    train, validation = {}, {}
    for index, bucket in enumerate(buckets):
        dataset_kwargs = dict(
            raw_dir=raw_dir,
            stats=metadata / "stats.json",
            bucket=bucket,
            target_key=settings.get("target_key", "ip"),
            cache_size=settings.get("cache_size", 2),
            conditioning=config.get("conditioning"),
        )
        dataset = ImpedanceBucketDataset(
            sample_ids=read_ids(metadata / "splits/train.txt"),
            samples_per_epoch=settings["train_samples_per_bucket"],
            seed=seed + index * 1009,
            horizontal_flip=settings.get("augmentation", {}).get("horizontal_flip", True),
            **dataset_kwargs,
        )
        batch_size = settings["microbatch_sizes"][bucket.name]
        train[bucket.name] = CyclingDataLoader(
            DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True,
                generator=torch.Generator().manual_seed(seed + index * 2027),
                **common,
            )
        )
        selected = [r for r in records if r["bucket"] == bucket.name]
        limit = settings["validation_samples_per_bucket"]
        if limit > 0:
            selected = selected[:limit]
        validation[bucket.name] = DataLoader(
            FixedValidationBucketDataset(
                records=selected,
                condition_seed=seed + index * 4049,
                **dataset_kwargs,
            ),
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(seed + index * 3037),
            **common,
        )
    return train, validation


def _kwargs(batch, device):
    keys = (
        "log2_scale_factors",
        "well_condition",
        "horizon_condition",
        "well_present",
        "horizon_present",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys if key in batch}


def _autocast(device, precision):
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(
        device.type, dtype=dtype, enabled=device.type == "cuda" and precision != "fp32"
    )


@torch.no_grad()
def _validate(model, loaders, loss_function, device, precision, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for loader in loaders.values():
        total, count = 0.0, 0
        for batch in loader:
            clean = batch["sample"].to(device)
            with _autocast(device, precision):
                loss = loss_function(
                    model, clean, generator=generator, model_kwargs=_kwargs(batch, device)
                )
            total += float(loss) * len(clean)
            count += len(clean)
        losses.append(total / count)
    return float(np.mean(losses))


def train(args):
    config = read_yaml(args.config)
    settings = config["training"]
    metadata = Path(args.metadata_dir or config["data"]["metadata_dir"])
    manifest = read_json(metadata / "manifest.json")
    raw_dir = Path(args.raw_dir) if args.raw_dir else metadata / manifest["raw_dir"]
    output = Path(args.run_dir)
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    buckets = parse_buckets(config["data"]["buckets"])
    if manifest["buckets"] != [b.to_dict() for b in buckets]:
        raise ValueError("Metadata and configured training buckets differ")
    model_config = {key: config[key] for key in ("model", "edm")}
    if bool(config.get("conditioning", {}).get("enabled")) != bool(
        config["model"].get("conditioning", {}).get("enabled")
    ):
        raise ValueError("Dataset and model conditioning must be enabled together")
    workers = args.num_workers
    if workers is None:
        workers = config["data"]["loader"]["num_workers"]
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    train_loaders, validation_loaders = _loaders(
        config,
        metadata,
        raw_dir,
        buckets,
        workers,
        seed,
    )
    model = create_edm_model(model_config).to(device)
    if any(
        any(size % model.network.downsample_factor for size in b.target_shape)
        or min(b.target_shape) < 2 * model.network.downsample_factor
        for b in buckets
    ):
        raise ValueError("Training bucket shapes must satisfy the network dimensions")
    loss_function = create_edm_loss(model_config)
    optimizer_config = settings["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"],
    )
    warmup = optimizer_config.get("warmup_updates", 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(warmup, 1), 1.0),
    )
    ema = ExponentialMovingAverage(model, **settings["ema"])
    precision = settings["mixed_precision"]
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be fp32, bf16 or fp16")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected device does not support BF16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    bucket_rng = np.random.default_rng(seed + 1009)
    noise_rng = torch.Generator(device=device).manual_seed(seed + 2017)
    stats = GlobalAIStats.from_dict(read_json(metadata / "stats.json"))
    fingerprint = hashlib.sha256()
    for name in (
        "stats.json",
        "validation_crops.json",
        "splits/train.txt",
        "splits/validation.txt",
        "splits/test.txt",
    ):
        fingerprint.update((metadata / name).read_bytes())
    data_fingerprint = fingerprint.hexdigest()
    update = 0
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved.get("format") != "geoforge_public_edm_v1":
            raise ValueError("Unsupported checkpoint format")
        if saved["config"] != config or saved["data_fingerprint"] != data_fingerprint:
            raise ValueError("Resume requires the same configuration and prepared metadata")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        for name, state in saved["loaders"].items():
            train_loaders[name].load_state_dict(state)
        bucket_rng.bit_generator.state = saved["bucket_rng_state"]
        noise_rng.set_state(saved["noise_generator_state"])
        restore_rng_state(saved["rng_state"])
        update = saved["update"]
    max_updates = args.max_updates if args.max_updates is not None else settings["max_updates"]
    if args.dry_run:
        max_updates = update + 1
    accumulation = settings["gradient_accumulation_steps"]
    intervals = {
        name: (1 if args.dry_run else settings[f"{name}_every_updates"])
        for name in ("log", "validate", "checkpoint")
    }
    if max_updates <= update or accumulation < 1 or min(intervals.values()) < 1:
        raise ValueError("Updates, accumulation and logging intervals must be positive")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    probabilities = np.array([b.probability for b in buckets])
    progress = trange(update, max_updates, initial=update, total=max_updates, unit="update")
    for _ in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(accumulation):
            bucket = buckets[int(bucket_rng.choice(len(buckets), p=probabilities))]
            batch = next(train_loaders[bucket.name])
            with _autocast(device, precision):
                loss = loss_function(
                    model,
                    batch["sample"].to(device),
                    generator=noise_rng,
                    model_kwargs=_kwargs(batch, device),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            scaler.scale(loss / accumulation).backward()
            train_loss += float(loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Training gradients are not finite")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        update += 1
        validation_loss = None
        if update % intervals["validate"] == 0 or update == max_updates:
            validation_loss = _validate(
                ema.module, validation_loaders, loss_function, device, precision, seed + 4099
            )
        if update % intervals["log"] == 0 or validation_loss is not None or update == max_updates:
            row = {
                "update": update,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            log = output / "metrics.csv"
            new = not log.exists()
            with log.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            progress.set_postfix(loss=f"{train_loss:.4f}", validation=validation_loss)
        if update % intervals["checkpoint"] == 0 or update == max_updates:
            save_checkpoint(
                output / "checkpoints/latest.pt",
                {
                    "format": "geoforge_public_edm_v1",
                    "update": update,
                    "config": config,
                    "model_config": model_config,
                    "stats": stats.to_dict(),
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "data_fingerprint": data_fingerprint,
                    "loaders": {
                        name: loader.state_dict() for name, loader in train_loaders.items()
                    },
                    "bucket_rng_state": bucket_rng.bit_generator.state,
                    "noise_generator_state": noise_rng.get_state(),
                    "rng_state": capture_rng_state(),
                },
            )
    print(f"Saved checkpoint: {output / 'checkpoints/latest.pt'}")
    return output
