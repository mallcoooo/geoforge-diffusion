# Geologically Constrained Subsurface Modeling for Constructing Conditional Diffusion Prior

An implementation of multi-resolution conditional EDM training and
sampling with synthetic acoustic-impedance data. The model supports
unconditional, well-conditioned, horizon-conditioned and joint sampling, plus
seismic-guided diffusion posterior sampling (DPS).

Dataset DOI: [10.5281/zenodo.23494028](https://doi.org/10.5281/zenodo.23494028).
Data formats, licensing and download instructions are described in
[data/README.md](data/README.md).

## Installation

Use Python 3.10 or later and install a PyTorch build suitable for your machine.
Run the following commands from the repository root:

```bash
python -m pip install -e .
```

The dependencies are NumPy, PyTorch, PyYAML and tqdm. The same commands work in
PowerShell and a Unix shell. CPU runs use FP32; the full training configuration
uses BF16 on CUDA.

## Small runnable example

Prepare the six included synthetic crops and train a small demonstration model:

```bash
python scripts/prepare_data.py --config configs/example.yaml
python scripts/train.py --config configs/example.yaml --run-dir runs/example
```

The demonstration runs two updates. Resume it for one more update with:

```bash
python scripts/train.py --config configs/example.yaml --run-dir runs/example --resume runs/example/checkpoints/latest.pt --max-updates 3
```

Choose any included `examples/sample_XXXXXX/` directory as the input grid.
The following commands use one of the included samples:

```bash
python scripts/sample.py --checkpoint runs/example/checkpoints/latest.pt --sample-dir examples/sample_000868 --output-dir runs/prior --condition-subset both --seeds 1001 2002
```

`--condition-subset` accepts `unconditional`, `well`, `horizon` and `both`.
Conditions are reproducibly extracted from the selected synthetic sample using
`--condition-seed`.

## synthetic case

The complete input case and archived reference results are also
included in `examples/paper_case/`. It contains the
exact test section, observed seismic,
wavelet, fixed horizons and well conditions, blind logs and low-frequency
background. Its `sampling.json` supplies the original sampling defaults;
`conditions.npz` supplies fixed maps rather than a new random condition draw.
Run guided or unguided DPS with a trained checkpoint:

```bash
python scripts/sample.py --checkpoint runs/full/checkpoints/latest.pt --sample-dir examples/paper_case --method dps --output-dir runs/paper_guided
python scripts/sample.py --checkpoint runs/full/checkpoints/latest.pt --sample-dir examples/paper_case --method dps --low-frequency-weight 0 --output-dir runs/paper_unguided
```

Case defaults select the original 32 steps and 32 seeds. Add `--seeds 1001
--num-steps 2` for a short execution check. Archived reference impedances and
background use kg m^-2 s^-1. 

## Full training

Download `metadata.zip` and all 20 sample archives, `synthetic_part_01.zip`
through `synthetic_part_20.zip`, from the
[synthetic training dataset](https://doi.org/10.5281/zenodo.23494028).
Extract all 21 ZIPs into the repository root. Each archive starts with `data/`,
so their contents merge into `data/metadata/` and `data/synthetic/`.
See [data/README.md](data/README.md) for the extraction command.
The fixed splits, training normalization and validation crops are included,
so metadata preparation can be skipped for this dataset:

```bash
python scripts/train.py --config configs/train.yaml --run-dir runs/train --dry-run
python scripts/train.py --config configs/train.yaml --run-dir runs/full
```

For a different collection of already generated samples, build new metadata:

```bash
python scripts/prepare_data.py --config configs/train.yaml --raw-dir data/synthetic --metadata-dir data/metadata --force
```

Sparse condition dropout trains all four condition subsets. A single reversible global
affine transform uses training samples only; there is no per-sample amplitude
normalization or spatial coordinate input. The U-Net, EDM loss, condition
encoders and EMA equations follow the retained research implementation.

The full configuration uses 200,000 updates, gradient accumulation, learning-rate
warmup and EMA. Training writes `metrics.csv`, `config.json` and
`checkpoints/latest.pt`. Checkpoints contain online/EMA weights, normalization,
optimizer, scheduler, scaler, loader positions and random states. Resuming
requires the same configuration and prepared metadata; use `--max-updates` to
extend a run. Pass both `--resume` and the desired `--run-dir`.

## Seismic-guided sampling

DPS reads the existing `poststack` array and a matching `wavelet.npy`, supplied
in `examples/paper_case/`. The full training dataset contains only impedance and
horizon picks. An alternative wavelet can be supplied with
`--wavelet`. `--dps-scale` controls the
data-consistency correction.

The differentiable operator is `d = 0.5 W D log(Z)`, with forward differences
and a zero terminal difference. It first crops the denoised grid to its native
extent, then differentiates and convolves. The seismic residual and its scaling
are evaluated only on that extent. The manuscript case provides the original
noisy observation made with this same time-domain operator. Its archived TV and
DPS results are supplied as numerical references; baseline inversion and
figure-generation code are excluded.

## Contents

```text
src/geoforge_diffusion/   model, data loading, training and sampling
scripts/                 prepare_data.py, train.py, sample.py
configs/                 full and small-example configurations
data/                    availability statement; metadata comes with the dataset
examples/                six small crops and the complete manuscript test case
```
