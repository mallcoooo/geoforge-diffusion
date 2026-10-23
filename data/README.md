# Synthetic training data

The dataset contains 4,000 synthetic acoustic-impedance models and horizon picks.
The fixed sample split is 3,200 training, 400 validation and 400 test samples.

**Dataset DOI:** [10.5281/zenodo.23494028](https://doi.org/10.5281/zenodo.23494028).

**Data license:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).


## Download and layout

Download all 21 ZIP files from the
[Zenodo dataset record](https://doi.org/10.5281/zenodo.23494028):

- `metadata.zip`: the six shared metadata files.
- `synthetic_part_01.zip` through `synthetic_part_20.zip`: 200 complete samples
  per archive, in ascending sample-number order.

Place the ZIPs in an `archives/` folder under the repository root, then run:

```bash
python -c "from pathlib import Path; from zipfile import ZipFile
names = ['metadata.zip'] + [f'synthetic_part_{i:02d}.zip' for i in range(1, 21)]
for name in names:
    with ZipFile(Path('archives') / name) as archive:
        archive.extractall('.')"
```

Each ZIP is independently extractable and starts with `data/`. Extract all ZIPs
into the same repository root; their contents merge into the layout below.
Sample files and metadata are preserved byte for byte, and the fixed split
is defined by `metadata/splits/`, independently of archive boundaries.

```text
data/
  README.md
  metadata/
    manifest.json
    stats.json
    validation_crops.json
    splits/{train,validation,test}.txt
  synthetic/
    sample_000001/data.npz
    ...
```

Each sample has one NPZ file containing only two float32 arrays:

| Key | Shape | Meaning / unit                                                   |
| --- | --- |------------------------------------------------------------------|
| `ip` | `[trace, time]` | Acoustic impedance (10^6 kg m^-2 s^-1)                           |
| `horizon_surfaces_time_samples` | `[horizon, trace]` | Fractional zero-based time-sample indices; NaN means unavailable |

Compression is lossless LZMA inside standard NPZ files. Impedance is stored in
Fortran order for better compression; array axes and numerical values are
unchanged. Read either array normally:

```python
import numpy

with numpy.load("data/synthetic/sample_000001/data.npz", allow_pickle=False) as sample:
    impedance = sample["ip"]
    horizons = sample["horizon_surfaces_time_samples"]
```

The training loader transposes impedance to `[time, trace]` and extracts sparse
well conditions from it. No seismic, wavelets, physical time axes or individual
sample descriptions are included in this training dataset. `configs/train.yaml`
reads horizon picks from the same `data.npz` as impedance.

Shared metadata preserves the original splits, training-only normalization and
fixed validation crops. Extracting all 21 archives is sufficient to run the full
training configuration without preparing metadata again.
