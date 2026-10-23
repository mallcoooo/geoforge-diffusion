"""Sample an EDM on an existing synthetic sample's grid with automatic minimal padding."""

import argparse
from pathlib import Path

from geoforge_diffusion.utils.runtime import configure_windows_conda_dll_search

_DLL_HANDLE = configure_windows_conda_dll_search()

from geoforge_diffusion.sampling.runner import sample  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/samples"))
    parser.add_argument("--method", choices=["prior", "dps"], default="prior")
    parser.add_argument(
        "--condition-subset",
        choices=["unconditional", "well", "horizon", "both"],
        default=None,
    )
    parser.add_argument("--condition-seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument(
        "--wavelet", type=Path, help="DPS wavelet NPY; default: sample-dir/wavelet.npy"
    )
    parser.add_argument("--dps-scale", type=float, default=None)
    parser.add_argument(
        "--low-frequency-weight",
        type=float,
        default=None,
        help="DPS background constraint; 0 disables it. Case settings supply defaults.",
    )
    sample(parser.parse_args())


if __name__ == "__main__":
    main()
