"""Train or resume the conditional acoustic-impedance EDM."""

import argparse
from pathlib import Path

from geoforge_diffusion.utils.runtime import configure_windows_conda_dll_search

_DLL_HANDLE = configure_windows_conda_dll_search()

from geoforge_diffusion.training.runner import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--dry-run", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
