"""Audit existing synthetic samples and prepare training metadata."""

import argparse
from pathlib import Path

from geoforge_diffusion.utils.runtime import configure_windows_conda_dll_search

_DLL_HANDLE = configure_windows_conda_dll_search()

from geoforge_diffusion.data.metadata import prepare_metadata  # noqa: E402
from geoforge_diffusion.utils.io import read_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = read_yaml(args.config)
    prepare_metadata(
        config,
        args.metadata_dir or config["data"]["metadata_dir"],
        raw_dir_override=args.raw_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
