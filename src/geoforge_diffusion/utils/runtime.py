"""Runtime setup shared by command-line entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_windows_conda_dll_search() -> object | None:
    """Expose the active Conda environment's DLL directory before importing PyTorch."""
    if sys.platform != "win32":
        return None
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if not dll_dir.is_dir():
        return None
    dll_dir_text = str(dll_dir)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if dll_dir_text.casefold() not in {entry.casefold() for entry in path_entries}:
        os.environ["PATH"] = dll_dir_text + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        return os.add_dll_directory(dll_dir_text)
    return None
